import os
import requests
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import time
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import google.generativeai as genai

# 한글 폰트 설정
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Configuration
# ==========================================
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

OUTPUT_DIR = "docs"
TEMPLATE_DIR = "templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDICES = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11', 'NASDAQ': 'IXIC'}

# 한미 섹터 분리
KR_SECTORS = {'반도체': '091160', '2차전지': '305700', '에너지': '117700', '금융': '117460', '바이오': '117620'}
US_SECTORS = {'기술(XLK)': 'XLK', '금융(XLF)': 'XLF', '에너지(XLE)': 'XLE', '헬스케어(XLV)': 'XLV', '반도체(SOXX)': 'SOXX'}

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ==========================================
# 2. Data & AI Analysis
# ==========================================
def get_stock_data(ticker, days=365):
    end = datetime.now()
    start = end - timedelta(days=days)
    for _ in range(3):
        try:
            df = fdr.DataReader(ticker, start, end)
            if not df.empty: return df
        except: time.sleep(1)
    return pd.DataFrame()

def get_naver_search(query, category='news'):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET: return []
    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": 5, "sort": "sim"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        return [{'title': i['title'].replace('<b>','').replace('</b>',''), 'link': i['link']} for i in res.json().get('items', [])]
    except: return []

def get_ranto_ai_insight():
    if not GEMINI_API_KEY: return "AI 키가 설정되지 않았습니다."
    try:
        # RSS에서 최신글 링크 추출
        rss_url = "https://rss.blog.naver.com/ranto28.xml"
        res = requests.get(rss_url, timeout=10)
        root = ET.fromstring(res.content)
        link = root.find('.//item/link').text
        
        # 블로그 본문 스크래핑
        blog_res = requests.get(link, timeout=10)
        soup = BeautifulSoup(blog_res.text, 'html.parser')
        iframe = soup.find('iframe', id='mainFrame')
        real_url = "https://blog.naver.com" + iframe['src']
        final_res = requests.get(real_url, timeout=10)
        content = BeautifulSoup(final_res.text, 'html.parser').find('div', class_='se-main-container').get_text()

        # Gemini 1.5 Flash 최신 모델 사용
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"다음 글을 읽고 오늘 주식 시장 투자 인사이트를 3문장으로 요약해줘: {content[:5000]}")
        return response.text
    except: return "블로그 글을 분석하는 중 오류가 발생했습니다."

def analyze_wave_weekly(df, name):
    df.index = pd.to_datetime(df.index)
    w_df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    curr = w_df['Close'].iloc[-1]
    h_20 = w_df['High'].rolling(window=20).max().iloc[-1]
    l_20 = w_df['Low'].rolling(window=20).min().iloc[-1]
    ratio = (curr - l_20) / (h_20 - l_20 + 1e-6)
    
    pos = "횡보/수렴"
    target = f"박스권 내 움직임 (저항: {h_20:,.0f})"
    if ratio > 0.8: 
        pos = "상승 파동 진행"
        target = f"전고점 돌파 시 목표: {l_20 + (h_20-l_20)*1.618:,.0f}"
    elif ratio < 0.2: 
        pos = "하락/조정 파동"
        target = f"지지선 확인 필요: {h_20 - (h_20-l_20)*0.618:,.0f}"
        
    return {'name': name, 'price': f"{curr:,.1f}", 'pos': pos, 'target': target, 'df': w_df}

# ==========================================
# 3. Main Logic
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}}
    
    # 파동 분석
    report_data['waves'] = []
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        res = analyze_wave_weekly(df, name)
        report_data['waves'].append(res)
        fig, ax = plt.subplots(figsize=(10, 4))
        res['df']['Close'].iloc[-50:].plot(ax=ax, color='black')
        ax.set_title(f"{name} Weekly")
        report_data['plots'][f'chart_{name}'] = fig_to_base64(fig)
        plt.close(fig)

    # 한미 섹터 로테이션 분석
    for country, sectors in [('KR', KR_SECTORS), ('US', US_SECTORS)]:
        perf = []
        for s_name, s_ticker in sectors.items():
            s_df = get_stock_data(s_ticker, 60)
            if not s_df.empty:
                p = (s_df['Close'].iloc[-1] / s_df['Close'].iloc[-22] - 1) * 100
                perf.append({'Sector': s_name, 'Perf': p})
        
        pdf = pd.DataFrame(perf).sort_values('Perf', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=pdf, x='Perf', y='Sector', palette='RdYlGn', ax=ax)
        ax.set_title(f"{country} Sector Rotation (1Mo)")
        report_data['plots'][f'sector_{country}'] = fig_to_base64(fig)
        plt.close(fig)

    report_data['ai_insight'] = get_ranto_ai_insight()
    report_data['naver_news'] = get_naver_search("코스피 전망 반도체")
    report_data['naver_blog'] = get_naver_search("미국 주식 시황", 'blog')
    
    # 구글 해외 뉴스 (항상 동작)
    g_url = "https://news.google.com/rss/search?q=US+Stock+Market&hl=ko&gl=KR&ceid=KR:ko"
    g_res = requests.get(g_url)
    report_data['google_news'] = [{'title': i.title.text, 'link': i.link.text} for i in BeautifulSoup(g_res.content, 'xml').find_all('item', limit=5)]

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
