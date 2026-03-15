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
import matplotlib.font_manager as fm

# 한글 폰트 설정 (GitHub Actions 우분투 환경)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Configuration & API Setup
# ==========================================
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

INDICES = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11', 'NASDAQ': 'IXIC'}
SECTOR_ETFs = {
    '반도체': '091160', '2차전지': '305700', '에너지': '117700',
    '철강/소재': '117460', '산업재': '117600', '헬스케어': '117620',
    '금융': '117460', 'IT소프트': '117690'
}

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ==========================================
# 2. Data Acquisition
# ==========================================
def get_stock_data(ticker, days=252):
    end = datetime.now()
    start = end - timedelta(days=days)
    for _ in range(3):
        try:
            df = fdr.DataReader(ticker, start, end)
            if not df.empty: return df
        except: time.sleep(1)
    return pd.DataFrame()

def get_market_sentiment():
    try: vix = fdr.DataReader('^VIX').iloc[-1, 0]
    except: vix = 20
    return {'VIX': vix, 'Label': '공포' if vix > 25 else ('탐욕' if vix < 15 else '중립')}

def get_google_news(query="미국 증시 OR 나스닥 OR 글로벌 경제", limit=5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, 'xml')
        items = soup.find_all('item', limit=limit)
        return [{'title': i.title.text, 'link': i.link.text} for i in items]
    except Exception as e:
        return [{'title': f'구글 뉴스 로드 실패', 'link': '#'}]

def get_naver_search(query, category='news', display=4):
    if not NAVER_CLIENT_ID: return [{'title': '네이버 API 키가 깃허브 시크릿에 없습니다.', 'link': '#', 'description': ''}]
    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": display, "sort": "sim"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        items = res.json().get('items', [])
        if not items: return [{'title': '검색 결과가 없습니다.', 'link': '#', 'description': ''}]
        return [{'title': i['title'].replace('<b>','').replace('</b>',''), 'link': i['link'], 'description': i.get('description', '').replace('<b>','').replace('</b>','')[:80]+'...'} for i in items]
    except: return [{'title': '네이버 API 호출 에러', 'link': '#', 'description': ''}]

def scrape_naver_blog_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        iframe = soup.find('iframe', id='mainFrame')
        if not iframe: return ""
        real_url = "https://blog.naver.com" + iframe['src']
        res2 = requests.get(real_url, timeout=10)
        soup2 = BeautifulSoup(res2.text, 'html.parser')
        content = soup2.find('div', class_='se-main-container')
        if content: return content.get_text(separator=' ', strip=True)
        return ""
    except: return ""

def get_ranto_ai_insights():
    url = "https://rss.blog.naver.com/ranto28.xml"
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        latest_item = root.find('.//item')
        if not latest_item: return "최신 글을 찾을 수 없습니다."
        
        link = latest_item.find('link').text
        title = latest_item.find('title').text
        blog_text = scrape_naver_blog_text(link)
        
        if not blog_text or not GEMINI_API_KEY:
            return f"[{title}] 글이 업데이트 되었습니다. (GEMINI API KEY를 깃허브 시크릿에 등록하시면 AI 요약이 제공됩니다.)"

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"다음은 주식/거시경제 전문 블로거 ranto28의 최신 글입니다. 이 글을 학습하여, 오늘 주식 투자를 하는 사람에게 도움이 될 핵심 시황 인사이트를 3개의 글머리 기호로 객관적이고 명확하게 요약해 주세요.\n\n[글 본문]\n{blog_text[:5000]}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 중 오류 발생: {e}"

def analyze_index_elliott(df, name):
    if df.empty: return {'name': name, 'current_price_str': '0', 'position': '데이터 없음'}
    curr = df['Close'].iloc[-1]
    high_60 = df['High'].rolling(window=60).max().iloc[-1]
    low_60 = df['Low'].rolling(window=60).min().iloc[-1]
    ratio = (curr - low_60) / (high_60 - low_60 + 1e-6)
    
    # 콤마 추가
    analysis = {'name': name, 'current_price_str': f"{curr:,.1f}", 'support': low_60, 'resistance': high_60}
    if ratio < 0.382: analysis['position'] = "조정/하락 파동 구간"
    elif ratio > 0.786: analysis['position'] = "강한 상승 파동 진행 중"
    else: analysis['position'] = "횡보 및 에너지 응축 구간"
    return analysis

def analyze_sector_rotation(sector_dfs):
    sector_perf = []
    for name, df in sector_dfs.items():
        if not df.empty and len(df) > 22:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-22]
            sector_perf.append({'Sector': name, 'Perf_1mo': (curr/prev - 1)*100})
    return pd.DataFrame(sector_perf)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}}
    report_data['market_sentiment'] = get_market_sentiment()
    
    # 지수 파동 분석
    report_data['index_analysis'] = []
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        analysis = analyze_index_elliott(df, name)
        report_data['index_analysis'].append(analysis)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            df['Close'].iloc[-100:].plot(ax=ax, color='black')
            ax.set_title(f"{name} Trend")
            ax.axhline(analysis['support'], color='red', ls='--')
            ax.axhline(analysis['resistance'], color='green', ls='--')
            report_data['plots'][f'chart_{name}'] = fig_to_base64(fig)
            plt.close(fig)

    # 섹터 로테이션
    sector_dfs = {name: get_stock_data(ticker, 60) for name, ticker in SECTOR_ETFs.items()}
    sector_df = analyze_sector_rotation(sector_dfs)
    if not sector_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sector_df.sort_values('Perf_1mo', ascending=False), x='Perf_1mo', y='Sector', palette='RdYlGn')
        ax.set_title("섹터별 자금 흐름 (최근 1개월 수익률 %)")
        report_data['plots']['sector_map'] = fig_to_base64(fig)
        plt.close(fig)

    # 뉴스 및 AI
    report_data['ranto_ai_insight'] = get_ranto_ai_insights()
    report_data['google_news'] = get_google_news()
    report_data['naver_news'] = get_naver_search("코스피 전망 OR 반도체 주식", 'news', 4)
    report_data['naver_blogs'] = get_naver_search("미국 주식 시황", 'blog', 4)

    # HTML 렌더링
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
