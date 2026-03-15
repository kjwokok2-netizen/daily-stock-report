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

# ==========================================
# 1. Configuration & API Setup
# ==========================================
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')
# 새로 추가된 AI 분석용 키 (무료)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)

INDICES = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11', 'NASDAQ': 'IXIC'}
SECTOR_ETFs = {'반도체': '091160', '2차전지': '305700', '에너지': '117700', '금융': '117460'}

plt.style.use('ggplot')

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ==========================================
# 2. Data Acquisition & AI Analysis
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

def scrape_naver_blog_text(url):
    """네이버 블로그 본문 iframe 우회 추출"""
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
    """란토님 블로그 최신글을 AI로 요약 분석하여 리포트에 도출"""
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
            return f"[{title}] 글이 업데이트 되었습니다. (AI 분석 대기 중)"

        # Gemini AI 프롬프트 명령
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        다음은 주식 시장과 거시 경제를 다루는 유명 블로거(ranto28)의 최신 글 본문입니다.
        이 글을 완벽하게 학습한 뒤, 오늘 주식 투자를 하는 패션 MD에게 도움이 될 만한 핵심 시황 인사이트를 3개의 글머리 기호(Bullet points)로 아주 명확하게 요약해 주세요.
        말투는 객관적이고 전문적인 리포트 톤으로 작성하세요.
        
        [글 본문]
        {blog_text[:5000]}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 중 오류 발생: {e}"

def analyze_index_elliott(df, name):
    if df.empty: return {'name': name, 'current_price': 0, 'change': 0, 'position': '데이터 없음'}
    curr = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    high_60 = df['High'].rolling(window=60).max().iloc[-1]
    low_60 = df['Low'].rolling(window=60).min().iloc[-1]
    
    ratio = (curr - low_60) / (high_60 - low_60 + 1e-6)
    analysis = {'name': name, 'current_price': curr, 'change': (curr/prev-1)*100, 'support': low_60, 'resistance': high_60}
    if ratio < 0.382: analysis['position'] = "조정/하락 파동 구간"
    elif ratio > 0.786: analysis['position'] = "강한 상승 파동 진행 중"
    else: analysis['position'] = "횡보 및 에너지 응축 구간"
    return analysis

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}}
    report_data['market_sentiment'] = get_market_sentiment()
    
    report_data['index_analysis'] = []
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        analysis = analyze_index_elliott(df, name)
        report_data['index_analysis'].append(analysis)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 3))
            df['Close'].iloc[-100:].plot(ax=ax, color='black')
            ax.set_title(f"{name} Trend")
            ax.axhline(analysis['support'], color='red', ls='--')
            report_data['plots'][f'chart_{name}'] = fig_to_base64(fig)
            plt.close(fig)

    # 핵심: AI가 학습하여 도출한 란토님 인사이트
    report_data['ranto_ai_insight'] = get_ranto_ai_insights()

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
