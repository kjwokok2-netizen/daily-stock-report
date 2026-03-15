import os
import requests
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import time

# ==========================================
# 1. Configuration & API Setup
# ==========================================
# 주완님이 이미 가지고 계신 네이버 키만 사용합니다.
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')

OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 지수 및 매크로 티커
INDICES = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11', 'NASDAQ': 'IXIC', 'PHLX Semico': 'SOX'}
MACRO_TICKERS = {'US10Y': 'DGS10', 'US02Y': 'DGS2', 'USD/KRW': 'USD/KRW', 'WTI': 'CL=F'}
SECTOR_ETFs = {
    '반도체': '091160', '2차전지': '305700', '에너지': '117700',
    '철강/소재': '117460', '산업재': '117600', '헬스케어': '117620',
    '금융': '117460', 'IT소프트': '117690'
}

plt.style.use('ggplot')

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
    for i in range(3):
        try:
            if ticker in ['DGS10', 'DGS2']:
                df = fdr.DataReader(ticker, start, end, data_source='fred')
            else:
                df = fdr.DataReader(ticker, start, end)
            if not df.empty: return df
        except: time.sleep(1)
    return pd.DataFrame()

def get_market_sentiment():
    try:
        us10y = get_stock_data('DGS10', 5)
        us02y = get_stock_data('DGS2', 5)
        spread_val = us10y.iloc[-1, 0] - us02y.iloc[-1, 0]
    except: spread_val = 0
    try:
        vix = fdr.DataReader('^VIX').iloc[-1, 0]
    except: vix = 20
    sentiment = {'VIX': vix, 'Spread': spread_val, 'Label': '중립'}
    if vix > 25: sentiment['Label'] = '공포'
    elif vix < 15: sentiment['Label'] = '탐욕'
    return sentiment

def get_naver_search(query, category='news', display=8):
    if not NAVER_CLIENT_ID: return []
    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": display, "sort": "sim"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        items = res.json().get('items', [])
        return [{'title': i['title'].replace('<b>','').replace('</b>',''), 'link': i['link'], 'description': i.get('description', '').replace('<b>','').replace('</b>','')[:100]+'...'} for i in items]
    except: return []

# ==========================================
# 3. Quantitative Analysis (파동, 섹터, VCP)
# ==========================================
def analyze_index_elliott(df, name):
    if df.empty: return {'name': name, 'current_price': 0, 'change': 0, 'position': '데이터 없음'}
    curr = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    high_60 = df['High'].rolling(window=60).max().iloc[-1]
    low_60 = df['Low'].rolling(window=60).min().iloc[-1]
    change = (curr / prev - 1) * 100
    ratio = (curr - low_60) / (high_60 - low_60 + 1e-6)
    
    analysis = {'name': name, 'current_price': curr, 'change': change, 'support': low_60, 'resistance': high_60}
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
# 4. Main Execution
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}}
    report_data['market_sentiment'] = get_market_sentiment()
    
    # 지수 분석 & 차트
    report_data['index_analysis'] = []
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        analysis = analyze_index_elliott(df, name)
        report_data['index_analysis'].append(analysis)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            df['Close'].iloc[-100:].plot(ax=ax, color='black')
            ax.set_title(f"{name} Wave Analysis")
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
        ax.set_title("1-Month Sector Rotation Map")
        report_data['plots']['sector_map'] = fig_to_base64(fig)
        plt.close(fig)
        report_data['sector_performance'] = sector_df.to_dict('records')

    # 네이버 크롤링 (주완님의 키워드)
    keywords = "코스피 전망 반도체 주식"
    report_data['naver_news'] = get_naver_search(keywords, 'news')
    report_data['naver_blogs'] = get_naver_search(keywords, 'blog')

    # HTML 렌더링
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
