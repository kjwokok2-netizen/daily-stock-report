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
import praw

# ==========================================
# 1. Configuration & API Setup
# ==========================================
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = "StockReportBot/1.0"

OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)

INDICES = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11', 'NASDAQ': 'IXIC', 'PHLX Semico': 'SOX'}
MACRO_TICKERS = {'US10Y': 'DGS10', 'US02Y': 'DGS2', 'USD/KRW': 'USD/KRW', 'WTI': 'CL=F'}
SECTOR_ETFs = {
    'Semi-conductor': '091160', 'Secondary Batt': '305700', 'Energy': '117700',
    'Materials': '117460', 'Industrials': '117600', 'HealthCare': '117600',
    'Financials': '117460', 'Info Tech': '117690'
}

plt.style.use('ggplot')

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ==========================================
# 2. Data Acquisition Functions
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
        us10y = get_stock_data('DGS10')
        us02y = get_stock_data('DGS2')
        spread_val = us10y.iloc[-1, 0] - us02y.iloc[-1, 0]
    except: spread_val = 0
    try:
        vix = fdr.DataReader('^VIX').iloc[-1, 0]
    except: vix = 20
    sentiment = {'VIX': vix, 'Spread': spread_val, 'Label': 'Neutral'}
    if vix > 25: sentiment['Label'] = 'Fear'
    elif vix < 15: sentiment['Label'] = 'Greed'
    return sentiment

def get_naver_search(query, category='news', display=5):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET: return []
    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": display, "sort": "sim"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        items = res.json().get('items', [])
        return [{'title': i['title'].replace('<b>','').replace('</b>',''), 'link': i['link'], 'description': i.get('description', '').replace('<b>','').replace('</b>','')[:80]+'...'} for i in items]
    except: return []

def get_reddit_threads(subreddits=['stocks', 'wallstreetbets'], limit=5):
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET: return []
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
        results = []
        for sub in subreddits:
            for post in reddit.subreddit(sub).hot(limit=limit):
                if not post.stickied:
                    results.append({'sub': sub, 'title': post.title, 'url': post.url, 'ups': post.ups})
        return results
    except: return []

# ==========================================
# 3. Quantitative Analysis Functions
# ==========================================
def analyze_index_elliott(df, name):
    if df.empty: return {'name': name, 'current_price': 0, 'change': 0, 'position': 'Data Missing', 'main_scenario': '-', 'alt_scenario': '-', 'support': 0, 'resistance': 0}
    curr = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    high_60 = df['High'].rolling(window=60).max().iloc[-1]
    low_60 = df['Low'].rolling(window=60).min().iloc[-1]
    
    change = (curr / prev - 1) * 100
    ratio = (curr - low_60) / (high_60 - low_60 + 1e-6)
    
    analysis = {'name': name, 'current_price': curr, 'change': change, 'support': low_60, 'resistance': high_60, 'main_scenario': "상승 추세 유지 시도 중", 'alt_scenario': "주요 지지선 이탈 시 조정 확대"}
    if ratio < 0.382:
        analysis['position'] = "과매도 또는 하락 파동 진행"
        analysis['main_scenario'] = "하락 A파 완료 후 반등 시도 구간"
    elif ratio > 0.786:
        analysis['position'] = "상승 파동(3파 또는 5파) 진행 중"
        analysis['main_scenario'] = "강력한 상승 추세 형성"
    else:
        analysis['position'] = "박스권 조정 및 에너지 응축"
    return analysis

def analyze_sector_rotation(sector_dfs):
    sector_perf = []
    for name, df in sector_dfs.items():
        if not df.empty and len(df) > 22:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-22]
            sector_perf.append({'Sector': name, 'Perf_1mo': (curr/prev - 1)*100})
    return pd.DataFrame(sector_perf)

def screen_vcp_candidates(data_dict):
    candidates = []
    for ticker, df in data_dict.items():
        if df.empty or len(df) < 200: continue
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        current_close = df['Close'].iloc[-1]
        if current_close > df['MA200'].iloc[-1] and df['MA50'].iloc[-1] > df['MA200'].iloc[-1]:
            recent_range = (df['High'].iloc[-10:].max() / df['Low'].iloc[-10:].min() - 1) * 100
            rs_score = (df['Close'].iloc[-1] / df['Close'].iloc[-200]) * 100
            if recent_range < 7 and rs_score > 100:
                candidates.append({'ticker': ticker, 'name': f'ETF {ticker}', 'rs_score': rs_score, 'recent_range_pct': recent_range, 'breakout_potential': 'High' if df['Volume'].iloc[-1] > df['Volume'].rolling(window=20).mean().iloc[-1] * 1.5 else 'Medium'})
    return candidates

def analyze_macro_correlations():
    kospi_df = get_stock_data(INDICES['KOSPI'])
    corr_data = pd.DataFrame({'KOSPI': kospi_df['Close']})
    for name, ticker in MACRO_TICKERS.items():
        m_df = get_stock_data(ticker)
        if not m_df.empty: corr_data[name] = m_df['Close']
    corr_matrix = corr_data.dropna().corr()
    
    insights = []
    if not corr_matrix.empty and 'KOSPI' in corr_matrix:
        kospi_corrs = corr_matrix['KOSPI'].drop('KOSPI')
        for name, corr in kospi_corrs.items():
            if abs(corr) > 0.6:
                insights.append(f"{name}와(과) 강한 {'양' if corr > 0 else '음'}의 상관관계 ({corr:.2f})")
    if not insights: insights.append("최근 KOSPI와 주요 매크로 지표 간 강한 상관관계가 관찰되지 않음.")
    return corr_matrix, insights

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}, 'index_analysis': []}
    
    # 마켓 센티먼트
    report_data['market_sentiment'] = get_market_sentiment()
    
    # 지수 및 파동 분석
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        analysis = analyze_index_elliott(df, name)
        report_data['index_analysis'].append(analysis)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            df['Close'].iloc[-100:].plot(ax=ax, color='black')
            ax.set_title(f"{name} Trend")
            ax.axhline(analysis['support'], color='red', linestyle='--')
            ax.axhline(analysis['resistance'], color='green', linestyle='--')
            report_data['plots'][f'index_elliott_{name}'] = fig_to_base64(fig)
            plt.close(fig)

    # 섹터 로테이션 & VCP 스크리닝
    sector_dfs = {name: get_stock_data(ticker, 252) for name, ticker in SECTOR_ETFs.items()}
    sector_df = analyze_sector_rotation(sector_dfs)
    if not sector_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sector_df.sort_values('Perf_1mo', ascending=False), x='Perf_1mo', y='Sector', ax=ax, palette='viridis')
        ax.set_title("1-Month Sector Rotation")
        report_data['plots']['sector_map'] = fig_to_base64(fig)
        plt.close(fig)
        report_data['sector_performance'] = sector_df.to_dict('records')
    
    report_data['vcp_candidates'] = screen_vcp_candidates(sector_dfs)
    
    # 매크로 상관관계
    corr_matrix, report_data['macro_insights'] = analyze_macro_correlations()
    if not corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        report_data['plots']['macro_heatmap'] = fig_to_base64(fig)
        plt.close(fig)

    # API 트렌드 크롤링 (네이버, 레딧)
    keyword = "주식 시장 전망 OR 반도체"
    report_data['naver_news'] = get_naver_search(keyword, 'news', 5)
    report_data['naver_blogs'] = get_naver_search(keyword, 'blog', 5)
    report_data['reddit_threads'] = get_reddit_threads()

    # HTML 렌더링
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
