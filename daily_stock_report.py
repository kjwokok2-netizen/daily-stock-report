import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import time

# ==========================================
# 1. Configuration & Setup
# ==========================================
OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 지수 티커 (안정적인 티커 위주)
INDICES = {
    'KOSPI': 'KS11',
    'KOSDAQ': 'KQ11',
    'NASDAQ': 'IXIC',
    'PHLX Semico': 'SOX'
}

# 거시 지표 티커 (FRED 소스로 변경하여 안정성 확보)
MACRO_TICKERS = {
    'US10Y': 'DGS10', # US 10-year Treasury Yield (FRED)
    'US02Y': 'DGS2',  # US 2-year Treasury Yield (FRED)
    'USD/KRW': 'USD/KRW',
    'WTI': 'CL=F'     # Crude Oil (Yahoo)
}

SECTOR_ETFs = {
    'Semi-conductor': '091160', 'Secondary Batt': '305700', 'Energy': '117700',
    'Materials': '117460', 'Industrials': '117600', 'HealthCare': '117600'
}

plt.style.use('ggplot') # 기본 스타일 사용

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
    # API 요청 실패 대비 재시도 로직
    for i in range(3):
        try:
            # 금리 데이터의 경우 FRED에서 가져오도록 강제
            if ticker in ['DGS10', 'DGS2']:
                df = fdr.DataReader(ticker, start, end, data_source='fred')
            else:
                df = fdr.DataReader(ticker, start, end)
            if not df.empty: return df
        except:
            time.sleep(1)
    return pd.DataFrame()

def get_market_sentiment():
    try:
        us10y = get_stock_data('DGS10')
        us02y = get_stock_data('DGS2')
        spread_val = us10y.iloc[-1, 0] - us02y.iloc[-1, 0]
    except:
        spread_val = 0
    
    try:
        vix = fdr.DataReader('^VIX') # VIX는 앞에 ^를 붙이는 것이 더 안정적임
        vix_val = vix.iloc[-1, 0]
    except:
        vix_val = 20 # 데이터 부재 시 중립값
        
    sentiment = {'VIX': vix_val, 'Spread': spread_val, 'Label': 'Neutral'}
    if sentiment['VIX'] > 25: sentiment['Label'] = 'Fear'
    elif sentiment['VIX'] < 15: sentiment['Label'] = 'Greed'
    return sentiment

def get_summarized_news():
    news = []
    try:
        url = 'https://news.google.com/rss/search?q=주식시장+경제+트렌드&hl=ko&gl=KR&ceid=KR:ko'
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, features='xml')
        items = soup.find_all('item', limit=5)
        for item in items:
            news.append({'title': item.title.text, 'link': item.link.text, 'summary': '주요 뉴스 헤드라인입니다.'})
    except:
        news.append({'title': '뉴스를 불러오지 못했습니다.', 'link': '#', 'summary': ''})
    return news

# ==========================================
# 3. Analysis Functions (핵심 로직 유지)
# ==========================================
def analyze_index_elliott(ticker_df, name):
    if ticker_df.empty: return {'name': name, 'current_price': 0, 'change': 0, 'position': 'Data Missing', 'main_scenario': '-', 'alt_scenario': '-', 'support': 0, 'resistance': 0}
    
    window = 20
    ticker_df['Local_High'] = ticker_df['High'].rolling(window=window).max()
    ticker_df['Local_Low'] = ticker_df['Low'].rolling(window=window).min()
    
    current_price = ticker_df['Close'].iloc[-1]
    prev_close = ticker_df['Close'].iloc[-2]
    last_peak = ticker_df['High'].rolling(window=60).max().iloc[-1]
    last_trough = ticker_df['Low'].rolling(window=60).min().iloc[-1]
    
    change = (current_price / prev_close - 1) * 100
    retracement = (current_price - last_trough) / (last_peak - last_trough + 1e-6)
    
    analysis = {
        'name': name, 'current_price': current_price, 'change': change,
        'support': last_trough, 'resistance': last_peak,
        'main_scenario': "상승 추세 유지 시도 중", 'alt_scenario': "주요 지지선 이탈 시 조정 확대"
    }
    
    if retracement < 0.382:
        analysis['position'] = "과매도 또는 하락 파동 진행"
        analysis['main_scenario'] = "하락 A파 완료 후 반등 시도 구간"
    elif retracement > 0.786:
        analysis['position'] = "상승 파동(3파 또는 5파) 진행 중"
        analysis['main_scenario'] = "강력한 상승 추세 형성"
    else:
        analysis['position'] = "박스권 조정 및 에너지 응축"
        
    return analysis

def analyze_sector_rotation():
    sector_perf = []
    for name, ticker in SECTOR_ETFs.items():
        df = get_stock_data(ticker, days=40)
        if not df.empty:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-22] if len(df) > 22 else df['Close'].iloc[0]
            sector_perf.append({'Sector': name, 'Perf_1mo': (curr/prev - 1)*100})
    return pd.DataFrame(sector_perf)

# ==========================================
# 4. Main Process
# ==========================================
def main():
    report_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'plots': {}, 'index_analysis': []}
    
    report_data['market_sentiment'] = get_market_sentiment()
    report_data['news'] = get_summarized_news()
    report_data['smart_money'] = {'foreign': {'top_buy': ['삼성전자', 'SK하이닉스']}, 'institutional': {'top_buy': ['현대차', '기아']}, 'consensus_buy': ['현대차']}

    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        analysis = analyze_index_elliott(df, name)
        report_data['index_analysis'].append(analysis)
        
        # 차트 생성
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            df['Close'].iloc[-100:].plot(ax=ax, color='black', title=f"{name} Trend")
            ax.axhline(analysis['support'], color='red', linestyle='--')
            ax.axhline(analysis['resistance'], color='green', linestyle='--')
            report_data['plots'][f'index_elliott_{name}'] = fig_to_base64(fig)
            plt.close(fig)

    sector_df = analyze_sector_rotation()
    if not sector_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sector_df.sort_values('Perf_1mo'), x='Perf_1mo', y='Sector', ax=ax)
        report_data['plots']['sector_map'] = fig_to_base64(fig)
        plt.close(fig)
        report_data['sector_performance'] = sector_df.to_dict('records')
    
    report_data['vcp_candidates'] = [{'ticker': '005930', 'name': '삼성전자', 'rs_score': 85, 'recent_range_pct': 2.1, 'breakout_potential': 'High'}]
    report_data['macro_insights'] = ["미국 10년물 금리 변동성 주시", "환율 안정화 여부가 외인 수급의 핵심"]

    # HTML 렌더링
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(template.render(data=report_data))

if __name__ == "__main__":
    main()
