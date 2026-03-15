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

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Output and Input paths
OUTPUT_DIR = "docs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define Tickers
INDICES = {
    'KOSPI': 'KS11',
    'KOSDAQ': 'KQ11',
    'NASDAQ': 'IXIC',
    'PHLX Semico': 'SOX'
}
MACRO_TICKERS = {
    'US10Y': 'US10Y', # US 10-year Treasury Yield
    'US02Y': 'US02Y', # US 2-year Treasury Yield
    'USD/KRW': 'USD/KRW',
    'WTI': 'WTI' # Crude Oil
}
# Example Sector ETFs (for Korea)
SECTOR_ETFs = {
    'Energy': '117700', 'Materials': '117460', 'Industrials': '117600',
    'Cons Disc': '117690', 'Cons Staples': '117700', 'HealthCare': '117600',
    'Financials': '117460', 'Info Tech': '117690', 'Comm Svc': '117700',
    'Utilities': '117600', 'Semi-conductor': '091160', 'Secondary Batt': '305700'
}

# Style setting for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 string for HTML embedding."""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ==========================================
# 2. Data Acquisition Functions
# ==========================================
def get_stock_data(ticker, days=252):
    """Fetches historical stock data."""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = fdr.DataReader(ticker, start, end)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_market_sentiment():
    """Fetches key market sentiment indices (Fear & Greed placeholder)."""
    # Note: Real-time Fear & Greed API is usually paid. This is a heuristic.
    # We can use US 10Y-2Y spread and VIX as proxies.
    spread = fdr.DataReader('US10Y') - fdr.DataReader('US02Y')
    vix = fdr.DataReader('VIX')
    sentiment = {
        'VIX': vix.iloc[-1, 0],
        'Spread': spread.iloc[-1, 0],
        'Label': 'Neutral' # Placeholder
    }
    if sentiment['VIX'] > 25: sentiment['Label'] = 'Fear'
    elif sentiment['VIX'] < 15: sentiment['Label'] = 'Greed'
    return sentiment

def get_smart_money_data():
    """Fetches recent Institutional/Foreign net buying for KOSPI top stocks."""
    # This often requires scraping/more advanced APIs.
    # Placeholder: Return a dictionary for structure.
    print("Warning: Real Smart Money data often requires advanced APIs. Returning placeholder structure.")
    return {
        'foreign': {'top_buy': ['Stock A', 'Stock B'], 'top_sell': ['Stock X', 'Stock Y']},
        'institutional': {'top_buy': ['Stock C', 'Stock A'], 'top_sell': ['Stock Y', 'Stock Z']},
        'consensus_buy': ['Stock A'] # Found in both top_buy
    }

def get_summarized_news():
    """Fetches and summarizes key news headlines."""
    # RSS feed example (e.g., Google News)
    # real summarization requires LLMs or text summarization libraries (nltk/gensim).
    # Placeholder: Simple headline retrieval.
    url = 'https://news.google.com/rss/search?q=주식시장+OR+경제+OR+KOSPI&hl=ko&gl=KR&ceid=KR:ko'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features='xml')
    items = soup.findall('item', limit=5)
    news = []
    for item in items:
        news.append({'title': item.title.text, 'link': item.link.text, 'summary': 'Summary not available in this simplified script.'})
    return news

# ==========================================
# 3. Quantitative Analysis Functions
# ==========================================
def analyze_index_elliott(ticker_df, name):
    """Performs heuristic Elliott Wave analysis for indices."""
    if ticker_df.empty: return {}
    
    # Simple heuristic: Peak-to-Peak/Trough-to-Trough detection
    # Note: True automated counting is extremely complex.
    window = 20
    ticker_df['Local_High'] = ticker_df['High'].rolling(window=window).max()
    ticker_df['Local_Low'] = ticker_df['Low'].rolling(window=window).min()

    last_peak = ticker_df[ticker_df['High'] == ticker_df['Local_High']].iloc[-1]
    last_trough = ticker_df[ticker_df['Low'] == ticker_df['Local_Low']].iloc[-1]
    
    current_price = ticker_df['Close'].iloc[-1]
    
    analysis = {
        'name': name,
        'current_price': current_price,
        'change': (current_price / ticker_df['Close'].iloc[-2] - 1) * 100,
        'last_peak_price': last_peak['High'],
        'last_trough_price': last_trough['Low'],
        'position': 'Between'
    }
    
    # Main/Alt Count Heuristics (Very Simplified)
    # Assume 5-wave impulse up as default.
    # If currently above last peak -> Potential 3 or 5?
    # If below -> Potential 조정파 or 시작?
    
    if current_price > last_peak['High']:
        analysis['position'] = 'New High (Potential Wave 3 or 5)'
        analysis['support'] = last_peak['High']
        analysis['resistance'] = current_price * 1.05 # Simple projection
    elif current_price < last_trough['Low']:
        analysis['position'] = 'New Low (Potential Corrective Wave or reversal)'
        analysis['support'] = current_price * 0.95
        analysis['resistance'] = last_trough['Low']
    else:
        # Check retracement levels from the move (trough to peak)
        retracement = (current_price - last_trough['Low']) / (last_peak['High'] - last_trough['Low'])
        analysis['position'] = f'Consolidating ({retracement:.0%} retracement from recent move)'
        # Define scenarios based on retracement levels
        if retracement < 0.382:
            analysis['main_scenario'] = "Corrective Wave A complete? Moving into Wave B bounce?"
            analysis['alt_scenario'] = "Corrective Wave A continues downwards towards 0.5/0.618 level."
        elif 0.382 <= retracement <= 0.618:
            analysis['main_scenario'] = "Potential Wave 2 or 4 complete (shallow retracement). Preparing for Wave 3 or 5?"
            analysis['alt_scenario'] = "Corrective Wave B complete? Wave C downwards beginning."
        else:
            analysis['main_scenario'] = "Deep Corrective Wave. Potential trend change or ABC complete?"
            analysis['alt_scenario'] = "Prolonged corrective structure. Need base building."
            
        analysis['support'] = last_trough['Low']
        analysis['resistance'] = last_peak['High']
        
    return analysis

def screen_vcp_candidates(data_dict, min_rs=70, max_volatility_window=10):
    """Screens stocks for VCP-like patterns."""
    # Minervini criteria simplified:
    # 1. Price above 200D MA
    # 2. 200D MA trending up (current > 1 month ago)
    # 3. 50D MA above 200D MA
    # 4. Current price above 50D MA
    # 5. Volatility contraction (narrowing price range)
    # 6. Relative Strength (RS) > threshold
    # 7. Volume contraction (volume declining while price range narrows)
    
    candidates = []
    print(f"Screening {len(data_dict)} stocks for VCP patterns...")
    for ticker, df in data_dict.items():
        if df.empty or len(df) < 252: continue
        
        # Calculate MAs
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 1-4. Trend Criteria
        current_close = df['Close'].iloc[-1]
        ma50 = df['MA50'].iloc[-1]
        ma200 = df['MA200'].iloc[-1]
        ma200_1mo = df['MA200'].iloc[-21] if len(df) > 21 else ma200
        
        if not (current_close > ma200 and ma200 > ma200_1mo and ma50 > ma200 and current_close > ma50): continue
        
        # 5-7. VCP-like Contraction Criteria (simplified)
        # Check max-to-min range in the last max_volatility_window days
        recent_range = (df['High'].iloc[-max_volatility_window:].max() / df['Low'].iloc[-max_volatility_window:].min() - 1) * 100
        # Placeholder RS check (requires comparison with market index, simplified here)
        print("Note: True Relative Strength (RS) requires market comparison. Simplified heuristic used.")
        df['RS_Heuristic'] = (df['Close'] / df['Close'].shift(252)) * 100 # Simple price change over 1 year
        rs_score = df['RS_Heuristic'].iloc[-1]
        
        if recent_range < 5 and rs_score > 100: # Narrow range and positive 1-year performance
            candidates.append({
                'ticker': ticker,
                'name': 'Ticker ' + ticker, # Ticker to name mapping needed for real use
                'recent_range_pct': recent_range,
                'rs_score': rs_score,
                'breakout_potential': 'High' if df['Volume'].iloc[-1] > df['Volume'].rolling(window=20).mean().iloc[-1] * 1.5 else 'Medium'
            })
            
    return pd.DataFrame(candidates)

def analyze_sector_rotation(sector_dfs):
    """Calculates sector performance (1-week, 1-month) and creates performance treemap data."""
    sector_perf = []
    for name, df in sector_dfs.items():
        if df.empty: continue
        
        # Calculate performance
        current = df['Close'].iloc[-1]
        prev_wk = df['Close'].iloc[-6] if len(df) > 6 else df['Close'].iloc[0]
        prev_mo = df['Close'].iloc[-22] if len(df) > 22 else df['Close'].iloc[0]
        
        # Calculate recent trading volume/money (proxy for money flow)
        money_flow = df['Close'].iloc[-5:].mean() * df['Volume'].iloc[-5:].mean()
        
        sector_perf.append({
            'Sector': name,
            'Perf_1wk': (current / prev_wk - 1) * 100,
            'Perf_1mo': (current / prev_mo - 1) * 100,
            'Money_Flow_Index': money_flow
        })
        
    return pd.DataFrame(sector_perf)

def analyze_macro_correlations():
    """Calculates correlations between KOSPI and macro variables."""
    # Note: Correlation analysis needs time to develop meaningful results.
    kospi_df = get_stock_data(INDICES['KOSPI'])
    correlation_data = pd.DataFrame({'KOSPI': kospi_df['Close']})
    
    for name, ticker in MACRO_TICKERS.items():
        macro_df = get_stock_data(ticker)
        if not macro_df.empty:
            correlation_data[name] = macro_df['Close']
            
    correlation_data = correlation_data.dropna()
    corr_matrix = correlation_data.corr()
    
    # Analyze main insights
    insights = []
    kospi_correlations = corr_matrix['KOSPI'].drop('KOSPI')
    strong_corrs = kospi_correlations[abs(kospi_correlations) > 0.6]
    for name, corr in strong_corrs.items():
        if corr > 0: insights.append(f"Strong Positive Correlation with {name}: {corr:.2f}")
        else: insights.append(f"Strong Negative Correlation with {name}: {corr:.2f}")
        
    if not insights: insights.append("No strong (>0.6) correlations detected with the predefined macro variables in the recent period.")
        
    return corr_matrix, insights

# ==========================================
# 4. Visualization Functions
# ==========================================
def plot_index_elliott(ticker_df, analysis):
    """Generates Elliott wave visualization for index report."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ticker_df['Close'].iloc[-120:].plot(ax=ax, color='black', lw=1.5, label='Close')
    
    # Plot Local Highs/Lows as points
    ticker_df[ticker_df['High'] == ticker_df['Local_High']].iloc[-120:]['High'].plot(ax=ax, style='go', label='Local Highs')
    ticker_df[ticker_df['Low'] == ticker_df['Local_Low']].iloc[-120:]['Low'].plot(ax=ax, style='ro', label='Local Lows')
    
    # Plot Support/Resistance lines
    ax.axhline(y=analysis['support'], color='red', linestyle='--', lw=1, label=f'Support ({analysis["support"]:,.0f})')
    ax.axhline(y=analysis['resistance'], color='green', linestyle='--', lw=1, label=f'Resistance ({analysis["resistance"]:,.0f})')
    
    ax.set_title(f"{analysis['name']} Elliott Wave Heuristics", fontsize=14, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    return fig

def plot_sector_rotation(sector_perf):
    """Creates a performance bar chart for sector rotation analysis."""
    if sector_perf.empty: return None
    
    # Bar chart for 1-month performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sector_perf_sorted = sector_perf.sort_values(by='Perf_1mo', ascending=False)
    
    # Color bar based on performance sign
    sector_perf_sorted['color'] = sector_perf_sorted['Perf_1mo'].apply(lambda x: '#34A853' if x > 0 else '#EA4335') # Green/Red

    ax.barh(sector_perf_sorted['Sector'], sector_perf_sorted['Perf_1mo'], color=sector_perf_sorted['color'], edgecolor='white', alpha=0.8)
    
    # Add text labels on bars
    for i, v in enumerate(sector_perf_sorted['Perf_1mo']):
        ax.text(v, i, f" {v:.1f}%", color='black', va='center', fontweight='bold' if v > 0 else 'normal')
        
    ax.axvline(x=0, color='black', lw=1)
    ax.set_title("Sector Rotation: 1-Month Performance Map", fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage Change (%)', fontsize=12)
    ax.set_ylabel('Sector ETF', fontsize=12)
    ax.invert_yaxis() # Leading sector on top
    plt.tight_layout()
    return fig

def plot_macro_correlations(corr_matrix):
    """Generates a correlation heatmap for macro variables."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1, linewidths=.5, cbar_kws={'shrink': 0.8})
    ax.set_title("KOSPI & Macro Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# 5. Main Execution & HTML Generation
# ==========================================
def main():
    print(f"--- Starting Daily Stock Report Generation for {datetime.now().strftime('%Y-%m-%d')} ---")
    report_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'plots': {},
        'index_analysis': []
    }
    
    # 1. Fetch data
    market_sentiment = get_market_sentiment()
    report_data['market_sentiment'] = market_sentiment
    
    smart_money = get_smart_money_data()
    report_data['smart_money'] = smart_money
    
    summarized_news = get_summarized_news()
    report_data['news'] = summarized_news
    
    # 2. Analyze Index & Elliot
    index_analysis_raw = []
    for name, ticker in INDICES.items():
        df = get_stock_data(ticker)
        if not df.empty:
            analysis = analyze_index_elliott(df, name)
            index_analysis_raw.append(analysis)
            fig = plot_index_elliott(df, analysis)
            report_data['plots'][f'index_elliott_{name}'] = fig_to_base64(fig)
            plt.close(fig)
            
    # Structure index analysis for the report
    for name in INDICES.keys():
        # Find analysis for the name
        analysis = next((a for a in index_analysis_raw if a['name'] == name), None)
        if analysis:
            report_data['index_analysis'].append(analysis)

    # 3. Analyze Sectors
    sector_dfs = {}
    for name, ticker in SECTOR_ETFs.items():
        sector_dfs[name] = get_stock_data(ticker, days=60) # Only need recent data
        
    sector_perf = analyze_sector_rotation(sector_dfs)
    report_data['sector_performance'] = sector_perf.to_dict('records')
    
    fig_sector = plot_sector_rotation(sector_perf)
    if fig_sector:
        report_data['plots']['sector_map'] = fig_to_base64(fig_sector)
        plt.close(fig_sector)
        
    # 4. Screen VCP (Example using Sector ETFs for demo, real use should have many more tickers)
    print("WARNING: Real VCP screening requires hundreds of tickers. Demo using predefined Sector ETFs.")
    vcp_results = screen_vcp_candidates(sector_dfs)
    report_data['vcp_candidates'] = vcp_results.to_dict('records')
    
    # 5. Analyze Macro Correlations
    corr_matrix, corr_insights = analyze_macro_correlations()
    report_data['macro_insights'] = corr_insights
    fig_macro = plot_macro_correlations(corr_matrix)
    report_data['plots']['macro_heatmap'] = fig_to_base64(fig_macro)
    plt.close(fig_macro)
    
    # 6. Generate HTML
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('daily_report_template.html')
    html_out = template.render(data=report_data)
    
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_out)
        
    print(f"--- Daily Stock Report successfully generated at {output_path} ---")

if __name__ == "__main__":
    main()
