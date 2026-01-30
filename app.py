import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr

# --- [Log Settings] ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- [Page Config] ---
st.set_page_config(page_title="Quant Nexus Pro", layout="wide", initial_sidebar_state="expanded")

# --- [Styling] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.1rem !important; }
    .streamlit-expanderHeader { background-color: #1e293b; color: white; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- [Constants] ---
RISK_KEYWORDS = ['유상증자', '횡령', '배임', '상장폐지', '감사의견', '거래정지', '불성실공시']

# --- [Data Layer] ---
@st.cache_data(ttl=3600*12)
def load_market_data():
    df_krx = fdr.StockListing('KRX')
    df_etf = fdr.StockListing('ETF/KR')
    
    # KOSPI 200 + KOSDAQ 100 filtering by Marcap
    sort_col = 'Marcap' if 'Marcap' in df_krx.columns else 'Close'
    df_kospi = df_krx[df_krx['Market'] == 'KOSPI'].sort_values(by=sort_col, ascending=False).head(200)
    df_kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].sort_values(by=sort_col, ascending=False).head(100)
    df_stocks = pd.concat([df_kospi, df_kosdaq])
    
    # ETF Top 50
    etf_sort_col = 'Marcap' if 'Marcap' in df_etf.columns else 'Amount'
    df_etf_top = df_etf.sort_values(by=etf_sort_col, ascending=False).head(50)

    # Sector Classification
    sectors = {
        "📊 Top 50 ETFs": [],
        "🚀 Semiconductor & IT": [],
        "🔋 Battery & Chemicals": [],
        "💊 Pharma & Biotech": [],
        "💰 Finance & Value-up": [],
        "🚗 Auto & Transport": [],
        "🛡️ Defense/Ship/Infra": [],
        "💄 Consumer/Food/Ent": [],
        "🎮 Platform & Game": [],
        "🌈 Other Large Caps": []
    }
    
    ticker_name_map = {}

    def process_tickers(df, is_etf=False):
        has_sector = 'Sector' in df.columns
        for _, row in df.iterrows():
            code = str(row['Symbol'])
            name = str(row['Name'])
            suffix = ".KS" if is_etf or (row.get('Market', 'KOSPI') == 'KOSPI') else ".KQ"
            yf_ticker = code + suffix
            ticker_name_map[yf_ticker] = name
            
            if is_etf:
                sectors["📊 Top 50 ETFs"].append(yf_ticker)
                continue

            combined_text = (name + " " + (str(row['Sector']) if has_sector and pd.notnull(row['Sector']) else "")).lower()
            
            if any(x in combined_text for x in ['semiconductor', 'elec', 'sk hynix', 'samsung el', 'hpsp']): sectors["🚀 Semiconductor & IT"].append(yf_ticker)
            elif any(x in combined_text for x in ['battery', 'chem', 'energy', 'ecopro', 'posco fut', 'kumyang']): sectors["🔋 Battery & Chemicals"].append(yf_ticker)
            elif any(x in combined_text for x in ['pharma', 'bio', 'life', 'alteogen', 'hlb']): sectors["💊 Pharma & Biotech"].append(yf_ticker)
            elif any(x in combined_text for x in ['finance', 'bank', 'insur', 'meritz', 'kb', 'shinhan']): sectors["💰 Finance & Value-up"].append(yf_ticker)
            elif any(x in combined_text for x in ['auto', 'motor', 'kia', 'hyundai', 'ship', 'air']): sectors["🚗 Auto & Transport"].append(yf_ticker)
            elif any(x in combined_text for x in ['heavy', 'defense', 'power', 'hanwha', 'rotem', 'lignex1']): sectors["🛡️ Defense/Ship/Infra"].append(yf_ticker)
            elif any(x in combined_text for x in ['food', 'cosmetic', 'ent', 'hybe', 'amore']): sectors["💄 Consumer/Food/Ent"].append(yf_ticker)
            elif any(x in combined_text for x in ['soft', 'game', 'internet', 'naver', 'kakao', 'krafton']): sectors["🎮 Platform & Game"].append(yf_ticker)
            else: sectors["🌈 Other Large Caps"].append(yf_ticker)

    process_tickers(df_stocks, is_etf=False)
    process_tickers(df_etf_top, is_etf=True)
    return sectors, ticker_name_map

with st.spinner("Initializing Quant Engine..."):
    SECTORS, TICKER_MAP = load_market_data()

# --- [Factor Layer] ---
def compute_factors(df):
    """
    Computes raw factor values from price history.
    """
    if len(df) < 60: return None
    
    # 1. Momentum Factors
    # Price Momentum: recent strength
    ret20 = df['Close'].pct_change(20).iloc[-1]  # 1 Month Return
    ret60 = df['Close'].pct_change(60).iloc[-1]  # 3 Month Return
    
    # 2. Liquidity Factors
    # Turnover Z (Time-series): How is today's volume compared to avg?
    turnover = df['Volume'] * df['Close']
    turnover_z = (turnover.iloc[-1] - turnover.mean()) / (turnover.std() + 1e-9)
    
    # 3. Risk Factors
    # MDD (Max Drawdown): Downside risk
    cummax = df['Close'].cummax()
    drawdown = (df['Close'] / cummax) - 1
    mdd = drawdown.min() # Should be negative
    
    # Volatility (Annualized)
    vol = df['Close'].pct_change().std() * np.sqrt(252)
    
    return {
        "ret20": ret20,
        "ret60": ret60,
        "turnover_z": turnover_z,
        "mdd": mdd,
        "vol": vol,
        "current_price": df['Close'].iloc[-1],
        "chart_data": df['Close'].tail(60)
    }

# --- [Risk Layer] ---
def check_event_risk(ticker):
    """
    Checks for critical risk keywords in recent news.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return False, []
        
        detected_risks = []
        for article in news[:3]:
            title = article.get('title', '').lower()
            # YFinance news is often English, but we check provided KR keywords just in case
            # + English equivalents
            for k in RISK_KEYWORDS:
                if k in title: detected_risks.append(k)
            
            # English risk keywords for Yahoo Finance
            en_risks = ['embezzle', 'delisting', 'suspension', 'audit', 'breach']
            for k in en_risks:
                if k in title: detected_risks.append(k)
                
        if detected_risks:
            return True, list(set(detected_risks))
        return False, []
    except:
        return False, []

# --- [Scoring Layer] ---
def cross_sectional_score(factor_df):
    """
    Calculates Z-scores across the universe and computes final weighted score.
    Logic: High Momentum + High Liquidity - High Risk
    """
    if factor_df.empty: return factor_df
    
    # Standardize (Z-Score)
    # (x - mean) / std -> converts absolute values to relative ranking (sigma)
    z = (factor_df - factor_df.mean()) / (factor_df.std() + 1e-9)
    
    # Weighted Score Formula
    # Momentum (60%) + Liquidity (20%) - Risk (20%)
    # Note: MDD is negative, so we ADD it to penalize deep drawdowns? 
    # Usually: Higher MDD (closer to 0) is better. Lower MDD (e.g. -50%) is worse.
    # Z-score of -50% MDD will be low (negative). Z-score of -5% MDD will be high (positive).
    # So we want High Z(MDD).
    # Volatility: High Vol is bad. We want Low Vol. So we subtract Z(Vol).
    
    final_score = (
        z['ret20'] * 0.3 +       # Short-term Mom
        z['ret60'] * 0.3 +       # Mid-term Mom
        z['turnover_z'] * 0.2 +  # Liquidity Shock
        z['mdd'] * 0.1 -         # Drawdown stability (Higher is better)
        z['vol'] * 0.1           # Volatility (Lower is better)
    )
    
    # Scaling to 0-100 for UI friendliness (Sigmoid-like scaling or MinMax)
    # Using Simple MinMax for readability 0 to 100
    min_scr = final_score.min()
    max_scr = final_score.max()
    scaled_score = ((final_score - min_scr) / (max_scr - min_scr + 1e-9)) * 100
    
    return scaled_score, z

# --- [Execution Engine] ---
def run_quant_analysis(tickers):
    results = {}
    risks = {}
    
    def worker(ticker):
        try:
            # 1. Fetch Data
            df = yf.Ticker(ticker).history(period="6mo") # 6 months needed for ret60
            
            # 2. Risk Check
            is_risky, risk_factors = check_event_risk(ticker)
            if is_risky:
                risks[ticker] = risk_factors
                # We typically exclude risky stocks or penalize them heavily
                return None 
            
            # 3. Compute Factors
            factors = compute_factors(df)
            if factors:
                return {**factors, 'ticker': ticker}
        except:
            return None
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results[res['ticker']] = res

    if not results:
        return pd.DataFrame(), risks

    # Create Factor DataFrame
    factor_df = pd.DataFrame.from_dict(results, orient='index')
    
    # 4. Cross Sectional Scoring
    # Only calculate z-scores on numeric columns
    numeric_cols = ['ret20', 'ret60', 'turnover_z', 'mdd', 'vol']
    scores, z_scores = cross_sectional_score(factor_df[numeric_cols])
    
    factor_df['Quant Score'] = scores
    
    # Add Z-scores for detailed view (optional)
    factor_df['Z-Mom'] = (z_scores['ret20'] + z_scores['ret60']) / 2
    factor_df['Z-Risk'] = (z_scores['mdd'] * -1 + z_scores['vol']) / 2 # High means High Risk
    
    return factor_df.sort_values(by='Quant Score', ascending=False), risks

# --- [Web UI] ---
st.title("🛡️ Quant Nexus Pro")
st.caption("Cross-Sectional Factor Scoring Model (Momentum / Liquidity / Risk)")

with st.sidebar:
    st.header("Strategy Settings")
    view_mode = st.radio("View Mode", ["📱 Card View", "💻 Table View"], horizontal=True)
    
    st.divider()
    selected_sector = st.selectbox("Universe Selection", list(SECTORS.keys()))
    
    st.divider()
    st.markdown("### ⚙️ Algorithm Logic")
    st.info("""
    **1. Scoring (Z-Score)**
    - Momentum (60%): 1M/3M Return
    - Liquidity (20%): Turnover Burst
    - Risk (20%): Volatility & MDD
    
    **2. Risk Filter**
    - Excludes: Embezzlement, Delisting, Suspension events.
    """)
    
    scan_button = st.button("🚀 Run Quant Engine", type="primary", use_container_width=True)

if scan_button:
    target_tickers = SECTORS[selected_sector]
    
    st.subheader(f"📊 Analysis: {selected_sector}")
    
    progress_bar = st.progress(0, text="Fetching Market Data...")
    
    # Run Engine
    df_result, risk_dict = run_quant_analysis(target_tickers)
    
    progress_bar.progress(100, text="Calculation Complete.")
    time.sleep(0.5)
    progress_bar.empty()
    
    if not df_result.empty:
        # Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        top_pick = df_result.iloc[0]
        name = TICKER_MAP.get(top_pick.name, top_pick.name)
        
        c1.metric("Universe Size", f"{len(df_result)} Stocks")
        c2.metric("Top Pick (No.1)", name)
        c3.metric("Avg Score", f"{df_result['Quant Score'].mean():.1f}")
        c4.metric("Risk Filtered", f"{len(risk_dict)}")

        # --- Card View ---
        if "Card" in view_mode:
            st.caption("Top 10 Ranked Stocks based on Factor Z-Scores")
            for idx, row in df_result.head(10).iterrows():
                stock_name = TICKER_MAP.get(idx, idx)
                score = row['Quant Score']
                
                # Dynamic Color based on Score
                if score >= 80: border_color = "🔴 Strong Buy"
                elif score >= 60: border_color = "🟡 Buy"
                else: border_color = "⚪ Hold"
                
                with st.expander(f"{border_color} | {stock_name} ({idx}) | Score: {score:.0f}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Momentum (1M)", f"{row['ret20']*100:.1f}%")
                    c2.metric("Risk (MDD)", f"{row['mdd']*100:.1f}%")
                    c3.metric("Liq Z-Score", f"{row['turnover_z']:.2f}")
                    
                    # Mini Chart
                    fig = go.Figure()
                    color = '#ef4444' if row['ret20'] > 0 else '#3b82f6'
                    fig.add_trace(go.Scatter(y=row['chart_data'], mode='lines', line=dict(color=color, width=2)))
                    fig.update_layout(height=150, margin=dict(t=10,b=10,l=10,r=10), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
                    fig.update_xaxes(visible=False) 
                    st.plotly_chart(fig, use_container_width=True)

        # --- Table View ---
        else:
            # Format DataFrame for Display
            display_df = df_result[['current_price', 'ret20', 'ret60', 'mdd', 'turnover_z', 'Quant Score']].copy()
            display_df.columns = ['Price', 'Mom(1M)', 'Mom(3M)', 'MDD', 'Liq(Z)', 'Score']
            display_df.index = [TICKER_MAP.get(x, x) for x in display_df.index]
            
            st.dataframe(
                display_df.style.background_gradient(subset=['Score'], cmap='RdYlGn')
                .format({
                    'Price': '{:,.0f}', 
                    'Mom(1M)': '{:+.1%}', 
                    'Mom(3M)': '{:+.1%}', 
                    'MDD': '{:.1%}', 
                    'Liq(Z)': '{:.2f}', 
                    'Score': '{:.1f}'
                }),
                use_container_width=True,
                height=600
            )

        # Risk Report
        if risk_dict:
            with st.expander("⚠️ Risk Alert (Excluded Stocks)", expanded=True):
                for t, r in risk_dict.items():
                    name = TICKER_MAP.get(t, t)
                    st.error(f"**{name} ({t})**: Detected keywords {r}")

    else:
        st.error("No valid data found or all stocks filtered by risk.")
