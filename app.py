import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr
import yfinance as yf
import requests # 크롤링 차단 우회용
import time

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Global)", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] { background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.1rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# [공통 로직: 팩터 계산 & 랭킹]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    # 데이터 길이가 너무 짧으면 계산 불가
    if len(price) < 60 + 20: return None 

    # 유동성 필터 (국가별 기준 적용)
    # 최근 20일 평균 거래대금이 기준 미달이면 탈락
    amt_series = price * volume
    avg_amt = amt_series.iloc[-20:].mean()
    if avg_amt < min_amt: return None

    # Factors Calculation
    try:
        mom_short = price.pct_change(20).iloc[-1] # 1개월 모멘텀
        mom_mid = price.pct_change(60).iloc[-1]   # 3개월 모멘텀
        # 변동성 (연환산)
        vol = price.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        # 유동성 점수 (로그 스케일)
        liquidity = np.log1p(avg_amt)
        
        # True Cumulative MDD (전고점 대비 하락률)
        window_price = price.tail(trading_days)
        roll_max = window_price.cummax()
        daily_dd = (window_price / roll_max) - 1.0
        mdd = daily_dd.min()
        
        return {
            'mom_short': mom_short, 'mom_mid': mom_mid,
            'volatility': vol, 'liquidity': liquidity, 'mdd': mdd,
            'price': price.iloc[-1]
        }
    except:
        return None

def rank_and_score(factor_df, weights):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    
    # Percentile Ranking (0.0 ~ 1.0)
    scored['R_Mom_S'] = scored['mom_short'].rank(pct=True)
    scored['R_Mom_M'] = scored['mom_mid'].rank(pct=True)
    scored['R_Vol'] = scored['volatility'].rank(pct=True, ascending=False) # 낮을수록 좋음
    scored['R_Liq'] = scored['liquidity'].rank(pct=True)
    scored['R_MDD'] = scored['mdd'].rank(pct=True) # 0(최고점)에 가까울수록 좋음
    
    # Weighted Sum
    total_score = (
        (scored['R_Mom_S'] * 0.5 + scored['R_Mom_M'] * 0.5) * weights['mom'] +
        scored['R_Vol'] * weights['vol'] +
        scored['R_Liq'] * weights['liq'] +
        scored['R_MDD'] * weights['risk']
    )
    
    weight_sum = sum(weights.values())
    if weight_sum == 0: weight_sum = 1
    
    scored['Total_Score'] = (total_score / weight_sum) * 100
    return scored.sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [데이터 로더: 차단 방지 적용 (Robust Loader)]
# ==============================================================================
@st.cache_data(ttl=3600*12)
def load_kr_data():
    """한국 시장 데이터 로딩 (FinanceDataReader)"""
    try:
        df = fdr.StockListing('KRX')
        if df.empty: raise ValueError("KRX 데이터 수신 실패")
        
        if 'Symbol' in df.columns: df.rename(columns={'Symbol':'Code'}, inplace=True)
        # 스팩, 우선주, 리츠 등 제외
        df = df[~df['Name'].str.contains('스팩|우B|우|리츠|홀딩스', na=False)]
        
        # 유동성 상위 500개만 필터링 (속도 최적화)
        if 'Amount' in df.columns:
            df = df.sort_values('Amount', ascending=False).head(500)
            
        ticker_info = df.set_index('Code')['Name'].to_dict()
        all_tickers = df['Code'].tolist()
        return ticker_info, all_tickers
    except Exception as e:
        st.error(f"KRX 로딩 실패: {e}")
        return {}, []

@st.cache_data(ttl=3600*24)
def load_us_data(index_name='S&P 500'):
    """미국 시장 데이터 로딩 (Wikipedia Crawling with Headers)"""
    cfg = {
        'S&P 500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'NASDAQ 100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'DOW 30': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    }
    
    try:
        url = cfg.get(index_name)
        # [중요] 봇 차단 방지용 헤더
        headers = {"User-Agent": "Mozilla/5.0"} 
        r = requests.get(url, headers=headers)
        
        tables = pd.read_html(r.text)
        df = None
        
        # 적절한 테이블 찾기
        keywords = ['Symbol', 'Ticker', 'Security', 'Company']
        for t in tables:
            if any(k in t.columns for k in keywords):
                df = t; break
        
        if df is None: raise ValueError("위키피디아 테이블 파싱 실패")
        
        # 컬럼 표준화
        rename_map = {'Symbol': 'Code', 'Ticker': 'Code', 'Security': 'Name', 'Company': 'Name'}
        df = df.rename(columns=rename_map)
        
        df = df[['Code', 'Name']].dropna()
        # yfinance 호환용 티커 변환 (BRK.B -> BRK-B)
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()

    except Exception as e:
        st.error(f"미국 데이터 로딩 실패 ({index_name}): {e}")
        # 비상용 폴백 데이터
        return {"AAPL":"Apple", "MSFT":"Microsoft", "TSLA":"Tesla"}, ["AAPL", "MSFT", "TSLA"]

# ==============================================================================
# [백테스트 데이터 페처]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_data_kr(universe, days=365*10):
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    kospi = fdr.DataReader('KS11', start)['Close']
    p, v = {}, {}
    
    def get(code):
        try:
            d = fdr.DataReader(code, start)
            if d.empty: return None
            return code, d['Close'], d['Volume']
        except: return None

    # 과도한 요청 방지를 위해 worker 수 조절 (30 -> 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(get, c) for c in universe]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res: p[res[0]], v[res[0]] = res[1], res[2]
            
    return pd.DataFrame(p), pd.DataFrame(v), kospi

@st.cache_data(ttl=3600*24)
def fetch_data_us(universe, days=365*10):
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    # Benchmark
    bm = yf.download("^GSPC", start=start, progress=False)['Close']
    
    # Bulk Download (yfinance는 이게 훨씬 빠르고 안전함)
    try:
        # threads=True로 내부 병렬 처리
        d = yf.download(universe, start=start, group_by='ticker', progress=False, threads=True)
        p, v = pd.DataFrame(), pd.DataFrame()
        
        for t in universe:
            if t in d.columns.levels[0]:
                # Adj Close 우선 사용
                col = 'Adj Close' if 'Adj Close' in d[t] else 'Close'
                p[t] = d[t][col]
                v[t] = d[t]['Volume']
        return p, v, bm
    except: return pd.DataFrame(), pd.DataFrame(), bm

# ==============================================================================
# [메인 컨트롤러]
# ==============================================================================
st.title("🌍 Alpha Seeking Pro (Global Ultimate)")

# --- 1. 사이드바: 설정 ---
with st.sidebar:
    st.header("🏳️ 시장 선택")
    market = st.radio("국가 선택", ["🇰🇷 한국 (Korea)", "🇺🇸 미국 (USA)"], horizontal=True)
    
    if "한국" in market:
        COUNTRY = "KR"
        CURRENCY = "원"
        COST_RATE = 0.002 # 0.2%
        MIN_AMT = 5_000_000_000 # 50억
        VIX_TICKER = "KS200VIX"
        BM_NAME = "KOSPI"
        st.info("대상: 유동성 상위 500개 종목")
        TICKER_INFO, ALL_STOCKS = load_kr_data()
        
    else:
        COUNTRY = "US"
        CURRENCY = "$"
        COST_RATE = 0.0005 # 0.05%
        MIN_AMT = 5_000_000 # $5M
        VIX_TICKER = "^VIX"
        BM_NAME = "S&P 500"
        us_index = st.selectbox("지수 선택", ["S&P 500", "NASDAQ 100", "DOW 30"])
        with st.spinner("데이터 수집 중..."):
            TICKER_INFO, ALL_STOCKS = load_us_data(us_index)

    st.divider()
    st.header("⚙️ 전략 설정")
    
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5),
        "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3),
        "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0),
        "🧘 마음의 평화": (0.3, 0.2, 1.0, 0.5),
        "🚑 좀비 헌터": (0.4, 0.3, 0.3, 1.0),
        "⚖️ 황금 밸런스": (0.5, 0.5, 0.5, 0.5),
        "💎 우상향 정석": (0.7, 0.3, 0.7, 0.4),
        "🐆 안전한 사냥": (0.8, 0.7, 0.1, 0.8), # 추천
        "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8)
    }
    
    sel_preset = st.selectbox("프리셋", list(PRESETS.keys()), index=9)
    dw = PRESETS[sel_preset]
    
    w_mom = st.slider("📈 추세", 0.0, 1.0, dw[0], 0.1)
    w_liq = st.slider("🌊 수급", 0.0, 1.0, dw[1], 0.1)
    w_vol = st.slider("⚖️ 저변동", 0.0, 1.0, dw[2], 0.1)
    w_risk = st.slider("🛡️ 방어", 0.0, 1.0, dw[3], 0.1)
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}
    
    st.divider()
    mode = st.radio("모드", ["📊 실시간 분석", "📉 백테스트"])

# --- VIX Helper ---
def get_vix_val(ticker):
    try:
        if ticker == "KS200VIX":
            # 한국 VIX는 FDR
            df = fdr.DataReader(ticker, (datetime.now()-timedelta(days=60)))
        else:
            # 미국 VIX는 YFinance
            df = yf.Ticker(ticker).history(period="3mo")
            
        if df.empty: return None, 0
        return df['Close'].iloc[-1], df['Close'].iloc[-1] - df['Close'].iloc[-2]
    except: return None, 0

# ==============================================================================
# [APP 실행]
# ==============================================================================

# -------------------- TAB 1: 실시간 --------------------
if mode == "📊 실시간 분석":
    st.subheader(f"📊 {market} 실시간 팩터 랭킹")
    
    if st.button("분석 실행", type="primary"):
        targets = ALL_STOCKS
        results = []
        bar = st.progress(0, f"데이터 분석 중... ({len(targets)}개)")
        
        # 실시간 스캐닝 (Rate Limit 방지를 위해 worker 수 조절: 5)
        def worker(t):
            try:
                if COUNTRY == "KR":
                    start = (datetime.now()-timedelta(days=400)).strftime('%Y-%m-%d')
                    df = fdr.DataReader(t, start)
                else:
                    df = yf.Ticker(t).history(period="2y")
                    
                if len(df) < 200: return None
                f = calculate_factors(df['Close'], df['Volume'], MIN_AMT)
                if f: f['code'] = t
                return f
            except: return None
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(worker, t) for t in targets]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res: results.append(res)
                bar.progress((i+1)/len(targets))
        bar.empty()
        
        if results:
            final = rank_and_score(pd.DataFrame(results).set_index('code'), weights)
            
            # Dashboard
            vix, v_delta = get_vix_val(VIX_TICKER)
            c1, c2, c3 = st.columns([1,1,2])
            top = final.iloc[0]
            
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            if vix:
                # 미국 VIX는 30 넘으면 공포, 한국은 22
                thr = 22 if COUNTRY=="KR" else 30
                state = "🔴 공포" if vix >= thr else ("🟠 주의" if vix >= (thr-5) else "🟢 안정")
                c3.metric(f"{market[:2]} VIX", f"{vix:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
                c3.caption(f"시장 상태: {state}")
                
            # Table
            cols = ['price', 'Total_Score', 'mom_short', 'mdd']
            disp = final[cols].copy()
            disp.columns = [f'Price({CURRENCY})', 'Score', 'Mom(1M)', 'MDD']
            disp.index = [TICKER_INFO.get(x,x) for x in disp.index]
            
            st.dataframe(disp, use_container_width=True, 
                         column_config={
                             f"Price({CURRENCY})": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
                             "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                             "Mom(1M)": st.column_config.NumberColumn(format="%.2%"),
                             "MDD": st.column_config.NumberColumn(format="%.2%")
                         })
        else:
            st.warning("분석 결과가 없습니다. (장시간 데이터 수집 실패 등)")

# -------------------- TAB 2: 백테스트 --------------------
else:
    st.subheader(f"📉 {market} 10년 백테스트")
    st.caption(f"⚡ 조건: 수수료 {COST_RATE*100}% | 벤치마크 {BM_NAME} | 최소거래 {MIN_AMT:,.0f}{CURRENCY}")
    
    if st.button("백테스트 시작", type="primary"):
        with st.spinner("10년치 데이터 시뮬레이션 중... (1~2분 소요)"):
            # 유니버스 샘플링 (속도: 400개)
            u = ALL_STOCKS[:400] if len(ALL_STOCKS) > 400 else ALL_STOCKS
            
            if COUNTRY == "KR":
                p, v, bm = fetch_data_kr(u)
            else:
                p, v, bm = fetch_data_us(u)
                
            if not p.empty:
                reb_dates = p.resample('M').last().index
                logs = []
                
                # Walk-Forward Logic
                for i in range(12, len(reb_dates)-1):
                    curr, next_d = reb_dates[i], reb_dates[i+1]
                    p_sub = p.loc[:curr].tail(300)
                    v_sub = v.loc[:curr].tail(300)
                    
                    # 팩터 계산
                    daily = []
                    active = p_sub.columns[p_sub.iloc[-1].notna()]
                    
                    for t in active:
                        f = calculate_factors(p_sub[t], v_sub[t], MIN_AMT)
                        if f: 
                            f['code'] = t; daily.append(f)
                    
                    if not daily: continue
                    
                    ranked = rank_and_score(pd.DataFrame(daily).set_index('code'), weights)
                    picks = ranked.head(20).index.tolist()
                    
                    # 수익률
                    ret_wd = p.loc[curr:next_d, picks].pct_change().dropna()
                    if ret_wd.empty: continue
                    
                    # 동일비중 포트폴리오
                    port_ret = (1+ret_wd).prod().mean() - 1 - COST_RATE
                    bm_ret = (1+bm.loc[curr:next_d].pct_change().dropna()).prod() - 1
                    
                    logs.append({
                        'Date': next_d, 'Port_Ret': port_ret, 'BM_Ret': bm_ret,
                        'Holdings': ", ".join([TICKER_INFO.get(x,x) for x in picks])
                    })
                
                if logs:
                    res = pd.DataFrame(logs).set_index('Date')
                    res['Cum_Port'] = (1+res['Port_Ret']).cumprod()
                    res['Cum_BM'] = (1+res['BM_Ret']).cumprod()
                    
                    # Metrics Calculation
                    tot = res['Cum_Port'].iloc[-1] - 1
                    y = len(res)/12
                    cagr = (tot+1)**(1/y)-1 if y>0 else 0
                    mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                    # Sharpe (Ann. Volatility)
                    ann_vol = res['Port_Ret'].std() * np.sqrt(12)
                    sharpe = cagr / ann_vol if ann_vol > 0 else 0
                    win = (res['Port_Ret']>0).sum()/len(res)
                    
                    # 1. Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], name="My Strategy", line=dict(color='#3b82f6', width=2)))
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_BM'], name=BM_NAME, line=dict(color='#94a3b8', dash='dot')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Metrics Board
                    st.divider()
                    st.subheader("📊 전략 성과 정밀 분석")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("CAGR", f"{cagr:.1%}", help="연평균 복리 수익률")
                    k2.metric("Total Return", f"{tot:.1%}")
                    k3.metric("MDD", f"{mdd:.1%}", help="최대 낙폭")
                    k4.metric("Sharpe", f"{sharpe:.2f}", help="위험 대비 수익 (1.0 이상 양호)")
                    k5.metric("Win Rate", f"{win:.1%}", help="월간 승률")
                    
                    # 3. Logs
                    st.divider()
                    st.dataframe(res[['Port_Ret', 'Holdings']].tail(5), use_container_width=True)
                else: st.error("백테스트 결과가 없습니다.")
            else: st.error("데이터 로딩 실패")
