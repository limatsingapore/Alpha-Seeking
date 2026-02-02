import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import logging
import FinanceDataReader as fdr
import yfinance as yf
import requests
import time

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Ultimate)", layout="wide", initial_sidebar_state="expanded")

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
# [설정 & 상수]
# ==============================================================================
CONST = {
    'TRADING_DAYS': 252,       
    'MOM_SHORT': 20,           
    'MOM_MID': 60,             
    'VOL_WINDOW': 252,         
    'COST_RATE': 0.002,        
    'TOP_N': 20,               
    'MIN_AMT': 5_000_000_000,
    'DEFAULT_START_DATE': datetime(2015, 1, 1)
}

# ==============================================================================
# [Helper: 날짜 및 데이터]
# ==============================================================================
def get_last_complete_month_end():
    today = datetime.now()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    return last_month_end

@st.cache_data(ttl=3600*12)
def load_kr_data():
    try:
        df = fdr.StockListing('KRX')
        if df.empty: raise ValueError("KRX Empty")
        if 'Symbol' in df.columns: df.rename(columns={'Symbol':'Code'}, inplace=True)
        df = df[~df['Name'].str.contains('스팩|우B|우|리츠|홀딩스', na=False)]
        if 'Amount' in df.columns:
            df = df.sort_values('Amount', ascending=False).head(600) # 분석 대상 확대
        
        # 섹터 정보 확보
        if 'Sector' not in df.columns: df['Sector'] = '기타'
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist(), df.set_index('Code')['Sector'].to_dict()
    except:
        return {"005930":"삼성전자"}, ["005930"], {}

@st.cache_data(ttl=3600*24)
def load_us_data(index_name='S&P 500'):
    cfg = {
        'S&P 500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'NASDAQ 100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'DOW 30': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    }
    try:
        url = cfg.get(index_name)
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        tables = pd.read_html(r.text)
        
        df = None
        keywords = ['Symbol', 'Ticker', 'Security', 'Company']
        for t in tables:
            if any(k in t.columns for k in keywords):
                df = t; break
        
        if df is None: raise ValueError("Table Not Found")
        
        rename_map = {'Symbol': 'Code', 'Ticker': 'Code', 'Security': 'Name', 'Company': 'Name', 'GICS Sector': 'Sector'}
        df = df.rename(columns=rename_map)
        
        # Sector 컬럼 확인
        if 'Sector' not in df.columns: df['Sector'] = 'Unknown'

        df = df[['Code', 'Name', 'Sector']].dropna()
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist(), df.set_index('Code')['Sector'].to_dict()
    except:
        return {"AAPL":"Apple"}, ["AAPL"], {}

# ==============================================================================
# [Core Logic: 월간 퀀트 & 단타 스캐너]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    """중장기 퀀트 팩터"""
    if len(price) < 60 + 20: return None 
    amt_series = price * volume
    avg_amt = amt_series.iloc[-20:].mean()
    if avg_amt < min_amt: return None

    try:
        mom_short = price.pct_change(20).iloc[-1]
        mom_mid = price.pct_change(60).iloc[-1]
        vol = price.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        liquidity = np.log1p(avg_amt)
        window_price = price.tail(trading_days)
        roll_max = window_price.cummax()
        daily_dd = (window_price / roll_max) - 1.0
        mdd = daily_dd.min()
        
        return {
            'mom_short': mom_short, 'mom_mid': mom_mid,
            'volatility': vol, 'liquidity': liquidity, 'mdd': mdd,
            'price': price.iloc[-1]
        }
    except: return None

def calculate_short_term_factors(price, volume):
    """단타/스윙 전용 팩터"""
    if len(price) < 20: return None
    
    try:
        # 1. 단기 모멘텀
        mom_1d = price.pct_change(1).iloc[-1]
        mom_3d = price.pct_change(3).iloc[-1]
        mom_5d = price.pct_change(5).iloc[-1]
        
        # 2. 거래량 급증 (Volume Spike)
        vol_ma_20 = volume.iloc[-21:-1].mean()
        if vol_ma_20 == 0: return None
        vol_spike = volume.iloc[-1] / vol_ma_20
        
        # 3. 변동성 (Intraday Volatility Proxy)
        # 고가/저가 데이터가 없으므로 종가 기준 표준편차로 대체
        recent_vol = price.tail(5).std() / price.tail(5).mean()
        
        # 4. 이격도 (Disparity)
        ma_20 = price.tail(20).mean()
        disparity = (price.iloc[-1] / ma_20) * 100
        
        return {
            'mom_1d': mom_1d, 'mom_3d': mom_3d, 'mom_5d': mom_5d,
            'vol_spike': vol_spike, 'recent_vol': recent_vol, 
            'disparity': disparity, 'price': price.iloc[-1]
        }
    except: return None

def rank_and_score(factor_df, weights):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    scored['R_Mom_S'] = scored['mom_short'].rank(pct=True)
    scored['R_Mom_M'] = scored['mom_mid'].rank(pct=True)
    scored['R_Vol'] = scored['volatility'].rank(pct=True, ascending=False)
    scored['R_Liq'] = scored['liquidity'].rank(pct=True)
    scored['R_MDD'] = scored['mdd'].rank(pct=True)
    
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
# [백테스트 데이터 페처]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_data_kr(universe, start_date, end_date):
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    try:
        kospi = fdr.DataReader('KS11', start_str, end_str)['Close']
        kospi.index = pd.to_datetime(kospi.index).tz_localize(None)
    except: kospi = pd.Series(dtype=float)

    p, v = {}, {}
    def get(code):
        try:
            d = fdr.DataReader(code, start_str, end_str)
            if d.empty: return None
            d.index = pd.to_datetime(d.index).tz_localize(None)
            return code, d['Close'], d['Volume']
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(get, c) for c in universe]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res: p[res[0]], v[res[0]] = res[1], res[2]
    return pd.DataFrame(p), pd.DataFrame(v), kospi

@st.cache_data(ttl=3600*24)
def fetch_data_us(universe, start_date, end_date):
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    try:
        bm = yf.download("^GSPC", start=start_str, end=end_str, progress=False)['Close']
        bm.index = pd.to_datetime(bm.index).tz_localize(None)
    except: bm = pd.Series(dtype=float)

    try:
        d = yf.download(universe, start=start_str, end=end_str, group_by='ticker', progress=False, threads=True)
        p, v = pd.DataFrame(), pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            tickers = d.columns.levels[0]
            for t in tickers:
                if t in universe:
                    if 'Adj Close' in d[t]: series = d[t]['Adj Close']
                    elif 'Close' in d[t]: series = d[t]['Close']
                    else: continue
                    vol = d[t]['Volume'] if 'Volume' in d[t] else None
                    series.index = pd.to_datetime(series.index).tz_localize(None)
                    if vol is not None: vol.index = pd.to_datetime(vol.index).tz_localize(None)
                    p[t], v[t] = series, vol
        return p, v, bm
    except: return pd.DataFrame(), pd.DataFrame(), bm

# ==============================================================================
# [백테스트 엔진]
# ==============================================================================
def run_backtest(prices, volumes, benchmark, weights, ticker_map):
    if prices.empty: return pd.DataFrame()
    reb_dates = prices.groupby(pd.Grouper(freq='MS')).apply(lambda x: x.index[0])
    logs = []
    start_idx = 12 
    
    for i in range(start_idx, len(reb_dates)-1):
        rebal_date = reb_dates[i]       
        next_rebal = reb_dates[i+1]     
        try:
            loc = prices.index.get_loc(rebal_date)
            if loc == 0: continue
            signal_date = prices.index[loc - 1] 
        except: continue
        
        p_sub = prices.loc[:signal_date].tail(300)
        v_sub = volumes.loc[:signal_date].tail(300)
        active_tickers = p_sub.columns[p_sub.iloc[-1].notna()]
        
        daily_factors = []
        for t in active_tickers:
            f = calculate_factors(p_sub[t], v_sub[t], CONST['MIN_AMT'])
            if f:
                f['code'] = t
                daily_factors.append(f)
        if not daily_factors: continue
        
        factor_df = pd.DataFrame(daily_factors).set_index('code')
        ranked = rank_and_score(factor_df, weights)
        picks = ranked.head(CONST['TOP_N']).index.tolist()
        if not picks: continue

        try:
            period_price = prices.loc[rebal_date:next_rebal, picks]
            if period_price.empty: continue
            period_daily_rets = period_price.pct_change().dropna()
        except KeyError: continue

        if not period_daily_rets.empty:
            port_daily_ret = period_daily_rets.mean(axis=1)
            port_period_ret = (1 + port_daily_ret).prod() - 1
            net_ret = port_period_ret - CONST['COST_RATE']
            try:
                bm_slice = benchmark.loc[rebal_date:next_rebal]
                bm_ret = (bm_slice.iloc[-1] / bm_slice.iloc[0]) - 1
            except: bm_ret = 0.0
            
            logs.append({
                'Date': next_rebal, 'Port_Ret': net_ret, 'BM_Ret': bm_ret,
                'Top1_Holding': ticker_map.get(picks[0], picks[0]) if picks else "",
                'Holdings_Full': ", ".join([ticker_map.get(x,x) for x in picks])
            })
    return pd.DataFrame(logs)

# ==============================================================================
# [UI 및 메인 실행]
# ==============================================================================
st.title("🌍 Alpha Seeking Pro (Ultimate)")

with st.sidebar:
    st.header("🏳️ 시장 선택")
    market = st.radio("국가 선택", ["🇰🇷 한국 (Korea)", "🇺🇸 미국 (USA)"], horizontal=True)
    us_index = "S&P 500" 
    
    if "한국" in market:
        COUNTRY = "KR"
        CURRENCY = "원"
        COST_RATE = 0.002
        MIN_AMT = 5_000_000_000
        VIX_TICKER = "KS200VIX"
        BM_NAME = "KOSPI"
        st.info("대상: 유동성 상위 600개")
        TICKER_INFO, ALL_STOCKS, SECTOR_INFO = load_kr_data()
    else:
        COUNTRY = "US"
        CURRENCY = "$"
        COST_RATE = 0.0005
        MIN_AMT = 5_000_000
        VIX_TICKER = "^VIX"
        BM_NAME = "S&P 500"
        us_index = st.selectbox("지수 선택", ["S&P 500", "NASDAQ 100", "DOW 30"])
        with st.spinner("데이터 수집 중..."):
            TICKER_INFO, ALL_STOCKS, SECTOR_INFO = load_us_data(us_index)
            
    CONST['MIN_AMT'] = MIN_AMT
    CONST['COST_RATE'] = COST_RATE

    if st.button("🧹 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    
    # [NEW] 확장된 모드 선택
    mode = st.radio("모드 선택", [
        "📊 실시간 퀀트 분석", 
        "⚡ 단타/스윙 스캐너", 
        "🔄 섹터 히트맵",
        "🎰 포트폴리오 진단",
        "📉 과거 백테스트"
    ])

    st.divider()
    
    # 전략 설정 (공통 사용)
    st.header("⚙️ 퀀트 전략 설정")
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5),
        "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3),
        "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0),
        "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8),
        "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1),
        "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9),
    }
    sel_preset = st.selectbox("프리셋", list(PRESETS.keys()), index=5)
    dw = PRESETS[sel_preset]
    w_mom = st.slider("📈 추세", 0.0, 1.0, dw[0], 0.1)
    w_liq = st.slider("🌊 수급", 0.0, 1.0, dw[1], 0.1)
    w_vol = st.slider("⚖️ 저변동", 0.0, 1.0, dw[2], 0.1)
    w_risk = st.slider("🛡️ 방어", 0.0, 1.0, dw[3], 0.1)
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}

# --- VIX Helper ---
def get_vix_val(ticker):
    try:
        if ticker == "KS200VIX":
            df = fdr.DataReader(ticker, (datetime.now()-timedelta(days=60)))
        else:
            df = yf.Ticker(ticker).history(period="3mo")
        if df.empty: return None, 0
        return df['Close'].iloc[-1], df['Close'].iloc[-1] - df['Close'].iloc[-2]
    except: return None, 0

# ==============================================================================
# [MODE 1] 실시간 퀀트 분석 (기존)
# ==============================================================================
if mode == "📊 실시간 퀀트 분석":
    st.subheader(f"📊 {market} 월간 퀀트 랭킹")
    if st.button("분석 실행", type="primary"):
        targets = ALL_STOCKS
        results = []
        bar = st.progress(0, f"데이터 분석 중... ({len(targets)}개)")
        
        def worker(t):
            try:
                if COUNTRY == "KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=400)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(t).history(period="2y")
                if len(df) < 200: return None
                f = calculate_factors(df['Close'], df['Volume'], MIN_AMT)
                if f: f['code'] = t
                return f
            except: return None
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(worker, t) for t in targets]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res: results.append(res)
                bar.progress((i+1)/len(targets))
        bar.empty()
        
        if results:
            final = rank_and_score(pd.DataFrame(results).set_index('code'), weights)
            
            # Add Sector Info
            final['Sector'] = [SECTOR_INFO.get(x, 'Unknown') for x in final.index]
            
            # Dashboard
            vix, v_delta = get_vix_val(VIX_TICKER)
            c1, c2, c3 = st.columns([1,1,2])
            top = final.iloc[0]
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            if vix:
                state = "🔴 공포" if vix >= (22 if COUNTRY=="KR" else 30) else "🟢 안정"
                c3.metric(f"VIX ({state})", f"{vix:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
                
            st.dataframe(final[['Total_Score', 'price', 'mom_short', 'mdd', 'Sector']].rename(index=TICKER_INFO), use_container_width=True)
        else: st.warning("결과 없음")

# ==============================================================================
# [MODE 2] 단타/스윙 스캐너 (NEW)
# ==============================================================================
elif mode == "⚡ 단타/스윙 스캐너":
    st.subheader("⚡ 오늘/내일 단기 급등 유망주")
    st.caption("조건: 거래량 급증(Volume Spike) + 단기 추세 살아있음 + 이격도 적정")
    
    col1, col2 = st.columns(2)
    with col1:
        scan_type = st.selectbox("스캔 타입", ["🚀 급등 출발 (거래량 터짐)", "🎣 눌림목 (3일 하락 후 반등)"])
    
    if st.button("스캔 시작", type="primary"):
        targets = ALL_STOCKS
        short_results = []
        bar = st.progress(0, "스캐닝 중...")
        
        def short_worker(t):
            try:
                # 단타는 최근 60일치만 있으면 됨
                if COUNTRY == "KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=100)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(t).history(period="3mo")
                
                if len(df) < 20: return None
                f = calculate_short_term_factors(df['Close'], df['Volume'])
                if f: f['code'] = t
                return f
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(short_worker, t) for t in targets]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res: short_results.append(res)
                bar.progress((i+1)/len(targets))
        bar.empty()
        
        if short_results:
            df_short = pd.DataFrame(short_results).set_index('code')
            df_short['Name'] = [TICKER_INFO.get(x,x) for x in df_short.index]
            
            # 필터링 로직
            if "급등 출발" in scan_type:
                # 거래량 2배 이상 + 1일 수익률 양수 + 5일 수익률 5% 이상
                picks = df_short[
                    (df_short['vol_spike'] >= 2.0) & 
                    (df_short['mom_1d'] > 0) & 
                    (df_short['mom_5d'] > 0.05)
                ].sort_values('vol_spike', ascending=False)
            else: # 눌림목
                # 3일 수익률 음수 + 5일 추세는 상승 + 거래량은 감소 중
                picks = df_short[
                    (df_short['mom_3d'] < -0.02) & 
                    (df_short['mom_5d'] > 0.05) & 
                    (df_short['vol_spike'] < 1.0)
                ].sort_values('mom_5d', ascending=False)
            
            st.success(f"검출된 종목: {len(picks)}개")
            st.dataframe(
                picks[['Name', 'price', 'mom_1d', 'mom_5d', 'vol_spike', 'disparity']],
                use_container_width=True,
                column_config={
                    "mom_1d": st.column_config.NumberColumn("1일 등락", format="%.2%"),
                    "vol_spike": st.column_config.NumberColumn("거래량 급증배수", format="%.1fx"),
                    "disparity": st.column_config.ProgressColumn("이격도(20일)", min_value=90, max_value=110, format="%.1f%%")
                }
            )
        else: st.warning("데이터가 없습니다.")

# ==============================================================================
# [MODE 3] 섹터 히트맵 (NEW)
# ==============================================================================
elif mode == "🔄 섹터 히트맵":
    st.subheader("🔄 실시간 섹터 자금 흐름")
    
    if not SECTOR_INFO:
        st.error("섹터 정보가 없습니다.")
    else:
        # 섹터별 종목 카운트
        sector_counts = pd.Series(SECTOR_INFO.values()).value_counts()
        
        # 트리맵 시각화
        fig = px.treemap(
            names=sector_counts.index,
            parents=["Market"] * len(sector_counts),
            values=sector_counts.values,
            title=f"{market} 섹터 비중 (종목 수 기준)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 팁: '실시간 퀀트 분석'을 먼저 실행한 후 결과를 보면, 어떤 섹터가 상위권에 많은지 알 수 있습니다.")

# ==============================================================================
# [MODE 4] 포트폴리오 진단 (NEW)
# ==============================================================================
elif mode == "🎰 포트폴리오 진단":
    st.subheader("🎰 내 보유 종목 점수 매기기")
    
    user_input = st.text_area("종목코드 입력 (쉼표로 구분)", "005930, 000660" if COUNTRY=="KR" else "AAPL, NVDA, TSLA")
    
    if st.button("진단 실행"):
        codes = [c.strip() for c in user_input.split(',')]
        results = []
        
        for t in codes:
            try:
                if COUNTRY == "KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=300)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(t).history(period="1y")
                
                if not df.empty:
                    f = calculate_factors(df['Close'], df['Volume'], 0) # 내 종목은 거래대금 필터 끔
                    if f: 
                        f['code'] = t
                        results.append(f)
            except: pass
            
        if results:
            df_pf = pd.DataFrame(results).set_index('code')
            scored = rank_and_score(df_pf, weights)
            scored['Name'] = [TICKER_INFO.get(x,x) for x in scored.index]
            
            avg_score = scored['Total_Score'].mean()
            st.metric("내 포트폴리오 평균 점수", f"{avg_score:.1f}점", help="80점 이상이면 매우 우수")
            
            st.dataframe(
                scored[['Name', 'Total_Score', 'mom_short', 'mdd']],
                use_container_width=True,
                column_config={"Total_Score": st.column_config.ProgressColumn("점수", min_value=0, max_value=100)}
            )
        else: st.error("유효한 종목이 없습니다.")

# ==============================================================================
# [MODE 5] 과거 백테스트 (기존)
# ==============================================================================
else:
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("시작일", CONST['DEFAULT_START_DATE'])
    with c2: end_date = st.date_input("종료일", get_last_complete_month_end())

    st.subheader(f"📉 {market} 백테스트")
    if st.button("백테스트 시작", type="primary"):
        with st.spinner("시뮬레이션 중..."):
            u = ALL_STOCKS[:400] if len(ALL_STOCKS) > 400 else ALL_STOCKS
            if COUNTRY == "KR": p, v, bm = fetch_data_kr(u, start_date, end_date)
            else: p, v, bm = fetch_data_us(u, start_date, end_date)
                
            if not p.empty:
                res = run_backtest(p, v, bm, weights, TICKER_INFO)
                if not res.empty:
                    res = res.set_index('Date')
                    res['Cum_Port'] = (1 + res['Port_Ret']).cumprod()
                    if not bm.empty:
                        bm_period = bm.loc[res.index[0]:res.index[-1]]
                        res['Cum_BM'] = (bm_period / bm_period.iloc[0]).reindex(res.index, method='ffill') if not bm_period.empty else 1.0
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], name="My Strategy"))
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_BM'], name=BM_NAME, line=dict(dash='dot')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    tot = res['Cum_Port'].iloc[-1] - 1
                    y = len(res)/12
                    cagr = (tot+1)**(1/y)-1 if y>0 else 0
                    mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("CAGR", f"{cagr:.1%}")
                    c2.metric("Total Return", f"{tot:.1%}")
                    c3.metric("MDD", f"{mdd:.1%}")
                    
                    st.dataframe(res[['Port_Ret', 'Holdings_Full']].tail(), use_container_width=True)
                else: st.error("결과 없음")
            else: st.error("데이터 로딩 실패")
