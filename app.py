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
# [Helper: 날짜 계산]
# ==============================================================================
def get_last_complete_month_end():
    today = datetime.now()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    return last_month_end

# ==============================================================================
# [데이터 로더]
# ==============================================================================
@st.cache_data(ttl=3600*12)
def load_kr_data():
    try:
        df = fdr.StockListing('KRX')
        if df.empty: raise ValueError("KRX Empty")
        if 'Symbol' in df.columns: df.rename(columns={'Symbol':'Code'}, inplace=True)
        df = df[~df['Name'].str.contains('스팩|우B|우|리츠|홀딩스', na=False)]
        if 'Amount' in df.columns:
            df = df.sort_values('Amount', ascending=False).head(600) # 유니버스 확대
        
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
        
        if 'Sector' not in df.columns: df['Sector'] = 'Unknown'

        df = df[['Code', 'Name', 'Sector']].dropna()
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist(), df.set_index('Code')['Sector'].to_dict()
    except:
        return {"AAPL":"Apple"}, ["AAPL"], {}

# ==============================================================================
# [Core Logic: 팩터 계산]
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
        mom_1d = price.pct_change(1).iloc[-1]
        mom_3d = price.pct_change(3).iloc[-1]
        mom_5d = price.pct_change(5).iloc[-1]
        
        vol_ma_20 = volume.iloc[-21:-1].mean()
        if vol_ma_20 == 0: return None
        vol_spike = volume.iloc[-1] / vol_ma_20
        
        recent_vol = price.tail(5).std() / price.tail(5).mean()
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
# [전략 최적화 엔진]
# ==============================================================================
def optimize_strategy(prices, volumes, benchmark, ticker_map, presets={}):
    results = []
    if prices.empty or volumes.empty: return pd.DataFrame()

    progress_bar = st.progress(0, text="전략 시뮬레이션 시작...")
    total_presets = len(presets)
    
    for i, (name, (mom, liq, vol, risk)) in enumerate(presets.items()):
        weights = {'mom': mom, 'liq': liq, 'vol': vol, 'risk': risk}
        try:
            res = run_backtest(prices, volumes, benchmark, weights, ticker_map)
            if not res.empty:
                res = res.set_index('Date')
                res['Cum_Port'] = (1 + res['Port_Ret']).cumprod()
                
                tot = res['Cum_Port'].iloc[-1] - 1
                y = len(res)/12
                if y <= 0: y = 1
                cagr = (tot+1)**(1/y)-1
                mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                ann_vol = res['Port_Ret'].std() * np.sqrt(12)
                sharpe = cagr / ann_vol if ann_vol != 0 else 0
                win_rate = (res['Port_Ret']>0).sum()/len(res)
                
                results.append({
                    '전략명': name, '승률': win_rate, '연수익률(CAGR)': cagr,
                    '누적수익률': tot, 'MDD': mdd, '샤프비율': sharpe, '변동성': ann_vol,
                    '가중치': f"추세{mom}|수급{liq}|저변동{vol}|방어{risk}"
                })
        except: pass
        progress_bar.progress((i + 1) / total_presets, text=f"분석 중: {name}")
    
    progress_bar.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values(by='연수익률(CAGR)', ascending=False)

def highlight_top3(s):
    is_volatility = s.name == '변동성'
    if is_volatility: sorted_vals = s.sort_values(ascending=True).unique()
    else: sorted_vals = s.sort_values(ascending=False).unique()
    
    top1 = sorted_vals[0] if len(sorted_vals) > 0 else None
    top2 = sorted_vals[1] if len(sorted_vals) > 1 else None
    top3 = sorted_vals[2] if len(sorted_vals) > 2 else None
    
    styles = []
    for v in s:
        if v == top1: styles.append('background-color: #FFD700; color: black; font-weight: bold')
        elif v == top2: styles.append('color: #FF4B4B; font-weight: bold')
        elif v == top3: styles.append('font-weight: bold')
        else: styles.append('')
    return styles

# ==============================================================================
# [메인 컨트롤러]
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
    mode = st.radio("모드 선택", [
        "📊 실시간 퀀트 분석", 
        "⚡ 단타/스윙 스캐너", 
        "🔄 섹터 히트맵",
        "🎰 포트폴리오 진단",
        "📉 과거 백테스트",
        "🔍 전략 전체 비교 (최적화)"
    ])

    st.divider()
    st.header("⚙️ 전략 설정")
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5),
        "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3),
        "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0),
        "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8),
        "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1),
        "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9),
        "🎯 퀄리티 그로스": (0.6, 0.6, 0.6, 0.6),
        "🌪️ 변동성 사냥꾼": (0.7, 0.5, 0.0, 0.2),
        "🦅 매파의 눈": (0.3, 0.9, 0.4, 0.7)
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
        if ticker == "KS200VIX": df = fdr.DataReader(ticker, (datetime.now()-timedelta(days=60)))
        else: df = yf.Ticker(ticker).history(period="3mo")
        if df.empty: return None, 0
        return df['Close'].iloc[-1], df['Close'].iloc[-1] - df['Close'].iloc[-2]
    except: return None, 0

# ==============================================================================
# [APP 실행 로직]
# ==============================================================================

# 1. 실시간 분석
if mode == "📊 실시간 퀀트 분석":
    st.subheader(f"📊 {market} 실시간 랭킹")
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
            final['Sector'] = [SECTOR_INFO.get(x, 'Unknown') for x in final.index]
            
            vix, v_delta = get_vix_val(VIX_TICKER)
            c1, c2, c3 = st.columns([1,1,2])
            top = final.iloc[0]
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            if vix:
                state = "🔴 공포" if vix >= (22 if COUNTRY=="KR" else 30) else "🟢 안정"
                c3.metric(f"VIX ({state})", f"{vix:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
                
            st.dataframe(
                final[['Total_Score', 'price', 'mom_short', 'mdd', 'Sector']].rename(index=TICKER_INFO),
                use_container_width=True,
                column_config={
                    "Total_Score": st.column_config.ProgressColumn("점수", min_value=0, max_value=100),
                    "price": st.column_config.NumberColumn("현재가", format=f"{CURRENCY}%.2f"),
                    "mom_short": st.column_config.NumberColumn("단기추세", format="%.2%"),
                    "mdd": st.column_config.NumberColumn("MDD", format="%.2%")
                }
            )
        else: st.warning("결과 없음")

# 2. 단타/스윙 스캐너
elif mode == "⚡ 단타/스윙 스캐너":
    st.subheader("⚡ 오늘/내일 단기 급등 유망주")
    col1, col2 = st.columns(2)
    with col1: scan_type = st.selectbox("스캔 타입", ["🚀 급등 출발 (거래량 터짐)", "🎣 눌림목 (3일 하락 후 반등)"])
    
    if st.button("스캔 시작", type="primary"):
        targets = ALL_STOCKS
        short_results = []
        bar = st.progress(0, "스캐닝 중...")
        
        def short_worker(t):
            try:
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
            
            if "급등 출발" in scan_type:
                picks = df_short[(df_short['vol_spike'] >= 2.0) & (df_short['mom_1d'] > 0) & (df_short['mom_5d'] > 0.05)].sort_values('vol_spike', ascending=False)
            else: 
                picks = df_short[(df_short['mom_3d'] < -0.02) & (df_short['mom_5d'] > 0.05) & (df_short['vol_spike'] < 1.0)].sort_values('mom_5d', ascending=False)
            
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
# [Helper] 한국 종목 섹터 추론기 (강화판)
# ==============================================================================
def infer_sector_kr(name):
    """종목명 키워드로 섹터를 역추적"""
    name = str(name)
    if any(x in name for x in ['스팩', '제호', '기업인수']): return '스팩/금융'
    if any(x in name for x in ['우', '우B']): return '우선주'
    
    # 주요 섹터 키워드 매핑
    keywords = {
        '반도체/IT': ['삼성전자', 'SK하이닉스', '반도체', '테크', '칩', '시스템', '전자', '디스플레이', '이노텍', 'DB하이텍', '주성', '한미반도체'],
        '2차전지': ['에코프로', '엘앤에프', 'LG에너지', '삼성SDI', 'SK이노베이션', '포스코퓨처', '천보', '엔켐', '금양'],
        '바이오/제약': ['바이오', '제약', '약품', '생명', '헬스', '셀트리온', '유한양행', '한미약품', 'HLB', '알테오젠', '케어'],
        '자동차/부품': ['현대차', '기아', '모비스', '타이어', '만도', '오토', '화신', '성우하이텍'],
        '인터넷/게임': ['NAVER', '카카오', '게임', '소프트', '엔씨', '펄어비스', '크래프톤', '위메이드', '넷마블'],
        '엔터/미디어': ['엔터', '스튜디오', '미디어', '에스엠', 'JYP', 'YG', '하이브', 'CJ ENM', '아프리카'],
        '금융/지주': ['금융', '지주', '은행', '증권', '보험', '카드', '투자', '홀딩스', '메리츠', 'KB', '신한'],
        '조선/중공업': ['중공업', '조선', '기계', '엔진', '현대미포', '한국조선', '삼성중공업', '두산', '한화오션'],
        '화학/정유': ['화학', '케미칼', '정유', 'S-Oil', '롯데정밀', '효성', '금호'],
        '건설/건자재': ['건설', '개발', '엔지니어링', '시멘트', '페인트', '현대건설', 'GS건설'],
        '소비재/유통': ['푸드', '식품', '제과', '쇼핑', '백화점', '이마트', '호텔', '투어', '항공', '화장품', '아모레']
    }
    
    for sector, keys in keywords.items():
        if any(k in name for k in keys):
            return sector
            
    return '기타/소형주'

# ==============================================================================
# [MODE 3] 섹터 히트맵 (탭 구성 적용)
# ==============================================================================
elif mode == "🔄 섹터 히트맵":
    st.subheader("🔄 시장 자금 흐름 분석 (Sector Flow)")
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📊 실시간 강도", "🔥 급등 테마 탐지", "📈 섹터 로테이션"])
    
    # --------------------------------------------------------------------------
    # TAB 1: 실시간 섹터 강도 (전체 종목 스캔)
    # --------------------------------------------------------------------------
    with tab1:
        st.caption("오늘 시장에서 가장 강한 업종을 찾습니다.")
        period_select = st.selectbox("기간 선택", ["오늘(1D)", "1주일(1W)", "1개월(1M)", "3개월(3Q)"])
        period_map = {"오늘(1D)": 2, "1주일(1W)": 5, "1개월(1M)": 20, "3개월(3Q)": 60} 
        # 오늘 수익률 계산을 위해 최소 2일치 데이터 필요
        
        if st.button("섹터 강도 분석 실행", type="primary"):
            lookback = period_map[period_select]
            targets = ALL_STOCKS[:400] # 상위 400개 샘플링
            sector_returns = {} # {섹터명: [수익률 리스트]}
            
            bar = st.progress(0, "시장 데이터 스캔 중...")
            
            # 병렬 처리로 데이터 수집
            def get_ret(t):
                try:
                    if COUNTRY == "KR":
                        # 넉넉하게 가져옴
                        df = fdr.DataReader(t, (datetime.now()-timedelta(days=lookback*2+10)).strftime('%Y-%m-%d'))
                    else:
                        df = yf.Ticker(t).history(period="6mo")
                        
                    if len(df) < lookback: return None
                    
                    # 수익률 계산
                    ret = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback]) - 1
                    
                    # 섹터 분류
                    name = TICKER_INFO.get(t, t)
                    if COUNTRY == "KR":
                        sec = infer_sector_kr(name)
                    else:
                        sec = SECTOR_INFO.get(t, 'Unknown')
                        
                    return sec, ret
                except: return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
                futures = [ex.submit(get_ret, t) for t in targets]
                for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                    res = fut.result()
                    if res:
                        s, r = res
                        if s not in sector_returns: sector_returns[s] = []
                        sector_returns[s].append(r)
                    bar.progress((i+1)/len(targets))
            bar.empty()
            
            # 결과 집계
            if sector_returns:
                # 평균 수익률 및 종목 수
                stats = []
                for s, rets in sector_returns.items():
                    if len(rets) >= 3: # 최소 3종목 이상인 섹터만
                        stats.append({
                            '섹터': s,
                            '수익률': np.mean(rets),
                            '종목수': len(rets)
                        })
                
                df_sec = pd.DataFrame(stats).sort_values('수익률', ascending=False)
                
                # 1. 바 차트
                fig = px.bar(
                    df_sec, x='섹터', y='수익률', color='수익률',
                    color_continuous_scale='RdYlGn',
                    title=f"📊 {period_select} 섹터별 평균 수익률"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. 강세/약세 리스트
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🔥 강세 섹터 Top 5")
                    st.dataframe(
                        df_sec.head(5).style.format({'수익률': '{:.2%}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                with c2:
                    st.subheader("❄️ 약세 섹터 Top 5")
                    st.dataframe(
                        df_sec.tail(5).sort_values('수익률').style.format({'수익률': '{:.2%}'}),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.warning("데이터가 부족합니다.")

    # --------------------------------------------------------------------------
    # TAB 2: 급등 테마 탐지 (거래량 급증 기반)
    # --------------------------------------------------------------------------
    with tab2:
        st.caption("거래량이 폭발하며 급등하는 '주도 테마'를 찾습니다.")
        
        if st.button("🚀 급등 테마 스캔", type="primary"):
            targets = ALL_STOCKS[:500]
            hot_stocks = []
            bar = st.progress(0, "거래량 분석 중...")
            
            def scan_hot(t):
                try:
                    if COUNTRY == "KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=20)).strftime('%Y-%m-%d'))
                    else: df = yf.Ticker(t).history(period="1mo")
                    
                    if len(df) < 10: return None
                    
                    # 조건: 1일 등락률 > 3% AND 거래량 > 5일 평균의 2배
                    ret = df['Close'].pct_change().iloc[-1]
                    vol_ratio = df['Volume'].iloc[-1] / df['Volume'].iloc[-6:-1].mean()
                    
                    if ret > 0.03 and vol_ratio >= 2.0:
                        name = TICKER_INFO.get(t, t)
                        sec = infer_sector_kr(name) if COUNTRY=="KR" else SECTOR_INFO.get(t, 'Unknown')
                        return {'종목명': name, '섹터': sec, '등락률': ret, '거래량급증': vol_ratio}
                except: return None
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
                futures = [ex.submit(scan_hot, t) for t in targets]
                for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                    res = fut.result()
                    if res: hot_stocks.append(res)
                    bar.progress((i+1)/len(targets))
            bar.empty()
            
            if hot_stocks:
                df_hot = pd.DataFrame(hot_stocks)
                
                # 섹터별로 묶어서 카운트
                hot_sectors = df_hot['섹터'].value_counts()
                
                # 1. 핫 섹터 랭킹
                st.subheader(f"🔥 오늘의 주도 테마: **{hot_sectors.index[0]}**")
                
                # 트리맵 시각화
                fig = px.treemap(
                    df_hot, path=['섹터', '종목명'], values='거래량급증',
                    color='등락률', color_continuous_scale='OrRd',
                    title="급등주 섹터 분포 (크기=거래강도)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. 상세 리스트
                st.dataframe(
                    df_hot.sort_values('거래량급증', ascending=False),
                    use_container_width=True,
                    column_config={
                        "등락률": st.column_config.NumberColumn(format="%.2%"),
                        "거래량급증": st.column_config.NumberColumn(format="%.1fx")
                    }
                )
            else:
                st.info("오늘 급등 조건(3%↑ + 거래량2배)을 만족하는 종목이 없습니다.")

    # --------------------------------------------------------------------------
    # TAB 3: 섹터 로테이션 (과거 1년 흐름)
    # --------------------------------------------------------------------------
    with tab3:
        st.caption("지난 12개월간 돈이 어떻게 돌았는지(Rotation) 보여줍니다.")
        
        if st.button("🔄 로테이션 분석 실행"):
            with st.spinner("과거 데이터 로딩 중..."):
                # 대표 섹터 ETF 또는 대표주 사용
                # KR: 주요 업종 대표주 2개씩
                # US: Sector ETF (XLK, XLV 등)
                
                if COUNTRY == "KR":
                    sectors = {
                        '반도체': ['000660', '005930'], # 하이닉스, 삼전
                        '2차전지': ['373220', '006400'], # LG엔솔, SDI
                        '바이오': ['207940', '068270'], # 삼바, 셀트
                        '자동차': ['005380', '000270'], # 현대, 기아
                        '금융': ['105560', '055550'],   # KB, 신한
                        '인터넷': ['035420', '035720']  # 네이버, 카카오
                    }
                else:
                    sectors = {
                        '기술(Tech)': ['XLK'], '헬스케어': ['XLV'], '금융': ['XLF'],
                        '에너지': ['XLE'], '소비재': ['XLY'], '산업재': ['XLI']
                    }
                
                # 월별 수익률 데이터 생성
                monthly_data = {}
                end_d = datetime.now()
                dates = [end_d - timedelta(days=30*i) for i in range(12)][::-1] # 최근 12개월
                date_labels = [d.strftime('%y-%m') for d in dates]
                
                for sec_name, codes in sectors.items():
                    sec_monthly_rets = []
                    
                    # 각 월별 수익률 계산 (약식)
                    # 실제로는 전체 데이터를 한번에 받아서 resample하는게 빠름
                    # 여기서는 로직 단순화를 위해 대표 종목의 최근 1년치 데이터를 받음
                    try:
                        code = codes[0]
                        if COUNTRY == "KR": 
                            df = fdr.DataReader(code, (end_d - timedelta(days=380)).strftime('%Y-%m-%d'))
                        else: 
                            df = yf.Ticker(code).history(period="1y")
                        
                        # 월간 리샘플링
                        df_m = df['Close'].resample('M').last().pct_change()
                        # 최근 12개만
                        sec_monthly_rets = df_m.tail(12).values.tolist()
                    except:
                        sec_monthly_rets = [0]*12
                        
                    # 길이 맞추기
                    if len(sec_monthly_rets) < 12:
                        sec_monthly_rets = [0]*(12-len(sec_monthly_rets)) + sec_monthly_rets
                        
                    monthly_data[sec_name] = sec_monthly_rets
                
                # 히트맵 그리기
                df_rot = pd.DataFrame(monthly_data, index=date_labels).T
                
                fig = px.imshow(
                    df_rot,
                    labels=dict(x="월", y="섹터", color="수익률"),
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="📅 월간 섹터 수익률 히트맵"
                )
                fig.update_traces(text=df_rot.applymap(lambda x: f"{x:.1%}").values, texttemplate="%{text}")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 붉은색(하락)에서 초록색(상승)으로 변하는 섹터가 다음 주도주일 가능성이 높습니다.")

# 4. 포트폴리오 진단
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
                    f = calculate_factors(df['Close'], df['Volume'], 0)
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
            st.dataframe(scored[['Name', 'Total_Score', 'mom_short', 'mdd']], use_container_width=True, column_config={"Total_Score": st.column_config.ProgressColumn("점수", min_value=0, max_value=100)})
        else: st.error("유효한 종목이 없습니다.")

# 5. 백테스트
elif mode == "📉 과거 백테스트":
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
                    
                    tot = res['Cum_Port'].iloc[-1] - 1
                    y = len(res)/12
                    if y<=0: y=1
                    cagr = (tot+1)**(1/y)-1
                    mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("CAGR", f"{cagr:.1%}")
                    c2.metric("Total Return", f"{tot:.1%}")
                    c3.metric("MDD", f"{mdd:.1%}")
                    st.dataframe(res[['Port_Ret', 'Holdings_Full']].tail(), use_container_width=True)
                else: st.error("결과 없음")
            else: st.error("데이터 로딩 실패")

# 6. 전략 최적화 (NEW)
else:
    st.subheader("🔍 전체 전략 성과 비교")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("시작일", CONST['DEFAULT_START_DATE'], key='opt_start')
    with c2: end_date = st.date_input("종료일", get_last_complete_month_end(), key='opt_end')
    
    if st.button("🚀 전체 전략 분석 실행", type="primary"):
        if start_date >= end_date: st.error("기간 설정 오류")
        else:
            with st.spinner("모든 전략 시뮬레이션 중..."):
                u = ALL_STOCKS[:400] if len(ALL_STOCKS) > 400 else ALL_STOCKS
                if COUNTRY == "KR": p, v, bm = fetch_data_kr(u, start_date, end_date)
                else: p, v, bm = fetch_data_us(u, start_date, end_date)
                
                if not p.empty:
                    test_presets = {k:v for k,v in PRESETS.items() if k != "사용자 정의"}
                    df_result = optimize_strategy(p, v, bm, TICKER_INFO, test_presets)
                    
                    if not df_result.empty:
                        st.success("분석 완료!")
                        st.dataframe(df_result.style.apply(highlight_top3, subset=['승률', '연수익률(CAGR)', '누적수익률', 'MDD', '샤프비율', '변동성'])
                            .format({'승률':'{:.1%}','연수익률(CAGR)':'{:.1%}','누적수익률':'{:.1%}','MDD':'{:.1%}','샤프비율':'{:.2f}','변동성':'{:.1%}'}), 
                            use_container_width=True, height=500)
                        
                        best_st = df_result.iloc[0]
                        st.divider()
                        st.subheader(f"👑 1위 전략: {best_st['전략명']}")
                        
                        ws_str = best_st['가중치']
                        parts = ws_str.split('|')
                        w_dict = {}
                        for part in parts:
                            if '추세' in part: w_dict['mom'] = float(part.replace('추세', ''))
                            elif '수급' in part: w_dict['liq'] = float(part.replace('수급', ''))
                            elif '저변동' in part: w_dict['vol'] = float(part.replace('저변동', ''))
                            elif '방어' in part: w_dict['risk'] = float(part.replace('방어', ''))
                        
                        res = run_backtest(p, v, bm, w_dict, TICKER_INFO)
                        if not res.empty:
                            res = res.set_index('Date')
                            res['Cum_Port'] = (1 + res['Port_Ret']).cumprod()
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], name=best_st['전략명'], line=dict(color='#FFD700', width=2)))
                            if not bm.empty:
                                try:
                                    bm_p = bm.loc[start_date:end_date]
                                    bm_re = bm_p / bm_p.iloc[0]
                                    fig.add_trace(go.Scatter(x=bm_re.index, y=bm_re.values, name=BM_NAME, line=dict(dash='dot', color='gray')))
                                except: pass
                            st.plotly_chart(fig, use_container_width=True)
                    else: st.error("분석 결과 없음")
                else: st.error("데이터 로딩 실패")
