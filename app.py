import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr
import yfinance as yf
import requests
import time

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Perfect)", layout="wide", initial_sidebar_state="expanded")

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
    # 기본 시작일
    'DEFAULT_START_DATE': datetime(2015, 1, 1)
}

# ==============================================================================
# [Helper: 날짜 계산]
# ==============================================================================
def get_last_complete_month_end():
    """지난달의 마지막 날짜 반환 (예: 오늘이 5월 15일 -> 4월 30일 반환)"""
    today = datetime.now()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    return last_month_end

# ==============================================================================
# [공통 로직: 팩터 계산 & 랭킹]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    if len(price) < 60 + 20: return None 

    # 유동성 필터
    amt_series = price * volume
    avg_amt = amt_series.iloc[-20:].mean()
    if avg_amt < min_amt: return None

    # Factors
    try:
        mom_short = price.pct_change(20).iloc[-1]
        mom_mid = price.pct_change(60).iloc[-1]
        vol = price.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        liquidity = np.log1p(avg_amt)
        
        # True Cumulative MDD
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
            df = df.sort_values('Amount', ascending=False).head(500)
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()
    except:
        return {"005930":"삼성전자"}, ["005930"]

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
        
        rename_map = {'Symbol': 'Code', 'Ticker': 'Code', 'Security': 'Name', 'Company': 'Name'}
        df = df.rename(columns=rename_map)[['Code', 'Name']].dropna()
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()
    except:
        return {"AAPL":"Apple"}, ["AAPL"]

# ==============================================================================
# [백테스트 데이터 페처 - 기간 선택 적용]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_data_kr(universe, start_date, end_date):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        kospi = fdr.DataReader('KS11', start_str, end_str)['Close']
        kospi.index = pd.to_datetime(kospi.index)
        if kospi.index.tz is not None: kospi.index = kospi.index.tz_localize(None)
    except: kospi = pd.Series(dtype=float)

    p, v = {}, {}
    def get(code):
        try:
            d = fdr.DataReader(code, start_str, end_str)
            if d.empty: return None
            d.index = pd.to_datetime(d.index)
            if d.index.tz is not None: d.index = d.index.tz_localize(None)
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
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        bm = yf.download("^GSPC", start=start_str, end=end_str, progress=False)['Close']
        bm.index = pd.to_datetime(bm.index)
        if bm.index.tz is not None: bm.index = bm.index.tz_localize(None)
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
                    
                    series.index = pd.to_datetime(series.index)
                    if series.index.tz is not None: series.index = series.index.tz_localize(None)
                    if vol is not None:
                        vol.index = pd.to_datetime(vol.index)
                        if vol.index.tz is not None: vol.index = vol.index.tz_localize(None)
                    p[t], v[t] = series, vol
        return p, v, bm
    except: return pd.DataFrame(), pd.DataFrame(), bm

# ==============================================================================
# [★ 핵심 백테스트 엔진 (Realistic)]
# ==============================================================================
def run_backtest(prices, volumes, benchmark, weights, ticker_map):
    if prices.empty: return pd.DataFrame()

    # [수정] 매월 첫 거래일(Month Start) 기준 리밸런싱
    reb_dates = prices.groupby(pd.Grouper(freq='MS')).apply(lambda x: x.index[0])
    
    logs = []
    start_idx = 12 
    
    for i in range(start_idx, len(reb_dates)-1):
        rebal_date = reb_dates[i]       # 리밸런싱 실행일 (매수일)
        next_rebal = reb_dates[i+1]     # 다음 리밸런싱일
        
        # 1. 팩터 계산: 리밸런싱 전일(D-1) 기준 데이터 사용
        # (현실성: 장 시작 전에 전날 종가 기준으로 종목을 뽑아옴)
        try:
            loc = prices.index.get_loc(rebal_date)
            if loc == 0: continue
            signal_date = prices.index[loc - 1] # 전 거래일
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
        
        # 2. 종목 선정
        factor_df = pd.DataFrame(daily_factors).set_index('code')
        ranked = rank_and_score(factor_df, weights)
        picks = ranked.head(CONST['TOP_N']).index.tolist()
        
        if not picks: continue

        # 3. 수익률 계산: 리밸런싱일(당일) 종가 매수 ~ 다음 리밸런싱일 전날 종가
        # (실제 보유 기간)
        try:
            # 매수일 종가 ~ 다음 매수일 종가
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
                'Date': next_rebal,
                'Port_Ret': net_ret,
                'BM_Ret': bm_ret,
                'Top1_Holding': TICKER_INFO.get(picks[0], picks[0]) if picks else "",
                'Holdings_Full': ", ".join([TICKER_INFO.get(x,x) for x in picks])
            })
            
    return pd.DataFrame(logs)

# ==============================================================================
# [메인 컨트롤러]
# ==============================================================================
st.title("🌍 Alpha Seeking Pro (Perfect)")

# --- 1. 사이드바 ---
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
        st.info("대상: 유동성 상위 500개")
        TICKER_INFO, ALL_STOCKS = load_kr_data()
    else:
        COUNTRY = "US"
        CURRENCY = "$"
        COST_RATE = 0.0005
        MIN_AMT = 5_000_000
        VIX_TICKER = "^VIX"
        BM_NAME = "S&P 500"
        us_index = st.selectbox("지수 선택", ["S&P 500", "NASDAQ 100", "DOW 30"])
        with st.spinner("데이터 수집 중..."):
            TICKER_INFO, ALL_STOCKS = load_us_data(us_index)
            
    CONST['MIN_AMT'] = MIN_AMT
    CONST['COST_RATE'] = COST_RATE

    if st.button("🧹 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

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
        "🐆 안전한 사냥": (0.8, 0.7, 0.1, 0.8),
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
            df = fdr.DataReader(ticker, (datetime.now()-timedelta(days=60)))
        else:
            df = yf.Ticker(ticker).history(period="3mo")
        if df.empty: return None, 0
        return df['Close'].iloc[-1], df['Close'].iloc[-1] - df['Close'].iloc[-2]
    except: return None, 0

# ==============================================================================
# [APP 실행]
# ==============================================================================

if mode == "📊 실시간 분석":
    st.subheader(f"📊 {market} 실시간 팩터 랭킹")
    if st.button("분석 실행", type="primary"):
        targets = ALL_STOCKS
        results = []
        bar = st.progress(0, f"분석 중... ({len(targets)}개)")
        
        def worker(t):
            try:
                if COUNTRY == "KR":
                    df = fdr.DataReader(t, (datetime.now()-timedelta(days=400)).strftime('%Y-%m-%d'))
                else:
                    df = yf.Ticker(t).history(period="2y")
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
            
            # Dashboard
            vix, v_delta = get_vix_val(VIX_TICKER)
            c1, c2, c3 = st.columns([1,1,2])
            top = final.iloc[0]
            
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            if vix:
                thr = 22 if COUNTRY=="KR" else 30
                state = "🔴 공포" if vix >= thr else ("🟠 주의" if vix >= (thr-5) else "🟢 안정")
                c3.metric(f"{market[:2]} VIX", f"{vix:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
                c3.caption(f"시장 상태: {state}")
                
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
            st.warning("결과 없음")

else:
    # [NEW] 기간 설정 UI
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("시작일", CONST['DEFAULT_START_DATE'])
    with c2:
        # 종료일 기본값: 지난달 말일
        default_end = get_last_complete_month_end()
        end_date = st.date_input("종료일 (최근 완료된 월 권장)", default_end)

    st.subheader(f"📉 {market} 백테스트 ({start_date} ~ {end_date})")
    st.caption(f"⚡ 조건: 매월 초 리밸런싱 | 수수료 {COST_RATE*100}% | 벤치마크 {BM_NAME}")
    
    if st.button("백테스트 시작", type="primary"):
        if start_date >= end_date:
            st.error("시작일은 종료일보다 빨라야 합니다.")
        else:
            with st.spinner("시뮬레이션 중..."):
                u = ALL_STOCKS[:400] if len(ALL_STOCKS) > 400 else ALL_STOCKS
                
                # Fetch Data with User Range
                if COUNTRY == "KR": p, v, bm = fetch_data_kr(u, start_date, end_date)
                else: p, v, bm = fetch_data_us(u, start_date, end_date)
                    
                if not p.empty:
                    with st.expander("🛠️ 데이터 검증 (Debug)", expanded=False):
                        st.write(f"📅 데이터 범위: {p.index.min().date()} ~ {p.index.max().date()}")
                        st.write(f"📊 종목 수: {len(p.columns)}개")

                    res = run_backtest(p, v, bm, weights, TICKER_INFO)
                    
                    if not res.empty:
                        res = res.set_index('Date')
                        res['Cum_Port'] = (1 + res['Port_Ret']).cumprod()
                        
                        if not bm.empty:
                            bm_period = bm.loc[res.index[0]:res.index[-1]]
                            if not bm_period.empty:
                                res['Cum_BM'] = bm_period / bm_period.iloc[0]
                                res['Cum_BM'] = res['Cum_BM'].reindex(res.index, method='ffill')
                            else: res['Cum_BM'] = 1.0
                        else: res['Cum_BM'] = 1.0
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], name="My Strategy", line=dict(color='#3b82f6', width=2)))
                        fig.add_trace(go.Scatter(x=res.index, y=res['Cum_BM'], name=BM_NAME, line=dict(color='#94a3b8', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        tot = res['Cum_Port'].iloc[-1] - 1
                        y = len(res)/12
                        cagr = (tot+1)**(1/y)-1 if y>0 else 0
                        mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                        ann_vol = res['Port_Ret'].std() * np.sqrt(12)
                        sharpe = cagr / ann_vol if ann_vol > 0 else 0
                        win = (res['Port_Ret']>0).sum()/len(res)
                        
                        st.divider()
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("CAGR", f"{cagr:.1%}")
                        k2.metric("Total Return", f"{tot:.1%}")
                        k3.metric("MDD", f"{mdd:.1%}")
                        k4.metric("Sharpe", f"{sharpe:.2f}")
                        k5.metric("Win Rate", f"{win:.1%}")
                        
                        st.divider()
                        st.dataframe(res[['Port_Ret', 'Holdings_Full', 'BM_Ret']].tail(5), use_container_width=True)
                    else: st.error("백테스트 결과가 없습니다.")
                else: st.error("데이터 로딩 실패")
