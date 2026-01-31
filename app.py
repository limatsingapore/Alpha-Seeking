import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr
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
    'BACKTEST_YEARS': 10,      # [수정] 다시 10년으로 복구
    'WARMUP_DAYS': 300,
    'MIN_AMT': 5_000_000_000   # 최소 거래대금 50억
}

# [테마 매핑]
THEME_MAP = {
    "🤖 AI/반도체": ["000660", "058470", "005930", "042700", "036540", "357780"],
    "🚀 방산/우주": ["012450", "064350", "005950", "079550", "047810"],
    "💊 바이오/헬스": ["207940", "068270", "214370", "290650", "145020"],
    "🔋 2차전지": ["006400", "003670", "247540", "051910"],
    "🚗 모빌리티": ["005380", "000270", "012330", "003620"],
    "🏦 밸류업(금융/지주)": ["003550", "055550", "105560", "086790", "000810"]
}

# ==============================================================================
# [Core Logic 1: Factor Calculation]
# ==============================================================================
def calculate_factors(price, volume):
    if len(price) < CONST['MOM_MID'] + 20: return None

    # 유동성 필터 (50억)
    amt_series = price * volume
    avg_amt = amt_series.iloc[-20:].mean()
    if avg_amt < CONST['MIN_AMT']: return None

    # Factors
    mom_short = price.pct_change(CONST['MOM_SHORT']).iloc[-1]
    mom_mid = price.pct_change(CONST['MOM_MID']).iloc[-1]
    vol = price.pct_change().tail(CONST['VOL_WINDOW']).std() * np.sqrt(CONST['TRADING_DAYS'])
    liquidity = np.log1p(avg_amt)
    
    # True Cumulative MDD
    window_price = price.tail(CONST['VOL_WINDOW'])
    roll_max = window_price.cummax()
    daily_dd = (window_price / roll_max) - 1.0
    mdd = daily_dd.min()
    
    return {
        'mom_short': mom_short, 'mom_mid': mom_mid,
        'volatility': vol, 'liquidity': liquidity, 'mdd': mdd,
        'price': price.iloc[-1]
    }

# ==============================================================================
# [Core Logic 2: Ranking]
# ==============================================================================
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
# [Data Loader]
# ==============================================================================
@st.cache_data(ttl=3600*12)
def load_market_data():
    try:
        df_krx = fdr.StockListing('KRX')
        if 'Symbol' in df_krx.columns: df_krx.rename(columns={'Symbol':'Code'}, inplace=True)
        
        df_krx = df_krx[~df_krx['Name'].str.contains('스팩|우B|우|리츠|홀딩스', na=False)]
        
        if 'Amount' in df_krx.columns:
            df_krx = df_krx.sort_values('Amount', ascending=False).head(500)
            
        ticker_info = df_krx.set_index('Code')['Name'].to_dict()
        
        ticker_theme = {}
        for code in df_krx['Code']:
            my_themes = []
            for theme, codes in THEME_MAP.items():
                if code in codes: my_themes.append(theme)
            ticker_theme[code] = ", ".join(my_themes) if my_themes else "기타"

        all_tickers = df_krx['Code'].tolist()
        return THEME_MAP, ticker_info, all_tickers, ticker_theme

    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return {}, {}, [], {}

with st.spinner("KRX 데이터 최적화 로딩 중..."):
    THEMES, TICKER_INFO, ALL_STOCKS_LIST, TICKER_THEME = load_market_data()

@st.cache_data(ttl=600)
def get_vix_enhanced():
    try:
        end = datetime.now()
        start = end - timedelta(days=100)
        df = fdr.DataReader('KS200VIX', start)
        if df.empty: return None, 0, []
        curr = df['Close'].iloc[-1]
        delta = curr - df['Close'].iloc[-2]
        return curr, delta, df['Close'].tail(30)
    except: return None, 0, []

# ==============================================================================
# [Backtest Engine]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_data_batch(universe, days=365*10): # [수정] 10년치 확보
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    kospi = fdr.DataReader('KS11', start_date)['Close']
    
    adj_price = {}
    adj_vol = {}
    
    def get_stock(code):
        try:
            d = fdr.DataReader(code, start_date)
            if d.empty: return None
            return code, d['Close'], d['Volume']
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(get_stock, code) for code in universe]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                c, p, v = res
                adj_price[c] = p
                adj_vol[c] = v
            
    return pd.DataFrame(adj_price), pd.DataFrame(adj_vol), kospi

def run_backtest(prices, volumes, benchmark, weights, ticker_map):
    reb_dates = prices.resample('M').last().index
    logs = []
    start_i = 12 
    
    for i in range(start_i, len(reb_dates)-1):
        curr_date = reb_dates[i]
        next_date = reb_dates[i+1]
        
        p_sub = prices.loc[:curr_date].tail(300)
        v_sub = volumes.loc[:curr_date].tail(300)
        active_tickers = p_sub.columns[p_sub.iloc[-1].notna()]
        
        daily_factors = []
        for t in active_tickers:
            f = calculate_factors(p_sub[t], v_sub[t])
            if f:
                f['code'] = t
                daily_factors.append(f)
                
        if not daily_factors: continue
        
        factor_df = pd.DataFrame(daily_factors).set_index('code')
        ranked = rank_and_score(factor_df, weights)
        top_picks = ranked.head(CONST['TOP_N']).index.tolist()
        
        fwd_ret = prices.loc[curr_date:next_date, top_picks].pct_change().dropna()
        if fwd_ret.empty: continue
            
        port_ret = (1 + fwd_ret).prod().mean() - 1
        bm_slice = benchmark.loc[curr_date:next_date].pct_change().dropna()
        bm_ret = (1 + bm_slice).prod() - 1
        net_ret = port_ret - CONST['COST_RATE']
        
        holdings_names = [ticker_map.get(x, x) for x in top_picks]
        top_1_name = holdings_names[0] if holdings_names else ""
        
        logs.append({
            'Date': next_date,
            'Port_Ret': net_ret,
            'BM_Ret': bm_ret,
            'Top1_Holding': top_1_name,
            'Holdings_Full': ", ".join(holdings_names)
        })
        
    return pd.DataFrame(logs)

# ==============================================================================
# [UI 구성]
# ==============================================================================
st.title("🧬 Alpha Seeking Pro (Ultimate)")

with st.sidebar:
    st.header("⚙️ 전략 설정")
    
    with st.expander("📚 전략 프리셋 가이드", expanded=False):
        st.markdown("""
        **1. 🔥 야수의 심장** (추세 1.0 / 수급 1.0)
        **2. 🐆 안전한 사냥** (추세 0.8 / 수급 0.7 / 방어 0.8) - *추천*
        **3. 🛡️ 철벽 방어** (저변동 1.0 / 방어 1.0)
        """)
    
    preset = st.selectbox("프리셋 선택", ["사용자 정의", "🐆 안전한 사냥 (추천)", "🔥 야수의 심장", "🛡️ 철벽 방어"])
    if preset == "🐆 안전한 사냥 (추천)": def_w = (0.8, 0.7, 0.1, 0.8)
    elif preset == "🔥 야수의 심장": def_w = (1.0, 1.0, 0.0, 0.0)
    elif preset == "🛡️ 철벽 방어": def_w = (0.1, 0.1, 1.0, 1.0)
    else: def_w = (0.4, 0.2, 0.2, 0.2)

    w_mom = st.slider("📈 추세", 0.0, 1.0, def_w[0], 0.1)
    w_liq = st.slider("🌊 수급", 0.0, 1.0, def_w[1], 0.1)
    w_vol = st.slider("⚖️ 저변동성", 0.0, 1.0, def_w[2], 0.1)
    w_risk = st.slider("🛡️ 방어력", 0.0, 1.0, def_w[3], 0.1)
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}
    
    st.divider()
    mode = st.radio("모드", ["📊 실시간 스크리닝", "📉 백테스트 (속도 최적화)"])

# ------------------------------------------------------------------------------
# TAB 1: 실시간
# ------------------------------------------------------------------------------
if mode == "📊 실시간 스크리닝":
    st.subheader("실시간 팩터 랭킹")
    
    filter_opt = st.radio("필터 기준", ["전체 유니버스 (Top 500)", "테마별 보기"], horizontal=True)
    target_list = []
    if filter_opt == "테마별 보기":
        thm = st.selectbox("테마 선택", list(THEMES.keys()))
        target_list = THEMES[thm]
    else:
        target_list = ALL_STOCKS_LIST
    
    if st.button("분석 실행", type="primary"):
        results = []
        bar = st.progress(0, "데이터 분석 중...")
        
        def get_snapshot(t):
            try:
                start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
                df = fdr.DataReader(t, start)
                if len(df) < 200: return None
                f = calculate_factors(df['Close'], df['Volume'])
                if f: f['code'] = t
                return f
            except: return None
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(get_snapshot, t) for t in target_list]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res: results.append(res)
                bar.progress((i+1)/len(target_list))
        bar.empty()
        
        if results:
            f_df = pd.DataFrame(results).set_index('code')
            final = rank_and_score(f_df, weights)
            
            # VIX
            v, vd, vh = get_vix_enhanced()
            c1, c2, c3 = st.columns([1,1,2])
            top = final.iloc[0]
            
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            if v:
                stt = "🔴 공포" if v>=22 else ("🟠 주의" if v>=17 else "🟢 안정")
                c3.metric("K-VIX", f"{v:.2f}", f"{vd:+.2f}", delta_color="inverse")
                c3.caption(f"시장 상태: {stt}")
            
            # Table
            disp = final[['price', 'Total_Score', 'mom_short', 'mdd']].copy()
            disp.columns = ['Price', 'Score', 'Mom(1M)', 'MDD']
            disp.index = [TICKER_INFO.get(x,x) for x in disp.index]
            
            st.dataframe(
                disp, use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="%d"),
                    "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                    "Mom(1M)": st.column_config.NumberColumn(format="%.2%"),
                    "MDD": st.column_config.NumberColumn(format="%.2%")
                }
            )
        else:
            st.warning("조건을 만족하는 종목이 없습니다.")

# ------------------------------------------------------------------------------
# TAB 2: 백테스트
# ------------------------------------------------------------------------------
else:
    st.subheader(f"📉 Walk-Forward Backtest ({CONST['BACKTEST_YEARS']} Years)")
    st.info("⚡ 속도 최적화 적용: 유동성 상위 400개 종목 샘플링 & 병렬 처리")
    
    if st.button("백테스트 시작", type="primary"):
        with st.spinner("과거 10년 데이터 로딩 및 시뮬레이션 중... (최대 1~2분 소요)"):
            universe = ALL_STOCKS_LIST[:400]
            p_df, v_df, bm = fetch_data_batch(universe, days=365 * CONST['BACKTEST_YEARS'])
            
            if not p_df.empty:
                res = run_backtest(p_df, v_df, bm, weights, TICKER_INFO)
                
                if not res.empty:
                    res['Date'] = pd.to_datetime(res['Date'])
                    res = res.set_index('Date')
                    
                    res['Cum_Port'] = (1 + res['Port_Ret']).cumprod()
                    res['Cum_BM'] = (1 + res['BM_Ret']).cumprod()
                    
                    # 차트 그리기
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=res.index, y=res['Cum_Port'], 
                        name='My Strategy',
                        line=dict(color='#ef4444', width=2),
                        hovertemplate='<b>Date</b>: %{x}<br><b>Return</b>: %{y:.2f}<br><b>Top1</b>: %{customdata}',
                        customdata=res['Top1_Holding']
                    ))
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_BM'], name='KOSPI', line=dict(color='#94a3b8', dash='dot')))
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], mode='markers', marker=dict(size=4, color='red'), name='Rebalancing', showlegend=False))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 성과 지표 (Metrics)
                    port_rets = res['Port_Ret']
                    cum_ret = res['Cum_Port'].iloc[-1]
                    total_ret = cum_ret - 1
                    months = len(port_rets)
                    years = months / 12
                    
                    cagr = (cum_ret) ** (1 / years) - 1 if years > 0 else 0
                    ann_vol = port_rets.std() * np.sqrt(12)
                    sharpe = cagr / ann_vol if ann_vol != 0 else 0
                    
                    running_max = res['Cum_Port'].cummax()
                    drawdown = (res['Cum_Port'] / running_max) - 1
                    max_dd = drawdown.min()
                    
                    win_months = (port_rets > 0).sum()
                    win_rate = win_months / months if months > 0 else 0
                    
                    st.divider()
                    st.subheader("📊 전략 성과 정밀 분석")
                    
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("연평균 수익률 (CAGR)", f"{cagr:.1%}", help="복리 개념이 적용된 실제 연 수익률입니다.")
                    k2.metric("총 누적 수익률", f"{total_ret:.1%}", help="백테스트 전체 기간 동안의 총 수익률입니다.")
                    k3.metric("최대 낙폭 (MDD)", f"{max_dd:.1%}", help="최악의 경우 겪을 수 있는 하락폭입니다. 낮을수록 좋습니다.")
                    k4.metric("샤프 지수 (위험대비)", f"{sharpe:.2f}", help="1.0 이상이면 양호, 2.0 이상이면 매우 훌륭한 전략입니다.")
                    k5.metric("월간 승률 (Win Rate)", f"{win_rate:.1%}", f"{win_months}/{months}개월", help="전체 기간 중 수익을 낸 달의 비율입니다.")
                    
                    st.divider()
                    st.caption("📜 월별 상세 운용 기록")
                    st.dataframe(res[['Port_Ret', 'Holdings_Full', 'BM_Ret']].tail(10), use_container_width=True)
                else:
                    st.error("결과 없음")
            else:
                st.error("데이터 로딩 실패")
