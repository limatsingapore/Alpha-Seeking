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
st.set_page_config(page_title="Alpha Seeking Pro (Final)", layout="wide", initial_sidebar_state="expanded")

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
    'BACKTEST_YEARS': 10,      
    'WARMUP_DAYS': 300,
    'MIN_AMT': 5_000_000_000   # 최소 거래대금 50억
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
            # 유동성 상위 500개만 유니버스로 사용
            df_krx = df_krx.sort_values('Amount', ascending=False).head(500)
            
        ticker_info = df_krx.set_index('Code')['Name'].to_dict()
        all_tickers = df_krx['Code'].tolist()
        
        return ticker_info, all_tickers

    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return {}, []

with st.spinner("KRX 데이터 최적화 로딩 중..."):
    TICKER_INFO, ALL_STOCKS_LIST = load_market_data()

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
def fetch_data_batch(universe, days=365*10): 
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
st.title("🧬 Alpha Seeking Pro (Final)")

with st.sidebar:
    st.header("⚙️ 전략 설정")
    
    # 10가지 프리셋 정의 (가중치: Mom, Liq, Vol, Risk)
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5),
        "🔥 야수의 심장 (공격 몰빵)": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말 (추세 추종)": (1.0, 0.5, 0.2, 0.3),
        "🌊 세력주 포착 (수급 집중)": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어 (극강 수비)": (0.1, 0.1, 1.0, 1.0),
        "🧘 마음의 평화 (저변동성)": (0.3, 0.2, 1.0, 0.5),
        "🚑 좀비 헌터 (낙폭과대)": (0.4, 0.3, 0.3, 1.0),
        "⚖️ 황금 밸런스 (중립)": (0.5, 0.5, 0.5, 0.5),
        "💎 우상향 정석 (성장+안정)": (0.7, 0.3, 0.7, 0.4),
        "🐆 안전한 사냥 (공격8:수비2)": (0.8, 0.7, 0.1, 0.8),
        "🧠 스마트 머니 (수급+방어)": (0.5, 0.8, 0.3, 0.8)
    }
    
    with st.expander("📚 전략 프리셋 가이드 (10선)", expanded=False):
        st.markdown("""
        ### ⚔️ 공격형
        1. **🔥 야수의 심장**: 리스크 무시, 오직 수익률과 거래량만 본다.
        2. **🚀 달리는 말**: 전형적인 추세 추종. 가는 놈이 더 간다.
        3. **🌊 세력주 포착**: 가격은 아직이나 돈(거래량)이 수상하게 몰리는 종목.

        ### 🛡️ 수비형
        4. **🏰 철벽 방어**: 하락장에서 내 돈을 지키는 것이 목표.
        5. **🧘 마음의 평화**: 밤에 발 뻗고 잘 수 있는 얌전한 주식.
        6. **🚑 좀비 헌터**: 바닥이 단단하여 더 떨어질 곳이 없는 종목.

        ### ⚖️ 균형형
        7. **⚖️ 황금 밸런스**: 모든 지표를 골고루 섞은 모범생.
        8. **💎 우상향 정석**: 적당한 상승 추세와 낮은 변동성의 조화.
        9. **🐆 안전한 사냥 (추천)**: 공격적으로 수익을 내되 안전벨트는 착용.
        10. **🧠 스마트 머니**: 메이저 수급이 들어와 가격 관리가 되는 종목.
        """)
    
    selected_preset = st.selectbox("전략 프리셋 선택", list(PRESETS.keys()), index=9) # 기본값: 안전한 사냥
    def_w = PRESETS[selected_preset]

    w_mom = st.slider("📈 추세 (Momentum)", 0.0, 1.0, def_w[0], 0.1)
    w_liq = st.slider("🌊 수급 (Liquidity)", 0.0, 1.0, def_w[1], 0.1)
    w_vol = st.slider("⚖️ 저변동성 (Low Vol)", 0.0, 1.0, def_w[2], 0.1)
    w_risk = st.slider("🛡️ 방어력 (MDD)", 0.0, 1.0, def_w[3], 0.1)
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}
    
    st.divider()
    mode = st.radio("모드", ["📊 실시간 스크리닝", "📉 백테스트 (속도 최적화)"])

# ------------------------------------------------------------------------------
# TAB 1: 실시간
# ------------------------------------------------------------------------------
if mode == "📊 실시간 스크리닝":
    st.subheader("실시간 팩터 랭킹 (Top 500 Universe)")
    st.info("💡 테마별 보기를 제거하고, **전체 유니버스(유동성 상위 500개)**를 대상으로 통합 분석합니다.")
    
    if st.button("전체 종목 분석 실행", type="primary"):
        # 전체 유니버스 대상 (속도 최적화를 위해 이미 500개로 필터링됨)
        target_list = ALL_STOCKS_LIST
        
        results = []
        bar = st.progress(0, "전체 시장 데이터 스캔 중...")
        
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
