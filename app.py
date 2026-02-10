import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr
import time

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Final v9.1)", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] { background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.1rem !important; }
    div[data-testid="stExpander"] { background-color: #1e293b; border-radius: 8px; }
    .stProgress { margin-top: 20px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# [상수 및 설정]
# ==============================================================================
CONST = {
    'TRADING_DAYS': 252,       
    'MOM_SHORT': 20,           
    'MOM_MID': 60,             
    'VOL_WINDOW': 252,         
    'DEFAULT_START_DATE': datetime(2018, 1, 1),
    'RISK_FREE_RATE': 0.035,
    'COST_RATE': 0.002, 
    'MIN_AMT': 5_000_000_000, 
    'TOP_N': 20 
}

# ==============================================================================
# [프리셋 정의] - (복구 완료)
# ==============================================================================
PRESETS = {
    "사용자 정의": (0.5, 0.5, 0.5, 0.5), "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
    "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3), "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
    "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0), "🧘 마음의 평화": (0.3, 0.2, 1.0, 0.5),
    "🚑 좀비 헌터": (0.4, 0.3, 0.3, 1.0), "⚖️ 황금 밸런스": (0.5, 0.5, 0.5, 0.5),
    "💎 우상향 정석": (0.7, 0.3, 0.7, 0.4), "🐆 안전한 사냥": (0.8, 0.7, 0.1, 0.8),
    "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8), "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1), 
    "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9), "🎯 퀄리티 그로스": (0.6, 0.6, 0.6, 0.6), 
    "🌪️ 변동성 사냥꾼": (0.7, 0.5, 0.0, 0.2), "🦅 매파의 눈": (0.3, 0.9, 0.4, 0.7),
    
    # [복구됨] 시장 상황별 최적화 프리셋
    "📈 상승장 최적화": (0.9, 0.7, 0.0, 0.1),
    "📉 하락장 최적화": (0.2, 0.3, 0.8, 0.9),
    "🦀 횡보장 최적화": (0.5, 0.8, 0.4, 0.6)
}

# ==============================================================================
# [Helper Functions]
# ==============================================================================
def get_last_complete_month_end():
    return datetime.now()

def z_score(x):
    if x.std() == 0: return x * 0
    return (x - x.mean()) / x.std()

def calculate_slippage(amount_traded, avg_daily_volume):
    if avg_daily_volume == 0: return 0.01 
    participation_rate = amount_traded / avg_daily_volume
    if participation_rate < 0.01: return 0.002  
    elif participation_rate < 0.05: return 0.003 
    elif participation_rate < 0.10: return 0.005 
    else: return 0.01 

def infer_sector_kr(name, code=None):
    exact_mapping = {
        '005930': '반도체/IT', '000660': '반도체/IT', 
        '373220': '2차전지', '006400': '2차전지',
        '207940': '바이오/제약', '068270': '바이오/제약',
        '005380': '자동차/부품', '000270': '자동차/부품',
        '035420': '인터넷/게임', '035720': '인터넷/게임',
        '105560': '금융/지주', '055550': '금융/지주'
    }
    if code and code in exact_mapping: return exact_mapping[code]

    name = str(name)
    if any(x in name for x in ['스팩', '제호', '기업인수']): return '스팩/금융'
    if any(x in name for x in ['우', '우B']): return '우선주'
    
    keywords = {
        '반도체/IT': ['반도체', '테크', '칩', '시스템', '전자', '이노텍', 'DB하이텍', '주성', '이오테크닉스', '솔브레인'],
        '2차전지': ['에코프로', '엘앤에프', '에너지', 'SK이노베이션', '포스코퓨처', '엔켐', '금양'],
        '바이오/제약': ['바이오', '제약', '약품', '생명', '헬스', '유한양행', '한미', 'HLB', '알테오젠'],
        '자동차/부품': ['모비스', '타이어', '만도', '오토', '화신', '에스엘'],
        '인터넷/게임': ['게임', '소프트', '엔씨', '펄어비스', '크래프톤', '넷마블', '위메이드'],
        '엔터/미디어': ['엔터', '스튜디오', '미디어', '에스엠', 'JYP', 'YG', '하이브', 'CJ ENM'],
        '금융/지주': ['금융', '지주', '은행', '증권', '보험', '카드', '투자', '홀딩스', '메리츠', '하나', '우리'],
        '조선/중공업': ['중공업', '조선', '기계', '엔진', '현대미포', '한국조선', '두산', '한화오션', '현대로템', 'LIG넥원'],
        '화학/정유': ['화학', '케미칼', '정유', 'S-Oil', '롯데정밀', '효성', '금호'],
        '건설/건자재': ['건설', '개발', '엔지니어링', '시멘트', '페인트', '현대건설', 'GS건설'],
        '소비재/유통': ['푸드', '식품', '제과', '쇼핑', '백화점', '이마트', '호텔', '항공', '화장품', '아모레']
    }
    for sector, keys in keywords.items():
        if any(k in name for k in keys): return sector
    return '기타/소형주'

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
            df = df.sort_values('Amount', ascending=False).head(300) 
        elif 'Marcap' in df.columns:
            df = df.sort_values('Marcap', ascending=False).head(300)
        else:
            df = df.head(300) 
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()
    except:
        return {"005930":"삼성전자"}, ["005930"]

@st.cache_data(ttl=3600*24)
def fetch_data_serial(universe, start_date, end_date):
    real_start_date = start_date - timedelta(days=400)
    s_str = real_start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    p_dict, v_dict, bm_dict = {}, {}, {}
    
    try:
        kospi = fdr.DataReader('KS11', s_str, e_str)
        if not kospi.empty: bm_dict['KOSPI'] = kospi['Close']
    except: pass
    
    total = len(universe)
    for i, code in enumerate(universe):
        try:
            d = fdr.DataReader(code, s_str, e_str)
            if len(d) > 120 and d['Close'].iloc[-1] > 0:
                if 'Close' in d.columns: p_dict[code] = d['Close']
                if 'Volume' in d.columns: v_dict[code] = d['Volume']
        except: pass
            
    df_p = pd.DataFrame(p_dict)
    df_v = pd.DataFrame(v_dict)
    
    if not df_p.empty:
        full_idx = pd.date_range(start=df_p.index.min(), end=df_p.index.max(), freq='B')
        df_p = df_p.reindex(full_idx).ffill()
        df_v = df_v.reindex(full_idx).fillna(0)
        for k in bm_dict:
            bm_dict[k] = bm_dict[k].reindex(full_idx).ffill()
            
    return df_p, df_v, bm_dict

# ==============================================================================
# [Core Logic]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    if len(price) < 120 or price.iloc[-1] == 0 or np.isnan(price.iloc[-1]): return None
    
    # [거래정지 감지 필터 유지]
    zero_volume_days = (volume.tail(20) == 0).sum()
    if zero_volume_days >= 3: return None 

    try:
        p = price
        v = volume
        amt = p * v
        if amt.iloc[-20:].mean() < min_amt: return None
        
        mom_short = p.pct_change(20).iloc[-1]
        mom_mid = p.pct_change(60).iloc[-1]
        vol = p.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        liquidity = np.log1p(amt.iloc[-20:].mean())
        
        roll_max = p.tail(trading_days).cummax()
        daily_dd = p.tail(trading_days) / roll_max - 1.0
        mdd = daily_dd.min()
        
        return {
            'mom_short': mom_short, 'mom_mid': mom_mid,
            'volatility': vol, 'liquidity': liquidity, 'mdd': mdd,
            'price': p.iloc[-1]
        }
    except: return None

def rank_and_score(factor_df, weights, ticker_map=None):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    
    if ticker_map:
        scored['sector'] = [infer_sector_kr(ticker_map.get(x, x), x) for x in scored.index]
    else:
        scored['sector'] = 'Unknown'

    for col in ['mom_short', 'mom_mid', 'volatility', 'liquidity', 'mdd']:
        if col in scored.columns:
            scored[col] = scored[col].fillna(scored[col].median())

    def sector_z(x): return (x - x.mean())/x.std() if len(x) > 2 else 0
    
    scored['abs_mdd'] = scored['mdd'].abs()
    scored['Z_MDD'] = scored.groupby('sector')['abs_mdd'].transform(sector_z).fillna(0) * -1

    scored['Z_Mom_S'] = scored.groupby('sector')['mom_short'].transform(sector_z).fillna(0)
    scored['Z_Mom_M'] = scored.groupby('sector')['mom_mid'].transform(sector_z).fillna(0)
    scored['Z_Vol'] = scored.groupby('sector')['volatility'].transform(sector_z).fillna(0) * -1 
    scored['Z_Liq'] = scored.groupby('sector')['liquidity'].transform(sector_z).fillna(0)
    
    scored['Total_Score'] = (
        scored['Z_Mom_S'] * weights['mom'] * 0.5 + 
        scored['Z_Mom_M'] * weights['mom'] * 0.5 +
        scored['Z_Vol'] * weights['vol'] + 
        scored['Z_Liq'] * weights['liq'] +
        scored['Z_MDD'] * weights['risk']
    )
    
    return scored.sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [백테스트 엔진]
# ==============================================================================
def run_backtest(prices, volumes, weights, ticker_map, const, benchmark=None):
    if prices.empty: return pd.DataFrame()
    
    reb_dates = prices.resample('BM').last().index
    logs = []
    prev_picks = [] 
    target_n = CONST['TOP_N'] 
    
    if benchmark is None or benchmark.empty:
        benchmark = pd.Series(1.0, index=prices.index)
    else:
        benchmark = benchmark.reindex(prices.index).ffill().fillna(method='bfill')

    valid_dates = [d for d in reb_dates if d >= prices.index[0] + timedelta(days=60)]
    if not valid_dates: return pd.DataFrame()

    start_idx = 0
    for i, d in enumerate(reb_dates):
        if d in valid_dates:
            start_idx = i
            break
            
    loop_range = range(start_idx, len(reb_dates))
    
    for i in loop_range:
        rebal_date = reb_dates[i] 
        if i < len(reb_dates) - 1: next_rebal = reb_dates[i+1]
        else: next_rebal = prices.index[-1]
        if rebal_date > prices.index[-1]: break
            
        try:
            min_trade_amt = CONST['MIN_AMT']
            rebal_idx = prices.index.searchsorted(rebal_date)
            if rebal_idx <= 0: continue
            selection_date = prices.index[rebal_idx - 1]
            
            p_sub = prices.loc[:selection_date].tail(300)
            v_sub = volumes.loc[:selection_date].tail(300)
            
            valid_cols = p_sub.columns[p_sub.iloc[-1].notna()]
            daily_factors = []
            for t in valid_cols:
                f = calculate_factors(p_sub[t], v_sub[t], min_trade_amt)
                if f:
                    f['code'] = t; daily_factors.append(f)
            
            if not daily_factors: continue 
            
            factor_df = pd.DataFrame(daily_factors).set_index('code')
            ranked = rank_and_score(factor_df, weights, ticker_map=ticker_map)
            
            picks = ranked.head(target_n).index.tolist()
            if not picks: continue
            
            buy_date = prices.index[rebal_idx]
            sell_idx = prices.index.searchsorted(next_rebal)
            if sell_idx >= len(prices): sell_idx = len(prices) - 1
            sell_date = prices.index[sell_idx]
            
            current_prices = {}
            for t in picks:
                buy_p = prices.at[buy_date, t]
                if not np.isnan(buy_p):
                    current_prices[t] = buy_p

            curr_prices = prices.loc[buy_date, picks].fillna(0).replace(0, np.nan).ffill()
            next_prices = prices.loc[sell_date, picks].fillna(0)
            
            ret_vec = (next_prices / curr_prices) - 1
            ret_vec = ret_vec.fillna(0)
            gross_ret = ret_vec.mean()
            
            # [비용 로직 유지: 회전율 반영]
            if not prev_picks: 
                turnover = 1.0
            else:
                kept = set(prev_picks) & set(picks)
                turnover = (len(picks) - len(kept)) / len(picks)
            
            assumed_capital = 100_000_000
            target_amt_per_stock = assumed_capital / len(picks)
            
            s_costs = []
            for t in picks:
                avg_v = volumes.loc[:buy_date, t].tail(20).mean()
                if np.isnan(avg_v) or avg_v == 0:
                    s_costs.append(0.01)
                else:
                    s_costs.append(calculate_slippage(target_amt_per_stock, avg_v))
            avg_slippage = np.mean(s_costs) if s_costs else 0.002
            
            total_cost = (CONST['COST_RATE'] + avg_slippage) * turnover
            net_ret = gross_ret - total_cost
            
            try:
                b_s = benchmark.asof(buy_date)
                b_e = benchmark.asof(sell_date)
                if isinstance(b_s, pd.Series): b_s = b_s.iloc[0]
                if isinstance(b_e, pd.Series): b_e = b_e.iloc[0]
                bm_ret = (b_e / b_s) - 1 if b_s != 0 else 0.0
            except: bm_ret = 0.0
                
            logs.append({
                'Date': buy_date, 
                'Sell_Date': sell_date, 
                'Gross_Ret': gross_ret, 
                'Net_Ret': net_ret,
                'BM_Ret': bm_ret, 
                'Holdings_List': picks, 
                'Prices_Dict': current_prices, 
                'Port_Ret': net_ret
            })
            prev_picks = picks 
            
        except: continue
            
    return pd.DataFrame(logs)

def calculate_metrics(res_df):
    if res_df.empty: return {}
    
    mean_ret = res_df['Port_Ret'].mean() * 12
    volatility = res_df['Port_Ret'].std() * np.sqrt(12)
    sharpe = (mean_ret - 0.035) / volatility if volatility != 0 else 0
    
    downside_returns = res_df.loc[res_df['Port_Ret'] < 0, 'Port_Ret']
    downside_vol = downside_returns.std() * np.sqrt(12)
    sortino = (mean_ret - 0.035) / downside_vol if downside_vol != 0 else 0
    
    cum = (1 + res_df['Port_Ret']).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    
    try:
        if len(res_df) > 1:
            slope, intercept = np.polyfit(res_df['BM_Ret'], res_df['Port_Ret'], 1)
            beta = slope
            alpha = intercept * 12 
        else:
            beta, alpha = 1.0, 0.0
    except:
        beta, alpha = 1.0, 0.0
        
    return {
        'CAGR': (cum.iloc[-1])**(1/(len(res_df)/12)) - 1,
        'MDD': mdd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Alpha': alpha,
        'Beta': beta,
        'Win_Rate': (res_df['Port_Ret'] > 0).sum() / len(res_df)
    }

# ==============================================================================
# [UI MAIN]
# ==============================================================================
st.title("🇰🇷 Alpha Seeking Pro (Final v9.1)")

def reset_results():
    st.session_state['bt_ran'] = False
    st.session_state['bt_res'] = pd.DataFrame()

def update_sliders():
    ps = st.session_state['preset_select']
    if ps in PRESETS:
        vals = PRESETS[ps]
        st.session_state['slider_mom'] = vals[0]
        st.session_state['slider_liq'] = vals[1]
        st.session_state['slider_vol'] = vals[2]
        st.session_state['slider_risk'] = vals[3]
    reset_results()

c_d1, c_d2 = st.columns(2)
with c_d1: 
    s_d = st.date_input("분석 시작일", CONST['DEFAULT_START_DATE'], key="start_date_common", on_change=reset_results)
with c_d2: 
    e_d = st.date_input("분석 종료일", get_last_complete_month_end(), key="end_date_common", on_change=reset_results)

with st.sidebar:
    st.info("대상: KOSPI/KOSDAQ 유동성 상위 300개")
    TICKER_INFO, ALL_STOCKS = load_kr_data()
            
    if st.button("🧹 캐시 초기화", key="clear_cache"): st.cache_data.clear(); st.rerun()
    st.divider()
    
    mode = st.radio("모드 선택", ["📉 백테스트", "🔍 전략 최적화"], key="mode_radio", on_change=reset_results)
    st.divider()
    
    if mode == "📉 백테스트":
        sel_preset = st.selectbox("전략 프리셋", list(PRESETS.keys()), index=9, key="preset_select", on_change=update_sliders)
        
        if 'slider_mom' not in st.session_state:
            init_vals = PRESETS["🐆 안전한 사냥"]
            st.session_state['slider_mom'] = init_vals[0]
            st.session_state['slider_liq'] = init_vals[1]
            st.session_state['slider_vol'] = init_vals[2]
            st.session_state['slider_risk'] = init_vals[3]

        w_mom = st.slider("📈 추세", 0.0, 1.0, key="slider_mom", step=0.1, on_change=reset_results)
        w_liq = st.slider("🌊 수급", 0.0, 1.0, key="slider_liq", step=0.1, on_change=reset_results)
        w_vol = st.slider("⚖️ 저변동", 0.0, 1.0, key="slider_vol", step=0.1, on_change=reset_results)
        w_risk = st.slider("🛡️ 방어", 0.0, 1.0, key="slider_risk", step=0.1, on_change=reset_results)

if mode == "📉 백테스트":
    st.write("") 
    if st.button("실행", type="primary", key="btn_run_backtest"):
        # [핵심] Force Injection
        forced_weights = {
            'mom': st.session_state['slider_mom'],
            'liq': st.session_state['slider_liq'],
            'vol': st.session_state['slider_vol'],
            'risk': st.session_state['slider_risk']
        }

        prog_bar = st.progress(0, text="데이터 불러오는 중...")
        p, v, bms = fetch_data_serial(ALL_STOCKS, s_d, e_d)
        prog_bar.progress(0.2, text="데이터 수집 완료. 백테스트 엔진 가동...")
        
        if not p.empty:
            main_bm = bms.get('KOSPI')
            if main_bm is None: main_bm = pd.Series(1.0, index=p.index)
            
            res = run_backtest(p, v, forced_weights, TICKER_INFO, CONST, benchmark=main_bm)
            prog_bar.progress(0.8, text="시뮬레이션 완료. 결과 분석 및 시각화 중...")
            
            st.session_state['bt_p'] = p
            st.session_state['bt_bms'] = bms
            st.session_state['bt_res'] = res
            st.session_state['bt_ran'] = True
            st.session_state['last_weights'] = forced_weights
            
            prog_bar.progress(1.0, text="완료!")
            time.sleep(0.3)
            prog_bar.empty()
        else:
            st.error("데이터 수집 실패")

    if st.session_state.get('bt_ran'):
        lw = st.session_state.get('last_weights', {})
        preset_name = st.session_state.get('preset_select', '사용자 정의')
        st.info(f"✅ **현재 적용된 전략:** [{preset_name}] (추세 {lw.get('mom')} / 수급 {lw.get('liq')} / 저변동 {lw.get('vol')} / 방어 {lw.get('risk')})")

        res = st.session_state['bt_res']
        bms = st.session_state.get('bt_bms', {})
        
        if not res.empty:
            res_chart = res.set_index('Date')
            res_chart['Cum'] = (1+res_chart['Port_Ret']).cumprod()
            
            mets = calculate_metrics(res_chart)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("CAGR", f"{mets['CAGR']:.1%}")
            m2.metric("MDD", f"{mets['MDD']:.1%}")
            m3.metric("Sharpe", f"{mets['Sharpe']:.2f}")
            m4.metric("Sortino", f"{mets['Sortino']:.2f}")
            m5.metric("Win Rate", f"{mets['Win_Rate']:.1%}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_chart.index, y=res_chart['Cum'], name="Port", line=dict(width=3, color='blue')))
            if 'KOSPI' in bms:
                b = bms['KOSPI'].reindex(res_chart.index, method='ffill')
                b = b / b.iloc[0]
                fig.add_trace(go.Scatter(x=b.index, y=b, name='KOSPI', line=dict(dash='dot', color='red')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("📅 월별 상세 분석")
            date_options = res['Date'].dt.strftime('%Y-%m-%d').tolist()
            sel_date_str = st.selectbox("매수 시점", date_options[::-1], index=0)
            
            if sel_date_str:
                row = res[res['Date'] == pd.to_datetime(sel_date_str)].iloc[0]
                codes = row['Holdings_List']
                prices = row['Prices_Dict']
                det = []
                for c in codes:
                    det.append({
                        "종목명": TICKER_INFO.get(c, c),
                        "코드": c,
                        "섹터": infer_sector_kr(TICKER_INFO.get(c, c), c),
                        "매수가(종가)": f"{prices.get(c,0):,.0f}원"
                    })
                st.dataframe(pd.DataFrame(det), use_container_width=True)
                st.caption(f"※ {sel_date_str} 종가 매수 (Standard Rebalancing)")

elif mode == "🔍 전략 최적화":
    st.info("모든 프리셋 전략을 비교 분석합니다.")
    st.write("") 
    if st.button("전략 비교 시작", key="btn_run_opt"):
        prog_bar = st.progress(0, text="데이터 불러오는 중...")
        p, v, bms = fetch_data_serial(ALL_STOCKS, s_d, e_d)
        prog_bar.progress(0.2, text="데이터 준비 완료. 시뮬레이션 시작...")
        
        if not p.empty:
            results = []
            main_bm = bms.get('KOSPI')
            if main_bm is None: main_bm = pd.Series(1.0, index=p.index)
            
            total_presets = len(PRESETS)
            for i, (name, w) in enumerate(PRESETS.items()):
                if name == "사용자 정의": continue
                ws = {'mom': w[0], 'liq': w[1], 'vol': w[2], 'risk': w[3]}
                res = run_backtest(p, v, ws, TICKER_INFO, CONST, benchmark=main_bm)
                if not res.empty:
                    res_c = res.set_index('Sell_Date')
                    mets = calculate_metrics(res_c)
                    mets['전략명'] = name
                    mets['가중치'] = f"{w[0]}|{w[1]}|{w[2]}|{w[3]}"
                    results.append(mets)
                
                current_prog = 0.2 + (0.7 * (i+1) / total_presets)
                prog_bar.progress(current_prog, text=f"분석 중: {name}")
                
            if results:
                prog_bar.progress(0.95, text="결과 표 생성 중...")
                res_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)
                disp_cols = ['전략명', 'CAGR', 'MDD', 'Sharpe', 'Sortino', 'Win_Rate', 'Alpha', 'Beta']
                st.dataframe(res_df[disp_cols].style.format({
                    "CAGR": "{:.1%}", "MDD": "{:.1%}", "Win_Rate": "{:.1%}", "Alpha": "{:.1%}",
                    "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Beta": "{:.2f}"
                }))
                
                best_strat = res_df.iloc[0]['전략명']
                st.success(f"🏆 추천 전략: {best_strat}")
                
                prog_bar.progress(1.0, text="완료!")
                time.sleep(0.3)
                prog_bar.empty()
            else:
                st.warning("결과 없음")
        else:
            st.error("데이터 수집 실패")
