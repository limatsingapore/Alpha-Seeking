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
st.set_page_config(page_title="Alpha Seeking Pro (Final v6.1)", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] { background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.1rem !important; }
    div[data-testid="stExpander"] { background-color: #1e293b; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# [설정 & 상수 (국장 전용)]
# ==============================================================================
CONST = {
    'TRADING_DAYS': 252,       
    'MOM_SHORT': 20,           
    'MOM_MID': 60,             
    'VOL_WINDOW': 252,         
    'COST_RATE': 0.002,        
    'TOP_N': 20,               
    'MIN_AMT': 5_000_000_000, 
    'DEFAULT_START_DATE': datetime(2018, 1, 1)
}

# ==============================================================================
# [Helper Functions]
# ==============================================================================
def get_last_complete_month_end():
    return datetime.now() 

def z_score(x):
    if x.std() == 0: return x * 0
    return (x - x.mean()) / x.std()

def infer_sector_kr(name):
    name = str(name)
    if any(x in name for x in ['스팩', '제호', '기업인수']): return '스팩/금융'
    if any(x in name for x in ['우', '우B']): return '우선주'
    
    keywords = {
        '반도체/IT': ['삼성전자', 'SK하이닉스', '반도체', '테크', '칩', '시스템', '전자', '이노텍', 'DB하이텍', '주성'],
        '2차전지': ['에코프로', '엘앤에프', 'LG에너지', '삼성SDI', 'SK이노베이션', '포스코퓨처', '엔켐', '금양'],
        '바이오/제약': ['바이오', '제약', '약품', '생명', '헬스', '셀트리온', '유한양행', '한미', 'HLB', '알테오젠'],
        '자동차/부품': ['현대차', '기아', '모비스', '타이어', '만도', '오토', '화신'],
        '인터넷/게임': ['NAVER', '카카오', '게임', '소프트', '엔씨', '펄어비스', '크래프톤', '넷마블'],
        '엔터/미디어': ['엔터', '스튜디오', '미디어', '에스엠', 'JYP', 'YG', '하이브', 'CJ ENM'],
        '금융/지주': ['금융', '지주', '은행', '증권', '보험', '카드', '투자', '홀딩스', '메리츠', 'KB', '신한'],
        '조선/중공업': ['중공업', '조선', '기계', '엔진', '현대미포', '한국조선', '두산', '한화오션', '현대로템', 'LIG넥원'],
        '화학/정유': ['화학', '케미칼', '정유', 'S-Oil', '롯데정밀', '효성', '금호'],
        '건설/건자재': ['건설', '개발', '엔지니어링', '시멘트', '페인트', '현대건설'],
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
            df = df.sort_values('Amount', ascending=False).head(200)
            
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()
    except:
        return {"005930":"삼성전자"}, ["005930"]

@st.cache_data(ttl=3600*24)
def fetch_data_serial(universe, start_date, end_date):
    """직렬 데이터 수집"""
    real_start_date = start_date - timedelta(days=400)
    s_str = real_start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    p_dict = {}
    v_dict = {}
    bm_dict = {}
    
    # 1. 벤치마크
    try:
        kospi = fdr.DataReader('KS11', s_str, e_str)
        if not kospi.empty: bm_dict['KOSPI'] = kospi['Close']
    except: pass
    
    # 2. 개별 종목
    progress_text = "데이터 수집 중... (0%)"
    my_bar = st.progress(0, text=progress_text)
    total = len(universe)
    
    for i, code in enumerate(universe):
        try:
            d = fdr.DataReader(code, s_str, e_str)
            if len(d) > 60: 
                if 'Close' in d.columns: p_dict[code] = d['Close']
                if 'Volume' in d.columns: v_dict[code] = d['Volume']
        except: pass
        
        if i % 10 == 0:
            my_bar.progress((i + 1) / total, text=f"데이터 수집 중... ({i+1}/{total})")
            
    my_bar.empty()
    
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
# [Core Logic: 팩터 계산]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    if len(price) < 60: return None
    try:
        p = price
        v = volume
        amt = p * v
        if amt.tail(20).mean() < min_amt: return None
        
        mom_short = p.pct_change(20).iloc[-1]
        mom_mid = p.pct_change(60).iloc[-1]
        vol = p.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        liquidity = np.log1p(amt.tail(20).mean())
        mdd = (p.tail(trading_days) / p.tail(trading_days).cummax() - 1).min()
        
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
        scored['sector'] = [infer_sector_kr(ticker_map.get(x, x)) for x in scored.index]
    else:
        scored['sector'] = 'Unknown'

    for col in ['mom_short', 'mom_mid', 'volatility', 'liquidity', 'mdd']:
        if col in scored.columns:
            scored[col] = scored[col].fillna(scored[col].median())

    scored['Z_Mom_S'] = z_score(scored['mom_short'])
    scored['Z_Mom_M'] = z_score(scored['mom_mid'])
    scored['Z_Vol'] = z_score(scored['volatility']) * -1 
    scored['Z_Liq'] = z_score(scored['liquidity'])
    scored['Z_MDD'] = z_score(scored['mdd']) 
    
    scored['Total_Score'] = (
        scored['Z_Mom_S'] * weights['mom'] * 0.5 + 
        scored['Z_Mom_M'] * weights['mom'] * 0.5 +
        scored['Z_Vol'] * weights['vol'] + 
        scored['Z_Liq'] * weights['liq'] +
        scored['Z_MDD'] * weights['risk']
    )
    
    return scored.sort_values(by='Total_Score', ascending=False)

def run_backtest(prices, volumes, weights, ticker_map, const, benchmark=None):
    if prices.empty: return pd.DataFrame()
    
    reb_dates = prices.resample('BM').last().index
    logs = []
    prev_picks = [] 
    
    if benchmark is None or benchmark.empty:
        benchmark = pd.Series(1.0, index=prices.index)
    else:
        benchmark = benchmark.reindex(prices.index).ffill().fillna(method='bfill')

    for i in range(12, len(reb_dates)):
        rebal_date = reb_dates[i]
        
        if i < len(reb_dates) - 1: next_rebal = reb_dates[i+1]
        else: next_rebal = prices.index[-1]
        
        if rebal_date >= next_rebal: break
            
        try:
            p_sub = prices.loc[:rebal_date].tail(300)
            v_sub = volumes.loc[:rebal_date].tail(300)
            
            valid_cols = p_sub.columns[p_sub.iloc[-1].notna()]
            if len(valid_cols) < 5: continue
            
            daily_factors = []
            for t in valid_cols:
                f = calculate_factors(p_sub[t], v_sub[t], const['MIN_AMT'])
                if f:
                    f['code'] = t; daily_factors.append(f)
            
            if not daily_factors: continue
            
            factor_df = pd.DataFrame(daily_factors).set_index('code')
            ranked = rank_and_score(factor_df, weights, ticker_map=ticker_map)
            
            n_picks = min(const['TOP_N'], len(ranked))
            picks = ranked.head(n_picks).index.tolist()
            
            if not picks: continue
            
            # [Table Data Prep]
            current_prices = {}
            for t in picks:
                if t in daily_factors[0]: # Check logic
                     for item in daily_factors:
                         if item['code'] == t:
                             current_prices[t] = item['price']
                             break
            
            buy_idx = prices.index.searchsorted(rebal_date) + 1
            if buy_idx >= len(prices): buy_idx = len(prices) - 1
            buy_date = prices.index[buy_idx]
            
            sell_idx = prices.index.searchsorted(next_rebal) + 1
            if sell_idx >= len(prices): sell_idx = len(prices) - 1
            sell_date = prices.index[sell_idx]
            
            curr_prices = prices.loc[buy_date, picks].fillna(0).replace(0, np.nan).ffill()
            next_prices = prices.loc[sell_date, picks].fillna(0)
            
            ret_vec = (next_prices / curr_prices) - 1
            ret_vec = ret_vec.fillna(0)
            gross_ret = ret_vec.mean()
            
            if not prev_picks: turnover = 1.0
            else:
                denom = len(picks) if len(picks)>0 else 1
                kept = set(prev_picks) & set(picks)
                turnover = (denom - len(kept)) / denom
                
            net_ret = gross_ret - (turnover * const['COST_RATE'])
            
            try:
                b_s = benchmark.asof(buy_date)
                b_e = benchmark.asof(sell_date)
                if isinstance(b_s, pd.Series): b_s = b_s.iloc[0]
                if isinstance(b_e, pd.Series): b_e = b_e.iloc[0]
                bm_ret = (b_e / b_s) - 1 if b_s != 0 else 0.0
            except: bm_ret = 0.0
                
            logs.append({
                'Date': rebal_date,
                'Sell_Date': sell_date, 
                'Gross_Ret': gross_ret, 
                'Net_Ret': net_ret,
                'BM_Ret': bm_ret, 
                'Turnover': turnover,
                'Holdings_List': picks, 
                'Prices_Dict': current_prices, 
                'Port_Ret': net_ret
            })
            prev_picks = picks
            
        except: continue
            
    return pd.DataFrame(logs)

def optimize_strategy(prices, volumes, ticker_map, presets, const):
    results = []
    if prices.empty: return pd.DataFrame()
    prog = st.progress(0, text="시뮬레이션 시작...")
    
    for i, (name, w) in enumerate(presets.items()):
        weights = {'mom': w[0], 'liq': w[1], 'vol': w[2], 'risk': w[3]}
        try:
            res = run_backtest(prices, volumes, weights, ticker_map, const, benchmark=None)
            if not res.empty:
                res_calc = res.set_index('Sell_Date') 
                cum = (1+res_calc['Port_Ret']).cumprod()
                
                tot = cum.iloc[-1]-1
                y = len(res_calc)/12
                cagr = (tot+1)**(1/y)-1 if y>0 else 0
                mdd = (cum/cum.cummax()-1).min()
                win = (res_calc['Port_Ret']>0).sum()/len(res_calc)
                
                results.append({
                    '전략명': name, '승률': win, 'CAGR': cagr, '누적수익': tot, 'MDD': mdd,
                    '가중치': f"{w[0]}|{w[1]}|{w[2]}|{w[3]}"
                })
        except: pass
        prog.progress((i+1)/len(presets), text=f"분석 중: {name}")
    
    prog.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values('CAGR', ascending=False)

# ==============================================================================
# [UI MAIN]
# ==============================================================================
st.title("🇰🇷 Alpha Seeking Pro (Final v6.1)")

with st.sidebar:
    st.info("대상: KOSPI/KOSDAQ 유동성 상위 200개")
    TICKER_INFO, ALL_STOCKS = load_kr_data()
            
    if st.button("🧹 캐시 초기화", key="clear_cache"): st.cache_data.clear(); st.rerun()
    st.divider()
    
    mode = st.radio("모드 선택", ["📉 백테스트", "🔍 전략 최적화"], key="mode_radio")
    st.divider()
    
    # [16개 프리셋 완전 복구]
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
        "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8),
        "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1), 
        "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9),
        "🎯 퀄리티 그로스": (0.6, 0.6, 0.6, 0.6), 
        "🌪️ 변동성 사냥꾼": (0.7, 0.5, 0.0, 0.2),
        "🦅 매파의 눈": (0.3, 0.9, 0.4, 0.7)
    }
    sel_preset = st.selectbox("전략 프리셋", list(PRESETS.keys()), index=5, key="preset_select")
    dw = PRESETS[sel_preset]
    w_mom = st.slider("📈 추세", 0.0, 1.0, dw[0], 0.1, key="slider_mom")
    w_liq = st.slider("🌊 수급", 0.0, 1.0, dw[1], 0.1, key="slider_liq")
    w_vol = st.slider("⚖️ 저변동", 0.0, 1.0, dw[2], 0.1, key="slider_vol")
    w_risk = st.slider("🛡️ 방어", 0.0, 1.0, dw[3], 0.1, key="slider_risk")
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}

if mode == "📉 백테스트":
    c1, c2 = st.columns(2)
    with c1: s_d = st.date_input("시작", CONST['DEFAULT_START_DATE'], key="start_date")
    with c2: e_d = st.date_input("종료", get_last_complete_month_end(), key="end_date")
    
    if st.button("실행", type="primary", key="btn_run_backtest"):
        p, v, bms = fetch_data_serial(ALL_STOCKS, s_d, e_d)
        
        if not p.empty:
            main_bm = bms.get('KOSPI')
            if main_bm is None: main_bm = pd.Series(1.0, index=p.index)
            
            res = run_backtest(p, v, weights, TICKER_INFO, CONST, benchmark=main_bm)
            
            if not res.empty:
                res_chart = res.set_index('Sell_Date')
                res_chart['Cum'] = (1+res_chart['Port_Ret']).cumprod()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_chart.index, y=res_chart['Cum'], name="Strategy", line=dict(width=3, color='blue')))
                
                if 'KOSPI' in bms:
                    try:
                        b = bms['KOSPI'].reindex(res_chart.index, method='ffill')
                        b = b / b.iloc[0]
                        fig.add_trace(go.Scatter(x=b.index, y=b, name='KOSPI', line=dict(dash='dot', color='red')))
                    except: pass
                    
                st.plotly_chart(fig, use_container_width=True)
                
                tot = res_chart['Cum'].iloc[-1]-1
                st.metric("Total Return", f"{tot:.1%}")
                
                st.divider()
                st.subheader("📅 월별 포트폴리오 상세 분석")
                
                date_options = res['Date'].dt.strftime('%Y-%m-%d').tolist()
                sel_date_str = st.selectbox("리밸런싱 날짜 선택", date_options[::-1], index=0)
                
                if sel_date_str:
                    sel_date = pd.to_datetime(sel_date_str)
                    row = res[res['Date'] == sel_date].iloc[0]
                    
                    codes = row['Holdings_List']
                    prices_dict = row['Prices_Dict']
                    
                    detail_data = []
                    for c in codes:
                        name = TICKER_INFO.get(c, c)
                        sector = infer_sector_kr(name)
                        price = prices_dict.get(c, 0)
                        detail_data.append({
                            "종목명": name,
                            "코드": c,
                            "섹터": sector,
                            "기준가(월말)": f"{price:,.0f}원"
                        })
                    
                    df_detail = pd.DataFrame(detail_data)
                    st.dataframe(df_detail, use_container_width=True)
                    st.caption(f"※ 위 종목들은 {sel_date_str} 종가 기준으로 선정되었으며, 실제 매수는 다음 거래일 시초가에 진행됩니다.")

            else:
                st.error("백테스트 결과가 없습니다. (기간 내 데이터 부족)")
        else:
            st.error("데이터 수집 실패")

elif mode == "🔍 전략 최적화":
    st.info("전략별 성과를 비교합니다.")
    if st.button("비교 시작", key="btn_run_opt"):
        p, v, bms = fetch_data_serial(ALL_STOCKS, datetime(2018,1,1), datetime.now())
        
        if not p.empty:
            results = []
            main_bm = bms.get('KOSPI')
            if main_bm is None: main_bm = pd.Series(1.0, index=p.index)
            
            progress_bar = st.progress(0)
            
            for i, (name, w) in enumerate(PRESETS.items()):
                if name == "사용자 정의": continue
                
                ws = {'mom': w[0], 'liq': w[1], 'vol': w[2], 'risk': w[3]}
                res = run_backtest(p, v, ws, TICKER_INFO, CONST, benchmark=main_bm)
                
                if not res.empty:
                    res_c = res.set_index('Sell_Date')
                    cum = (1+res_c['Port_Ret']).cumprod()
                    tot = cum.iloc[-1]-1
                    mdd = (cum/cum.cummax()-1).min()
                    win = (res_c['Port_Ret'] > 0).sum() / len(res_c)
                    
                    results.append({
                        "전략": name, "수익률": tot, "MDD": mdd, "승률": win,
                        "가중치": f"{w[0]}|{w[1]}|{w[2]}|{w[3]}"
                    })
                
                progress_bar.progress((i+1)/len(PRESETS))
                
            if results:
                res_df = pd.DataFrame(results).sort_values("수익률", ascending=False)
                st.dataframe(res_df.style.format({"수익률": "{:.1%}", "MDD": "{:.1%}", "승률": "{:.1%}"}))
                
                best_strat = res_df.iloc[0]['전략']
                st.success(f"🏆 추천 전략: {best_strat}")
                
                best_w_str = res_df.iloc[0]['가중치']
                best_w_list = list(map(float, best_w_str.split('|')))
                best_weights = {'mom': best_w_list[0], 'liq': best_w_list[1], 'vol': best_w_list[2], 'risk': best_w_list[3]}
                
                res_best = run_backtest(p, v, best_weights, TICKER_INFO, CONST, benchmark=main_bm)
                if not res_best.empty:
                    res_best = res_best.set_index('Sell_Date')
                    res_best['Cum'] = (1+res_best['Port_Ret']).cumprod()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res_best.index, y=res_best['Cum'], name=best_strat, line=dict(color='#FFD700', width=2)))
                    if 'KOSPI' in bms:
                        b = bms['KOSPI'].reindex(res_best.index, method='ffill')
                        b = b / b.iloc[0]
                        fig.add_trace(go.Scatter(x=b.index, y=b, name='KOSPI', line=dict(dash='dot', color='red')))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("결과 없음")
        else:
            st.error("데이터 수집 실패")
