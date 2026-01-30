import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr
import time
from typing import Tuple, Union

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링] ---
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

# ==============================================================================
# [설정]
# ==============================================================================
CONST = {
    'TRADING_DAYS': 252,       
    'MOM_SHORT': 20,           
    'MOM_MID': 60,             
    'VOL_WINDOW': 252,         
    'TURNOVER_WINDOW': 60,     
    'COST_RATE': 0.002,        
    'TOP_N': 20,               
    'Z_CLIP': 3.0,             
    'BACKTEST_YEARS': 10,      
    'WARMUP_DAYS': 400         
}

# ==============================================================================
# [핵심 로직: 팩터 계산]
# ==============================================================================

def calculate_raw_factors_vectorized(price_series, volume_series):
    # 1. Momentum
    ret_short = price_series.pct_change(CONST['MOM_SHORT'])
    ret_mid = price_series.pct_change(CONST['MOM_MID'])
    
    # 2. Volatility
    volatility = price_series.pct_change().rolling(CONST['VOL_WINDOW']).std() * np.sqrt(CONST['TRADING_DAYS'])
    
    # 3. Liquidity
    turnover = price_series * volume_series
    to_mean = turnover.rolling(CONST['TURNOVER_WINDOW']).mean()
    to_std = turnover.rolling(CONST['TURNOVER_WINDOW']).std()
    turnover_z = (turnover - to_mean) / (to_std + 1e-9)
    
    # 4. Risk
    roll_max = price_series.rolling(CONST['VOL_WINDOW'], min_periods=1).max()
    drawdown = (price_series / roll_max) - 1.0
    mdd = drawdown.rolling(CONST['VOL_WINDOW'], min_periods=1).min()
    
    return ret_short, ret_mid, turnover_z, volatility, mdd

def normalize_and_score(factor_df, weights):
    if factor_df.empty: return factor_df
    
    directions = {
        'ret_short': 1, 'ret_mid': 1, 
        'turnover_z': 1, 
        'volatility': -1, 
        'max_drawdown': 1 
    }
    
    z_df = pd.DataFrame(index=factor_df.index)
    
    for col, direction in directions.items():
        if col not in factor_df.columns: continue
        
        mu = factor_df[col].mean()
        sigma = factor_df[col].std()
        
        z_val = (factor_df[col] - mu) / (sigma + 1e-9)
        z_df[f'z_{col}'] = z_val.clip(-CONST['Z_CLIP'], CONST['Z_CLIP']) * direction

    final_score = (
        z_df.get('z_ret_short', 0) * weights['mom'] * 0.5 +
        z_df.get('z_ret_mid', 0) * weights['mom'] * 0.5 +
        z_df.get('z_turnover_z', 0) * weights['liq'] +
        z_df.get('z_volatility', 0) * weights['vol'] +
        z_df.get('z_max_drawdown', 0) * weights['risk']
    )
    
    min_s, max_s = final_score.min(), final_score.max()
    if max_s - min_s == 0:
        factor_df['Total_Score'] = 50
    else:
        factor_df['Total_Score'] = ((final_score - min_s) / (max_s - min_s)) * 100
    
    return pd.concat([factor_df, z_df], axis=1).sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [데이터 로딩]
# ==============================================================================
@st.cache_data(ttl=3600*12)
def load_market_data():
    try:
        df_krx = fdr.StockListing('KRX')
        df_etf = fdr.StockListing('ETF/KR')
        
        # --- [KeyError 방어 로직] ---
        if 'Code' not in df_krx.columns and 'Symbol' in df_krx.columns:
            df_krx.rename(columns={'Symbol': 'Code'}, inplace=True)
        
        # 시가총액 컬럼 확인
        sort_col = 'Marcap' if 'Marcap' in df_krx.columns else 'Close'
        amt_col = 'Amount' if 'Amount' in df_krx.columns else 'Volume'
        
        # 데이터 정렬 및 추출
        top_cap = df_krx.sort_values(by=sort_col, ascending=False).head(500)
        top_vol = df_krx.sort_values(by=amt_col, ascending=False).head(300)
        df_stocks = pd.concat([top_cap, top_vol]).drop_duplicates(subset=['Code'])
        
        # ETF 처리
        if 'Symbol' not in df_etf.columns and 'Code' in df_etf.columns:
            df_etf.rename(columns={'Code': 'Symbol'}, inplace=True)
        etf_sort = 'Marcap' if 'Marcap' in df_etf.columns else 'Amount'
        if etf_sort not in df_etf.columns: df_etf[etf_sort] = 0
        df_etf_top = df_etf.sort_values(by=etf_sort, ascending=False).head(50)
        
        # 섹터 분류
        sectors = {
            "📊 ETF Top 50": [],
            "🚀 반도체/하드웨어": [], "💻 SW/플랫폼/게임": [],
            "🔋 2차전지/에너지": [], "🧪 화학/정유/소재": [],
            "💊 바이오/헬스케어": [], "🏥 의료기기/서비스": [],
            "💰 금융(은행/지주)": [], "📈 증권/보험": [],
            "🚗 자동차/모빌리티": [], "🚢 조선/해운/운송": [],
            "🛡️ 방산/우주항공": [], "⚡ 전력/전선/원전": [],
            "🏗️ 건설/인프라/기계": [], "🛍️ 유통/상사/지주": [],
            "💄 화장품/패션/의류": [], "🍜 식음료(F&B)": [],
            "🎬 미디어/엔터/광고": [], "📡 통신/네트워크": [],
            "🌈 기타 대형주": []
        }
        
        ticker_info = {}

        def categorize(name, sec):
            txt = (str(name) + " " + str(sec)).lower()
            if any(x in txt for x in ['etf', 'kodex', 'tiger']): return "📊 ETF Top 50"
            if any(x in txt for x in ['반도체', 'sk하이닉스', '삼성전자', 'hpsp', '이수페타', 'pcb', '디스플레이', 'lg이노텍']): return "🚀 반도체/하드웨어"
            if any(x in txt for x in ['게임', '소프트웨어', '네이버', '카카오', '크래프톤', '보안', '클라우드', 'ai']): return "💻 SW/플랫폼/게임"
            if any(x in txt for x in ['에코프로', '엘앤에프', '포스코퓨처', '전지', '머티리얼', '양극재', '금양', '캠']): return "🔋 2차전지/에너지"
            if any(x in txt for x in ['화학', '정유', 'oil', 'sk이노', '효성', '롯데케미', '금호', '제철', '고려아연']): return "🧪 화학/정유/소재"
            if any(x in txt for x in ['삼성바이오', '셀트리온', '알테오젠', 'hlb', '유한양행', '한미약품', '리가켐', 'sk바이오']): return "💊 바이오/헬스케어"
            if any(x in txt for x in ['클래시스', '덴티움', '미용', '의료', '휴젤', '파마리서치']): return "🏥 의료기기/서비스"
            if any(x in txt for x in ['금융', '은행', '지주', 'kb', '신한', '하나', '우리']): return "💰 금융(은행/지주)"
            if any(x in txt for x in ['증권', '보험', '삼성생명', '화재', '메리츠', '키움']): return "📈 증권/보험"
            if any(x in txt for x in ['자동차', '현대차', '기아', '모비스', '타이어', '한온']): return "🚗 자동차/모빌리티"
            if any(x in txt for x in ['조선', '해운', 'hmm', '오션', '중공업', '팬오션', '글로비스', '대한항공']): return "🚢 조선/해운/운송"
            if any(x in txt for x in ['방산', '한화에어', 'lig', '한국항공', '현대로템', '풍산']): return "🛡️ 방산/우주항공"
            if any(x in txt for x in ['전력', '전선', '일렉', '변압기', 'ls', '효성중공업', '두산에너', '한전']): return "⚡ 전력/전선/원전"
            if any(x in txt for x in ['건설', '기계', '두산밥캣', '현대건설', 'gs건설', '엔지니어링']): return "🏗️ 건설/인프라/기계"
            if any(x in txt for x in ['유통', '백화점', '상사', '포스코인터', '물산', '지주', '이마트', '편의점']): return "🛍️ 유통/상사/지주"
            if any(x in txt for x in ['화장품', '아모레', '코스맥스', '영원무역', 'f&f', '패션', '의류']): return "💄 화장품/패션/의류"
            if any(x in txt for x in ['식품', '음료', '농심', '삼양', '오리온', 'cj제일', '롯데칠성']): return "🍜 식음료(F&B)"
            if any(x in txt for x in ['엔터', '하이브', 'jyp', '에스엠', '스튜디오', '광고', '제일기획', '미디어']): return "🎬 미디어/엔터/광고"
            if any(x in txt for x in ['통신', '텔레콤', 'kt', 'lg유플러스', '네트워크']): return "📡 통신/네트워크"
            return "🌈 기타 대형주"

        # 주식 처리
        for _, row in df_stocks.iterrows():
            code = str(row['Code'])
            name = str(row['Name'])
            sec_raw = str(row['Sector']) if 'Sector' in row and pd.notnull(row['Sector']) else ""
            
            cat = categorize(name, sec_raw)
            if cat in sectors: sectors[cat].append(code)
            else: sectors["🌈 기타 대형주"].append(code)
            ticker_info[code] = {'Name': name}
            
        # ETF 처리
        for _, row in df_etf_top.iterrows():
            code = str(row['Symbol'])
            name = str(row['Name'])
            sectors["📊 ETF Top 50"].append(code)
            ticker_info[code] = {'Name': name}
            
        all_tickers = df_stocks['Code'].tolist()
        return sectors, ticker_info, all_tickers

    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return {}, {}, []

with st.spinner("시장 데이터 로딩 중..."):
    SECTORS, TICKER_INFO, ALL_STOCKS_LIST = load_market_data()

# --- [VIX 조회] ---
@st.cache_data(ttl=600)
def get_vix():
    try:
        df = fdr.DataReader('KS200VIX', datetime.now() - timedelta(days=7))
        if not df.empty: return df['Close'].iloc[-1]
        return None
    except: return None

# ==============================================================================
# [실시간 분석]
# ==============================================================================
def compute_current_factors_safe(ticker):
    try:
        start_date = (datetime.now() - timedelta(days=CONST['WARMUP_DAYS'] + 60)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start_date)
        
        if len(df) < CONST['TRADING_DAYS'] * 0.8: return None 

        ret_s, ret_m, tz, vol, mdd = calculate_raw_factors_vectorized(df['Close'], df['Volume'])
        
        return {
            "code": ticker,
            "current_price": df['Close'].iloc[-1],
            "ret_short": ret_s.iloc[-1],
            "ret_mid": ret_m.iloc[-1],
            "turnover_z": tz.iloc[-1],
            "volatility": vol.iloc[-1],
            "max_drawdown": mdd.iloc[-1],
            "chart_data": df['Close'].tail(60)
        }
    except Exception as e:
        return None

# ==============================================================================
# [백테스트 엔진] (버그 수정됨)
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_rolling_backtest_data(universe, start_str, end_str):
    try:
        kospi = fdr.DataReader('KS11', start_str, end_str)['Close']
    except: return None, None, None
    
    def worker(t):
        try:
            d = fdr.DataReader(t, start_str, end_str)
            return t, d['Close'], d['Volume']
        except: return None, None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        res = list(ex.map(worker, universe))
    
    c_dict, v_dict = {}, {}
    for t, c, v in res:
        if t: c_dict[t], v_dict[t] = c, v
            
    return pd.DataFrame(c_dict).ffill(), pd.DataFrame(v_dict).ffill(), kospi

def run_rolling_backtest(prices, volumes, benchmark, weights):
    reb_dates = prices.resample('M').last().index
    start_idx = 0
    for i, d in enumerate(reb_dates):
        if (d - prices.index[0]).days > CONST['WARMUP_DAYS']:
            start_idx = i
            break
    if start_idx >= len(reb_dates) - 1: return pd.DataFrame()

    logs = []
    prev_holdings = set()
    
    for i in range(start_idx, len(reb_dates)-1):
        curr = reb_dates[i]
        next_d = reb_dates[i+1] # [수정] next_date 변수명 통일 (오타 수정)
        
        p_slice = prices.loc[:curr].tail(int(CONST['WARMUP_DAYS'] * 1.2))
        v_slice = volumes.loc[:curr].tail(int(CONST['WARMUP_DAYS'] * 1.2))
        
        if len(p_slice) < CONST['TRADING_DAYS']: continue
        
        rs, rm, tz, vol, mdd = calculate_raw_factors_vectorized(p_slice, v_slice)
        
        factors_t = pd.DataFrame({
            'ret_short': rs.iloc[-1], 'ret_mid': rm.iloc[-1],
            'turnover_z': tz.iloc[-1], 'volatility': vol.iloc[-1],
            'max_drawdown': mdd.iloc[-1]
        }).dropna()
        
        if factors_t.empty: continue
        
        scored = normalize_and_score(factors_t, weights)
        top_picks = scored.nlargest(CONST['TOP_N'], 'Total_Score').index.tolist()
        current_holdings = set(top_picks)
        
        # [수정] next_date -> next_d
        fwd_ret = prices.loc[curr:next_d, top_picks].pct_change().dropna()
        port_ret = (1 + fwd_ret).prod() - 1
        gross = port_ret.mean()
        
        if not prev_holdings: turnover_rate = 1.0
        else:
            kept_count = len(prev_holdings.intersection(current_holdings))
            turnover_rate = (CONST['TOP_N'] - kept_count) / CONST['TOP_N']
            
        cost = turnover_rate * CONST['COST_RATE']
        net = gross - cost
        
        bm_period = benchmark.loc[curr:next_d].pct_change().dropna()
        bm_ret = (1 + bm_period).prod() - 1
        
        logs.append({
            'Date': next_d, 'Gross': gross, 'Net': net, 'Cost': cost,
            'BM': bm_ret, 'Alpha': net - bm_ret, 'Turnover': turnover_rate
        })
        prev_holdings = current_holdings
        
    return pd.DataFrame(logs)

# ==============================================================================
# [UI 구성]
# ==============================================================================
st.title("🧬 Alpha Seeking Pro (Master)")

with st.sidebar:
    st.header("⚙️ 전략 설정")
    w_mom = st.slider("📈 모멘텀 (추세)", 0.0, 1.0, 0.4, 0.1, help="최근 주가가 상승세인 종목을 선호합니다.")
    w_liq = st.slider("🌊 수급 (거래강도)", 0.0, 1.0, 0.2, 0.1, help="평소보다 거래가 폭발하는 종목을 찾습니다.")
    w_vol = st.slider("⚖️ 저변동성 (안정)", 0.0, 1.0, 0.2, 0.1, help="주가 등락폭이 적은 안정적인 종목을 선호합니다.")
    w_risk = st.slider("🛡️ 방어력 (MDD)", 0.0, 1.0, 0.2, 0.1, help="과거 폭락장에서 잘 버틴 종목을 찾습니다.")
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}
    
    st.divider()
    mode = st.radio("모드 선택", ["📊 실시간 분석", "📉 과거 백테스트"])

# ------------------------------------------------------------------------------
# TAB 1: 실시간 랭킹
# ------------------------------------------------------------------------------
if mode == "📊 실시간 분석":
    st.subheader("실시간 팩터 스코어링")
    sec = st.selectbox("업종/테마 선택", list(SECTORS.keys()))
    
    if st.button("분석 실행", type="primary"):
        targets = SECTORS[sec]
        if not targets: st.error("해당 섹터에 종목이 없습니다.")
        else:
            data = []
            bar = st.progress(0, text="데이터 수집 중...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(compute_current_factors_safe, t): t for t in targets}
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    res = f.result()
                    if res: data.append(res)
                    bar.progress((i+1)/len(targets))
            bar.empty()
            
            if data:
                f_df = pd.DataFrame(data).set_index('code')
                final = normalize_and_score(f_df, weights)
                
                # Top Pick
                top = final.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("🏆 1위 종목", f"{TICKER_INFO.get(top.name, {}).get('Name', top.name)}")
                c2.metric("⭐ 종합 점수", f"{top['Total_Score']:.1f}점")
                
                vix = get_vix()
                c3.metric("📉 시장 공포지수 (VIX)", f"{vix:.2f}" if vix else "-")
                
                # --- [테이블 구성: 사용자 친화적 용어 & 콤마 적용] ---
                disp = final[['current_price', 'Total_Score', 'ret_short', 'turnover_z', 'max_drawdown']].copy()
                disp.columns = ['Price', 'Score', 'Mom', 'Liq', 'MDD']
                
                # 인덱스를 종목명으로 변경
                disp.index = [TICKER_INFO.get(x,{}).get('Name',x) for x in disp.index]
                
                # 천단위 콤마를 위해 문자열로 변환 (정렬에는 불리하지만 보기 좋음)
                # 정렬을 유지하고 싶다면 column_config를 써야 하는데, 
                # Streamlit의 NumberColumn은 천단위 콤마를 기본 지원하지 않음(locale 의존).
                # 따라서 가장 확실한 방법인 문자열 포맷팅 적용.
                
                st.dataframe(
                    disp, 
                    use_container_width=True, 
                    height=600,
                    column_config={
                        "Price": st.column_config.NumberColumn(
                            "현재가 (원)", 
                            format="%d" # 기본 숫자 포맷 (브라우저 로케일에 따라 콤마 들어갈 수 있음)
                        ),
                        "Score": st.column_config.ProgressColumn(
                            "종합 점수", 
                            format="%.1f점", 
                            min_value=0, 
                            max_value=100
                        ),
                        "Mom": st.column_config.NumberColumn(
                            "📈 상승 추세", 
                            help="최근 1개월 주가 상승 강도 (높을수록 좋음)",
                            format="%.2f (Z)"
                        ),
                        "Liq": st.column_config.NumberColumn(
                            "🌊 수급 강도",
                            help="평소 대비 거래 폭발 정도 (2.0 이상이면 강력)",
                            format="%.2f (Z)"
                        ),
                        "MDD": st.column_config.NumberColumn(
                            "🛡️ 방어력",
                            help="최근 1년 최대 낙폭 (0%에 가까울수록 안전)",
                            format="%.1f%%"
                        )
                    }
                )
            else:
                st.error("데이터 수집 실패. 잠시 후 다시 시도해주세요.")

# ------------------------------------------------------------------------------
# TAB 2: Rolling Backtest
# ------------------------------------------------------------------------------
else:
    today = datetime.today()
    end_s = today.strftime('%Y-%m-%d')
    start_s = (today - timedelta(days=365*(CONST['BACKTEST_YEARS']+1))).strftime('%Y-%m-%d')
    
    st.subheader(f"📉 과거 10년 시뮬레이션 ({start_s} ~ {end_s})")
    st.info(f"💡 과거 데이터만 사용하여 매월 말 종목을 교체했을 때의 성과입니다. (거래비용 {CONST['COST_RATE']*100}% 차감)")
    
    if st.button("🚀 시뮬레이션 시작", type="primary"):
        st.write("데이터 로딩 중... (최대 1~2분 소요)")
        p_df, v_df, bm = fetch_rolling_backtest_data(ALL_STOCKS_LIST[:300], start_s, end_s)
        
        if p_df is not None:
            st.write("연산 수행 중...")
            res = run_rolling_backtest(p_df, v_df, bm, weights)
            
            if not res.empty:
                res['Date'] = pd.to_datetime(res['Date'])
                res = res.set_index('Date')
                
                res['Cum_Port'] = (1 + res['Net']).cumprod()
                res['Cum_BM'] = (1 + res['BM']).cumprod()
                
                tot = res['Cum_Port'].iloc[-1] - 1
                days = (res.index[-1] - res.index[0]).days
                cagr = (1 + tot) ** (365 / days) - 1
                mdd = (res['Cum_Port'] / res['Cum_Port'].cummax() - 1).min()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("연평균 수익률 (CAGR)", f"{cagr:.1%}")
                m2.metric("최대 낙폭 (MDD)", f"{mdd:.1%}")
                m3.metric("총 누적 수익", f"{tot:.1%}")
                m4.metric("평균 종목 교체율", f"{res['Turnover'].mean():.1%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Port'], name='내 전략', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_BM'], name='시장(KOSPI)', line=dict(color='grey', dash='dot')))
                st.plotly_chart(fig, use_container_width=True)
                
                # 결과 테이블 한글화 및 포맷팅
                st.dataframe(
                    res,
                    use_container_width=True,
                    column_config={
                        "Gross": st.column_config.NumberColumn("수익률(비용전)", format="%.2%"),
                        "Net": st.column_config.NumberColumn("수익률(비용후)", format="%.2%"),
                        "Cost": st.column_config.NumberColumn("비용", format="%.4f"),
                        "BM": st.column_config.NumberColumn("시장수익률", format="%.2%"),
                        "Alpha": st.column_config.NumberColumn("초과수익", format="%.2%"),
                        "Turnover": st.column_config.NumberColumn("교체율", format="%.2%"),
                        "Cum_Port": st.column_config.NumberColumn("누적(전략)", format="%.2f"),
                        "Cum_BM": st.column_config.NumberColumn("누적(시장)", format="%.2f")
                    }
                )
            else:
                st.error("결과 산출 실패 (데이터 부족)")
        else:
            st.error("데이터 로딩 실패")
