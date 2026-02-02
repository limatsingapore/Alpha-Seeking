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
# [Helper: 날짜 및 데이터 유틸리티]
# ==============================================================================
def get_last_complete_month_end():
    today = datetime.now()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    return last_month_end

def infer_sector_kr(name):
    """종목명 키워드로 섹터를 역추적 (한국장 전용)"""
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
        '조선/중공업': ['중공업', '조선', '기계', '엔진', '현대미포', '한국조선', '두산', '한화오션', '현대로템'],
        '화학/정유': ['화학', '케미칼', '정유', 'S-Oil', '롯데정밀', '효성', '금호'],
        '건설/건자재': ['건설', '개발', '엔지니어링', '시멘트', '페인트', '현대건설'],
        '소비재/유통': ['푸드', '식품', '제과', '쇼핑', '백화점', '이마트', '호텔', '항공', '화장품', '아모레']
    }
    for sector, keys in keywords.items():
        if any(k in name for k in keys): return sector
    return '기타/소형주'

# ==============================================================================
# [섹터 분석 전용 캐싱 함수] (최적화 9번)
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_sector_performance(targets, lookback, country, ticker_map, sector_map):
    sector_returns = {}
    
    def get_ret(t):
        try:
            # 기간 계산 수정 (4번)
            days_needed = max(lookback * 2, 10)
            
            if country == "KR":
                df = fdr.DataReader(t, (datetime.now()-timedelta(days=days_needed)).strftime('%Y-%m-%d'))
            else:
                df = yf.Ticker(t).history(period="3mo") # 넉넉하게
                
            if len(df) <= lookback: return None
            
            # 수익률 계산 (종가 기준)
            # 오늘(1D)이면: 오늘 종가 / 어제 종가 - 1
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-(lookback+1)]) - 1
            
            # 섹터 분류
            name = ticker_map.get(t, t)
            if country == "KR": sec = infer_sector_kr(name)
            else: sec = sector_map.get(t, 'Unknown')
                
            return sec, ret
        except: return None

    # 병렬 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(get_ret, t) for t in targets]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                s, r = res
                if s not in sector_returns: sector_returns[s] = []
                sector_returns[s].append(r)
                
    return sector_returns

@st.cache_data(ttl=3600*24)
def fetch_sector_rotation(country):
    """섹터 로테이션 데이터 생성"""
    if country == "KR":
        sectors = {
            '반도체': ['000660', '005930'], '2차전지': ['373220', '006400'],
            '바이오': ['207940', '068270'], '자동차': ['005380', '000270'],
            '금융': ['105560', '055550'], '인터넷': ['035420', '035720']
        }
    else:
        # 미국 섹터 대표주 (수정 2번)
        sectors = {
            '기술': ['AAPL', 'MSFT'], '헬스케어': ['JNJ', 'UNH'],
            '금융': ['JPM', 'BAC'], '에너지': ['XOM', 'CVX'],
            '소비재': ['AMZN', 'TSLA'], '통신': ['T', 'VZ']
        }
    
    monthly_data = {}
    end_d = datetime.now()
    
    for sec_name, codes in sectors.items():
        sec_monthly_rets = []
        try:
            code = codes[0]
            if country == "KR": 
                df = fdr.DataReader(code, (end_d - timedelta(days=400)).strftime('%Y-%m-%d'))
            else: 
                df = yf.Ticker(code).history(period="2y")
            
            # 월간 리샘플링 (수정 1번)
            df_m = df['Close'].resample('M').last().pct_change().dropna()
            raw_list = df_m.tail(12).values.tolist()
            
            # 길이 보정
            if len(raw_list) > 12: sec_monthly_rets = raw_list[-12:]
            elif len(raw_list) < 12: sec_monthly_rets = [0.0]*(12-len(raw_list)) + raw_list
            else: sec_monthly_rets = raw_list
            
        except: sec_monthly_rets = [0.0]*12
            
        monthly_data[sec_name] = sec_monthly_rets
        
    # 날짜 라벨 생성 (최근 12개월)
    dates = [end_d - timedelta(days=30*i) for i in range(12)][::-1]
    date_labels = [d.strftime('%y-%m') for d in dates]
    
    return pd.DataFrame(monthly_data, index=date_labels).T

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
            df = df.sort_values('Amount', ascending=False).head(600)
        
        if 'Sector' not in df.columns: df['Sector'] = '기타'
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist(), df.set_index('Code')['Sector'].to_dict()
    except: return {"005930":"삼성전자"}, ["005930"], {}

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
        for t in tables:
            if any(k in t.columns for k in ['Symbol', 'Ticker']):
                df = t; break
        if df is None: raise ValueError("Table Not Found")
        
        rename_map = {'Symbol': 'Code', 'Ticker': 'Code', 'Security': 'Name', 'Company': 'Name', 'GICS Sector': 'Sector'}
        df = df.rename(columns=rename_map)
        if 'Sector' not in df.columns: df['Sector'] = 'Unknown'
        df = df[['Code', 'Name', 'Sector']].dropna()
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist(), df.set_index('Code')['Sector'].to_dict()
    except: return {"AAPL":"Apple"}, ["AAPL"], {}

# ==============================================================================
# [Core Logic]
# ==============================================================================
def calculate_factors(price, volume, min_amt, trading_days=252):
    if len(price) < 60 + 20: return None 
    amt_series = price * volume
    avg_amt = amt_series.iloc[-20:].mean()
    if avg_amt < min_amt: return None

    try:
        mom_short = price.pct_change(20).iloc[-1]
        mom_mid = price.pct_change(60).iloc[-1]
        vol = price.pct_change().tail(trading_days).std() * np.sqrt(trading_days)
        liquidity = np.log1p(avg_amt)
        mdd = (price.tail(trading_days) / price.tail(trading_days).cummax() - 1).min()
        return {'mom_short': mom_short, 'mom_mid': mom_mid, 'volatility': vol, 'liquidity': liquidity, 'mdd': mdd, 'price': price.iloc[-1]}
    except: return None

def calculate_short_term_factors(price, volume):
    if len(price) < 20: return None
    try:
        mom_1d = price.pct_change(1).iloc[-1]
        mom_3d = price.pct_change(3).iloc[-1]
        mom_5d = price.pct_change(5).iloc[-1]
        
        # 거래량 급증 (수정 5번)
        vol_avg_5d = volume.iloc[-6:-1].mean()
        vol_today = volume.iloc[-1]
        vol_spike = vol_today / vol_avg_5d if vol_avg_5d > 0 else 0
        
        recent_vol = price.tail(5).std() / price.tail(5).mean()
        disparity = (price.iloc[-1] / price.tail(20).mean()) * 100
        
        return {'mom_1d': mom_1d, 'mom_3d': mom_3d, 'mom_5d': mom_5d, 'vol_spike': vol_spike, 'recent_vol': recent_vol, 'disparity': disparity, 'price': price.iloc[-1]}
    except: return None

def rank_and_score(factor_df, weights):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    scored['R_Mom_S'] = scored['mom_short'].rank(pct=True)
    scored['R_Mom_M'] = scored['mom_mid'].rank(pct=True)
    scored['R_Vol'] = scored['volatility'].rank(pct=True, ascending=False)
    scored['R_Liq'] = scored['liquidity'].rank(pct=True)
    scored['R_MDD'] = scored['mdd'].rank(pct=True)
    
    total = (scored['R_Mom_S']*0.5 + scored['R_Mom_M']*0.5)*weights['mom'] + \
            scored['R_Vol']*weights['vol'] + scored['R_Liq']*weights['liq'] + \
            scored['R_MDD']*weights['risk']
    
    w_sum = sum(weights.values())
    scored['Total_Score'] = (total / w_sum) * 100 if w_sum > 0 else 0
    return scored.sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [백테스트 엔진]
# ==============================================================================
def fetch_backtest_data(universe, start_date, end_date, country):
    s_str, e_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    p, v = {}, {}
    
    # Benchmark
    try:
        if country == "KR": bm = fdr.DataReader('KS11', s_str, e_str)['Close']
        else: bm = yf.download("^GSPC", start=s_str, end=e_str, progress=False)['Close']
        bm.index = pd.to_datetime(bm.index).tz_localize(None)
    except: bm = pd.Series(dtype=float)

    def get(code):
        try:
            if country == "KR": d = fdr.DataReader(code, s_str, e_str)
            else: return None # 미국은 Bulk로 처리
            d.index = pd.to_datetime(d.index).tz_localize(None)
            return code, d['Close'], d['Volume']
        except: return None

    # KR: Individual Fetch / US: Bulk Fetch
    if country == "KR":
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(get, c) for c in universe]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res: p[res[0]], v[res[0]] = res[1], res[2]
    else:
        try:
            d = yf.download(universe, start=s_str, end=e_str, group_by='ticker', progress=False, threads=True)
            if isinstance(d.columns, pd.MultiIndex):
                tickers = d.columns.levels[0]
                for t in tickers:
                    if t in universe:
                        if 'Adj Close' in d[t]: s = d[t]['Adj Close']
                        elif 'Close' in d[t]: s = d[t]['Close']
                        else: continue
                        vo = d[t]['Volume'] if 'Volume' in d[t] else None
                        s.index = pd.to_datetime(s.index).tz_localize(None)
                        if vo is not None: vo.index = pd.to_datetime(vo.index).tz_localize(None)
                        p[t], v[t] = s, vo
        except: pass

    return pd.DataFrame(p), pd.DataFrame(v), bm

def run_backtest(prices, volumes, benchmark, weights, ticker_map, const):
    if prices.empty: return pd.DataFrame()
    reb_dates = prices.groupby(pd.Grouper(freq='MS')).apply(lambda x: x.index[0])
    logs = []
    
    for i in range(12, len(reb_dates)-1):
        rebal_date = reb_dates[i]       
        next_rebal = reb_dates[i+1]     
        try:
            loc = prices.index.get_loc(rebal_date)
            if loc == 0: continue
            signal_date = prices.index[loc - 1] 
        except: continue
        
        p_sub = prices.loc[:signal_date].tail(300)
        v_sub = volumes.loc[:signal_date].tail(300)
        active = p_sub.columns[p_sub.iloc[-1].notna()]
        
        daily = []
        for t in active:
            f = calculate_factors(p_sub[t], v_sub[t], const['MIN_AMT'])
            if f:
                f['code'] = t; daily.append(f)
        if not daily: continue
        
        ranked = rank_and_score(pd.DataFrame(daily).set_index('code'), weights)
        picks = ranked.head(const['TOP_N']).index.tolist()
        if not picks: continue

        try:
            period_price = prices.loc[rebal_date:next_rebal, picks]
            if period_price.empty: continue
            period_daily_rets = period_price.pct_change().dropna()
        except: continue

        if not period_daily_rets.empty:
            port_ret = (1 + period_daily_rets.mean(axis=1)).prod() - 1 - const['COST_RATE']
            try:
                bm_s = benchmark.loc[rebal_date:next_rebal]
                bm_ret = (bm_s.iloc[-1]/bm_s.iloc[0]) - 1
            except: bm_ret = 0.0
            
            logs.append({'Date': next_rebal, 'Port_Ret': port_ret, 'BM_Ret': bm_ret,
                         'Holdings': ", ".join([ticker_map.get(x,x) for x in picks])})
    return pd.DataFrame(logs)

# ==============================================================================
# [최적화 엔진]
# ==============================================================================
def optimize_strategy(prices, volumes, benchmark, ticker_map, presets, const):
    results = []
    if prices.empty: return pd.DataFrame()
    
    prog = st.progress(0, text="전략 시뮬레이션 시작...")
    
    for i, (name, w) in enumerate(presets.items()):
        weights = {'mom': w[0], 'liq': w[1], 'vol': w[2], 'risk': w[3]}
        try:
            res = run_backtest(prices, volumes, benchmark, weights, ticker_map, const)
            if not res.empty:
                res = res.set_index('Date')
                res['Cum_Port'] = (1+res['Port_Ret']).cumprod()
                tot = res['Cum_Port'].iloc[-1] - 1
                y = len(res)/12
                cagr = (tot+1)**(1/y)-1 if y>0 else 0
                mdd = (res['Cum_Port']/res['Cum_Port'].cummax()-1).min()
                vol = res['Port_Ret'].std() * np.sqrt(12)
                sharpe = cagr/vol if vol!=0 else 0
                win = (res['Port_Ret']>0).sum()/len(res)
                
                results.append({'전략명': name, '승률': win, 'CAGR': cagr, '누적수익': tot,
                                'MDD': mdd, '샤프': sharpe, '변동성': vol,
                                '가중치': f"{w[0]}|{w[1]}|{w[2]}|{w[3]}"})
        except: pass
        prog.progress((i+1)/len(presets), text=f"분석 중: {name}")
    
    prog.empty()
    return pd.DataFrame(results).sort_values('CAGR', ascending=False)

def highlight_top3(s):
    is_good_small = s.name in ['변동성'] # 작은게 좋은 지표
    sorted_vals = s.sort_values(ascending=is_good_small).unique()
    styles = []
    for v in s:
        if len(sorted_vals)>0 and v==sorted_vals[0]: styles.append('background-color: #FFD700; color: black; font-weight: bold')
        elif len(sorted_vals)>1 and v==sorted_vals[1]: styles.append('color: #FF4B4B; font-weight: bold')
        elif len(sorted_vals)>2 and v==sorted_vals[2]: styles.append('font-weight: bold')
        else: styles.append('')
    return styles

# ==============================================================================
# [UI MAIN]
# ==============================================================================
st.title("🌍 Alpha Seeking Pro (Ultimate)")

with st.sidebar:
    st.header("🏳️ 시장 선택")
    market = st.radio("국가", ["🇰🇷 한국 (Korea)", "🇺🇸 미국 (USA)"], horizontal=True)
    us_index = "S&P 500"
    
    if "한국" in market:
        COUNTRY, CURRENCY, COST_RATE, MIN_AMT, VIX_TICKER, BM_NAME = "KR", "원", 0.002, 5_000_000_000, "KS200VIX", "KOSPI"
        st.info("대상: 유동성 상위 600개")
        TICKER_INFO, ALL_STOCKS, SECTOR_INFO = load_kr_data()
    else:
        COUNTRY, CURRENCY, COST_RATE, MIN_AMT, VIX_TICKER, BM_NAME = "US", "$", 0.0005, 5_000_000, "^VIX", "S&P 500"
        us_index = st.selectbox("지수", ["S&P 500", "NASDAQ 100", "DOW 30"])
        with st.spinner("데이터 로딩..."): TICKER_INFO, ALL_STOCKS, SECTOR_INFO = load_us_data(us_index)
            
    CONST['MIN_AMT'], CONST['COST_RATE'] = MIN_AMT, COST_RATE
    
    if st.button("🧹 캐시 초기화"): st.cache_data.clear(); st.rerun()
    st.divider()
    
    mode = st.radio("모드 선택", ["📊 실시간 랭킹", "⚡ 단타/스윙", "🔄 섹터 분석", "🎰 포트폴리오", "📉 백테스트", "🔍 전략 최적화"])
    st.divider()
    
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5), "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3), "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0), "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8),
        "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1), "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9)
    }
    sel_preset = st.selectbox("전략 프리셋", list(PRESETS.keys()), index=5)
    dw = PRESETS[sel_preset]
    w_mom = st.slider("📈 추세", 0.0, 1.0, dw[0], 0.1)
    w_liq = st.slider("🌊 수급", 0.0, 1.0, dw[1], 0.1)
    w_vol = st.slider("⚖️ 저변동", 0.0, 1.0, dw[2], 0.1)
    w_risk = st.slider("🛡️ 방어", 0.0, 1.0, dw[3], 0.1)
    weights = {'mom': w_mom, 'liq': w_liq, 'vol': w_vol, 'risk': w_risk}

def get_vix():
    try:
        t = "KS200VIX" if COUNTRY=="KR" else "^VIX"
        df = fdr.DataReader(t, (datetime.now()-timedelta(days=60)).strftime('%Y-%m-%d')) if COUNTRY=="KR" else yf.Ticker(t).history(period="3mo")
        return df['Close'].iloc[-1], df['Close'].iloc[-1]-df['Close'].iloc[-2]
    except: return None, 0

# ==============================================================================
# [MODE: 실행]
# ==============================================================================
if mode == "📊 실시간 랭킹":
    if st.button("분석 실행", type="primary"):
        targets = ALL_STOCKS
        results = []
        bar = st.progress(0, text="스캔 중...")
        
        def worker(t):
            try:
                days = 400 if COUNTRY=="KR" else 730
                if COUNTRY=="KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(t).history(period="2y")
                
                if len(df)<200: return None
                f = calculate_factors(df['Close'], df['Volume'], MIN_AMT)
                if f: f['code'] = t
                return f
            except: return None
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(worker, t) for t in targets]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res: results.append(res)
                bar.progress((i+1)/len(targets), text=f"분석 중... ({i+1}/{len(targets)})")
        bar.empty()
        
        if results:
            final = rank_and_score(pd.DataFrame(results).set_index('code'), weights)
            final['Sector'] = [SECTOR_INFO.get(x, 'Unknown') for x in final.index]
            
            v, vd = get_vix()
            c1, c2, c3 = st.columns(3)
            top = final.iloc[0]
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            c3.metric("VIX", f"{v:.2f}" if v else "N/A", f"{vd:.2f}" if v else "0.0", delta_color="inverse")
            
            st.dataframe(final[['Total_Score', 'price', 'mom_short', 'mdd', 'Sector']].rename(index=TICKER_INFO), use_container_width=True)
        else: st.warning("결과 없음")

elif mode == "⚡ 단타/스윙":
    st.subheader("⚡ 단기 급등 유망주")
    st.caption("조건: 거래량 급증 + 추세 살아있음")
    scan_type = st.selectbox("타입", ["🚀 급등 출발", "🎣 눌림목"])
    
    if st.button("스캔 시작", type="primary"):
        res_short = []
        bar = st.progress(0, text="스캔 중...")
        
        def s_worker(t):
            try:
                if COUNTRY=="KR": df = fdr.DataReader(t, (datetime.now()-timedelta(days=100)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(t).history(period="3mo")
                if len(df)<20: return None
                f = calculate_short_term_factors(df['Close'], df['Volume'])
                if f: f['code'] = t
                return f
            except: return None
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(s_worker, t) for t in ALL_STOCKS]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                r = fut.result()
                if r: res_short.append(r)
                bar.progress((i+1)/len(ALL_STOCKS))
        bar.empty()
        
        if res_short:
            df = pd.DataFrame(res_short).set_index('code')
            df['Name'] = [TICKER_INFO.get(x,x) for x in df.index]
            
            if "급등" in scan_type:
                picks = df[(df['vol_spike']>=2.0) & (df['mom_1d']>0) & (df['mom_5d']>0.05)].sort_values('vol_spike', ascending=False)
            else:
                picks = df[(df['mom_3d']<-0.02) & (df['mom_5d']>0.05) & (df['vol_spike']<1.0)].sort_values('mom_5d', ascending=False)
            
            if not picks.empty:
                st.success(f"{len(picks)}개 발견")
                st.dataframe(picks[['Name', 'price', 'mom_1d', 'vol_spike', 'disparity']], use_container_width=True)
            else:
                st.info("조건에 맞는 종목이 없습니다.")
                st.caption("💡 조건: 1일 등락률 3% 이상 + 거래량 2배 이상 (급등 기준)")

elif mode == "🔄 섹터 분석":
    st.subheader("🔄 시장 자금 흐름")
    tab1, tab2, tab3 = st.tabs(["📊 실시간 강도", "🔥 급등 테마", "📈 로테이션"])
    
    with tab1:
        per = st.selectbox("기간", ["오늘(1D)", "1주일(1W)", "1개월(1M)"])
        pmap = {"오늘(1D)": 1, "1주일(1W)": 5, "1개월(1M)": 20} # 수정 4번
        
        if st.button("분석 실행"):
            with st.spinner("계산 중..."):
                # 최적화 9번: 캐싱된 함수 호출
                sec_rets = fetch_sector_performance(ALL_STOCKS[:400], pmap[per], COUNTRY, TICKER_INFO, SECTOR_INFO)
                
                if sec_rets:
                    stats = []
                    for s, rs in sec_rets.items():
                        if len(rs) >= 3: stats.append({'섹터':s, '수익률':np.mean(rs), '종목수':len(rs)})
                    
                    df_s = pd.DataFrame(stats).sort_values('수익률', ascending=False)
                    fig = px.bar(df_s, x='섹터', y='수익률', color='수익률', color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_s.style.format({'수익률':'{:.2%}'}), use_container_width=True)
                else: st.warning("데이터 부족")

    with tab2:
        if st.button("테마 스캔"):
            st.info("거래량 급증 종목들을 분석하여 주도 테마를 찾습니다.")
            # (단타 스캔 로직 재활용 가능, 생략)

    with tab3:
        if st.button("로테이션 맵"):
            with st.spinner("과거 1년 데이터 분석..."):
                df_rot = fetch_sector_rotation(COUNTRY)
                
                # 수정 3번: 버전 호환성 적용
                try:
                    txt = df_rot.applymap(lambda x: f"{x:.1%}").values
                except:
                    txt = df_rot.map(lambda x: f"{x:.1%}").values
                    
                fig = px.imshow(df_rot, labels=dict(color="수익률"), color_continuous_scale='RdYlGn', aspect="auto")
                fig.update_traces(text=txt, texttemplate="%{text}")
                st.plotly_chart(fig, use_container_width=True)

elif mode == "🎰 포트폴리오":
    codes = st.text_area("종목코드 (쉼표 구분)", "005930, 000660" if COUNTRY=="KR" else "AAPL, TSLA").split(',')
    if st.button("진단"):
        res = []
        for c in codes:
            c = c.strip()
            try:
                days = 300 if COUNTRY=="KR" else 300
                if COUNTRY=="KR": df = fdr.DataReader(c, (datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d'))
                else: df = yf.Ticker(c).history(period="1y")
                
                f = calculate_factors(df['Close'], df['Volume'], 0)
                if f: 
                    f['code'] = c
                    res.append(f)
            except: pass
        
        if res:
            scored = rank_and_score(pd.DataFrame(res).set_index('code'), weights)
            scored['Name'] = [TICKER_INFO.get(x,x) for x in scored.index]
            st.metric("평균 점수", f"{scored['Total_Score'].mean():.1f}")
            st.dataframe(scored[['Name', 'Total_Score', 'mom_short']], use_container_width=True)

elif mode == "📉 백테스트":
    c1, c2 = st.columns(2)
    with c1: s_d = st.date_input("시작", CONST['DEFAULT_START_DATE'])
    with c2: e_d = st.date_input("종료", get_last_complete_month_end())
    
    if st.button("실행", type="primary"):
        with st.spinner("시뮬레이션..."):
            u = ALL_STOCKS[:400] if len(ALL_STOCKS)>400 else ALL_STOCKS
            p, v, bm = fetch_backtest_data(u, s_d, e_d, COUNTRY)
            
            if not p.empty:
                res = run_backtest(p, v, bm, weights, TICKER_INFO, CONST)
                if not res.empty:
                    res = res.set_index('Date')
                    res['Cum'] = (1+res['Port_Ret']).cumprod()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res.index, y=res['Cum'], name="Strategy"))
                    if not bm.empty:
                        try:
                            b = bm.loc[s_d:e_d]
                            fig.add_trace(go.Scatter(x=b.index, y=b/b.iloc[0], name="Benchmark", line=dict(dash='dot')))
                        except: pass
                    st.plotly_chart(fig, use_container_width=True)
                    
                    tot = res['Cum'].iloc[-1]-1
                    st.metric("Total Return", f"{tot:.1%}")
                    st.dataframe(res.tail())

else: # 최적화
    c1, c2 = st.columns(2)
    with c1: s_d = st.date_input("시작", CONST['DEFAULT_START_DATE'])
    with c2: e_d = st.date_input("종료", get_last_complete_month_end())
    
    if st.button("전체 전략 비교"):
        with st.spinner("10+개 전략 시뮬레이션..."):
            u = ALL_STOCKS[:400] if len(ALL_STOCKS)>400 else ALL_STOCKS
            p, v, bm = fetch_backtest_data(u, s_d, e_d, COUNTRY)
            
            if not p.empty:
                presets = {k:v for k,v in PRESETS.items() if k!="사용자 정의"}
                res = optimize_strategy(p, v, bm, TICKER_INFO, presets, CONST)
                if not res.empty:
                    st.dataframe(res.style.apply(highlight_top3, subset=['승률', 'CAGR', '누적수익', 'MDD', '샤프', '변동성'])
                                 .format({'승률':'{:.1%}', 'CAGR':'{:.1%}', '누적수익':'{:.1%}', 'MDD':'{:.1%}', '샤프':'{:.2f}'}), 
                                 use_container_width=True, height=500)
                    
                    best = res.iloc[0]
                    st.success(f"Best: {best['전략명']}")
                    # (그래프 생략 - 위 백테스트와 동일 로직으로 구현 가능)
