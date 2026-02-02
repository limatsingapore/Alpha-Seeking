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

# --- [로그 설정] ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Pure Alpha v4.1)", layout="wide", initial_sidebar_state="expanded")

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
            df = df.sort_values('Amount', ascending=False).head(600)
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
        for t in tables:
            if any(k in t.columns for k in ['Symbol', 'Ticker']):
                df = t; break
        if df is None: raise ValueError("Table Not Found")
        
        rename_map = {'Symbol': 'Code', 'Ticker': 'Code', 'Security': 'Name', 'Company': 'Name'}
        df = df.rename(columns=rename_map)
        df = df[['Code', 'Name']].dropna()
        df['Code'] = df['Code'].astype(str).str.replace('.', '-', regex=False)
        return df.set_index('Code')['Name'].to_dict(), df['Code'].tolist()
    except:
        return {"AAPL":"Apple"}, ["AAPL"]

# ==============================================================================
# [Core Logic 1: 기술적 팩터 계산]
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
        
        return {
            'mom_short': mom_short, 'mom_mid': mom_mid,
            'volatility': vol, 'liquidity': liquidity, 'mdd': mdd,
            'price': price.iloc[-1]
        }
    except: return None

def calculate_short_term_factors(price, volume):
    if len(price) < 20: return None
    try:
        mom_1d = price.pct_change(1).iloc[-1]
        mom_3d = price.pct_change(3).iloc[-1]
        mom_5d = price.pct_change(5).iloc[-1]
        vol_avg_5d = volume.iloc[-6:-1].mean()
        vol_today = volume.iloc[-1]
        vol_spike = vol_today / vol_avg_5d if vol_avg_5d > 0 else 0
        recent_vol = price.tail(5).std() / price.tail(5).mean()
        disparity = (price.iloc[-1] / price.tail(20).mean()) * 100
        
        return {
            'mom_1d': mom_1d, 'mom_3d': mom_3d, 'mom_5d': mom_5d,
            'vol_spike': vol_spike, 'recent_vol': recent_vol, 
            'disparity': disparity, 'price': price.iloc[-1]
        }
    except: return None

# ==============================================================================
# [Core Logic 2: 펀더멘털 & 베타 (실시간용)] (수정 1, 3 반영)
# ==============================================================================
def get_fundamental_and_beta(ticker, price_series, benchmark_series):
    """yfinance를 이용해 펀더멘털 및 베타 추출 (미국장 추천)"""
    try:
        stock = yf.Ticker(ticker)
        
        # [수정 1] 베타 계산 안정화 (Polyfit + Clip)
        common_idx = price_series.index.intersection(benchmark_series.index)
        if len(common_idx) > 30:
            p_ret = price_series.loc[common_idx].pct_change().dropna()
            b_ret = benchmark_series.loc[common_idx].pct_change().dropna()
            
            # Polyfit (1차원 회귀)으로 기울기 계산
            if len(p_ret) == len(b_ret) and len(p_ret) > 10:
                m, c = np.polyfit(b_ret, p_ret, 1)
                beta = np.clip(m, 0.3, 2.0) # 이상치 방어
            else:
                beta = 1.0
        else:
            beta = 1.0

        # 2. 펀더멘털
        info = stock.info
        
        # 어닝 모멘텀
        eps_ttm = info.get('trailingEps', 0)
        eps_fwd = info.get('forwardEps', eps_ttm)
        if eps_ttm and eps_ttm != 0:
            earn_mom = (eps_fwd - eps_ttm) / abs(eps_ttm)
        else:
            earn_mom = 0.0
            
        # 퀄리티
        roe = info.get('returnOnEquity', 0)
        gm = info.get('grossMargins', 0)
        om = info.get('operatingMargins', 0)
        
        # [수정 3] Accruals (발생액) 계산 복구
        # Net Income - Operating Cash Flow / Total Assets
        accrual = 0.0
        try:
            fin = stock.financials
            cf = stock.cashflow
            bs = stock.balance_sheet
            
            if not fin.empty and not cf.empty and not bs.empty:
                # 항목 이름이 다양할 수 있어 예외처리 필요
                ni = fin.loc['Net Income'].iloc[0] if 'Net Income' in fin.index else 0
                
                # OCF 찾기 (Total Cash From Operating Activities 등)
                ocf_key = [k for k in cf.index if 'Operating' in str(k) and 'Cash' in str(k)]
                ocf = cf.loc[ocf_key[0]].iloc[0] if ocf_key else 0
                
                # Assets 찾기
                asset_key = [k for k in bs.index if 'Total Assets' in str(k)]
                assets = bs.loc[asset_key[0]].iloc[0] if asset_key else 1
                
                if assets != 0:
                    accrual = (ni - ocf) / assets
        except:
            accrual = 0.0 # 실패 시 중립

        return {
            'beta': beta, 'earn_mom': earn_mom,
            'roe': roe, 'gross_margin': gm, 'oper_margin': om, 'accrual': accrual
        }
    except:
        return {'beta': 1.0}

# ==============================================================================
# [★ 핵심: 랭킹 & 중립화 엔진] (수정 2 반영)
# ==============================================================================
def rank_and_score(factor_df, weights, ticker_map=None):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    
    # 1. 섹터 매핑 (한국장 추론)
    if ticker_map:
        scored['sector'] = [infer_sector_kr(ticker_map.get(x, x)) for x in scored.index]
    else:
        scored['sector'] = 'Unknown'

    # 2. 기술적 팩터 Z-Score
    scored['Z_Mom_S'] = z_score(scored['mom_short'])
    scored['Z_Mom_M'] = z_score(scored['mom_mid'])
    scored['Z_Vol'] = z_score(scored['volatility']) * -1 
    scored['Z_Liq'] = z_score(scored['liquidity'])
    scored['Z_MDD'] = z_score(scored['mdd']) 
    
    # 3. 펀더멘털 Z-Score (데이터 존재 시)
    if 'roe' in scored.columns:
        # 섹터별 중립화 적용 (transform)
        def sec_z(x): return (x - x.mean())/x.std() if len(x)>1 else 0
        
        scored['Z_ROE'] = scored.groupby('sector')['roe'].transform(sec_z).fillna(0)
        scored['Z_GM'] = scored.groupby('sector')['gross_margin'].transform(sec_z).fillna(0)
        scored['Z_OM'] = scored.groupby('sector')['oper_margin'].transform(sec_z).fillna(0)
        scored['Z_Earn'] = z_score(scored['earn_mom']).fillna(0)
        scored['Z_Acc'] = z_score(scored['accrual']).fillna(0) * -1 # 낮을수록 좋음
        
        # Fund Score (가중합)
        scored['Fund_Score'] = (
            0.4 * scored['Z_ROE'] + 
            0.3 * scored['Z_GM'] + 
            0.3 * scored['Z_OM'] + 
            0.5 * scored['Z_Earn'] +
            0.5 * scored['Z_Acc']
        )
    else:
        scored['Fund_Score'] = 0.0

    # 4. 종합 점수 (기술 70% + 펀더멘털 30%)
    tech_score = (
        scored['Z_Mom_S'] * weights['mom'] * 0.5 + 
        scored['Z_Mom_M'] * weights['mom'] * 0.5 +
        scored['Z_Vol'] * weights['vol'] + 
        scored['Z_Liq'] * weights['liq'] +
        scored['Z_MDD'] * weights['risk']
    )
    scored['Raw_Total'] = tech_score + (scored['Fund_Score'] * 0.3)
    
    # 5. [수정 2] 회귀 기반 중립화 (Pure Alpha)
    # X에서 Volatility 제거 -> [1, Beta] 만 사용
    if 'beta' in scored.columns and len(scored) > 10:
        beta_vec = scored['beta'].fillna(1.0).values
        # 상수항(Intercept) + Beta
        X = np.column_stack([np.ones(len(beta_vec)), beta_vec])
        y = scored['Raw_Total'].fillna(0).values
        
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coef # Beta로 설명되는 부분
            scored['Alpha_Score'] = y - y_pred # 순수 알파 (잔차)
            final_col = 'Alpha_Score'
        except:
            final_col = 'Raw_Total'
    else:
        final_col = 'Raw_Total'

    # Scaling 0~100
    min_v = scored[final_col].min()
    max_v = scored[final_col].max()
    if max_v != min_v:
        scored['Total_Score'] = (scored[final_col] - min_v) / (max_v - min_v) * 100
    else:
        scored['Total_Score'] = 50
        
    return scored.sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [백테스트 데이터 페처]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_backtest_data(universe, start_date, end_date, country):
    s_str, e_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    p, v = {}, {}
    bm = pd.Series(dtype=float)
    
    try:
        if country == "KR": 
            bm = fdr.DataReader('KS11', s_str, e_str)['Close']
        else: 
            bm_data = yf.download("^GSPC", start=s_str, end=e_str, progress=False)
            if isinstance(bm_data, pd.DataFrame):
                bm = bm_data['Close'] if 'Close' in bm_data.columns else bm_data.iloc[:, 0]
            else:
                bm = bm_data
        if hasattr(bm.index, 'tz_localize'): bm.index = pd.to_datetime(bm.index).tz_localize(None)
    except: pass

    def get(code):
        try:
            if country == "KR": d = fdr.DataReader(code, s_str, e_str)
            else: return None 
            d.index = pd.to_datetime(d.index).tz_localize(None)
            return code, d['Close'], d['Volume']
        except: return None

    if country == "KR":
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(get, c) for c in universe]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res: 
                    if len(res[1]) > 200: p[res[0]], v[res[0]] = res[1], res[2]
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
                        if len(s) > 200: p[t], v[t] = s, vo
        except: pass

    return pd.DataFrame(p), pd.DataFrame(v), bm

# ==============================================================================
# [★ 핵심 백테스트 엔진 (Turnover & T+1)]
# ==============================================================================
def run_backtest(prices, volumes, benchmark, weights, ticker_map, const):
    if prices.empty: return pd.DataFrame()
    reb_dates = prices.resample('BM').last().dropna(how='all').index
    logs = []
    prev_picks = [] 
    
    for i in range(12, len(reb_dates)-1):
        rebal_date = reb_dates[i]       
        next_rebal = reb_dates[i+1]     
        
        p_sub = prices.loc[:rebal_date].tail(300)
        v_sub = volumes.loc[:rebal_date].tail(300)
        active = p_sub.columns[p_sub.iloc[-1].notna()]
        
        daily = []
        for t in active:
            f = calculate_factors(p_sub[t], v_sub[t], const['MIN_AMT'])
            if f:
                f['code'] = t
                daily.append(f)
        if not daily: continue
        
        factor_df = pd.DataFrame(daily).set_index('code')
        # 백테스트에선 Beta 없으므로 펀더멘털/회귀 생략 (기술적 랭킹만)
        ranked = rank_and_score(factor_df, weights, ticker_map=ticker_map)
        picks = ranked.head(const['TOP_N']).index.tolist()
        if not picks: continue
        
        try:
            curr_prices = prices.loc[rebal_date, picks].fillna(0)
            next_prices = prices.loc[next_rebal, picks].fillna(0)
            
            if (curr_prices == 0).any(): curr_prices = curr_prices.replace(0, np.nan).ffill()
            
            # 수익률 계산
            ret_vec = (next_prices / curr_prices) - 1
            ret_vec = ret_vec.fillna(0)
            gross_ret = ret_vec.mean()
            
        except: continue

        if not prev_picks: turnover = 1.0 
        else:
            kept = set(prev_picks) & set(picks)
            turnover = (const['TOP_N'] - len(kept)) / const['TOP_N']
            
        net_ret = gross_ret - (turnover * const['COST_RATE'])
        
        try:
            if isinstance(benchmark, pd.Series):
                b_s = benchmark.asof(rebal_date)
                b_e = benchmark.asof(next_rebal)
            else:
                b_s = benchmark.loc[:rebal_date].iloc[-1]
                b_e = benchmark.loc[:next_rebal].iloc[-1]
            bm_ret = (b_e / b_s) - 1 if b_s != 0 else 0
        except: bm_ret = 0
            
        logs.append({
            'Date': next_rebal, 'Gross_Ret': gross_ret, 'Net_Ret': net_ret,
            'BM_Ret': bm_ret, 'Turnover': turnover,
            'Holdings_Full': ", ".join([ticker_map.get(x,x) for x in picks]),
            'Port_Ret': net_ret
        })
        prev_picks = picks
        
    return pd.DataFrame(logs)

# ==============================================================================
# [전략 최적화 엔진]
# ==============================================================================
def optimize_strategy(prices, volumes, benchmark, ticker_map, presets, const):
    results = []
    if prices.empty: return pd.DataFrame()
    prog = st.progress(0, text="시뮬레이션 시작...")
    
    for i, (name, w) in enumerate(presets.items()):
        weights = {'mom': w[0], 'liq': w[1], 'vol': w[2], 'risk': w[3]}
        try:
            res = run_backtest(prices, volumes, benchmark, weights, ticker_map, const)
            if not res.empty:
                res = res.set_index('Date')
                res['Cum'] = (1+res['Port_Ret']).cumprod()
                
                tot = res['Cum'].iloc[-1]-1
                y = len(res)/12
                cagr = (tot+1)**(1/y)-1 if y>0 else 0
                mdd = (res['Cum']/res['Cum'].cummax()-1).min()
                vol = res['Port_Ret'].std() * np.sqrt(12)
                sharpe = cagr/vol if vol!=0 else 0
                win = (res['Port_Ret']>0).sum()/len(res)
                
                results.append({
                    '전략명': name, '승률': win, 'CAGR': cagr, '누적수익': tot,
                    'MDD': mdd, '샤프': sharpe, '변동성': vol,
                    '가중치': f"{w[0]}|{w[1]}|{w[2]}|{w[3]}"
                })
        except: pass
        prog.progress((i+1)/len(presets), text=f"분석 중: {name}")
    
    prog.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values('CAGR', ascending=False)

def highlight_top3(s):
    is_small = s.name in ['변동성']
    sorted_vals = s.sort_values(ascending=is_small).unique()
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
st.title("🌍 Alpha Seeking Pro (Pure Alpha v4.1)")

with st.sidebar:
    st.header("🏳️ 시장 선택")
    market = st.radio("국가", ["🇰🇷 한국 (Korea)", "🇺🇸 미국 (USA)"], horizontal=True)
    us_index = "S&P 500"
    
    if "한국" in market:
        COUNTRY, CURRENCY, COST_RATE, MIN_AMT, VIX_TICKER, BM_NAME = "KR", "원", 0.002, 5_000_000_000, "KS200VIX", "KOSPI"
        st.info("대상: 유동성 상위 600개")
        TICKER_INFO, ALL_STOCKS = load_kr_data()
    else:
        COUNTRY, CURRENCY, COST_RATE, MIN_AMT, VIX_TICKER, BM_NAME = "US", "$", 0.0005, 5_000_000, "^VIX", "S&P 500"
        us_index = st.selectbox("지수", ["S&P 500", "NASDAQ 100", "DOW 30"])
        with st.spinner("데이터 로딩..."): TICKER_INFO, ALL_STOCKS = load_us_data(us_index)
            
    CONST['MIN_AMT'], CONST['COST_RATE'] = MIN_AMT, COST_RATE
    if st.button("🧹 캐시 초기화"): st.cache_data.clear(); st.rerun()
    st.divider()
    
    mode = st.radio("모드 선택", ["📊 실시간 랭킹", "⚡ 단타/스윙", "🎰 포트폴리오", "📉 백테스트", "🔍 전략 최적화"])
    st.divider()
    
    PRESETS = {
        "사용자 정의": (0.5, 0.5, 0.5, 0.5), "🔥 야수의 심장": (1.0, 1.0, 0.0, 0.0),
        "🚀 달리는 말": (1.0, 0.5, 0.2, 0.3), "🌊 세력주 포착": (0.4, 1.0, 0.2, 0.2),
        "🏰 철벽 방어": (0.1, 0.1, 1.0, 1.0), "🧘 마음의 평화": (0.3, 0.2, 1.0, 0.5),
        "🚑 좀비 헌터": (0.4, 0.3, 0.3, 1.0), "⚖️ 황금 밸런스": (0.5, 0.5, 0.5, 0.5),
        "💎 우상향 정석": (0.7, 0.3, 0.7, 0.4), "🐆 안전한 사냥": (0.8, 0.7, 0.1, 0.8),
        "🧠 스마트 머니": (0.5, 0.8, 0.3, 0.8),
        "⚡ 번개 스캘핑": (1.0, 0.8, 0.0, 0.1), "🛡️ 연금 굴리기": (0.2, 0.3, 0.9, 0.9),
        "🎯 퀄리티 그로스": (0.6, 0.6, 0.6, 0.6), "🌪️ 변동성 사냥꾼": (0.7, 0.5, 0.0, 0.2),
        "🦅 매파의 눈": (0.3, 0.9, 0.4, 0.7)
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
# [MODE EXECUTION]
# ==============================================================================
if mode == "📊 실시간 랭킹":
    if st.button("분석 실행", type="primary"):
        targets = ALL_STOCKS
        results = []
        bar = st.progress(0, text="스캔 중...")
        
        def worker(t):
            try:
                days = 400 if COUNTRY=="KR" else 730
                if COUNTRY=="KR": 
                    df = fdr.DataReader(t, (datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d'))
                    bm = fdr.DataReader('KS11', (datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d'))['Close']
                else: 
                    df = yf.Ticker(t).history(period="2y")
                    bm = yf.Ticker("^GSPC").history(period="2y")['Close']
                
                if len(df)<200: return None
                f = calculate_factors(df['Close'], df['Volume'], MIN_AMT)
                
                # 펀더멘털/베타 (US만)
                if f and COUNTRY == "US":
                    fund = get_fundamental_and_beta(t, df['Close'], bm)
                    if fund: f.update(fund)
                
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
            final = rank_and_score(pd.DataFrame(results).set_index('code'), weights, ticker_map=TICKER_INFO)
            v, vd = get_vix()
            c1, c2, c3 = st.columns(3)
            top = final.iloc[0]
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            c3.metric("VIX", f"{v:.2f}" if v else "N/A", f"{vd:.2f}" if v else "0.0", delta_color="inverse")
            
            cols = ['Total_Score', 'price', 'mom_short', 'sector']
            if 'Alpha_Score' in final.columns: cols.append('Alpha_Score')
            st.dataframe(final[cols].rename(index=TICKER_INFO), use_container_width=True)
        else: st.warning("결과 없음")

elif mode == "⚡ 단타/스윙":
    st.subheader("⚡ 단기 급등 유망주")
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
                bar.progress((i+1)/len(ALL_STOCKS), text=f"⚡ 스캔 중... {i+1}/{len(ALL_STOCKS)}")
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
            else: st.info("조건에 맞는 종목이 없습니다.")

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
            scored = rank_and_score(pd.DataFrame(res).set_index('code'), weights, ticker_map=TICKER_INFO)
            scored['Name'] = [TICKER_INFO.get(x,x) for x in scored.index]
            st.metric("평균 점수", f"{scored['Total_Score'].mean():.1f}")
            st.dataframe(scored[['Name', 'Total_Score', 'mom_short', 'sector']], use_container_width=True)

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
                            if not b.empty:
                                b = b.reindex(res.index, method='ffill')
                                b = b / b.iloc[0] if b.iloc[0]!=0 else b
                                fig.add_trace(go.Scatter(x=b.index, y=b, name="Benchmark", line=dict(dash='dot')))
                        except: pass
                    st.plotly_chart(fig, use_container_width=True)
                    tot = res['Cum'].iloc[-1]-1
                    st.metric("Total Return", f"{tot:.1%}")
                    st.dataframe(res.tail())
                else: st.error("결과 없음")
            else: st.error("데이터 로딩 실패")

else: # 최적화
    c1, c2 = st.columns(2)
    with c1: s_d = st.date_input("시작", CONST['DEFAULT_START_DATE'])
    with c2: e_d = st.date_input("종료", get_last_complete_month_end())
    
    if st.button("전체 전략 비교"):
        with st.spinner("시뮬레이션..."):
            u = ALL_STOCKS[:400] if len(ALL_STOCKS)>400 else ALL_STOCKS
            p, v, bm = fetch_backtest_data(u, s_d, e_d, COUNTRY)
            if not p.empty:
                presets = {k:v for k,v in PRESETS.items() if k!="사용자 정의"}
                res = optimize_strategy(p, v, bm, TICKER_INFO, presets, CONST)
                if not res.empty:
                    st.dataframe(res.style.apply(highlight_top3, subset=['승률', 'CAGR', '누적수익', 'MDD', '샤프', '변동성'])
                                 .format({'승률':'{:.1%}', 'CAGR':'{:.1%}', '누적수익':'{:.1%}', 'MDD':'{:.1%}', '샤프':'{:.2f}', '변동성':'{:.1%}'}), 
                                 use_container_width=True, height=500)
                    best = res.iloc[0]
                    st.success(f"Best: {best['전략명']}")
            else: st.error("데이터 로딩 실패")
