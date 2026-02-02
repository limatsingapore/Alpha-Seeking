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
st.set_page_config(page_title="Alpha Seeking Pro (The Real)", layout="wide", initial_sidebar_state="expanded")

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
# [★ 핵심: 벡터화된 팩터 계산 (속도 10배 향상)]
# ==============================================================================
def calculate_factors_vectorized(p_sub, v_sub, min_amt, trading_days=252):
    """
    for문을 없애고 DataFrame 전체 연산으로 처리
    """
    # 1. 유동성 필터 (평균 거래대금)
    amt = p_sub * v_sub
    avg_amt = amt.tail(20).mean()
    valid_stocks = avg_amt[avg_amt >= min_amt].index
    
    if len(valid_stocks) == 0: return pd.DataFrame()
    
    # 유효 종목만 남김
    p = p_sub[valid_stocks]
    # v = v_sub[valid_stocks] # v는 이제 안씀
    avg_amt = avg_amt[valid_stocks]

    # 2. 팩터 일괄 계산 (Vectorized)
    # 마지막 날짜 기준
    current_price = p.iloc[-1]
    
    # Momentum
    mom_short = p.pct_change(20).iloc[-1]
    mom_mid = p.pct_change(60).iloc[-1]
    
    # Volatility (Return Std * sqrt(252))
    # pct_change 후 std 계산
    daily_ret = p.pct_change()
    vol = daily_ret.tail(trading_days).std() * np.sqrt(trading_days)
    
    # Liquidity Score (Log)
    liquidity = np.log1p(avg_amt)
    
    # MDD (1년)
    window = p.tail(trading_days)
    roll_max = window.cummax()
    dd = (window / roll_max) - 1.0
    mdd = dd.min()
    
    # DataFrame 조립
    factors = pd.DataFrame({
        'mom_short': mom_short,
        'mom_mid': mom_mid,
        'volatility': vol,
        'liquidity': liquidity,
        'mdd': mdd,
        'price': current_price
    })
    
    return factors.dropna()

# ==============================================================================
# [NEW] 펀더멘털 & 베타 팩터 계산기 (실시간 분석용)
# ==============================================================================
def get_fundamental_factors(ticker, price_series, benchmark_series):
    """
    재무제표 데이터와 베타를 계산 (yfinance 전용)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. 베타(Beta) & 시장 중립화 계수 계산
        # 가격 데이터 길이를 맞춤
        common_idx = price_series.index.intersection(benchmark_series.index)
        if len(common_idx) < 60: return None
        
        p_ret = price_series.loc[common_idx].pct_change().dropna()
        b_ret = benchmark_series.loc[common_idx].pct_change().dropna()
        
        # 공분산 / 분산
        cov = np.cov(p_ret, b_ret)[0][1]
        var = np.var(b_ret)
        beta = cov / var if var != 0 else 1.0
        
        # 2. 재무 데이터 가져오기 (비동기 처리 안 하면 느림, 여기선 로직 구현 위주)
        info = stock.info
        fin = stock.financials
        bal = stock.balance_sheet
        cash = stock.cashflow
        
        # 데이터가 없으면 None 반환
        if fin.empty or bal.empty or cash.empty: return None

        # --- [A] 어닝 모멘텀 (EPS Growth) ---
        # (EPS TTM - EPS 1년전) / abs(EPS 1년전)
        # yfinance info에서 trailingEps 제공, 과거 데이터는 financials에서 추정
        eps_ttm = info.get('trailingEps', None)
        
        # 재무제표는 연단위 혹은 분기단위. 여기선 최근 연간 기준 근사치 사용
        # (정교한 4Q 전 데이터는 쿼터별 데이터가 필수이나 속도상 연간 데이터로 대용)
        try:
            eps_prev = fin.loc['Basic EPS'].iloc[1] # 작년 EPS
        except: 
            eps_prev = eps_ttm # 데이터 없으면 모멘텀 0 처리
            
        if eps_ttm and eps_prev:
            earn_mom = (eps_ttm - eps_prev) / abs(eps_prev)
        else:
            earn_mom = 0.0

        # --- [B] 퀄리티 (Quality) ---
        # ROE, Gross Margin, Operating Margin
        roe = info.get('returnOnEquity', 0)
        gm = info.get('grossMargins', 0)
        om = info.get('operatingMargins', 0)
        
        # --- [C] 회계 정직성 (Accruals) ---
        # Accrual = (Net Income - Operating Cash Flow) / Total Assets
        # 발생액이 높으면(현금흐름보다 순이익이 과도하게 높으면) 분식회계/이익조정 의심 -> 낮을수록 좋음
        try:
            ni = cash.loc['Net Income'].iloc[0] if 'Net Income' in cash.index else fin.loc['Net Income'].iloc[0]
            ocf = cash.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash.index else cash.loc['Total Cash From Operating Activities'].iloc[0]
            assets = bal.loc['Total Assets'].iloc[0]
            
            accrual = (ni - ocf) / assets if assets != 0 else 0
        except:
            accrual = 0.0 # 데이터 없으면 중립 처리

        return {
            'beta': beta,
            'earn_mom': earn_mom,
            'roe': roe,
            'gross_margin': gm,
            'oper_margin': om,
            'accrual': accrual
        }
        
    except Exception as e:
        return None

def rank_and_score(factor_df, weights):
    if factor_df.empty: return factor_df
    scored = factor_df.copy()
    
    # 1. 기술적 팩터 랭킹 (기존)
    scored['R_Mom_S'] = scored['mom_short'].rank(pct=True)
    scored['R_Mom_M'] = scored['mom_mid'].rank(pct=True)
    scored['R_Vol'] = scored['volatility'].rank(pct=True, ascending=False)
    scored['R_Liq'] = scored['liquidity'].rank(pct=True)
    scored['R_MDD'] = scored['mdd'].rank(pct=True)
    
    # 2. 펀더멘털 팩터 랭킹 (데이터가 있는 경우에만)
    if 'earn_mom' in scored.columns:
        # Quality Composite Score 계산
        # 0.4 * ROE + 0.3 * GM + 0.3 * OM
        scored['Quality_Raw'] = (
            0.4 * scored['roe'].fillna(0).rank(pct=True) + 
            0.3 * scored['gross_margin'].fillna(0).rank(pct=True) + 
            0.3 * scored['oper_margin'].fillna(0).rank(pct=True)
        )
        
        scored['R_Qual'] = scored['Quality_Raw'].rank(pct=True) # 퀄리티 높을수록 좋음
        scored['R_Earn'] = scored['earn_mom'].fillna(0).rank(pct=True) # 이익모멘텀 높을수록 좋음
        scored['R_Acc'] = scored['accrual'].fillna(0).rank(pct=True, ascending=False) # 발생액 낮을수록 좋음 (Ascending=False)
    else:
        # 펀더멘털 데이터 없으면 0 처리
        scored['R_Qual'] = 0.5
        scored['R_Earn'] = 0.5
        scored['R_Acc'] = 0.5

    # 3. 종합 점수 계산 (가중치 적용)
    # 기존 가중치에 펀더멘털 가중치(임의 설정: 퀄리티/이익/회계 각각 0.5 정도의 영향력 가정)
    # 사용자가 슬라이더로 조절하게 하려면 weights 딕셔너리에 키를 추가해야 함.
    # 여기서는 "스마트 머니" 프리셋 등의 논리에 녹여내기 위해 기본 점수에 가산점 형태로 추가합니다.
    
    base_score = (
        (scored['R_Mom_S'] * 0.5 + scored['R_Mom_M'] * 0.5) * weights['mom'] +
        scored['R_Vol'] * weights['vol'] +
        scored['R_Liq'] * weights['liq'] +
        scored['R_MDD'] * weights['risk']
    )
    
    # 펀더멘털 가산점 (Fundamental Boost) - 약 30% 비중
    fund_score = (scored['R_Qual'] + scored['R_Earn'] + scored['R_Acc']) / 3
    
    # 최종 점수 (기술적 70% + 펀더멘털 30%)
    total_score = base_score * 0.7 + fund_score * 0.3
    
    # 정규화 (0~100)
    scored['Raw_Total'] = (total_score / (sum(weights.values()) * 0.7 + 0.3)) * 100
    
    # 4. 시장 중립화 (Beta Neutralization)
    # Alpha = Score - (Beta * Market_Factor)
    # Market_Factor는 시장의 평균적인 과열도라고 가정 (여기선 50점으로 고정하거나 전체 평균 사용)
    if 'beta' in scored.columns:
        market_bias = 50 # 기준점
        # 베타가 1보다 크면 점수를 깎고(고위험), 1보다 작으면 점수를 높여줌(저변동)
        # 단, 상승장에서는 베타 높은게 좋으므로, 이 로직은 "안정성"을 중시하는 Alpha 로직임.
        scored['Alpha_Score'] = scored['Raw_Total'] - (scored['beta'] * (scored['Raw_Total'].mean() * 0.2)) 
        # 설명: 베타가 높을수록 전체 평균점수의 20%만큼 페널티를 부여 (로우 베타 선호)
        
        # 최종적으로 Alpha Score 사용
        scored['Total_Score'] = scored['Alpha_Score']
    else:
        scored['Total_Score'] = scored['Raw_Total']

    return scored.sort_values(by='Total_Score', ascending=False)

# ==============================================================================
# [백테스트 데이터 페처]
# ==============================================================================
@st.cache_data(ttl=3600*24)
def fetch_backtest_data(universe, start_date, end_date, country):
    s_str, e_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    p, v = {}, {}
    bm = pd.Series(dtype=float)
    
    # Benchmark
    try:
        if country == "KR": 
            bm = fdr.DataReader('KS11', s_str, e_str)['Close']
        else: 
            bm_data = yf.download("^GSPC", start=s_str, end=e_str, progress=False)
            if isinstance(bm_data, pd.DataFrame):
                bm = bm_data['Close'] if 'Close' in bm_data.columns else bm_data.iloc[:, 0]
            else:
                bm = bm_data
        
        if hasattr(bm.index, 'tz_localize'):
            bm.index = pd.to_datetime(bm.index).tz_localize(None)
    except: pass

    # Data Fetching
    def get(code):
        try:
            if country == "KR": d = fdr.DataReader(code, s_str, e_str)
            else: return None
            d.index = pd.to_datetime(d.index).tz_localize(None)
            return code, d['Close'], d['Volume']
        except: return None

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

# ==============================================================================
# [★ 핵심 백테스트 엔진 (T+1 실행 & 벡터화 적용)]
# ==============================================================================
def run_backtest(prices, volumes, benchmark, weights, ticker_map, const):
    if prices.empty: return pd.DataFrame()
    
    # 1. 리밸런싱 날짜: 월말 (Signal Date)
    reb_dates = prices.resample('BM').last().dropna(how='all').index
    
    logs = []
    prev_picks = [] 
    
    for i in range(12, len(reb_dates)-1):
        signal_date = reb_dates[i]      # 이번달 말일 (시그널 발생)
        next_signal_date = reb_dates[i+1] # 다음달 말일 (다음 시그널)
        
        # 2. 실행 날짜 (T+1) 계산: signal_date 다음 거래일 찾기
        # 전체 인덱스에서 signal_date의 위치를 찾고 +1
        try:
            sig_loc = prices.index.get_loc(signal_date)
            if sig_loc + 1 >= len(prices): break # 데이터 끝이면 중단
            
            buy_date = prices.index[sig_loc + 1] # 익월 첫 거래일 (매수)
            
            # 매도 날짜: 다음달 시그널 다음 거래일 (다음달 리밸런싱 매수일 = 이번달 매도일)
            next_sig_loc = prices.index.get_loc(next_signal_date)
            if next_sig_loc + 1 >= len(prices): 
                sell_date = prices.index[-1] # 데이터 끝이면 마지막 날 매도
            else:
                sell_date = prices.index[next_sig_loc + 1] # 익익월 첫 거래일 (매도)
                
        except (KeyError, IndexError):
            continue

        # -----------------------------------------------------------
        # [A] 팩터 계산 (Vectorized)
        # -----------------------------------------------------------
        # signal_date 기준 데이터 슬라이싱
        p_sub = prices.loc[:signal_date].tail(300)
        v_sub = volumes.loc[:signal_date].tail(300)
        
        # 벡터화 함수 호출 (Loop 없음! 쾌적!)
        factor_df = calculate_factors_vectorized(p_sub, v_sub, const['MIN_AMT'])
        if factor_df.empty: continue
        
        ranked = rank_and_score(factor_df, weights)
        picks = ranked.head(const['TOP_N']).index.tolist()
        
        if not picks: continue
        
        # -----------------------------------------------------------
        # [B] 수익률 계산 (T+1 ~ Next T+1)
        # -----------------------------------------------------------
        try:
            # 매수가: buy_date 종가
            # 매도가: sell_date 종가
            curr_prices = prices.loc[buy_date, picks].fillna(0)
            next_prices = prices.loc[sell_date, picks].fillna(0)
            
            # 0원(상폐/정지) 처리: 0이면 매수 불가 -> 제외하거나 ffill 시도
            # 보수적 접근: 매수가 0이면 수익률 0 (현금보유 효과)
            valid_idx = (curr_prices > 0) & (next_prices > 0)
            
            if valid_idx.sum() == 0:
                gross_ret = 0.0
            else:
                curr_p = curr_prices[valid_idx]
                next_p = next_prices[valid_idx]
                asset_returns = (next_p / curr_p) - 1
                gross_ret = asset_returns.mean()
            
        except Exception as e:
            continue

        # -----------------------------------------------------------
        # [C] 비용 계산 (Turnover)
        # -----------------------------------------------------------
        if not prev_picks: turnover_rate = 1.0 
        else:
            kept_stocks = set(prev_picks) & set(picks)
            turnover_rate = (const['TOP_N'] - len(kept_stocks)) / const['TOP_N']
            
        real_cost = turnover_rate * const['COST_RATE']
        net_ret = gross_ret - real_cost
        
        # -----------------------------------------------------------
        # [D] 벤치마크 (T+1 기간 매칭)
        # -----------------------------------------------------------
        try:
            bm_s = benchmark.asof(buy_date)
            bm_e = benchmark.asof(sell_date)
            bm_ret = (bm_e / bm_s) - 1 if (bm_s > 0) else 0.0
        except: bm_ret = 0.0
            
        # -----------------------------------------------------------
        # [E] 기록
        # -----------------------------------------------------------
        logs.append({
            'Date': sell_date, # 수익 실현일 기준
            'Gross_Ret': gross_ret,     
            'Net_Ret': net_ret,         
            'BM_Ret': bm_ret,
            'Turnover': turnover_rate,  
            'Holdings_Full': ", ".join([ticker_map.get(x, x) for x in picks]),
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
    
    prog = st.progress(0, text="전략 시뮬레이션 시작...")
    total = len(presets)
    
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
        
        prog.progress((i+1)/total, text=f"📊 분석 중... {i+1}/{total} ({name})")
    
    prog.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values('CAGR', ascending=False)

def highlight_top3(s):
    is_good_small = s.name in ['변동성'] 
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
st.title("🌍 Alpha Seeking Pro (The Real)")

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
    
    # [16개 프리셋 복구]
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
                f = calculate_factors(df['Close'], df['Volume'], MIN_AMT) # 실시간은 기존 함수 사용 (한 종목씩)
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
            v, vd = get_vix()
            c1, c2, c3 = st.columns(3)
            top = final.iloc[0]
            c1.metric("🏆 Top Pick", TICKER_INFO.get(top.name, top.name))
            c2.metric("⭐ Score", f"{top['Total_Score']:.1f}")
            c3.metric("VIX", f"{v:.2f}" if v else "N/A", f"{vd:.2f}" if v else "0.0", delta_color="inverse")
            st.dataframe(final[['Total_Score', 'price', 'mom_short', 'mdd']].rename(index=TICKER_INFO), use_container_width=True)
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
            else:
                st.info("조건에 맞는 종목이 없습니다.")

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
                            if not b.empty:
                                b = b.reindex(res.index, method='ffill')
                                b = b / b.iloc[0] if b.iloc[0] != 0 else b
                                fig.add_trace(go.Scatter(x=b.index, y=b, name="Benchmark", line=dict(dash='dot')))
                        except: pass
                    st.plotly_chart(fig, use_container_width=True)
                    
                    tot = res['Cum'].iloc[-1]-1
                    st.metric("Total Return", f"{tot:.1%}")
                    st.dataframe(res.tail())
                else: st.error("백테스트 결과가 없습니다.")
            else:
                st.error("❌ 데이터 로딩 실패")

else: # 최적화
    c1, c2 = st.columns(2)
    with c1: s_d = st.date_input("시작", CONST['DEFAULT_START_DATE'])
    with c2: e_d = st.date_input("종료", get_last_complete_month_end())
    
    if st.button("전체 전략 비교"):
        with st.spinner("15+개 전략 시뮬레이션..."):
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
                    
                    # 1등 전략 차트
                    ws_str = best['가중치']
                    parts = ws_str.split('|')
                    w_dict = {'mom': float(parts[0]), 'liq': float(parts[1]), 'vol': float(parts[2]), 'risk': float(parts[3])}
                    
                    res_best = run_backtest(p, v, bm, w_dict, TICKER_INFO, CONST)
                    if not res_best.empty:
                        res_best = res_best.set_index('Date')
                        res_best['Cum'] = (1+res_best['Port_Ret']).cumprod()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=res_best.index, y=res_best['Cum'], name=best['전략명'], line=dict(color='#FFD700', width=2)))
                        if not bm.empty:
                            try:
                                b = bm.loc[s_d:e_d]
                                b = b.reindex(res_best.index, method='ffill')
                                b = b / b.iloc[0] if b.iloc[0] !=0 else b
                                fig.add_trace(go.Scatter(x=b.index, y=b, name="Benchmark", line=dict(dash='dot', color='gray')))
                            except: pass
                        st.plotly_chart(fig, use_container_width=True)
                    
            else: st.error("데이터 로딩 실패")
