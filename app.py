import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr

# --- [로그 설정] ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking Pro (Quant)", layout="wide", initial_sidebar_state="expanded")

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

# --- [VIX 가져오기] ---
@st.cache_data(ttl=600)
def get_vix():
    try:
        kvix = yf.Ticker("^KS200VIX").history(period="1d")
        if not kvix.empty: return kvix['Close'].iloc[-1]
        us_vix = yf.Ticker("^VIX").history(period="1d")
        if not us_vix.empty: return us_vix['Close'].iloc[-1]
        return 0.0
    except: return 0.0

# --- [데이터 로딩: 유니버스 대폭 확장] ---
@st.cache_data(ttl=3600*12)
def load_market_data():
    # 1. KRX 전체 종목 (코스피 + 코스닥)
    df_krx = fdr.StockListing('KRX')
    
    # 2. 유니버스 확장: 시가총액 상위 500개 + 거래대금 상위 300개 (중복 제거)
    # Marcap이 없는 경우 대비 Close * Stocks로 계산 시도하거나 Close로 대체
    sort_col = 'Marcap' if 'Marcap' in df_krx.columns else 'Close'
    amount_col = 'Amount' if 'Amount' in df_krx.columns else 'Close'
    
    top_cap = df_krx.sort_values(by=sort_col, ascending=False).head(500)
    top_vol = df_krx.sort_values(by=amount_col, ascending=False).head(300)
    df_stocks = pd.concat([top_cap, top_vol]).drop_duplicates(subset=['Code'])

    # 3. ETF 데이터
    df_etf = fdr.StockListing('ETF/KR')
    etf_sort = 'Marcap' if 'Marcap' in df_etf.columns else 'Amount'
    df_etf_top = df_etf.sort_values(by=etf_sort, ascending=False).head(50)

    # --- 섹터 분류 (GICS 유사 방식 + 테마) ---
    sectors = {
        "📊 Top 50 ETFs": [],
        "🚀 IT/반도체/하드웨어": [],
        "🔋 2차전지/화학/에너지": [],
        "💊 헬스케어/바이오": [],
        "💰 금융/지주/보험": [],
        "🚗 자동차/운송/물류": [],
        "🛡️ 산업재 (방산/조선/건설/전력)": [],
        "🛍️ 소비재 (식품/화장품/유통)": [],
        "🎮 커뮤니케이션 (게임/미디어/통신)": [],
        "🌈 기타 대형주": []
    }
    
    ticker_name_map = {}

    def process_tickers(df, is_etf=False):
        code_col = 'Code' if 'Code' in df.columns else 'Symbol'
        has_sector = 'Sector' in df.columns
        
        for _, row in df.iterrows():
            code = str(row[code_col])
            name = str(row['Name'])
            
            if is_etf: suffix = ".KS"
            else:
                market = row.get('Market', 'KOSPI')
                suffix = ".KS" if market == 'KOSPI' else ".KQ"

            yf_ticker = code + suffix
            ticker_name_map[yf_ticker] = name
            
            if is_etf:
                sectors["📊 Top 50 ETFs"].append(yf_ticker)
                continue

            # 키워드 기반 섹터 매핑
            sector_val = str(row['Sector']) if has_sector and pd.notnull(row['Sector']) else ""
            combined_text = (name + " " + sector_val).lower()
            
            if any(x in combined_text for x in ['반도체', '전자', '디스플레이', '통신장비', 'sk하이닉스', 'hpsp', '가온칩스', '이수페타시스']):
                sectors["🚀 IT/반도체/하드웨어"].append(yf_ticker)
            elif any(x in combined_text for x in ['전지', '화학', '에너지', '금속', '에코프로', '포스코', '엘앤에프', '캠']):
                sectors["🔋 2차전지/화학/에너지"].append(yf_ticker)
            elif any(x in combined_text for x in ['제약', '바이오', '의료', '생명', '알테오젠', 'hlb', '셀트리온', '삼천당']):
                sectors["💊 헬스케어/바이오"].append(yf_ticker)
            elif any(x in combined_text for x in ['금융', '지주', '은행', '증권', '보험', '메리츠', '기업은행']):
                sectors["💰 금융/지주/보험"].append(yf_ticker)
            elif any(x in combined_text for x in ['자동차', '부품', '운송', '항공', '해운', '기아', '현대차', '글로비스']):
                sectors["🚗 자동차/운송/물류"].append(yf_ticker)
            elif any(x in combined_text for x in ['기계', '방산', '조선', '건설', '전력', '전선', '한화', '현대일렉', '로템']):
                sectors["🛡️ 산업재 (방산/조선/건설/전력)"].append(yf_ticker)
            elif any(x in combined_text for x in ['식품', '음료', '화장품', '의복', '유통', '백화점', '하이브', '엔터', '아모레', 'cj']):
                sectors["🛍️ 소비재 (식품/화장품/유통)"].append(yf_ticker)
            elif any(x in combined_text for x in ['소프트웨어', '게임', '서비스', '통신', '네이버', '카카오', '크래프톤', '텔레콤']):
                sectors["🎮 커뮤니케이션 (게임/미디어/통신)"].append(yf_ticker)
            else:
                sectors["🌈 기타 대형주"].append(yf_ticker)

    process_tickers(df_stocks, is_etf=False)
    process_tickers(df_etf_top, is_etf=True)

    return sectors, ticker_name_map

# --- [데이터 로딩 실행] ---
with st.spinner("시장 데이터 유니버스 구축 중 (상위 800+ 종목)..."):
    SECTORS, TICKER_MAP = load_market_data()

# --- [헬퍼 함수] ---
def get_stock_name(ticker):
    return TICKER_MAP.get(ticker, ticker)

def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        pos_words = ['체결', '수주', '흑자', '개발', '승인', '제휴', '공급', 'M&A', '협력', '상향']
        neg_words = ['적자', '소송', '해지', '반려', '거절', '횡령', '배임', '하향', '유상증자', '거래정지']
        score = 0
        risk_flag = False
        for article in news[:3]:
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 5
            for nw in neg_words:
                if nw in title: 
                    score -= 5
                    if nw in ['횡령', '배임', '상장폐지', '거래정지']: risk_flag = True
        return max(-10, min(10, score)), risk_flag
    except: return 0, False

# --- [★ 퀀트 팩터 계산 함수] ---
def calculate_quant_factors(hist):
    """모멘텀, 변동성, 수급, 회전성, 볼린저밴드 등 팩터 계산"""
    if len(hist) < 252: return None # 최소 1년 데이터 권장
    
    factors = {}
    
    # 1. 모멘텀 (Momentum)
    ret_1m = hist['Close'].pct_change(21).iloc[-1]
    ret_3m = hist['Close'].pct_change(63).iloc[-1]
    ret_6m = hist['Close'].pct_change(126).iloc[-1]
    ret_12m = hist['Close'].pct_change(252).iloc[-1]
    # 12개월 수익률에서 최근 1개월 제외 (단기 과열 방지 로직 적용 시)
    factors['momentum_score'] = (ret_12m * 0.4) + (ret_6m * 0.3) + (ret_3m * 0.2) + (ret_1m * 0.1)
    
    # 2. 변동성 (Volatility) - 낮을수록 좋음 (안정성)
    daily_vol = hist['Close'].pct_change().std() * np.sqrt(252)
    factors['volatility'] = daily_vol
    
    # 3. 거래량 추세 (Volume Trend)
    vol_20 = hist['Volume'].rolling(20).mean().iloc[-1]
    vol_60 = hist['Volume'].rolling(60).mean().iloc[-1]
    factors['volume_trend'] = vol_20 / (vol_60 + 1e-9)
    
    # 4. 평균 회귀 (Mean Reversion) - RSI & CCI
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # CCI
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = ((tp - sma_tp) / (0.015 * mad)).iloc[-1]
    
    factors['rsi'] = rsi
    factors['cci'] = cci
    
    # 5. 볼린저 밴드 위치 (BB Position)
    sma20 = hist['Close'].rolling(20).mean().iloc[-1]
    std20 = hist['Close'].rolling(20).std().iloc[-1]
    factors['bb_position'] = (hist['Close'].iloc[-1] - sma20) / (2 * std20) # 1.0이면 상단, -1.0이면 하단
    
    return factors

# --- [★ 퀀트 스코어링 함수] ---
def score_quant(factors, weights):
    """팩터 값에 가중치를 적용하여 0~100점 산출"""
    score = 0
    max_score = sum(weights.values())
    
    # 1. 모멘텀 (높을수록 좋음)
    mom = factors['momentum_score']
    if mom > 0.5: s = 100
    elif mom > 0.2: s = 80
    elif mom > 0: s = 60
    elif mom > -0.2: s = 40
    else: s = 20
    score += s * (weights['momentum'] / 100)
    
    # 2. 변동성 (낮을수록 좋음)
    vol = factors['volatility']
    if vol < 0.2: s = 100
    elif vol < 0.3: s = 80
    elif vol < 0.4: s = 60
    elif vol < 0.6: s = 40
    else: s = 20
    score += s * (weights['volatility'] / 100)
    
    # 3. 수급 (거래량 추세)
    vt = factors['volume_trend']
    if vt > 2.0: s = 100
    elif vt > 1.2: s = 80
    elif vt > 1.0: s = 60
    else: s = 40
    score += s * (weights['volume'] / 100)
    
    # 4. 회전성 (과매도 구간에서 높은 점수)
    rsi = factors['rsi']
    if rsi <= 30: s = 100 # 과매도 (매수 기회)
    elif rsi <= 45: s = 80
    elif rsi <= 60: s = 60
    elif rsi <= 75: s = 40
    else: s = 20 # 과매수 (위험)
    score += s * (weights['reversion'] / 100)
    
    # 정규화 (100점 만점)
    # 가중치 합이 100이라고 가정 시 그대로 사용, 아니면 비율 조정
    return score

# --- [TA(기술적 분석) 스코어링 - 기존 로직 유지] ---
def score_ta(hist):
    # 기존 SMA 골든크로스, 정배열 로직 등을 간단히 점수화
    cur_price = hist['Close'].iloc[-1]
    sma20 = hist['Close'].rolling(20).mean().iloc[-1]
    sma60 = hist['Close'].rolling(60).mean().iloc[-1]
    sma120 = hist['Close'].rolling(120).mean().iloc[-1]
    
    score = 0
    if cur_price > sma20: score += 20
    if cur_price > sma60: score += 20
    if cur_price > sma120: score += 20
    if sma20 > sma60: score += 20 # 정배열 초기
    if sma60 > sma120: score += 20
    return score

# --- [통합 데이터 수집 및 계산] ---
def fetch_enhanced_stock_data(ticker, weights):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 250: return {"error": f"{ticker}: 데이터 부족"}
        
        # 1. 기본 정보
        cur_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_chg = ((cur_price - prev_close) / prev_close) * 100
        
        # 2. 팩터 계산
        q_factors = calculate_quant_factors(hist)
        if not q_factors: return {"error": f"{ticker}: 팩터 계산 실패"}
        
        # 3. 점수 산출
        quant_score = score_quant(q_factors, weights)
        ta_score = score_ta(hist)
        
        # 뉴스 및 리스크
        news_score, risk_flag = get_news_sentiment(ticker)
        
        # 최종 점수 (Final Score) = Quant(60%) + TA(30%) + News(10%) + VIX보정
        final_score = (quant_score * 0.6) + (ta_score * 0.3) + (news_score)
        
        # 리스크 발생 시 점수 대폭 삭감
        if risk_flag: final_score = 0
        
        # VIX 보정
        if get_vix() > 25: final_score -= 10
        
        final_score
