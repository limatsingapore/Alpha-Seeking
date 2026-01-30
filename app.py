import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging
import time
import requests_cache # 캐싱을 위한 라이브러리 활용 (없으면 일반 요청)

# --- [로그 설정] ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="K-Quant Nexus Web", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    [data-testid="stDataFrame"] { background-color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- [섹터 데이터 정의] ---
SECTORS = {
    "🚀 반도체 & IT": {
        "반도체 대장주": ["005930.KS", "000660.KS", "000990.KS"],
        "소부장": ["005290.KS", "036540.KQ", "277810.KQ"]
    },
    "🔋 2차전지": {
        "배터리 셀": ["373220.KS", "006400.KS", "051910.KS"],
        "소재(양극재)": ["003670.KS", "247540.KQ", "066570.KS"]
    },
    "💄 소비재 (화장품/식품)": {
        "화장품 & 뷰티": ["090430.KS", "002790.KS", "247540.KQ", "192820.KS", "000100.KS"],
        "식음료": ["097950.KS", "271560.KS", "004370.KS"]
    },
    "🚗 자동차 & 운송": {
        "완성차": ["005380.KS", "000270.KS"],
        "자동차 부품": ["012330.KS", "011210.KS"]
    },
    "💊 제약 & 바이오": {
        "바이오시밀러": ["207940.KS", "068270.KS"],
        "신약 개발": ["000100.KS", "128940.KS"]
    },
    "💰 금융": {
        "금융지주": ["105560.KS", "055550.KS", "316140.KS"],
        "증권": ["005940.KS", "000370.KS"]
    },
    "🎮 플랫폼 & 게임": {
        "플랫폼": ["035420.KS", "035720.KS"],
        "게임": ["259960.KS", "036570.KQ"]
    }
}

# --- [뉴스 센티멘트 함수] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        pos_words = ['상승', '상향', '돌파', '최고', '급등', '흑자', '수주', '강세', 'surge', 'growth']
        neg_words = ['하락', '하향', '이탈', '최저', '급락', '적자', '약세', 'fall', 'loss']
        score = 0
        for article in news[:3]: # 뉴스 개수 줄임 (속도 향상)
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 4
            for nw in neg_words:
                if nw in title: score -= 4
        return max(-12, min(12, score))
    except:
        return 0

# --- [VIX 가져오기] ---
@st.cache_data(ttl=600)
def get_vix():
    try:
        # VIX 조회 실패 시 기본값 반환하여 전체 로직 멈춤 방지
        return yf.Ticker("^KS200VIX").history(period="1d")['Close'].iloc[-1]
    except:
        try:
            return yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        except:
            return 20.0

# --- [주식 데이터 가져오기 (디버깅 모드)] ---
# 캐시 제거 (실시간성 확보 및 에러 확인용)
def fetch_stock_data(ticker):
    try:
        # 1. 티커 객체 생성
        stock = yf.Ticker(ticker)
        
        # 2. 데이터 가져오기 (오류 발생 지점 확인)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": f"{ticker}: 데이터 없음 (Yahoo 차단 가능성)"}
        
        if len(hist) < 10:
            return {"error": f"{ticker}: 데이터 부족 (<10일)"}

        # 3. 지표 계산
        cur_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_chg = ((cur_price - prev_close) / prev_close) * 100
        
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        rvol = (hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 0
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        loss_val = loss.iloc[-1]
        if loss_val == 0:
            rsi = 50
        else:
            rs = gain.iloc[-1] / loss_val
            rsi = 100 - (100 / (1 + rs))
        
        # 4. 부가 정보
        news_score = get_news_sentiment(ticker)
        vix = get_vix()
        
        score = 50
        if 40 <= rsi <= 60: score += 10
        elif rsi < 35: score += 15
        elif rsi > 75: score -= 10
        
        score += news_score
        if day_chg > 2: score += 10
        if rvol > 1.5: score += 10
        if vix > 25: score -= 5 
        
        signal = "관망"
        if score >= 70: signal = "🚀 강력 매수"
        elif score >= 55: signal = "👍 매수"
        elif score <= 35: signal = "⚠️ 주의"
        
        chart_data = hist['Close'].tail(30)
        
        # 성공 시 데이터 반환
        return {
            "티커": ticker, 
            "종목명": ticker, # info 호출은 느려서 생략하고 티커로 대체
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi,
            "뉴스": news_score, "종합점수": score, "신호": signal,
            "chart_data": chart_data
        }

    except Exception as e:
        return {"error": f"{ticker} 에러: {str(e)}"}

# --- [웹 UI 구성] ---
st.title("🛡️ QuantNexus KR Web Edition")

with st.sidebar:
    st.header("🎯 타겟 설정")
    tab1, tab2 = st.tabs(["📂 섹터 선택", "⌨️ 직접 입력"])
    
    with tab1:
        category = st.selectbox("대분류", list(SECTORS.keys()))
        sub_category = st.selectbox("세부 섹터", list(SECTORS[category].keys()))
        
    with tab2:
        st.info("⚠️ 반드시 .KS(코스피) 또는 .KQ(코스닥)를 붙여주세요.")
        custom_input = st.text_area("티커 입력 (쉼표 구분)", 
                                  placeholder="005930.KS, 000660.KS",
                                  height=150)

    st.divider()
    scan_button = st.button("📊 분석 시작 (START)", type="primary", use_container_width=True)

if scan_button:
    is_custom = bool(custom_input.strip())
    
    if is_custom:
        tickers = [t.strip() for t in custom_input.split(',') if t.strip()]
        title_text = "커스텀 포트폴리오 분석"
    else:
        tickers = SECTORS[category][sub_category]
        title_text = f"{sub_category} 분석"
    
    st.subheader(f"🔍 {title_text}")
    
    if len(tickers) > 0:
        results = []
        errors = []
        
        # 프로그레스 바 설정
        progress_text = "데이터 수집 중..."
        my_bar = st.progress(0, text=progress_text)
        
        # [중요] 스레드 수 2개로 제한 (서버 차단 방지)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in tickers}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                res = future.result()
                
                # 에러와 정상 결과 분리
                if res and "error" in res:
                    errors.append(res['error'])
                elif res:
                    results.append(res)
                
                # 진행률 업데이트
                progress = (i + 1) / len(tickers)
                my_bar.progress(progress, text=f"분석 중... ({i+1}/{len(tickers)})")
        
        my_bar.empty() # 바 제거

        # 결과 화면 출력
        if results:
            df = pd.DataFrame(results).sort_values(by="종합점수", ascending=False)
            df_display = df.drop(columns=['chart_data'])
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("분석 성공", f"{len(df)}개")
            c2.metric("실패/차단", f"{len(errors)}개")
            c3.metric("평균 수익률", f"{df['등락률'].mean():.2f}%")
            c4.metric("VIX", f"{get_vix():.2f}")

            def color_signal(val):
                if '강력' in val: color = '#ef4444'
                elif '매수' in val: color = '#facc15'
                elif '주의' in val: color = '#3b82f6'
                else: color = 'white'
                return f'color: {color}'

            st.dataframe(df_display.style.map(color_signal, subset=['신호'])
                         .format({'현재가': '{:,.0f} 원', '등락률': '{:+.2f}%',
