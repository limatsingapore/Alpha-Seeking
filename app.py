import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging
import time

# --- [로그 설정] ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- [페이지 설정] ---
st.set_page_config(page_title="Alpha Seeking", layout="wide", initial_sidebar_state="expanded")

# --- [스타일링: 모바일 가독성 최적화] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.1rem !important; }
    .streamlit-expanderHeader { background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- [통합 티커 맵 (약 150개 핵심 종목)] ---
TICKER_MAP = {
    # 반도체
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "000990.KS": "DB하이텍", "403870.KQ": "HPSP", "058470.KQ": "리노공업",
    "005290.KS": "동진쎄미켐", "084370.KQ": "유진테크", "240810.KQ": "원익IPS",
    "036540.KQ": "SFA반도체", "121600.KQ": "나노신소재", "005935.KS": "삼성전자우",
    "253450.KQ": "스튜디오드래곤", # (임시 매핑 수정)
    "353200.KQ": "대명에너지", "093370.KS": "후성",

    # 2차전지
    "373220.KS": "LG에너지솔루션", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠", "247540.KQ": "에코프로비엠", "086520.KQ": "에코프로",
    "066970.KQ": "엘앤에프", "361610.KQ": "SKIET", "278280.KQ": "대주전자재료",
    "005070.KS": "코스모신소재", "096770.KS": "SK이노베이션", "137400.KS": "피엔티",

    # 바이오
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "196170.KQ": "알테오젠",
    "028300.KQ": "HLB", "000100.KS": "유한양행", "128940.KS": "한미약품",
    "000250.KS": "삼천당제약", "214150.KQ": "클래시스", "145020.KQ": "휴젤",
    "243070.KQ": "휴온스", "095700.KQ": "제넥신", "235980.KQ": "메드팩토",

    # 금융
    "105560.KS": "KB금융", "055550.KS": "신한지주", "316140.KS": "우리금융지주",
    "086790.KS": "하나금융지주", "032830.KS": "삼성생명", "000810.KS": "삼성화재",
    "138040.KS": "메리츠금융지주", "024110.KS": "기업은행", "323410.KS": "카카오뱅크",
    "006800.KS": "미래에셋증권", "005940.KS": "NH투자증권", "071050.KS": "한국금융지주",
    "039490.KQ": "키움증권", "028260.KS": "삼성물산",

    # 자동차/운송
    "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스",
    "003490.KS": "대한항공", "011200.KS": "HMM", "086280.KS": "현대글로비스",
    "011210.KS": "현대위아", "028670.KQ": "팬오션",

    # 방산/조선/전력
    "329180.KS": "HD현대중공업", "042660.KS": "한화오션", "010140.KS": "삼성중공업",
    "012450.KS": "한화에어로스페이스", "047810.KS": "한국항공우주", "079550.KS": "LIG넥스원",
    "064350.KS": "현대로템", "267260.KS": "HD현대일렉트릭", "010120.KS": "LS ELECTRIC",
    "024110.KS": "기업은행", "009540.KS": "HD한국조선해양", "000150.KS": "두산",

    # 플랫폼/게임/엔터
    "035420.KS": "NAVER", "035720.KS": "카카오", "259960.KS": "크래프톤",
    "251270.KQ": "넷마블", "036570.KQ": "엔씨소프트", "352820.KS": "하이브",
    "035900.KQ": "JYP Ent.", "041510.KQ": "에스엠", "122870.KQ": "와이지엔터",

    # 소비재
    "003230.KS": "삼양식품", "090430.KS": "아모레퍼시픽", "097950.KS": "CJ제일제당",
    "004370.KS": "농심", "271560.KS": "오리온", "192820.KS": "코스맥스",
    "023530.KS": "롯데쇼핑", "280360.KS": "롯데웰푸드",

    # 원전/에너지/화학
    "034020.KS": "두산에너빌리티", "052690.KS": "한전기술", "015760.KS": "한국전력",
    "011070.KS": "LG이노텍", "009150.KS": "삼성전기", "005490.KS": "POSCO홀딩스",
    "010950.KS": "S-Oil", "078930.KS": "GS", "010130.KS": "고려아연"
}

# --- [통합 섹터 정의 (세부섹터 제거)] ---
SECTORS = {
    "🚀 반도체 & AI": [
        "005930.KS", "000660.KS", "042700.KS", "000990.KS", "403870.KQ", 
        "058470.KQ", "005290.KS", "084370.KQ", "240810.KQ", "121600.KQ", 
        "036540.KQ", "005935.KS", "011070.KS", "009150.KS"
    ],
    "🔋 2차전지 & 화학": [
        "373220.KS", "006400.KS", "051910.KS", "003670.KS", "247540.KQ", 
        "086520.KQ", "066970.KQ", "361610.KQ", "278280.KQ", "005070.KS", 
        "096770.KS", "137400.KS", "051910.KS", "005490.KS", "010950.KS"
    ],
    "🛡️ 방산/조선/전력 (슈퍼사이클)": [
        "267260.KS", "010120.KS", "012450.KS", "047810.KS", "079550.KS", 
        "064350.KS", "329180.KS", "042660.KS", "010140.KS", "009540.KS",
        "034020.KS", "052690.KS", "015760.KS", "000150.KS"
    ],
    "💊 제약 & 바이오": [
        "207940.KS", "068270.KS", "196170.KQ", "028300.KQ", "000100.KS", 
        "128940.KS", "000250.KS", "214150.KQ", "145020.KQ", "243070.KQ", 
        "235980.KQ", "095700.KQ"
    ],
    "💰 금융 & 밸류업": [
        "105560.KS", "055550.KS", "086790.KS", "316140.KS", "138040.KS", 
        "032830.KS", "000810.KS", "024110.KS", "323410.KS", "006800.KS", 
        "071050.KS", "005940.KS", "039490.KQ", "028260.KS"
    ],
    "🚗 자동차 & 운송": [
        "005380.KS", "000270.KS", "012330.KS", "011210.KS", "086280.KS", 
        "003490.KS", "011200.KS", "028670.KQ"
    ],
    "🎮 플랫폼/게임/엔터": [
        "035420.KS", "035720.KS", "259960.KS", "251270.KQ", "036570.KQ", 
        "352820.KS", "035900.KQ", "041510.KQ", "122870.KQ"
    ],
    "🍜 음식료 & 소비재": [
        "003230.KS", "004370.KS", "097950.KS", "271560.KS", "280360.KS",
        "090430.KS", "192820.KS", "023530.KS"
    ]
}

# --- [뉴스 센티멘트] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        pos_words = ['체결', '수주', '흑자', '개발', '승인', '제휴', '공급', 'M&A', '협력', '목표가상향']
        neg_words = ['적자', '소송', '해지', '반려', '거절', '횡령', '배임', '목표가하향', '유상증자']
        score = 0
        for article in news[:3]:
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 5
            for nw in neg_words:
                if nw in title: score -= 5
        return max(-10, min(10, score))
    except: return 0

# --- [VIX 가져오기] ---
@st.cache_data(ttl=600)
def get_vix():
    try: return yf.Ticker("^KS200VIX").history(period="1d")['Close'].iloc[-1]
    except: return 20.0

# --- [종목명 가져오기] ---
def get_stock_name(ticker):
    if ticker in TICKER_MAP: return TICKER_MAP[ticker]
    try: return yf.Ticker(ticker).info.get('shortName', ticker)
    except: return ticker

# --- [보조지표 계산 함수] ---
def calculate_indicators(hist):
    # MACD
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = hist['Close'].rolling(window=20).mean()
    std20 = hist['Close'].rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    
    # SMA
    sma120 = hist['Close'].rolling(window=120).mean()
    sma50 = hist['Close'].rolling(window=50).mean()
    
    return macd, signal_line, upper_band, sma120, sma50

# --- [주식 데이터 및 점수 계산] ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 120: return {"error": f"{ticker}: 데이터 부족"}

        cur_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_chg = ((cur_price - prev_close) / prev_close) * 100
        
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        rvol = (hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 0
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss_val = loss.iloc[-1]
        rsi = 100 if loss_val == 0 else 100 - (100 / (1 + gain.iloc[-1] / loss_val))
        
        macd, signal_line, upper_band, sma120, sma50 = calculate_indicators(hist)
        
        cur_macd, cur_signal = macd.iloc[-1], signal_line.iloc[-1]
        cur_upper, cur_sma120 = upper_band.iloc[-1], sma120.iloc[-1]
        cur_sma50, prev_sma50 = sma50.iloc[-1], sma50.iloc[-2]

        # --- [점수 산정] ---
        score = 0
        
        # 1. 추세 (Trend)
        if cur_price >= cur_sma120: score += 20 # 장기 정배열
        else: score -= 10
        
        # 2. 50일선 돌파 (Golden Cross)
        is_golden_cross_50 = (prev_close < prev_sma50) and (cur_price >= cur_sma50)
        
        if is_golden_cross_50: score += 15
        elif cur_price >= cur_sma50: score += 10
        else: score -= 5
        
        # 3. MACD
        if cur_macd > cur_signal: score += 20
        else: score -= 10
        
        # 4. RSI
        if 40 <= rsi <= 60: score += 10
        elif rsi <= 30: score += 20
        elif rsi >= 75: score -= 20
        
        # 5. 기타
        if cur_price > cur_upper: score -= 10
        if rvol > 1.5: score += 10
        score += get_news_sentiment(ticker)
        if get_vix() > 25: score -= 10

        score = max(0, min(100, score))
        
        signal = "관망"
        if rsi >= 80 or cur_price > cur_upper * 1.05:
            signal = "🔥 과열 (매수금지)"
        elif is_golden_cross_50:
            signal = "✨ 50일선 돌파!"
        elif score >= 80:
            signal = "🚀 강력 매수"
        elif score >= 60:
            signal = "👍 매수"
        elif score <= 30:
            signal = "⚠️ 매도 우위"
            
        chart_data = hist['Close'].tail(60)
        stock_name = get_stock_name(ticker)

        return {
            "티커": ticker, "종목명": stock_name,
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi, 
            "종합점수": score, "신호": signal,
            "chart_data": chart_data, "SMA50_Cross": is_golden_cross_50
        }

    except Exception as e:
        return {"error": f"{ticker} 에러: {str(e)}"}

# --- [웹 UI 구성] ---
st.title("🛡️ Alpha Seeking (Pro)")

with st.sidebar:
    st.header("🎯 타겟 설정")
    
    view_mode = st.radio("화면 모드", ["📱 모바일 카드", "💻 PC 테이블"], horizontal=True)
    st.divider()
    
    tab1, tab2 = st.tabs(["📂 통합 섹터", "⌨️ 직접 입력"])
    with tab1:
        # [수정] 세부 섹터 없이 대분류만 선택하면 끝!
        selected_sector = st.selectbox("분석할 섹터를 선택하세요", list(SECTORS.keys()))
        
    with tab2:
        st.info("예시: 005930.KS, 000660.KS")
        custom_input = st.text_area("티커 입력", height=100)
    
    st.divider()
    scan_button = st.button("📊 분석 시작 (START)", type="primary", use_container_width=True)

if scan_button:
    # 섹터 선택이면 해당 섹터의 모든 종목 가져오기
    target_tickers = [t.strip() for t in custom_input.split(',') if t.strip()] if custom_input.strip() else SECTORS[selected_sector]
    title_text = "커스텀 분석" if custom_input.strip() else f"{selected_sector} 분석"
    
    st.subheader(f"🔍 {title_text}")
    
    if len(target_tickers) > 0:
        results = []
        errors = []
        my_bar = st.progress(0, text=f"데이터 수집 중... (총 {len(target_tickers)}개)")
        
        # 20개 이상일 경우 시간 소요를 고려해 스레드 2개 유지
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in target_tickers}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                res = future.result()
                if res and "error" in res: errors.append(res['error'])
                elif res: results.append(res)
                my_bar.progress((i + 1) / len(target_tickers))
        my_bar.empty()

        if results:
            df = pd.DataFrame(results).sort_values(by="종합점수", ascending=False)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("분석 종목", f"{len(df)}개")
            c2.metric("50일선 돌파", f"{len(df[df['SMA50_Cross']])}개")
            c3.metric("평균 수익률", f"{df['등락률'].mean():.2f}%")
            c4.metric("VIX", f"{get_vix():.2f}")

            # --- [모바일 카드 뷰] ---
            if "모바일" in view_mode:
                st.caption("💡 카드를 눌러 차트 확인 | 50일선 돌파는 별도 표시됨")
                for idx, row in df.iterrows():
                    header_icon = "✨" if row['SMA50_Cross'] else ("🔥" if "과열" in row['신호'] else ("🚀" if "강력" in row['신호'] else "📊"))
                    with st.expander(f"{header_icon} {row['종목명']} | {row['신호']} ({row['등락률']:+.2f}%)", expanded=False):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("점수", f"{row['종합점수']:.0f}")
                        m2.metric("가", f"{row['현재가']:,.0f}")
                        m3.metric("RSI", f"{row['RSI']:.0f}")
                        m4.metric("RVOL", f"{row['RVOL']:.1f}")
                        
                        fig = go.Figure()
                        color = '#ef4444' if row['등락률'] > 0 else '#3b82f6'
                        fig.add_trace(go.Scatter(x=row['chart_data'].index, y=row['chart_data'], mode='lines', line=dict(color=color, width=2)))
                        fig.update_layout(height=180, margin=dict(t=10,b=10,l=10,r=10), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                        st.plotly_chart(fig, use_container_width=True)

            # --- [PC 테이블 뷰] ---
            else:
                df_display = df.drop(columns=['chart_data', 'SMA50_Cross'])
                def color_signal(val):
                    if '강력' in val: return 'color: #ef4444; font-weight: bold'
                    if '돌파' in val: return 'color: #f59e0b; font-weight: bold'
                    if '매수' in val: return 'color: #facc15'
                    if '과열' in val: return 'color: #dc2626'
                    if '매도' in val: return 'color: #3b82f6'
                    return 'color: white'

                format_dict = {'현재가': '{:,.0f} 원', '등락률': '{:+.2f}%', 'RSI': '{:.1f}', 'RVOL': '{:.1f}x', '종합점수': '{:.0f}점'}
                st.dataframe(df_display.style.map(color_signal, subset=['신호']).format(format_dict), use_container_width=True, height=800)

        if errors:
            with st.expander("⚠️ 일부 데이터 누락", expanded=False):
                for e in errors: st.write(e)
        if not results and not errors: st.error("데이터를 가져오지 못했습니다.")
    else: st.warning("티커가 없습니다.")
