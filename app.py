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

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        color: white !important;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }
    [data-testid="stDataFrame"] { background-color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- [티커-종목명 매핑 (시총 상위 150위권 대거 포함)] ---
# * yfinance는 한글 종목명을 지원하지 않으므로, 이 사전이 가장 빠르고 정확합니다.
TICKER_MAP = {
    # [반도체]
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "403870.KQ": "HPSP", "005290.KS": "동진쎄미켐", "058470.KQ": "리노공업",
    "000990.KS": "DB하이텍", "084370.KQ": "유진테크", "240810.KQ": "원익IPS",
    "036540.KQ": "SFA반도체", "121600.KQ": "나노신소재", "005935.KS": "삼성전자우",

    # [2차전지]
    "373220.KS": "LG에너지솔루션", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠", "247540.KQ": "에코프로비엠", "086520.KQ": "에코프로",
    "066970.KQ": "엘앤에프", "361610.KQ": "SKIET", "278280.KQ": "대주전자재료",
    "005070.KS": "코스모신소재", "096770.KS": "SK이노베이션",

    # [바이오/헬스케어]
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "196170.KQ": "알테오젠",
    "028300.KQ": "HLB", "000100.KS": "유한양행", "128940.KS": "한미약품",
    "000250.KS": "삼천당제약", "214150.KQ": "클래시스", "145020.KQ": "휴젤",
    "243070.KQ": "휴온스", "235980.KQ": "메드팩토", "095700.KQ": "제넥신",

    # [금융/지주/밸류업] (시총 순위 반영)
    "105560.KS": "KB금융", "055550.KS": "신한지주", "032830.KS": "삼성생명",
    "138040.KS": "메리츠금융지주", "086790.KS": "하나금융지주", "000810.KS": "삼성화재",
    "316140.KS": "우리금융지주", "323410.KS": "카카오뱅크", "024110.KS": "기업은행",
    "006800.KS": "미래에셋증권", "071050.KS": "한국금융지주", "005940.KS": "NH투자증권",
    "039490.KQ": "키움증권", "028260.KS": "삼성물산", "003550.KS": "LG",

    # [자동차/운송]
    "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스",
    "032350.KS": "롯데관광개발", "003490.KS": "대한항공", "011200.KS": "HMM",
    "086280.KS": "현대글로비스", "011210.KS": "현대위아", "028670.KQ": "팬오션",

    # [전력/방산/조선]
    "329180.KS": "HD현대중공업", "012450.KS": "한화에어로스페이스", "267260.KS": "HD현대일렉트릭",
    "042660.KS": "한화오션", "010140.KS": "삼성중공업", "047810.KS": "한국항공우주",
    "064350.KS": "현대로템", "079550.KS": "LIG넥스원", "010120.KS": "LS ELECTRIC",
    "009540.KS": "HD한국조선해양",

    # [플랫폼/게임/엔터]
    "035420.KS": "NAVER", "035720.KS": "카카오", "259960.KS": "크래프톤",
    "251270.KQ": "넷마블", "036570.KQ": "엔씨소프트", "352820.KS": "하이브",
    "035900.KQ": "JYP Ent.", "041510.KQ": "에스엠", "122870.KQ": "와이지엔터",

    # [소비재]
    "003230.KS": "삼양식품", "090430.KS": "아모레퍼시픽", "097950.KS": "CJ제일제당",
    "004370.KS": "농심", "271560.KS": "오리온", "028050.KS": "삼성엔지니어링",
    "192820.KS": "코스맥스", "023530.KS": "롯데쇼핑", "007090.KS": "글로벌텍스프리"
}

# --- [섹터 데이터 정의 (시총 순위별 재정렬)] ---
SECTORS = {
    "🚀 반도체 & AI": {
        "메모리 & 시스템 (시총상위)": ["005930.KS", "000660.KS", "000990.KS"],
        "HBM & 소부장 주도주": ["042700.KS", "403870.KQ", "058470.KQ", "084370.KQ"],
        "온디바이스 & 소재": ["121600.KQ", "005290.KS", "036540.KQ"]
    },
    "💰 금융 & 밸류업 (시총순)": {
        "4대 금융지주": ["105560.KS", "055550.KS", "086790.KS", "316140.KS"],
        "보험 & 특수은행": ["032830.KS", "000810.KS", "138040.KS", "024110.KS", "323410.KS"],
        "증권사 (초대형IB)": ["006800.KS", "071050.KS", "005940.KS", "039490.KQ"]
    },
    "🔋 2차전지 & 에너지": {
        "셀 메이커 (TOP3)": ["373220.KS", "006400.KS", "051910.KS"],
        "양극재 (시총상위)": ["247540.KQ", "003670.KS", "086520.KQ", "066970.KQ"],
        "전해액/분리막": ["278280.KQ", "361610.KQ", "005070.KS"]
    },
    "💊 제약 & 바이오": {
        "바이오시밀러 (대장)": ["207940.KS", "068270.KS"],
        "K-바이오 3대장 (알테/HLB/삼천당)": ["196170.KQ", "028300.KQ", "000250.KS"],
        "전통제약 & 톡신": ["000100.KS", "128940.KS", "214150.KQ", "145020.KQ"]
    },
    "⚡ 전력 & 방산 & 조선": {
        "전력설비 (슈퍼사이클)": ["322000.KS", "267260.KS", "010120.KS"], # HD현대일렉트릭 코드 수정(322000 X -> 267260)
        "방산 (수출주)": ["012450.KS", "047810.KS", "079550.KS", "064350.KS"],
        "조선 (빅3)": ["329180.KS", "010140.KS", "042660.KS"]
    },
    "🚗 자동차 & 운송": {
        "완성차": ["005380.KS", "000270.KS"],
        "부품 & 물류": ["012330.KS", "086280.KS", "011200.KS", "003490.KS"]
    },
    "🍜 소비재 (K-Food/Beauty)": {
        "라면 & 식품": ["003230.KS", "004370.KS", "097950.KS", "271560.KS"],
        "화장품": ["090430.KS", "192820.KS", "214150.KQ"]
    },
    "🎮 플랫폼 & 엔터": {
        "인터넷": ["035420.KS", "035720.KS"],
        "게임 (크래프톤 외)": ["259960.KS", "251270.KQ", "036570.KQ"],
        "엔터 4사": ["352820.KS", "035900.KQ", "041510.KQ", "122870.KQ"]
    }
}

# --- [뉴스 센티멘트 함수] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        
        pos_words = ['체결', '수주', '흑자', '개발', '승인', '제휴', '공급', 'M&A', '협력', '목표가상향',
                     'contract', 'agreement', 'approval', 'profit', 'launch', 'partnership']
        neg_words = ['적자', '소송', '해지', '반려', '거절', '횡령', '배임', '목표가하향', '유상증자',
                     'loss', 'lawsuit', 'reject', 'cancel', 'investigation']
        
        score = 0
        for article in news[:3]:
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 5
            for nw in neg_words:
                if nw in title: score -= 5
        return max(-10, min(10, score))
    except:
        return 0

# --- [VIX 가져오기] ---
@st.cache_data(ttl=600)
def get_vix():
    try:
        return yf.Ticker("^KS200VIX").history(period="1d")['Close'].iloc[-1]
    except:
        try:
            return yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        except:
            return 20.0

# --- [종목명 가져오기 (Fallback 기능 추가)] ---
def get_stock_name(ticker):
    # 1. 내장 사전에서 찾기 (가장 빠름)
    if ticker in TICKER_MAP:
        return TICKER_MAP[ticker]
    
    # 2. 사전에 없으면 yfinance로 시도 (느리지만 커스텀 입력을 위해 필요)
    try:
        info = yf.Ticker(ticker).info
        # shortName이 있으면 반환, 없으면 ticker 반환
        return info.get('shortName', ticker)
    except:
        return ticker

# --- [주식 데이터 가져오기] ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty: return {"error": f"{ticker}: 데이터 없음"}
        if len(hist) < 10: return {"error": f"{ticker}: 데이터 부족"}

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
        
        news_score = get_news_sentiment(ticker)
        vix = get_vix()
        
        # 점수 로직
        score = 50
        if rsi >= 80: score -= 30
        elif rsi >= 70: score -= 10
        elif rsi <= 30: score += 20
        elif 40 <= rsi <= 60: score += 10
            
        if rvol > 1.5: score += 10
        if day_chg > 2: score += 10
        if vix > 25: score -= 5
        score += news_score
        
        signal = "관망"
        if rsi >= 80: signal = "🔥 과열 (매수금지)"
        elif score >= 75: signal = "🚀 강력 매수"
        elif score >= 60: signal = "👍 매수"
        elif score <= 35: signal = "⚠️ 주의"
            
        chart_data = hist['Close'].tail(30)
        
        # [수정] 종목명 가져오기 함수 사용
        stock_name = get_stock_name(ticker)

        return {
            "티커": ticker, "종목명": stock_name,
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi,
            "뉴스": news_score, "종합점수": score, "신호": signal,
            "chart_data": chart_data
        }

    except Exception as e:
        return {"error": f"{ticker} 에러: {str(e)}"}

# --- [웹 UI 구성] ---
st.title("🛡️ Alpha Seeking")

with st.sidebar:
    st.header("🎯 타겟 설정")
    tab1, tab2 = st.tabs(["📂 섹터 선택", "⌨️ 직접 입력"])
    with tab1:
        category = st.selectbox("대분류", list(SECTORS.keys()))
        sub_category = st.selectbox("세부 섹터", list(SECTORS[category].keys()))
    with tab2:
        st.info("⚠️ 예시: 005930.KS, 000660.KS")
        custom_input = st.text_area("티커 입력 (쉼표 구분)", height=150)
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
        progress_text = "데이터 수집 중..."
        my_bar = st.progress(0, text=progress_text)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in tickers}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                res = future.result()
                if res and "error" in res: errors.append(res['error'])
                elif res: results.append(res)
                my_bar.progress((i + 1) / len(tickers), text=f"분석 중... ({i+1}/{len(tickers)})")
        my_bar.empty()

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
                elif '매수' in val and '과열' not in val: color = '#facc15'
                elif '과열' in val: color = '#dc2626'
                elif '주의' in val: color = '#3b82f6'
                else: color = 'white'
                return f'color: {color}'

            format_dict = {'현재가': '{:,.0f} 원', '등락률': '{:+.2f}%', 'RSI': '{:.1f}', 'RVOL': '{:.1f}x'}
            st.dataframe(df_display.style.map(color_signal, subset=['신호']).format(format_dict), use_container_width=True, height=500)
            
            with st.expander("📈 차트 보기 (Dark Mode)", expanded=True):
                cols = st.columns(3)
                for i, (idx, row) in enumerate(df.head(6).iterrows()):
                    with cols[i % 3]:
                        fig = go.Figure()
                        color = '#ef4444' if row['등락률'] > 0 else '#3b82f6'
                        fig.add_trace(go.Scatter(x=row['chart_data'].index, y=row['chart_data'], mode='lines', line=dict(color=color, width=2)))
                        fig.update_layout(title=f"{row['종목명']} ({row['등락률']:+.1f}%)", height=250, margin=dict(t=40,b=20,l=20,r=20), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                        st.plotly_chart(fig, use_container_width=True)
        
        if errors:
            with st.expander("⚠️ 분석 실패 로그", expanded=True):
                for e in errors: st.write(e)
        if not results and not errors: st.error("데이터를 가져오지 못했습니다.")
    else: st.warning("티커가 없습니다.")
