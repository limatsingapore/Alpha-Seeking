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

# --- [티커-종목명 매핑 사전] ---
TICKER_MAP = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "000990.KS": "DB하이텍",
    "005290.KS": "동진쎄미켐", "036540.KQ": "SFA반도체", "277810.KQ": "천보", "042700.KS": "한미반도체",
    "373220.KS": "LG에너지솔루션", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠", "247540.KQ": "에코프로비엠", "066570.KS": "LG전자", "086520.KQ": "에코프로",
    "090430.KS": "아모레퍼시픽", "002790.KS": "아모레G", "192820.KS": "코스맥스", "000100.KS": "유한양행",
    "097950.KS": "CJ제일제당", "271560.KS": "오리온", "004370.KS": "농심",
    "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스", "011210.KS": "현대위아",
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "128940.KS": "한미약품", "000250.KS": "삼천당제약",
    "105560.KS": "KB금융", "055550.KS": "신한지주", "316140.KS": "우리금융지주",
    "005940.KS": "NH투자증권", "000370.KS": "한화손해보험",
    "035420.KS": "NAVER", "035720.KS": "카카오", "259960.KS": "크래프톤", "036570.KQ": "엔씨소프트",
    "042660.KS": "한화오션", "010140.KS": "삼성중공업", "011200.KS": "HMM",
    "012450.KS": "한화에어로스페이스", "047810.KS": "한국항공우주", "079550.KS": "LIG넥스원"
}

# --- [섹터 데이터 정의] ---
SECTORS = {
    "🚀 반도체 & IT": {
        "반도체 대장주": ["005930.KS", "000660.KS", "000990.KS"],
        "소부장 (장비/소재)": ["005290.KS", "036540.KQ", "277810.KQ", "042700.KS"]
    },
    "🔋 2차전지": {
        "배터리 셀": ["373220.KS", "006400.KS", "051910.KS"],
        "양극재/음극재": ["003670.KS", "247540.KQ", "066570.KS", "086520.KQ"]
    },
    "💄 소비재 (화장품/식품)": {
        "화장품 & 뷰티": ["090430.KS", "002790.KS", "247540.KQ", "192820.KS"],
        "식음료": ["097950.KS", "271560.KS", "004370.KS"]
    },
    "🚗 자동차 & 운송": {
        "완성차": ["005380.KS", "000270.KS"],
        "자동차 부품": ["012330.KS", "011210.KS"]
    },
    "💊 제약 & 바이오": {
        "바이오시밀러": ["207940.KS", "068270.KS"],
        "신약 개발": ["000100.KS", "128940.KS", "000250.KS"]
    },
    "💰 금융": {
        "금융지주": ["105560.KS", "055550.KS", "316140.KS"],
        "증권": ["005940.KS", "000370.KS"]
    },
    "🎮 플랫폼 & 게임": {
        "플랫폼": ["035420.KS", "035720.KS"],
        "게임": ["259960.KS", "036570.KQ"]
    },
    "🏗️ 중공업 & 방산": {
        "조선/해운": ["042660.KS", "010140.KS", "011200.KS"],
        "방산": ["012450.KS", "047810.KS", "079550.KS"]
    }
}

# --- [뉴스 센티멘트 함수 (로직 개선됨)] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        
        # [수정] 단순 등락('상승', '하락')은 제외하고 실질적 호재/악재 키워드만 남김
        pos_words = ['체결', '수주', '흑자', '개발', '승인', '제휴', '공급', 'M&A', '협력', '목표가상향',
                     'contract', 'agreement', 'approval', 'profit', 'launch', 'partnership']
        
        neg_words = ['적자', '소송', '해지', '반려', '거절', '횡령', '배임', '목표가하향', '유상증자',
                     'loss', 'lawsuit', 'reject', 'cancel', 'investigation']
        
        score = 0
        for article in news[:3]:
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 5  # 핵심 호재는 점수 상향
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

# --- [주식 데이터 가져오기 (RSI 로직 개선됨)] ---
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
        
        # --- [점수 산정 로직 수정] ---
        score = 50
        
        # 1. RSI (과매수 페널티 강화)
        if rsi >= 80:
            score -= 30  # [수정] 과매수 구간 대폭 감점 (위험)
        elif rsi >= 70:
            score -= 10  # [수정] 과열 주의
        elif rsi <= 30:
            score += 20  # [수정] 과매도 구간 (기술적 반등 기대)
        elif 40 <= rsi <= 60:
            score += 10  # 안정적 추세
            
        # 2. 거래량 & 변동성
        if rvol > 1.5: score += 10
        if day_chg > 2: score += 10
        if vix > 25: score -= 5
        
        # 3. 뉴스 점수 반영
        score += news_score
        
        # --- [신호 결정 로직] ---
        signal = "관망"
        
        # RSI 80 이상이면 점수가 높아도 무조건 경고
        if rsi >= 80:
            signal = "🔥 과열 (매수금지)"
        elif score >= 75:
            signal = "🚀 강력 매수"
        elif score >= 60:
            signal = "👍 매수"
        elif score <= 35:
            signal = "⚠️ 주의"
            
        chart_data = hist['Close'].tail(30)
        stock_name = TICKER_MAP.get(ticker, ticker)

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
                elif '매수' in val and '과열' not in val: color = '#facc15' # 과열이 아닐때만 노랑
                elif '과열' in val: color = '#dc2626' # 과열은 진한 빨강(경고)
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
