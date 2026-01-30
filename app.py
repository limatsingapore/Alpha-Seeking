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

# --- [통합 티커 맵 (섹터 확장 및 ETF 추가)] ---
TICKER_MAP = {
    # 반도체
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "000990.KS": "DB하이텍", "403870.KQ": "HPSP", "058470.KQ": "리노공업",
    "005290.KS": "동진쎄미켐", "084370.KQ": "유진테크", "240810.KQ": "원익IPS",
    "036540.KQ": "SFA반도체", "121600.KQ": "나노신소재",

    # 2차전지
    "373220.KS": "LG에너지솔루션", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠", "247540.KQ": "에코프로비엠", "086520.KQ": "에코프로",
    "066970.KQ": "엘앤에프", "361610.KQ": "SKIET", "278280.KQ": "대주전자재료",

    # 석유화학 & 정유
    "096770.KS": "SK이노베이션", "010950.KS": "S-Oil", "011780.KS": "금호석유",
    "011170.KS": "롯데케미칼", "009830.KS": "한화솔루션",

    # 건설 & 상사
    "028260.KS": "삼성물산", "000720.KS": "현대건설", "006360.KS": "GS건설",
    "047040.KS": "대우건설", "000150.KS": "두산", "010130.KS": "고려아연",

    # 제약/바이오/화장품
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "196170.KQ": "알테오젠",
    "028300.KQ": "HLB", "000100.KS": "유한양행", "128940.KS": "한미약품",
    "090430.KS": "아모레퍼시픽", "192820.KS": "코스맥스", "247540.KQ": "에코프로비엠",
    "214150.KQ": "클래시스", "145020.KQ": "휴젤", "002790.KS": "아모레G",
    "243070.KQ": "휴온스",

    # 금융
    "105560.KS": "KB금융", "055550.KS": "신한지주", "316140.KS": "우리금융지주",
    "086790.KS": "하나금융지주", "032830.KS": "삼성생명", "000810.KS": "삼성화재",
    "138040.KS": "메리츠금융지주", "024110.KS": "기업은행", "323410.KS": "카카오뱅크",
    "006800.KS": "미래에셋증권", "005940.KS": "NH투자증권", "071050.KS": "한국금융지주",
    "039490.KQ": "키움증권",

    # 자동차/운송/조선/방산/전력
    "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스",
    "003490.KS": "대한항공", "011200.KS": "HMM", "010140.KS": "삼성중공업",
    "329180.KS": "HD현대중공업", "042660.KS": "한화오션", "012450.KS": "한화에어로스페이스",
    "047810.KS": "한국항공우주", "079550.KS": "LIG넥스원", "064350.KS": "현대로템",
    "267260.KS": "HD현대일렉트릭", "010120.KS": "LS ELECTRIC",

    # 플랫폼/게임/엔터
    "035420.KS": "NAVER", "035720.KS": "카카오", "259960.KS": "크래프톤",
    "251270.KQ": "넷마블", "036570.KQ": "엔씨소프트", "352820.KS": "하이브",
    "035900.KQ": "JYP Ent.", "041510.KQ": "에스엠", "122870.KQ": "와이지엔터",
    "003230.KS": "삼양식품", "097950.KS": "CJ제일제당",

    # ETF (주요 20개)
    "069500.KS": "KODEX 200", "122630.KS": "KODEX 레버리지", "252670.KS": "KODEX 200선물인버스2X",
    "102110.KS": "TIGER 200", "360750.KS": "TIGER 미국테크TOP10 INDXX", "091160.KS": "KODEX 반도체",
    "305720.KS": "KODEX 2차전지산업", "371460.KS": "TIGER 차이나전기차SOLACTIVE",
    "102780.KS": "KODEX 삼성그룹", "233740.KS": "KODEX 코스닥150레버리지",
    "114800.KS": "KODEX 인버스", "305540.KS": "TIGER 2차전지테마", "229200.KS": "KODEX 코스닥150",
    "133690.KS": "TIGER 미국나스닥100", "329750.KS": "TIGER 미국S&P500",
    "379800.KS": "KODEX 미국나스닥100TR", "364960.KS": "TIGER KRX2차전지K-뉴딜",
    "261220.KS": "KODEX WTI원유선물", "139260.KS": "TIGER 200 IT", "091210.KS": "TIGER 반도체"
}

# --- [섹터 정의] ---
SECTORS = {
    "📊 국내 ETF (Top 20)": [
        "069500.KS", "122630.KS", "252670.KS", "102110.KS", "360750.KS", 
        "091160.KS", "305720.KS", "371460.KS", "102780.KS", "233740.KS",
        "114800.KS", "305540.KS", "229200.KS", "133690.KS", "329750.KS",
        "379800.KS", "364960.KS", "261220.KS", "139260.KS", "091210.KS"
    ],
    "🚀 반도체 & AI": [
        "005930.KS", "000660.KS", "042700.KS", "000990.KS", "403870.KQ", 
        "058470.KQ", "005290.KS", "084370.KQ", "240810.KQ", "121600.KQ", "036540.KQ"
    ],
    "🏗️ 건설 & 석유화학": [
        "028260.KS", "000720.KS", "006360.KS", "047040.KS", "000150.KS",
        "051910.KS", "010950.KS", "011780.KS", "011170.KS", "009830.KS", "010130.KS"
    ],
    "🔋 2차전지": [
        "373220.KS", "006400.KS", "051910.KS", "003670.KS", "247540.KQ", 
        "086520.KQ", "066970.KQ", "361610.KQ", "278280.KQ"
    ],
    "💄 화장품 & 엔터 & 게임": [
        "090430.KS", "192820.KS", "214150.KQ", "002790.KS", "352820.KS",
        "035900.KQ", "041510.KQ", "122870.KQ", "259960.KS", "251270.KQ", "036570.KQ"
    ],
    "💊 제약 & 바이오": [
        "207940.KS", "068270.KS", "196170.KQ", "028300.KQ", "000100.KS", 
        "128940.KS", "145020.KQ", "243070.KQ"
    ],
    "🛡️ 방산/조선/전력": [
        "012450.KS", "047810.KS", "079550.KS", "064350.KS", "329180.KS", 
        "042660.KS", "010140.KS", "267260.KS", "010120.KS"
    ],
    "💰 금융 (지주/증권/은행)": [
        "105560.KS", "055550.KS", "086790.KS", "316140.KS", "138040.KS", 
        "032830.KS", "024110.KS", "323410.KS", "006800.KS", "071050.KS", 
        "005940.KS", "039490.KQ"
    ],
    "🚗 자동차 & 플랫폼": [
        "005380.KS", "000270.KS", "012330.KS", "035420.KS", "035720.KS", "003490.KS"
    ]
}

# --- [뉴스 센티멘트] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        pos_words = ['체결', '수주', '흑자', '개발', '승인', '제휴', '공급', 'M&A', '협력', '상향']
        neg_words = ['적자', '소송', '해지', '반려', '거절', '횡령', '배임', '하향', '유상증자']
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

def get_stock_name(ticker):
    return TICKER_MAP.get(ticker, ticker)

# --- [보조지표 계산 (CCI 추가)] ---
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
    lower_band = sma20 - (std20 * 2)
    
    # SMA
    sma120 = hist['Close'].rolling(window=120).mean()
    sma60 = hist['Close'].rolling(window=60).mean()
    sma50 = hist['Close'].rolling(window=50).mean()
    
    # CCI (Commodity Channel Index) - 새로운 변수
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = (tp - sma_tp).abs().rolling(window=20).mean()
    cci = (tp - sma_tp) / (0.015 * mad)
    
    return macd, signal_line, upper_band, lower_band, sma120, sma60, sma50, cci

# --- [주식 데이터 및 점수 계산] ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 120: return {"error": f"{ticker}: 데이터 부족"}

        # 가격 & RVOL
        cur_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_chg = ((cur_price - prev_close) / prev_close) * 100
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        rvol = (hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 0
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 if loss.iloc[-1] == 0 else 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))
        
        # 지표 계산
        macd, signal_line, upper_band, lower_band, sma120, sma60, sma50, cci = calculate_indicators(hist)
        
        cur_macd, cur_signal = macd.iloc[-1], signal_line.iloc[-1]
        cur_upper, cur_lower = upper_band.iloc[-1], lower_band.iloc[-1]
        cur_sma120, cur_sma60, cur_sma50 = sma120.iloc[-1], sma60.iloc[-1], sma50.iloc[-1]
        prev_sma50 = sma50.iloc[-2]
        cur_cci = cci.iloc[-1]

        # --- [점수 산정: 객관성 강화] ---
        score = 0
        reasons = [] # 매수 추천 이유 저장 리스트
        
        # 1. 추세 (Trend) - 정배열 가중치 증가
        if cur_price > cur_sma120:
            if cur_sma60 > cur_sma120: # 60일선도 120일선 위에 있음 (완전 정배열 초기)
                score += 25
                reasons.append("정배열")
            else:
                score += 15
        else:
            score -= 15 # 역배열은 감점

        # 2. 50일선 골든크로스
        is_golden_cross_50 = (prev_close < prev_sma50) and (cur_price >= cur_sma50)
        if is_golden_cross_50:
            score += 20
            reasons.append("50일선돌파")
        elif cur_price >= cur_sma50:
            score += 10
        
        # 3. MACD
        if cur_macd > cur_signal:
            score += 15
            reasons.append("MACD상승")
        else: score -= 5
        
        # 4. RSI & CCI (더블 체크)
        if 40 <= rsi <= 60: score += 5
        elif rsi <= 30: 
            score += 15
            reasons.append("과매도반등")
        elif rsi >= 70: score -= 20 # 과열 감점 강화
        
        # CCI 체크 (-100 이하면 과매도, +100 이상이면 과매수)
        if cur_cci < -100: 
            score += 10
            reasons.append("CCI침체")
        elif cur_cci > 100: 
            score -= 10
            
        # 5. 거래량 & 뉴스 & 변동성
        if rvol > 2.0: 
            score += 10
            reasons.append("거래량폭발")
        elif rvol > 1.3: 
            score += 5
            
        news_score = get_news_sentiment(ticker)
        score += news_score
        if news_score > 0: reasons.append("호재뉴스")
        
        # 변동성 및 이격도 페널티
        if get_vix() > 25: score -= 5
        if cur_price > cur_upper: score -= 15 # 볼밴 상단 돌파는 매도 시그널일 수 있음

        # 최종 점수 제한
        score = max(0, min(100, score))
        
        # 신호 결정
        signal = "관망"
        if rsi >= 80 or cur_price > cur_upper * 1.05:
            signal = "🔥 과열 (매수금지)"
            reasons = ["단기과열"]
        elif is_golden_cross_50:
            signal = "✨ 추세전환"
        elif score >= 75:
            signal = "🚀 강력 매수"
        elif score >= 60:
            signal = "👍 매수"
        elif score <= 30:
            signal = "⚠️ 매도 우위"
            reasons = ["추세하락"]
            
        # 이유가 없는데 점수만 높은 경우 방지
        reason_str = ", ".join(reasons) if reasons else "-"

        return {
            "티커": ticker, "종목명": get_stock_name(ticker),
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi, "CCI": cur_cci,
            "종합점수": score, "신호": signal, "분석요약": reason_str,
            "chart_data": hist['Close'].tail(60), "SMA50_Cross": is_golden_cross_50
        }
    except Exception as e:
        return {"error": f"{ticker}: {str(e)}"}

# --- [웹 UI 구성] ---
st.title("🛡️ Alpha Seeking Pro")

with st.sidebar:
    st.header("🎯 타겟 설정")
    view_mode = st.radio("화면 모드", ["📱 모바일 카드", "💻 PC 테이블"], horizontal=True)
    st.divider()
    
    tab1, tab2 = st.tabs(["📂 통합 섹터", "⌨️ 직접 입력"])
    with tab1:
        selected_sector = st.selectbox("분석할 섹터를 선택하세요", list(SECTORS.keys()))
    with tab2:
        st.info("예시: 005930.KS, 000660.KS")
        custom_input = st.text_area("티커 입력", height=100)
    
    st.divider()
    scan_button = st.button("📊 분석 시작 (START)", type="primary", use_container_width=True)

if scan_button:
    target_tickers = [t.strip() for t in custom_input.split(',') if t.strip()] if custom_input.strip() else SECTORS[selected_sector]
    title_text = "커스텀 분석" if custom_input.strip() else f"{selected_sector} 분석"
    
    st.subheader(f"🔍 {title_text}")
    
    if len(target_tickers) > 0:
        results = []
        errors = []
        my_bar = st.progress(0, text=f"데이터 정밀 분석 중... (총 {len(target_tickers)}개)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
            c4.metric("VIX Index", f"{get_vix():.2f}")

            if "모바일" in view_mode:
                st.caption("💡 카드를 누르면 상세 차트가 펼쳐집니다.")
                for idx, row in df.iterrows():
                    header_icon = "✨" if row['SMA50_Cross'] else ("🔥" if "과열" in row['신호'] else ("🚀" if "강력" in row['신호'] else "📊"))
                    with st.expander(f"{header_icon} {row['종목명']} | {row['신호']} ({row['등락률']:+.2f}%)", expanded=False):
                        st.markdown(f"**📌 분석 요약:** :blue[{row['분석요약']}]")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("점수", f"{row['종합점수']:.0f}")
                        m2.metric("현재가", f"{row['현재가']:,.0f}")
                        m3.metric("RSI/CCI", f"{row['RSI']:.0f} / {row['CCI']:.0f}")
                        m4.metric("RVOL", f"{row['RVOL']:.1f}")
                        
                        fig = go.Figure()
                        color = '#ef4444' if row['등락률'] > 0 else '#3b82f6'
                        fig.add_trace(go.Scatter(x=row['chart_data'].index, y=row['chart_data'], mode='lines', line=dict(color=color, width=2)))
                        fig.update_layout(height=180, margin=dict(t=10,b=10,l=10,r=10), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                        st.plotly_chart(fig, use_container_width=True)

            else:
                df_display = df.drop(columns=['chart_data', 'SMA50_Cross'])
                
                def color_signal(val):
                    if '강력' in val: return 'color: #ef4444; font-weight: bold'
                    if '전환' in val: return 'color: #f59e0b; font-weight: bold'
                    if '매수' in val: return 'color: #facc15'
                    if '과열' in val: return 'color: #dc2626'
                    if '매도' in val: return 'color: #3b82f6'
                    return 'color: white'

                # [UI 수정] 가격 컬럼 너비 강제 지정 및 분석요약 컬럼 추가
                st.dataframe(
                    df_display.style.map(color_signal, subset=['신호']).format({
                        '현재가': '{:,.0f} 원', '등락률': '{:+.2f}%', 
                        'RSI': '{:.0f}', 'CCI': '{:.0f}', 'RVOL': '{:.1f}x', '종합점수': '{:.0f}점'
                    }),
                    use_container_width=True, 
                    height=800,
                    column_config={
                        "현재가": st.column_config.NumberColumn("현재가 (KRW)", format="%d 원"),
                        "분석요약": st.column_config.TextColumn("매수/매도 핵심 근거", width="medium"),
                        "종목명": st.column_config.TextColumn("종목명", width="small")
                    }
                )

        if errors:
            with st.expander("⚠️ 일부 데이터 누락", expanded=False):
                for e in errors: st.write(e)
        if not results and not errors: st.error("데이터를 가져오지 못했습니다.")
    else: st.warning("티커가 없습니다.")
