import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging
import FinanceDataReader as fdr  # [추가] 한국 주식 데이터 전문 라이브러리

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

# --- [VIX 가져오기] ---
@st.cache_data(ttl=600)
def get_vix():
    try: return yf.Ticker("^KS200VIX").history(period="1d")['Close'].iloc[-1]
    except: return 20.0

# --- [★ 핵심: 전 종목 자동 수집 및 분류 함수] ---
@st.cache_data(ttl=3600*12)  # 12시간마다 갱신 (장 시작 전 한 번이면 충분)
def load_market_data():
    # 1. KOSPI, KOSDAQ 전 종목 리스트 가져오기 (Marcap: 시가총액)
    df_kospi = fdr.StockListing('KOSPI')
    df_kosdaq = fdr.StockListing('KOSDAQ')
    
    # 2. 시가총액 상위 필터링 (우선주 제외 패턴 등은 간단히 처리)
    df_kospi = df_kospi.sort_values(by='Marcap', ascending=False).head(200)
    df_kosdaq = df_kosdaq.sort_values(by='Marcap', ascending=False).head(100)
    
    combined_df = pd.concat([df_kospi, df_kosdaq])
    
    # 3. 섹터 자동 분류 로직 (Keyword matching)
    sectors = {
        "🚀 반도체 & IT": [],
        "🔋 2차전지 & 화학": [],
        "💊 제약 & 바이오": [],
        "💰 금융 (지주/증권/은행)": [],
        "🚗 자동차 & 운송": [],
        "🛡️ 방산/조선/전력/건설": [],
        "💄 소비재 (화장품/식품/엔터)": [],
        "🎮 플랫폼 & 게임 & 통신": [],
        "🌈 기타 대형주": []
    }
    
    # 티커 맵핑용 딕셔너리 생성
    ticker_name_map = {}

    for _, row in combined_df.iterrows():
        code = row['Code']
        name = row['Name']
        sector = str(row['Sector']) if pd.notnull(row['Sector']) else ""
        industry = str(row['Industry']) if pd.notnull(row['Industry']) else ""
        
        # 야후 파이낸스용 티커 변환 (.KS / .KQ)
        # FinanceDataReader는 숫자만 주므로 접미사 붙여야 함
        # (간단한 로직: 코스피 리스트에 있으면 KS, 아니면 KQ)
        suffix = ".KS" if code in df_kospi['Code'].values else ".KQ"
        yf_ticker = code + suffix
        ticker_name_map[yf_ticker] = name
        
        # 키워드 기반 분류
        combined_text = (name + sector + industry).lower()
        
        if any(x in combined_text for x in ['반도체', '전자', '디스플레이', 'sk하이닉스', 'hpsp', '리노']):
            sectors["🚀 반도체 & IT"].append(yf_ticker)
        elif any(x in combined_text for x in ['전지', '화학', '에너지', '에코프로', '포스코퓨처', '금양']):
            sectors["🔋 2차전지 & 화학"].append(yf_ticker)
        elif any(x in combined_text for x in ['제약', '바이오', '생명', '헬스', '알테오젠', 'hlb', '셀트리온']):
            sectors["💊 제약 & 바이오"].append(yf_ticker)
        elif any(x in combined_text for x in ['금융', '지주', '은행', '증권', '보험', '메리츠']):
            sectors["💰 금융 (지주/증권/은행)"].append(yf_ticker)
        elif any(x in combined_text for x in ['자동차', '부품', '에어', '항공', '해운', '글로비스', '기아', '현대차']):
            sectors["🚗 자동차 & 운송"].append(yf_ticker)
        elif any(x in combined_text for x in ['중공업', '방산', '기계', '전력', '전선', '건설', '조선', '한화', '현대일렉']):
            sectors["🛡️ 방산/조선/전력/건설"].append(yf_ticker)
        elif any(x in combined_text for x in ['식품', '음료', '화장품', '엔터', '투어', '쇼핑', 'f&b', '하이브', '푸드']):
            sectors["💄 소비재 (화장품/식품/엔터)"].append(yf_ticker)
        elif any(x in combined_text for x in ['소프트웨어', '게임', '통신', '서비스', '인터넷', '네이버', '카카오', '크래프톤']):
            sectors["🎮 플랫폼 & 게임 & 통신"].append(yf_ticker)
        else:
            sectors["🌈 기타 대형주"].append(yf_ticker)

    return sectors, ticker_name_map

# --- [데이터 로딩 (앱 시작 시 최초 1회 실행)] ---
with st.spinner("최신 시가총액 상위 300개 종목을 불러오는 중입니다..."):
    SECTORS, TICKER_MAP = load_market_data()

# --- [보조지표 및 점수 함수 (동일)] ---
def get_stock_name(ticker):
    return TICKER_MAP.get(ticker, ticker)

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
    sma20 = hist['Close'].rolling(window=20).mean()
    sma5 = hist['Close'].rolling(window=5).mean()
    # CCI
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = (tp - sma_tp).abs().rolling(window=20).mean()
    cci = (tp - sma_tp) / (0.015 * mad)
    
    return macd, signal_line, upper_band, sma120, sma50, sma20, sma5, cci

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
        rsi = 100 if loss.iloc[-1] == 0 else 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))
        
        macd, signal_line, upper_band, sma120, sma50, sma20, sma5, cci = calculate_indicators(hist)
        
        cur_macd, cur_signal = macd.iloc[-1], signal_line.iloc[-1]
        cur_upper = upper_band.iloc[-1]
        cur_sma120, cur_sma50 = sma120.iloc[-1], sma50.iloc[-1]
        cur_sma20, cur_sma5 = sma20.iloc[-1], sma5.iloc[-1]
        prev_sma50 = sma50.iloc[-2]
        prev_sma5, prev_sma20 = sma5.iloc[-2], sma20.iloc[-2]
        cur_cci = cci.iloc[-1]

        score = 0
        reasons = [] 
        
        # 1. 추세
        if cur_price >= cur_sma120: score += 20; reasons.append("장기정배열")
        else: score -= 10
        
        # 2. 골든크로스 (단기 & 중기)
        is_short_gc = (prev_sma5 < prev_sma20) and (cur_sma5 >= cur_sma20)
        if is_short_gc: score += 20; reasons.append("단기골든크로스")
        elif cur_sma5 > cur_sma20: score += 10
        
        is_mid_gc = (prev_close < prev_sma50) and (cur_price >= cur_sma50)
        if is_mid_gc: score += 15; reasons.append("50일선돌파")
        elif cur_price >= cur_sma50: score += 5
        
        # 3. 보조지표
        if cur_macd > cur_signal: score += 10
        if cur_cci < -100: score += 10; reasons.append("CCI침체")
        
        # 4. RSI (과열/침체)
        if rsi <= 35: score += 15; reasons.append("RSI과매도")
        elif 40 <= rsi <= 65: score += 5
        elif rsi >= 75: score -= 15; reasons.append("RSI과열")

        # 5. 기타
        vix = get_vix()
        if vix > 25: score -= 10
        news_score = get_news_sentiment(ticker)
        score += news_score
        if news_score > 0: reasons.append("호재뉴스")
        if rvol > 1.5: score += 5

        score = max(0, min(100, score))
        
        signal = "⚖️ 중립 (관망)"
        if rsi >= 80 or cur_price > cur_upper * 1.05:
            signal = "🔥 과열 (진입금지)"
            reasons.insert(0, "단기급등")
        elif is_mid_gc or is_short_gc:
            signal = "✨ 추세전환 시도"
        elif score >= 75:
            signal = "🚀 강력 매수"
        elif score >= 60:
            signal = "👍 매수"
        elif score <= 35:
            signal = "⚠️ 매도/관망"
            
        reason_str = ", ".join(reasons) if reasons else "-"

        return {
            "티커": ticker, "종목명": get_stock_name(ticker),
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi, "CCI": cur_cci,
            "종합점수": score, "신호": signal, "분석요약": reason_str,
            "chart_data": hist['Close'].tail(60), "SMA50_Cross": is_mid_gc
        }
    except Exception as e: return {"error": f"{ticker}: {str(e)}"}

# --- [웹 UI 구성] ---
st.title("🛡️ Alpha Seeking Pro (Top 300)")

with st.sidebar:
    st.header("🎯 타겟 설정")
    view_mode = st.radio("화면 모드", ["📱 모바일 카드", "💻 PC 테이블"], horizontal=True)
    st.divider()
    
    tab1, tab2 = st.tabs(["📂 섹터 (자동분류)", "⌨️ 직접 입력"])
    with tab1:
        st.info("KOSPI 200 / KOSDAQ 100 기준 자동 분류됨")
        selected_sector = st.selectbox("분석할 섹터를 선택하세요", list(SECTORS.keys()))
    with tab2:
        st.info("예시: 005930.KS, 000660.KS")
        custom_input = st.text_area("티커 입력", height=100)
    
    st.divider()
    scan_button = st.button("📊 분석 시작 (START)", type="primary", use_container_width=True)

if scan_button:
    target_tickers = [t.strip() for t in custom_input.split(',') if t.strip()] if custom_input.strip() else SECTORS[selected_sector]
    
    # 너무 많으면 오래 걸리므로 메시지 표시
    if len(target_tickers) > 50:
        st.info(f"선택하신 섹터의 종목 수({len(target_tickers)}개)가 많아 시간이 조금 걸릴 수 있습니다.")
    
    st.subheader(f"🔍 {selected_sector if not custom_input.strip() else '커스텀'} 분석")
    
    if len(target_tickers) > 0:
        results = []
        errors = []
        my_bar = st.progress(0, text=f"데이터 정밀 분석 중... (총 {len(target_tickers)}개)")
        
        # 종목 수가 늘어났으므로 스레드 조금 늘림 (5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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

            if "모바일" in view_mode:
                st.caption("💡 카드를 누르면 상세 차트가 펼쳐집니다.")
                for idx, row in df.iterrows():
                    header_icon = "✨" if row['SMA50_Cross'] else ("🔥" if "과열" in row['신호'] else ("🚀" if "강력" in row['신호'] else "📊"))
                    with st.expander(f"{header_icon} {row['종목명']} | {row['신호']} ({row['등락률']:+.2f}%)", expanded=False):
                        st.markdown(f"**📌 요약:** :blue[{row['분석요약']}]")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("점수", f"{row['종합점수']:.0f}")
                        m2.metric("현재가", f"{row['현재가']:,.0f}")
                        m3.metric("RSI", f"{row['RSI']:.0f}")
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
                    if '중립' in val: return 'color: #94a3b8'
                    return 'color: white'

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
