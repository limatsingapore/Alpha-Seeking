import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging

# --- [로그 설정 (Streamlit Cloud 호환성 고려)] ---
# 클라우드에서는 파일 로그보다 콘솔 로그가 확인하기 쉽습니다.
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
    "🚀 주도주 & 반도체": {
        "반도체 대형주": ["005930.KS", "000660.KS", "000990.KS", "047810.KS"],
        "HBM & 장비": ["066970.KQ", "222800.KQ", "042700.KQ", "121600.KQ"],
    },
    "🔋 이차전지 & 신성장": {
        "이차전지 제조": ["373220.KS", "006400.KS", "051910.KS"],
        "로봇 & AI": ["440820.KS", "304100.KQ", "389120.KQ"],
    },
    "🎮 플랫폼 & 엔터": {
        "IT 플랫폼": ["035420.KS", "035720.KS"],
        "엔터/게임": ["352820.KS", "041510.KQ", "253450.KQ"]
    }
}

# --- [뉴스 센티멘트 함수] ---
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0
        pos_words = ['상승', '상향', '돌파', '최고', '급등', '흑자', '수주', '공급계약', '강세', 'M&A', '협력',
                     'surge', 'beat', 'strong', 'gain', 'rise', 'rally', 'upgrade', 'growth', 'profit']
        neg_words = ['하락', '하향', '이탈', '최저', '급락', '적자', '논란', '조사', '검찰', '약세', '유상증자',
                     'fall', 'plunge', 'miss', 'weak', 'drop', 'decline', 'downgrade', 'loss', 'concern']
        score = 0
        for article in news[:5]:
            title = article.get('title', '').lower()
            for pw in pos_words:
                if pw in title: score += 4
            for nw in neg_words:
                if nw in title: score -= 4
        return max(-12, min(12, score))
    except Exception as e:
        return 0

# --- [VIX 가져오기] ---
@st.cache_data(ttl=300)
def get_vix():
    try:
        # 한국 VIX 데이터가 없을 경우 미국 VIX(^VIX)를 대체재로 사용하거나 기본값 반환
        vix_ticker = yf.Ticker("^KS200VIX") 
        hist = vix_ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        
        # 폴백: 데이터 없으면 미국 VIX라도 참고
        us_vix = yf.Ticker("^VIX").history(period="1d")
        if not us_vix.empty:
            return us_vix['Close'].iloc[-1]
            
        return 20.0 
    except:
        return 20.0

# --- [주식 데이터 가져오기] ---
@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) < 30: return None
        
        cur_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_chg = ((cur_price - prev_close) / prev_close) * 100
        
        # RVOL
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        rvol = (hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 0
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        loss_val = loss.iloc[-1]
        if loss_val == 0:
            rsi = 100
        else:
            rs = gain.iloc[-1] / loss_val
            rsi = 100 - (100 / (1 + rs))
        
        news_score = get_news_sentiment(ticker)
        
        # 스코어링
        vix = get_vix()
        score = 50
        if 40 <= rsi <= 60: score += 10
        elif rsi < 35: score += 15
        elif rsi > 75: score -= 10 # 과매수 감점 추가
        
        score += news_score
        if day_chg > 2: score += 10
        if rvol > 1.5: score += 10
        if vix > 25: score -= 5 
        
        signal = "관망"
        if score >= 70: signal = "🚀 강력 매수"
        elif score >= 55: signal = "👍 매수"
        elif score <= 35: signal = "⚠️ 주의"
        
        # 차트 데이터 (최근 30일)
        chart_data = hist['Close'].tail(30)
        
        return {
            "티커": ticker, "종목명": stock.info.get('shortName', ticker),
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi,
            "뉴스": news_score, "종합점수": score, "신호": signal,
            "chart_data": chart_data # 이 객체는 나중에 분리해야 함
        }
    except Exception as e:
        logging.error(f"Error {ticker}: {e}")
        return None

# --- [웹 UI 구성] ---
st.title("🛡️ QuantNexus KR Web Edition")

with st.sidebar:
    st.header("설정")
    category = st.selectbox("카테고리 선택", list(SECTORS.keys()))
    sub_category = st.selectbox("세부 섹터 선택", list(SECTORS[category].keys()))
    
    st.divider()
    custom_input = st.text_area("커스텀 티커 (쉼표 구분)", placeholder="005930.KS, 000660.KS")
    scan_button = st.button("📊 실시간 스캔 시작", type="primary")

if scan_button:
    # 티커 리스트 결정
    if custom_input.strip():
        tickers = [t.strip() for t in custom_input.split(',') if t.strip()]
        if len(tickers) > 20:
            st.toast("⚠️ 티커가 20개를 초과하여 상위 20개만 분석합니다.", icon="⚠️")
            tickers = tickers[:20]
        title_text = "커스텀 포트폴리오 분석"
    else:
        tickers = SECTORS[category][sub_category]
        title_text = f"{sub_category} 분석"
    
    st.subheader(f"🔍 {title_text}")
    
    with st.spinner('데이터를 분석 중입니다...'):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res: results.append(res)
        
        if results:
            df = pd.DataFrame(results).sort_values(by="종합점수", ascending=False)
            
            # [중요] 차트 데이터와 표 데이터를 분리
            df_display = df.drop(columns=['chart_data'])
            
            # 메트릭
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("분석 종목", f"{len(df)}개")
            top_stock = df.iloc[0]
            c2.metric("최고 점수", f"{top_stock['종합점수']:.0f}점", top_stock['종목명'])
            c3.metric("평균 RSI", f"{df['RSI'].mean():.1f}")
            c4.metric("시장 공포지수(VIX)", f"{get_vix():.2f}")
            
            # 스타일링 함수
            def color_signal(val):
                if '강력' in val: color = '#ef4444'
                elif '매수' in val: color = '#facc15'
                elif '주의' in val: color = '#3b82f6'
                else: color = 'white'
                return f'color: {color}'

            # 메인 테이블 출력 (차트 데이터 제외된 df_display 사용)
            st.dataframe(df_display.style.map(color_signal, subset=['신호'])
                         .format({'현재가': '{:,.0f} 원', '등락률': '{:+.2f}%', 
                                  'RSI': '{:.1f}', 'RVOL': '{:.1f}x'}),
                         use_container_width=True, height=400)
            
            # 차트 확장 패널
            with st.expander("📈 상세 차트 보기 (상위 5개)", expanded=False):
                # 성능을 위해 상위 5개만 먼저 보여줌
                for i, row in df.head(5).iterrows():
                    fig = go.Figure()
                    # 캔들차트 느낌의 선 그래프
                    color = '#ef4444' if row['등락률'] > 0 else '#3b82f6'
                    fig.add_trace(go.Scatter(
                        x=row['chart_data'].index, 
                        y=row['chart_data'], 
                        mode='lines', 
                        line=dict(color=color, width=2),
                        name=row['티커']
                    ))
                    fig.update_layout(
                        title=f"{row['종목명']} ({row['티커']}) - Score: {row['종합점수']}",
                        height=250, 
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # CSV 다운로드 (차트 데이터 제외된 df_display 사용)
            csv = df_display.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ 결과 CSV 다운로드", csv, 
                               f"quant_report_{datetime.now().strftime('%Y%m%d')}.csv", 
                               "text/csv")
            
        else:
            st.error("데이터를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
