import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import plotly.graph_objects as go
import logging

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

# --- [대규모 섹터 데이터 확장] ---
SECTORS = {
    "🚀 반도체 & IT 하드웨어": {
        "반도체 대장주": ["005930.KS", "000660.KS", "000990.KS"],
        "HBM & 장비": ["042700.KS", "000250.KS", "066970.KQ", "222800.KQ", "121600.KQ"],
        "소부장 (소재/부품)": ["005290.KS", "036540.KQ", "277810.KQ", "138080.KQ"]
    },
    "🔋 2차전지 & 에너지": {
        "배터리 셀 (대형)": ["373220.KS", "006400.KS", "051910.KS"],
        "양극재/음극재": ["003670.KS", "247540.KQ", "066570.KS", "291230.KQ"],
        "전해액/분리막": ["278280.KQ", "361610.KQ", "096770.KS"]
    },
    "🚗 자동차 & 운송": {
        "완성차": ["005380.KS", "000270.KS"],
        "자동차 부품": ["012330.KS", "009240.KS", "011210.KS", "298050.KS"],
        "해운 & 항공": ["011200.KS", "003490.KS", "086280.KS", "005880.KS"]
    },
    "💊 제약 & 바이오": {
        "바이오시밀러/CDMO": ["207940.KS", "068270.KS", "000100.KS"],
        "신약 개발 & 플랫폼": ["196170.KQ", "000250.KS", "298040.KQ", "235980.KQ"],
        "의료기기 & 미용": ["214150.KQ", "145020.KQ", "243070.KQ"]
    },
    "🏗️ 중공업 & 방산": {
        "조선 & 기자재": ["329180.KS", "042660.KS", "009540.KS", "010620.KS"],
        "방위산업": ["012450.KS", "047810.KS", "079550.KS", "272210.KS"],
        "전력설비 (AI수혜)": ["024110.KS", "267260.KS", "003550.KS"]
    },
    "💰 금융 & 지주": {
        "금융지주": ["105560.KS", "055550.KS", "316140.KS", "086790.KS"],
        "증권/보험": ["005940.KS", "000370.KS", "000810.KS", "000030.KS"],
        "주요 지주사": ["003550.KS", "000120.KS", "004020.KS"]
    },
    "💄 소비재 (화장품/식품)": {
        "화장품 & 뷰티": ["090430.KS", "002790.KS", "247540.KQ", "192820.KS"],
        "식음료": ["097950.KS", "271560.KS", "004370.KS", "005610.KS"],
        "유통/면세": ["004170.KS", "023530.KS", "007090.KS"]
    },
    "🎮 플랫폼 & 콘텐츠": {
        "인터넷 플랫폼": ["035420.KS", "035720.KS"],
        "게임": ["259960.KS", "251270.KQ", "041510.KQ", "036570.KQ"],
        "엔터테인먼트": ["352820.KS", "253450.KQ", "122870.KQ"]
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
        vix_ticker = yf.Ticker("^KS200VIX") 
        hist = vix_ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        
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
        
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        rvol = (hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 0
        
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
        
        return {
            "티커": ticker, "종목명": stock.info.get('shortName', ticker),
            "현재가": cur_price, "등락률": day_chg,
            "RVOL": rvol, "RSI": rsi,
            "뉴스": news_score, "종합점수": score, "신호": signal,
            "chart_data": chart_data
        }
    except Exception as e:
        return None

# --- [웹 UI 구성] ---
st.title("🛡️ QuantNexus KR Web Edition")

with st.sidebar:
    st.header("🎯 타겟 설정")
    
    # 탭으로 기능 분리
    tab1, tab2 = st.tabs(["📂 섹터 선택", "⌨️ 직접 입력"])
    
    with tab1:
        category = st.selectbox("대분류", list(SECTORS.keys()))
        sub_category = st.selectbox("세부 섹터", list(SECTORS[category].keys()))
        
    with tab2:
        st.info("⚠️ 반드시 .KS(코스피) 또는 .KQ(코스닥)를 붙여주세요.")
        custom_input = st.text_area("티커 입력 (쉼표 구분)", 
                                  placeholder="예시:\n005930.KS (삼성전자)\n000660.KS (SK하이닉스)\n086520.KQ (에코프로)",
                                  height=150)

    st.divider()
    scan_button = st.button("📊 분석 시작 (START)", type="primary", use_container_width=True)

if scan_button:
    # 1. 입력 모드 확인 (커스텀 입력란에 내용이 있으면 커스텀 모드)
    is_custom = bool(custom_input.strip())
    
    if is_custom:
        tickers = [t.strip() for t in custom_input.split(',') if t.strip()]
        title_text = "커스텀 포트폴리오 분석"
    else:
        tickers = SECTORS[category][sub_category]
        title_text = f"{sub_category} 분석"
    
    st.subheader(f"🔍 {title_text}")
    
    # 2. 티커 유효성 검사 및 개수 제한
    if len(tickers) == 0:
        st.error("입력된 티커가 없습니다.")
    elif len(tickers) > 30:
        st.warning(f"티커가 너무 많습니다. 상위 30개만 분석합니다. (입력: {len(tickers)}개)")
        tickers = tickers[:30]
        
    # 3. 데이터 수집
    if len(tickers) > 0:
        with st.spinner(f'{len(tickers)}개 종목 분석 중... (잠시만 기다려주세요)'):
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in tickers}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    res = future.result()
                    if res: results.append(res)
            
            # 4. 결과 출력
            if results:
                df = pd.DataFrame(results).sort_values(by="종합점수", ascending=False)
                df_display = df.drop(columns=['chart_data'])
                
                # 상단 지표
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("분석 성공", f"{len(df)} / {len(tickers)}")
                top_stock = df.iloc[0]
                c2.metric("★ 1위 종목", top_stock['종목명'], f"{top_stock['종합점수']:.0f}점")
                c3.metric("평균 등락률", f"{df['등락률'].mean():.2f}%")
                c4.metric("공포지수(VIX)", f"{get_vix():.2f}")
                
                # 스타일링
                def color_signal(val):
                    if '강력' in val: color = '#ef4444'
                    elif '매수' in val: color = '#facc15'
                    elif '주의' in val: color = '#3b82f6'
                    else: color = 'white'
                    return f'color: {color}'

                st.dataframe(df_display.style.map(color_signal, subset=['신호'])
                             .format({'현재가': '{:,.0f} 원', '등락률': '{:+.2f}%', 
                                      'RSI': '{:.1f}', 'RVOL': '{:.1f}x'}),
                             use_container_width=True, height=500)
                
                # 차트 펼치기
                with st.expander("📈 상위 종목 차트 확인하기 (Top 5)", expanded=True):
                    cols = st.columns(3) # 3열로 배치
                    for i, (idx, row) in enumerate(df.head(6).iterrows()):
                        with cols[i % 3]:
                            fig = go.Figure()
                            color = '#ef4444' if row['등락률'] > 0 else '#3b82f6'
                            fig.add_trace(go.Scatter(
                                x=row['chart_data'].index, y=row['chart_data'], 
                                mode='lines', line=dict(color=color, width=2)
                            ))
                            fig.update_layout(
                                title=f"{row['종목명']} ({row['등락률']:+.1f}%)",
                                height=200, margin=dict(l=10, r=10, t=30, b=10),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=10),
                                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # CSV 다운로드
                csv = df_display.to_csv(index=False).encode('utf-8-sig')
                st.download_button("⬇️ 엑셀용 CSV 다운로드", csv, 
                                   f"quant_report_{datetime.now().strftime('%H%M%S')}.csv", "text/csv")
            else:
                st.error("❌ 분석된 데이터가 없습니다.\n\n"
                         "1. 티커 형식이 올바른지 확인하세요 (예: 005930.KS)\n"
                         "2. 상장폐지되거나 거래정지된 종목인지 확인하세요.")
