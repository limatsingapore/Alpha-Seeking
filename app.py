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

# --- [스타일링: 가독성 강화] ---
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 설정 */
    .main { background-color: #0f172a; color: #f8fafc; }
    
    /* 메트릭 박스 스타일 */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        color: white !important;
    }
    
    /* 메트릭 텍스트 강제 흰색 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }
    
    /* 데이터프레임 스타일 */
    [data-testid="stDataFrame"] { background-color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- [티커-종목명 매핑 사전 (속도 향상용)] ---
TICKER_MAP = {
    # 반도체
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "000990.KS": "DB하이텍",
    "005290.KS": "동진쎄미켐", "036540.KQ": "SFA반도체", "277810.KQ": "천보", "042700.KS": "한미반도체",
    # 2차전지
    "373220.KS": "LG에너지솔루션", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠", "247540.KQ": "에코프로비엠", "066570.KS": "LG전자", "086520.KQ": "에코프로",
    # 화장품/식품
    "090430.KS": "아모레퍼시픽", "002790.KS": "아모레G", "192820.KS": "코스맥스", "000100.KS": "유한양행",
    "097950.KS": "CJ제일제당", "271560.KS": "오리온", "004370.KS": "농심",
    # 자동차
    "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스", "011210.KS": "현대위아",
    # 바이오
    "207940.KS": "삼성바이오로직스", "068270.KS": "셀트리온", "128940.KS": "한미약품", "000250.KS": "삼천당제약",
    # 금융
    "105560.KS": "KB금융", "055550.KS": "신한지주", "316140.KS": "우리금융지주",
    "005940.KS": "NH투자증권", "000370.KS": "한화손해보험",
    # 플랫폼/게임
    "035420.KS": "NAVER", "035720.KS": "카카오", "259960.KS": "크래프톤", "036570.KQ": "엔씨소프트",
    # 중공업/방산
    "042660.KS": "대우조선해양", "010140.KS": "삼성중공업", "011200.KS": "HMM",
    "012450.KS": "한화에어로스페이스", "047810.KS": "한국항공우주", "079550.KS": "LIG넥스원"
}

# --- [섹터 데이터 정의] ---
SECTORS = {
    "🚀 반도체 & IT": {
        "반도체 대장주": ["005930.KS", "000660.KS",
