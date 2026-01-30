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
        "소부장 (장비/소재)": ["005290.KS", "036540.KQ", "277810.KQ", "042700.KS"]
    },
    "🔋 2차전지": {
        "배터리 셀": ["373220.KS", "006400.KS", "051910.KS"],
        "양극재/음극재": ["003670.KS", "247540.KQ", "066570.KS", "086520.KQ"]
    },
    "💄 소비재 (화장품/식품)": {
        "화장품 & 뷰티": ["090430.KS", "002790.KS", "247540.KQ", "192820.KS", "000100.KS"],
        "식음료": ["097950.KS", "271560.KS", "004370.KS"]
    },
    "🚗 자동차 & 운송": {
        "완성차": ["005380.KS", "000270.KS"],
        "자동차 부품": ["012330.KS", "011210
