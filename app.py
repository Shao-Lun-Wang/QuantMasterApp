import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

TW_HOT_STOCKS = [
    '2330', '2317', '2454', '2308', '2412', '6505', '2303', '2610', '1301', '2881',
    '2882', '2382', '5880', '1101', '1303', '1326', '1402', '2002', '3008', '2301',
    '1216', '2385', '2886', '2834', '2891', '5871', '2883', '1305', '2305', '2892',
    '2884', '2885', '2890', '2474', '2609', '2337', '2475', '1325', '2006', '2302',
    '2357', '3045', '2408', '3545', '2409', '4938', '3044', '3514', '2897', '2451',
    '2603', '2615', '2618', '3037', '3034', '3231', '2356', '2376', '2388', '3017',
    '6669', '3443', '3661', '3529', '5269', '6415', '6756', '8069', '8299', '9910',
    '9958', '1513', '1519', '1504', '1605', '1722', '1708', '2059', '2345', '2368',
    '2449', '3035', '3189', '3324', '3711', '4919', '4958', '4966', '5347', '5483',
    '6147', '6182', '6213', '6278', '6488', '8046', '8081', '8454', '9921', '9904'
]

st.set_page_config(page_title="QuantMaster Pro (KD + 動態回測)", layout="wide", page_icon="📈")

class DataManager:
    @staticmethod
    @st.cache_data(ttl=300)  # 確保此函式只接收純字串參數
    def fetch_price_data(symbol):
        yf_symbol = symbol if symbol.endswith('.TW') else f"{symbol}.TW"
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="6mo")
            if df.empty or len(df) < 60:
                return None, None
            df = df.reset_index()
            df['Date'] = df['Date'].dt.tz_localize(None)
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA5_Vol'] = df['Volume'].rolling(5).mean()

            low_min = df['Low'].rolling(9).min()
            high_max = df['High'].rolling(9).max()
            df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
            df['RSV'] = df['RSV'].fillna(50)
            k, d = 50, 50
            k_list, d_list = [], []
            for rsv in df['RSV']:
                k = (2 / 3) * k + (1 / 3) * rsv
                d = (2 / 3) * d + (1 / 3) * k
                k
