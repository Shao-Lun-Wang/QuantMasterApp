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
                k_list.append(k)
                d_list.append(d)
            df['K'] = k_list
            df['D'] = d_list
            return df, yf_symbol
        except:
            return None, None

    def get_real_news_sentiment(self, yf_symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
            news_list = ticker.news
            if not news_list:
                return "無新聞", 50
            pos_words = ['up', 'rise', 'gain', 'high', 'strong', 'bull', '新高', '上漲', '獲利', '大增', '搶眼']
            neg_words = ['down', 'fall', 'loss', 'low', 'weak', 'bear', '新低', '下跌', '虧損', '衰退', '重挫']
            score = 50
            for news in news_list[:3]:
                t = news.get('title', '').lower()
                if any(p in t for p in pos_words):
                    score += 10
                if any(n in t for n in neg_words):
                    score -= 10
            return news_list[0].get('title'), min(max(score, 0), 100)
        except:
            return "新聞異常", 50

class DecisionEngine:
    def analyze(self, df, sentiment_score, w_tech, w_theme):
        if df is None or df.empty:
            return None
        curr, prev = df.iloc[-1], df.iloc[-2]
        tech_score = 0
        if curr['Close'] > curr['MA20'] and curr['MA20'] > curr['MA60']:
            tech_score += 40
        vol_ratio = curr['Volume'] / curr['MA5_Vol'] if curr['MA5_Vol'] > 0 else 1
        if vol_ratio > 1.2:
            tech_score += 20
        k_cross = (curr['K'] > curr['D']) and (prev['K'] < prev['D'])
        if k_cross:
            tech_score += 40 if curr['K'] < 40 else 20
        if (curr['K'] < curr['D']) and (prev['K'] > prev['D']):
            tech_score -= 20
        tech_score = max(0, min(100, tech_score))
        total_score = tech_score * w_tech + sentiment_score * w_theme
        if total_score >= 80:
            rating = "強力買進"
        elif total_score >= 60:
            rating = "買進"
        elif total_score <= 40:
            rating = "賣出"
        else:
            rating = "觀望"
        return {'score': round(total_score, 1), 'rating': rating, 'price': curr['Close'], 'K': curr['K'], 'D': curr['D'], 'vol_ratio': vol_ratio}

def recommend_stocks(period, w_tech, w_theme):
    dm = DataManager()
    de = DecisionEngine()
    if period == 'short':
        w_t, w_th = w_tech * 1.2, w_theme * 0.8
    elif period == 'mid':
        w_t, w_th = w_tech, w_theme
    else:
        w_t, w_th = w_tech * 0.7, w_theme * 1.3
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(TW_HOT_STOCKS)
    for i, sym in enumerate(TW_HOT_STOCKS):
        status.text(f"分析中: {sym} ({i+1}/{total})")
        progress.progress((i+1) / total)
        df, yf_sym = DataManager.fetch_price_data(sym)
        if df is None:
            continue
        _, sent = dm.get_real_news_sentiment(yf_sym)
        res = de.analyze(df, sent, w_t, w_th)
        if res and res['score'] >= 60:
            results.append((sym, res['score'], res['rating'], res['K'], res['D']))
    status.empty()
    progress.empty()
    return sorted(results, key=lambda x: x[1], reverse=True)[:5]

def backtest(stock, days=30):
    dm = DataManager()
    de = DecisionEngine()
    df, yf_sym = DataManager.fetch_price_data(stock)
    if df is None or len(df) < days:
        return None
    df_bt = df.tail(days)
    sent_title, sent_score = dm.get_real_news_sentiment(yf_sym)
    start_price = df_bt['Close'].iloc[0]
    end_price = df_bt['Close'].iloc[-1]
    ret_pct = (end_price - start_price) / start_price * 100
    return round(ret_pct, 2)

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

def add_position():
    pos = {'symbol': st.session_state.pos_symbol, 'buy_price': st.session_state.pos_price, 'shares': st.session_state.pos_shares,
           'date': st.session_state.pos_date}
    st.session_state.portfolio.append(pos)

def show_portfolio():
    dm = DataManager()
    if len(st.session_state.portfolio) == 0:
        st.info("尚無持倉資料")
        return
    st.write("### 持倉清單")
    total_val = 0
    for p in st.session_state.portfolio:
        df, _ = DataManager.fetch_price_data(p['symbol'])
        if df is None:
            st.write(f"{p['symbol']} 無法取得價格")
            continue
        current_price = df['Close'].iloc[-1]
        change_pct = (current_price - p['buy_price']) / p['buy_price'] * 100
        val = current_price * p['shares']
        total_val += val
        st.write(
            f"{p['symbol']} 買價:{p['buy_price']} 現價:{current_price:.2f} 漲跌:{change_pct:.2f}% 股數:{p['shares']} 持倉價值:{val:.0f}")

    st.write(f"#### 持倉總市值: {total_val:.0f} 元")

def main():
    st.sidebar.header("推薦參數調整")
    w_tech = st.sidebar.slider("技術面 (含KD) 權重", 0.0, 1.0, 0.7)
    w_theme = st.sidebar.slider("新聞面權重", 0.0, 1.0, 0.3)

    st.title("QuantMaster Pro (KD + 動態回測) 修正版")
    st.info("加入 @staticmethod 修正快取錯誤")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("短期推薦 (1~3天)"):
            recs = recommend_stocks('short', w_tech, w_theme)
            st.subheader("短期推薦股票")
            for r in recs:
                st.success(f"{r[0]} 分數:{r[1]} 評級:{r[2]} K值:{r[3]:.1f} D值:{r[4]:.1f}")

    with col2:
        if st.button("中期推薦 (數週)"):
            recs = recommend_stocks('mid', w_tech, w_theme)
            st.subheader("中期推薦股票")
            for r in recs:
                st.info(f"{r[0]} 分數:{r[1]} 評級:{r[2]} K值:{r[3]:.1f} D值:{r[4]:.1f}")

    with col3:
        if st.button("長期推薦 (1年)"):
            recs = recommend_stocks('long', w_tech, w_theme)
            st.subheader("長期推薦股票")
            for r in recs:
                st.warning(f"{r[0]} 分數:{r[1]} 評級:{r[2]} K值:{r[3]:.1f} D值:{r[4]:.1f}")

    st.markdown("---")
    st.markdown("### 個股回測 (單檔)")
    stock = st.text_input("輸入股票代碼回測", value="2330")
    bt_days = st.number_input("回測天數", min_value=10, max_value=365, value=30, step=1)
    if st.button("開始回測"):
        ret = backtest(stock, bt_days)
        if ret is None:
            st.error("沒有足夠資料作回測")
        else:
            st.success(f"{stock} {bt_days}日回測收益率：{ret}%")

    st.markdown("---")
    st.markdown("### 持倉管理")
    with st.form("pos_form"):
        st.text_input("股票代碼", key="pos_symbol")
        st.number_input("買入價格", min_value=0.01, max_value=1000000.0, step=0.01, key="pos_price")
        st.number_input("持股數量", min_value=1, max_value=1000000, step=1, key="pos_shares")
        st.date_input("交易日期", key="pos_date")
        submitted = st.form_submit_button("新增持倉")
        if submitted:
            add_position()
            st.success("持倉新增成功")

    show_portfolio()

if __name__ == "__main__":
    main()
