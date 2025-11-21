import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

TW50_STOCKS = ['2330', '2317', '2454', '2308', '2412', '6505', '2303', '2610', '1301', '2881',
               '2882', '2382', '5880', '1101', '1303', '1326', '1402', '2002', '3008', '2301',
               '1216', '2385', '2886', '2834', '2891', '5871', '2883', '1305', '2305', '2892',
               '2884', '2885', '2890', '2474', '2609', '2337', '2475', '1325', '2006', '2302',
               '2357', '3045', '2408', '3545', '2409', '4938', '3044', '3514', '2897', '2451']

# --- 資料管理 ---
class DataManager:
    @st.cache_data(ttl=60)
    def fetch_price_data(_self, symbol):
        yf_symbol = symbol if symbol.endswith('.TW') else f"{symbol}.TW"
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="1y")
            if df.empty:
                return None, None
            df = df.reset_index()
            df['Date'] = df['Date'].dt.tz_localize(None)
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA5_Vol'] = df['Volume'].rolling(5).mean()
            return df, yf_symbol
        except:
            return None, None

    def get_real_news_sentiment(self, yf_symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
            news_list = ticker.news
            if not news_list:
                return "無近期新聞", 50
            pos_list = ['up', 'rise', 'gain', 'profit', 'buy', 'strong', 'growth', 'bull', '新高', '上漲', '獲利']
            neg_list = ['down', 'fall', 'loss', 'weak', 'risk', 'sell', '下跌', '虧損', '新低']
            score = 50
            for news in news_list[:5]:
                title = news.get('title','').lower()
                score += 5 if any(p in title for p in pos_list) else 0
                score -= 5 if any(n in title for n in neg_list) else 0
            score = min(max(score, 0), 100)
            return news_list[0].get('title'), score
        except:
            return "新聞錯誤", 50

# --- 推薦引擎 ---
class DecisionEngine:
    def analyze(self, df, sentiment_score, w_tech, w_theme):
        if df is None or df.empty:
            return None
        current_price = df['Close'].iloc[-1]
        if len(df) < 60:
            ma20 = ma60 = current_price
            vol_ratio = 1.0
        else:
            ma20 = df['MA20'].iloc[-1]
            ma60 = df['MA60'].iloc[-1]
            vol_ma5 = df['MA5_Vol'].iloc[-1]
            vol_ratio = df['Volume'].iloc[-1] / vol_ma5 if vol_ma5 else 1.0
        tech_score = (current_price > ma20)*40 + (ma20 > ma60)*30 + (vol_ratio > 1.2)*30
        weighted_tech = tech_score * w_tech
        weighted_theme = sentiment_score * w_theme
        total_score = weighted_tech + weighted_theme
        rating = "觀望"
        if total_score>=75: rating="積極買進"
        elif total_score>=60: rating="買進"
        elif total_score<40: rating="賣出"
        return {"score":round(total_score,1), "rating":rating, "price":current_price, "ma20":ma20, "ma60":ma60, "vol_ratio":vol_ratio}

def recommend_stocks(period, w_tech, w_theme):
    dm = DataManager()
    de = DecisionEngine()
    pool = TW50_STOCKS
    results = []
    w_map = {'short':(0.8,0.2), 'mid':(0.5,0.5), 'long':(0.3,0.7)}
    w1, w2 = w_map.get(period, (0.5,0.5))
    w_tech_p = w_tech * w1
    w_theme_p = w_theme * w2
    for sym in pool:
        df, yf_sym = dm.fetch_price_data(sym)
        if df is None: continue
        _, sent_score = dm.get_real_news_sentiment(yf_sym)
        res = de.analyze(df, sent_score, w_tech_p, w_theme_p)
        if res:
            results.append((sym, res['score'], res['rating']))
    return sorted(results, key=lambda x: x[1], reverse=True)[:3]

# --- 回測功能 ---
def backtest(stock, days=90):
    dm = DataManager()
    de = DecisionEngine()
    df, yf_sym = dm.fetch_price_data(stock)
    if df is None or len(df)<days: return None
    df_bt = df.tail(days)
    sent_title, sent_score = dm.get_real_news_sentiment(yf_sym)
    # 用簡化假設：整期內每日當日價及均線打分，計算買入持有報酬率
    daily_scores = []
    for i in range(len(df_bt)):
        sub_df = df_bt.iloc[:i+1] if i>0 else df_bt.iloc[:1]
        score = de.analyze(sub_df, sent_score, 0.6, 0.4)
        if score:
            daily_scores.append(score['score'])
    start_price = df_bt['Close'].iloc[0]
    end_price = df_bt['Close'].iloc[-1]
    returns_pct = (end_price - start_price)/start_price*100
    return round(returns_pct,2), daily_scores

# --- 持倉紀錄 ---

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

def add_position():
    pos = {
        'symbol': st.session_state.pos_symbol,
        'buy_price': st.session_state.pos_price,
        'shares': st.session_state.pos_shares,
        'date': st.session_state.pos_date
    }
    st.session_state.portfolio.append(pos)

def show_portfolio():
    dm = DataManager()
    if not st.session_state.portfolio:
        st.info("尚無持倉紀錄。請輸入股票、買入價格與數量。")
        return
    st.write("### 持倉檢視")
    total_val = 0
    for pos in st.session_state.portfolio:
        df, yf_sym = dm.fetch_price_data(pos['symbol'])
        if df is None:
            st.write(f"{pos['symbol']}: 無法獲取價格")
            continue
        current_price = df['Close'].iloc[-1]
        change_pct = (current_price - pos['buy_price']) / pos['buy_price'] * 100
        val = current_price * pos['shares']
        total_val += val
        st.write(f"{pos['symbol']} - 買價: {pos['buy_price']} - 現價: {current_price:.2f} - 漲跌: {change_pct:.2f}% - 持股: {pos['shares']} 股 - 持倉價值: {val:.0f}")

    st.write(f"#### 持倉總市值: {total_val:.0f} 元")

def main():
    st.sidebar.header("⚙️ 權重調整與推薦")
    w_tech = st.sidebar.slider("技術面權重", 0.0,1.0,0.7,0.05)
    w_theme = st.sidebar.slider("新聞面權重", 0.0,1.0,0.3,0.05)
    st.sidebar.markdown("---")

    if st.sidebar.button("短期推薦(1~3天)"):
        recs = recommend_stocks('short', w_tech, w_theme)
        st.sidebar.markdown("### 短期推薦")
        for r in recs: st.sidebar.write(f"{r[0]} 分數:{r[1]} 建議:{r[2]}")

    if st.sidebar.button("中期推薦(數月)"):
        recs = recommend_stocks('mid', w_tech, w_theme)
        st.sidebar.markdown("### 中期推薦")
        for r in recs: st.sidebar.write(f"{r[0]} 分數:{r[1]} 建議:{r[2]}")

    if st.sidebar.button("長期推薦(1年)"):
        recs = recommend_stocks('long', w_tech, w_theme)
        st.sidebar.markdown("### 長期推薦")
        for r in recs: st.sidebar.write(f"{r[0]} 分數:{r[1]} 建議:{r[2]}")

    st.title("QuantMaster 多功能投資管理")
    st.markdown("## 持倉紀錄與盈虧計算")
    with st.form("pos_form"):
        st.text_input("股票代碼", key="pos_symbol")
        st.number_input("買入價格", 0.01, 1000000.0, step=0.01, key="pos_price")
        st.number_input("持股數量", 1, 1000000, step=1, key="pos_shares")
        st.date_input("買入日期", key="pos_date")
        submitted = st.form_submit_button("新增持倉")
        if submitted:
            add_position()
            st.success("新增持倉成功！")

    show_portfolio()

    st.markdown("---")
    st.markdown("## 簡單回測範例 (選一檔股檢視近90日報酬)")
    stock_for_bt = st.text_input("輸入股票代碼進行回測", value="2330")
    if st.button("開始回測"):
        result = backtest(stock_for_bt)
        if result is None:
            st.error("找不到資料或資料不足")
        else:
            st.success(f"{stock_for_bt} 近90日回測總報酬：{result[0]}%")

if __name__=="__main__":
    main()
