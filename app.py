import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 擴大股票池 (上市櫃熱門股 + 權值股) ---
# 包含台積電、鴻海、聯發科等權值股，以及航運、AI概念、重電等熱門股
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

st.set_page_config(page_title="QuantMaster Pro (KD版)", layout="wide", page_icon="📈")

# --- 資料與指標計算 ---
class DataManager:
    @st.cache_data(ttl=300) # 緩存 5 分鐘
    def fetch_price_data(_self, symbol):
        yf_symbol = symbol if symbol.endswith('.TW') else f"{symbol}.TW"
        try:
            # 下載足夠長度以計算 MA60 與 KD
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="6mo")
            if df.empty or len(df) < 60:
                return None, None
            
            df = df.reset_index()
            df['Date'] = df['Date'].dt.tz_localize(None)
            
            # 基礎均線
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA5_Vol'] = df['Volume'].rolling(5).mean()
            
            # --- KD 指標計算 (9,3,3) ---
            # RSV = (今日收盤 - 最近9天最低) / (最近9天最高 - 最近9天最低) * 100
            # K = 2/3 * 昨日K + 1/3 * RSV
            # D = 2/3 * 昨日D + 1/3 * K
            low_min = df['Low'].rolling(9).min()
            high_max = df['High'].rolling(9).max()
            df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
            df['RSV'] = df['RSV'].fillna(50) # 補值
            
            # 遞迴計算 KD
            k_list, d_list = [], []
            k, d = 50, 50 # 初始值
            for rsv in df['RSV']:
                k = (2/3) * k + (1/3) * rsv
                d = (2/3) * d + (1/3) * k
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
            if not news_list: return "無新聞", 50
            
            pos_words = ['up','rise','gain','high','strong','bull','新高','上漲','獲利','大增','搶眼']
            neg_words = ['down','fall','loss','low','weak','bear','新低','下跌','虧損','衰退','重挫']
            
            score = 50
            for news in news_list[:3]:
                t = news.get('title','').lower()
                if any(w in t for w in pos_words): score += 10
                if any(w in t for w in neg_words): score -= 10
            
            return news_list[0].get('title'), min(max(score, 0), 100)
        except:
            return "新聞異常", 50

# --- 決策引擎 (加入 KD) ---
class DecisionEngine:
    def analyze(self, df, sentiment_score, w_tech, w_theme):
        if df is None or df.empty: return None
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 技術面評分 (滿分 100) ---
        tech_raw = 0
        
        # 1. 均線多頭 (40分)
        if curr['Close'] > curr['MA20'] and curr['MA20'] > curr['MA60']:
            tech_raw += 40
            
        # 2. 量能放大 (20分)
        vol_ratio = curr['Volume'] / curr['MA5_Vol'] if curr['MA5_Vol'] > 0 else 1
        if vol_ratio > 1.2:
            tech_raw += 20
            
        # 3. KD 指標 (40分)
        # 黃金交叉：K > D 且 前一天 K < D
        kd_gold_cross = (curr['K'] > curr['D']) and (prev['K'] < prev['D'])
        # 處於低檔 (K < 40) 更有力
        if kd_gold_cross:
            if curr['K'] < 40:
                tech_raw += 40 # 低檔金叉 (強烈買進)
            else:
                tech_raw += 20 # 一般金叉
        # 死亡交叉扣分
        elif (curr['K'] < curr['D']) and (prev['K'] > prev['D']):
            tech_raw -= 20

        final_tech = max(0, min(100, tech_raw))
        
        # 總分加權
        total_score = final_tech * w_tech + sentiment_score * w_theme
        
        rating = "觀望"
        if total_score >= 80: rating = "強力買進"
        elif total_score >= 60: rating = "買進"
        elif total_score <= 40: rating = "賣出"
        
        return {
            "score": round(total_score, 1),
            "rating": rating,
            "price": curr['Close'],
            "k": curr['K'],
            "d": curr['D'],
            "vol_ratio": vol_ratio
        }

def run_recommendation(period, w_tech, w_theme):
    dm = DataManager()
    de = DecisionEngine()
    
    # 依週期調整內部權重 (長線更看基本面/新聞，短線更看技術/KD)
    if period == 'short':
        w_t, w_th = w_tech * 1.2, w_theme * 0.8 # 短線重技術
    elif period == 'mid':
        w_t, w_th = w_tech, w_theme
    else:
        w_t, w_th = w_tech * 0.7, w_theme * 1.3 # 長線重題材
        
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    total = len(TW_HOT_STOCKS)
    for i, sym in enumerate(TW_HOT_STOCKS):
        status.text(f"正在分析: {sym} ... ({i+1}/{total})")
        progress.progress((i+1)/total)
        
        df, yf_sym = dm.fetch_price_data(sym)
        if df is None: continue
        
        _, sent = dm.get_real_news_sentiment(yf_sym)
        res = de.analyze(df, sent, w_t, w_th)
        
        if res and res['score'] >= 60: # 只列出及格的
            results.append((sym, res['score'], res['rating'], res['k'], res['d']))
            
    status.empty()
    progress.empty()
    return sorted(results, key=lambda x: x[1], reverse=True)[:5] # 取前5名

# --- 主介面 ---
def main():
    st.sidebar.header("🔥 推薦設定")
    w_tech = st.sidebar.slider("技術面 (含KD) 權重", 0.0, 1.0, 0.7)
    w_theme = st.sidebar.slider("新聞面權重", 0.0, 1.0, 0.3)
    
    st.title("QuantMaster Pro (KD 策略版)")
    st.info("已升級：加入 **KD黃金交叉** 判斷，股票池擴大至 **100+ 熱門台股**。")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 短線衝刺 (1-3天)", use_container_width=True):
            recs = run_recommendation('short', w_tech, w_theme)
            st.subheader("短線推薦")
            for r in recs:
                st.success(f"**{r[0]}** | 分數:{r[1]} | {r[2]} | K:{r[3]:.1f}")

    with col2:
        if st.button("📈 波段操作 (數週)", use_container_width=True):
            recs = run_recommendation('mid', w_tech, w_theme)
            st.subheader("波段推薦")
            for r in recs:
                st.info(f"**{r[0]}** | 分數:{r[1]} | {r[2]} | K:{r[3]:.1f}")

    with col3:
        if st.button("💎 長線存股 (1年)", use_container_width=True):
            recs = run_recommendation('long', w_tech, w_theme)
            st.subheader("長線推薦")
            for r in recs:
                st.warning(f"**{r[0]}** | 分數:{r[1]} | {r[2]} | K:{r[3]:.1f}")

    st.markdown("---")
    st.markdown("#### 📊 個股詳細檢測 (含回測)")
    stock = st.text_input("輸入代碼", value="2330")
    
    if st.button("分析個股"):
        dm = DataManager()
        de = DecisionEngine()
        df, _ = dm.fetch_price_data(stock)
        if df is not None:
            res = de.analyze(df, 60, 0.7, 0.3) # 預設參數
            
            k1, k2, k3 = st.columns(3)
            k1.metric("現價", f"{res['price']:.2f}")
            k2.metric("KD值", f"K={res['k']:.1f}, D={res['d']:.1f}")
            k3.metric("評級", res['rating'])
            
            # 畫圖
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                        low=df['Low'], close=df['Close'], name='K線'))
            # 畫KD (副圖概念，這裡簡單畫在一起或分開)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**技術狀態**: KD {'金叉向上' if res['k']>res['d'] else '死叉向下'}")

if __name__ == "__main__":
    main()
