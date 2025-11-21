import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# 約200檔高流動台股清單（示意請自行補充完整）
TW_TOP_200 = [
    '6770','2344','2408','1802','1815','2317','2337','1303','2881','8358','6173','1605','2409','2887','2330','2615','2885','8043','2883','3481','2449','2801','3715','3231','2324','2884','2891','2327','3702','2303','2890','3037','3711','1504','8112','2002','8069','2610','3624','8042','6163','8150','3706','2609','2312','2301','1314','2882','1101','6191','2486','2481','8021','1301','2886','4958','2377','6282','2618','1326','4989','2515','9813','2353','2472','2880','2892','2371','3260','2368','2308','3026','5425','6274','2375','5498','2867','5328','3189','2382','2834','4577','6919','8110','1519','3017','1216','6505','8422','2356','2492','6208','3036','8046','6209','5880','2027','2329','8048','3048','1402','5314','4904','3236','5351','3006','5469','2105','2201','3264','5876','4938','5243','9805','1717','3167','4967','2498','8034','1409','3450','2347','9904','4979','1513','2421','2478','2412','8039','2495','2354','8111','6442','6239','5871','2603','4763','2374','2454','1514','2641','2812','2637','3105'
]

st.set_page_config(page_title="QuantMaster Pro (高流動池+進度條)", layout="wide", page_icon="📈")

class DataManager:
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_price_and_financials(symbol):
        yf_symbol = symbol if symbol.endswith('.TW') else f"{symbol}.TW"
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="6mo")
            fin = ticker.financials.T
            if df.empty or len(df) < 20 or fin.empty:
                return None, None, None
            df = df.reset_index()
            df['Date'] = df['Date'].dt.tz_localize(None)
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA5_Vol'] = df['Volume'].rolling(5).mean()
            low_min = df['Low'].rolling(9).min()
            high_max = df['High'].rolling(9).max()
            df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min) * 100).fillna(50)
            k, d = 50, 50
            k_list, d_list = [], []
            for r in df['RSV']:
                k = (2/3)*k + (1/3)*r
                d = (2/3)*d + (1/3)*k
                k_list.append(k)
                d_list.append(d)
            df['K'] = k_list
            df['D'] = d_list
            try:
                eps = fin.get('Net Income', pd.Series()).tail(3)
                eps_growth = eps.pct_change().mean() * 100 if len(eps) >= 3 else 0
            except:
                eps_growth = 0
            return df, eps_growth, yf_symbol
        except:
            return None, None, None

    def get_news_sentiment(self, yf_symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
            news_list = ticker.news
            if not news_list:
                return "無新聞", 50
            pos_words = ['up', 'rise', 'gain', 'high', 'strong', 'bull', '新高', '上漲', '獲利']
            neg_words = ['down', 'fall', 'loss', 'low', 'weak', 'bear', '新低', '下跌', '虧損']
            score = 50
            for n in news_list[:3]:
                title = n.get('title', '').lower()
                if any(p in title for p in pos_words):
                    score += 10
                if any(n in title for n in neg_words):
                    score -= 10
            return news_list[0].get('title'), min(max(score, 0), 100)
        except:
            return "新聞異常", 50

class DecisionEngine:
    def analyze(self, df, eps_growth, sentiment, w_tech, w_theme, period):
        if df is None or df.empty:
            return None
        curr, prev = df.iloc[-1], df.iloc[-2]
        tech_score = 0
        if period == 'short':
            cross = (curr['MA5'] > curr['MA20']) and (prev['MA5'] <= prev['MA20'])
            if cross:
                tech_score += 70
            if curr['Volume'] > curr['MA5_Vol'] * 1.2:
                tech_score += 30
        elif period == 'mid':
            if curr['Close'] > curr['MA20'] > curr['MA60']:
                tech_score += 40
            vol_ratio = curr['Volume'] / curr['MA5_Vol'] if curr['MA5_Vol'] > 0 else 1
            if vol_ratio > 1.2:
                tech_score += 20
            kd_gold_cross = (curr['K'] > curr['D']) and (prev['K'] < prev['D'])
            if kd_gold_cross:
                tech_score += 40 if curr['K'] < 40 else 20
            death_cross = (curr['K'] < curr['D']) and (prev['K'] > prev['D'])
            if death_cross:
                tech_score -= 20
            tech_score = max(0, min(100, tech_score))
        else:
            if curr['Close'] > curr['MA60']:
                tech_score += 40
            if curr['K'] > curr['D']:
                tech_score += 30
            tech_score += min(max(eps_growth, 0), 30)
            tech_score = min(100, tech_score)
        total = tech_score * w_tech + sentiment * w_theme
        if total >= 80:
            rating = "強力買進"
        elif total >= 60:
            rating = "買進"
        elif total <= 40:
            rating = "賣出"
        else:
            rating = "觀望"
        return {
            'score': round(total, 1),
            'rating': rating,
            'price': curr['Close'],
            'K': curr['K'],
            'D': curr['D'],
            'eps_growth': eps_growth
        }

def recommend(period, w_tech, w_theme):
    dm = DataManager()
    de = DecisionEngine()
    weight_map = {'short': (1.0, 0.0), 'mid': (0.7, 0.3), 'long': (0.4, 0.6)}
    wt, wth = weight_map.get(period, (0.7, 0.3))
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(TW_TOP_200)
    for i, sym in enumerate(TW_TOP_200):
        status.text(f"分析中: {sym} ({i + 1}/{total})")
        progress.progress((i + 1) / total)
        df, eps, yf_sym = dm.fetch_price_and_financials(sym)
        if df is None:
            continue
        _, sent = dm.get_news_sentiment(yf_sym)
        res = de.analyze(df, eps, sent, w_tech * wt, w_theme * wth, period)
        if res and res['score'] > 60:
            results.append((sym, res['score'], res['rating'], res['K'], res['D'], res['eps_growth']))
    status.empty()
    progress.empty()
    return sorted(results, key=lambda x: x[1], reverse=True)[:5]

def main():
    st.sidebar.header("推薦設定與權重調整")
    w_tech = st.sidebar.slider("技術面權重", 0.0, 1.0, 0.7)
    w_theme = st.sidebar.slider("新聞面權重", 0.0, 1.0, 0.3)
    period = st.sidebar.radio("選擇推薦週期", ['short', 'mid', 'long'], index=0)

    st.title("QuantMaster Pro (高流動池+進度條)")
    st.info("短期用5/20日線黃金交叉判斷，中期維持KD+均線，長期加入EPS基本面")

    if st.sidebar.button("開始推薦"):
        recs = recommend(period, w_tech, w_theme)
        st.subheader(f"{period}期推薦")
        for r in recs:
            st.write(f"{r[0]}：分數 {r[1]}，建議 {r[2]}，K={r[3]:.1f} D={r[4]:.1f}，EPS增長={r[5]:.2f}%")

if __name__ == "__main__":
    main()
