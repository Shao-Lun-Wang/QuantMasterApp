import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# 約200檔高成交量台股代碼示意，請按需替換或擴充
TW_TOP_200 = [
    '0050','0051','0053','0055','0056','0057','0061','006201','006203','006204','006205','006206','006207','006208','00625K','00631L','00632R','00633L','00634R','00635U','00636','00636K','00637L','00638R','00639','00640L','00641R','00642U','00643','00643K','00645','00646','00647L','00648R','00650L','00651R','00652','00653L','00654R','00655L','00656R','00657','00657K','00660','00661','00662','00663L','00664R','00665L','00666R','00668','00668K','00669R','00670L','00671R','00673R','00674R','00675L','00676R','00678','00679B','00680L','00681R','00682U','00683L','00684R','00685L','00686R','00687B','00687C','00688L','00689R','00690','00692','00693U','00694B','00695B','00696B','00697B','00700','00701','00702','00703','00706L','00707R','00708L','00709','00710B','00711B','00712','00713','00714','00715L','00717','00719B','00720B','00722B','00723B','00724B','00725B','00726B','00727B','00728','00730','00731','00733','00734B','00735','00736','00737','00738U','00739','00740B','00741B','00746B','00749B','00750B','00751B','00752','00753L','00754B','00755B','00756B','00757','00758B','00759B','00760B','00761B','00762','00763U','00764B','00768B','00770','00771','00772B','00773B','00775B','00777B','00778B','00779B','00780B','00781B','00782B','00783','00785B','00786B','00787B','00788B','00789B','00791B','00792B','00793B','00795B','00799B','00830','00834B','00836B','00840B','00841B','00842B','00844B','00845B','00846B','00847B','00848B','00849B','00850','00851','00852L','00853B','00856B','00857B','00858','00859B','00860B','00861','00862B','00863B','00864B','00865B','00867B','00870B','00875','00876','00877','00878','00881','00882','00883B','00884B','00885','00886','00887','00888','00890B','00891','00892','00893','00894','00895','00896','00897','00898','00899','00900','00901','00902','00903','00904','00905','00907','00908','00909','00910','00911','00912','00913','00915','00916','00917','00918','00919','00920','00921','00922','00923','00924','00926','00927','00928','00929','00930','00931B','00932','00933B','00934','00935','00936','00937B','00938','00939','00940','00941','00942B','00943','00944','00945B','00946','00947','00948B','00949','00950B','00951','00952','00953B','00954','00955','00956','00957B','00958B','00959B','00960','00961','00962','00963','00964','00965','00966B','00967B','00968B','00969B','00970B','00971','00972','009800','009801','009802','009803','009804','009805','009806','009807','009808','009809','00980A','00980B','00980D','00980T','009810','009811','009812','009813','00981A','00981B','00981D','00981T','00982A','00982B','00982D','00983A'

]

st.set_page_config(page_title="QuantMaster Pro (高流動池版)", layout="wide", page_icon="📈")

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
            # 短期用5日線與20日線交叉判斷，且量能放大
            cross = (curr['MA5'] > curr['MA20']) and (prev['MA5'] <= prev['MA20'])
            if cross:
                tech_score += 70
            if curr['Volume'] > curr['MA5_Vol'] * 1.2:
                tech_score += 30
        elif period == 'mid':
            # 中期技術面 (收盤 > MA20 > MA60) + KD黃金交叉
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
            # 長期：技術面重季線趨勢＋KD + 基本面EPS增長
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

    st.title("QuantMaster Pro (高流動池+基本面長期)")
    st.info(
        "短期以5/20日線黃金交叉為主，中期維持KD與均線判斷，長期加入基本面EPS增長判斷")

    c1, c2, c3 = st.columns(3)
    period_list = ['short', 'mid', 'long']
    labels = ['短期 (1~3天)', '中期 (數週)', '長期 (1年)']

    for i, p in enumerate(period_list):
        with [c1, c2, c3][i]:
            if st.button(labels[i]):
                recs = recommend(p, w_tech, w_theme)
                st.subheader(f"{labels[i]}推薦")
                for r in recs:
                    st.write(f"{r[0]}：分數 {r[1]}，建議 {r[2]}，K={r[3]:.1f} D={r[4]:.1f}，EPS增長={r[5]:.2f}%")

if __name__ == "__main__":
    main()
