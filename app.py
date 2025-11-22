import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import numpy as np
import random
import requests
import json

# ==========================================
# 1. Streamlit 設定與 Dark Mode 風格
# ==========================================

st.set_page_config(
    page_title="AI Stock Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 強制深色金融終端機風格
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.5rem;
    }
    .card {
        background-color: #1e232a;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .sub-text { color: #8b949e; font-size: 0.9rem; }
    .stProgress > div > div > div > div {
        background-color: #00ff7f;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 股票分析後端類別
# ==========================================

class StockAnalyzer:
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.df = None
        self.info = {}

    def fetch_data(self, period: str = "1y") -> bool:
        try:
            self.df = self.ticker.history(period=period)
            if self.df.empty:
                return False
            try:
                self.info = self.ticker.info
            except Exception:
                self.info = {}
            return True
        except Exception as e:
            st.error(f"數據獲取失敗: {e}")
            return False

    def calculate_technicals(self):
        if self.df is None or self.df.empty:
            return
        # RSI
        self.df.ta.rsi(length=14, append=True)
        # MACD
        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
        # Bollinger Bands
        self.df.ta.bbands(length=20, std=2, append=True)
        # SMA
        self.df["SMA_20"] = ta.sma(self.df["Close"], length=20)
        self.df["SMA_60"] = ta.sma(self.df["Close"], length=60)
        self.df.dropna(inplace=True)

    def generate_signals(self):
        if self.df is None or len(self.df) < 2:
            return []

        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]
        signals = []

        # 黃金/死亡交叉
        if prev_row["SMA_20"] < prev_row["SMA_60"] and last_row["SMA_20"] > last_row["SMA_60"]:
            signals.append("🔥 [translate:黃金交叉] (Bullish)")
        elif prev_row["SMA_20"] > prev_row["SMA_60"] and last_row["SMA_20"] < last_row["SMA_60"]:
            signals.append("❄️ [translate:死亡交叉] (Bearish)")

        # RSI
        if last_row["RSI_14"] < 30:
            signals.append("🟢 RSI [translate:超賣] (Oversold)")
        elif last_row["RSI_14"] > 70:
            signals.append("🔴 RSI [translate:超買] (Overbought)")

        # 布林通道
        if last_row["Close"] < last_row["BBL_20_2.0"]:
            signals.append("🟢 [translate:跌破下軌] (Potential Rebound)")
        elif last_row["Close"] > last_row["BBU_20_2.0"]:
            signals.append("🔴 [translate:突破上軌] (Overextended)")

        return signals

# ==========================================
# 3. 消息面：OpenAI 情緒分析 (選填)
# ==========================================

def analyze_sentiment_with_openai(ticker_symbol: str, api_key: str = None):
    # 1. 嘗試抓取新聞
    news_list = []
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if news:
            news_list = [n.get("title", "") for n in news[:5]]
    except Exception:
        pass

    # 沒抓到就用模擬標題
    if not news_list:
        news_list = [
            f"{ticker_symbol} beats earnings expectations by 15%",
            f"Market volatility affects {ticker_symbol} short-term outlook",
            f"Analysts upgrade {ticker_symbol} following product launch",
            f"Supply chain issues may impact {ticker_symbol} Q4 results",
        ]

    # 2. 有 OpenAI Key 才分析，否則 Mock
    if api_key:
        try:
            joined_titles = "\n".join(f"- {t}" for t in news_list)
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""
            You are a senior equity analyst. Analyze the sentiment of the following news headlines for {ticker_symbol}:
            {joined_titles}
            Return one line only in the format: score|short summary
            where score is -1 to 1.
            """
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content.strip()
            parts = content.split("|", 1)
            score = float(parts[0].strip())
            summary = parts[1].strip() if len(parts) > 1 else "Sentiment analyzed."
            return score, summary, news_list
        except Exception as e:
            st.sidebar.error(f"OpenAI Error: {e} (Switching to Demo Mode)")

    mock_score = round(random.uniform(-0.5, 0.8), 2)
    mock_summary = "Market sentiment is mixed with both opportunities and risks."
    return mock_score, mock_summary, news_list

# ==========================================
# 4. 基本面：Perplexity API 分析 (KEY 已填入)
# ==========================================

# [KEY 設定區] 您的 Perplexity API Key 填在這裡
DEFAULT_PPLX_KEY = "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"

def get_fundamental_target_with_perplexity(ticker_symbol: str, pplx_key: str = None, current_price: float = None):
    """
    利用 Perplexity 搜尋最新分析師目標價與共識
    """
    if current_price is None:
        try:
            current_price = yf.Ticker(ticker_symbol).history(period="1d")["Close"].iloc[-1]
        except:
            current_price = 100.0

    # 優先使用傳入的 Key，若無則使用預設 Key
    key_to_use = pplx_key if pplx_key else DEFAULT_PPLX_KEY

    # 若完全無 Key，回退至 Mock
    if not key_to_use:
        return _mock_fundamental(current_price)

    try:
        # 這裡使用 Perplexity 的 Base URL
        client = openai.OpenAI(
            api_key=key_to_use,
            base_url="https://api.perplexity.ai",
        )

        system_prompt = (
            "You are a professional equity research analyst. "
            "Search the latest web news and broker reports. "
            "Extract the LATEST consensus analyst 12-month target price, rating consensus, and a summary. "
            "Respond ONLY in valid JSON."
        )

        user_prompt = f"""
        Stock: {ticker_symbol}
        Current Price: {current_price}

        Please search for the latest analyst target price and consensus.
        Return JSON format:
        {{
          "target_price": (number),
          "consensus": "Buy/Hold/Sell",
          "summary": "Traditional Chinese summary (max 50 words)"
        }}
        """

        # 調用 sonar-pro 模型
        completion = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        raw = completion.choices[0].message.content.strip()
        
        # 清理 JSON 格式 (有時候模型會包 ```
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"): raw = raw[4:]
        
        data = json.loads(raw)

        target_price = float(data.get("target_price", current_price))
        consensus = data.get("consensus", "Hold")
        summary = data.get("summary", "目前市場缺乏明確共識。")
        
        upside_pct = (target_price - current_price) / current_price * 100.0
        return target_price, upside_pct, consensus, summary, False

    except Exception as e:
        st.sidebar.error(f"Perplexity API Error: {e}")
        return _mock_fundamental(current_price)

def _mock_fundamental(current_price):
    """模擬數據 (當 API 失敗時使用)"""
    target_price = round(current_price * random.uniform(1.05, 1.25), 2)
    upside_pct = (target_price - current_price) / current_price * 100.0
    return target_price, upside_pct, "Buy", "Mock Data: Analysts apply cautious optimism.", True

# ==========================================
# 5. 綜合評分計算
# ==========================================

def calculate_confidence(tech_signals, sentiment_score, upside_pct):
    base = 50
    tech_score = 0
    for sig in tech_signals:
        if any(x in sig for x in ["Bullish", "Oversold", "Potential Rebound"]): tech_score += 20
        if any(x in sig for x in ["Bearish", "Overbought", "Overextended"]): tech_score -= 20
    
    tech_norm = max(0, min(100, base + tech_score))
    sent_norm = (sentiment_score + 1) * 50
    fund_norm = max(0, min(100, ((upside_pct + 10) / 40) * 100)) # -10% ~ 30% range
    
    return int(tech_norm * 0.4 + sent_norm * 0.3 + fund_norm * 0.3)

# ==========================================
# 6. 繪圖函數
# ==========================================

def plot_chart(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="OHLC", increasing_line_color="#00ff7f", decreasing_line_color="#ff4b4b"
    ), row=1, col=1)
    
    # 均線與布林
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='cyan', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='BB Up', line=dict(color='gray', dash='dot', width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='BB Low', line=dict(color='gray', dash='dot', width=0.5), fill='tonexty'), row=1, col=1)

    # MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Hist', marker_color='gray'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD', line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal', line=dict(color='orange')), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} Technical Chart", template="plotly_dark", xaxis_rangeslider_visible=False,
        height=600, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 7. 主程式介面
# ==========================================

def main():
    with st.sidebar:
        st.title("🤖 AI Quant Pro")
        ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
        period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
        
        st.markdown("---")
        st.caption("🔑 API Settings")
        
        # OpenAI Key (Optional)
        openai_key = st.text_input("OpenAI API Key", type="password", help="Optional for sentiment analysis")
        
        # Perplexity Key (UI 預設值會讀取上面的變數)
        pplx_key = st.text_input("Perplexity API Key", value=DEFAULT_PPLX_KEY, type="password")
        
        st.markdown("---")
        run_btn = st.button("🚀 Run Analysis", type="primary")

    if run_btn:
        analyzer = StockAnalyzer(ticker_input)
        
        with st.spinner(f"Fetching Data for {ticker_input}..."):
            if not analyzer.fetch_data(period):
                return

        # 1. 技術分析
        analyzer.calculate_technicals()
        signals = analyzer.generate_signals()
        
        # 2. 價格數據
        curr_price = analyzer.df['Close'].iloc[-1]
        pct_chg = ((curr_price - analyzer.df['Close'].iloc[-2]) / analyzer.df['Close'].iloc[-2]) * 100
        
        # 3. AI 分析
        with st.spinner("AI Analyzing Sentiment & Targets..."):
            # 新聞情緒
            sent_score, sent_summary, headlines = analyze_sentiment_with_openai(ticker_input, openai_key)
            
            # 基本面目標價 (使用 Perplexity)
            target, upside, consensus, target_sum, is_mock = get_fundamental_target_with_perplexity(
                ticker_input, pplx_key, curr_price
            )
            
        # 4. 綜合評分
        conf_score = calculate_confidence(signals, sent_score, upside)

        # --- UI 顯示 ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${curr_price:.2f}", f"{pct_chg:.2f}%")
        col2.metric("Sentiment Score", f"{sent_score:.2f}")
        col3.metric("Target Upside", f"{upside:.1f}%", f"Target: ${target}")
        col4.metric("Confidence", f"{conf_score}/100")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(plot_chart(analyzer.df, ticker_input), use_container_width=True)
        with c2:
            rec_color = "#00ff7f" if conf_score >= 60 else "#ff4b4b"
            rec_text = "BUY" if conf_score >= 60 else "SELL/HOLD"
            
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {rec_color};">
                <h2 style="color:{rec_color}; margin:0;">{rec_text}</h2>
                <p class="sub-text">Score: {conf_score}</p>
                <hr style="border-color: #333;">
                <p><strong>Analyst Consensus:</strong> {consensus}</p>
                <p style="font-size:0.9rem">{target_sum}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Recent News"):
                for h in headlines: st.write(f"- {h}")
            
            if is_mock: st.caption("⚠️ Using Mock Data (API Error)")

        st.dataframe(analyzer.df.tail(5)[['Close', 'RSI_14', 'SMA_20', 'MACD_12_26_9']])

if __name__ == "__main__":
    main()
