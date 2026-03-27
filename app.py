from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh
from scipy.optimize import minimize
import openai

# --- MODULAR UTILS (INLINE FOR DEPLOYMENT) ---

class FinanceAdvisor:
    @staticmethod
    def get_ai_response(user_input, context=""):
        # This acts as a smart advisor using context
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful Fintech AI. Context: {context}"},
                        {"role": "user", "content": user_input}
                    ]
                )
                return response.choices[0].message.content
            except:
                pass

        # Rule-based fallback
        low_input = user_input.lower()
        if "bullish" in low_input: return "Market shows bullish signals when the 20-day MA stays above the 50-day MA."
        if "risk" in low_input: return "Diversification and monitoring the Sharpe Ratio are key to managing risk."
        return "I can help with stock trends, sentiment, and portfolio optimization. Try asking about a specific stock!"

# Optional ML Imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(
    page_title="FinAI Super App",
    page_icon="🧠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(11, 132, 255, 0.08), transparent 30%),
                    linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
        color: #111827;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- CACHING & DATA ---

@st.cache_resource
def load_sentiment_model():
    if TRANSFORMERS_AVAILABLE:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return None

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    data = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if data.empty: return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data.columns = [str(col).title() for col in data.columns]
    return data

@st.cache_data(ttl=600)
def fetch_news(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        news = t.news
        return [{
            "title": n.get("title", ""),
            "publisher": n.get("publisher", ""),
            "time": datetime.fromtimestamp(n.get("providerPublishTime", 0)).strftime('%Y-%m-%d %H:%M')
        } for n in news[:10]]
    except: return []

# --- FEATURES ---

def run_prediction_ui(ticker, years, pred_days):
    df = fetch_stock_data(ticker, years)
    if df.empty:
        st.error("Invalid Ticker")
        return

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    curr, ma20, ma50 = df["Close"].iloc[-1], df["MA20"].iloc[-1], df["MA50"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"₹{curr:.2f}")
    c2.metric("20-Day MA", f"₹{ma20:.2f}")
    c3.metric("50-Day MA", f"₹{ma50:.2f}")
    c4.metric("Trend", "📈 Uptrend" if ma20 > ma50 else "📉 Downtrend")

    if PROPHET_AVAILABLE:
        m = Prophet(changepoint_prior_scale=0.15, daily_seasonality=False)
        m.fit(df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}))
        future = m.make_future_dataframe(periods=pred_days)
        forecast = m.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="History"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="AI Forecast", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Forecasting disabled: Prophet not installed.")

def run_sentiment_ui():
    ticker = st.text_input("Enter Ticker for News", value="RELIANCE.NS")
    if ticker:
        with st.spinner("Analyzing Sentiment..."):
            news = fetch_news(ticker)
            model = load_sentiment_model()
            results = []
            for n in news:
                label, score = ("Neutral", 0.0)
                if model:
                    res = model(n['title'])[0]
                    label, score = res['label'], res['score']
                results.append({
                    "Headline": n['title'], "Sentiment": label.upper(), "Impact": "📈" if label=="positive" else "📉" if label=="negative" else "➖"
                })

            res_df = pd.DataFrame(results)
            col1, col2 = st.columns([2, 1])
            col1.dataframe(res_df, use_container_width=True)
            if not res_df.empty:
                fig = px.pie(res_df, names='Sentiment', title="Sentiment Distribution")
                col2.plotly_chart(fig, use_container_width=True)

def run_portfolio_ui():
    stocks = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, RELIANCE.NS")
    if st.button("Optimize Portfolio"):
        tickers = [s.strip() for s in stocks.split(",")]
        data = yf.download(tickers, period="1y")['Close'].pct_change().dropna()

        num = len(tickers)
        weights = np.array(num * [1./num])

        ret = np.sum(data.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(data.cov() * 252, weights)))

        st.subheader("Current Portfolio Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Return", f"{ret:.2%}")
        c2.metric("Volatility", f"{vol:.2%}")
        c3.metric("Sharpe Ratio", f"{(ret - 0.06)/vol:.2f}")

def run_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask me anything about stocks..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        resp = FinanceAdvisor.get_ai_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"): st.markdown(resp)

# --- MAIN ---

def main():
    st_autorefresh(interval=30000, key="sync")
    st.sidebar.title("🧠 FinAI Super App")
    st.sidebar.caption("AI-Powered Financial Intelligence")

    menu = ["📈 Stock Prediction", "📰 News Sentiment", "📊 Portfolio Analyzer", "🤖 AI Advisor"]
    selected = st.sidebar.radio("Navigation", menu)
    st.sidebar.divider()
    st.sidebar.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    if selected == "📈 Stock Prediction":
        ticker = st.sidebar.text_input("Symbol", value="INFY.NS")
        run_prediction_ui(ticker, 2, 15)
    elif selected == "📰 News Sentiment": run_sentiment_ui()
    elif selected == "📊 Portfolio Analyzer": run_portfolio_ui()
    elif selected == "🤖 AI Advisor": run_chat_ui()

if __name__ == "__main__":
    main()
