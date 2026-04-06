import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    try:
        data = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data.columns = [str(col).title() for col in data.columns]
        return data
    except: return pd.DataFrame()

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

@st.cache_data(ttl=300)
def get_portfolio_data(tickers, start_date):
    try:
        data = yf.download(tickers, start=start_date)['Close']
        if isinstance(data, pd.Series): data = data.to_frame()
        return data.pct_change().dropna()
    except: return pd.DataFrame()
