from prophet import Prophet
import pandas as pd
import io
import streamlit as st
from transformers import pipeline
import torch
import numpy as np
from scipy.optimize import minimize

@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None

@st.cache_resource
def train_prophet_model(df_json: str):
    try:
        pdf = pd.read_json(io.StringIO(df_json), orient="split")
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        m = Prophet(changepoint_prior_scale=0.15, daily_seasonality=False)
        m.fit(pdf.rename(columns={"Date": "ds", "Close": "y"}))
        return m
    except Exception:
        return None

def portfolio_performance(weights, returns):
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return port_return, port_vol

def minimize_sharpe(weights, returns, risk_free_rate):
    ret, vol = portfolio_performance(weights, returns)
    return -(ret - risk_free_rate) / vol

def optimize_portfolio(returns, risk_free_rate, tickers):
    num_assets = len(tickers)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    optimized = minimize(minimize_sharpe, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized.x
