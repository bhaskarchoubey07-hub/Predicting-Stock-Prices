from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from prophet import Prophet
from streamlit_autorefresh import st_autorefresh

try:
    import openai
except Exception:
    openai = None

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


st.set_page_config(
    page_title="FinAI Super App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


REFRESH_INTERVAL_MS = 60_000
DATA_CACHE_TTL = 55
MASTER_CACHE_TTL = 24 * 60 * 60
NSE_MASTER_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
BSE_MASTER_URL = "https://api.bseindia.com/BseIndiaAPI/api/LitsOfScripCSVDownload/w?Scripcode=&Group=&status=&segment="
BUNDLED_COMPANIES = [
    ("Infosys", "INFY.NS"),
    ("Reliance Industries", "RELIANCE.NS"),
    ("Tata Consultancy Services", "TCS.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("Larsen & Toubro", "LT.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("ITC", "ITC.NS"),
]


CUSTOM_CSS = """
<style>
    :root {
        --ink: #14213d;
        --muted: #5b6577;
        --accent: #0b6e4f;
        --accent-warm: #d97706;
        --accent-pop: #7c3aed;
        --danger: #be123c;
        --card: rgba(255,255,255,0.80);
        --border: rgba(20,33,61,0.08);
        --shadow: 0 22px 60px rgba(20,33,61,0.10);
    }
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(16px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes softFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes gradientDrift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(11,110,79,0.18), transparent 26%),
            radial-gradient(circle at 100% 12%, rgba(217,119,6,0.16), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(124,58,237,0.10), transparent 28%),
            linear-gradient(180deg, #f7f2e9 0%, #f2f6f3 54%, #eef3fa 100%);
        color: var(--ink);
        background-size: 120% 120%;
        animation: gradientDrift 22s ease-in-out infinite;
    }
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, rgba(20,33,61,0.98) 0%, rgba(16,24,40,0.96) 100%);
        color: #f8fafc;
        border-right: 1px solid rgba(255,255,255,0.08);
        animation: fadeSlideUp 0.7s ease-out;
    }
    .stSidebar label, .stSidebar p, .stSidebar div[data-testid="stMarkdownContainer"] {
        color: #e5edf8;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.6rem 1.8rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(20,33,61,0.96) 0%, rgba(11,110,79,0.92) 56%, rgba(217,119,6,0.88) 100%);
        color: #fff7ed;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        animation: fadeSlideUp 0.75s ease-out;
    }
    .hero:before {
        content: "";
        position: absolute;
        inset: auto -8% -45% auto;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        animation: softFloat 6s ease-in-out infinite;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
        line-height: 1.02;
        letter-spacing: -0.04em;
        font-family: Georgia, "Palatino Linotype", serif;
    }
    .hero p {
        margin: 0.7rem 0 0;
        max-width: 780px;
        color: rgba(255,247,237,0.90);
        font-size: 1rem;
    }
    .hero-strip {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .hero-pill, .subtle-chip {
        display: inline-block;
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .hero-pill {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        color: #fffaf2;
    }
    .glass-card, .metric-tile, .chat-frame {
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        animation: fadeSlideUp 0.8s ease-out both;
    }
    .glass-card {
        border-radius: 24px;
        padding: 1rem 1.1rem;
    }
    .metric-tile {
        border-radius: 22px;
        padding: 1rem 1.05rem;
        min-height: 128px;
    }
    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .metric-value {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-top: 0.35rem;
    }
    .metric-delta {
        margin-top: 0.5rem;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .metric-foot, .footer-note, .section-copy {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtle-chip {
        background: rgba(20,33,61,0.05);
        border: 1px solid rgba(20,33,61,0.08);
        color: var(--ink);
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .positive { color: var(--accent); }
    .negative { color: var(--danger); }
    .neutral { color: var(--accent-warm); }
    .chat-frame {
        border-radius: 24px;
        padding: 1rem;
    }
</style>
"""


PLOT_TEMPLATE = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(color="#14213d", family="Trebuchet MS, Segoe UI, sans-serif"),
    margin=dict(l=20, r=20, t=56, b=20),
)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def bundled_company_master() -> pd.DataFrame:
    fallback = pd.DataFrame(BUNDLED_COMPANIES, columns=["name", "ticker"])
    fallback["exchange"] = "NSE"
    fallback["source"] = "Bundled fallback"
    fallback["label"] = fallback["name"] + " (" + fallback["ticker"] + ")"
    return fallback


def parse_nse_company_master(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={column: str(column).strip().upper() for column in frame.columns})
    if "SYMBOL" not in frame.columns or "NAME OF COMPANY" not in frame.columns:
        raise ValueError("Unexpected NSE company master format.")
    if "SERIES" in frame.columns:
        frame = frame[frame["SERIES"].astype(str).str.upper().isin(["EQ", "BE", "BZ", "SM", "ST"])]
    parsed = frame[["SYMBOL", "NAME OF COMPANY"]].copy()
    parsed["SYMBOL"] = parsed["SYMBOL"].astype(str).str.strip().str.upper()
    parsed["NAME OF COMPANY"] = parsed["NAME OF COMPANY"].astype(str).str.strip()
    parsed = parsed[(parsed["SYMBOL"] != "") & (parsed["NAME OF COMPANY"] != "")]
    parsed["ticker"] = parsed["SYMBOL"] + ".NS"
    parsed["name"] = parsed["NAME OF COMPANY"].str.title()
    parsed["exchange"] = "NSE"
    parsed["source"] = "Live NSE master"
    parsed["label"] = parsed["name"] + " (" + parsed["ticker"] + ")"
    return parsed[["name", "ticker", "exchange", "source", "label"]].drop_duplicates(subset="ticker")


def parse_bse_company_master(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={column: str(column).strip().upper() for column in frame.columns})
    symbol_column = next((column for column in ["SECURITY ID", "SECURITYID", "SECURITY_ID", "SC_NAME"] if column in frame.columns), None)
    code_column = next((column for column in ["SECURITY CODE", "SECURITYCODE", "SC_CODE", "SCRIP CODE"] if column in frame.columns), None)
    name_column = next((column for column in ["ISSUER NAME", "SECURITY NAME", "COMPANY NAME", "NAME"] if column in frame.columns), None)
    if symbol_column is None and code_column is None:
        raise ValueError("Unexpected BSE company master format.")

    parsed = pd.DataFrame()
    parsed["symbol"] = frame[symbol_column].astype(str).str.strip().str.upper() if symbol_column else ""
    parsed["code"] = frame[code_column].astype(str).str.strip() if code_column else ""
    parsed["name"] = frame[name_column].astype(str).str.strip().str.title() if name_column else parsed["symbol"]
    parsed = parsed[(parsed["symbol"] != "") | (parsed["code"] != "")]
    parsed["ticker_base"] = np.where(parsed["symbol"] != "", parsed["symbol"], parsed["code"])
    parsed["ticker"] = parsed["ticker_base"] + ".BO"
    parsed["exchange"] = "BSE"
    parsed["source"] = "Live BSE master"
    parsed["label"] = parsed["name"] + " (" + parsed["ticker"] + ")"
    return parsed[["name", "ticker", "exchange", "source", "label"]].drop_duplicates(subset="ticker")


@st.cache_data(ttl=MASTER_CACHE_TTL, show_spinner=False)
def fetch_indian_company_master() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
    }
    datasets: list[pd.DataFrame] = []
    for url in NSE_MASTER_URLS:
        try:
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=10)
            response = session.get(url, timeout=20)
            response.raise_for_status()
            datasets.append(parse_nse_company_master(pd.read_csv(io.StringIO(response.text))))
            break
        except Exception:
            continue

    try:
        bse_session = requests.Session()
        bse_session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,text/plain,*/*",
                "Referer": "https://www.bseindia.com/corporates/List_Scrips.html",
            }
        )
        bse_session.get("https://www.bseindia.com/corporates/List_Scrips.html", timeout=10)
        bse_response = bse_session.get(BSE_MASTER_URL, timeout=25)
        bse_response.raise_for_status()
        datasets.append(parse_bse_company_master(pd.read_csv(io.StringIO(bse_response.text))))
    except Exception:
        pass

    datasets.append(bundled_company_master())
    combined = pd.concat(datasets, ignore_index=True)
    return combined.sort_values(["name", "ticker"]).drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)


def normalize_ticker(raw_ticker: str) -> str:
    cleaned = raw_ticker.strip().upper().replace(" ", "")
    if not cleaned:
        return ""
    cleaned = cleaned.replace(".NS.NS", ".NS").replace(".BO.BO", ".BO")
    if cleaned in {"NS", ".NS", "BO", ".BO", "NSE", "BSE"}:
        return ""
    if "." not in cleaned:
        return f"{cleaned}.NS"
    return cleaned


def ticker_candidates(ticker: str) -> list[str]:
    if not ticker:
        return []
    candidates = [ticker]
    if ticker.endswith(".NS"):
        candidates.append(f"{ticker[:-3]}.BO")
    elif ticker.endswith(".BO"):
        candidates.append(f"{ticker[:-3]}.NS")
    return list(dict.fromkeys(candidates))


class FinanceAdvisor:
    @staticmethod
    def get_ai_response(user_input: str, context: str = "") -> str:
        api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
        if api_key and openai is not None:
            openai.api_key = api_key
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful Fintech AI assistant. Context: {context}"},
                        {"role": "user", "content": user_input},
                    ],
                )
                return response.choices[0].message.content
            except Exception:
                pass
        prompt = user_input.lower()
        if "bullish" in prompt:
            return "Bullish structure usually strengthens when shorter moving averages stay above longer ones and price respects that slope."
        if "risk" in prompt:
            return "Risk control starts with sizing, diversification, and knowing your acceptable drawdown before entering the trade."
        if "portfolio" in prompt:
            return "A healthy portfolio balances return, volatility, and correlation instead of chasing the strongest recent winner."
        return "I can help with stock trends, sentiment, forecast signals, and portfolio questions. Ask about a ticker or investing scenario."


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if TRANSFORMERS_AVAILABLE:
        try:
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            return None
    return None


@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def fetch_stock_data(ticker: str, years: int) -> pd.DataFrame:
    for candidate in ticker_candidates(ticker):
        data = yf.download(candidate, period=f"{years}y", interval="1d", auto_adjust=True, progress=False, threads=False)
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data.columns = [str(col).title() for col in data.columns]
        data["Date"] = pd.to_datetime(data["Date"])
        return data
    return pd.DataFrame()


@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def fetch_news(ticker_symbol: str) -> list[dict]:
    for candidate in ticker_candidates(ticker_symbol):
        try:
            ticker = yf.Ticker(candidate)
            news = ticker.news or []
            if not news:
                continue
            return [
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "time": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                }
                for item in news[:10]
            ]
        except Exception:
            continue
    return []


@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def fetch_live_snapshot(ticker_symbol: str) -> dict[str, float | str]:
    for candidate in ticker_candidates(ticker_symbol):
        try:
            stock = yf.Ticker(candidate)
            intraday = stock.history(period="2d", interval="1m", auto_adjust=True)
            if not intraday.empty:
                latest = intraday.iloc[-1]
                previous_close = stock.fast_info.get("previous_close")
                live_price = float(latest["Close"])
                change_pct = ((live_price - previous_close) / previous_close) * 100 if previous_close else 0.0
                return {
                    "ticker": candidate,
                    "price": live_price,
                    "change_pct": change_pct,
                    "timestamp": pd.Timestamp(intraday.index[-1]).strftime("%d %b %Y %I:%M:%S %p"),
                }
            fast_info = stock.fast_info
            price = fast_info.get("last_price") or fast_info.get("regular_market_price")
            previous_close = fast_info.get("previous_close")
            if price is not None:
                change_pct = ((price - previous_close) / previous_close) * 100 if previous_close else 0.0
                return {
                    "ticker": candidate,
                    "price": float(price),
                    "change_pct": change_pct,
                    "timestamp": datetime.now().strftime("%d %b %Y %I:%M:%S %p"),
                }
        except Exception:
            continue
    return {"ticker": ticker_symbol, "price": 0.0, "change_pct": 0.0, "timestamp": datetime.now().strftime("%d %b %Y %I:%M:%S %p")}


def apply_plot_style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(title=title, height=height, template="plotly_white", **PLOT_TEMPLATE)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(20,33,61,0.08)", zeroline=False)
    return fig


def render_metric_tile(label: str, value: str, delta: str, foot: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {tone}">{delta}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def run_prediction_ui(ticker: str, years: int, pred_days: int) -> None:
    df = fetch_stock_data(ticker, years)
    if df.empty:
        st.error("No market data was returned for this ticker. Try a valid NSE symbol such as `INFY.NS` or a BSE symbol such as `500325.BO`.")
        return

    live_snapshot = fetch_live_snapshot(ticker)
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["Daily Return"] = df["Close"].pct_change().fillna(0)

    current_price = float(live_snapshot["price"]) if live_snapshot["price"] else float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    ma50 = float(df["MA50"].iloc[-1])
    trend_up = ma20 > ma50
    trend_text = "Uptrend" if trend_up else "Downtrend"
    trend_icon = "📈" if trend_up else "📉"
    day_change = float(live_snapshot["change_pct"]) if live_snapshot["price"] else float(df["Daily Return"].iloc[-1] * 100)

    forecast = None
    predicted_price = current_price
    expected_change = 0.0
    if len(df) > 90:
        prophet_frame = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet(changepoint_prior_scale=0.15, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_frame)
        forecast = model.predict(model.make_future_dataframe(periods=pred_days))
        predicted_price = float(forecast["yhat"].iloc[-1])
        expected_change = ((predicted_price - current_price) / current_price) * 100 if current_price else 0.0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_tile("Current Price", f"₹{current_price:,.2f}", f"{day_change:+.2f}% today", f"Live snapshot at {live_snapshot['timestamp']}", "positive" if day_change >= 0 else "negative")
    with metric_cols[1]:
        render_metric_tile("20-Day MA", f"₹{ma20:,.2f}", f"{trend_icon} {trend_text}", "Fast trend signal", "positive" if trend_up else "negative")
    with metric_cols[2]:
        render_metric_tile("50-Day MA", f"₹{ma50:,.2f}", f"{(ma20 - ma50):+,.2f}", "Longer market rhythm", "positive" if ma20 >= ma50 else "negative")
    with metric_cols[3]:
        render_metric_tile("AI Forecast", f"₹{predicted_price:,.2f}", f"{expected_change:+.2f}% outlook", f"Next {pred_days} trading days", "positive" if expected_change >= 0 else "negative")

    overview_col, chips_col = st.columns([3.2, 1.2])
    with overview_col:
        render_section_header("Market Storyboard", "Historical price, moving-average structure, and the Prophet forecast are layered together so you can read momentum and forward drift in one place.")
    with chips_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="subtle-chip">Ticker: {ticker}</div>
                <div class="subtle-chip">Window: {years}Y</div>
                <div class="subtle-chip">Forecast: {pred_days}D</div>
                <div class="subtle-chip">Refresh: 1m</div>
                <div class="subtle-chip">Feed: {live_snapshot['ticker']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    main_chart = go.Figure()
    main_chart.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(color="#14213d", width=3)))
    main_chart.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="20-Day MA", line=dict(color="#0b6e4f", width=2.5)))
    main_chart.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="50-Day MA", line=dict(color="#d97706", width=2.5)))
    if forecast is not None:
        future = forecast.tail(pred_days)
        main_chart.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines+markers", name="AI Forecast", line=dict(color="#7c3aed", width=3, dash="dash")))
        main_chart.add_trace(go.Scatter(x=future["ds"], y=future["yhat_upper"], mode="lines", line=dict(color="rgba(124,58,237,0)"), showlegend=False))
        main_chart.add_trace(go.Scatter(x=future["ds"], y=future["yhat_lower"], mode="lines", line=dict(color="rgba(124,58,237,0)"), fill="tonexty", fillcolor="rgba(124,58,237,0.12)", name="Forecast Range"))
    apply_plot_style(main_chart, "Historical Price and Forecast Canvas", height=500)
    st.plotly_chart(main_chart, use_container_width=True)

    lower, upper = st.columns([1.6, 1])
    with lower:
        signal_chart = go.Figure()
        signal_chart.add_trace(go.Bar(x=df["Date"].tail(45), y=df["Daily Return"].tail(45) * 100, marker_color=np.where(df["Daily Return"].tail(45) >= 0, "#0b6e4f", "#be123c"), name="Daily Return %"))
        apply_plot_style(signal_chart, "Recent Return Pulse", height=360)
        st.plotly_chart(signal_chart, use_container_width=True)
    with upper:
        forecast_table = pd.DataFrame()
        if forecast is not None:
            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(pred_days).copy()
            forecast_table.columns = ["Date", "Forecast", "Lower", "Upper"]
            forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
            forecast_table["Expected Change %"] = ((forecast_table["Forecast"] - current_price) / current_price) * 100
            forecast_table["Signal"] = np.where(forecast_table["Expected Change %"] >= 0, "Growth", "Pullback")
        render_section_header("Forward Price Map", "A compact forecast table for the upcoming horizon.")
        st.dataframe(
            forecast_table.style.format({"Forecast": "₹{:,.2f}", "Lower": "₹{:,.2f}", "Upper": "₹{:,.2f}", "Expected Change %": "{:+.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )


def run_sentiment_ui(default_ticker: str) -> None:
    render_section_header("Headline Moodboard", "News headlines are clustered into a fast sentiment snapshot so you can gauge whether the narrative around a stock is supportive, cautious, or mixed.")
    ticker = st.text_input("Ticker for news sentiment", value=default_ticker)
    news = fetch_news(ticker)
    if not news:
        st.warning("No recent news was returned for this ticker.")
        return
    with st.spinner("Reading the tone of recent headlines..."):
        model = load_sentiment_model()
        results = []
        for item in news:
            label, score = ("neutral", 0.0)
            if model:
                result = model(item["title"])[0]
                label, score = result["label"].lower(), float(result["score"])
            impact = "📈" if "pos" in label else "📉" if "neg" in label else "➖"
            results.append({"Headline": item["title"], "Publisher": item["publisher"], "Time": item["time"], "Sentiment": label.title(), "Confidence": score, "Impact": impact})
    sentiment_df = pd.DataFrame(results)
    left, right = st.columns([1.7, 1])
    with left:
        st.dataframe(sentiment_df.style.format({"Confidence": "{:.2f}"}), use_container_width=True, hide_index=True)
    with right:
        pie = px.pie(sentiment_df, names="Sentiment", color="Sentiment", color_discrete_map={"Positive": "#0b6e4f", "Negative": "#be123c", "Neutral": "#d97706"}, hole=0.58)
        pie.update_traces(textposition="inside", textinfo="percent+label")
        apply_plot_style(pie, "Narrative Balance", height=380)
        st.plotly_chart(pie, use_container_width=True)


def run_portfolio_ui() -> None:
    render_section_header("Portfolio Atmosphere", "Get a quick read on return, volatility, and diversification pressure from a basket of tickers.")
    stocks = st.text_input("Tickers", value="AAPL, MSFT, RELIANCE.NS")
    if not st.button("Analyze Portfolio", use_container_width=True):
        return
    tickers = [symbol.strip() for symbol in stocks.split(",") if symbol.strip()]
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    returns = data.pct_change().dropna()
    if returns.empty:
        st.error("Not enough data was returned for that basket.")
        return
    count = len(tickers)
    weights = np.array(count * [1.0 / count])
    annual_return = np.sum(returns.mean() * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (annual_return - 0.06) / volatility if volatility else 0.0
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_tile("Annual Return", f"{annual_return:.2%}", "Portfolio drift", "Higher is better", "positive" if annual_return >= 0 else "negative")
    with c2:
        render_metric_tile("Volatility", f"{volatility:.2%}", "Risk profile", "Annualized variation", "neutral")
    with c3:
        render_metric_tile("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return", "Using 6% risk-free rate", "positive" if sharpe >= 1 else "neutral")
    chart_left, chart_right = st.columns([1.5, 1])
    with chart_left:
        cumulative = (1 + returns).cumprod()
        line = go.Figure()
        for column in cumulative.columns:
            line.add_trace(go.Scatter(x=cumulative.index, y=cumulative[column], mode="lines", name=str(column), line=dict(width=2.2)))
        apply_plot_style(line, "Relative Performance Wave", height=390)
        st.plotly_chart(line, use_container_width=True)
    with chart_right:
        alloc = pd.DataFrame({"Ticker": tickers, "Weight": weights})
        donut = px.pie(alloc, names="Ticker", values="Weight", hole=0.55, color_discrete_sequence=["#14213d", "#0b6e4f", "#d97706", "#7c3aed", "#be123c", "#2563eb"])
        apply_plot_style(donut, "Equal-Weight Mix", height=390)
        st.plotly_chart(donut, use_container_width=True)


def run_chat_ui(active_ticker: str) -> None:
    render_section_header("AI Advisor Lounge", "A cleaner space for conversation, strategy questions, and quick market explanations with context from your active ticker.")
    st.markdown('<div class="chat-frame">', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask about momentum, risk, entries, or portfolio health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = FinanceAdvisor.get_ai_response(prompt, context=f"Current selected ticker: {active_ticker}")
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="sync")
    rerun_time = datetime.now().strftime("%d %b %Y %I:%M:%S %p")
    company_master = fetch_indian_company_master()
    labels = company_master["label"].tolist()
    default_label = "Infosys (INFY.NS)" if "Infosys (INFY.NS)" in labels else labels[0]

    st.markdown(
        f"""
        <div class="hero">
            <h1>FinAI Super App</h1>
            <p>
                Live market storytelling for Indian equities with a searchable all-company NSE universe,
                1-minute refresh cycles, richer news context, forecasting, and portfolio diagnostics.
            </p>
            <div class="hero-strip">
                <span class="hero-pill">Auto refresh every 1 minute</span>
                <span class="hero-pill">Refresh count: {int(refresh_count)}</span>
                <span class="hero-pill">Last rerun: {rerun_time}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("FinAI Control Deck")
        st.caption("Search a broad Indian equity universe powered by live exchange metadata and yfinance market data.")
        input_mode = st.radio("Ticker source", ["Search Indian companies", "Enter custom ticker"], index=0)
        if input_mode == "Search Indian companies":
            selected_company = st.selectbox("Indian listed companies", options=labels, index=labels.index(default_label))
            active_ticker = company_master.loc[company_master["label"] == selected_company, "ticker"].iloc[0]
            exchange_count = company_master["exchange"].fillna("Unknown").nunique()
            st.caption(f"{company_master['ticker'].nunique():,} Indian market symbols loaded across {exchange_count} exchange sources. Custom `.NS` and `.BO` symbols can still be entered manually.")
        else:
            active_ticker = normalize_ticker(st.text_input("Primary ticker", value="INFY.NS"))
        history_years = st.select_slider("History window", options=[1, 2, 3], value=2)
        forecast_days = st.slider("Forecast horizon", min_value=7, max_value=30, value=15)
        selected = st.radio("Choose experience", ["Stock Prediction", "News Sentiment", "Portfolio Analyzer", "AI Advisor"])
        st.markdown("---")
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title" style="font-size:1rem;">Live Session</div>
                <div class="footer-note">Ticker: <strong>{active_ticker}</strong></div>
                <div class="footer-note">Last refresh: {rerun_time}</div>
                <div class="footer-note">Auto refresh cycle: 1 minute</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if selected == "Stock Prediction":
        run_prediction_ui(active_ticker, history_years, forecast_days)
    elif selected == "News Sentiment":
        run_sentiment_ui(active_ticker)
    elif selected == "Portfolio Analyzer":
        run_portfolio_ui()
    else:
        run_chat_ui(active_ticker)


if __name__ == "__main__":
    main()
