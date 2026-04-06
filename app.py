from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


CUSTOM_CSS = """
<style>
    :root {
        --bg: #f4efe6;
        --bg-soft: #fbf7f1;
        --ink: #14213d;
        --muted: #5b6577;
        --accent: #0b6e4f;
        --accent-2: #d97706;
        --accent-3: #7c3aed;
        --rose: #be123c;
        --card: rgba(255,255,255,0.78);
        --border: rgba(20, 33, 61, 0.08);
        --shadow: 0 22px 60px rgba(20, 33, 61, 0.10);
    }
    @keyframes fadeSlideUp {
        0% {
            opacity: 0;
            transform: translateY(18px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes softFloat {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-6px);
        }
    }
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.18);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(217, 119, 6, 0.02);
        }
    }
    @keyframes gradientDrift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(11, 110, 79, 0.18), transparent 26%),
            radial-gradient(circle at 100% 10%, rgba(217, 119, 6, 0.16), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(124, 58, 237, 0.10), transparent 28%),
            linear-gradient(180deg, #f7f2e9 0%, #f2f6f3 50%, #eef3fa 100%);
        color: var(--ink);
        background-size: 120% 120%;
        animation: gradientDrift 22s ease-in-out infinite;
    }
    .stSidebar > div:first-child {
        background:
            linear-gradient(180deg, rgba(20,33,61,0.97) 0%, rgba(16,24,40,0.96) 100%);
        color: #f8fafc;
        border-right: 1px solid rgba(255,255,255,0.08);
        animation: fadeSlideUp 0.75s ease-out;
    }
    .stSidebar label, .stSidebar p, .stSidebar div[data-testid="stMarkdownContainer"] {
        color: #e5edf8;
    }
    .stSidebar .stRadio label, .stSidebar .stSelectbox label, .stSidebar .stTextInput label {
        color: #f9fafb;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 16px !important;
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
        background:
            linear-gradient(135deg, rgba(20,33,61,0.95) 0%, rgba(11,110,79,0.92) 55%, rgba(217,119,6,0.88) 100%);
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
        background: rgba(255,255,255,0.09);
        filter: blur(2px);
        animation: softFloat 6s ease-in-out infinite;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
        line-height: 1.02;
        letter-spacing: -0.04em;
        font-family: "Georgia", "Palatino Linotype", serif;
    }
    .hero p {
        margin: 0.7rem 0 0 0;
        max-width: 760px;
        color: rgba(255,247,237,0.88);
        font-size: 1rem;
    }
    .hero-strip {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .hero-pill {
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.88rem;
        color: #fffaf2;
        animation: fadeSlideUp 0.7s ease-out both, glowPulse 3.8s ease-in-out infinite;
    }
    .hero-pill:nth-child(1) { animation-delay: 0.08s, 0s; }
    .hero-pill:nth-child(2) { animation-delay: 0.16s, 0s; }
    .hero-pill:nth-child(3) { animation-delay: 0.24s, 0s; }
    .glass-card {
        border-radius: 24px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 1rem 1.1rem;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeSlideUp 0.85s ease-out both;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 26px 60px rgba(20, 33, 61, 0.14);
        border-color: rgba(11, 110, 79, 0.18);
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    .section-copy {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .metric-tile {
        border-radius: 22px;
        padding: 1rem 1.05rem;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(20,33,61,0.08);
        box-shadow: 0 14px 32px rgba(20,33,61,0.08);
        min-height: 128px;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeSlideUp 0.7s ease-out both;
    }
    .metric-tile:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 22px 44px rgba(20, 33, 61, 0.14);
        border-color: rgba(124, 58, 237, 0.16);
    }
    .metric-tile:nth-of-type(1) { animation-delay: 0.06s; }
    .metric-tile:nth-of-type(2) { animation-delay: 0.12s; }
    .metric-tile:nth-of-type(3) { animation-delay: 0.18s; }
    .metric-tile:nth-of-type(4) { animation-delay: 0.24s; }
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
        margin-top: 0.55rem;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .metric-foot {
        margin-top: 0.35rem;
        font-size: 0.82rem;
        color: var(--muted);
    }
    .positive { color: var(--accent); }
    .negative { color: var(--rose); }
    .neutral { color: var(--accent-2); }
    .subtle-chip {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(20,33,61,0.05);
        border: 1px solid rgba(20,33,61,0.08);
        color: var(--ink);
        font-size: 0.83rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .chat-frame {
        border-radius: 24px;
        padding: 1rem;
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        animation: fadeSlideUp 0.85s ease-out;
    }
    div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
        border-radius: 22px;
        overflow: hidden;
        animation: fadeSlideUp 0.95s ease-out both;
    }
    div[data-testid="stPlotlyChart"] {
        transition: transform 0.24s ease, box-shadow 0.24s ease;
    }
    div[data-testid="stPlotlyChart"]:hover {
        transform: translateY(-4px);
    }
    div[data-testid="column"] {
        animation: fadeSlideUp 0.75s ease-out both;
    }
    div[data-testid="column"]:nth-child(1) { animation-delay: 0.05s; }
    div[data-testid="column"]:nth-child(2) { animation-delay: 0.12s; }
    div[data-testid="column"]:nth-child(3) { animation-delay: 0.19s; }
    div[data-testid="column"]:nth-child(4) { animation-delay: 0.26s; }
    .stButton > button {
        border-radius: 999px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
        box-shadow: 0 10px 24px rgba(20, 33, 61, 0.12);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 28px rgba(20, 33, 61, 0.18);
    }
    .stChatMessage {
        animation: fadeSlideUp 0.45s ease-out both;
    }
    .footer-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.6rem;
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
            return "Risk control starts with sizing, diversification, and knowing how much drawdown you can accept before entering a trade."
        if "portfolio" in prompt:
            return "A healthy portfolio balances return, volatility, and correlation rather than chasing the strongest recent winner."
        return "I can help with stock trends, sentiment, forecast signals, and portfolio questions. Ask about a ticker or investing scenario."


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if TRANSFORMERS_AVAILABLE:
        try:
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            return None
    return None


@st.cache_data(ttl=25, show_spinner=False)
def fetch_stock_data(ticker: str, years: int) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data.columns = [str(col).title() for col in data.columns]
    data["Date"] = pd.to_datetime(data["Date"])
    return data


@st.cache_data(ttl=60, show_spinner=False)
def fetch_news(ticker_symbol: str) -> list[dict]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news or []
        return [
            {
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "time": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
            }
            for item in news[:10]
        ]
    except Exception:
        return []


def apply_plot_style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        **PLOT_TEMPLATE,
    )
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
        st.error("No market data was returned for this ticker. Try a valid NSE or BSE symbol such as `INFY.NS` or `RELIANCE.NS`.")
        return

    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["Daily Return"] = df["Close"].pct_change().fillna(0)

    current_price = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    ma50 = float(df["MA50"].iloc[-1])
    trend_up = ma20 > ma50
    trend_text = "Uptrend" if trend_up else "Downtrend"
    trend_icon = "📈" if trend_up else "📉"
    day_change = float(df["Daily Return"].iloc[-1] * 100)

    prophet_frame = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    forecast = None
    if len(df) > 90:
        model = Prophet(
            changepoint_prior_scale=0.15,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.fit(prophet_frame)
        forecast = model.predict(model.make_future_dataframe(periods=pred_days))
        predicted_price = float(forecast["yhat"].iloc[-1])
        expected_change = ((predicted_price - current_price) / current_price) * 100 if current_price else 0.0
    else:
        predicted_price = current_price
        expected_change = 0.0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_tile("Current Price", f"₹{current_price:,.2f}", f"{day_change:+.2f}% today", "Latest adjusted close", "positive" if day_change >= 0 else "negative")
    with metric_cols[1]:
        render_metric_tile("20-Day MA", f"₹{ma20:,.2f}", trend_icon + " " + trend_text, "Fast trend signal", "positive" if trend_up else "negative")
    with metric_cols[2]:
        render_metric_tile("50-Day MA", f"₹{ma50:,.2f}", f"{(ma20 - ma50):+,.2f}", "Longer market rhythm", "positive" if ma20 >= ma50 else "negative")
    with metric_cols[3]:
        render_metric_tile("AI Forecast", f"₹{predicted_price:,.2f}", f"{expected_change:+.2f}% outlook", f"Next {pred_days} trading days", "positive" if expected_change >= 0 else "negative")

    overview_col, chips_col = st.columns([3.2, 1.2])
    with overview_col:
        render_section_header(
            "Market Storyboard",
            "Historical price, moving-average structure, and the Prophet forecast are layered together so you can read momentum and forward drift in one place.",
        )
    with chips_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="subtle-chip">Ticker: {ticker}</div>
                <div class="subtle-chip">Window: {years}Y</div>
                <div class="subtle-chip">Forecast: {pred_days}D</div>
                <div class="subtle-chip">Refresh: 30s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    main_chart = go.Figure()
    main_chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#14213d", width=3),
        )
    )
    main_chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            mode="lines",
            name="20-Day MA",
            line=dict(color="#0b6e4f", width=2.5),
        )
    )
    main_chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA50"],
            mode="lines",
            name="50-Day MA",
            line=dict(color="#d97706", width=2.5),
        )
    )
    if forecast is not None:
        future = forecast.tail(pred_days)
        main_chart.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat"],
                mode="lines+markers",
                name="AI Forecast",
                line=dict(color="#7c3aed", width=3, dash="dash"),
            )
        )
        main_chart.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_upper"],
                mode="lines",
                line=dict(color="rgba(124,58,237,0)"),
                showlegend=False,
            )
        )
        main_chart.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_lower"],
                mode="lines",
                line=dict(color="rgba(124,58,237,0)"),
                fill="tonexty",
                fillcolor="rgba(124,58,237,0.12)",
                name="Forecast Range",
            )
        )
    apply_plot_style(main_chart, "Historical Price and Forecast Canvas", height=500)
    st.plotly_chart(main_chart, use_container_width=True)

    lower, upper = st.columns([1.6, 1])
    with lower:
        signal_chart = go.Figure()
        signal_chart.add_trace(
            go.Bar(
                x=df["Date"].tail(45),
                y=(df["Daily Return"].tail(45) * 100),
                marker_color=np.where(df["Daily Return"].tail(45) >= 0, "#0b6e4f", "#be123c"),
                name="Daily Return %",
            )
        )
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
            forecast_table.style.format(
                {
                    "Forecast": "₹{:,.2f}",
                    "Lower": "₹{:,.2f}",
                    "Upper": "₹{:,.2f}",
                    "Expected Change %": "{:+.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def run_sentiment_ui(default_ticker: str) -> None:
    render_section_header(
        "Headline Moodboard",
        "News headlines are clustered into a fast sentiment snapshot so you can gauge whether the narrative around a stock is supportive, cautious, or mixed.",
    )
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
            results.append(
                {
                    "Headline": item["title"],
                    "Publisher": item["publisher"],
                    "Time": item["time"],
                    "Sentiment": label.title(),
                    "Confidence": score,
                    "Impact": impact,
                }
            )

    sentiment_df = pd.DataFrame(results)
    left, right = st.columns([1.7, 1])
    with left:
        st.dataframe(
            sentiment_df.style.format({"Confidence": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        pie = px.pie(
            sentiment_df,
            names="Sentiment",
            color="Sentiment",
            color_discrete_map={"Positive": "#0b6e4f", "Negative": "#be123c", "Neutral": "#d97706"},
            hole=0.58,
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        apply_plot_style(pie, "Narrative Balance", height=380)
        st.plotly_chart(pie, use_container_width=True)


def run_portfolio_ui() -> None:
    render_section_header(
        "Portfolio Atmosphere",
        "Get a quick read on return, volatility, and diversification pressure from a basket of tickers.",
    )
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
            line.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative[column],
                    mode="lines",
                    name=str(column),
                    line=dict(width=2.2),
                )
            )
        apply_plot_style(line, "Relative Performance Wave", height=390)
        st.plotly_chart(line, use_container_width=True)
    with chart_right:
        alloc = pd.DataFrame({"Ticker": tickers, "Weight": weights})
        donut = px.pie(
            alloc,
            names="Ticker",
            values="Weight",
            hole=0.55,
            color_discrete_sequence=["#14213d", "#0b6e4f", "#d97706", "#7c3aed", "#be123c", "#2563eb"],
        )
        apply_plot_style(donut, "Equal-Weight Mix", height=390)
        st.plotly_chart(donut, use_container_width=True)


def run_chat_ui(active_ticker: str) -> None:
    render_section_header(
        "AI Advisor Lounge",
        "A cleaner space for conversation, strategy questions, and quick market explanations with context from your active ticker.",
    )
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

        context = f"Current selected ticker: {active_ticker}"
        response = FinanceAdvisor.get_ai_response(prompt, context=context)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    refresh_count = st_autorefresh(interval=30_000, key="sync")
    rerun_time = datetime.now().strftime("%d %b %Y %I:%M:%S %p")

    st.markdown(
        f"""
        <div class="hero">
            <h1>FinAI Super App</h1>
            <p>
                Live market storytelling for Indian equities with elegant forecasting, richer news context,
                portfolio diagnostics, and a more cinematic investing dashboard.
            </p>
            <div class="hero-strip">
                <span class="hero-pill">Auto refresh every 30 seconds</span>
                <span class="hero-pill">Refresh count: {int(refresh_count)}</span>
                <span class="hero-pill">Last rerun: {rerun_time}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("FinAI Control Deck")
        st.caption("A richer command center for prediction, sentiment, portfolio signals, and AI guidance.")

        active_ticker = st.text_input("Primary ticker", value="INFY.NS")
        history_years = st.select_slider("History window", options=[1, 2, 3], value=2)
        forecast_days = st.slider("Forecast horizon", min_value=7, max_value=30, value=15)
        selected = st.radio(
            "Choose experience",
            ["Stock Prediction", "News Sentiment", "Portfolio Analyzer", "AI Advisor"],
        )
        st.markdown("---")
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title" style="font-size:1rem;">Live Session</div>
                <div class="footer-note">Ticker: <strong>{active_ticker}</strong></div>
                <div class="footer-note">Last refresh: {rerun_time}</div>
                <div class="footer-note">Auto refresh cycle: 30 seconds</div>
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
