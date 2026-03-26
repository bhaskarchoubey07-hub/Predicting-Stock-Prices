from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    TENSORFLOW_AVAILABLE = True
except Exception:
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    TENSORFLOW_AVAILABLE = False


st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(11, 132, 255, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(18, 184, 134, 0.18), transparent 28%),
            linear-gradient(180deg, #f6fbff 0%, #eef6f5 100%);
        color: #12303f;
    }
    .hero-card {
        padding: 1.3rem 1.5rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(18, 48, 63, 0.08);
        box-shadow: 0 18px 45px rgba(18, 48, 63, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #45606d;
        font-size: 1rem;
    }
    .trend-pill {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        margin-top: 0.6rem;
    }
    .trend-up {
        background: rgba(18, 184, 134, 0.12);
        color: #0f8c65;
    }
    .trend-down {
        background: rgba(230, 73, 128, 0.12);
        color: #c2255c;
    }
    .data-note {
        color: #4b6774;
        font-size: 0.95rem;
    }
</style>
"""


MIN_HISTORY_ROWS = 120
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 12
NSE_MASTER_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
INDIAN_COMPANIES = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("Tata Consultancy Services", "TCS.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Infosys", "INFY.NS"),
    ("ITC", "ITC.NS"),
    ("Larsen & Toubro", "LT.NS"),
    ("Hindustan Unilever", "HINDUNILVR.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("Kotak Mahindra Bank", "KOTAKBANK.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("Bajaj Finserv", "BAJAJFINSV.NS"),
    ("Sun Pharmaceutical", "SUNPHARMA.NS"),
    ("Maruti Suzuki", "MARUTI.NS"),
    ("NTPC", "NTPC.NS"),
    ("UltraTech Cement", "ULTRACEMCO.NS"),
    ("Asian Paints", "ASIANPAINT.NS"),
    ("Titan Company", "TITAN.NS"),
    ("Mahindra & Mahindra", "M&M.NS"),
    ("HCL Technologies", "HCLTECH.NS"),
    ("Power Grid", "POWERGRID.NS"),
    ("Tata Motors", "TATAMOTORS.NS"),
    ("Adani Ports", "ADANIPORTS.NS"),
    ("Adani Enterprises", "ADANIENT.NS"),
    ("Bajaj Auto", "BAJAJ-AUTO.NS"),
    ("Bharat Electronics", "BEL.NS"),
    ("Trent", "TRENT.NS"),
    ("Wipro", "WIPRO.NS"),
    ("Nestle India", "NESTLEIND.NS"),
    ("ONGC", "ONGC.NS"),
    ("Coal India", "COALINDIA.NS"),
    ("HDFC Life", "HDFCLIFE.NS"),
    ("SBI Life Insurance", "SBILIFE.NS"),
    ("Grasim Industries", "GRASIM.NS"),
    ("JSW Steel", "JSWSTEEL.NS"),
    ("Hindalco Industries", "HINDALCO.NS"),
    ("Tata Steel", "TATASTEEL.NS"),
    ("Cipla", "CIPLA.NS"),
    ("Dr Reddys Laboratories", "DRREDDY.NS"),
    ("IndusInd Bank", "INDUSINDBK.NS"),
    ("Eicher Motors", "EICHERMOT.NS"),
    ("Hero MotoCorp", "HEROMOTOCO.NS"),
    ("Divis Laboratories", "DIVISLAB.NS"),
    ("Tech Mahindra", "TECHM.NS"),
    ("Tata Consumer Products", "TATACONSUM.NS"),
    ("Britannia Industries", "BRITANNIA.NS"),
    ("Apollo Hospitals", "APOLLOHOSP.NS"),
    ("Shriram Finance", "SHRIRAMFIN.NS"),
    ("Adani Green Energy", "ADANIGREEN.NS"),
    ("Adani Power", "ADANIPOWER.NS"),
    ("Pidilite Industries", "PIDILITIND.NS"),
    ("Ambuja Cements", "AMBUJACEM.NS"),
    ("Siemens India", "SIEMENS.NS"),
    ("Godrej Consumer Products", "GODREJCP.NS"),
    ("Dabur India", "DABUR.NS"),
    ("Vedanta", "VEDL.NS"),
    ("BPCL", "BPCL.NS"),
    ("Indian Oil Corporation", "IOC.NS"),
    ("Havells India", "HAVELLS.NS"),
    ("DLF", "DLF.NS"),
    ("Avenue Supermarts", "DMART.NS"),
    ("InterGlobe Aviation", "INDIGO.NS"),
    ("TVS Motor Company", "TVSMOTOR.NS"),
    ("ABB India", "ABB.NS"),
    ("Bank of Baroda", "BANKBARODA.NS"),
    ("Canara Bank", "CANBK.NS"),
    ("Punjab National Bank", "PNB.NS"),
    ("REC", "RECLTD.NS"),
    ("Power Finance Corporation", "PFC.NS"),
    ("Indian Railway Finance Corporation", "IRFC.NS"),
    ("NHPC", "NHPC.NS"),
    ("Torrent Pharmaceuticals", "TORNTPHARM.NS"),
    ("Samvardhana Motherson", "MOTHERSON.NS"),
    ("Bosch", "BOSCHLTD.NS"),
    ("Colgate-Palmolive India", "COLPAL.NS"),
    ("ICICI Prudential Life", "ICICIPRULI.NS"),
    ("LTIMindtree", "LTIM.NS"),
    ("Mphasis", "MPHASIS.NS"),
    ("Info Edge", "NAUKRI.NS"),
    ("Persistent Systems", "PERSISTENT.NS"),
    ("Coforge", "COFORGE.NS"),
    ("Aurobindo Pharma", "AUROPHARMA.NS"),
    ("Polycab India", "POLYCAB.NS"),
    ("CG Power", "CGPOWER.NS"),
    ("BHEL", "BHEL.NS"),
    ("HAL", "HAL.NS"),
    ("Indian Hotels Company", "INDHOTEL.NS"),
    ("Max Healthcare", "MAXHEALTH.NS"),
    ("Lupin", "LUPIN.NS"),
    ("Marico", "MARICO.NS"),
    ("United Spirits", "MCDOWELL-N.NS"),
    ("Page Industries", "PAGEIND.NS"),
    ("PI Industries", "PIIND.NS"),
    ("SRF", "SRF.NS"),
    ("Berger Paints", "BERGEPAINT.NS"),
    ("Container Corporation", "CONCOR.NS"),
    ("Cummins India", "CUMMINSIND.NS"),
    ("GAIL India", "GAIL.NS"),
    ("ICICI Lombard", "ICICIGI.NS"),
    ("Jindal Steel & Power", "JINDALSTEL.NS"),
    ("Life Insurance Corporation", "LICI.NS"),
    ("NMDC", "NMDC.NS"),
    ("Oberoi Realty", "OBEROIRLTY.NS"),
    ("Petronet LNG", "PETRONET.NS"),
    ("SAIL", "SAIL.NS"),
    ("Supreme Industries", "SUPREMEIND.NS"),
    ("Union Bank of India", "UNIONBANK.NS"),
    ("Zydus Lifesciences", "ZYDUSLIFE.NS"),
    ("One 97 Communications", "PAYTM.NS"),
    ("Jio Financial Services", "JIOFIN.NS"),
    ("Indus Towers", "INDUSTOWER.NS"),
    ("AU Small Finance Bank", "AUBANK.NS"),
    ("Max Financial Services", "MFSL.NS"),
    ("Tata Power", "TATAPOWER.NS"),
    ("ACC", "ACC.NS"),
]
COMPANY_LABELS = [f"{name} ({ticker})" for name, ticker in INDIAN_COMPANIES]
COMPANY_LABEL_TO_TICKER = {label: ticker for label, (_, ticker) in zip(COMPANY_LABELS, INDIAN_COMPANIES)}
KNOWN_SYMBOL_LOOKUP = {ticker.split(".")[0].upper(): ticker for _, ticker in INDIAN_COMPANIES}


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)


def to_naive_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def normalize_ticker(raw_ticker: str) -> str:
    cleaned = raw_ticker.strip().upper().replace(" ", "")
    if not cleaned:
        return ""

    cleaned = cleaned.replace(".NS.NS", ".NS").replace(".BO.BO", ".BO")
    if cleaned in {"NS", ".NS", "BO", ".BO", "NSE", "BSE"}:
        raise ValueError("Please choose a valid company or enter a stock symbol like RELIANCE, TCS, INFY, or SBIN.")

    if cleaned in KNOWN_SYMBOL_LOOKUP:
        return KNOWN_SYMBOL_LOOKUP[cleaned]

    if cleaned.endswith("NS") and not cleaned.endswith(".NS"):
        return f"{cleaned[:-2]}.NS"
    if cleaned.endswith("BO") and not cleaned.endswith(".BO"):
        return f"{cleaned[:-2]}.BO"
    if "." not in cleaned:
        return f"{cleaned}.NS"
    return cleaned


def bundled_company_master() -> pd.DataFrame:
    bundled = pd.DataFrame(INDIAN_COMPANIES, columns=["name", "ticker"])
    bundled["exchange"] = "NSE"
    bundled["source"] = "Bundled large-cap fallback"
    bundled["label"] = bundled["name"] + " (" + bundled["ticker"] + ")"
    return bundled


def parse_nse_company_master(frame: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {column: str(column).strip().upper() for column in frame.columns}
    frame = frame.rename(columns=normalized_columns)

    if "SYMBOL" not in frame.columns or "NAME OF COMPANY" not in frame.columns:
        raise ValueError("NSE company master is missing expected columns.")

    if "SERIES" in frame.columns:
        frame = frame[frame["SERIES"].astype(str).str.upper().isin(["EQ", "BE", "BZ", "SM", "ST"])]

    parsed = frame[["SYMBOL", "NAME OF COMPANY"]].copy()
    parsed["SYMBOL"] = parsed["SYMBOL"].astype(str).str.strip().str.upper()
    parsed["NAME OF COMPANY"] = parsed["NAME OF COMPANY"].astype(str).str.strip()
    parsed = parsed[(parsed["SYMBOL"] != "") & (parsed["NAME OF COMPANY"] != "")]
    parsed["ticker"] = parsed["SYMBOL"] + ".NS"
    parsed["name"] = parsed["NAME OF COMPANY"].str.title()
    parsed["exchange"] = "NSE"
    parsed["source"] = "Live NSE equity master"
    parsed["label"] = parsed["name"] + " (" + parsed["ticker"] + ")"
    return parsed[["name", "ticker", "exchange", "source", "label"]].drop_duplicates(subset="ticker")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_indian_company_master() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
    }
    errors: list[str] = []

    for url in NSE_MASTER_URLS:
        try:
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=10)
            response = session.get(url, timeout=20)
            response.raise_for_status()
            live_master = parse_nse_company_master(pd.read_csv(io.StringIO(response.text)))

            combined = pd.concat([live_master, bundled_company_master()], ignore_index=True)
            combined = combined.sort_values(["name", "ticker"]).drop_duplicates(subset="ticker", keep="first")
            return combined.reset_index(drop=True)
        except Exception as error:
            errors.append(str(error))

    fallback = bundled_company_master()
    fallback["source"] = "Bundled large-cap fallback"
    return fallback


def ticker_candidates(ticker: str) -> list[str]:
    candidates = [ticker]
    if ticker.endswith(".NS"):
        candidates.append(f"{ticker[:-3]}.BO")
    elif ticker.endswith(".BO"):
        candidates.append(f"{ticker[:-3]}.NS")
    return list(dict.fromkeys(candidates))


@st.cache_data(ttl=25, show_spinner=False)
def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    for candidate in ticker_candidates(ticker):
        data = yf.download(
            candidate,
            period=f"{years}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data.columns = [str(col).title() for col in data.columns]

        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        data["Date"] = pd.to_datetime(data["Date"]).apply(to_naive_timestamp)
        return data

    raise ValueError("No historical data was returned for this ticker on NSE or BSE.")


@st.cache_data(ttl=25, show_spinner=False)
def fetch_live_snapshot(ticker: str) -> dict[str, Any]:
    for candidate in ticker_candidates(ticker):
        stock = yf.Ticker(candidate)

        intraday = stock.history(period="2d", interval="1m", auto_adjust=True)
        if not intraday.empty:
            intraday = intraday.reset_index()
            intraday.columns = [str(col).title() for col in intraday.columns]
            last_row = intraday.iloc[-1]
            previous_close = stock.fast_info.get("previous_close")
            live_price = float(last_row["Close"])
            if previous_close and previous_close != 0:
                change_pct = ((live_price - previous_close) / previous_close) * 100
            else:
                change_pct = np.nan
            timestamp = to_naive_timestamp(last_row.iloc[0])
            return {
                "price": live_price,
                "change_pct": change_pct,
                "timestamp": timestamp,
            }

        fast_info = stock.fast_info
        price = fast_info.get("last_price") or fast_info.get("regular_market_price")
        previous_close = fast_info.get("previous_close")

        if price is not None:
            if previous_close and previous_close != 0:
                change_pct = ((price - previous_close) / previous_close) * 100
            else:
                change_pct = np.nan

            return {
                "price": float(price),
                "change_pct": change_pct,
                "timestamp": datetime.now(),
            }

    raise ValueError("Live price is currently unavailable for this ticker on NSE or BSE.")


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    processed = data.copy()
    processed = processed.sort_values("Date").drop_duplicates(subset="Date")
    processed[["Open", "High", "Low", "Close", "Volume"]] = (
        processed[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill()
    )

    processed["Daily_Return"] = processed["Close"].pct_change().fillna(0.0)
    processed["MA20"] = processed["Close"].rolling(window=20, min_periods=1).mean()
    processed["MA50"] = processed["Close"].rolling(window=50, min_periods=1).mean()
    return processed


def serialize_frame(frame: pd.DataFrame) -> str:
    return frame.to_json(date_format="iso", orient="split")


@st.cache_resource(show_spinner=False)
def train_prophet_model(serialized_df: str) -> Prophet:
    prophet_df = pd.read_json(io.StringIO(serialized_df), orient="split")
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.08,
        seasonality_mode="multiplicative",
    )
    model.fit(prophet_df)
    return model


def run_prophet_forecast(processed: pd.DataFrame, prediction_days: int) -> dict[str, Any]:
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed in this environment.")

    prophet_df = processed[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    full_model = train_prophet_model(serialize_frame(prophet_df))
    future = full_model.make_future_dataframe(periods=prediction_days, freq="D")
    forecast = full_model.predict(future)

    holdout = min(max(14, prediction_days), max(14, len(prophet_df) // 5))
    if len(prophet_df) <= holdout + 30:
        raise ValueError("Not enough data to evaluate Prophet predictions. Try a longer history.")

    train_df = prophet_df.iloc[:-holdout].copy()
    test_df = prophet_df.iloc[-holdout:].copy()
    backtest_model = train_prophet_model(serialize_frame(train_df))
    backtest_future = backtest_model.make_future_dataframe(periods=holdout, freq="D")
    backtest_forecast = backtest_model.predict(backtest_future)
    comparison = backtest_forecast[["ds", "yhat"]].tail(holdout).merge(test_df, on="ds", how="inner")
    comparison.rename(columns={"y": "Actual", "yhat": "Predicted"}, inplace=True)

    future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(prediction_days).copy()
    future_forecast.rename(
        columns={
            "ds": "Date",
            "yhat": "Predicted_Close",
            "yhat_lower": "Lower_Bound",
            "yhat_upper": "Upper_Bound",
        },
        inplace=True,
    )

    metrics = {
        "mae": float(mean_absolute_error(comparison["Actual"], comparison["Predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(comparison["Actual"], comparison["Predicted"]))),
        "mape": safe_mape(comparison["Actual"].to_numpy(), comparison["Predicted"].to_numpy()),
    }

    return {
        "forecast_frame": forecast,
        "comparison": comparison,
        "future_predictions": future_forecast,
        "metrics": metrics,
    }


def create_lstm_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for index in range(sequence_length, len(values)):
        x_values.append(values[index - sequence_length:index, 0])
        y_values.append(values[index, 0])

    if not x_values:
        return np.array([]), np.array([])

    x_array = np.array(x_values)
    y_array = np.array(y_values)
    return x_array.reshape((x_array.shape[0], x_array.shape[1], 1)), y_array


@st.cache_resource(show_spinner=False)
def train_lstm_model(serialized_values: str, sequence_length: int, epochs: int) -> dict[str, Any]:
    close_values = np.array(pd.read_json(io.StringIO(serialized_values)).iloc[:, 0]).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_values)
    x_values, y_values = create_lstm_sequences(scaled_values, sequence_length)

    if len(x_values) < 50:
        raise ValueError("LSTM needs more data points to train reliably.")

    split_index = max(int(len(x_values) * 0.8), 1)
    x_train, x_test = x_values[:split_index], x_values[split_index:]
    y_train, y_test = y_values[:split_index], y_values[split_index:]

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    predicted_scaled = model.predict(x_test, verbose=0).flatten() if len(x_test) else np.array([])
    actual_scaled = y_test.flatten() if len(y_test) else np.array([])

    if len(predicted_scaled):
        predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
        actual = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()
    else:
        predicted = np.array([])
        actual = np.array([])

    return {
        "model": model,
        "scaler": scaler,
        "scaled_values": scaled_values,
        "split_index": split_index,
        "predicted": predicted,
        "actual": actual,
    }


def run_lstm_forecast(processed: pd.DataFrame, prediction_days: int) -> dict[str, Any]:
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed in this environment.")

    close_frame = processed[["Close"]].copy()
    serialized_values = close_frame.to_json()
    artifacts = train_lstm_model(serialized_values, LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS)
    scaler: MinMaxScaler = artifacts["scaler"]
    model = artifacts["model"]
    scaled_values = artifacts["scaled_values"]

    split_index = artifacts["split_index"]
    comparison_dates = processed["Date"].iloc[LSTM_SEQUENCE_LENGTH + split_index :].reset_index(drop=True)
    comparison = pd.DataFrame(
        {
            "ds": comparison_dates[: len(artifacts["actual"])],
            "Actual": artifacts["actual"],
            "Predicted": artifacts["predicted"],
        }
    )

    rolling_window = scaled_values[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, 1)
    future_scaled_predictions = []
    for _ in range(prediction_days):
        next_scaled = model.predict(rolling_window, verbose=0)[0][0]
        future_scaled_predictions.append(next_scaled)
        rolling_window = np.append(rolling_window[:, 1:, :], [[[next_scaled]]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_scaled_predictions).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(
        start=processed["Date"].iloc[-1] + timedelta(days=1),
        periods=prediction_days,
        freq="D",
    )
    future_frame = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Close": future_predictions,
            "Lower_Bound": future_predictions * 0.97,
            "Upper_Bound": future_predictions * 1.03,
        }
    )

    metrics = {
        "mae": float(mean_absolute_error(comparison["Actual"], comparison["Predicted"])) if not comparison.empty else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(comparison["Actual"], comparison["Predicted"])))
        if not comparison.empty
        else np.nan,
        "mape": safe_mape(comparison["Actual"].to_numpy(), comparison["Predicted"].to_numpy())
        if not comparison.empty
        else np.nan,
    }

    return {
        "comparison": comparison,
        "future_predictions": future_frame,
        "metrics": metrics,
    }


def train_model(model_name: str, processed: pd.DataFrame, prediction_days: int) -> dict[str, Any]:
    if model_name == "Prophet":
        return run_prophet_forecast(processed, prediction_days)
    if model_name == "LSTM":
        return run_lstm_forecast(processed, prediction_days)
    raise ValueError(f"Unsupported model selection: {model_name}")


def predict_prices(model_name: str, processed: pd.DataFrame, prediction_days: int) -> pd.DataFrame:
    model_output = train_model(model_name, processed, prediction_days)
    return model_output["future_predictions"]


def determine_trend(processed: pd.DataFrame) -> tuple[str, str]:
    latest = processed.iloc[-1]
    if latest["MA20"] > latest["MA50"]:
        return "Uptrend", "📈"
    return "Downtrend", "📉"


def build_historical_chart(processed: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=processed["Date"],
            y=processed["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#0b84ff", width=3),
        )
    )
    figure.update_layout(
        title="Historical Price Chart",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return figure


def build_ma_chart(processed: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=processed["Date"], y=processed["Close"], name="Close", line=dict(color="#194b69", width=2)))
    figure.add_trace(go.Scatter(x=processed["Date"], y=processed["MA20"], name="20-Day MA", line=dict(color="#12b886", width=2.5)))
    figure.add_trace(go.Scatter(x=processed["Date"], y=processed["MA50"], name="50-Day MA", line=dict(color="#f08c00", width=2.5)))
    figure.update_layout(
        title="Moving Averages",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return figure


def build_prediction_chart(processed: pd.DataFrame, comparison: pd.DataFrame, future_predictions: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": False}]])
    figure.add_trace(
        go.Scatter(
            x=processed["Date"],
            y=processed["Close"],
            name="Historical Close",
            mode="lines",
            line=dict(color="#0b84ff", width=2.5),
        )
    )

    if not comparison.empty:
        figure.add_trace(
            go.Scatter(
                x=comparison["ds"],
                y=comparison["Actual"],
                name="Actual (Validation)",
                mode="lines",
                line=dict(color="#495057", width=2),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=comparison["ds"],
                y=comparison["Predicted"],
                name="Predicted (Validation)",
                mode="lines",
                line=dict(color="#c2255c", width=2, dash="dash"),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=future_predictions["Date"],
            y=future_predictions["Predicted_Close"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#12b886", width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_predictions["Date"],
            y=future_predictions["Upper_Bound"],
            mode="lines",
            line=dict(color="rgba(18, 184, 134, 0.0)"),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_predictions["Date"],
            y=future_predictions["Lower_Bound"],
            mode="lines",
            line=dict(color="rgba(18, 184, 134, 0.0)"),
            fill="tonexty",
            fillcolor="rgba(18, 184, 134, 0.12)",
            name="Forecast Range",
        )
    )
    figure.update_layout(
        title="Actual vs Predicted Prices",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
    )
    return figure


def build_forecast_chart(future_predictions: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=future_predictions["Date"],
            y=future_predictions["Predicted_Close"],
            marker_color=np.where(
                future_predictions["Predicted_Close"].diff().fillna(0) >= 0,
                "#12b886",
                "#e64980",
            ),
            name="Predicted Close",
        )
    )
    figure.update_layout(
        title="Forecast Future Trend",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Forecast Date",
        yaxis_title="Predicted Price",
    )
    return figure


def plot_results(processed: pd.DataFrame, comparison: pd.DataFrame, future_predictions: pd.DataFrame) -> None:
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(build_historical_chart(processed), use_container_width=True)
    with right_col:
        st.plotly_chart(build_ma_chart(processed), use_container_width=True)

    st.plotly_chart(build_prediction_chart(processed, comparison, future_predictions), use_container_width=True)
    st.plotly_chart(build_forecast_chart(future_predictions), use_container_width=True)


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st_autorefresh(interval=30_000, key="stock-data-refresh")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AI Stock Price Predictor</div>
            <div class="hero-subtitle">
                Live market tracking, trend detection, and machine learning forecasts with automatic refresh every 30 seconds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    company_master = fetch_indian_company_master()
    default_label = "Infosys (INFY.NS)"
    if default_label not in company_master["label"].values:
        default_label = company_master["label"].iloc[0]
    live_company_count = int(company_master["ticker"].nunique())
    live_master_loaded = company_master["source"].eq("Live NSE equity master").any()

    with st.sidebar:
        st.header("Controls")
        input_mode = st.radio(
            "Ticker input",
            options=["Search live Indian company master", "Enter custom ticker"],
            index=0,
        )
        if input_mode == "Search live Indian company master":
            selected_company = st.selectbox(
                "Indian companies",
                options=company_master["label"].tolist(),
                index=company_master["label"].tolist().index(default_label),
            )
            ticker = company_master.loc[company_master["label"] == selected_company, "ticker"].iloc[0]
            if live_master_loaded:
                st.caption(
                    f"{live_company_count:,} live NSE-listed symbols loaded. BSE-only tickers can still be entered manually with `.BO`."
                )
            else:
                st.caption("Live exchange master is unavailable right now, so the app is using the bundled 110+ company fallback.")
        else:
            custom_ticker = st.text_input(
                "Custom stock ticker",
                value="INFY.NS",
                help="Examples: RELIANCE, TCS, INFY, RELIANCE.NS, TCS.NS, SBIN.BO",
            )
            ticker = normalize_ticker(custom_ticker)
        years = st.select_slider("Historical window", options=[1, 2, 3], value=2)
        prediction_days = st.slider("Prediction horizon (days)", min_value=7, max_value=30, value=15)
        model_options = ["Prophet"] + (["LSTM"] if TENSORFLOW_AVAILABLE else [])
        model_name = st.selectbox("Forecast model", options=model_options)
        st.caption("Data refreshes automatically every 30 seconds.")

        if model_name == "Prophet" and not PROPHET_AVAILABLE:
            st.warning("Prophet is not available in this environment. Install Prophet or switch to LSTM.")
        if not TENSORFLOW_AVAILABLE:
            st.info("LSTM is hidden on this machine because TensorFlow is not available for the current Python environment.")

    if not ticker:
        st.info("Enter a stock ticker in the sidebar to begin.")
        return

    try:
        with st.spinner("Fetching live market data..."):
            raw_data = fetch_data(ticker, years)
            live_snapshot = fetch_live_snapshot(ticker)

        processed = preprocess_data(raw_data)
        if len(processed) < MIN_HISTORY_ROWS:
            st.error("Insufficient historical data. Try a stock with at least 120 daily records.")
            return

        if model_name == "LSTM" and len(processed) < LSTM_SEQUENCE_LENGTH + 40:
            st.error("LSTM needs a longer history window. Increase the historical window or use Prophet.")
            return

        with st.spinner(f"Training {model_name} model and generating predictions..."):
            model_output = train_model(model_name, processed, prediction_days)
            future_predictions = model_output["future_predictions"].copy()
            comparison = model_output.get("comparison", pd.DataFrame())
            metrics = model_output["metrics"]

        current_price = float(live_snapshot["price"])
        latest_close = float(processed["Close"].iloc[-1])
        trend_label, trend_icon = determine_trend(processed)
        expected_price = float(future_predictions["Predicted_Close"].iloc[-1])
        expected_change_pct = ((expected_price - current_price) / current_price) * 100 if current_price else np.nan
        latest_return = float(processed["Daily_Return"].iloc[-1]) * 100
        last_updated = pd.to_datetime(live_snapshot["timestamp"]).strftime("%d %b %Y %I:%M:%S %p")

        trend_class = "trend-up" if trend_label == "Uptrend" else "trend-down"
        trend_text = f"{trend_icon} {trend_label}"

        st.markdown(
            f"""
            <div class="hero-card">
                <div class="data-note">Last updated at {last_updated}</div>
                <div class="trend-pill {trend_class}">{trend_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(5)
        metric_cols[0].metric(
            "Current Price",
            format_currency(current_price),
            delta=None if pd.isna(live_snapshot["change_pct"]) else f"{live_snapshot['change_pct']:.2f}%",
        )
        metric_cols[1].metric("Latest Close", format_currency(latest_close))
        metric_cols[2].metric("Expected Price", format_currency(expected_price))
        metric_cols[3].metric("Expected Change", f"{expected_change_pct:.2f}%")
        metric_cols[4].metric("Daily Return", f"{latest_return:.2f}%")

        accuracy_cols = st.columns(3)
        accuracy_cols[0].metric("MAE", f"{metrics['mae']:.2f}" if not pd.isna(metrics["mae"]) else "N/A")
        accuracy_cols[1].metric("RMSE", f"{metrics['rmse']:.2f}" if not pd.isna(metrics["rmse"]) else "N/A")
        accuracy_cols[2].metric("MAPE", f"{metrics['mape']:.2f}%" if not pd.isna(metrics["mape"]) else "N/A")

        summary_text = (
            "Predicted growth likely"
            if expected_change_pct >= 0
            else "Predicted decline likely"
        )
        summary_color = "#0f8c65" if expected_change_pct >= 0 else "#c2255c"
        st.markdown(
            f"<div class='data-note' style='font-weight:700;color:{summary_color};'>{summary_text}: {expected_change_pct:.2f}% over the next {prediction_days} days.</div>",
            unsafe_allow_html=True,
        )

        plot_results(processed, comparison, future_predictions)

        display_predictions = future_predictions.copy()
        display_predictions["Expected Change %"] = (
            (display_predictions["Predicted_Close"] - current_price) / current_price
        ) * 100
        display_predictions["Signal"] = np.where(
            display_predictions["Expected Change %"] >= 0,
            "Growth",
            "Decline",
        )
        display_predictions["Date"] = pd.to_datetime(display_predictions["Date"]).dt.strftime("%Y-%m-%d")

        st.subheader("Predicted Prices")
        st.dataframe(
            display_predictions.style.format(
                {
                    "Predicted_Close": "{:,.2f}",
                    "Lower_Bound": "{:,.2f}",
                    "Upper_Bound": "{:,.2f}",
                    "Expected Change %": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )

    except ValueError as error:
        st.error(f"Unable to process ticker `{ticker}`: {error}")
        st.info("Check the symbol and exchange suffix, then try again. Examples: INFY.NS, TCS.NS, AAPL, MSFT.")
    except ImportError as error:
        st.error(str(error))
        st.info("Install the missing package from requirements.txt, then rerun the app.")
    except Exception as error:
        st.error("Something unexpected happened while running the app.")
        st.exception(error)


if __name__ == "__main__":
    main()
