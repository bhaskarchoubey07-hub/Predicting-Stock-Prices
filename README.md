# AI Stock Price Predictor

A production-oriented Streamlit web app for live Indian stock tracking and forecasting using `yfinance`, `Prophet`, and an optional `LSTM` path.

## Features

- Live stock data with automatic refresh every 30 seconds
- Searchable Indian stock universe using a live NSE company master with bundled fallback symbols
- Historical price analysis for 1 to 3 years
- Moving averages, daily returns, and trend detection
- Forecasting with Prophet (optimized for volatility with 0.15 changepoint scale)
- Deep LSTM support (128-64-32 unit architecture) for complex pattern recognition
- Interactive Plotly charts
- Prediction table with expected change and growth or decline signal
- Caching for faster reloads
- Error handling for invalid or unsupported symbols

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- .streamlit/
    `-- config.toml
```

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH, use:

```bash
python -m streamlit run app.py
```

## Deployment Notes

- The default forecast path is `Prophet`.
- The `LSTM` option is shown only when TensorFlow is available in the Python environment.
- On Python `3.14`, TensorFlow wheels may not be available yet, so the UI automatically hides LSTM in that case.
- The app is ready for Streamlit Community Cloud or GitHub-based deployment workflows.

## Advanced Forecasting Models

The app uses two high-performance models:
1. **Prophet**: Configured with a `changepoint_prior_scale` of `0.15` to capture rapid market shifts and high volatility.
2. **Deep LSTM**: A 3-layer neural network (128, 64, and 32 units) trained over 20 epochs for deep sequence learning.

## Local LSTM Setup

Use a separate Python `3.10` to `3.13` virtual environment for LSTM.

Example:

```bash
py -3.13 -m venv .venv-lstm
.venv-lstm\Scripts\activate
pip install -r requirements-lstm-local.txt
streamlit run app.py
```

If TensorFlow installs successfully in that environment, the app will automatically show the `LSTM` option again.

## GitHub Push

Initialize git and push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Data Sources

- NSE company master and market symbol metadata
- Yahoo Finance via `yfinance` for historical and latest price data

## Important Note

Live market data and exchange symbol discovery require network access at runtime. If the live NSE company master is unavailable, the app falls back to a bundled large-cap Indian stock list so the UI remains usable.
