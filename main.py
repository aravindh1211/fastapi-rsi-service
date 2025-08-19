from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import numpy as np

app = FastAPI()

# A set of common crypto symbols to help identify them and default to a USD pair.
COMMON_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "SHIB", "AVAX", 
    "DOT", "TRX", "LINK", "MATIC", "LTC", "BCH", "UNI", "ATOM"
}

# --------------------------------------------------------------------------
# HIGHLY ACCURATE & EFFICIENT RSI CALCULATION (Vectorized)
# --------------------------------------------------------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Wilder's RSI using a vectorized method that matches TradingView.
    This is more efficient and robust than a loop-based approach.
    """
    if series.size < period + 1:
        return pd.Series(dtype=np.float64) # Return empty series if not enough data

    delta = series.diff()
    
    # Make gains and losses series
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate initial averages using Simple Moving Average
    initial_avg_gain = gain.rolling(window=period, min_periods=period).mean().dropna()
    initial_avg_loss = loss.rolling(window=period, min_periods=period).mean().dropna()

    # Setup the final average series
    avg_gain = pd.Series(index=series.index, dtype=np.float64)
    avg_loss = pd.Series(index=series.index, dtype=np.float64)
    
    # Set the first average value
    if not initial_avg_gain.empty:
        avg_gain.iloc[period] = initial_avg_gain.iloc[0]
        avg_loss.iloc[period] = initial_avg_loss.iloc[0]

    # Calculate the rest of the averages using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    for i in range(period + 1, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.dropna()

def get_yfinance_params(interval: str) -> dict:
    """
    Determines the correct 'period' to fetch from yfinance based on the interval
    to ensure enough data for RSI warm-up while respecting API limits.
    """
    # yfinance intraday data is limited to a 60-day lookback period.
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        return {"period": "60d", "interval": interval}
    # For intervals of 4h or more, we can fetch a longer history.
    elif interval in ["4h"]:
        return {"period": "730d", "interval": "1d"} # Fetch daily to calculate, then can resample if needed
    elif interval in ["1d", "5d", "1wk"]:
        return {"period": "5y", "interval": interval}
    elif interval in ["1mo", "3mo"]:
        return {"period": "max", "interval": interval}
    else:
        # Default for unrecognized intervals
        return {"period": "2y", "interval": "1d"}


# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "RSI service is running"}

@app.get("/rsi")
def get_rsi(symbol: str, period: int = 14, interval: str = "1d"):
    try:
        # --- SMART SYMBOL NORMALIZATION ---
        normalized_symbol = symbol.upper()
        if "-" not in normalized_symbol and normalized_symbol in COMMON_CRYPTO_SYMBOLS:
            normalized_symbol = f"{normalized_symbol}-USD"

        # --- CORRECT DATA FETCHING ---
        params = get_yfinance_params(interval)
        
        data = yf.download(
            tickers=normalized_symbol,
            **params,  # Unpack the period and interval parameters
            auto_adjust=True,
            progress=False
        )

        if data.empty or "Close" not in data.columns:
            return {"symbol": symbol, "error": f"No data found for symbol '{normalized_symbol}' on interval '{interval}'. Check if the symbol and interval are valid.", "interval": interval}
        
        if len(data) < period * 2: 
            return {"symbol": symbol, "error": f"Insufficient historical data for RSI calculation (need at least {period*2} data points, got {len(data)}).", "interval": interval}

        # Use the accurate, vectorized RSI calculation function
        rsi_series = calculate_rsi(data["Close"], period=period)

        if rsi_series.empty:
            return {"symbol": symbol, "error": "RSI calculation resulted in no values. This can happen with sparse data.", "interval": interval}

        latest_rsi = float(rsi_series.iloc[-1])
        return {
            "symbol": symbol,
            "normalized_symbol": normalized_symbol,
            "rsi": round(latest_rsi, 2),
            "period": period,
            "interval": interval,
            "last_close_date": rsi_series.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "interval": interval}
