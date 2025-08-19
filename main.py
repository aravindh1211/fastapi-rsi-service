from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import numpy as np
import requests # Import the requests library

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
    """
    if series.size < period + 1:
        return pd.Series(dtype=np.float64)

    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use Exponential Moving Average (EMA) with alpha = 1/period for Wilder's smoothing
    # com (center of mass) = period - 1 for alpha = 1 / (1 + com)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.dropna()


def get_yfinance_params(interval: str) -> dict:
    """
    Determines the correct 'period' to fetch from yfinance based on the interval
    to ensure enough data while respecting API limits.
    """
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        return {"period": "60d", "interval": interval}
    if interval in ["4h"]: # yfinance doesn't have 4h, so we can't reliably use it
        return {"period": "2y", "interval": "1d"} # Fallback to daily
    if interval in ["1d", "5d", "1wk"]:
        return {"period": "5y", "interval": interval}
    if interval in ["1mo", "3mo"]:
        return {"period": "max", "interval": interval}
    return {"period": "2y", "interval": "1d"} # Default


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

        # --- CREATE A SESSION WITH A BROWSER USER-AGENT ---
        # This is the crucial fix for cloud/server environments like Render
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        })
        
        # --- CORRECT DATA FETCHING ---
        params = get_yfinance_params(interval)
        
        data = yf.download(
            tickers=normalized_symbol,
            **params,
            auto_adjust=True,
            progress=False,
            session=session # Pass the custom session here
        )

        if data.empty:
            return {"symbol": symbol, "error": f"No data returned for symbol '{normalized_symbol}' on interval '{params['interval']}'. This can be due to an invalid symbol, delisting, or network issues.", "interval": interval}
        
        if len(data) < period * 2: 
            return {"symbol": symbol, "error": f"Insufficient historical data for RSI (need >{period*2}, got {len(data)}).", "interval": interval}

        # Use the accurate, vectorized RSI calculation function
        rsi_series = calculate_rsi(data["Close"], period=period)

        if rsi_series.empty:
            return {"symbol": symbol, "error": "RSI calculation resulted in no values.", "interval": interval}

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
        return {"symbol": symbol, "error": f"An unexpected error occurred: {str(e)}", "interval": interval}
