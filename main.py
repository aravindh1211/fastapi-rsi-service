from fastapi import FastAPI
import pandas as pd
import numpy as np
import yfinance as yf
import requests

app = FastAPI()

# --------------------------------------------------------------------------
# ACCURATE RSI CALCULATION (Matches TradingView)
# --------------------------------------------------------------------------
def calculate_rsi_accurate(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Wilder's RSI using the standard, optimized method (EMA).
    This is the method used by TradingView and other major platforms.
    """
    if series.size < period + 1:
        return pd.Series(dtype=np.float64)

    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing is an EMA with alpha = 1/period.
    # In pandas, com (center of mass) = period - 1 achieves this.
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.dropna()

# --------------------------------------------------------------------------
# DATA FETCHING FUNCTIONS (Restored to Simple, Working Versions)
# --------------------------------------------------------------------------

def get_stock_data(symbol: str, interval: str = "1d"):
    """
    Fetches stock data (US & NSE) using a simple, reliable yfinance call.
    """
    # Use a safe lookback period that works for both daily and intraday.
    period_map = {
        "1d": "2y", "5d": "5y", "1wk": "5y",
        "1h": "60d", "90m": "60d", "30m": "60d"
    }
    fetch_period = period_map.get(interval, "1y") # Default to 1 year
    
    data = yf.download(
        tickers=symbol,
        period=fetch_period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    return data

def get_crypto_data(symbol: str, interval: str = "1d"):
    """
    Fetches crypto data directly from Binance API.
    """
    try:
        # Convert symbol format (e.g., BTC-USD -> BTCUSDT)
        pair = symbol.upper().replace("-", "")
        if pair.endswith("USD"):
            pair = pair[:-3] + "USDT" # Binance prefers USDT for USD pairs
            
        # Map interval to Binance's format
        interval_map = {"1d": "1d", "4h": "4h", "1h": "1h", "1wk": "1w"}
        binance_interval = interval_map.get(interval, "1d")

        url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={binance_interval}&limit=500"
        
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset vol", "Trades", "TB Base vol", "TB Quote vol", "Ignore"
        ])
        df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
        df.set_index("Date", inplace=True)
        df["Close"] = df["Close"].astype(float)
        return df

    except Exception:
        return pd.DataFrame()

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "RSI service is running"}

@app.get("/rsi")
def get_rsi(symbol: str, period: int = 14, interval: str = "1d"):
    try:
        data = pd.DataFrame()
        # --- Simple Logic to Choose Data Source ---
        # If it contains a hyphen, it's crypto.
        if "-" in symbol:
            data = get_crypto_data(symbol, interval)
        # Otherwise, it's a stock.
        else:
            data = get_stock_data(symbol, interval)

        if data.empty or "Close" not in data.columns:
            return {"symbol": symbol, "error": f"No data found for symbol '{symbol}' on interval '{interval}'.", "interval": interval}
        
        if len(data) < period * 2: 
            return {"symbol": symbol, "error": f"Insufficient historical data for RSI (need >{period*2}, got {len(data)}).", "interval": interval}

        # Use the ACCURATE RSI calculation function
        rsi_series = calculate_rsi_accurate(data["Close"], period=period)

        if rsi_series.empty:
            return {"symbol": symbol, "error": "RSI calculation resulted in no values.", "interval": interval}

        latest_rsi = float(rsi_series.iloc[-1])
        return {
            "symbol": symbol,
            "rsi": round(latest_rsi, 2),
            "period": period,
            "interval": interval,
            "last_close_date": rsi_series.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"An unexpected error occurred: {str(e)}", "interval": interval}
