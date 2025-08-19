from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import requests

app = FastAPI()

# --------------------------------------------------------------------------
# ACCURATE RSI CALCULATION (Matching TradingView's Wilder's RSI)
# --------------------------------------------------------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) using the Wilder's smoothing method,
    which is standard on TradingView and other platforms.
    """
    delta = series.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the initial average gain and loss using a simple moving average
    avg_gain = gain.rolling(window=period, min_periods=period).mean()[:period]
    avg_loss = loss.rolling(window=period, min_periods=period).mean()[:period]
    
    # After the initial period, use Wilder's smoothing
    # avg_gain_prev = avg_gain.iloc[-1]
    # avg_loss_prev = avg_loss.iloc[-1]
    
    for i in range(period, len(series)):
        avg_gain = avg_gain.append(pd.Series([(avg_gain.iloc[-1] * (period - 1) + gain.iloc[i]) / period], index=[series.index[i]]))
        avg_loss = avg_loss.append(pd.Series([(avg_loss.iloc[-1] * (period - 1) + loss.iloc[i]) / period], index=[series.index[i]]))

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# --------------------------------------------------------------------------
# UNIFIED DATA FETCHER (Yahoo Finance & Binance)
# --------------------------------------------------------------------------
def get_stock_data(symbol: str, interval: str = "1d"):
    """
    Fetches historical data for both stocks (US, NSE) and crypto.
    - For stocks (e.g., 'AAPL', 'RELIANCE.NS'), uses yfinance.
    - For crypto (e.g., 'BTC-USD'), uses Binance API.
    
    Fetches enough data for RSI warm-up.
    """
    # 1. Handle Crypto Symbols (e.g., BTC-USD, ETH-USDT)
    if "-" in symbol:
        try:
            # Convert to Binance format (e.g., BTC-USD -> BTCUSDT)
            pair = symbol.replace("-", "")
            
            # Map common intervals to Binance format
            interval_map = {"1d": "1d", "4h": "4h", "1h": "1h", "1wk": "1w"}
            binance_interval = interval_map.get(interval, "1d")
            
            # Fetch at least 500 candles for proper RSI calculation
            url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={binance_interval}&limit=500"
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status() # Raise an exception for bad status codes
            data = resp.json()

            df = pd.DataFrame(data, columns=[
                "Open time", "Open", "High", "Low", "Close", "Volume",
                "Close time", "Quote asset vol", "Trades", "TB Base vol",
                "TB Quote vol", "Ignore"
            ])
            df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
            df.set_index("Date", inplace=True)
            df["Close"] = df["Close"].astype(float)
            return df

        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

    # 2. Handle Stock Symbols (e.g., AAPL, RELIANCE.NS)
    else:
        try:
            # yfinance requires a longer period to fetch enough data points for daily interval
            # Fetch ~2 years of data for daily, 60d for hourly, etc.
            period_map = {"1d": "2y", "1wk": "5y", "1h": "730d"}
            fetch_period = period_map.get(interval, "1y")

            data = yf.download(
                tickers=symbol,
                period=fetch_period,
                interval=interval,
                auto_adjust=True, # Recommended for simplicity
                progress=False
            )
            return data
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
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
        # Unified data fetching
        data = get_stock_data(symbol, interval)

        if data.empty or "Close" not in data.columns:
            return {"symbol": symbol, "error": f"No data found for interval '{interval}'.", "interval": interval}
        
        if len(data) < period * 2: # Check if there's enough data for a meaningful calculation
            return {"symbol": symbol, "error": "Insufficient historical data for RSI calculation.", "interval": interval}

        # Use the accurate RSI calculation function
        rsi_series = calculate_rsi(data["Close"], period=period).dropna()

        if rsi_series.empty:
            return {"symbol": symbol, "error": "Could not calculate RSI, not enough data after dropping NaNs.", "interval": interval}

        latest_rsi = float(rsi_series.iloc[-1])
        return {
            "symbol": symbol,
            "rsi": round(latest_rsi, 2),
            "period": period,
            "interval": interval,
            "last_close_date": rsi_series.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "interval": interval}
