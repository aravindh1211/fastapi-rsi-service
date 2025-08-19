from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import requests

app = FastAPI()

# ----------------------------
# RSI calculation (Wilder's method)
# ----------------------------
def rsi_wilder(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder’s smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ----------------------------
# Get NSE data
# ----------------------------
def get_nse_data(symbol: str, lookback: str = "6mo"):
    # NSE API requires headers
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }

    url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from=2023-01-01&to=2024-12-31"
    try:
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=10)
        data = resp.json()["data"]

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.set_index("Date", inplace=True)
        df["Close"] = df["CH_CLOSING_PRICE"].astype(float)
        return df[["Close"]]
    except Exception:
        return pd.DataFrame()

# ----------------------------
# Get Crypto data (Binance)
# ----------------------------
def get_crypto_data(symbol: str, interval: str = "1d", lookback: str = "90d"):
    # Convert TradingView-style (BTC-USD) to Binance (BTCUSDT)
    base, quote = symbol.split("-")
    if quote == "USD":
        pair = base + "USDT"
    else:
        pair = base + quote

    binance_interval = interval.replace("m", "m").replace("h", "h").replace("d", "d").replace("wk", "w")

    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={binance_interval}&limit=500"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "Open time","Open","High","Low","Close","Volume",
            "Close time","Quote asset vol","Trades","TB Base vol",
            "TB Quote vol","Ignore"
        ])
        df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
        df.set_index("Date", inplace=True)
        df["Close"] = df["Close"].astype(float)
        return df[["Close"]]
    except Exception:
        return pd.DataFrame()

# ----------------------------
# API endpoints
# ----------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "RSI service running"}

@app.get("/rsi")
def get_rsi(symbol: str, period: int = 14, interval: str = "1d", lookback: str = "1y"):
    try:
        # Indian stock (.NS suffix expected from n8n)
        if symbol.endswith(".NS"):
            data = get_nse_data(symbol.replace(".NS", ""))
        # Crypto (BTC-USD, ETH-USD, etc.)
        elif "-" in symbol:
            data = get_crypto_data(symbol, interval=interval)
        # US/Global stock (AAPL, MSFT, TSLA)
        else:
            data = yf.download(symbol, period=lookback, interval=interval, auto_adjust=False, progress=False)

        if data.empty or "Close" not in data:
            return {"symbol": symbol, "error": "no_data", "interval": interval}

        rsi_series = rsi_wilder(data["Close"], period=period).dropna()
        if rsi_series.empty:
            return {"symbol": symbol, "error": "insufficient_data", "interval": interval}

        latest_rsi = float(rsi_series.iloc[-1])
        return {
            "symbol": symbol,
            "rsi": round(latest_rsi, 2),
            "period": period,
            "interval": interval
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "interval": interval}
