from fastapi import FastAPI
import yfinance as yf
import pandas as pd

app = FastAPI()

def rsi_wilder(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (Exponential moving average with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@app.get("/")
def home():
    return {"status": "ok", "message": "RSI service running"}

@app.get("/rsi")
def get_rsi(symbol: str, period: int = 14):
    try:
        # Fetch 1 year daily data, unadjusted
        data = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)

        if data.empty:
            return {"symbol": symbol, "error": "no_data"}

        rsi_series = rsi_wilder(data["Close"], period=period).dropna()

        if rsi_series.empty:
            return {"symbol": symbol, "error": "insufficient_data"}

        latest_rsi = float(rsi_series.iloc[-1])
        return {"symbol": symbol, "rsi": round(latest_rsi, 2), "period": period}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
