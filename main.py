from fastapi import FastAPI
import yfinance as yf
import pandas as pd

app = FastAPI()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

@app.get("/")
def home():
    return {"status": "ok", "message": "RSI service running"}

@app.get("/rsi")
def get_rsi(symbol: str):
    try:
        data = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if data.empty:
            return {"symbol": symbol, "error": "no_data"}
        r = rsi(data["Close"]).dropna()
        if r.empty:
            return {"symbol": symbol, "error": "insufficient_data"}
        return {"symbol": symbol, "rsi": round(r.iloc[-1], 2)}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
