import time
from datetime import datetime
import requests
import streamlit as st

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")

st.title("📊 Trading Dashboard")
st.caption("Stable cloud version • Finnhub data • Indices via ETF proxies (SPY/QQQ/DIA)")

refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 30, 600, 60, step=30)
st.sidebar.caption("SPX/NDX/US30 via proxies: SPY/QQQ/DIA")

API_KEY = st.secrets.get("FINNHUB_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()

# --- Markets (stable)
MARKETS = {
    "S&P 500 (proxy SPY)": "SPY",
    "Nasdaq 100 (proxy QQQ)": "QQQ",
    "US30 (proxy DIA)": "DIA",
    "EURUSD": "OANDA:EUR_USD",
    "XAUUSD": "OANDA:XAU_USD",  # if blocked, we auto-fallback to GLD
}

def finnhub_quote(symbol: str) -> dict:
    r = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": API_KEY},
        timeout=20,
    )
    # if unauthorized, raise a clean error (no URL leak)
    if r.status_code == 401:
        raise RuntimeError("401 Unautho
