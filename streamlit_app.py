import time
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")

st.title("📊 Trading Dashboard")
st.caption("Stable free data version • Indices via Stooq • EURUSD via exchangerate.host • XAUUSD coming soon")

refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 60, 600, 120, step=30)

# ---- Markets ----
INDICES = {
    "S&P 500 (SPX)": "spx",
    "Nasdaq 100 (NDX)": "ndx",
    "US30 (DJIA)": "djia",
}

FX = ["EURUSD"]
METALS = ["XAUUSD"]  # placeholder

# ---- Data fetchers ----
@st.cache_data(ttl=300)
def fetch_stooq_daily(symbol: str, days: int = 180) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))

    # Robust checks
    if df.empty:
        raise ValueError(f"No data returned from Stooq for symbol '{symbol}'")
    if "Date" not in df.columns:
        raise ValueError(f"Unexpected Stooq format for '{symbol}'. Columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.tail(days).reset_index(drop=True)

    if "Close" not in df.columns or df["Close"].dropna().empty:
        raise ValueError(f"No Close prices for '{symbol}'")

    return df

@st.cache_data(ttl=120)
def fetch_eurusd() -> float:
    # exchangerate.host is free; base EUR -> USD rate
    url = "https://api.exchangerate.host/latest"
    r = requests.get(url, params={"base": "EUR", "symbols": "USD"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    rate = js.get("rates", {}).get("USD")
    if rate is None:
        raise ValueError("EURUSD rate unavailable from exchangerate.host")
    return float(rate)

def calc_bias_from_close(close: pd.Series) -> str:
    close = close.dropna()
    if len(close) < 20:
        return "NEUTRAL"

    # simple momentum + slope
    mom = (close.iloc[-1] / close.iloc[-5] - 1) * 100
    y = close.tail(20).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    slope_pct = (slope / np.mean(y)) * 100

    score = 0
    if mom > 0.3:
        score += 1
    elif mom < -0.3:
        score -= 1

    if slope_pct > 0.05:
        score += 1
    elif slope_pct < -0.05:
        score -= 1

    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"

def bias_badge(label: str) -> str:
    if label == "BULLISH":
        return "🟢 BULLISH"
    if label == "BEARISH":
        return "🔴 BEARISH"
    return "⚪ NEUTRAL"

# ---- Cards ----
cols = st.columns(5)

cards = []

# Indices cards (3)
for name, sym in INDICES.items():
    try:
        df = fetch_stooq_daily(sym)
        close = df["Close"]
        price = float(close.iloc[-1])
        change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        bias = calc_bias_from_close(close)
        cards.append((name, price, change_pct, bias, df))
    except Exception as e:
        cards.append((name, None, None, "NEUTRAL", str(e)))

# EURUSD card
try:
    eurusd = fetch_eurusd()
    # fake tiny change (no previous from this endpoint) -> show none
    cards.append(("EURUSD", eurusd, None, "NEUTRAL", None))
except Exception as e:
    cards.append(("EURUSD", None, None, "NEUTRAL", str(e)))

# XAUUSD placeholder
cards.append(("XAUUSD", None, None, "NEUTRAL", "Free stable XAUUSD source not configured yet."))

# Render 5 cards
for i in range(5):
    name, price, change_pct, bias, extra = cards[i]
    with cols[i]:
        st.subheader(name)
        st.write(bias_badge(bias))
        if price is None:
            st.write("—")
            st.caption(f"⚠️ {extra}")
        else:
            if name in ["EURUSD"]:
                st.metric("Price", f"{price:.5f}", None)
            else:
                st.metric("Price", f"{price:,.2f}", f"{change_pct:+.2f}%" if change_pct is not None else None)

st.divider()

# ---- Chart ----
chart_market = st.selectbox("Select market for chart", list(INDICES.keys()), index=0)
dfc = fetch_stooq_daily(INDICES[chart_market])
st.line_chart(dfc.set_index("Date")["Close"])

st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} • Auto refresh every {refresh_seconds}s")
time.sleep(refresh_seconds)
st.rerun()
