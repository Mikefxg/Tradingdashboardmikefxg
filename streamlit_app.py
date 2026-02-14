import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")

st.title("📊 Trading Dashboard")
st.caption("Stable free data version (Stooq only)")

STOOQ = {
    "S&P 500": "spx",
    "Nasdaq 100": "ndx",
    "US30 (Dow)": "djia",
    "XAUUSD": "xauusd",
    "EURUSD": "eurusd",
}

@st.cache_data(ttl=300)
def fetch_stooq(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

def calc_bias(series):
    if len(series) < 10:
        return "NEUTRAL"
    mom = (series.iloc[-1] / series.iloc[-5] - 1) * 100
    slope = np.polyfit(range(10), series.tail(10), 1)[0]

    if mom > 0.3 and slope > 0:
        return "BULLISH"
    elif mom < -0.3 and slope < 0:
        return "BEARISH"
    else:
        return "NEUTRAL"

cols = st.columns(5)

for i, (name, symbol) in enumerate(STOOQ.items()):
    with cols[i]:
        try:
            df = fetch_stooq(symbol)
            price = df["Close"].iloc[-1]
            change_pct = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            bias = calc_bias(df["Close"])

            st.subheader(name)
            st.metric("Price", f"{price:,.4f}" if "USD" in name else f"{price:,.2f}", f"{change_pct:+.2f}%")
            st.write("Bias:", bias)

        except Exception as e:
            st.write("Data unavailable")
            st.caption(str(e))

st.divider()

selected = st.selectbox("Select market for chart", list(STOOQ.keys()))
df = fetch_stooq(STOOQ[selected])
st.line_chart(df.set_index("Date")["Close"])

st.caption("Auto refresh every 60 seconds")
time.sleep(60)
st.rerun()
