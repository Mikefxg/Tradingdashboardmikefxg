import time
from datetime import datetime
import requests
import numpy as np
import streamlit as st

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")

st.title("📊 Trading Dashboard")
st.caption("Stable cloud version • Finnhub data • Indices via ETF proxies (SPY/QQQ/DIA)")

# --- Settings
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 30, 600, 60, step=30)
st.sidebar.caption("Note: SPX/NDX/US30 are shown via ETF proxies (SPY/QQQ/DIA) for stability.")

# --- Secrets
API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
if not API_KEY:
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets. Add it in Manage app → Settings → Secrets.")
    st.stop()

# --- Markets (stable)
MARKETS = {
    "S&P 500 (proxy SPY)": "SPY",
    "Nasdaq 100 (proxy QQQ)": "QQQ",
    "US30 (proxy DIA)": "DIA",
    "EURUSD": "OANDA:EUR_USD",
    "XAUUSD": "OANDA:XAU_USD",  # if this returns empty on your plan, we can switch to GLD
}

def finnhub_quote(symbol: str) -> dict:
    r = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_quotes():
    out = {}
    for name, sym in MARKETS.items():
        try:
            q = finnhub_quote(sym)
            # Finnhub quote response: c=current, pc=prev close, dp=change %
            out[name] = {
                "symbol": sym,
                "price": q.get("c"),
                "prev": q.get("pc"),
                "change_pct": q.get("dp"),
            }
        except Exception as e:
            out[name] = {"symbol": sym, "error": str(e)}
    return out

def bias_from_change(dp):
    if dp is None:
        return "NEUTRAL"
    if dp > 0.25:
        return "BULLISH"
    if dp < -0.25:
        return "BEARISH"
    return "NEUTRAL"

def badge(label: str) -> str:
    return {"BULLISH": "🟢 BULLISH", "BEARISH": "🔴 BEARISH"}.get(label, "⚪ NEUTRAL")

quotes = get_quotes()

# --- Cards row
cols = st.columns(5)
for i, (name, sym) in enumerate(MARKETS.items()):
    with cols[i]:
        st.subheader(name)
        data = quotes.get(name, {})
        if "error" in data:
            st.write(badge("NEUTRAL"))
            st.write("—")
            st.caption(f"⚠️ {data['error'][:120]}")
            continue

        price = data.get("price")
        dp = data.get("change_pct")
        bias = bias_from_change(dp)

        st.write(badge(bias))

        if price is None or price == 0:
            st.write("—")
            st.caption("⚠️ No price returned")
        else:
            # formatting
            if "EURUSD" in name or "XAUUSD" in name:
                st.metric("Price", f"{price:.5f}", f"{dp:+.2f}%" if dp is not None else None)
            else:
                st.metric("Price", f"{price:,.2f}", f"{dp:+.2f}%" if dp is not None else None)

st.divider()

# --- Mini details panel (no candles yet)
selected = st.selectbox("Select market", list(MARKETS.keys()), index=0)
sel = quotes.get(selected, {})

st.subheader("Details")
st.write("Symbol:", MARKETS[selected])
if "error" in sel:
    st.error(sel["error"])
else:
    st.write("Price:", sel.get("price"))
    st.write("Prev close:", sel.get("prev"))
    st.write("Change %:", sel.get("change_pct"))

st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} • Auto refresh every {refresh_seconds}s")
time.sleep(refresh_seconds)
st.rerun()
