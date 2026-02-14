import time
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .badge {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600;
      }
      .bull { background:#0f5132; color:#d1e7dd; }
      .bear { background:#842029; color:#f8d7da; }
      .neutral { background:#41464b; color:#e2e3e5; }
      .subtle { color: rgba(255,255,255,0.65); font-size: 12px; }
      .title { font-size: 42px; font-weight: 800; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60, step=30)
lookback_minutes = st.sidebar.select_slider("Momentum lookback", options=[15, 30, 60, 120], value=60)
show_news = st.sidebar.checkbox("Show news", value=True)
news_limit = st.sidebar.slider("News items per market", 3, 10, 5)

st.sidebar.divider()
st.sidebar.caption("Gratis data = best effort. Voor echte realtime feeds heb je paid data nodig.")

# Auto-refresh
st.sidebar.write("⏱️ Last refresh:", datetime.now().strftime("%H:%M:%S"))
time.sleep(0.1)
st.autorefresh = st.empty()
# Streamlit has no built-in autorefresh in core; we do a lightweight rerun timer:
# We implement it by a small JS-free hack: rerun after a short sleep at bottom.
# (Works fine on Streamlit Community Cloud.)

# -----------------------------
# DATA SOURCES
# -----------------------------
# Stooq symbols (daily intraday not guaranteed) - good free fallback
STOOQ = {
    "S&P 500": "spx",
    "Nasdaq 100": "ndx",
    "US30 (Dow)": "djia",
}

# Yahoo symbols for live-ish quotes (can rate-limit)
YAHOO = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "US30 (Dow)": "^DJI",
    "XAUUSD": "XAUUSD=X",
    "EURUSD": "EURUSD=X",
}

MARKETS = ["S&P 500", "Nasdaq 100", "US30 (Dow)", "XAUUSD", "EURUSD"]

# GDELT query keywords per market
NEWS_QUERY = {
    "S&P 500": "S&P 500 OR SPX OR US stocks",
    "Nasdaq 100": "Nasdaq OR NDX OR tech stocks",
    "US30 (Dow)": "Dow Jones OR DJIA OR US stocks",
    "XAUUSD": "gold price OR XAUUSD OR bullion",
    "EURUSD": "EURUSD OR euro dollar OR ECB OR Fed",
}

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data(ttl=60)
def fetch_yahoo_quote(symbol: str):
    """
    Uses Yahoo's public quote endpoint. Not officially supported; best-effort.
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": symbol}, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteResponse", {}).get("result", [])
    return result[0] if result else None

@st.cache_data(ttl=300)
def fetch_stooq_daily(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Stooq daily CSV, free. Great fallback for trend/momentum if Yahoo fails.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd.compat, "StringIO") else pd.read_csv(pd.io.common.StringIO(r.text))
    # Ensure columns: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(days).reset_index(drop=True)
    return df

def calc_bias_from_series(prices: pd.Series) -> dict:
    """
    Simple daytrader-friendly bias:
    - Momentum: last vs lookback
    - Trend: slope over last N points
    Returns score and label.
    """
    prices = prices.dropna()
    if len(prices) < 5:
        return {"score": 0, "label": "NEUTRAL"}

    # momentum
    lb = min(len(prices)-1, max(3, int(len(prices) * 0.25)))
    mom = (prices.iloc[-1] / prices.iloc[-1-lb] - 1.0) * 100.0

    # slope
    y = prices.tail(min(20, len(prices))).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    slope_pct = (slope / np.mean(y)) * 100.0

    score = 0
    # momentum thresholds
    if mom > 0.15:
        score += 1
    elif mom < -0.15:
        score -= 1

    # slope thresholds
    if slope_pct > 0.03:
        score += 1
    elif slope_pct < -0.03:
        score -= 1

    if score >= 2:
        label = "BULLISH"
    elif score <= -2:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {"score": score, "label": label, "mom_pct": mom, "slope_pct": slope_pct}

def badge(label: str) -> str:
    if label == "BULLISH":
        return '<span class="badge bull">BULLISH</span>'
    if label == "BEARISH":
        return '<span class="badge bear">BEARISH</span>'
    return '<span class="badge neutral">NEUTRAL</span>'

@st.cache_data(ttl=300)
def fetch_gdelt_news(query: str, limit: int = 5):
    """
    GDELT 2 DOC API - free news search.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": limit,
        "sort": "HybridRel",
        "formatdate": "YmdHis",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    articles = js.get("articles", []) or []
    out = []
    for a in articles:
        out.append(
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("sourceCountry", "") or a.get("sourceCollection", ""),
                "datetime": a.get("seendate", ""),
            }
        )
    return out

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<p class="title">📊 Trading Dashboard</p>', unsafe_allow_html=True)
st.caption(f"Live-ish dashboard (auto refresh every {refresh_seconds}s) • Markets: SPX / NDX / US30 / XAUUSD / EURUSD")

# -----------------------------
# FETCH + DISPLAY
# -----------------------------
cols = st.columns(5)
market_cards = {}

for i, m in enumerate(MARKETS):
    with cols[i]:
        symbol = YAHOO[m]
        q = None
        err = None
        try:
            q = fetch_yahoo_quote(symbol)
        except Exception as e:
            err = str(e)

        price = None
        change = None
        change_pct = None

        if q:
            price = safe_float(q.get("regularMarketPrice"))
            change = safe_float(q.get("regularMarketChange"))
            change_pct = safe_float(q.get("regularMarketChangePercent"))

        # Fallback: use Stooq daily close for indices (not FX)
        series = None
        if price is None and m in STOOQ:
            try:
                df = fetch_stooq_daily(STOOQ[m], days=60)
                series = df["Close"]
                price = float(series.iloc[-1])
                # Daily change approximation
                if len(series) >= 2:
                    change = float(series.iloc[-1] - series.iloc[-2])
                    change_pct = float((series.iloc[-1] / series.iloc[-2] - 1) * 100)
            except Exception as e:
                err = err or str(e)

        # For bias calculation: prefer daily series fallback if no intraday
        if series is None and m in STOOQ:
            try:
                df = fetch_stooq_daily(STOOQ[m], days=60)
                series = df["Close"]
            except Exception:
                series = None

        bias = {"score": 0, "label": "NEUTRAL", "mom_pct": 0.0, "slope_pct": 0.0}
        if series is not None and len(series) >= 5:
            bias = calc_bias_from_series(series)

        # UI card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**{m}**  {badge(bias['label'])}", unsafe_allow_html=True)

        if price is None:
            st.write("—")
            st.caption("Data unavailable (free feed limit)")
        else:
            st.metric(
                label="Price",
                value=f"{price:,.4f}" if m in ["EURUSD", "XAUUSD"] else f"{price:,.2f}",
                delta=(f"{change_pct:+.2f}%" if change_pct is not None else None),
            )

        st.markdown(
            f'<div class="subtle">Score: {bias.get("score",0)} • Momentum: {bias.get("mom_pct",0):+.2f}% • Trend: {bias.get("slope_pct",0):+.2f}%</div>',
            unsafe_allow_html=True,
        )

        if err:
            st.caption(f"⚠️ {err[:90]}...")

        st.markdown("</div>", unsafe_allow_html=True)

        market_cards[m] = {"symbol": symbol, "quote": q, "bias": bias}

st.divider()

# -----------------------------
# DETAIL VIEW
# -----------------------------
tab1, tab2 = st.tabs(["📈 Charts", "📰 News"])

with tab1:
    left, right = st.columns([2, 1])

    with right:
        selected = st.selectbox("Select market", MARKETS, index=0)
        st.write(f"Yahoo symbol: `{YAHOO[selected]}`")

    with left:
        # Chart data: for indices from stooq; for FX from Yahoo is hard without paid/intraday endpoint
        if selected in STOOQ:
            try:
                df = fetch_stooq_daily(STOOQ[selected], days=180)
                df = df.rename(columns={"Date": "date", "Close": "close"})
                st.line_chart(df.set_index("date")["close"])
            except Exception as e:
                st.error(f"Chart unavailable: {e}")
        else:
            st.info("Free intraday history for FX/XAU is limited. Later we can plug a dedicated FX data source.")

with tab2:
    if not show_news:
        st.info("News is turned off in settings.")
    else:
        ncols = st.columns(2)
        for idx, m in enumerate(MARKETS):
            with ncols[idx % 2]:
                st.subheader(m)
                try:
                    items = fetch_gdelt_news(NEWS_QUERY[m], limit=news_limit)
                    if not items:
                        st.write("No articles found.")
                    for it in items:
                        title = it["title"] or "Untitled"
                        url = it["url"]
                        dt = it.get("datetime", "")
                        st.markdown(f"- [{title}]({url})  \n  <span class='subtle'>{dt}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.caption(f"⚠️ News unavailable: {e}")

# ---------------------------
