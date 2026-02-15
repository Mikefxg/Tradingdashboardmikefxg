import streamlit as st
from datetime import datetime, timezone
from tradingview_ta import TA_Handler, Interval

# -----------------------------
# SETTINGS
# -----------------------------
st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

REFRESH_MINUTES = 10
INTERVAL = Interval.INTERVAL_15_MINUTES  # good balance for 10-min refresh

# Primary: FOREXCOM (TradingView broker feed)
# Fallback: TVC (indices), OANDA (fx/metals) if something fails on FOREXCOM
MARKETS = [
    # name, symbol, primary_exchange, screener, fallbacks (exchange, screener, symbol optional)
    ("Nasdaq 100", "NAS100", "FOREXCOM", "america", [("TVC", "america", "NDX")]),
    ("S&P 500",   "SPX500", "FOREXCOM", "america", [("TVC", "america", "SPX")]),
    ("US30",      "US30",   "FOREXCOM", "america", [("TVC", "america", "DJI")]),
    ("XAUUSD",    "XAUUSD", "FOREXCOM", "forex",   [("OANDA", "forex", "XAUUSD")]),
    ("EURUSD",    "EURUSD", "FOREXCOM", "forex",   [("OANDA", "forex", "EURUSD")]),
    # DXY can be tricky; FOREXCOM sometimes uses USDOLLAR; keep both attempts + TVC fallback
    ("DXY",       "DXY",    "FOREXCOM", "america", [("FOREXCOM", "america", "USDOLLAR"), ("TVC", "america", "DXY")]),
]

# -----------------------------
# HELPERS
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

@st.cache_data(ttl=REFRESH_MINUTES * 60, show_spinner=False)
def fetch_ta(symbol: str, exchange: str, screener: str, interval: Interval):
    h = TA_Handler(symbol=symbol, exchange=exchange, screener=screener, interval=interval)
    a = h.get_analysis()
    return a

def get_analysis_with_fallback(symbol, exchange, screener, interval, fallbacks):
    # Try primary
    try:
        a = fetch_ta(symbol, exchange, screener, interval)
        return a, (exchange, symbol)
    except Exception:
        pass
    # Try fallbacks
    for ex, sc, sym in fallbacks:
        try:
            a = fetch_ta(sym, ex, sc, interval)
            return a, (ex, sym)
        except Exception:
            continue
    raise RuntimeError("No working feed found for this market on configured exchanges.")

def sentiment_from_indicators(ind):
    """
    Sentiment op basis van "movement":
    - Trend: close vs SMA50
    - Momentum: RSI
    - Structure: SMA20 vs SMA50
    - MACD: MACD histogram sign (MACD - signal)
    Score -> Bullish/Neutral/Bearish
    """
    close = safe_float(ind.get("close"))
    sma20 = safe_float(ind.get("SMA20"))
    sma50 = safe_float(ind.get("SMA50"))
    rsi   = safe_float(ind.get("RSI"))
    macd  = safe_float(ind.get("MACD.macd"))
    macds = safe_float(ind.get("MACD.signal"))

    if close is None:
        return {
            "label": "NEUTRAL",
            "score": 0.0,
            "bias": "WAIT",
            "why": ["No close price available from feed."],
            "metrics": {}
        }

    score = 0.0
    why = []

    # Trend vs SMA50
    if sma50 is not None:
        if close > sma50:
            score += 0.9
            why.append("Price above SMA50 (trend up).")
        elif close < sma50:
            score -= 0.9
            why.append("Price below SMA50 (trend down).")

    # SMA20 vs SMA50 (structure)
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 0.6
            why.append("SMA20 above SMA50 (bull structure).")
        elif sma20 < sma50:
            score -= 0.6
            why.append("SMA20 below SMA50 (bear structure).")

    # RSI (momentum)
    if rsi is not None:
        if rsi >= 58:
            score += 0.5
            why.append(f"RSI {rsi:.1f} bullish momentum.")
        elif rsi <= 42:
            score -= 0.5
            why.append(f"RSI {rsi:.1f} bearish momentum.")
        else:
            why.append(f"RSI {rsi:.1f} neutral zone.")

    # MACD histogram sign
    if macd is not None and macds is not None:
        hist = macd - macds
        if hist > 0:
            score += 0.4
            why.append("MACD histogram positive.")
        elif hist < 0:
            score -= 0.4
            why.append("MACD histogram negative.")

    # Clamp score
    score = clamp(score, -2.0, 2.0)

    # Label
    if score >= 0.75:
        label = "BULLISH"
    elif score <= -0.75:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    # Bias suggestion
    # (Indicatie, geen “trade advice”: trend-following bias)
    if label == "BULLISH":
        bias = "BUY BIAS"
    elif label == "BEARISH":
        bias = "SELL BIAS"
    else:
        bias = "WAIT / RANGE"

    metrics = {
        "close": close,
        "SMA20": sma20,
        "SMA50": sma50,
        "RSI": rsi,
        "MACD_hist": (macd - macds) if (macd is not None and macds is not None) else None
    }

    return {"label": label, "score": score, "bias": bias, "why": why, "metrics": metrics}

def icon(label: str):
    return "🟢" if label == "BULLISH" else ("🔴" if label == "BEARISH" else "⚪️")

# -----------------------------
# UI
# -----------------------------
st.title("📈 Market Sentiment (TradingView × FOREXCOM)")
st.caption(f"Refresh elke {REFRESH_MINUTES} min • Interval: 15m • Feed: FOREXCOM (fallbacks actief)")

# Auto-refresh
st.markdown(
    f"""
    <script>
    setTimeout(function(){{
        window.location.reload();
    }}, {REFRESH_MINUTES * 60 * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

cols = st.columns(6)

for idx, (name, symbol, ex, scr, fallbacks) in enumerate(MARKETS):
    with cols[idx]:
        st.subheader(name)

        try:
            analysis, used = get_analysis_with_fallback(symbol, ex, scr, INTERVAL, fallbacks)
            ind = analysis.indicators or {}
            rec = analysis.summary.get("RECOMMENDATION") if analysis.summary else None

            s = sentiment_from_indicators(ind)

            st.markdown(f"### {icon(s['label'])} {s['label']}")
            st.metric("Bias", s["bias"], f"score {s['score']:+.2f}")

            # Core numbers
            m = s["metrics"]
            if m.get("close") is not None:
                st.write(f"**Last:** `{m['close']}`")
            if m.get("RSI") is not None:
                st.write(f"**RSI:** `{m['RSI']:.1f}`")
            if m.get("SMA20") is not None and m.get("SMA50") is not
