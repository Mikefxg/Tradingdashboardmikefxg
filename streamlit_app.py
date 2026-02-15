import streamlit as st
from datetime import datetime, timezone
from tradingview_ta import TA_Handler, Interval

# -----------------------------
# SETTINGS
# -----------------------------
st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

REFRESH_MINUTES = 10
INTERVAL = Interval.INTERVAL_15_MINUTES  # good balance with 10-min refresh

# Primary: FOREXCOM (TradingView broker feed)
# Fallback: TVC / OANDA if a symbol isn't available on FOREXCOM in TradingView TA
MARKETS = [
    ("Nasdaq 100", "NAS100", "FOREXCOM", "america", [("TVC", "america", "NDX")]),
    ("S&P 500",    "SPX500", "FOREXCOM", "america", [("TVC", "america", "SPX")]),
    ("US30",       "US30",   "FOREXCOM", "america", [("TVC", "america", "DJI")]),
    ("XAUUSD",     "XAUUSD", "FOREXCOM", "forex",   [("OANDA", "forex", "XAUUSD")]),
    ("EURUSD",     "EURUSD", "FOREXCOM", "forex",   [("OANDA", "forex", "EURUSD")]),
    ("DXY",        "DXY",    "FOREXCOM", "america", [("FOREXCOM", "america", "USDOLLAR"), ("TVC", "america", "DXY")]),
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
    handler = TA_Handler(symbol=symbol, exchange=exchange, screener=screener, interval=interval)
    return handler.get_analysis()

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
    Sentiment op basis van chart movement:
    - Trend: close vs SMA50
    - Structure: SMA20 vs SMA50
    - Momentum: RSI
    - MACD histogram sign
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

    # Structure: SMA20 vs SMA50
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 0.6
            why.append("SMA20 above SMA50 (bull structure).")
        elif sma20 < sma50:
            score -= 0.6
            why.append("SMA20 below SMA50 (bear structure).")

    # Momentum: RSI
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

    score = clamp(score, -2.0, 2.0)

    if score >= 0.75:
        label = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.75:
        label = "BEARISH"
        bias = "SELL BIAS"
    else:
        label = "NEUTRAL"
        bias = "WAIT / RANGE"

    metrics = {
        "close": close,
        "SMA20": sma20,
        "SMA50": sma50,
        "RSI": rsi,
        "MACD_hist": (macd - macds) if (macd is not None and macds is not None) else None,
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

            m = s["metrics"]
            if m.get("close") is not None:
                st.write(f"**Last:** `{m['close']}`")
            if m.get("RSI") is not None:
                st.write(f"**RSI:** `{m['RSI']:.1f}`")
            if m.get("SMA20") is not None and m.get("SMA50") is not None:
                st.write(f"**SMA20/SMA50:** `{m['SMA20']:.2f}` / `{m['SMA50']:.2f}`")
            if m.get("MACD_hist") is not None:
                st.write(f"**MACD hist:** `{m['MACD_hist']:+.4f}`")

            if rec:
                st.caption(f"TradingView TA: **{rec}** • feed: `{used[0]}:{used[1]}`")

            with st.expander("Waarom deze score?"):
                for line in s["why"]:
                    st.write("• " + line)

        except Exception as e:
            st.error("Feed error")
            st.caption(str(e))

st.divider()
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Last refresh: {now}")
