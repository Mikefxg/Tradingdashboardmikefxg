import streamlit as st
from datetime import datetime
from tradingview_ta import TA_Handler, Interval

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trading Outlook Dashboard", layout="wide")

REFRESH_SECONDS = 60

# TradingView sources:
# - Indices via TVC (works reliably with tradingview_ta)
# - FX/Gold via OANDA
MARKETS = {
    "US100 (Nasdaq Index)": {"symbol": "NDX", "exchange": "TVC", "screener": "america"},
    "US30 (Dow Index)":     {"symbol": "DJI", "exchange": "TVC", "screener": "america"},
    "SPX500 (S&P Index)":   {"symbol": "SPX", "exchange": "TVC", "screener": "america"},
    "XAUUSD (Gold Spot)":   {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex"},
    "EURUSD":               {"symbol": "EURUSD", "exchange": "OANDA", "screener": "forex"},
    "DXY (Dollar Index)":   {"symbol": "DXY", "exchange": "TVC", "screener": "america"},
}

TA_SCORE_MAP = {
    "STRONG_BUY":  +2,
    "BUY":         +1,
    "NEUTRAL":      0,
    "SELL":        -1,
    "STRONG_SELL": -2,
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def rec_to_score(rec: str) -> int:
    return TA_SCORE_MAP.get((rec or "").upper(), 0)

def score_to_label(score: float):
    if score >= 0.75:
        return "BULLISH"
    if score <= -0.75:
        return "BEARISH"
    return "NEUTRAL"

def label_icon(label: str):
    return "🟢" if label == "BULLISH" else ("🔴" if label == "BEARISH" else "⚪️")

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_ta(symbol: str, exchange: str, screener: str, interval: Interval):
    handler = TA_Handler(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        interval=interval
    )
    analysis = handler.get_analysis()
    return {
        "recommendation": analysis.summary.get("RECOMMENDATION"),
        "indicators": analysis.indicators,
        "summary": analysis.summary,
    }

def volatility_proxy(indicators: dict):
    """
    Simple volatility proxy (0..1) using:
    - ADX (trend strength)
    - RSI distance from 50 (momentum)
    """
    adx = safe_float(indicators.get("ADX"))
    rsi = safe_float(indicators.get("RSI"))

    if adx is None and rsi is None:
        return None

    # ADX normalize roughly 10..40 -> 0..1
    adx_n = None
    if adx is not None:
        adx_n = clamp((adx - 10) / 30, 0, 1)

    # RSI distance normalize 0..25 -> 0..1
    rsi_n = None
    if rsi is not None:
        rsi_n = clamp(abs(rsi - 50) / 25, 0, 1)

    parts = [p for p in [adx_n, rsi_n] if p is not None]
    return sum(parts) / len(parts) if parts else None

def outlook_score(ta_rec: str, vol: float | None):
    """
    Outlook score -1..+1 from:
    - TA direction (main driver)
    - Volatility proxy adds small confidence
    """
    ta_s = rec_to_score(ta_rec)          # -2..+2
    ta_n = ta_s / 2.0                    # -1..+1

    # Vol term small: -0.1..+0.1
    vol_term = 0.0 if vol is None else (vol - 0.5) * 0.2

    score = (0.80 * ta_n) + (0.20 * vol_term)
    return clamp(score, -1.0, 1.0)

# -----------------------------
# UI
# -----------------------------
st.title("📊 Trading Outlook Dashboard")
st.caption(f"Auto refresh ~ elke {REFRESH_SECONDS}s • Spot/CFD via TradingView • Outlook: TA + Volatility (zonder nieuws)")

# Auto refresh
st.markdown(
    f"""
    <script>
    setTimeout(function(){{
        window.location.reload();
    }}, {REFRESH_SECONDS * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

cols = st.columns(len(MARKETS))

for i, (name, cfg) in enumerate(MARKETS.items()):
    with cols[i]:
        st.subheader(name)

        try:
            ta = fetch_ta(cfg["symbol"], cfg["exchange"], cfg["screener"], Interval.INTERVAL_1_HOUR)
            rec = ta["recommendation"] or "NEUTRAL"
            ind = ta["indicators"] or {}

            last = safe_float(ind.get("close"))
            rsi = safe_float(ind.get("RSI"))
            adx = safe_float(ind.get("ADX"))

            vol = volatility_proxy(ind)
            score = outlook_score(rec, vol)
            label = score_to_label(score)

            st.markdown(f"### {label_icon(label)} {label}")
            st.metric("Outlook score", f"{score:+.2f}")

            st.write(f"**TV Rec:** `{rec}`")
            if last is not None:
                st.write(f"**Last (TV):** `{last}`")
            if rsi is not None and adx is not None:
                st.write(f"**RSI:** `{rsi:.1f}` • **ADX:** `{adx:.1f}`")
            if vol is not None:
                st.write(f"**Volatility (proxy):** `{vol:.2f}`")

        except Exception as e:
            st.error(f"TA error: {e}")

st.divider()
st.caption(f"Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
