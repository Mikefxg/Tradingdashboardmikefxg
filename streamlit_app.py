import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import streamlit as st

# TradingView TA library (reads TradingView technicals, not broker API)
from tradingview_ta import TA_Handler, Interval, Exchange, TradingView

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Market Sentiment (TradingView x Capital.com)", layout="wide")

REFRESH_MINUTES = 10
INTERVAL = Interval.INTERVAL_15_MINUTES  # 15m is a good balance for intraday bias

# IMPORTANT:
# You MUST set these to the exact TradingView tickers you see (Exchange:Symbol).
# Examples (may differ!):
#   "CAPITALCOM:US100" or "CAPITALCOM:NAS100"
#   "CAPITALCOM:US500" or "CAPITALCOM:SPX500"
# If a symbol fails: open TradingView, search the instrument, copy the exact ticker.

MARKETS = {
    "US100 (Nasdaq CFD)": "CAPITALCOM:US100",
    "US500 (S&P CFD)": "CAPITALCOM:US500",
    "US30 (Dow CFD)": "CAPITALCOM:US30",
    "XAUUSD (Gold Spot)": "CAPITALCOM:XAUUSD",
    "EURUSD": "CAPITALCOM:EURUSD",
    # DXY often not available on broker feeds; keep Capitalcom first and fallback to TVC:DXY
    "DXY (Dollar Index)": "CAPITALCOM:DXY|TVC:DXY",
}

# -----------------------------
# HELPERS
# -----------------------------
@dataclass
class Outlook:
    label: str            # BULLISH / BEARISH / NEUTRAL
    bias: str             # BUY BIAS / SELL BIAS / WAIT
    score: float
    details: Dict[str, float]
    tv_recommendation: str
    last: Optional[float]
    error: Optional[str] = None


def parse_tv_ticker(ticker: str) -> Tuple[str, str]:
    """Split EXCHANGE:SYMBOL"""
    if ":" not in ticker:
        # If user gave only symbol, assume CAPITALCOM (best guess)
        return "CAPITALCOM", ticker
    ex, sym = ticker.split(":", 1)
    return ex.strip().upper(), sym.strip().upper()


def get_ta(exchange: str, symbol: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener="forex",  # works for most; TradingView still resolves based on exchange
            interval=INTERVAL,
            timeout=10,
        )
        analysis = handler.get_analysis()
        return {
            "summary": analysis.summary,
            "indicators": analysis.indicators,
        }, None
    except Exception as e:
        return None, str(e)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_outlook(ta: dict) -> Outlook:
    ind = ta["indicators"]
    summ = ta["summary"] or {}
    tv_rec = (summ.get("RECOMMENDATION") or "NEUTRAL").upper()

    rsi = safe_float(ind.get("RSI"))
    adx = safe_float(ind.get("ADX"))
    macd = safe_float(ind.get("MACD.macd"))
    macd_signal = safe_float(ind.get("MACD.signal"))
    close = safe_float(ind.get("close"))

    sma20 = safe_float(ind.get("SMA20"))
    sma50 = safe_float(ind.get("SMA50"))
    ema20 = safe_float(ind.get("EMA20"))
    ema50 = safe_float(ind.get("EMA50"))

    # Score components (simple + robust)
    score = 0.0
    details = {}

    # Trend via moving averages
    trend_score = 0.0
    if close and sma20 and sma50:
        if close > sma20 > sma50:
            trend_score += 0.8
        elif close < sma20 < sma50:
            trend_score -= 0.8
        elif close > sma50:
            trend_score += 0.3
        elif close < sma50:
            trend_score -= 0.3

    # fallback to EMA if SMA missing
    if trend_score == 0.0 and close and ema20 and ema50:
        if close > ema20 > ema50:
            trend_score += 0.7
        elif close < ema20 < ema50:
            trend_score -= 0.7

    details["trend"] = trend_score
    score += trend_score

    # RSI momentum
    rsi_score = 0.0
    if rsi is not None:
        if rsi >= 60:
            rsi_score += 0.4
        elif rsi <= 40:
            rsi_score -= 0.4
        else:
            rsi_score += 0.0
    details["rsi"] = rsi_score
    score += rsi_score

    # MACD histogram sign
    macd_score = 0.0
    if macd is not None and macd_signal is not None:
        hist = macd - macd_signal
        if hist > 0:
            macd_score += 0.35
        elif hist < 0:
            macd_score -= 0.35
    details["macd"] = macd_score
    score += macd_score

    # ADX = strength multiplier (not direction)
    strength = 1.0
    if adx is not None:
        if adx >= 25:
            strength = 1.15
        elif adx <= 15:
            strength = 0.85
    details["strength_mult"] = strength

    score *= strength

    # Nudge score using TradingView recommendation
    tv_nudge = 0.0
    if "STRONG_BUY" in tv_rec or tv_rec == "BUY":
        tv_nudge = 0.25
    elif "STRONG_SELL" in tv_rec or tv_rec == "SELL":
        tv_nudge = -0.25
    details["tv_nudge"] = tv_nudge
    score += tv_nudge

    # Final label
    if score >= 0.55:
        label = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.55:
        label = "BEARISH"
        bias = "SELL BIAS"
    else:
        label = "NEUTRAL"
        bias = "WAIT / NO CLEAR EDGE"

    return Outlook(
        label=label,
        bias=bias,
        score=round(score, 2),
        details={
            "RSI": round(rsi, 1) if rsi is not None else None,
            "ADX": round(adx, 1) if adx is not None else None,
            "SMA20": round(sma20, 5) if sma20 is not None else None,
            "SMA50": round(sma50, 5) if sma50 is not None else None,
            "Last": round(close, 5) if close is not None else None,
        },
        tv_recommendation=tv_rec,
        last=close,
    )


def tv_widget(symbol: str, title: str):
    # TradingView embedded chart (works if symbol string is correct)
    # symbol must be like "CAPITALCOM:US100"
    return f"""
    <div class="tradingview-widget-container" style="height:420px;">
      <div id="tv_{title.replace(" ", "_")}" style="height:420px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": 420,
          "symbol": "{symbol}",
          "interval": "15",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_side_toolbar": false,
          "details": true,
          "container_id": "tv_{title.replace(" ", "_")}"
        }});
      </script>
    </div>
    """


@st.cache_data(ttl=REFRESH_MINUTES * 60)
def load_market_outlooks() -> Dict[str, Outlook]:
    results: Dict[str, Outlook] = {}

    for name, ticker in MARKETS.items():
        # support fallback like "CAPITALCOM:DXY|TVC:DXY"
        candidates = [t.strip() for t in ticker.split("|") if t.strip()]
        last_err = None
        ta_data = None

        for cand in candidates:
            ex, sym = parse_tv_ticker(cand)
            data, err = get_ta(ex, sym)
            if data:
                ta_data = data
                last_err = None
                # store which symbol succeeded
                results[name] = compute_outlook(ta_data)
                results[name].details["Feed"] = cand
                break
            last_err = err

        if ta_data is None:
            results[name] = Outlook(
                label="—",
                bias="—",
                score=0.0,
                details={"Feed": candidates[0] if candidates else ticker},
                tv_recommendation="—",
                last=None,
                error=f"No working feed. Last error: {last_err}",
            )

    return results


def pill(label: str):
    if label == "BULLISH":
        st.markdown("🟢 **BULLISH**")
    elif label == "BEARISH":
        st.markdown("🔴 **BEARISH**")
    elif label == "NEUTRAL":
        st.markdown("⚪ **NEUTRAL**")
    else:
        st.markdown("⚠️ **FEED ERROR**")


# -----------------------------
# UI
# -----------------------------
st.title("📈 Market Sentiment (TradingView × Capital.com)")
st.caption(f"Refresh elke {REFRESH_MINUTES} min • Interval: 15m • Charts: TradingView widget • Score: MA + RSI + MACD + ADX + TV rec")

outlooks = load_market_outlooks()

# Summary row
cols = st.columns(len(MARKETS))
for (name, _), col in zip(MARKETS.items(), cols):
    o = outlooks[name]
    with col:
        st.subheader(name.split(" (")[0])
        if o.error:
            st.error("Feed error")
            st.caption(o.error)
            st.caption(f"Feed: {o.details.get('Feed')}")
        else:
            pill(o.label)
            st.markdown(f"**{o.bias}**")
            st.metric("Score", f"{o.score:+.2f}")
            st.caption(f"TV: {o.tv_recommendation} • Last: {o.last}")

st.divider()

# Charts + details
st.subheader("Charts & details")

# Show 2 rows of charts
names = list(MARKETS.keys())
row1 = names[:3]
row2 = names[3:]

for row in (row1, row2):
    c = st.columns(len(row))
    for name, col in zip(row, c):
        ticker = MARKETS[name].split("|")[0].strip()  # chart uses primary symbol
        with col:
            st.markdown(f"### {name}")
            st.components.v1.html(tv_widget(ticker, name), height=460)
            o = outlooks[name]
            if o.error:
                st.warning(o.error)
            else:
                st.write(
                    {
                        "Outlook": o.label,
                        "Bias": o.bias,
                        "Score": o.score,
                        "TV Rec": o.tv_recommendation,
                        "Last": o.last,
                        "RSI": o.details.get("RSI"),
                        "ADX": o.details.get("ADX"),
                        "SMA20": o.details.get("SMA20"),
                        "SMA50": o.details.get("SMA50"),
                        "Feed used": o.details.get("Feed"),
                    }
                )

st.info(
    "Tip: Als je 'Feed error' krijgt bij US100/US500/US30/XAUUSD/EURUSD, "
    "dan klopt de ticker niet. Open TradingView → zoek instrument → kopieer exact 'EXCHANGE:SYMBOL' "
    "(bv. CAPITALCOM:US100 of CAPITALCOM:NAS100) en vervang bovenaan in MARKETS."
)
