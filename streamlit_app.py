# streamlit_app.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from tradingview_ta import TA_Handler, Interval  # <-- GEEN TradingViewTAError import


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Market Sentiment (TradingView x Capital.com)", layout="wide")

DEFAULT_EXCHANGE = "CAPITALCOM"

DEFAULT_MARKETS = {
    "US100 (Nasdaq CFD)": {"symbol": "US100", "screener": "cfd"},
    "US500 (S&P CFD)": {"symbol": "US500", "screener": "cfd"},
    "US30 (Dow CFD)": {"symbol": "US30", "screener": "cfd"},
    "GOLD (XAUUSD)": {"symbol": "GOLD", "screener": "cfd"},
    "EURUSD": {"symbol": "EURUSD", "screener": "forex"},
    "DXY (US Dollar Index)": {"symbol": "DXY", "screener": "cfd"},
}


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class MarketResult:
    ok: bool
    reason: str
    score: float
    bias: str
    tv_rec: str
    last: Optional[float]
    rsi: Optional[float]
    adx: Optional[float]
    vol_pct: Optional[float]
    details: Dict[str, float]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def compute_score(ind: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Score in range ~[-3, +3]
    Based on:
    - Price vs SMA20/SMA50 (trend)
    - RSI (momentum)
    - MACD histogram (momentum)
    - ADX (trend strength) boosts trend signals
    - ATR% (volatility proxy) shown, not used heavily in bias
    """
    d: Dict[str, float] = {}

    close = _safe_float(ind.get("close"))
    sma20 = _safe_float(ind.get("SMA20"))
    sma50 = _safe_float(ind.get("SMA50"))
    rsi = _safe_float(ind.get("RSI"))
    adx = _safe_float(ind.get("ADX"))

    macd = _safe_float(ind.get("MACD.macd"))
    macd_signal = _safe_float(ind.get("MACD.signal"))
    macd_hist = None
    if macd is not None and macd_signal is not None:
        macd_hist = macd - macd_signal

    atr = _safe_float(ind.get("ATR"))
    vol_pct = None
    if atr is not None and close is not None and close != 0:
        vol_pct = (atr / close) * 100.0

    score = 0.0

    # Trend (price vs averages)
    if close is not None and sma20 is not None:
        s = 0.6 if close > sma20 else (-0.6 if close < sma20 else 0.0)
        score += s
        d["trend_vs_sma20"] = s

    if close is not None and sma50 is not None:
        s = 0.6 if close > sma50 else (-0.6 if close < sma50 else 0.0)
        score += s
        d["trend_vs_sma50"] = s

    # Momentum (RSI)
    if rsi is not None:
        # Map RSI: 50 is neutral, >60 bullish, <40 bearish
        if rsi >= 60:
            s = 0.6
        elif rsi <= 40:
            s = -0.6
        else:
            s = (rsi - 50.0) / 20.0  # [-0.5, +0.5]
        score += s
        d["rsi_signal"] = s

    # Momentum (MACD histogram)
    if macd_hist is not None:
        s = 0.6 if macd_hist > 0 else (-0.6 if macd_hist < 0 else 0.0)
        score += s
        d["macd_hist_signal"] = s

    # Trend strength boost (ADX)
    if adx is not None:
        # ADX >= 25 means trend is stronger -> amplify trend part a bit
        boost = 1.0
        if adx >= 25:
            boost = 1.15
        elif adx <= 15:
            boost = 0.9
        score *= boost
        d["adx_boost"] = boost

    # Keep within a nice band
    score = max(-3.0, min(3.0, score))
    if vol_pct is not None:
        d["vol_pct"] = vol_pct

    return score, d


def score_to_bias(score: float) -> str:
    if score >= 0.7:
        return "BULLISH"
    if score <= -0.7:
        return "BEARISH"
    return "NEUTRAL"


@st.cache_data(ttl=600)  # 10 min cache
def fetch_tv_analysis(symbol: str, exchange: str, screener: str, interval: str) -> Dict:
    handler = TA_Handler(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        interval=interval,
    )
    analysis = handler.get_analysis()
    return {
        "summary": dict(analysis.summary) if analysis.summary else {},
        "indicators": dict(analysis.indicators) if analysis.indicators else {},
    }


def tv_symbol(exchange: str, symbol: str) -> str:
    return f"{exchange}:{symbol}"


def tradingview_iframe(symbol_full: str, interval: str, height: int = 420) -> None:
    # TradingView widget embed (no key needed)
    # interval mapping to widget:
    interval_map = {
        Interval.INTERVAL_1_MINUTE: "1",
        Interval.INTERVAL_5_MINUTES: "5",
        Interval.INTERVAL_15_MINUTES: "15",
        Interval.INTERVAL_30_MINUTES: "30",
        Interval.INTERVAL_1_HOUR: "60",
        Interval.INTERVAL_4_HOURS: "240",
        Interval.INTERVAL_1_DAY: "D",
    }
    i = interval_map.get(interval, "15")

    html = f"""
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol={symbol_full}&interval={i}&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1"
      style="width: 100%; height: {height}px; border: 0; border-radius: 12px;"
      loading="lazy"
    ></iframe>
    """
    components.html(html, height=height + 20)


def badge(text: str) -> str:
    return f"<span style='padding:6px 10px;border-radius:999px;background:#f2f2f2;font-weight:700;'>{text}</span>"


def bias_badge(bias: str) -> str:
    if bias == "BULLISH":
        color = "#19a34a"
        bg = "#e9f9ef"
    elif bias == "BEARISH":
        color = "#d11a2a"
        bg = "#fdecee"
    else:
        color = "#444"
        bg = "#f3f4f6"
    return f"<span style='padding:8px 12px;border-radius:999px;background:{bg};color:{color};font-weight:900;'>{bias}</span>"


# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")

refresh_minutes = st.sidebar.number_input("Auto refresh (minutes)", min_value=1, max_value=60, value=10, step=1)
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

exchange = st.sidebar.text_input("TradingView exchange", value=DEFAULT_EXCHANGE)

interval_label = st.sidebar.selectbox(
    "Interval",
    options=["15m", "30m", "1h", "4h", "1D"],
    index=0,
)

interval_map = {
    "15m": Interval.INTERVAL_15_MINUTES,
    "30m": Interval.INTERVAL_30_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1D": Interval.INTERVAL_1_DAY,
}
interval = interval_map[interval_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Tickers (instelbaar)")

markets_cfg = {}
for name, cfg in DEFAULT_MARKETS.items():
    sym = st.sidebar.text_input(f"{name} • symbol", value=cfg["symbol"])
    scr = st.sidebar.selectbox(f"{name} • screener", options=["cfd", "forex"], index=(0 if cfg["screener"] == "cfd" else 1))
    markets_cfg[name] = {"symbol": sym.strip().upper(), "screener": scr}

st.sidebar.markdown("---")
show_charts = st.sidebar.checkbox("Show charts", value=True)
st.sidebar.caption("Tip: als je ‘Feed error’ krijgt, pas exchange/screener/symbol aan totdat TradingView TA het vindt.")

# -----------------------------
# UI
# -----------------------------
st.title("📈 Market Sentiment (TradingView × Capital.com)")
st.caption(f"Refresh elke {refresh_minutes} min • Interval: {interval_label} • Exchange: {exchange}")

# Summary row (global)
results: Dict[str, MarketResult] = {}

for name, cfg in markets_cfg.items():
    symbol = cfg["symbol"]
    screener = cfg["screener"]

    try:
        data = fetch_tv_analysis(symbol=symbol, exchange=exchange, screener=screener, interval=interval)
        ind = data.get("indicators", {})
        summ = data.get("summary", {})

        score, details = compute_score(ind)
        bias = score_to_bias(score)

        tv_rec = (summ.get("RECOMMENDATION") or "—").upper()
        last = _safe_float(ind.get("close"))
        rsi = _safe_float(ind.get("RSI"))
        adx = _safe_float(ind.get("ADX"))
        vol_pct = _safe_float(details.get("vol_pct"))

        results[name] = MarketResult(
            ok=True,
            reason="",
            score=score,
            bias=bias,
            tv_rec=tv_rec,
            last=last,
            rsi=rsi,
            adx=adx,
            vol_pct=vol_pct,
            details=details,
        )

    except Exception as e:
        results[name] = MarketResult(
            ok=False,
            reason=str(e),
            score=0.0,
            bias="NEUTRAL",
            tv_rec="—",
            last=None,
            rsi=None,
            adx=None,
            vol_pct=None,
            details={},
        )

# Cards grid
cols = st.columns(3)
i = 0
for name, cfg in markets_cfg.items():
    r = results[name]
    symbol_full = tv_symbol(exchange, cfg["symbol"])

    with cols[i % 3]:
        st.subheader(name)

        if not r.ok:
            st.error("Feed error")
            st.caption("No working feed found / TA not available for deze combinatie.")
            with st.expander("Debug (wat ging mis?)"):
                st.code(r.reason)
            st.caption(f"Probeer: exchange={exchange}, screener={cfg['screener']}, symbol={cfg['symbol']}")
        else:
            st.markdown(bias_badge(r.bias), unsafe_allow_html=True)
            st.caption(f"Bias: {'BUY' if r.bias=='BULLISH' else ('SELL' if r.bias=='BEARISH' else 'WAIT')} • score {r.score:+.2f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Last", f"{r.last:.5f}" if r.last is not None else "—")
            c2.metric("RSI", f"{r.rsi:.1f}" if r.rsi is not None else "—")
            c3.metric("ADX", f"{r.adx:.1f}" if r.adx is not None else "—")

            st.caption(f"TV Rec: **{r.tv_rec}** • Volatility (ATR%): **{r.vol_pct:.2f}%**" if r.vol_pct is not None else f"TV Rec: **{r.tv_rec}**")

            if show_charts:
                tradingview_iframe(symbol_full, interval)

            with st.expander("Waarom deze score?"):
                st.write(
                    {
                        "Trend vs SMA20": r.details.get("trend_vs_sma20"),
                        "Trend vs SMA50": r.details.get("trend_vs_sma50"),
                        "RSI signal": r.details.get("rsi_signal"),
                        "MACD hist signal": r.details.get("macd_hist_signal"),
                        "ADX boost": r.details.get("adx_boost"),
                        "ATR% (volatility)": r.details.get("vol_pct"),
                    }
                )

    i += 1

st.markdown("---")
st.caption(
    "Let op: dit is een **sentiment-indicator**, geen financieel advies. "
    "Als TradingView TA geen data teruggeeft voor een ticker op CAPITALCOM, probeer een andere symbol-naam of screener."
)
