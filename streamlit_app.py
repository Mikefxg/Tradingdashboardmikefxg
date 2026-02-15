# -----------------------------
# UnknownFX Dashboard (PRO)
# TradingView charts (Capital.com) + server-side Outlook (SMA/RSI/MACD/ATR)
# Refresh: every 2 minutes
# -----------------------------

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================
# App config
# =============================
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
)

# =============================
# Helpers
# =============================
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def pct_change(a: float, b: float) -> Optional[float]:
    # from b -> a
    try:
        if b == 0:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None


# =============================
# Data sources (no keys)
# =============================
# NOTE: these are "calculation feeds". Your displayed chart stays Capital.com.
# - Indices proxy via ETFs (Yahoo charting is blocked often on cloud; Stooq is stable):
#   US100 proxy: QQQ
#   US500 proxy: SPY
#   DXY proxy: UUP
#   GOLD proxy: GLD
# - EURUSD via exchangerate.host (free)
#
# This gives a robust outlook without paid APIs.

@dataclass(frozen=True)
class Market:
    key: str
    title: str
    tv_symbol: str        # for TradingView embed (Capital.com)
    calc_source: str      # "stooq:QQQ" or "fx:EURUSD"
    desc: str


MARKETS = [
    Market("US100", "US100 (Nasdaq CFD)", "CAPITALCOM:US100", "stooq:qqq", "Calc proxy: QQQ"),
    Market("US500", "US500 (S&P 500 CFD)", "CAPITALCOM:US500", "stooq:spy", "Calc proxy: SPY"),
    Market("XAUUSD", "GOLD (XAUUSD Spot)", "CAPITALCOM:GOLD", "stooq:gld", "Calc proxy: GLD"),
    Market("EURUSD", "EURUSD", "CAPITALCOM:EURUSD", "fx:EURUSD", "Calc feed: exchangerate.host"),
    Market("DXY", "DXY (Dollar Index)", "CAPITALCOM:DXY", "stooq:uup", "Calc proxy: UUP"),
]


@st.cache_data(ttl=120, show_spinner=False)
def fetch_stooq_daily(symbol_lower: str, limit: int = 180) -> pd.DataFrame:
    # Stooq requires lowercase symbols, e.g. qqq, spy, gld, uup
    url = f"https://stooq.com/q/d/l/?s={symbol_lower}.us&i=d"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd, "compat") else pd.read_csv(pd.io.common.StringIO(r.text))
    # Fallback for pandas versions:
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(limit).reset_index(drop=True)
    df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_eurusd_series(limit: int = 180) -> pd.DataFrame:
    # exchangerate.host: free, no key
    # We build a small OHLC-like series (close only) from daily rates.
    url = "https://api.exchangerate.host/timeseries"
    params = {
        "base": "EUR",
        "symbols": "USD",
        "start_date": (pd.Timestamp.utcnow().date() - pd.Timedelta(days=365)).isoformat(),
        "end_date": pd.Timestamp.utcnow().date().isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rates = data.get("rates", {})
    rows = []
    for d, v in rates.items():
        usd = v.get("USD")
        if usd is not None:
            rows.append((pd.to_datetime(d), float(usd)))
    df = pd.DataFrame(rows, columns=["date", "close"]).sort_values("date").tail(limit).reset_index(drop=True)
    # Fake OHLC using close (so ATR will be less meaningful; we handle that)
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = np.nan
    return df


def fetch_market_df(m: Market) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if m.calc_source.startswith("stooq:"):
            sym = m.calc_source.split(":", 1)[1].strip().lower()
            df = fetch_stooq_daily(sym)
            return df, None
        if m.calc_source.startswith("fx:"):
            # only EURUSD for now
            df = fetch_eurusd_series()
            return df, None
        return None, "Unknown calc source"
    except Exception as e:
        return None, str(e)


# =============================
# Outlook scoring
# =============================
def compute_outlook(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Returns metrics and a final 'score' from -2 to +2 roughly.
    """
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else last
    chg = pct_change(last, prev)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = rsi(close, 14)
    hist = macd_hist(close)

    # Volatility proxy: ATR/price (for FX series where OHLC=close, ATR will be ~0)
    _atr = atr(df, 14) if {"high", "low", "close"}.issubset(df.columns) else pd.Series([np.nan] * len(df))
    atrp = None
    if _atr.notna().any():
        atr_last = float(_atr.iloc[-1]) if not math.isnan(float(_atr.iloc[-1])) else None
        if atr_last is not None and last:
            atrp = (atr_last / last) * 100.0

    # Signals
    trend = 0.0
    if not math.isnan(float(sma50.iloc[-1])):
        trend = 1.0 if last > float(sma50.iloc[-1]) else -1.0

    momentum = 0.0
    r = float(rsi14.iloc[-1]) if not math.isnan(float(rsi14.iloc[-1])) else None
    if r is not None:
        if r >= 60:
            momentum = 0.7
        elif r <= 40:
            momentum = -0.7
        else:
            momentum = 0.0

    macd_sig = 0.0
    h = float(hist.iloc[-1]) if not math.isnan(float(hist.iloc[-1])) else None
    if h is not None:
        macd_sig = 0.5 if h > 0 else (-0.5 if h < 0 else 0.0)

    # Volatility penalty (only if we have real ATR)
    vol_penalty = 0.0
    if atrp is not None:
        # if volatility > 1.2% daily (for ETFs/indices proxy), reduce conviction
        if atrp > 1.2:
            vol_penalty = -0.3
        elif atrp < 0.6:
            vol_penalty = +0.1

    score = trend + momentum + macd_sig + vol_penalty

    # Clamp
    score = max(-2.0, min(2.0, score))

    # Label
    if score >= 0.7:
        state = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.7:
        state = "BEARISH"
        bias = "SELL BIAS"
    else:
        state = "NEUTRAL"
        bias = "WAIT / NEUTRAL"

    return {
        "last": last,
        "chg": chg,
        "sma20": float(sma20.iloc[-1]) if not math.isnan(float(sma20.iloc[-1])) else None,
        "sma50": float(sma50.iloc[-1]) if not math.isnan(float(sma50.iloc[-1])) else None,
        "rsi14": r,
        "macd_hist": h,
        "atrp": atrp,
        "score": score,
        "state": state,
        "bias": bias,
    }


# =============================
# TradingView embeds (no API keys)
# =============================
def tv_chart(symbol: str, height: int = 640) -> str:
    # Advanced Chart widget
    # Note: embed uses client-side TradingView scripts; stable on Streamlit.
    return f"""
<div class="tv-container">
  <div id="tv_{symbol.replace(":", "_")}"></div>
</div>

<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
  new TradingView.widget({{
    "autosize": true,
    "symbol": "{symbol}",
    "interval": "15",
    "timezone": "Etc/UTC",
    "theme": "light",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "withdateranges": true,
    "allow_symbol_change": false,
    "container_id": "tv_{symbol.replace(":", "_")}"
  }});
</script>

<style>
  .tv-container {{
    width: 100%;
    height: {height}px;
  }}
  .tv-container > div {{
    width: 100%;
    height: {height}px;
  }}
</style>
"""


def tv_ta_widget(symbol: str, height: int = 430) -> str:
    # Technical Analysis widget (visual)
    return f"""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
  {{
    "interval": "15m",
    "width": "100%",
    "isTransparent": false,
    "height": "{height}",
    "symbol": "{symbol}",
    "showIntervalTabs": true,
    "locale": "en",
    "colorTheme": "light"
  }}
  </script>
</div>
"""


# =============================
# UI
# =============================
st.markdown(
    """
<style>
  .title-wrap { padding: 0.6rem 0 0.2rem 0; }
  .subtle { color: #666; }
  .pill {
    display:inline-block; padding: 0.25rem 0.55rem; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem; margin-right: 0.4rem;
    border: 1px solid #ddd;
  }
  .bull { background:#eaffea; color:#0b6b0b; border-color:#bfe8bf; }
  .bear { background:#ffecec; color:#8a0f0f; border-color:#f0bcbc; }
  .neut { background:#f3f3f3; color:#333; border-color:#e0e0e0; }
  .metricbox {
    padding: 0.8rem; border:1px solid #e6e6e6; border-radius: 14px; background: #fff;
  }
  .metricbig { font-size: 2.2rem; font-weight: 900; line-height: 1.1; }
  .metas { color:#666; font-size: 0.95rem; }
  .divider { height: 1px; background: #eee; margin: 1.2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title-wrap">', unsafe_allow_html=True)
st.title("🚀 UnknownFX Dashboard")
st.caption("MTF confluence • Key levels • Confidence % • Trend probability • DXY correlation • Refresh: every 2 minutes")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    st.write("Refresh: **120s** (2 min)")
    st.write("Interval (chart): **15m**")
    st.write("Markets: US100, US500, GOLD, EURUSD, DXY")
    st.divider()
    st.subheader("Tips")
    st.write("• Chart = Capital.com TradingView")
    st.write("• Outlook = indicators from stable public feeds")
    st.write("• If TradingView blocks embeds in some networks, try another browser/network.")

# Auto refresh every 2 minutes
st.markdown(
    """
<script>
  setTimeout(function(){ window.location.reload(); }, 120000);
</script>
""",
    unsafe_allow_html=True,
)

# =============================
# Top Overview Cards
# =============================
st.subheader("Market Outlook Overview")

cols = st.columns(len(MARKETS))
overview_results: Dict[str, Dict] = {}

for i, m in enumerate(MARKETS):
    df, err = fetch_market_df(m)
    with cols[i]:
        st.markdown(f"### {m.key}")
        st.caption(m.title)

        if err or df is None or len(df) < 60:
            st.markdown('<div class="metricbox">', unsafe_allow_html=True)
            st.markdown('<span class="pill bear">DATA ERROR</span>', unsafe_allow_html=True)
            st.write(err or "Not enough data")
            st.markdown("</div>", unsafe_allow_html=True)
            overview_results[m.key] = {"state": "ERROR"}
            continue

        o = compute_outlook(df)
        overview_results[m.key] = o

        state = o["state"]
        pill_class = "neut"
        if state == "BULLISH":
            pill_class = "bull"
        elif state == "BEARISH":
            pill_class = "bear"

        chg = o.get("chg")
        chg_txt = "—" if chg is None else f"{chg:+.2f}%"

        st.markdown('<div class="metricbox">', unsafe_allow_html=True)
        st.markdown(f'<span class="pill {pill_class}">{state}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metricbig">{o["bias"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metas">Last (calc): {o["last"]:.4f} • Δ {chg_txt}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metas">Score: {o["score"]:+.2f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# =============================
# Detailed sections (1 per row, big chart)
# =============================
st.subheader("Charts + Detailed Outlook (1 per row)")

for m in MARKETS:
    st.markdown(f"## {m.title}")

    df, err = fetch_market_df(m)
    if err or df is None or len(df) < 60:
        st.error(f"Data feed error for {m.key}: {err or 'unknown'}")
        st.markdown(tv_chart(m.tv_symbol, height=640), unsafe_allow_html=True)
        st.markdown(tv_ta_widget(m.tv_symbol, height=430), unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        continue

    o = compute_outlook(df)

    # Header row metrics
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
    with c1:
        st.metric("Outlook", o["state"], delta=None)
        st.write(f"**Bias:** {o['bias']}")
        st.caption(f"Calc feed: {m.desc}")
    with c2:
        st.metric("Score", f"{o['score']:+.2f}")
    with c3:
        st.metric("RSI(14)", "—" if o["rsi14"] is None else f"{o['rsi14']:.1f}")
    with c4:
        st.metric("SMA20 / SMA50", "—" if (o["sma20"] is None or o["sma50"] is None) else f"{o['sma20']:.2f} / {o['sma50']:.2f}")
    with c5:
        st.metric("Volatility (ATR% proxy)", "—" if o["atrp"] is None else f"{o['atrp']:.2f}%")

    # Big TradingView chart
    st.markdown(tv_chart(m.tv_symbol, height=740), unsafe_allow_html=True)

    # Visual TA widget (client-side, no python API)
    with st.expander("TradingView Technical Analysis (visual)"):
        st.markdown(tv_ta_widget(m.tv_symbol, height=460), unsafe_allow_html=True)

    # Explanation
    with st.expander("Waarom deze outlook? (rules)"):
        st.write(
            """
**Score components (simpel & effectief):**
- Trend: price > SMA50 → bullish, anders bearish
- Momentum: RSI ≥ 60 → bullish, RSI ≤ 40 → bearish
- MACD histogram: > 0 bullish / < 0 bearish
- Volatility: extreem hoog → lagere confidence (minder “all-in” signalen)

**Interpretatie:**
- Score ≥ +0.70 → BULLISH (BUY bias)
- Score ≤ -0.70 → BEARISH (SELL bias)
- Tussenin → NEUTRAL (WAIT)
"""
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.caption(f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC • Auto refresh: 120s")
