from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "🚀 UnknownFX Dashboard — PRO++"
DEFAULT_REFRESH_SEC = 120  # 2 minutes

st.set_page_config(page_title="UnknownFX Dashboard", page_icon="🚀", layout="wide")


# =========================================================
# UI STYLE
# =========================================================
st.markdown(
    """
<style>
  .subtle { color:#6b6b6b; }
  .pill{
    display:inline-block; padding:0.28rem 0.62rem; border-radius:999px;
    font-weight:800; font-size:0.85rem; border:1px solid #e6e6e6; margin-right:.4rem;
  }
  .bull{ background:#eaffea; color:#0b6b0b; border-color:#bfe8bf; }
  .bear{ background:#ffecec; color:#8a0f0f; border-color:#f0bcbc; }
  .neut{ background:#f3f3f3; color:#333; border-color:#e0e0e0; }
  .warn{ background:#fff6d7; color:#7a5b00; border-color:#ffe08a; }

  .card{
    padding:0.9rem; border:1px solid #eaeaea; border-radius:16px; background:#fff;
  }
  .big{ font-size:2.1rem; font-weight:900; line-height:1.05; }
  .mid{ font-size:1.05rem; font-weight:700; }
  .meta{ color:#666; font-size:0.95rem; }
  .divider{ height:1px; background:#efefef; margin:1.2rem 0; }

  .tvwrap{ width:100%; border:1px solid #ededed; border-radius:16px; overflow:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MARKET MAP (Capital.com TradingView symbols)
# =========================================================
@dataclass(frozen=True)
class Market:
    key: str
    title: str
    tv_symbol: str     # TradingView embed symbol
    calc_source: str   # server-side calc source (stable, no key)
    desc: str


# NOTE: calc_source uses proxies where needed; chart is always Capital.com TradingView.
MARKETS = [
    Market("US100", "US100 (Nasdaq CFD)", "CAPITALCOM:US100", "stooq:qqq", "Calc proxy: QQQ"),
    Market("US500", "US500 (S&P 500 CFD)", "CAPITALCOM:US500", "stooq:spy", "Calc proxy: SPY"),
    Market("XAUUSD", "GOLD (XAUUSD Spot)", "CAPITALCOM:GOLD", "stooq:gld", "Calc proxy: GLD"),
    Market("EURUSD", "EURUSD", "CAPITALCOM:EURUSD", "fx:EURUSD", "Calc feed: exchangerate.host"),
    Market("DXY", "DXY (Dollar Index)", "CAPITALCOM:DXY", "stooq:uup", "Calc proxy: UUP"),
]


# =========================================================
# INDICATORS
# =========================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def pct_change(last: float, prev: float) -> Optional[float]:
    if prev == 0 or prev is None or last is None:
        return None
    return (last - prev) / prev * 100.0


# =========================================================
# DATA FETCH (NO KEYS)
# =========================================================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_stooq_daily(symbol_lower: str, limit: int = 260) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol_lower}.us&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]

    # Normalize columns
    if "date" not in df.columns:
        raise ValueError("Stooq response missing date column")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(limit).reset_index(drop=True)

    # Ensure OHLC exists
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        raise ValueError("Stooq response missing OHLC columns")

    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_eurusd_series(limit: int = 260) -> pd.DataFrame:
    url = "https://api.exchangerate.host/timeseries"
    params = {
        "base": "EUR",
        "symbols": "USD",
        "start_date": (pd.Timestamp.utcnow().date() - pd.Timedelta(days=400)).isoformat(),
        "end_date": pd.Timestamp.utcnow().date().isoformat(),
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()

    rates = data.get("rates", {})
    rows = []
    for d, v in rates.items():
        usd = v.get("USD")
        if usd is not None:
            rows.append((pd.to_datetime(d), float(usd)))

    df = pd.DataFrame(rows, columns=["date", "close"]).sort_values("date").tail(limit).reset_index(drop=True)
    # Fake OHLC (ATR not meaningful here; we handle that)
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = np.nan
    return df


def fetch_market_df(m: Market) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if m.calc_source.startswith("stooq:"):
            sym = m.calc_source.split(":", 1)[1].strip().lower()
            return fetch_stooq_daily(sym), None
        if m.calc_source.startswith("fx:"):
            return fetch_eurusd_series(), None
        return None, "Unknown calc source"
    except Exception as e:
        return None, str(e)


# =========================================================
# PRO++ FEATURES
# =========================================================
def current_session_bias() -> Dict[str, str]:
    # UTC-based sessions (simple & practical)
    h = pd.Timestamp.utcnow().hour
    if 0 <= h < 7:
        return {"session": "ASIA", "note": "Ranges/false breaks common. Wait for London/NY confirmation."}
    if 7 <= h < 13:
        return {"session": "LONDON", "note": "Highest FX momentum window. Breakouts more valid."}
    if 13 <= h < 21:
        return {"session": "NEW YORK", "note": "Continuation/reversals. Watch US news + DXY."}
    return {"session": "LATE / ROLL", "note": "Liquidity lower. Prefer higher timeframe levels."}


def weekly_resample(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.set_index("date").sort_index()
    w = pd.DataFrame()
    w["open"] = d["open"].resample("W-FRI").first()
    w["high"] = d["high"].resample("W-FRI").max()
    w["low"] = d["low"].resample("W-FRI").min()
    w["close"] = d["close"].resample("W-FRI").last()
    w = w.dropna().reset_index()
    return w


def key_levels(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    # Swing high/low + classic pivot + fib zone from recent swing
    d = df.tail(lookback).copy()
    high = float(d["high"].max())
    low = float(d["low"].min())
    last_close = float(d["close"].iloc[-1])

    # Pivot points (classic) using last candle
    ph = float(d["high"].iloc[-1])
    pl = float(d["low"].iloc[-1])
    pc = float(d["close"].iloc[-1])
    pivot = (ph + pl + pc) / 3.0
    r1 = (2 * pivot) - pl
    s1 = (2 * pivot) - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)

    # Fib (from low -> high)
    fib_382 = low + (high - low) * 0.382
    fib_618 = low + (high - low) * 0.618

    return {
        "swing_high": high,
        "swing_low": low,
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
        "fib_382": fib_382,
        "fib_618": fib_618,
        "last_close": last_close,
    }


def compute_outlook(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else last
    chg = pct_change(last, prev)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    rsi14 = rsi(close, 14)
    hist = macd_hist(close)

    # ATR% (only if OHLC real; for EURUSD via exchangerate host -> ATR ~ 0)
    atrp = None
    try:
        a = atr(df, 14)
        a_last = float(a.iloc[-1])
        if not math.isnan(a_last) and last != 0:
            atrp = (a_last / last) * 100.0
            # EURUSD fake OHLC => atrp will be 0; treat as None
            if atrp < 0.0001:
                atrp = None
    except Exception:
        atrp = None

    # Trend & momentum signals
    trend = 0.0
    if not math.isnan(float(sma50.iloc[-1])):
        trend = 1.0 if last > float(sma50.iloc[-1]) else -1.0

    momentum = 0.0
    r = float(rsi14.iloc[-1]) if not math.isnan(float(rsi14.iloc[-1])) else None
    if r is not None:
        if r >= 60:
            momentum = 0.8
        elif r <= 40:
            momentum = -0.8
        else:
            momentum = 0.0

    macd_sig = 0.0
    h = float(hist.iloc[-1]) if not math.isnan(float(hist.iloc[-1])) else None
    if h is not None:
        macd_sig = 0.6 if h > 0 else (-0.6 if h < 0 else 0.0)

    # Higher timeframe filter proxy: SMA200
    ht = 0.0
    if len(close) >= 200 and not math.isnan(float(sma200.iloc[-1])):
        ht = 0.4 if last > float(sma200.iloc[-1]) else -0.4

    # Volatility penalty
    vol_adj = 0.0
    if atrp is not None:
        if atrp > 1.3:
            vol_adj = -0.35
        elif atrp < 0.7:
            vol_adj = +0.10

    score = trend + momentum + macd_sig + ht + vol_adj
    score = max(-2.2, min(2.2, score))

    if score >= 0.8:
        state, bias = "BULLISH", "BUY BIAS"
    elif score <= -0.8:
        state, bias = "BEARISH", "SELL BIAS"
    else:
        state, bias = "NEUTRAL", "WAIT / NEUTRAL"

    # Confidence
    base_conf = min(100.0, abs(score) / 2.2 * 100.0)  # 0..100
    # Penalize if volatility very high (when available)
    if atrp is not None and atrp > 1.3:
        base_conf = max(0.0, base_conf - 15.0)
    # Boost slightly if RSI agrees with trend
    if r is not None:
        if (trend > 0 and r > 55) or (trend < 0 and r < 45):
            base_conf = min(100.0, base_conf + 7.0)

    return {
        "last": last,
        "chg": chg,
        "sma20": float(sma20.iloc[-1]) if not math.isnan(float(sma20.iloc[-1])) else None,
        "sma50": float(sma50.iloc[-1]) if not math.isnan(float(sma50.iloc[-1])) else None,
        "sma200": float(sma200.iloc[-1]) if len(close) >= 200 and not math.isnan(float(sma200.iloc[-1])) else None,
        "rsi14": r,
        "macd_hist": h,
        "atrp": atrp,
        "score": score,
        "state": state,
        "bias": bias,
        "confidence": float(base_conf),
    }


def mtf_confluence(df_daily: pd.DataFrame) -> Dict[str, str]:
    # Daily + Weekly trend agreement
    d_close = df_daily["close"].astype(float)
    d_sma50 = d_close.rolling(50).mean()
    d_last = float(d_close.iloc[-1])
    d_trend = "UP" if not math.isnan(float(d_sma50.iloc[-1])) and d_last > float(d_sma50.iloc[-1]) else "DOWN"

    w = weekly_resample(df_daily)
    w_close = w["close"].astype(float)
    w_sma20 = w_close.rolling(20).mean()  # weekly SMA20
    if len(w_close) < 25 or math.isnan(float(w_sma20.iloc[-1])):
        w_trend = "—"
    else:
        w_trend = "UP" if float(w_close.iloc[-1]) > float(w_sma20.iloc[-1]) else "DOWN"

    if w_trend == "—":
        verdict = "MTF: Daily only"
    elif d_trend == w_trend:
        verdict = f"MTF: STRONG ({d_trend} on Daily & Weekly)"
    else:
        verdict = f"MTF: MIXED (Daily {d_trend} vs Weekly {w_trend})"

    return {"daily": d_trend, "weekly": w_trend, "verdict": verdict}


@st.cache_data(ttl=120, show_spinner=False)
def eurusd_dxy_correlation() -> Tuple[Optional[float], Optional[str]]:
    # EURUSD returns vs DXY proxy (UUP) returns - rolling corr 30D
    try:
        eur = fetch_eurusd_series(limit=260)[["date", "close"]].rename(columns={"close": "eurusd"})
        dxy = fetch_stooq_daily("uup", limit=260)[["date", "close"]].rename(columns={"close": "dxy"})
        merged = pd.merge(eur, dxy, on="date", how="inner").sort_values("date")
        if len(merged) < 60:
            return None, "Not enough overlap data"

        re = merged["eurusd"].pct_change()
        rd = merged["dxy"].pct_change()
        corr = re.rolling(30).corr(rd).iloc[-1]
        if corr is None or (isinstance(corr, float) and math.isnan(corr)):
            return None, "Correlation unavailable"
        return float(corr), None
    except Exception as e:
        return None, str(e)


# =========================================================
# TRADINGVIEW EMBEDS (CLIENT-SIDE)
# =========================================================
def tv_chart(symbol: str, height: int = 760) -> str:
    sid = symbol.replace(":", "_").replace("/", "_")
    return f"""
<div class="tvwrap" style="height:{height}px;">
  <div id="tv_{sid}" style="height:{height}px; width:100%;"></div>
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
    "toolbar_bg": "#f5f6f8",
    "enable_publishing": false,
    "withdateranges": true,
    "allow_symbol_change": false,
    "container_id": "tv_{sid}"
  }});
</script>
"""


def tv_ta_widget(symbol: str, height: int = 460) -> str:
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


# =========================================================
# SIDEBAR + AUTO REFRESH
# =========================================================
with st.sidebar:
    st.header("Settings")
    refresh = st.number_input("Auto refresh (seconds)", min_value=30, max_value=600, value=DEFAULT_REFRESH_SEC, step=30)
    st.caption("Tip: 120s is stable. Too low can cause rate limits on free feeds.")
    st.divider()
    s = current_session_bias()
    st.subheader("Session (UTC)")
    st.write(f"**{s['session']}**")
    st.caption(s["note"])
    st.divider()
    st.subheader("Markets")
    st.write("US100 • US500 • GOLD(XAUUSD) • EURUSD • DXY")
    st.caption("Charts = Capital.com TradingView\n\nOutlook = server-side indicators (stable feeds)")


# Auto refresh (client-side)
st.markdown(
    f"""
<script>
  setTimeout(function(){{ window.location.reload(); }}, {int(refresh)*1000});
</script>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================
st.title(APP_TITLE)
st.caption("MTF confluence • Key levels • Confidence % • Session bias • DXY correlation • Professional clean layout")

corr, corr_err = eurusd_dxy_correlation()
cA, cB, cC = st.columns([1.2, 1, 1])

with cA:
    s = current_session_bias()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="mid">Session (UTC): <b>{s["session"]}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta">{s["note"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mid">EURUSD ↔ DXY Correlation (30D)</div>', unsafe_allow_html=True)
    if corr_err or corr is None:
        st.markdown(f'<div class="meta">— ({corr_err or "unavailable"})</div>', unsafe_allow_html=True)
    else:
        tag = "warn" if corr > -0.20 else "neut"
        st.markdown(f'<span class="pill {tag}">{corr:+.2f}</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="meta">Normally negative. If it becomes less negative/positive, be careful with EURUSD signals.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with cC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mid">Refresh</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{int(refresh)}s</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta">Last refresh: {time.strftime("%Y-%m-%d %H:%M:%S")} UTC</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# =========================================================
# OVERVIEW ROW
# =========================================================
st.subheader("Market Outlook — Overview")
overview_cols = st.columns(len(MARKETS))

market_cache: Dict[str, Dict] = {}

for i, m in enumerate(MARKETS):
    df, err = fetch_market_df(m)
    with overview_cols[i]:
        st.markdown(f"### {m.key}")
        st.caption(m.title)

        if err or df is None or len(df) < 80:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="pill bear">DATA ERROR</span>', unsafe_allow_html=True)
            st.write(err or "Not enough data")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        o = compute_outlook(df)
        mtf = mtf_confluence(df)
        lv = key_levels(df, lookback=60)

        market_cache[m.key] = {"df": df, "outlook": o, "mtf": mtf, "levels": lv}

        state = o["state"]
        cls = "neut"
        if state == "BULLISH":
            cls = "bull"
        elif state == "BEARISH":
            cls = "bear"

        chg = o.get("chg")
        chg_txt = "—" if chg is None else f"{chg:+.2f}%"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<span class="pill {cls}">{state}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="big">{o["bias"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">Confidence: <b>{o["confidence"]:.0f}%</b> • Score: {o["score"]:+.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">Last (calc): {o["last"]:.4f} • Δ {chg_txt}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">{mtf["verdict"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# =========================================================
# DETAILS: 1 MARKET PER ROW
# =========================================================
st.subheader("Pro++ Detail View (1 per row)")

for m in MARKETS:
    st.markdown(f"## {m.title}")

    df, err = fetch_market_df(m)
    if err or df is None or len(df) < 80:
        st.error(f"Feed error for {m.key}: {err or 'unknown'}")
        st.markdown(tv_chart(m.tv_symbol, height=780), unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        continue

    o = compute_outlook(df)
    mtf = mtf_confluence(df)
    lv = key_levels(df, lookback=60)

    # TOP METRICS
    a, b, c, d, e = st.columns([1.1, 1, 1, 1, 1])
    with a:
        st.metric("Outlook", o["state"])
        st.write(f"**Bias:** {o['bias']}")
        st.caption(f"Chart: {m.tv_symbol} • {m.desc}")
    with b:
        st.metric("Confidence", f"{o['confidence']:.0f}%")
    with c:
        st.metric("RSI(14)", "—" if o["rsi14"] is None else f"{o['rsi14']:.1f}")
    with d:
        st.metric("SMA20/SMA50", "—" if (o["sma20"] is None or o["sma50"] is None) else f"{o['sma20']:.2f} / {o['sma50']:.2f}")
    with e:
        st.metric("Volatility (ATR%)", "—" if o["atrp"] is None else f"{o['atrp']:.2f}%")

    # KEY LEVELS + MTF BOX
    k1, k2 = st.columns([1.15, 1.85])
    with k1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='mid'>Key Levels (last ~60D)</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="meta">
<b>Pivot:</b> {lv["pivot"]:.2f}<br/>
<b>R1/R2:</b> {lv["r1"]:.2f} / {lv["r2"]:.2f}<br/>
<b>S1/S2:</b> {lv["s1"]:.2f} / {lv["s2"]:.2f}<br/>
<b>Swing High/Low:</b> {lv["swing_high"]:.2f} / {lv["swing_low"]:.2f}<br/>
<b>Fib Zone:</b> {lv["fib_382"]:.2f} – {lv["fib_618"]:.2f}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='mid'>MTF Confluence</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="meta">
<b>Daily trend:</b> {mtf["daily"]} (price vs SMA50)<br/>
<b>Weekly trend:</b> {mtf["weekly"]} (price vs weekly SMA20)<br/>
<b>Verdict:</b> {mtf["verdict"]}<br/><br/>
<b>Rule of thumb:</b> Only take aggressive entries when Daily & Weekly agree.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # BIG CHART
    st.markdown(tv_chart(m.tv_symbol, height=820), unsafe_allow_html=True)

    # OPTIONAL: TradingView TA widget (visual multi-interval)
    with st.expander("TradingView Technical Analysis (visual, multi-interval)"):
        st.markdown(tv_ta_widget(m.tv_symbol, height=520), unsafe_allow_html=True)

    # RULES EXPLAINER
    with st.expander("Waarom deze outlook? (PRO++ rules)"):
        st.write(
            """
**Outlook score bestaat uit:**
- Trend: price > SMA50 → bullish, anders bearish
- Momentum: RSI ≥ 60 bullish / RSI ≤ 40 bearish
- MACD histogram: >0 bullish / <0 bearish
- HT filter: price > SMA200 → extra bullish bias (anders bearish)
- Volatility adjust: extreem hoog = lagere conviction

**Confidence %**
- gebaseerd op absolute score + volatility penalty + RSI/trend agreement

**Key levels**
- Classic pivot + swing high/low + fib zone (60D) → perfecte zones om entries te plannen.
"""
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.caption(f"✅ Running stable • Auto refresh: {int(refresh)}s • UTC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
