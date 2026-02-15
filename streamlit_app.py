import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Market Outlook", page_icon="📊", layout="wide")

st.title("📊 Market Outlook — Bull/Bear Engine (Sentiment • Vol • News • DXY)")
st.caption("TradingView charts + eigen outlook score. Calendar: alleen RED/ORANGE. DXY proxy via UUP.")

# ---- Sidebar
st.sidebar.header("⚙️ Settings")
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 30, 600, 60, step=30)
tf_fast = st.sidebar.selectbox("Fast timeframe", ["1", "5", "15"], index=2)
tf_slow = st.sidebar.selectbox("Slow timeframe", ["60", "240", "D"], index=0)
lookback_bars = st.sidebar.slider("Lookback bars (trend/vol)", 60, 400, 180, step=30)
use_dxy = st.sidebar.checkbox("Include DXY movement in score", value=True)

st.sidebar.divider()
st.sidebar.subheader("📅 ForexFactory calendar")
ff_file = st.sidebar.file_uploader("Upload ForexFactory Calendar CSV (week export)", type=["csv"])
st.sidebar.caption("We nemen alleen RED/ORANGE (High/Medium impact) mee.")

# ---- Secrets
API_KEY = st.secrets.get("FINNHUB_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()

# =========================
# SYMBOLS
# =========================
# TradingView for "CFD look" (view-only)
TV = {
    "US100 (Nasdaq CFD)": "OANDA:NAS100USD",
    "US30 (Dow CFD)": "OANDA:US30USD",
    "SPX500 (S&P CFD)": "OANDA:SPX500USD",
    "XAUUSD": "OANDA:XAUUSD",
    "EURUSD": "OANDA:EURUSD",
}

# Calculation feed (Finnhub) - stable proxies where needed
CALC = {
    "US100 (Nasdaq CFD)": "QQQ",
    "US30 (Dow CFD)": "DIA",
    "SPX500 (S&P CFD)": "SPY",
    "XAUUSD": "GLD",              # spot XAU via broker/provider later; GLD is stable now
    "EURUSD": "OANDA:EUR_USD",    # if your Finnhub plan blocks this, swap to FX feed provider later
}

# DXY proxy (UUP = dollar ETF)
DXY_PROXY = "UUP"

MARKETS = list(TV.keys())

# =========================
# TRADINGVIEW WIDGETS
# =========================
def tv_symbol_info(symbol: str) -> str:
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
      {{
        "symbol": "{symbol}",
        "width": "100%",
        "locale": "en",
        "colorTheme": "light",
        "isTransparent": true
      }}
      </script>
    </div>
    """

def tv_chart(symbol: str, interval: str = "15") -> str:
    return f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{interval}",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "save_image": false,
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """

# =========================
# FOREX FACTORY CSV
# =========================
def _impact_class(x: str) -> str:
    """
    Map FF impact string to {high, medium, low, unknown}
    We treat RED=high, ORANGE=medium.
    """
    s = str(x).strip().lower()
    if any(k in s for k in ["high", "red", "3", "high impact"]):
        return "high"
    if any(k in s for k in ["medium", "orange", "2", "med impact"]):
        return "medium"
    if any(k in s for k in ["low", "yellow", "1"]):
        return "low"
    return "unknown"

def parse_ff_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    colmap = {}
    for c in df.columns:
        if "date" in c or "time" in c:
            colmap[c] = "datetime"
        elif c in ["currency", "cur"]:
            colmap[c] = "currency"
        elif "impact" in c:
            colmap[c] = "impact"
        elif "event" in c:
            colmap[c] = "event"
        elif "actual" in c:
            colmap[c] = "actual"
        elif "forecast" in c:
            colmap[c] = "forecast"
        elif "previous" in c:
            colmap[c] = "previous"

    df = df.rename(columns=colmap)

    needed = ["datetime", "currency", "impact", "event"]
    for k in needed:
        if k not in df.columns:
            return pd.DataFrame(columns=["datetime","currency","impact","impact_class","event","actual","forecast","previous"])

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for c in ["actual", "forecast", "previous"]:
        if c not in df.columns:
            df[c] = np.nan

    df["impact_class"] = df["impact"].apply(_impact_class)

    # keep only RED/ORANGE (high/medium)
    df = df[df["impact_class"].isin(["high", "medium"])].reset_index(drop=True)
    return df

def _to_float(v):
    try:
        s = str(v).strip().replace("%", "").replace(",", "")
        if s in ["", "nan", "None"]:
            return None
        return float(s)
    except Exception:
        return None

# =========================
# FINNHUB CANDLES (stock vs forex)
# =========================
def finnhub_candles(symbol: str, resolution: str, bars: int = 180) -> pd.DataFrame:
    """
    Uses:
    - /stock/candle for stocks/ETFs
    - /forex/candle for forex symbols like OANDA:EUR_USD
    """
    end = int(datetime.now(timezone.utc).timestamp())
    sec = {"1":60,"5":300,"15":900,"60":3600,"240":14400,"D":86400}[resolution]
    start = end - bars * sec

    is_forex = ":" in symbol and symbol.upper().startswith(("OANDA:", "FXCM:", "FOREX:"))

    endpoint = "https://finnhub.io/api/v1/forex/candle" if is_forex else "https://finnhub.io/api/v1/stock/candle"

    r = requests.get(
        endpoint,
        params={"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": API_KEY},
        timeout=20,
    )

    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized (API key invalid/not active)")
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden (symbol/feed not allowed on your plan)")
    r.raise_for_status()

    js = r.json()
    if js.get("s") != "ok":
        raise RuntimeError(f"No candle data ({js.get('s')})")

    df = pd.DataFrame({
        "t": js["t"],
        "o": js["o"],
        "h": js["h"],
        "l": js["l"],
        "c": js["c"],
        "v": js.get("v", [0] * len(js["t"])),
    })
    df["dt"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df

# =========================
# INDICATORS
# =========================
def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])

def realized_vol(df: pd.DataFrame) -> float:
    c = df["c"].astype(float)
    rets = np.log(c / c.shift(1)).dropna()
    if len(rets) < 10:
        return 0.0
    return float(rets.std() * math.sqrt(len(rets)) * 100)

def momentum_score(df_fast: pd.DataFrame, df_slow: pd.DataFrame) -> int:
    def slope_score(c: pd.Series) -> int:
        y = c.tail(min(50, len(c))).values
        x = np.arange(len(y))
        m = np.polyfit(x, y, 1)[0]
        return 1 if m > 0 else -1 if m < 0 else 0

    def ema_score(c: pd.Series, span: int = 20) -> int:
        e = c.ewm(span=span).mean()
        return 1 if c.iloc[-1] > e.iloc[-1] else -1

    c_fast = df_fast["c"].astype(float)
    c_slow = df_slow["c"].astype(float)

    score = 0
    score += slope_score(c_fast)
    score += ema_score(c_fast, 20)
    score += slope_score(c_slow)
    score += ema_score(c_slow, 20)
    return int(np.clip(score, -4, 4))

def vol_score(df_fast: pd.DataFrame) -> int:
    rv = realized_vol(df_fast)
    a = atr(df_fast, 14)
    p = float(df_fast["c"].iloc[-1])
    atr_pct = (a / p) * 100 if p else 0.0

    score = 0
    # vol spikes -> risk-off bias
    if rv > 2.0 or atr_pct > 0.6:
        score -= 1
    if rv > 3.5 or atr_pct > 1.0:
        score -= 1
    return score

# =========================
# NEWS SCORING (FF) - only RED/ORANGE already filtered
# =========================
def news_impact_score(ff: pd.DataFrame) -> int:
    """
    Score based on:
    - number of upcoming RED/ORANGE events in next 12h (risk -> bearish)
    - recent surprises (Actual vs Forecast) in last 12h (bigger surprise -> more vol -> bearish)
    """
    if ff is None or ff.empty:
        return 0

    now = datetime.now(timezone.utc)
    recent = ff[(ff["datetime"] >= now - timedelta(hours=12)) & (ff["datetime"] <= now)]
    upcoming = ff[(ff["datetime"] > now) & (ff["datetime"] <= now + timedelta(hours=12))]

    score = 0

    # Upcoming load
    n_up = len(upcoming)
    if n_up >= 2:
        score -= 1
    if n_up >= 5:
        score -= 1

    # Recent surprises
    if not recent.empty and ("actual" in recent.columns) and ("forecast" in recent.columns):
        surprises = []
        for _, r in recent.iterrows():
            a = _to_float(r.get("actual"))
            f = _to_float(r.get("forecast"))
            if a is None or f is None:
                continue
            surprises.append(abs(a - f))

        if surprises:
            mx = max(surprises)
            if mx > 0.5:
                score -= 1
            if mx > 1.0:
                score -= 1

    return score

# =========================
# DXY MOVEMENT SCORE (via UUP)
# =========================
def dxy_movement_score(df_dxy_fast: pd.DataFrame) -> int:
    """
    Score based on short-term DXY proxy (UUP) trend.
    If USD strengthening -> tends to pressure EURUSD (bear) and XAUUSD (bear).
    For indices: mild risk-off when USD spikes.
    """
    c = df_dxy_fast["c"].astype(float)
    if len(c) < 20:
        return 0

    # simple pct change over last 20 bars
    pct = (c.iloc[-1] / c.iloc[-20] - 1) * 100
    if pct > 0.25:
        return 1
    if pct < -0.25:
        return -1
    return 0

# =========================
# LABELING
# =========================
def label_from_total(total: int) -> str:
    if total >= 3:
        return "BULLISH"
    if total <= -3:
        return "BEARISH"
    return "NEUTRAL"

def badge(label: str) -> str:
    return {"BULLISH":"🟢 BULLISH","BEARISH":"🔴 BEARISH"}.get(label, "⚪ NEUTRAL")

# =========================
# LOAD CALENDAR (RED/ORANGE only)
# =========================
ff_df = parse_ff_csv(ff_file) if ff_file else pd.DataFrame()

# Precompute calendar score (macro risk)
ff_score = news_impact_score(ff_df)

# Precompute DXY proxy score (macro USD move)
dxy_score = 0
if use_dxy:
    try:
        dxy_fast = finnhub_candles(DXY_PROXY, tf_fast, bars=lookback_bars)
        dxy_score = dxy_movement_score(dxy_fast)  # +1 = USD up, -1 = USD down
    except Exception:
        dxy_score = 0

# =========================
# TOP CARDS
# =========================
cols = st.columns(5)
results = {}

for i, m in enumerate(MARKETS):
    with cols[i]:
        st.subheader(m)

        sym = CALC[m]
        try:
            df_fast = finnhub_candles(sym, tf_fast, bars=lookback_bars)
            df_slow = finnhub_candles(sym, tf_slow, bars=lookback_bars)

            mom = momentum_score(df_fast, df_slow)   # -4..+4
            vol = vol_score(df_fast)                 # 0..-2
            news = ff_score                           # 0..-2 (macro calendar)

            # DXY effect depends on market:
            # USD up -> bearish EURUSD, bearish Gold; mild bearish indices.
            dxy_component = 0
            if use_dxy:
                if m == "EURUSD":
                    dxy_component = -dxy_score   # USD up => EURUSD down (bear)
                elif m == "XAUUSD":
                    dxy_component = -dxy_score   # USD up => gold pressured (bear)
                else:
                    dxy_component = -1 if dxy_score == 1 else 0  # only penalize on strong USD

            total = int(mom + vol + news + dxy_component)
            lbl = label_from_total(total)

            price = float(df_fast["c"].iloc[-1])
            if m == "EURUSD":
                st.metric("Price (calc feed)", f"{price:.5f}")
            else:
                st.metric("Price (calc feed)", f"{price:,.2f}")

            st.write(badge(lbl), f"Score: **{total}**")
            st.caption(
                f"Sentiment: {mom:+d} • Vol: {vol:+d} • News: {news:+d} • DXY: {dxy_component:+d}"
                if use_dxy else
                f"Sentiment: {mom:+d} • Vol: {vol:+d} • News: {news:+d}"
            )

            results[m] = {"label": lbl, "score": total, "symbol": sym}

        except Exception as e:
            st.write("⚪ NEUTRAL")
            st.caption(f"⚠️ Calc feed error: {str(e)[:120]}")

st.divider()

# =========================
# DETAILS
# =========================
left, right = st.columns([2, 1])

with right:
    selected = st.selectbox("Select market", MARKETS, index=0)
    st.write("TradingView symbol:", f"`{TV[selected]}`")
    st.write("Calc symbol:", f"`{CALC[selected]}`")

    if use_dxy:
        st.write("DXY proxy (UUP) move score:", dxy_score, "(+1 USD up, -1 USD down, 0 flat)")

    if selected in results:
        st.write("Bias:", badge(results[selected]["label"]))
        st.write("Score:", results[selected]["score"])

with left:
    components.html(tv_chart(TV[selected], interval=tf_fast), height=560)

st.divider()

# =========================
# CALENDAR VIEW (RED/ORANGE ONLY)
# =========================
st.subheader("📅 ForexFactory — RED/ORANGE only (from CSV upload)")
if ff_df.empty:
    st.info("Upload ForexFactory week-CSV via de sidebar. We filteren automatisch naar RED/ORANGE.")
else:
    now = datetime.now(timezone.utc)
    recent = ff_df[(ff_df["datetime"] >= now - timedelta(hours=12)) & (ff_df["datetime"] <= now)].tail(50)
    upcoming = ff_df[(ff_df["datetime"] > now) & (ff_df["datetime"] <= now + timedelta(days=2))].head(80)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Recent (last 12h)")
        st.dataframe(
            recent[["datetime","currency","impact","impact_class","event","actual","forecast","previous"]],
            use_container_width=True
        )
    with c2:
        st.markdown("### Upcoming (next 48h)")
        st.dataframe(
            upcoming[["datetime","currency","impact","impact_class","event","forecast","previous"]],
            use_container_width=True
        )

st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} • Auto refresh every {refresh_seconds}s")
time.sleep(refresh_seconds)
st.rerun()
