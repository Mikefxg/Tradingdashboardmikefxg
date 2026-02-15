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

TITLE = "📊 Market Outlook (CFD-style) — Bull/Bear Engine"
st.title(TITLE)
st.caption("TradingView look + eigen bias-engine (sentiment + vol + news events)")

# ---- Sidebar
st.sidebar.header("⚙️ Settings")
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 30, 600, 60, step=30)
tf_fast = st.sidebar.selectbox("Fast timeframe", ["1", "5", "15"], index=2)
tf_slow = st.sidebar.selectbox("Slow timeframe", ["60", "240", "D"], index=0)
lookback_bars = st.sidebar.slider("Lookback bars (for trend/vol)", 30, 300, 120, step=30)

st.sidebar.divider()
st.sidebar.subheader("Datafeed (voor berekeningen)")
st.sidebar.caption(
    "TradingView widgets zijn view-only. Voor berekeningen gebruiken we Finnhub (gratis key) "
    "met proxies waar nodig."
)

API_KEY = st.secrets.get("FINNHUB_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()

# =========================
# SYMBOLS
# =========================
# TradingView symbols for “CFD look”
TV = {
    "US100 (Nasdaq CFD)": "OANDA:NAS100USD",
    "US30 (Dow CFD)": "OANDA:US30USD",
    "SPX500 (S&P CFD)": "OANDA:SPX500USD",
    "XAUUSD": "OANDA:XAUUSD",
    "EURUSD": "OANDA:EURUSD",
}

# Finnhub symbols for numeric calculations (stable proxies)
# Indices -> ETF proxies; FX ok; Gold spot sometimes restricted so use GLD fallback.
CALC = {
    "US100 (Nasdaq CFD)": "QQQ",
    "US30 (Dow CFD)": "DIA",
    "SPX500 (S&P CFD)": "SPY",
    "XAUUSD": "GLD",        # echte XAU spot via broker/datafeed later; GLD is stabiel nu
    "EURUSD": "OANDA:EUR_USD",
}

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
# FINNHUB (numeric feed)
# =========================
def finnhub_candles(symbol: str, resolution: str, bars: int = 120) -> pd.DataFrame:
    """
    Finnhub candles:
    resolution: '1','5','15','60','240','D'
    """
    end = int(datetime.now(timezone.utc).timestamp())
    # rough seconds per bar
    sec = {"1":60,"5":300,"15":900,"60":3600,"240":14400,"D":86400}[resolution]
    start = end - bars * sec

    r = requests.get(
        "https://finnhub.io/api/v1/stock/candle",
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
        "v": js["v"],
    })
    df["dt"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df

def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])

def realized_vol(df: pd.DataFrame) -> float:
    # simple realized vol % over window using log returns
    c = df["c"].astype(float)
    rets = np.log(c / c.shift(1)).dropna()
    if len(rets) < 10:
        return 0.0
    return float(rets.std() * math.sqrt(len(rets)) * 100)

def momentum_score(df_fast: pd.DataFrame, df_slow: pd.DataFrame) -> int:
    # trend via slope + position vs EMA
    def slope_score(c: pd.Series) -> int:
        y = c.tail(min(40, len(c))).values
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
    # higher vol can mean risk-off; we score “bearish” when vol spikes
    rv = realized_vol(df_fast)
    a = atr(df_fast, 14)
    # normalize ATR to price
    p = float(df_fast["c"].iloc[-1])
    atr_pct = (a / p) * 100 if p else 0.0

    # thresholds tuned for “feel”, not academically perfect
    score = 0
    if rv > 2.0 or atr_pct > 0.6:
        score -= 1
    if rv > 3.5 or atr_pct > 1.0:
        score -= 1
    return score

# =========================
# FOREX FACTORY EVENTS (CSV upload)
# =========================
st.sidebar.divider()
st.sidebar.subheader("📅 ForexFactory calendar")
ff_file = st.sidebar.file_uploader("Upload ForexFactory Calendar CSV (week export)", type=["csv"])

def parse_ff_csv(file) -> pd.DataFrame:
    """
    Verwacht: een CSV export met kolommen zoals:
    date/time, currency, impact, event, actual, forecast, previous
    (Format kan verschillen; we mappen defensief.)
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    # best-effort mapping
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
            # don't crash, just return empty
            return pd.DataFrame(columns=["datetime","currency","impact","event","actual","forecast","previous"])

    # parse datetime loosely
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for c in ["actual","forecast","previous"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def news_impact_score(ff: pd.DataFrame, market: str) -> int:
    """
    Score op basis van:
    - upcoming high-impact (risico -> bearish bias)
    - recent outcomes (Actual vs Forecast) (risk-on/off simpel)
    """
    if ff is None or ff.empty:
        return 0

    now = datetime.now(timezone.utc)
    recent = ff[(ff["datetime"] >= now - timedelta(hours=12)) & (ff["datetime"] <= now)]
    upcoming = ff[(ff["datetime"] > now) & (ff["datetime"] <= now + timedelta(hours=12))]

    # simpele impact filter: high/medium keywords
    def is_high(x):
        s = str(x).lower()
        return ("high" in s) or ("red" in s) or ("3" == s.strip())

    high_upcoming = upcoming[upcoming["impact"].apply(is_high)]
    score = 0

    # upcoming high impact -> risk (bearish)
    if len(high_upcoming) >= 2:
        score -= 1
    if len(high_upcoming) >= 4:
        score -= 1

    # outcomes: if actual better than forecast for USD growth/inflation events etc is complex
    # MVP approach: any big surprise increases vol -> bearish
    def surprise_row(r):
        try:
            a = float(str(r["actual"]).replace("%","").replace(",",""))
            f = float(str(r["forecast"]).replace("%","").replace(",",""))
            return abs(a - f)
        except Exception:
            return 0.0

    if not recent.empty:
        surprises = recent.apply(surprise_row, axis=1)
        if surprises.max() > 0:
            # big surprises -> more vol -> slightly bearish
            if surprises.max() > np.nanmedian(surprises) * 3 if len(surprises) > 3 else surprises.max() > 0.5:
                score -= 1

    return score

# =========================
# ENGINE
# =========================
def label_from_total(total: int) -> str:
    if total >= 3:
        return "BULLISH"
    if total <= -3:
        return "BEARISH"
    return "NEUTRAL"

def badge(label: str) -> str:
    return {"BULLISH":"🟢 BULLISH","BEARISH":"🔴 BEARISH"}.get(label, "⚪ NEUTRAL")

# Load FF
ff_df = parse_ff_csv(ff_file) if ff_file else pd.DataFrame()

# =========================
# TOP CARDS
# =========================
cols = st.columns(5)
results = {}

for i, m in enumerate(MARKETS):
    with cols[i]:
        st.subheader(m)

        # Numeric feed
        sym = CALC[m]
        try:
            df_fast = finnhub_candles(sym, tf_fast, bars=lookback_bars)
            df_slow = finnhub_candles(sym, tf_slow, bars=lookback_bars)
            mom = momentum_score(df_fast, df_slow)      # -4..+4
            vol = vol_score(df_fast)                    # 0..-2
            news = news_impact_score(ff_df, m)          # 0..-2 (MVP)

            total = int(mom + vol + news)
            lbl = label_from_total(total)

            price = float(df_fast["c"].iloc[-1])
            st.metric("Price (calc feed)", f"{price:,.2f}" if "EUR" not in m else f"{price:.5f}")
            st.write(badge(lbl), f"Score: **{total}**")
            st.caption(f"Sentiment: {mom:+d} • Vol: {vol:+d} • News: {news:+d}")

            results[m] = {"label": lbl, "score": total, "symbol": sym}

        except Exception as e:
            st.write("⚪ NEUTRAL")
            st.caption(f"⚠️ Calc feed error: {str(e)[:120]}")

st.divider()

# =========================
# DETAILS SECTION
# =========================
left, right = st.columns([2, 1])

with right:
    selected = st.selectbox("Select market", MARKETS, index=0)
    st.write("TradingView symbol:", f"`{TV[selected]}`")
    st.write("Calc symbol:", f"`{CALC[selected]}`")
    if selected in results:
        st.write("Bias:", badge(results[selected]["label"]))
        st.write("Score:", results[selected]["score"])

with left:
    components.html(tv_chart(TV[selected], interval=tf_fast), height=560)

st.divider()

# =========================
# FOREX FACTORY EVENTS VIEW
# =========================
st.subheader("📅 ForexFactory — Recent & Upcoming (from CSV upload)")
if ff_df.empty:
    st.info("Upload de ForexFactory week-CSV via de sidebar om events/outcomes mee te nemen in de score.")
else:
    now = datetime.now(timezone.utc)
    recent = ff_df[(ff_df["datetime"] >= now - timedelta(hours=12)) & (ff_df["datetime"] <= now)].tail(30)
    upcoming = ff_df[(ff_df["datetime"] > now) & (ff_df["datetime"] <= now + timedelta(days=2))].head(50)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Recent (last 12h)")
        st.dataframe(recent[["datetime","currency","impact","event","actual","forecast","previous"]], use_container_width=True)
    with c2:
        st.markdown("### Upcoming (next 48h)")
        st.dataframe(upcoming[["datetime","currency","impact","event","forecast","previous"]], use_container_width=True)

st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} • Auto refresh every {refresh_seconds}s")
time.sleep(refresh_seconds)
st.rerun()
