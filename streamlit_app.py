import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# =========================================================
# UnknownFX Dashboard
# =========================================================

st.set_page_config(page_title="UnknownFX Dashboard", layout="wide")
st.title("📈 UnknownFX Dashboard")
st.caption("Charts via TradingView (Capital.com) • Sentiment: TA + volatility + regime + DXY correlation")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Settings")

refresh_minutes = st.sidebar.slider("Refresh interval (minutes)", min_value=1, max_value=30, value=10, step=1)
tf_choice = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=0)

# Auto refresh
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# Map timeframe to yfinance interval + resample
TF_MAP = {
    "15m": {"yf_interval": "15m", "tv_interval": "15", "resample": None, "period": "10d"},
    "1h":  {"yf_interval": "60m", "tv_interval": "60", "resample": None, "period": "60d"},
    "4h":  {"yf_interval": "60m", "tv_interval": "240", "resample": "4H", "period": "60d"},
    "1d":  {"yf_interval": "1d",  "tv_interval": "D",  "resample": None, "period": "2y"},
}

tf_cfg = TF_MAP[tf_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Market symbols (edit if needed)")

# Default mapping based on what you shared (Capital.com labels)
DEFAULT_MARKETS = {
    "US100":  {"tv": "CAPITALCOM:US100",  "yf": "^NDX"},
    "US500":  {"tv": "CAPITALCOM:US500",  "yf": "^GSPC"},
    "US30":   {"tv": "CAPITALCOM:US30",   "yf": "^DJI"},
    "GOLD":   {"tv": "CAPITALCOM:GOLD",   "yf": "GC=F"},        # Gold futures proxy
    "EURUSD": {"tv": "CAPITALCOM:EURUSD", "yf": "EURUSD=X"},
    "DXY":    {"tv": "CAPITALCOM:DXY",    "yf": "DX-Y.NYB"},
}

# Editable tickers in sidebar (100% instelbaar)
markets = {}
for k, v in DEFAULT_MARKETS.items():
    tv_sym = st.sidebar.text_input(f"{k} TradingView symbol", value=v["tv"])
    yf_sym = st.sidebar.text_input(f"{k} Data ticker (yfinance)", value=v["yf"])
    markets[k] = {"tv": tv_sym.strip(), "yf": yf_sym.strip()}

st.sidebar.markdown("---")
show_debug = st.sidebar.checkbox("Show debug panels", value=False)

# =========================================================
# Indicators
# =========================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Classic ADX calculation
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df).values
    atr_vals = pd.Series(tr).rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr_vals)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr_vals)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum" if "Volume" in df.columns else "sum"
    }).dropna()
    return ohlc

# =========================================================
# Data fetch
# =========================================================

@st.cache_data(ttl=60 * 10)  # cache 10 minutes
def fetch_market_data(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    # yfinance sometimes returns multiindex columns; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

# =========================================================
# Sentiment scoring
# =========================================================

def compute_signals(df: pd.DataFrame) -> dict:
    close = df["Close"]
    df = df.copy()

    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI14"] = rsi(close, 14)
    df["ATR14"] = atr(df, 14)
    df["ADX14"] = adx(df, 14)
    df["MACDH"] = macd_hist(close)

    latest = df.iloc[-1]
    price = float(latest["Close"])
    ema20v = float(latest["EMA20"])
    ema50v = float(latest["EMA50"])
    rsiv = float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else np.nan
    atrv = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else np.nan
    adxv = float(latest["ADX14"]) if pd.notna(latest["ADX14"]) else np.nan
    macdh = float(latest["MACDH"]) if pd.notna(latest["MACDH"]) else np.nan

    # Slope (trend direction) via EMA20 slope over last N bars
    n = min(10, len(df) - 1)
    ema20_slope = float(df["EMA20"].iloc[-1] - df["EMA20"].iloc[-1 - n]) if n > 0 else 0.0

    # Score components (-5..+5)
    score = 0
    reasons = []

    # Trend stack
    if price > ema20v and ema20v > ema50v:
        score += 2
        reasons.append("+2 Price>EMA20>EMA50 (uptrend stack)")
    elif price < ema20v and ema20v < ema50v:
        score -= 2
        reasons.append("-2 Price<EMA20<EMA50 (downtrend stack)")
    else:
        reasons.append("0 Mixed EMA stack")

    # RSI zone
    if not np.isnan(rsiv):
        if rsiv >= 60:
            score += 1
            reasons.append("+1 RSI>=60 (bullish momentum)")
        elif rsiv <= 40:
            score -= 1
            reasons.append("-1 RSI<=40 (bearish momentum)")
        else:
            reasons.append("0 RSI neutral zone")

    # MACD histogram
    if not np.isnan(macdh):
        if macdh > 0:
            score += 1
            reasons.append("+1 MACD hist > 0 (bullish impulse)")
        elif macdh < 0:
            score -= 1
            reasons.append("-1 MACD hist < 0 (bearish impulse)")
        else:
            reasons.append("0 MACD hist flat")

    # EMA slope
    if ema20_slope > 0:
        score += 1
        reasons.append("+1 EMA20 slope up")
    elif ema20_slope < 0:
        score -= 1
        reasons.append("-1 EMA20 slope down")

    # Volatility filter (ATR%): too high = reduce confidence
    atr_pct = None
    if not np.isnan(atrv) and price != 0:
        atr_pct = (atrv / price) * 100
        if atr_pct >= 1.2:
            score -= 1
            reasons.append("-1 ATR% high → reduce confidence")
        elif atr_pct <= 0.4:
            score += 0  # keep neutral; low vol isn't always good
            reasons.append("0 ATR% low/normal")

    score = int(np.clip(score, -5, 5))

    # Sentiment label + trade bias
    if score >= 2:
        sentiment = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -2:
        sentiment = "BEARISH"
        bias = "SELL BIAS"
    else:
        sentiment = "NEUTRAL"
        bias = "WAIT / NO EDGE"

    # Regime: ADX trend vs range (fallback to slope)
    if not np.isnan(adxv):
        regime = "TRENDING" if adxv >= 25 else "RANGING"
    else:
        regime = "TRENDING" if abs(ema20_slope) > 0 else "RANGING"

    # Volatility label
    vol_label = None
    if atr_pct is not None:
        if atr_pct >= 1.2:
            vol_label = "HIGH"
        elif atr_pct <= 0.4:
            vol_label = "LOW"
        else:
            vol_label = "NORMAL"

    return {
        "price": price,
        "ema20": ema20v,
        "ema50": ema50v,
        "rsi": rsiv,
        "adx": adxv,
        "macdh": macdh,
        "atr_pct": atr_pct,
        "vol_label": vol_label,
        "score": score,
        "sentiment": sentiment,
        "bias": bias,
        "regime": regime,
        "reasons": reasons,
    }

# =========================================================
# Sessions
# =========================================================

def session_status(now_utc: datetime) -> dict:
    # Simple UTC-based sessions:
    # Asia (Tokyo): 00-09 UTC
    # London: 07-16 UTC
    # New York: 13-22 UTC
    h = now_utc.hour
    asia = (0 <= h < 9)
    london = (7 <= h < 16)
    ny = (13 <= h < 22)
    overlap1 = london and ny
    overlap2 = asia and london  # small overlap
    return {"Asia": asia, "London": london, "New York": ny, "London/NY overlap": overlap1, "Asia/London overlap": overlap2}

# =========================================================
# Header: session + refresh
# =========================================================

now = datetime.now(timezone.utc)
sess = session_status(now)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2])
with c1:
    st.metric("⏱ Refresh", f"{refresh_minutes} min", delta=f"TF: {tf_choice}")
with c2:
    st.metric("🕒 UTC Time", now.strftime("%H:%M"))
with c3:
    open_sessions = [k for k, v in sess.items() if v and "overlap" not in k]
    st.metric("🌍 Sessions open", ", ".join(open_sessions) if open_sessions else "None")
with c4:
    overlaps = [k for k, v in sess.items() if v and "overlap" in k]
    st.write("**Overlaps:**", ", ".join(overlaps) if overlaps else "—")

st.markdown("---")

# =========================================================
# Main grid: 6 markets
# =========================================================

def tv_embed(symbol: str, interval: str) -> None:
    # interval: "15", "60", "240", "D"
    if interval == "D":
        tv_interval = "D"
    else:
        tv_interval = interval

    html = f"""
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol={symbol}&interval={tv_interval}&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light"
      style="width:100%;height:360px;"
      frameborder="0"
      allowtransparency="true"
      scrolling="no">
    </iframe>
    """
    components.html(html, height=380)

def score_bar(score: int):
    # map -5..+5 to 0..100
    pct = int(((score + 5) / 10) * 100)
    st.progress(pct)
    st.caption(f"Score: **{score}** (range -5..+5)")

# Build cards
cols = st.columns(3)
market_results = {}

for idx, (name, cfg) in enumerate(markets.items()):
    col = cols[idx % 3]
    with col:
        st.subheader(name)

        df = fetch_market_data(cfg["yf"], period=tf_cfg["period"], interval=tf_cfg["yf_interval"])

        if df.empty or len(df) < 60:
            st.error("No data / too little history from data feed.")
            if show_debug:
                st.info(f"yfinance symbol: {cfg['yf']}")
            tv_embed(cfg["tv"], tf_cfg["tv_interval"])
            continue

        # If 4h requested, resample
        if tf_cfg["resample"]:
            try:
                df = df.copy()
                # Ensure datetime index is tz-aware-ish; yfinance index is typically tz-naive
                df.index = pd.to_datetime(df.index)
                df = resample_ohlc(df, tf_cfg["resample"])
            except Exception as e:
                st.warning("Resample failed; using base timeframe.")
                if show_debug:
                    st.write(e)

        sig = compute_signals(df)
        market_results[name] = {"df": df, "sig": sig}

        # Sentiment badge
        if sig["sentiment"] == "BULLISH":
            st.success("🟢 BULLISH")
        elif sig["sentiment"] == "BEARISH":
            st.error("🔴 BEARISH")
        else:
            st.warning("⚪ NEUTRAL")

        st.write(f"**Bias:** {sig['bias']}")
        score_bar(sig["score"])

        # Quick stats
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Last", f"{sig['price']:.5f}" if sig["price"] < 10 else f"{sig['price']:.2f}")
        with cB:
            st.metric("Regime", sig["regime"])
        with cC:
            if sig["atr_pct"] is not None:
                st.metric("Vol (ATR%)", f"{sig['atr_pct']:.2f}% ({sig['vol_label']})")
            else:
                st.metric("Vol (ATR%)", "—")

        # Breakdown (optional)
        if show_debug:
            with st.expander("Why this score?"):
                for r in sig["reasons"]:
                    st.write(r)
                st.write("RSI:", sig["rsi"])
                st.write("EMA20:", sig["ema20"])
                st.write("EMA50:", sig["ema50"])
                st.write("ADX:", sig["adx"])
                st.write("MACD hist:", sig["macdh"])
                st.write("TV symbol:", cfg["tv"])
                st.write("Data ticker:", cfg["yf"])

        # TradingView chart
        tv_embed(cfg["tv"], tf_cfg["tv_interval"])

st.markdown("---")

# =========================================================
# DXY movement + Correlation section
# =========================================================

st.header("💵 DXY Movement & Correlations")

if "DXY" in market_results and market_results["DXY"]["df"] is not None:
    dxy_df = market_results["DXY"]["df"].copy()
    dxy_df["ret"] = dxy_df["Close"].pct_change()

    # Show DXY direction (last N bars)
    lookback = 12 if tf_choice in ["15m", "1h"] else 20
    lookback = min(lookback, len(dxy_df) - 1)
    dxy_move = (dxy_df["Close"].iloc[-1] / dxy_df["Close"].iloc[-1 - lookback] - 1) * 100 if lookback > 0 else 0

    st.metric("DXY change (recent)", f"{dxy_move:.2f}%")

    # Correlations with indices
    corr_markets = ["US100", "US500", "US30"]
    rows = []
    for m in corr_markets:
        if m in market_results:
            dfm = market_results[m]["df"].copy()
            dfm["ret"] = dfm["Close"].pct_change()

            joined = pd.concat([dxy_df["ret"], dfm["ret"]], axis=1).dropna()
            joined.columns = ["dxy_ret", "m_ret"]

            if len(joined) >= 30:
                rolling = joined["dxy_ret"].rolling(30).corr(joined["m_ret"])
                corr_now = float(rolling.iloc[-1]) if pd.notna(rolling.iloc[-1]) else np.nan
                rows.append({"Market": m, "Rolling Corr (30 bars)": corr_now})
            else:
                rows.append({"Market": m, "Rolling Corr (30 bars)": np.nan})

    corr_table = pd.DataFrame(rows)
    st.dataframe(corr_table, use_container_width=True)

    st.caption("Tip: vaak is DXY negatief gecorreleerd met indices. Als DXY hard stijgt, kan risk-off toenemen.")

else:
    st.warning("DXY data not available. Check your yfinance ticker for DXY (default: DX-Y.NYB).")

st.markdown("---")

# =========================================================
# “What fits here?” section (practical extras)
# =========================================================

st.header("✅ What else fits well for trading decisions?")

st.markdown(
    """
**Aanraders (super nuttig in praktijk):**
- **Multi-timeframe confluence**: laat 15m + 1h + 4h tegelijk een score geven → “aligned” of “mixed”.
- **Key levels / pivots**: yesterday high/low + weekly open + session highs.
- **Risk mode**: als volatility HIGH is, verlaag score confidence of zet “WAIT”.
- **Alerts**: score crossing (bijv. van -1 naar -2 = SELL bias trigger).
"""
)

st.caption("Let op: dit is een indicatie-dashboard, geen financieel advies.")

