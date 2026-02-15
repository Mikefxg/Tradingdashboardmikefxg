import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone, timedelta

# =========================================================
# UnknownFX Dashboard (MTF Confluence + Key Levels)
# =========================================================

st.set_page_config(page_title="UnknownFX Dashboard", layout="wide")
st.title("📈 UnknownFX Dashboard")
st.caption(
    "Charts via TradingView (Capital.com) • Sentiment: TA + volatility + regime + DXY correlation • "
    "MTF Confluence + Key Levels (YH/YL/WO + London/NY ranges)"
)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Settings")

refresh_minutes = st.sidebar.slider("Refresh interval (minutes)", 1, 30, 10, 1)
main_tf_choice = st.sidebar.selectbox("Main chart timeframe", ["15m", "1h", "4h", "1d"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Multi-timeframe confluence")
use_15m = st.sidebar.checkbox("Use 15m", True)
use_1h  = st.sidebar.checkbox("Use 1h", True)
use_4h  = st.sidebar.checkbox("Use 4h", True)
use_1d  = st.sidebar.checkbox("Use 1d", False)

vote_threshold = st.sidebar.slider("Votes needed for ALIGNED (out of selected TFs)", 2, 4, 2, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Key Levels sensitivity")
near_atr_mult = st.sidebar.slider("Near-level threshold (ATR x)", 0.05, 0.50, 0.20, 0.05)
show_debug = st.sidebar.checkbox("Show debug panels", value=False)

# Auto refresh
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# ---------------------------------------------------------
# Timeframe configs (yfinance interval + optional resample)
# ---------------------------------------------------------
TF_CFG = {
    "15m": {"yf_interval": "15m", "tv_interval": "15",  "resample": None, "period": "10d"},
    "1h":  {"yf_interval": "60m", "tv_interval": "60",  "resample": None, "period": "60d"},
    "4h":  {"yf_interval": "60m", "tv_interval": "240", "resample": "4H", "period": "60d"},
    "1d":  {"yf_interval": "1d",  "tv_interval": "D",   "resample": None, "period": "2y"},
}

# -------------------------
# Editable market tickers
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Market symbols (edit if needed)")

DEFAULT_MARKETS = {
    "US100":  {"tv": "CAPITALCOM:US100",  "yf": "^NDX"},
    "US500":  {"tv": "CAPITALCOM:US500",  "yf": "^GSPC"},
    "US30":   {"tv": "CAPITALCOM:US30",   "yf": "^DJI"},
    "GOLD":   {"tv": "CAPITALCOM:GOLD",   "yf": "GC=F"},
    "EURUSD": {"tv": "CAPITALCOM:EURUSD", "yf": "EURUSD=X"},
    "DXY":    {"tv": "CAPITALCOM:DXY",    "yf": "DX-Y.NYB"},
}

markets = {}
for k, v in DEFAULT_MARKETS.items():
    tv_sym = st.sidebar.text_input(f"{k} TradingView symbol", value=v["tv"])
    yf_sym = st.sidebar.text_input(f"{k} Data ticker (yfinance)", value=v["yf"])
    markets[k] = {"tv": tv_sym.strip(), "yf": yf_sym.strip()}

# =========================================================
# Helpers / Indicators
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
    return true_range(df).rolling(period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["High"], df["Low"]
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    # Ensure tz-aware in UTC (yfinance may be naive)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

# =========================================================
# Sentiment scoring (per TF)
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

    n = min(10, len(df) - 1)
    ema20_slope = float(df["EMA20"].iloc[-1] - df["EMA20"].iloc[-1 - n]) if n > 0 else 0.0

    score = 0
    reasons = []

    # Trend stack
    if price > ema20v and ema20v > ema50v:
        score += 2; reasons.append("+2 Price>EMA20>EMA50")
    elif price < ema20v and ema20v < ema50v:
        score -= 2; reasons.append("-2 Price<EMA20<EMA50")
    else:
        reasons.append("0 Mixed EMA stack")

    # RSI
    if not np.isnan(rsiv):
        if rsiv >= 60:
            score += 1; reasons.append("+1 RSI>=60")
        elif rsiv <= 40:
            score -= 1; reasons.append("-1 RSI<=40")
        else:
            reasons.append("0 RSI neutral")

    # MACD hist
    if not np.isnan(macdh):
        if macdh > 0:
            score += 1; reasons.append("+1 MACD hist > 0")
        elif macdh < 0:
            score -= 1; reasons.append("-1 MACD hist < 0")
        else:
            reasons.append("0 MACD hist flat")

    # EMA slope
    if ema20_slope > 0:
        score += 1; reasons.append("+1 EMA20 slope up")
    elif ema20_slope < 0:
        score -= 1; reasons.append("-1 EMA20 slope down")

    # Volatility filter (ATR%)
    atr_pct = None
    if not np.isnan(atrv) and price != 0:
        atr_pct = (atrv / price) * 100
        if atr_pct >= 1.2:
            score -= 1; reasons.append("-1 ATR% high (reduce confidence)")
        else:
            reasons.append("0 ATR% ok")

    score = int(np.clip(score, -5, 5))

    if score >= 2:
        sentiment = "BULLISH"; bias = "BUY"
    elif score <= -2:
        sentiment = "BEARISH"; bias = "SELL"
    else:
        sentiment = "NEUTRAL"; bias = "WAIT"

    if not np.isnan(adxv):
        regime = "TRENDING" if adxv >= 25 else "RANGING"
    else:
        regime = "TRENDING" if abs(ema20_slope) > 0 else "RANGING"

    vol_label = None
    if atr_pct is not None:
        if atr_pct >= 1.2: vol_label = "HIGH"
        elif atr_pct <= 0.4: vol_label = "LOW"
        else: vol_label = "NORMAL"

    return {
        "price": price,
        "score": score,
        "sentiment": sentiment,
        "bias": bias,
        "regime": regime,
        "atr": atrv,
        "atr_pct": atr_pct,
        "vol_label": vol_label,
        "rsi": rsiv,
        "adx": adxv,
        "reasons": reasons,
    }

# =========================================================
# MTF confluence
# =========================================================

def sentiment_vote(score: int) -> int:
    if score >= 2: return 1
    if score <= -2: return -1
    return 0

def confluence_verdict(tf_scores: dict, threshold: int) -> dict:
    votes = {tf: sentiment_vote(s) for tf, s in tf_scores.items()}
    bull = sum(1 for v in votes.values() if v == 1)
    bear = sum(1 for v in votes.values() if v == -1)
    total = len(votes)

    if bull >= threshold:
        return {"verdict": "ALIGNED BUY", "label": "🟢 ALIGNED BUY", "bull": bull, "bear": bear, "total": total}
    if bear >= threshold:
        return {"verdict": "ALIGNED SELL", "label": "🔴 ALIGNED SELL", "bull": bull, "bear": bear, "total": total}
    return {"verdict": "MIXED / WAIT", "label": "⚪ MIXED / WAIT", "bull": bull, "bear": bear, "total": total}

# =========================================================
# Key Levels (YH/YL/WO + London/NY ranges)
# =========================================================

def _to_day(df: pd.DataFrame) -> pd.DataFrame:
    # Daily OHLC in UTC
    d = df.resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
    return d

def compute_key_levels(df_intraday: pd.DataFrame) -> dict:
    """
    df_intraday must be UTC tz-aware.
    Uses:
      - Yesterday High/Low from daily resample
      - Weekly Open from first daily bar of the current week (Monday)
      - London session (07:00-16:00 UTC) High/Low today
      - NY session (13:00-22:00 UTC) High/Low today
    """
    if df_intraday is None or df_intraday.empty:
        return {"error": "No data"}

    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()

    # Daily bars
    d = _to_day(df_intraday)

    # Yesterday levels (last fully completed day)
    # Use yesterday date explicitly
    y_date = today - timedelta(days=1)
    yh = yl = None
    if y_date in d.index.date:
        row = d.loc[d.index.date == y_date].iloc[-1]
        yh = float(row["High"])
        yl = float(row["Low"])

    # Weekly Open (Monday open)
    # Determine Monday date of current week
    monday = today - timedelta(days=today.weekday())
    wo = None
    if monday in d.index.date:
        row = d.loc[d.index.date == monday].iloc[0]
        wo = float(row["Open"])

    # Session ranges for today
    df_today = df_intraday.loc[df_intraday.index.date == today]
    london_h = london_l = ny_h = ny_l = None

    if not df_today.empty:
        # London 07-16 UTC
        london = df_today.between_time("07:00", "16:00")
        if not london.empty:
            london_h = float(london["High"].max())
            london_l = float(london["Low"].min())

        # NY 13-22 UTC
        ny = df_today.between_time("13:00", "22:00")
        if not ny.empty:
            ny_h = float(ny["High"].max())
            ny_l = float(ny["Low"].min())

    return {
        "yh": yh, "yl": yl, "wo": wo,
        "london_h": london_h, "london_l": london_l,
        "ny_h": ny_h, "ny_l": ny_l,
    }

def format_px(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.5f}" if x < 10 else f"{x:.2f}"

def nearest_levels(price: float, levels: dict, atr_val: float, near_mult: float) -> tuple[list, float]:
    """
    Returns list of near level labels and the threshold used.
    """
    if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
        thresh = max(price * 0.001, 0.0)  # fallback: 0.1%
    else:
        thresh = atr_val * near_mult

    candidates = []
    for key, val in levels.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if abs(price - float(val)) <= thresh:
            candidates.append(key)

    return candidates, thresh

# =========================================================
# TradingView embed
# =========================================================

def tv_embed(symbol: str, interval: str) -> None:
    html = f"""
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol={symbol}&interval={interval}&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light"
      style="width:100%;height:360px;"
      frameborder="0"
      allowtransparency="true"
      scrolling="no">
    </iframe>
    """
    components.html(html, height=380)

def score_bar(score: int):
    pct = int(((score + 5) / 10) * 100)
    st.progress(pct)
    st.caption(f"Score: **{score}** (range -5..+5)")

# =========================================================
# Header status
# =========================================================

now = datetime.now(timezone.utc)
st.metric("⏱ Refresh", f"{refresh_minutes} min", delta=f"Main TF: {main_tf_choice}")
st.caption(f"UTC time: {now.strftime('%H:%M')}")

st.markdown("---")

# TF list for confluence
selected_tfs = []
if use_15m: selected_tfs.append("15m")
if use_1h:  selected_tfs.append("1h")
if use_4h:  selected_tfs.append("4h")
if use_1d:  selected_tfs.append("1d")

if len(selected_tfs) == 0:
    st.error("Select at least one timeframe for confluence.")
    st.stop()

vote_threshold = min(vote_threshold, len(selected_tfs))

# =========================================================
# Main grid
# =========================================================

cols = st.columns(3)
market_results = {}

for idx, (name, cfg) in enumerate(markets.items()):
    col = cols[idx % 3]
    with col:
        st.subheader(name)

        # --- Compute MTF scores ---
        tf_scores = {}
        tf_details = {}

        for tf in selected_tfs:
            cfg_tf = TF_CFG[tf]
            df = fetch_market_data(cfg["yf"], period=cfg_tf["period"], interval=cfg_tf["yf_interval"])

            if df.empty or len(df) < 60:
                tf_scores[tf] = 0
                tf_details[tf] = {"error": "No data"}
                continue

            if cfg_tf["resample"]:
                try:
                    df = resample_ohlc(df, cfg_tf["resample"])
                except Exception:
                    pass

            sig = compute_signals(df)
            tf_scores[tf] = sig["score"]
            tf_details[tf] = sig

        conf = confluence_verdict(tf_scores, vote_threshold)

        # Display confluence verdict
        if "BUY" in conf["verdict"]:
            st.success(conf["label"])
        elif "SELL" in conf["verdict"]:
            st.error(conf["label"])
        else:
            st.warning(conf["label"])

        st.write(f"**Votes:** 🟢 {conf['bull']} / 🔴 {conf['bear']} (need {vote_threshold} to align)")
        st.write(f"**Trade bias:** **{conf['verdict']}**")

        tf_table = pd.DataFrame([{
            "TF": tf,
            "Score": tf_scores[tf],
            "State": "BULL" if tf_scores[tf] >= 2 else ("BEAR" if tf_scores[tf] <= -2 else "NEUTRAL")
        } for tf in selected_tfs])
        st.dataframe(tf_table, hide_index=True, use_container_width=True)

        # MAIN TF data + key levels should use intraday if possible
        main_cfg = TF_CFG[main_tf_choice]
        main_df = fetch_market_data(cfg["yf"], period=main_cfg["period"], interval=main_cfg["yf_interval"])
        if not main_df.empty and len(main_df) >= 60 and main_cfg["resample"]:
            try:
                main_df = resample_ohlc(main_df, main_cfg["resample"])
            except Exception:
                pass

        # For key levels, prefer 15m data (more precise)
        kl_df = fetch_market_data(cfg["yf"], period=TF_CFG["15m"]["period"], interval=TF_CFG["15m"]["yf_interval"])
        if kl_df.empty or len(kl_df) < 60:
            # fallback to main df
            kl_df = main_df

        if main_df is not None and not main_df.empty and len(main_df) >= 60:
            main_sig = compute_signals(main_df)
            market_results[name] = {"df": main_df, "sig": main_sig}

            score_bar(main_sig["score"])

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Last", f"{main_sig['price']:.5f}" if main_sig["price"] < 10 else f"{main_sig['price']:.2f}")
            with m2:
                st.metric("Regime", main_sig["regime"])
            with m3:
                if main_sig["atr_pct"] is not None:
                    st.metric("Vol (ATR%)", f"{main_sig['atr_pct']:.2f}% ({main_sig['vol_label']})")
                else:
                    st.metric("Vol (ATR%)", "—")

            # ---- Key Levels panel ----
            levels = compute_key_levels(kl_df)
            if "error" not in levels:
                price = main_sig["price"]
                atr_val = main_sig["atr"] if main_sig["atr"] is not None else np.nan
                near, thresh = nearest_levels(price, {
                    "YH (Yesterday High)": levels["yh"],
                    "YL (Yesterday Low)": levels["yl"],
                    "WO (Weekly Open)": levels["wo"],
                    "London High": levels["london_h"],
                    "London Low": levels["london_l"],
                    "NY High": levels["ny_h"],
                    "NY Low": levels["ny_l"],
                }, atr_val, near_atr_mult)

                with st.expander("🎯 Key Levels (YH/YL/WO + London/NY)", expanded=True):
                    lvl_rows = [
                        ("YH (Yesterday High)", levels["yh"]),
                        ("YL (Yesterday Low)", levels["yl"]),
                        ("WO (Weekly Open)", levels["wo"]),
                        ("London High (today)", levels["london_h"]),
                        ("London Low (today)", levels["london_l"]),
                        ("NY High (today)", levels["ny_h"]),
                        ("NY Low (today)", levels["ny_l"]),
                    ]
                    df_levels = pd.DataFrame([{"Level": a, "Price": format_px(b)} for a, b in lvl_rows])
                    st.dataframe(df_levels, hide_index=True, use_container_width=True)

                    st.caption(f"Near-level threshold: ~ {format_px(thresh)} (≈ ATR x {near_atr_mult})")
                    if near:
                        st.warning("⚠️ Price is near: **" + "**, **".join(near) + "**")
                        st.caption("Tip: nabij key levels → fakeouts/stop-hunts vaker, wacht op bevestiging (break & retest / rejection).")
                    else:
                        st.success("✅ Price is not near a major key level right now.")

        if show_debug:
            with st.expander("Debug / breakdown"):
                st.write("TV:", cfg["tv"])
                st.write("YF:", cfg["yf"])
                st.write("TF scores:", tf_scores)

        # TradingView chart
        tv_interval = TF_CFG[main_tf_choice]["tv_interval"]
        tv_embed(cfg["tv"], tv_interval)

st.markdown("---")

# =========================================================
# DXY correlations (Main TF)
# =========================================================

st.header("💵 DXY Movement & Correlations (Main TF)")

if "DXY" in market_results:
    dxy_df = market_results["DXY"]["df"].copy()
    dxy_df["ret"] = dxy_df["Close"].pct_change()

    lookback = 12 if main_tf_choice in ["15m", "1h"] else 20
    lookback = min(lookback, len(dxy_df) - 1)
    dxy_move = (dxy_df["Close"].iloc[-1] / dxy_df["Close"].iloc[-1 - lookback] - 1) * 100 if lookback > 0 else 0
    st.metric("DXY change (recent)", f"{dxy_move:.2f}%")

    rows = []
    for m in ["US100", "US500", "US30", "EURUSD", "GOLD"]:
        if m in market_results:
            dfm = market_results[m]["df"].copy()
            dfm["ret"] = dfm["Close"].pct_change()
            joined = pd.concat([dxy_df["ret"], dfm["ret"]], axis=1).dropna()
            joined.columns = ["dxy_ret", "m_ret"]
            if len(joined) >= 30:
                corr_now = float(joined["dxy_ret"].rolling(30).corr(joined["m_ret"]).iloc[-1])
                rows.append({"Market": m, "Rolling Corr (30 bars)": corr_now})
            else:
                rows.append({"Market": m, "Rolling Corr (30 bars)": np.nan})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.warning("DXY data not available. Check your yfinance ticker for DXY (default: DX-Y.NYB).")

st.markdown("---")
st.caption("Let op: dit is een indicatie-dashboard, geen financieel advies.")
