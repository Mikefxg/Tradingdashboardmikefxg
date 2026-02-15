import os
import math
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone, timedelta

# =========================================================
# UnknownFX Dashboard PRO+ (Mobile + Telegram + Calendar + "AI" Probability)
# =========================================================

st.set_page_config(page_title="UnknownFX Dashboard", layout="wide")

# -------------------------
# STYLE (dark + mobile-ish)
# -------------------------
st.markdown("""
<style>
  :root { --bg:#0e1117; --card:#151a22; --muted:#9aa4b2; --line:#232a36; }
  body { background-color: var(--bg); color: #e6e8ee; }
  .ufx-title { font-size: 42px; font-weight: 900; margin: 0; }
  .ufx-sub { color: var(--muted); margin-top: 6px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px 14px; }
  .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 800; font-size: 12px; }
  .bull { background:#0f2a1b; color:#4ade80; border:1px solid #1b5a33; }
  .bear { background:#2a0f12; color:#fb7185; border:1px solid #6b1a24; }
  .neut { background:#1d2430; color:#cbd5e1; border:1px solid #334155; }
  .kv { color: var(--muted); font-size: 12px; }
  .big { font-size: 18px; font-weight: 800; }
  .small { font-size: 12px; color: var(--muted); }
  /* Make charts a bit more compact on small screens */
  iframe { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ufx-title">🚀 UnknownFX Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="ufx-sub">MTF Confluence • Key Levels • Confidence % • Alerts • DXY Correlation • Economic Calendar • Trend Probability</div>', unsafe_allow_html=True)

# -------------------------
# Refresh + caching
# -------------------------
REFRESH_SECONDS = 120  # 2 minutes
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("⚙️ Settings")

main_tf_choice = st.sidebar.selectbox("Main chart timeframe", ["15m", "1h", "4h", "1d"], index=0)
show_charts = st.sidebar.checkbox("Show charts", True)
show_debug = st.sidebar.checkbox("Show debug", False)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Multi-timeframe confluence")
use_15m = st.sidebar.checkbox("Use 15m", True)
use_1h  = st.sidebar.checkbox("Use 1h", True)
use_4h  = st.sidebar.checkbox("Use 4h", True)
use_1d  = st.sidebar.checkbox("Use 1d", False)
vote_threshold = st.sidebar.slider("Votes needed for ALIGNED", 1, 4, 2, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Key Levels")
near_atr_mult = st.sidebar.slider("Near-level threshold (ATR x)", 0.05, 0.50, 0.20, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Telegram alerts")
st.sidebar.caption("Set env vars on Streamlit Cloud:\nTELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
enable_telegram = st.sidebar.checkbox("Enable Telegram alerts", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Economic Calendar")
st.sidebar.caption("Uses TradingEconomics API if keys are set.\nSet env vars:\nTE_API_KEY + TE_API_SECRET (or TE_API_KEY only if your plan supports it)")
calendar_enabled = st.sidebar.checkbox("Show Economic Calendar (High impact only)", value=True)

# -------------------------
# Markets (TV embed symbols + data tickers)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Symbols (editable)")

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
    tv_sym = st.sidebar.text_input(f"{k} TradingView", value=v["tv"])
    yf_sym = st.sidebar.text_input(f"{k} Data ticker", value=v["yf"])
    markets[k] = {"tv": tv_sym.strip(), "yf": yf_sym.strip()}

# =========================================================
# Timeframe configs (yfinance interval + optional resample)
# =========================================================
TF_CFG = {
    "15m": {"yf_interval": "15m", "tv_interval": "15",  "resample": None, "period": "10d"},
    "1h":  {"yf_interval": "60m", "tv_interval": "60",  "resample": None, "period": "60d"},
    "4h":  {"yf_interval": "60m", "tv_interval": "240", "resample": "4H", "period": "60d"},
    "1d":  {"yf_interval": "1d",  "tv_interval": "D",   "resample": None, "period": "2y"},
}

selected_tfs = []
if use_15m: selected_tfs.append("15m")
if use_1h:  selected_tfs.append("1h")
if use_4h:  selected_tfs.append("4h")
if use_1d:  selected_tfs.append("1d")
if not selected_tfs:
    st.error("Select at least one timeframe for confluence.")
    st.stop()

vote_threshold = min(vote_threshold, len(selected_tfs))

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
# Data fetch (cache ~110s, matches 120s refresh)
# =========================================================
@st.cache_data(ttl=110)
def fetch_market_data(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

# =========================================================
# Sentiment + Score + Regime + Volatility + Probability
# =========================================================
def compute_signals(df: pd.DataFrame) -> dict:
    df = df.copy()
    close = df["Close"]

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

    # Score (-5..+5)
    score = 0
    reasons = []

    # Trend stack
    if price > ema20v and ema20v > ema50v:
        score += 2; reasons.append("+2 Trend stack up (P>20>50)")
    elif price < ema20v and ema20v < ema50v:
        score -= 2; reasons.append("-2 Trend stack down (P<20<50)")
    else:
        reasons.append("0 Mixed stack")

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
            score += 1; reasons.append("+1 MACD hist>0")
        elif macdh < 0:
            score -= 1; reasons.append("-1 MACD hist<0")
        else:
            reasons.append("0 MACD flat")

    # EMA slope
    if ema20_slope > 0:
        score += 1; reasons.append("+1 EMA20 slope up")
    elif ema20_slope < 0:
        score -= 1; reasons.append("-1 EMA20 slope down")

    atr_pct = None
    vol_label = None
    if not np.isnan(atrv) and price != 0:
        atr_pct = (atrv / price) * 100
        if atr_pct >= 1.2:
            score -= 1; reasons.append("-1 ATR% high → reduce confidence")
            vol_label = "HIGH"
        elif atr_pct <= 0.4:
            vol_label = "LOW"
        else:
            vol_label = "NORMAL"

    score = int(np.clip(score, -5, 5))

    if score >= 2:
        sentiment = "BULLISH"; bias = "BUY"
    elif score <= -2:
        sentiment = "BEARISH"; bias = "SELL"
    else:
        sentiment = "NEUTRAL"; bias = "WAIT"

    # Regime
    if not np.isnan(adxv):
        regime = "TRENDING" if adxv >= 25 else "RANGING"
    else:
        regime = "TRENDING" if abs(ema20_slope) > 0 else "RANGING"

    # "AI-style" probability model (no external AI required)
    # Features -> probability via sigmoid:
    # - score
    # - RSI distance
    # - ADX strength
    # - volatility penalty
    rsi_feat = 0.0 if np.isnan(rsiv) else (rsiv - 50.0) / 10.0       # ~[-3..+3]
    adx_feat = 0.0 if np.isnan(adxv) else (adxv - 20.0) / 10.0       # ~[-? .. +?]
    vol_pen  = 0.0
    if atr_pct is not None:
        if atr_pct >= 1.2: vol_pen = 0.8
        elif atr_pct <= 0.4: vol_pen = -0.1

    # Logit: positive -> bullish probability
    logit = (0.85 * score) + (0.35 * rsi_feat) + (0.25 * adx_feat) - (0.55 * vol_pen)
    p_bull = 1.0 / (1.0 + math.exp(-logit))
    p_bear = 1.0 - p_bull

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
        "macdh": macdh,
        "p_bull": float(p_bull),
        "p_bear": float(p_bear),
        "reasons": reasons,
    }

# =========================================================
# Confluence
# =========================================================
def sentiment_vote(score: int) -> int:
    if score >= 2: return 1
    if score <= -2: return -1
    return 0

def confluence_verdict(tf_scores: dict, threshold: int) -> dict:
    votes = {tf: sentiment_vote(s) for tf, s in tf_scores.items()}
    bull = sum(1 for v in votes.values() if v == 1)
    bear = sum(1 for v in votes.values() if v == -1)

    if bull >= threshold:
        return {"verdict": "ALIGNED BUY", "pill": "bull", "bull": bull, "bear": bear}
    if bear >= threshold:
        return {"verdict": "ALIGNED SELL", "pill": "bear", "bull": bull, "bear": bear}
    return {"verdict": "MIXED / WAIT", "pill": "neut", "bull": bull, "bear": bear}

def confidence_percent(tf_scores: dict) -> int:
    votes = [sentiment_vote(s) for s in tf_scores.values()]
    bull = sum(1 for v in votes if v == 1)
    bear = sum(1 for v in votes if v == -1)
    strongest = max(bull, bear)
    return int(round((strongest / max(1, len(votes))) * 100))

# =========================================================
# Key Levels
# =========================================================
def _to_day(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()

def compute_key_levels(df_intraday: pd.DataFrame) -> dict:
    if df_intraday is None or df_intraday.empty:
        return {"error": "No data"}
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()

    d = _to_day(df_intraday)

    y_date = today - timedelta(days=1)
    yh = yl = None
    if any(d.index.date == y_date):
        row = d.loc[d.index.date == y_date].iloc[-1]
        yh = float(row["High"])
        yl = float(row["Low"])

    monday = today - timedelta(days=today.weekday())
    wo = None
    if any(d.index.date == monday):
        row = d.loc[d.index.date == monday].iloc[0]
        wo = float(row["Open"])

    df_today = df_intraday.loc[df_intraday.index.date == today]
    london_h = london_l = ny_h = ny_l = None

    if not df_today.empty:
        london = df_today.between_time("07:00", "16:00")
        if not london.empty:
            london_h = float(london["High"].max())
            london_l = float(london["Low"].min())
        ny = df_today.between_time("13:00", "22:00")
        if not ny.empty:
            ny_h = float(ny["High"].max())
            ny_l = float(ny["Low"].min())

    return {"yh": yh, "yl": yl, "wo": wo, "london_h": london_h, "london_l": london_l, "ny_h": ny_h, "ny_l": ny_l}

def format_px(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.5f}" if x < 10 else f"{x:.2f}"

def nearest_levels(price: float, levels: dict, atr_val: float, near_mult: float):
    if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
        thresh = price * 0.001  # 0.1%
    else:
        thresh = atr_val * near_mult

    near = []
    for label, val in levels.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if abs(price - float(val)) <= thresh:
            near.append(label)
    return near, thresh

# =========================================================
# TradingView embed
# =========================================================
def tv_embed(symbol: str, interval: str) -> None:
    html = f"""
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol={symbol}&interval={interval}&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark"
      style="width:100%;height:320px;"
      frameborder="0"
      allowtransparency="true"
      scrolling="no">
    </iframe>
    """
    components.html(html, height=340)

# =========================================================
# Telegram alerts
# =========================================================
def send_telegram(msg: str) -> Tuple[bool, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=12)
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {r.text[:120]}"
        return True, "sent"
    except Exception as e:
        return False, str(e)

# =========================================================
# Economic Calendar (TradingEconomics - optional)
# =========================================================
@st.cache_data(ttl=110)
def fetch_te_calendar_high() -> pd.DataFrame:
    """
    TradingEconomics calendar requires API credentials.
    If none found -> returns empty df (no errors).
    """
    api_key = os.getenv("TE_API_KEY", "").strip()
    api_secret = os.getenv("TE_API_SECRET", "").strip()

    if not api_key:
        return pd.DataFrame()

    # TradingEconomics supports multiple auth methods depending on plan.
    # We'll try "key:secret" if secret exists, else key only.
    if api_secret:
        auth = f"{api_key}:{api_secret}"
    else:
        auth = api_key

    # Next 24h events (High impact only if supported; otherwise filter later)
    # Endpoint commonly used: /calendar
    # We'll request next day and then filter keywords/importance if returned.
    url = f"https://api.tradingeconomics.com/calendar?c={auth}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Normalize
        rows = []
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=24)

        for e in data if isinstance(data, list) else []:
            # fields vary: "Date", "Country", "Category", "Event", "Importance", "Actual", "Forecast", "Previous"
            dt_raw = e.get("Date") or e.get("date") or e.get("Datetime")
            try:
                dt = pd.to_datetime(dt_raw, utc=True)
            except Exception:
                dt = None

            if dt is not None and (dt < now - timedelta(minutes=1) or dt > end):
                continue

            imp = str(e.get("Importance") or e.get("importance") or "").lower()
            # "High" often appears as 3, "high", or "3"
            is_high = ("high" in imp) or (imp.strip() in ["3", "high importance", "high"])
            # If API doesn't provide importance, we keep it but mark unknown
            rows.append({
                "Time (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt is not None else "",
                "Country": e.get("Country") or e.get("country") or "",
                "Event": e.get("Event") or e.get("event") or e.get("Category") or "",
                "Importance": e.get("Importance") or e.get("importance") or "—",
                "Actual": e.get("Actual") or "",
                "Forecast": e.get("Forecast") or "",
                "Previous": e.get("Previous") or "",
                "_is_high": is_high,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Filter to high if possible; if no importance signal, show all upcoming.
        if df["_is_high"].any():
            df = df[df["_is_high"]].copy()
        df = df.drop(columns=["_is_high"], errors="ignore")
        return df

    except Exception:
        return pd.DataFrame()

# =========================================================
# Header metrics
# =========================================================
now = datetime.now(timezone.utc)
st.markdown(f"<div class='small'>Auto-refresh: every <b>{REFRESH_SECONDS}</b>s • UTC: <b>{now.strftime('%H:%M')}</b></div>", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# MAIN GRID (cards)
# =========================================================
# Use Streamlit columns; on mobile it stacks naturally.
cols = st.columns(3)
market_main_tf_data = {}

for idx, (mkt, sym) in enumerate(markets.items()):
    with cols[idx % 3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='big'>{mkt}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>{sym['yf']} • {sym['tv']}</div>", unsafe_allow_html=True)

        # --- MTF scores ---
        tf_scores = {}
        for tf in selected_tfs:
            cfg = TF_CFG[tf]
            df = fetch_market_data(sym["yf"], period=cfg["period"], interval=cfg["yf_interval"])
            if df.empty or len(df) < 80:
                tf_scores[tf] = 0
                continue
            if cfg["resample"]:
                try:
                    df = resample_ohlc(df, cfg["resample"])
                except Exception:
                    pass
            sig = compute_signals(df)
            tf_scores[tf] = sig["score"]

        conf = confluence_verdict(tf_scores, vote_threshold)
        conf_pct = confidence_percent(tf_scores)

        # --- Alerts on change ---
        state_key = f"state_{mkt}"
        prev = st.session_state.get(state_key)
        current = conf["verdict"]
        if prev is None:
            st.session_state[state_key] = current
            prev = current

        changed = (prev != current)
        if changed:
            st.session_state[state_key] = current

        # show verdict pill
        st.markdown(f"<span class='pill {conf['pill']}'>{conf['verdict']}</span> <span class='kv'>votes 🟢{conf['bull']} / 🔴{conf['bear']} • threshold {vote_threshold}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>Confidence: <b>{conf_pct}%</b></div>", unsafe_allow_html=True)

        if changed:
            st.warning(f"🚨 Confluence changed: {prev} → {current}")

            if enable_telegram:
                msg = f"UnknownFX ALERT: {mkt} changed {prev} -> {current} | votes bull={conf['bull']} bear={conf['bear']} | conf={conf_pct}%"
                ok, info = send_telegram(msg)
                if show_debug:
                    st.caption(f"Telegram: {ok} ({info})")

        # --- Main TF signals & key levels ---
        main_cfg = TF_CFG[main_tf_choice]
        main_df = fetch_market_data(sym["yf"], period=main_cfg["period"], interval=main_cfg["yf_interval"])
        if main_cfg["resample"] and not main_df.empty:
            try:
                main_df = resample_ohlc(main_df, main_cfg["resample"])
            except Exception:
                pass

        if main_df.empty or len(main_df) < 80:
            st.error("No data (main TF).")
            if show_charts:
                tv_embed(sym["tv"], main_cfg["tv_interval"])
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        main_sig = compute_signals(main_df)
        market_main_tf_data[mkt] = {"df": main_df, "sig": main_sig}

        # sentiment pill
        s = main_sig["sentiment"]
        pill = "bull" if s == "BULLISH" else ("bear" if s == "BEARISH" else "neut")
        st.markdown(f"<span class='pill {pill}'>{s}</span> <span class='kv'>Bias: <b>{main_sig['bias']}</b> • Score: <b>{main_sig['score']}</b></span>", unsafe_allow_html=True)

        # probability
        st.markdown(f"<div class='kv'>Trend probability: 🟢 <b>{int(main_sig['p_bull']*100)}%</b> / 🔴 <b>{int(main_sig['p_bear']*100)}%</b></div>", unsafe_allow_html=True)

        # quick stats
        cA, cB, cC = st.columns(3)
        cA.metric("Last", f"{main_sig['price']:.5f}" if main_sig["price"] < 10 else f"{main_sig['price']:.2f}")
        cB.metric("Regime", main_sig["regime"])
        vol_text = "—" if main_sig["atr_pct"] is None else f"{main_sig['atr_pct']:.2f}% ({main_sig['vol_label']})"
        cC.metric("Vol (ATR%)", vol_text)

        # key levels uses 15m data for precision
        kl_df = fetch_market_data(sym["yf"], period=TF_CFG["15m"]["period"], interval=TF_CFG["15m"]["yf_interval"])
        if kl_df.empty:
            kl_df = main_df

        levels = compute_key_levels(kl_df)
        if "error" not in levels:
            lvl_map = {
                "YH": levels["yh"],
                "YL": levels["yl"],
                "WO": levels["wo"],
                "LON-H": levels["london_h"],
                "LON-L": levels["london_l"],
                "NY-H": levels["ny_h"],
                "NY-L": levels["ny_l"],
            }

            near, thresh = nearest_levels(main_sig["price"], lvl_map, main_sig["atr"], near_atr_mult)

            with st.expander("🎯 Key Levels", expanded=True):
                df_lvl = pd.DataFrame([{"Level": k, "Price": format_px(v)} for k, v in lvl_map.items()])
                st.dataframe(df_lvl, hide_index=True, use_container_width=True)
                st.caption(f"Near threshold ≈ {format_px(thresh)} (ATR x {near_atr_mult})")
                if near:
                    st.warning("⚠️ Price near: " + ", ".join(near))
                else:
                    st.success("✅ Not near major levels")

        # show chart
        if show_charts:
            tv_embed(sym["tv"], main_cfg["tv_interval"])

        if show_debug:
            with st.expander("Debug"):
                st.write("TF scores:", tf_scores)
                st.write("Main reasons:", main_sig["reasons"])

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# Economic Calendar section
# =========================================================
st.header("🌍 Economic Calendar (next 24h)")
if calendar_enabled:
    df_cal = fetch_te_calendar_high()
    if df_cal.empty:
        st.info("Calendar disabled or no API key set. Set env vars TE_API_KEY (and optionally TE_API_SECRET) to enable.")
    else:
        st.dataframe(df_cal, use_container_width=True, height=320)
else:
    st.caption("Calendar hidden (disabled in sidebar).")

st.markdown("---")

# =========================================================
# DXY correlations (main TF)
# =========================================================
st.header("💵 DXY Correlation Snapshot (Main TF)")

if "DXY" in market_main_tf_data:
    dxy_df = market_main_tf_data["DXY"]["df"].copy()
    dxy_df["ret"] = dxy_df["Close"].pct_change()

    rows = []
    for m in ["US100", "US500", "US30", "EURUSD", "GOLD"]:
        if m in market_main_tf_data:
            dfm = market_main_tf_data[m]["df"].copy()
            dfm["ret"] = dfm["Close"].pct_change()
            joined = pd.concat([dxy_df["ret"], dfm["ret"]], axis=1).dropna()
            joined.columns = ["dxy_ret", "m_ret"]
            if len(joined) >= 30:
                corr = joined["dxy_ret"].rolling(30).corr(joined["m_ret"]).iloc[-1]
                rows.append({"Market": m, "Rolling Corr (30 bars)": float(corr) if pd.notna(corr) else np.nan})
            else:
                rows.append({"Market": m, "Rolling Corr (30 bars)": np.nan})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.warning("DXY data not available (check your DXY yfinance ticker).")

st.caption("⚠️ Educational dashboard – not financial advice.")
