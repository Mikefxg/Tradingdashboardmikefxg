# streamlit_app.py
# UnknownFX Dashboard — PRO+++ (Capital.com candles + TradingView charts) — NO STOOQ
# Public mode: sidebar hidden (visitors)
# Admin mode: sidebar visible (EPIC setup + Capital login tools)
#
# Secrets required (Streamlit Cloud -> App -> Settings -> Secrets):
#   CAPITAL_API_KEY = "..."
#   CAPITAL_IDENTIFIER = "your@email.com"
#   CAPITAL_PASSWORD = "yourPassword"
#   CAPITAL_API_BASE = "https://demo-api-capital.backend-capital.com"   # demo
#   # or live base: "https://api-capital.backend-capital.com"
#   ADMIN_PASSWORD = "set-a-strong-admin-password"

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import requests
import streamlit as st


# =========================
# App config
# =========================
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EPICS_FILE = "epics.json"

# TradingView interval mapping (string -> TradingView interval)
TV_INTERVALS = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1D": "D",
}

# Capital.com resolution mapping (string -> Capital resolution)
CAP_RES = {
    "1m": "MINUTE",
    "5m": "MINUTE_5",
    "15m": "MINUTE_15",
    "30m": "MINUTE_30",
    "1h": "HOUR",
    "4h": "HOUR_4",
    "1D": "DAY",
}


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    tv_symbol: str          # TradingView symbol
    epic_hint: str          # hint only, user fills actual Capital EPIC


MARKETS: List[Market] = [
    Market("US100", "US Tech 100", "CAPITALCOM:USTECH100", "bv. 'USTECH100' EPIC van Capital"),
    Market("US500", "US 500",      "CAPITALCOM:US500",     "bv. 'US500' EPIC van Capital"),
    Market("GOLD",  "Gold (XAUUSD)","CAPITALCOM:GOLD",     "bv. 'GOLD' / 'XAUUSD' EPIC van Capital"),
    Market("EURUSD","EURUSD",      "CAPITALCOM:EURUSD",    "bv. 'EURUSD' EPIC van Capital"),
    Market("DXY",   "DXY",         "CAPITALCOM:DXY",       "bv. 'DXY' EPIC van Capital"),
]


# =========================
# Styling (institutional look)
# =========================
st.markdown(
    """
<style>
/* Page background */
.main {background: radial-gradient(1200px 800px at 10% 10%, rgba(60,80,120,.15), transparent 60%),
                    radial-gradient(1000px 600px at 90% 0%, rgba(120,60,80,.10), transparent 55%);
      }
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}

/* Header */
.ufx-header {
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 18px; border-radius: 16px;
  background: rgba(20, 24, 32, 0.65);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.22);
  backdrop-filter: blur(12px);
}
.ufx-title {font-size: 28px; font-weight: 800; letter-spacing: .3px;}
.ufx-sub {opacity: .70; font-size: 13px; margin-top: 2px;}
.ufx-badge {
  padding: 6px 10px; border-radius: 999px;
  background: rgba(0, 180, 120, .16);
  border: 1px solid rgba(0, 180, 120, .30);
  font-weight: 700; font-size: 12px;
}

/* Cards */
.card {
  padding: 14px 14px;
  border-radius: 16px;
  background: rgba(20, 24, 32, 0.55);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.18);
  backdrop-filter: blur(10px);
  height: 100%;
}
.card-title {font-size: 12px; opacity:.70; margin-bottom: 6px;}
.card-value {font-size: 20px; font-weight: 800; letter-spacing: .2px;}
.card-sub {font-size: 12px; opacity:.70; margin-top: 6px;}

/* TradingView container */
.tv-wrap {
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 14px 40px rgba(0,0,0,.22);
}

/* Hide Streamlit footer/menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Helpers: EPIC storage
# =========================
def load_epics() -> Dict[str, str]:
    try:
        with open(EPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # ensure all keys exist
            out = {m.key: "" for m in MARKETS}
            for k, v in data.items():
                out[str(k)] = str(v)
            return out
    except Exception:
        pass
    return {m.key: "" for m in MARKETS}


def save_epics(epics: Dict[str, str]) -> None:
    try:
        with open(EPICS_FILE, "w", encoding="utf-8") as f:
            json.dump(epics, f, indent=2)
    except Exception:
        # If filesystem write fails, we still keep it in session_state.
        pass


# =========================
# Public/Admin mode
# =========================
def is_admin() -> bool:
    # URL param ?admin=1 makes admin available (for you)
    qp = st.query_params
    if str(qp.get("admin", "")).lower() in ("1", "true", "yes"):
        st.session_state["is_admin"] = True

    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False

    if not st.session_state["is_admin"]:
        with st.sidebar:
            st.markdown("### 🔒 Admin login")
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                if pw and pw == st.secrets.get("ADMIN_PASSWORD", ""):
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Wrong password")
    return bool(st.session_state["is_admin"])


ADMIN = is_admin()

# Hide sidebar for public visitors
if not ADMIN:
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] {display:none !important;}
          div[data-testid="collapsedControl"] {display:none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Capital.com API client
# =========================
def _secrets_get(key: str) -> str:
    v = st.secrets.get(key, "")
    return str(v).strip()


def capital_required_secrets_ok() -> Tuple[bool, List[str]]:
    needed = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD", "CAPITAL_API_BASE"]
    missing = [k for k in needed if not _secrets_get(k)]
    return (len(missing) == 0, missing)


def capital_login(force: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (success, info). On success, saves tokens in session_state.
    Tokens: CST, X-SECURITY-TOKEN
    """
    ok, missing = capital_required_secrets_ok()
    if not ok:
        return False, {"error": "Missing secrets", "missing": missing}

    # Throttle login attempts (avoid 429)
    now = time.time()
    cooldown_until = st.session_state.get("capital_login_cooldown_until", 0.0)
    if now < cooldown_until and not force:
        return False, {"error": "Cooldown (429 protection)", "retry_in_sec": int(cooldown_until - now)}

    # Reuse tokens if fresh (20 min)
    token = st.session_state.get("capital_token")
    if token and not force:
        age = now - float(token.get("ts", 0))
        if age < 20 * 60 and token.get("cst") and token.get("xst"):
            return True, {"status": "reused", "age_sec": int(age)}

    base = _secrets_get("CAPITAL_API_BASE").rstrip("/")
    url = f"{base}/api/v1/session"
    api_key = _secrets_get("CAPITAL_API_KEY")
    identifier = _secrets_get("CAPITAL_IDENTIFIER")
    password = _secrets_get("CAPITAL_PASSWORD")

    headers = {
        "X-CAP-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "identifier": identifier,
        "password": password,
        "encryptedPassword": False,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
    except Exception as e:
        return False, {"error": "Network error", "detail": str(e), "url": url}

    cst = r.headers.get("CST")
    xst = r.headers.get("X-SECURITY-TOKEN")

    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}

    # Handle rate limit
    if r.status_code == 429:
        st.session_state["capital_login_cooldown_until"] = time.time() + 90  # 1.5 min cooldown
        return False, {"status": 429, "body": body, "url": url, "note": "Too many requests. Wait 1-2 min."}

    # Success criteria: 2xx and tokens exist
    if 200 <= r.status_code < 300 and cst and xst:
        st.session_state["capital_token"] = {
            "cst": cst,
            "xst": xst,
            "ts": time.time(),
            "base": base,
        }
        return True, {"status": r.status_code, "body": body, "url": url}

    return False, {
        "status": r.status_code,
        "body": body,
        "got_CST": bool(cst),
        "got_X_SECURITY_TOKEN": bool(xst),
        "url": url,
    }


def capital_headers() -> Dict[str, str]:
    token = st.session_state.get("capital_token") or {}
    api_key = _secrets_get("CAPITAL_API_KEY")
    return {
        "X-CAP-API-KEY": api_key,
        "CST": token.get("cst", ""),
        "X-SECURITY-TOKEN": token.get("xst", ""),
        "Accept": "application/json",
    }


def fetch_candles(epic: str, timeframe_key: str, max_points: int = 300) -> Tuple[bool, Any]:
    """
    Uses Capital /api/v1/prices/{epic}?resolution=...&max=...
    Returns (success, df_or_error)
    """
    token = st.session_state.get("capital_token")
    if not token:
        ok, info = capital_login(force=False)
        if not ok:
            return False, info
        token = st.session_state.get("capital_token")

    base = (token.get("base") or _secrets_get("CAPITAL_API_BASE")).rstrip("/")
    resolution = CAP_RES.get(timeframe_key, "MINUTE_15")
    url = f"{base}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": int(max_points)}

    try:
        r = requests.get(url, headers=capital_headers(), params=params, timeout=15)
    except Exception as e:
        return False, {"error": "Network error", "detail": str(e), "url": url}

    # If unauthorized/expired -> try relogin once
    if r.status_code in (401, 403):
        ok, info = capital_login(force=True)
        if not ok:
            return False, {"error": "Login refresh failed", "login": info}
        r = requests.get(url, headers=capital_headers(), params=params, timeout=15)

    if r.status_code == 429:
        return False, {"error": "429 too many requests", "url": url, "hint": "Increase refresh interval."}

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}

    if not (200 <= r.status_code < 300):
        return False, {"status": r.status_code, "body": data, "url": url}

    prices = data.get("prices") or []
    if not prices:
        return False, {"error": "No prices returned", "body": data, "url": url}

    rows = []
    for p in prices:
        snap = p.get("snapshotTimeUTC") or p.get("snapshotTime") or ""
        o = p.get("openPrice", {})
        c = p.get("closePrice", {})
        h = p.get("highPrice", {})
        l = p.get("lowPrice", {})

        def mid(x):
            if x is None:
                return None
            bid = x.get("bid")
            ask = x.get("ask")
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
            if bid is not None:
                return float(bid)
            if ask is not None:
                return float(ask)
            return None

        rows.append(
            {
                "time": snap,
                "open": mid(o),
                "high": mid(h),
                "low": mid(l),
                "close": mid(c),
            }
        )

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.dropna(subset=["close"])
    if df.empty:
        return False, {"error": "Parsed DF is empty", "body": data}
    return True, df


# =========================
# Indicators + Market Sentiment
# =========================
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 2:
        return float("nan")
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 2:
        return float("nan")
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
    atr = tr.rolling(period).mean().iloc[-1]
    last = close.iloc[-1]
    if last == 0 or pd.isna(last) or pd.isna(atr):
        return float("nan")
    return float((atr / last) * 100.0)


def market_sentiment(df: pd.DataFrame) -> str:
    """
    Simple, stable institutional-style sentiment:
    - +1 if last close > previous close (momentum)
    - +1 if RSI > 55
    - -1 if RSI < 45
    score >= 2 => BULLISH
    score <= -2 => BEARISH
    else NEUTRAL
    """
    if df is None or len(df) < 20:
        return "NEUTRAL"

    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    rsi = compute_rsi(close, 14)

    score = 0
    # price momentum
    score += 1 if last > prev else -1

    # rsi momentum
    if pd.notna(rsi):
        if rsi > 55:
            score += 1
        elif rsi < 45:
            score -= 1

    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"


# =========================
# TradingView embed (BIGGER)
# =========================
def tradingview_embed(symbol: str, interval: str, height: int = 900) -> None:
    # Bigger chart: 900px default
    html = f"""
<div class="tv-wrap">
  <div class="tradingview-widget-container">
    <div id="tv_chart"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "autosize": false,
        "width": "100%",
        "height": {height},
        "symbol": "{symbol}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "save_image": false,
        "calendar": false,
        "details": false,
        "hotlist": false,
        "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
        "withdateranges": true,
        "support_host": "https://www.tradingview.com"
      }});
    </script>
  </div>
</div>
"""
    st.components.v1.html(html, height=height + 40, scrolling=False)


# =========================
# Auto refresh (NO external module)
# =========================
def meta_refresh(seconds: int) -> None:
    if seconds <= 0:
        return
    st.markdown(
        f"""<meta http-equiv="refresh" content="{int(seconds)}">""",
        unsafe_allow_html=True,
    )


# =========================
# Header
# =========================
st.markdown(
    """
<div class="ufx-header">
  <div>
    <div class="ufx-title">🚀 UnknownFX Dashboard</div>
    <div class="ufx-sub">Institutional view · Capital.com candles · TradingView charts · Bull/Bear/Neutral · Public/Private mode</div>
  </div>
  <div class="ufx-badge">PRO+++</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# =========================
# Load EPICs (persisted)
# =========================
if "epics" not in st.session_state:
    st.session_state["epics"] = load_epics()


# =========================
# Sidebar (ADMIN ONLY)
# =========================
if ADMIN:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        auto_refresh_on = st.toggle("Auto refresh", value=True)
        refresh_min = st.slider("Refresh interval (min)", 1, 30, 10, 1)
        chart_tf = st.selectbox("TradingView interval (chart)", list(TV_INTERVALS.keys()), index=2)  # 15m default

        st.markdown("---")
        st.markdown("## 🔑 Capital.com (required)")

        ok, missing = capital_required_secrets_ok()
        if not ok:
            st.error(f"Missing secrets: {', '.join(missing)}")
        else:
            colA, colB = st.columns(2)
            with colA:
                if st.button("Login / Refresh", use_container_width=True):
                    success, info = capital_login(force=True)
                    if success:
                        st.success("Capital login OK ✅")
                    else:
                        st.error(f"Capital login failed: {info}")
            with colB:
                if st.button("Clear session", use_container_width=True):
                    st.session_state.pop("capital_token", None)
                    st.session_state.pop("capital_login_cooldown_until", None)
                    st.success("Session cleared.")

            token = st.session_state.get("capital_token")
            if token:
                age = int(time.time() - float(token.get("ts", 0)))
                st.caption(f"Session active · age {age}s · base: {token.get('base','')}")
            else:
                st.caption("No active session yet.")

        st.markdown("---")
        st.markdown("## 🧭 EPIC map (1x instellen)")
        st.caption("Plak jouw EPICs hier (van Capital).")

        for m in MARKETS:
            val = st.text_input(
                f"{m.key} EPIC",
                value=st.session_state["epics"].get(m.key, ""),
                placeholder=m.epic_hint,
                key=f"epic_{m.key}",
            )
            st.session_state["epics"][m.key] = val.strip()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save EPICs", use_container_width=True):
                save_epics(st.session_state["epics"])
                st.success("EPICs saved ✅")
        with c2:
            if st.button("Clear EPICs", use_container_width=True):
                st.session_state["epics"] = {m.key: "" for m in MARKETS}
                save_epics(st.session_state["epics"])
                st.warning("EPICs cleared.")

        st.caption("Tip: krijg je 429 → refresh hoger (10+ min) en niet spam klikken.")


# =========================
# Public mode defaults (no sidebar)
# =========================
if not ADMIN:
    auto_refresh_on = True
    refresh_min = 10
    chart_tf = "15m"


# Apply refresh
if auto_refresh_on:
    meta_refresh(refresh_min * 60)


# =========================
# Main controls
# =========================
top_left, top_right = st.columns([2.2, 1.0], vertical_alignment="top")

with top_left:
    market_key = st.selectbox(
        "Market",
        options=[m.key for m in MARKETS],
        format_func=lambda k: next(x.label for x in MARKETS if x.key == k),
        index=0,
        label_visibility="collapsed",
    )

with top_right:
    tf = st.selectbox("Timeframe", options=list(TV_INTERVALS.keys()), index=list(TV_INTERVALS.keys()).index(chart_tf))
    st.caption("Sentiment komt uit Capital candles (RSI + momentum).")


market = next(m for m in MARKETS if m.key == market_key)
epic = (st.session_state.get("epics", {}) or {}).get(market.key, "").strip()
tv_interval = TV_INTERVALS.get(tf, "15")
cap_tf = tf if tf in CAP_RES else "15m"


# =========================
# Data fetch + KPI cards + Sentiment
# =========================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1.0, 1.0, 1.0, 1.0, 1.3], gap="large")

price_text = "—"
chg_text = "—"
rsi_text = "—"
atr_text = "—"
sentiment = "NEUTRAL"
status_text = ""

df = None

if epic:
    ok, out = fetch_candles(epic=epic, timeframe_key=cap_tf, max_points=300)
    if ok:
        df = out
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
        chg = ((last - prev) / prev) * 100 if prev else 0.0
        rsi = compute_rsi(df["close"], 14)
        atrp = compute_atr_pct(df, 14)

        sentiment = market_sentiment(df)

        price_text = f"{last:,.2f}"
        chg_text = f"{chg:+.2f}%"
        rsi_text = f"{rsi:.1f}" if pd.notna(rsi) else "—"
        atr_text = f"{atrp:.2f}%" if pd.notna(atrp) else "—"
        status_text = "Capital candles: OK ✅"
    else:
        status_text = f"Capital candles: ERROR — {out}"
else:
    status_text = "EPIC ontbreekt voor deze market (admin: vul EPIC map in)."

with kpi1:
    st.markdown(
        '<div class="card"><div class="card-title">Price (Capital)</div>'
        f'<div class="card-value">{price_text}</div>'
        f'<div class="card-sub">{market.label}</div></div>',
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        '<div class="card"><div class="card-title">Change</div>'
        f'<div class="card-value">{chg_text}</div>'
        f'<div class="card-sub">Last candle vs prev</div></div>',
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        '<div class="card"><div class="card-title">RSI (14)</div>'
        f'<div class="card-value">{rsi_text}</div>'
        f'<div class="card-sub">Momentum</div></div>',
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        '<div class="card"><div class="card-title">Volatility (ATR%)</div>'
        f'<div class="card-value">{atr_text}</div>'
        f'<div class="card-sub">{status_text}</div></div>',
        unsafe_allow_html=True,
    )

with kpi5:
    # Big clear sentiment indicator
    if sentiment == "BULLISH":
        st.success("🟢 BULLISH")
    elif sentiment == "BEARISH":
        st.error("🔴 BEARISH")
    else:
        st.warning("🟡 NEUTRAL")
    st.caption("Bias: price momentum + RSI (55/45 zones).")

st.write("")


# =========================
# Big Chart (TradingView) — even bigger
# =========================
tradingview_embed(symbol=market.tv_symbol, interval=tv_interval, height=980)


# =========================
# Optional: show candles table (ADMIN only)
# =========================
if ADMIN:
    with st.expander("Debug · Capital candles dataframe", expanded=False):
        if df is None:
            st.warning("No DF available (missing epic or error).")
        else:
            st.dataframe(df.tail(80), use_container_width=True)


# =========================
# Footer note
# =========================
st.caption(
    "Public mode: sidebar hidden. Admin mode: open with `?admin=1` and login (ADMIN_PASSWORD). "
    "EPICs stored in epics.json + session_state."
)
