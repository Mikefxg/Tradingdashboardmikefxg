# streamlit_app.py
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ----------------------------
# Page config + Institutional CSS
# ----------------------------
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
/* --- Base --- */
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }
small, .muted { color: rgba(250,250,250,0.70); }

/* --- Header strip --- */
.ufx-hero {
  border-radius: 16px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,24,39,0.95));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,0.28);
}
.ufx-hero-title {
  font-size: 36px;
  font-weight: 800;
  margin: 0;
  color: rgba(255,255,255,0.98);
}
.ufx-hero-sub {
  margin-top: 4px;
  font-size: 13px;
  color: rgba(255,255,255,0.70);
}

/* --- Cards --- */
.ufx-card {
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  background: rgba(17,24,39,0.75);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 25px rgba(0,0,0,0.20);
}
.ufx-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.ufx-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(2,6,23,0.35);
  color: rgba(255,255,255,0.88);
}
.ufx-kpi {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(2,6,23,0.35);
  border: 1px solid rgba(255,255,255,0.08);
  min-width: 160px;
}
.ufx-kpi .label { font-size: 11px; color: rgba(255,255,255,0.70); }
.ufx-kpi .value { font-size: 20px; font-weight: 850; color: rgba(255,255,255,0.98); }

/* --- Badges --- */
.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
}
.badge-buy { background: rgba(16,185,129,0.18); color: rgba(209,250,229,0.95); }
.badge-sell { background: rgba(239,68,68,0.18); color: rgba(254,226,226,0.95); }
.badge-neutral { background: rgba(245,158,11,0.16); color: rgba(255,237,213,0.95); }

/* --- TradingView frame --- */
.ufx-tv {
  width: 100%;
  height: 560px;
  border: none;
  border-radius: 14px;
  overflow: hidden;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] { background: rgba(2,6,23,0.95); border-right: 1px solid rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------
# Auto refresh (no extra package)
# ----------------------------
def meta_refresh(seconds: int) -> None:
    st.markdown(
        f"""<meta http-equiv="refresh" content="{int(seconds)}">""",
        unsafe_allow_html=True,
    )


# ----------------------------
# Helpers: Indicators
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def bias_from_indicators(close: float, sma50: float, rsi14: float) -> str:
    # simple, stable "institutional-like" bias rule
    if np.isnan(sma50) or np.isnan(rsi14):
        return "NEUTRAL"
    if close > sma50 and rsi14 >= 52:
        return "BUY"
    if close < sma50 and rsi14 <= 48:
        return "SELL"
    return "NEUTRAL"


def badge_html(text: str) -> str:
    t = text.upper().strip()
    if t == "BUY":
        cls = "badge badge-buy"
    elif t == "SELL":
        cls = "badge badge-sell"
    else:
        cls = "badge badge-neutral"
    return f"<span class='{cls}'>{t}</span>"


# ----------------------------
# Capital.com API client
# ----------------------------
class CapitalClient:
    def __init__(self, base_url: str, api_key: str, identifier: str, password: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.timeout = timeout

    def _headers_base(self) -> Dict[str, str]:
        return {
            "X-CAP-API-KEY": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def login(self) -> Tuple[bool, Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/session"
        payload = {"identifier": self.identifier, "password": self.password, "encryptedPassword": False}
        r = requests.post(url, headers=self._headers_base(), json=payload, timeout=self.timeout)

        cst = r.headers.get("CST")
        xst = r.headers.get("X-SECURITY-TOKEN")

        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        ok = (200 <= r.status_code < 300) and bool(cst) and bool(xst)
        return ok, {
            "status": r.status_code,
            "body": body,
            "cst": cst,
            "x_security_token": xst,
            "url": url,
        }

    def _authed_headers(self, cst: str, xst: str) -> Dict[str, str]:
        h = self._headers_base()
        h["CST"] = cst
        h["X-SECURITY-TOKEN"] = xst
        return h

    def search_markets(self, cst: str, xst: str, search_term: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/v1/markets"
        params = {"searchTerm": search_term}
        r = requests.get(url, headers=self._authed_headers(cst, xst), params=params, timeout=self.timeout)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return (200 <= r.status_code < 300), {"status": r.status_code, "body": body, "url": url}

    def get_prices(
        self,
        cst: str,
        xst: str,
        epic: str,
        resolution: str,
        points: int = 200,
    ) -> Tuple[bool, Any]:
        # Uses /prices/{epic}?resolution=...&max=...
        url = f"{self.base_url}/api/v1/prices/{epic}"
        params = {"resolution": resolution, "max": int(points)}
        r = requests.get(url, headers=self._authed_headers(cst, xst), params=params, timeout=self.timeout)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return (200 <= r.status_code < 300), {"status": r.status_code, "body": body, "url": url}

    def get_market_details(self, cst: str, xst: str, epic: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/v1/markets/{epic}"
        r = requests.get(url, headers=self._authed_headers(cst, xst), timeout=self.timeout)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return (200 <= r.status_code < 300), {"status": r.status_code, "body": body, "url": url}


def prices_to_df(prices_body: Dict[str, Any]) -> pd.DataFrame:
    # Capital format: body["prices"] list, each has snapshotTimeUTC, openPrice/closePrice/highPrice/lowPrice with bid/ask/lastTraded
    items = prices_body.get("prices") or []
    rows = []
    for p in items:
        t = p.get("snapshotTimeUTC") or p.get("snapshotTime") or None
        # choose mid price from bid/ask if available, else lastTraded
        def mid(x: Dict[str, Any]) -> float:
            if not isinstance(x, dict):
                return np.nan
            b = x.get("bid")
            a = x.get("ask")
            lt = x.get("lastTraded")
            if b is not None and a is not None:
                return (float(b) + float(a)) / 2.0
            if lt is not None:
                return float(lt)
            return np.nan

        o = mid(p.get("openPrice", {}))
        h = mid(p.get("highPrice", {}))
        l = mid(p.get("lowPrice", {}))
        c = mid(p.get("closePrice", {}))
        rows.append({"time": t, "open": o, "high": h, "low": l, "close": c})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------------
# App state
# ----------------------------
DEFAULT_WATCHLIST = {
    "US100": {"tv": "CAPITALCOM:US100"},
    "US500": {"tv": "CAPITALCOM:US500"},
    "GOLD (XAUUSD)": {"tv": "CAPITALCOM:GOLD"},
    "EURUSD": {"tv": "CAPITALCOM:EURUSD"},
    "DXY": {"tv": "CAPITALCOM:DXY"},
}

if "epic_map" not in st.session_state:
    st.session_state["epic_map"] = {k: "" for k in DEFAULT_WATCHLIST.keys()}

if "capital_tokens" not in st.session_state:
    st.session_state["capital_tokens"] = {"cst": "", "xst": "", "ts": 0.0, "ok": False}

if "last_login_error" not in st.session_state:
    st.session_state["last_login_error"] = None


# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="ufx-hero">
  <div class="ufx-hero-title">🚀 UnknownFX Dashboard</div>
  <div class="ufx-hero-sub">Institutional view • Capital.com candles • TradingView charts • MTF confluence • Auto refresh</div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ----------------------------
# Sidebar: Settings + Capital creds
# ----------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")

    auto_refresh = st.toggle("Auto refresh", value=True, help="Ververs automatisch (zonder extra packages).")
    refresh_minutes = st.slider("Refresh interval (min)", min_value=1, max_value=10, value=2)
    if auto_refresh:
        meta_refresh(refresh_minutes * 60)

    tv_interval = st.selectbox(
        "TradingView interval (chart)",
        options=["1", "5", "15", "60", "240", "D"],
        index=2,
        help="Alleen voor de chart weergave; TA komt uit Capital candles.",
    )

    st.divider()
    st.subheader("🔑 Capital.com (required)")

    # Read secrets safely
    def sget(key: str) -> str:
        try:
            return str(st.secrets.get(key, "")).strip()
        except Exception:
            return ""

    cap_api_key = sget("CAPITAL_API_KEY")
    cap_identifier = sget("CAPITAL_IDENTIFIER")
    cap_password = sget("CAPITAL_PASSWORD")
    cap_base = sget("CAPITAL_API_BASE") or "https://demo-api-capital.backend-capital.com"

    st.caption("Secrets keys: CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD, CAPITAL_API_BASE")

    missing = [k for k, v in {
        "CAPITAL_API_KEY": cap_api_key,
        "CAPITAL_IDENTIFIER": cap_identifier,
        "CAPITAL_PASSWORD": cap_password,
        "CAPITAL_API_BASE": cap_base,
    }.items() if not v]

    if missing:
        st.error("Missing secrets: " + ", ".join(missing))
        st.stop()

    client = CapitalClient(
        base_url=cap_base,
        api_key=cap_api_key,
        identifier=cap_identifier,
        password=cap_password,
    )

    # Login control: do NOT spam login every rerun (prevents 429)
    tokens = st.session_state["capital_tokens"]
    token_age = time.time() - float(tokens.get("ts", 0) or 0)
    token_ok = bool(tokens.get("cst")) and bool(tokens.get("xst")) and token_age < 20 * 60

    colA, colB = st.columns(2)
    with colA:
        do_login = st.button("Login / Refresh", use_container_width=True)
    with colB:
        do_logout = st.button("Clear session", use_container_width=True)

    if do_logout:
        st.session_state["capital_tokens"] = {"cst": "", "xst": "", "ts": 0.0, "ok": False}
        st.session_state["last_login_error"] = None
        st.success("Session cleared.")
        st.rerun()

    if do_login or (not token_ok and tokens.get("ok") is not True):
        ok, info = client.login()
        if ok:
            st.session_state["capital_tokens"] = {
                "cst": info["cst"],
                "xst": info["x_security_token"],
                "ts": time.time(),
                "ok": True,
            }
            st.session_state["last_login_error"] = None
            st.success("Capital login OK ✅")
        else:
            st.session_state["capital_tokens"]["ok"] = False
            st.session_state["last_login_error"] = info
            st.error(f"Capital login failed ({info.get('status')}): {info.get('body')}")
            st.stop()

    tokens = st.session_state["capital_tokens"]
    token_ok = bool(tokens.get("cst")) and bool(tokens.get("xst"))
    if token_ok:
        st.success("Capital session active ✅")
    else:
        st.error("No active Capital session. Click Login.")
        st.stop()

    st.divider()
    st.subheader("🧭 EPIC finder (1x instellen)")
    st.caption("Zoek markt → kies EPIC → klik 'Use this EPIC'. Daarna gebruikt het dashboard Capital-data.")

    market_to_set = st.selectbox("Market to set EPIC for", list(DEFAULT_WATCHLIST.keys()))
    search_term = st.text_input("Search term", value=market_to_set)

    if st.button("Search EPICs", use_container_width=True):
        ok, res = client.search_markets(tokens["cst"], tokens["xst"], search_term.strip())
        if not ok:
            st.error(f"Search failed ({res.get('status')}): {res.get('body')}")
        else:
            # Try to normalize
            data = res.get("body", {})
            markets = data.get("markets") or data.get("marketDetails") or []
            if not markets:
                st.warning("Geen resultaten.")
            else:
                # Build a compact table
                rows = []
                for m in markets[:30]:
                    instrument = m.get("instrumentName") or m.get("instrument") or ""
                    epic = m.get("epic") or ""
                    mtype = m.get("instrumentType") or m.get("type") or ""
                    expiry = m.get("expiry") or ""
                    rows.append({"instrument": instrument, "epic": epic, "type": mtype, "expiry": expiry})

                st.session_state["last_market_search"] = rows

    if st.button("Clear EPICs", use_container_width=True):
        st.session_state["epic_map"] = {k: "" for k in DEFAULT_WATCHLIST.keys()}
        st.success("EPICs cleared.")
        st.rerun()

    rows = st.session_state.get("last_market_search", [])
    if rows:
        st.markdown("**Results**")
        for i, r in enumerate(rows):
            cols = st.columns([4, 2, 1])
            cols[0].write(f"**{r['instrument']}**  \n_{r['type']}_")
            cols[1].code(r["epic"] or "-", language="text")
            if cols[2].button("Use", key=f"use_epic_{i}", use_container_width=True):
                if not r["epic"]:
                    st.warning("Geen EPIC in deze row.")
                else:
                    st.session_state["epic_map"][market_to_set] = r["epic"]
                    st.success(f"EPIC opgeslagen voor {market_to_set}: {r['epic']}")
                    st.rerun()

    st.divider()
    st.subheader("✅ Current EPIC map")
    st.json(st.session_state["epic_map"])


# ----------------------------
# Main: Dashboard
# ----------------------------
tokens = st.session_state["capital_tokens"]
epic_map = st.session_state["epic_map"]

# MTF settings
RES_MAP = {
    "5m": "MINUTE_5",
    "15m": "MINUTE_15",
    "1h": "HOUR",
    "4h": "HOUR_4",
    "1D": "DAY",
}
MTF_ORDER = ["5m", "15m", "1h", "4h", "1D"]

st.write("")
st.subheader("📊 Markets (1 per rij, big chart)")

for market_name, cfg in DEFAULT_WATCHLIST.items():
    epic = epic_map.get(market_name, "").strip()
    tv_symbol = cfg["tv"]

    st.markdown(f"<div class='ufx-card'>", unsafe_allow_html=True)
    top = st.columns([1.4, 1, 1, 1, 1.2])

    top[0].markdown(f"### {market_name}")
    top[0].markdown(f"<div class='ufx-pill'>TradingView: <b>{tv_symbol}</b></div>", unsafe_allow_html=True)

    if not epic:
        top[1].warning("EPIC ontbreekt. Stel ‘m in via sidebar (EPIC finder).")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        continue

    # Pull latest 15m candles for KPIs (stable)
    ok, prices_res = client.get_prices(tokens["cst"], tokens["xst"], epic=epic, resolution="MINUTE_15", points=200)
    if not ok:
        status = prices_res.get("status")
        body = prices_res.get("body")
        if status == 429:
            top[1].error("Rate limit (429). Wacht even / zet refresh hoger.")
        else:
            top[1].error(f"Capital prices error ({status}): {body}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        continue

    df = prices_to_df(prices_res["body"])
    if df.empty or df["close"].isna().all():
        top[1].error("Geen candle data ontvangen.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        continue

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    price = float(last["close"])
    change = float(price - float(prev["close"]))
    change_pct = (change / float(prev["close"])) * 100 if float(prev["close"]) else 0.0

    rsi14_val = float(last["rsi14"]) if not np.isnan(last["rsi14"]) else np.nan
    sma20_val = float(last["sma20"]) if not np.isnan(last["sma20"]) else np.nan
    sma50_val = float(last["sma50"]) if not np.isnan(last["sma50"]) else np.nan
    atrpct = (float(last["atr14"]) / price) * 100 if (not np.isnan(last["atr14"]) and price) else np.nan

    bias_15m = bias_from_indicators(price, sma50_val, rsi14_val)

    top[1].markdown(
        f"<div class='ufx-kpi'><div class='label'>Last price (15m)</div><div class='value'>{price:,.2f}</div></div>",
        unsafe_allow_html=True,
    )
    top[2].markdown(
        f"<div class='ufx-kpi'><div class='label'>Δ (prev candle)</div><div class='value'>{change:+,.2f} ({change_pct:+.2f}%)</div></div>",
        unsafe_allow_html=True,
    )
    top[3].markdown(
        f"<div class='ufx-kpi'><div class='label'>RSI(14)</div><div class='value'>{rsi14_val:,.1f}</div></div>",
        unsafe_allow_html=True,
    )
    top[4].markdown(
        f"<div class='ufx-kpi'><div class='label'>SMA20 / SMA50</div><div class='value'>{sma20_val:,.2f} / {sma50_val:,.2f}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='ufx-row' style='margin-top:10px;'>"
        f"{badge_html(bias_15m)}"
        f"<span class='ufx-pill'>EPIC: <b>{epic}</b></span>"
        f"<span class='ufx-pill'>Volatility (ATR%): <b>{atrpct:,.2f}%</b></span>"
        f"<span class='ufx-pill'>Last update: <b>{pd.to_datetime(last['time']).tz_convert('Europe/Amsterdam'):%Y-%m-%d %H:%M}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    # TradingView big chart (1 per row)
    interval = tv_interval
    tv_url = f"https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={interval}&hidetoptoolbar=0&hidesidetoolbar=0&symboledit=0&saveimage=1&toolbarbg=0F172A&studies=[]&theme=dark&style=1&timezone=Europe%2FAmsterdam&withdateranges=1&hideideas=1"
    st.components.v1.iframe(tv_url, height=580, scrolling=False)

    st.write("")

    # MTF Confluence (Capital candles)
    st.markdown("#### 🧠 MTF Confluence (Capital candles)")
    mtf_rows = []
    votes = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}

    for tf in MTF_ORDER:
        res = RES_MAP[tf]
        ok, pr = client.get_prices(tokens["cst"], tokens["xst"], epic=epic, resolution=res, points=220)
        if not ok:
            mtf_rows.append({"TF": tf, "Bias": "NEUTRAL", "Reason": f"API error {pr.get('status')}"})
            votes["NEUTRAL"] += 1
            continue
        d = prices_to_df(pr["body"])
        if d.empty:
            mtf_rows.append({"TF": tf, "Bias": "NEUTRAL", "Reason": "No data"})
            votes["NEUTRAL"] += 1
            continue
        d["sma50"] = d["close"].rolling(50).mean()
        d["rsi14"] = rsi(d["close"], 14)
        ll = d.iloc[-1]
        bb = bias_from_indicators(float(ll["close"]), float(ll["sma50"]) if not np.isnan(ll["sma50"]) else np.nan, float(ll["rsi14"]) if not np.isnan(ll["rsi14"]) else np.nan)
        votes[bb] += 1
        reason = f"close vs SMA50, RSI14"
        mtf_rows.append({"TF": tf, "Bias": bb, "Reason": reason})

    total = sum(votes.values()) or 1
    verdict = "NEUTRAL"
    if votes["BUY"] >= 3:
        verdict = "BUY"
    elif votes["SELL"] >= 3:
        verdict = "SELL"

    confidence = int(round((max(votes["BUY"], votes["SELL"]) / total) * 100, 0))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.markdown(f"<div class='ufx-kpi'><div class='label'>Votes BUY</div><div class='value'>{votes['BUY']}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='ufx-kpi'><div class='label'>Votes SELL</div><div class='value'>{votes['SELL']}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='ufx-kpi'><div class='label'>Confidence</div><div class='value'>{confidence}%</div></div>", unsafe_allow_html=True)
    c4.markdown(
        f"<div class='ufx-kpi'><div class='label'>Verdict</div><div class='value'>{verdict}</div></div>",
        unsafe_allow_html=True,
    )

    mtf_df = pd.DataFrame(mtf_rows)
    # Pretty bias column
    def pretty_bias(b: str) -> str:
        return "🟢 BUY" if b == "BUY" else ("🔴 SELL" if b == "SELL" else "🟠 NEUTRAL")

    mtf_df["Bias"] = mtf_df["Bias"].map(pretty_bias)
    st.dataframe(mtf_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='muted'>Rule of thumb: agressieve entries pas wanneer hogere TF’s dezelfde richting bevestigen.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")


st.caption("UnknownFX Dashboard • PRO+++ • Capital candles + TradingView • No stooq • No telegram")
