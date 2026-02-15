from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# Page / Theme
# =========================================================
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
)

CSS = """
<style>
/* --- Global --- */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.92rem; }
.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }

/* --- Institutional cards --- */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}
.card-title {
  display:flex; align-items:center; justify-content:space-between;
  font-weight: 700; font-size: 1.05rem;
}
.badge {
  padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
}
.badge.green { background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.25); color: rgb(34,197,94); }
.badge.red   { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.25); color: rgb(239,68,68); }
.badge.gray  { background: rgba(148,163,184,0.12); border-color: rgba(148,163,184,0.25); color: rgb(226,232,240); }

.grid4 { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }
.kpi { border-radius: 14px; padding: 12px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.02); }
.kpi .lbl { font-size: 0.82rem; color: rgba(255,255,255,0.65); }
.kpi .val { font-size: 1.12rem; font-weight: 800; margin-top: 4px; }

.note { color: rgba(255,255,255,0.70); font-size: 0.95rem; }
.warn { padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(239,68,68,0.25);
        background: rgba(239,68,68,0.10); color: rgb(254,226,226); }
.ok { padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(34,197,94,0.25);
      background: rgba(34,197,94,0.10); color: rgb(220,252,231); }

/* --- Make TradingView iframe nicer --- */
.tv-wrap { border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# Helpers: indicators
# =========================================================
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig

def atr_percent(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return (atr / close) * 100

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =========================================================
# Capital.com API client (IG-like)
# =========================================================
@dataclass
class CapitalSession:
    cst: str
    xst: str

class CapitalClient:
    def __init__(self, base_url: str, api_key: str, identifier: str, password: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.timeout = timeout

    def _headers_base(self) -> Dict[str, str]:
        return {
            "X-CAP-API-KEY": self.api_key,  # important
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def login(self) -> Tuple[bool, Dict]:
        """
        Returns: (ok, payload)
        ok=True => payload has {"cst":..., "xst":...}
        ok=False => payload has debug info
        """
        url = f"{self.base_url}/api/v1/session"
        payload = {"identifier": self.identifier, "password": self.password, "encryptedPassword": False}
        try:
            r = requests.post(url, headers=self._headers_base(), json=payload, timeout=self.timeout)
        except Exception as e:
            return False, {"error": "request_failed", "exception": str(e), "url": url}

        cst = r.headers.get("CST")
        xst = r.headers.get("X-SECURITY-TOKEN")
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        if 200 <= r.status_code < 300 and cst and xst:
            return True, {"cst": cst, "xst": xst, "body": body, "status": r.status_code}
        return False, {
            "status": r.status_code,
            "body": body,
            "got_CST": bool(cst),
            "got_X_SECURITY_TOKEN": bool(xst),
            "resp_header_keys": list(r.headers.keys()),
            "url": url,
        }

    def _auth_headers(self, sess: CapitalSession) -> Dict[str, str]:
        h = self._headers_base()
        h["CST"] = sess.cst
        h["X-SECURITY-TOKEN"] = sess.xst
        return h

    def search_markets(self, sess: CapitalSession, term: str) -> Tuple[bool, Dict]:
        url = f"{self.base_url}/api/v1/markets"
        params = {"searchTerm": term}
        try:
            r = requests.get(url, headers=self._auth_headers(sess), params=params, timeout=self.timeout)
            return (200 <= r.status_code < 300), {"status": r.status_code, "body": r.json() if r.text else {}}
        except Exception as e:
            return False, {"error": "request_failed", "exception": str(e), "url": url}

    def get_prices(self, sess: CapitalSession, epic: str, resolution: str, max_points: int = 250) -> Tuple[bool, Dict]:
        # IG-style endpoint
        url = f"{self.base_url}/api/v1/prices/{epic}"
        params = {"resolution": resolution, "max": max_points}
        try:
            r = requests.get(url, headers=self._auth_headers(sess), params=params, timeout=self.timeout)
            body = r.json() if r.text else {}
            return (200 <= r.status_code < 300), {"status": r.status_code, "body": body}
        except Exception as e:
            return False, {"error": "request_failed", "exception": str(e), "url": url}

# =========================================================
# TradingView embeds (NO API scraping, so no 429)
# =========================================================
def tradingview_chart_embed(symbol: str, interval: str, height: int = 560, theme: str = "dark") -> str:
    # symbol e.g. "CAPITALCOM:US100"
    # interval: "1", "5", "15", "60", "240", "D"
    # Note: TradingView widget runs client-side.
    cfg = {
        "autosize": True,
        "symbol": symbol,
        "interval": interval,
        "timezone": "Etc/UTC",
        "theme": theme,
        "style": "1",
        "locale": "en",
        "enable_publishing": False,
        "allow_symbol_change": False,
        "hide_side_toolbar": False,
        "withdateranges": True,
        "details": True,
        "hotlist": False,
        "calendar": False,
        "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies", "Moving Average@tv-basicstudies"],
        "support_host": "https://www.tradingview.com",
        "height": height,
    }
    # Embed uses JS; Streamlit needs raw HTML.
    return f"""
<div class="tv-wrap">
  <div class="tradingview-widget-container">
    <div id="tv_{symbol.replace(':','_')}_{interval}"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({cfg});
    </script>
  </div>
</div>
""".replace("{cfg}", str(cfg).replace("'", '"'))

def tradingview_ta_embed(symbol: str, interval: str, theme: str = "dark") -> str:
    # TradingView technical analysis widget (client-side)
    cfg = {
        "interval": interval,
        "width": "100%",
        "isTransparent": True,
        "height": 260,
        "symbol": symbol,
        "showIntervalTabs": False,
        "locale": "en",
        "colorTheme": theme,
    }
    return f"""
<div class="tv-wrap">
  <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {str(cfg).replace("'", '"')}
    </script>
  </div>
</div>
"""

# =========================================================
# Data transforms: Capital prices -> OHLC dataframe
# =========================================================
def prices_to_ohlc(prices_body: Dict) -> Optional[pd.DataFrame]:
    """
    Tries to normalize Capital/IG-like price response to df with columns:
    time, open, high, low, close
    """
    if not isinstance(prices_body, dict):
        return None

    items = prices_body.get("prices") or prices_body.get("Prices") or prices_body.get("data") or []
    if not isinstance(items, list) or len(items) < 30:
        return None

    rows = []
    for it in items:
        # IG format often has: it["snapshotTimeUTC"] and it["openPrice"]["bid"/"ask"/"lastTraded"] etc.
        ts = it.get("snapshotTimeUTC") or it.get("snapshotTime") or it.get("time") or it.get("timestamp")
        op = it.get("openPrice") or {}
        hp = it.get("highPrice") or {}
        lp = it.get("lowPrice") or {}
        cp = it.get("closePrice") or {}

        def pick(px: Dict) -> Optional[float]:
            if not isinstance(px, dict):
                return None
            # prefer "lastTraded" then "mid" then average bid/ask
            if px.get("lastTraded") is not None:
                return float(px["lastTraded"])
            if px.get("mid") is not None:
                return float(px["mid"])
            b = px.get("bid")
            a = px.get("ask")
            if b is not None and a is not None:
                return float((float(b) + float(a)) / 2.0)
            if b is not None:
                return float(b)
            if a is not None:
                return float(a)
            return None

        o = pick(op)
        h = pick(hp)
        l = pick(lp)
        c = pick(cp)

        if ts is None or any(v is None for v in [o, h, l, c]):
            continue

        rows.append((ts, o, h, l, c))

    if len(rows) < 30:
        return None

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    # best-effort datetime parse
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df if len(df) >= 30 else None

# =========================================================
# Scoring / bias model (simple, stable, explainable)
# =========================================================
def compute_bias(df: pd.DataFrame) -> Dict[str, float | str]:
    close = df["close"]
    out: Dict[str, float | str] = {}

    df = df.copy()
    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["rsi14"] = rsi(close, 14)
    df["macd_hist"] = macd_hist(close)
    df["atrp"] = atr_percent(df, 14)

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_sma20 = float(last["sma20"]) if not math.isnan(last["sma20"]) else np.nan
    last_sma50 = float(last["sma50"]) if not math.isnan(last["sma50"]) else np.nan
    last_rsi = float(last["rsi14"]) if not math.isnan(last["rsi14"]) else np.nan
    last_macd_h = float(last["macd_hist"]) if not math.isnan(last["macd_hist"]) else np.nan
    last_atrp = float(last["atrp"]) if not math.isnan(last["atrp"]) else np.nan

    # Score components
    score = 0.0

    # Trend (SMA20 vs SMA50)
    if not np.isnan(last_sma20) and not np.isnan(last_sma50):
        score += 0.8 if last_sma20 > last_sma50 else -0.8

    # Price vs SMA50
    if not np.isnan(last_sma50):
        score += 0.6 if last_close > last_sma50 else -0.6

    # Momentum (RSI)
    if not np.isnan(last_rsi):
        if last_rsi >= 55:
            score += 0.6
        elif last_rsi <= 45:
            score -= 0.6

    # MACD histogram
    if not np.isnan(last_macd_h):
        score += 0.5 if last_macd_h > 0 else -0.5

    # Confidence: higher when |score| bigger, slightly adjusted by volatility
    conf = clamp((abs(score) / 2.5) * 100.0, 0, 100)
    if not np.isnan(last_atrp):
        # if extreme ATR%, reduce confidence a bit
        conf *= clamp(1.0 - (last_atrp / 20.0), 0.55, 1.0)

    if score >= 0.8:
        sentiment = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.8:
        sentiment = "BEARISH"
        bias = "SELL BIAS"
    else:
        sentiment = "NEUTRAL"
        bias = "WAIT / NEUTRAL"

    out["last"] = last_close
    out["sma20"] = last_sma20
    out["sma50"] = last_sma50
    out["rsi14"] = last_rsi
    out["macd_hist"] = last_macd_h
    out["atrp"] = last_atrp
    out["score"] = float(score)
    out["confidence"] = float(conf)
    out["sentiment"] = sentiment
    out["bias"] = bias
    return out

def mtf_confluence(trends: Dict[str, str]) -> str:
    # Count up/down across TFs
    up = sum(1 for v in trends.values() if v == "UP")
    dn = sum(1 for v in trends.values() if v == "DOWN")
    if up >= 3:
        return "BULLISH (3+ TF aligned)"
    if dn >= 3:
        return "BEARISH (3+ TF aligned)"
    return "MIXED"

def trend_from_df(df: pd.DataFrame) -> str:
    df = df.copy()
    df["sma50"] = sma(df["close"], 50)
    last = df.iloc[-1]
    if math.isnan(last["sma50"]):
        return "MIXED"
    return "UP" if float(last["close"]) > float(last["sma50"]) else "DOWN"

# =========================================================
# Sidebar settings
# =========================================================
st.markdown("## 🚀 UnknownFX Dashboard")
st.markdown('<div class="small-muted">Institutional view • Capital.com candles • TradingView charts • MTF confluence • Auto refresh</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    refresh_minutes = st.slider("Auto refresh (minutes)", min_value=1, max_value=10, value=2, step=1)
    tv_interval_label = st.selectbox("TradingView interval", ["1m", "5m", "15m", "1h", "4h", "1D"], index=2)

    tv_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "D"}
    tv_interval = tv_map[tv_interval_label]

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### 🔑 Capital.com (required)")

    # Read secrets
    def sec(name: str) -> str:
        try:
            return str(st.secrets.get(name, "")).strip()
        except Exception:
            return ""

    CAPITAL_API_KEY = sec("CAPITAL_API_KEY")
    CAPITAL_IDENTIFIER = sec("CAPITAL_IDENTIFIER")
    CAPITAL_PASSWORD = sec("CAPITAL_PASSWORD")
    CAPITAL_API_BASE = sec("CAPITAL_API_BASE") or "https://demo-api-capital.backend-capital.com"

    missing = [k for k, v in [
        ("CAPITAL_API_KEY", CAPITAL_API_KEY),
        ("CAPITAL_IDENTIFIER", CAPITAL_IDENTIFIER),
        ("CAPITAL_PASSWORD", CAPITAL_PASSWORD),
        ("CAPITAL_API_BASE", CAPITAL_API_BASE),
    ] if not v]

    if missing:
        st.error("Missing secrets: " + ", ".join(missing))
    else:
        st.success("Capital secrets loaded ✅")

    st.markdown('<div class="small-muted">Tip: demo base is usually <code>https://demo-api-capital.backend-capital.com</code></div>', unsafe_allow_html=True)

# Auto refresh
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

# =========================================================
# App state: EPIC mapping (user-controlled)
# =========================================================
DEFAULT_MARKETS = [
    # (display_name, TradingView symbol, default search term)
    ("US100", "CAPITALCOM:US100", "US100"),
    ("US500", "CAPITALCOM:US500", "US500"),
    ("GOLD (XAUUSD)", "CAPITALCOM:GOLD", "GOLD"),
    ("EURUSD", "CAPITALCOM:EURUSD", "EURUSD"),
    ("DXY", "CAPITALCOM:DXY", "DXY"),
]

if "epic_map" not in st.session_state:
    # user can override via EPIC finder
    st.session_state.epic_map = {m[0]: "" for m in DEFAULT_MARKETS}

# =========================================================
# Capital login (once per refresh) + EPIC Finder
# =========================================================
capital_ok = False
capital_client: Optional[CapitalClient] = None
capital_sess: Optional[CapitalSession] = None
capital_debug: Optional[Dict] = None

if not missing:
    capital_client = CapitalClient(
        base_url=CAPITAL_API_BASE,
        api_key=CAPITAL_API_KEY,
        identifier=CAPITAL_IDENTIFIER,
        password=CAPITAL_PASSWORD,
    )
    ok, payload = capital_client.login()
    capital_ok = ok
    if ok:
        capital_sess = CapitalSession(cst=payload["cst"], xst=payload["xst"])
    else:
        capital_debug = payload

with st.sidebar:
    st.markdown("### 🧭 EPIC finder (1x instellen)")
    st.markdown('<div class="small-muted">Capital gebruikt EPICs. Zoek je market → kies EPIC → klik “Use this EPIC”.</div>', unsafe_allow_html=True)

    market_sel = st.selectbox("Market to set EPIC for", [m[0] for m in DEFAULT_MARKETS], index=0)
    term_default = dict((m[0], m[2]) for m in DEFAULT_MARKETS).get(market_sel, market_sel)
    search_term = st.text_input("Search term", value=term_default)

    colA, colB = st.columns(2)
    do_search = colA.button("Search EPICs", use_container_width=True)
    do_clear = colB.button("Clear EPICs", use_container_width=True)

    if do_clear:
        st.session_state.epic_map = {m[0]: "" for m in DEFAULT_MARKETS}
        st.success("EPIC map cleared.")

    if not capital_ok:
        st.error(f"Capital login failed: {capital_debug}")
    else:
        st.success("Capital login OK ✅")

    results = []
    if do_search and capital_ok and capital_client and capital_sess:
        ok, res = capital_client.search_markets(capital_sess, search_term.strip())
        if not ok:
            st.error(f"Search failed: {res}")
        else:
            body = res.get("body", {})
            # IG-like: markets may be in "markets"
            mkts = body.get("markets") or body.get("Markets") or []
            if isinstance(mkts, list):
                for it in mkts[:20]:
                    epic = it.get("epic") or it.get("EPIC")
                    name = it.get("instrumentName") or it.get("instrument") or it.get("name") or ""
                    if epic:
                        results.append((epic, name))
            if not results:
                st.warning("No results. Try another term (e.g. 'NASDAQ', 'US 100', 'Gold', 'EUR/USD').")

    if results:
        pick = st.selectbox("Pick EPIC", [f"{e} — {n}" for e, n in results])
        if st.button("Use this EPIC", use_container_width=True):
            epic = pick.split(" — ")[0].strip()
            st.session_state.epic_map[market_sel] = epic
            st.success(f"Saved EPIC for {market_sel}: {epic}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### ✅ Current EPIC map")
    st.json(st.session_state.epic_map)

# =========================================================
# Main dashboard rendering
# =========================================================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def badge_html(text: str) -> str:
    t = text.upper()
    if "BULL" in t or "BUY" in t:
        cls = "green"
    elif "BEAR" in t or "SELL" in t:
        cls = "red"
    else:
        cls = "gray"
    return f'<span class="badge {cls}">{text}</span>'

# Global risk index (simple average score)
scores = []
global_status = "NEUTRAL"
global_conf = 0.0

# One market per row
for display_name, tv_symbol, _ in DEFAULT_MARKETS:
    left, right = st.columns([1.25, 1.0], gap="large")

    # -------- LEFT: TradingView chart (big) --------
    with left:
        st.markdown(f"### {display_name}")
        st.components.v1.html(tradingview_chart_embed(tv_symbol, tv_interval, height=580), height=600, scrolling=False)

    # -------- RIGHT: Institutional card (Capital indicators + MTF + TA widget) --------
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        epic = st.session_state.epic_map.get(display_name, "").strip()
        top_line = f"""
        <div class="card-title">
          <div>Outlook • {display_name}</div>
          <div>{badge_html("EPIC OK" if epic else "EPIC MISSING")}</div>
        </div>
        <div class="small-muted">TradingView: <code>{tv_symbol}</code> • Capital EPIC: <code>{epic or "—"}</code></div>
        """
        st.markdown(top_line, unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        if not (capital_ok and capital_client and capital_sess):
            st.markdown('<div class="warn">Capital API not ready. Check sidebar “Capital login failed” + Secrets.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        if not epic:
            st.markdown('<div class="warn">EPIC ontbreekt. Gebruik links de <b>EPIC finder</b> → zoek → “Use this EPIC”.</div>', unsafe_allow_html=True)
            # Still show TradingView TA widget
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.components.v1.html(tradingview_ta_embed(tv_symbol, tv_interval), height=280, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        # Fetch 15m candles for KPIs
        okp, resp = capital_client.get_prices(capital_sess, epic=epic, resolution="MINUTE_15", max_points=260)
        if not okp:
            st.markdown(f'<div class="warn">Capital prices fetch failed: <code>{resp}</code></div>', unsafe_allow_html=True)
            st.components.v1.html(tradingview_ta_embed(tv_symbol, tv_interval), height=280, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        df15 = prices_to_ohlc(resp.get("body", {}))
        if df15 is None:
            st.markdown('<div class="warn">Geen bruikbare candle data terug. Check of EPIC klopt (soms meerdere varianten).</div>', unsafe_allow_html=True)
            st.components.v1.html(tradingview_ta_embed(tv_symbol, tv_interval), height=280, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        k = compute_bias(df15)

        # Save for global risk
        scores.append(float(k["score"]))

        # KPI grid
        sentiment = str(k["sentiment"])
        bias = str(k["bias"])
        score = float(k["score"])
        conf = float(k["confidence"])
        last = float(k["last"])

        st.markdown(
            f"""
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
              {badge_html(sentiment)}
              {badge_html(bias)}
              <span class="badge">score {score:+.2f}</span>
              <span class="badge">confidence {conf:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="grid4">
              <div class="kpi"><div class="lbl">Last</div><div class="val">{last:,.2f}</div></div>
              <div class="kpi"><div class="lbl">RSI(14)</div><div class="val">{float(k["rsi14"]):.1f}</div></div>
              <div class="kpi"><div class="lbl">SMA20 / SMA50</div><div class="val">{float(k["sma20"]):,.2f} / {float(k["sma50"]):,.2f}</div></div>
              <div class="kpi"><div class="lbl">Volatility (ATR%)</div><div class="val">{float(k["atrp"]):.2f}%</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # MTF confluence (Capital candles)
        tf_map = {
            "5m": "MINUTE_5",
            "15m": "MINUTE_15",
            "1h": "HOUR",
            "4h": "HOUR_4",
            "1D": "DAY",
        }
        trends: Dict[str, str] = {}
        for label, reso in tf_map.items():
            okx, rx = capital_client.get_prices(capital_sess, epic=epic, resolution=reso, max_points=220)
            dfx = prices_to_ohlc(rx.get("body", {})) if okx else None
            trends[label] = trend_from_df(dfx) if dfx is not None else "MIXED"

        verdict = mtf_confluence(trends)
        st.markdown("#### 🧠 MTF Confluence")
        st.markdown(
            f"""
            <div class="note">
              5m: <b>{trends['5m']}</b> • 15m: <b>{trends['15m']}</b> • 1h: <b>{trends['1h']}</b> • 4h: <b>{trends['4h']}</b> • 1D: <b>{trends['1D']}</b><br/>
              Verdict: <b>{verdict}</b><br/>
              <span class="small-muted">Rule of thumb: only take aggressive entries when Daily & Weekly agree (or 3+ TF aligned).</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # TradingView TA widget (secondary confirmation)
        st.markdown("#### 📌 TradingView Technical Analysis")
        st.components.v1.html(tradingview_ta_embed(tv_symbol, tv_interval), height=280, scrolling=False)

        st.markdown("</div>", unsafe_allow_html=True)

# Global risk
if scores:
    avg = float(np.mean(scores))
    if avg >= 0.4:
        global_status = "BULLISH"
    elif avg <= -0.4:
        global_status = "BEARISH"
    else:
        global_status = "NEUTRAL"
    global_conf = clamp((abs(avg) / 2.5) * 100.0, 0, 100)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown("## 🌍 Global Risk Snapshot")
st.markdown(
    f"""
    <div class="card">
      <div class="card-title">
        <div>Global Risk (avg of model scores)</div>
        <div>{badge_html(global_status)} <span class="badge">avg score {avg:+.2f}</span> <span class="badge">confidence {global_conf:.0f}%</span></div>
      </div>
      <div class="small-muted">This is a directional dashboard signal — not financial advice. Always confirm with structure + risk management.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
