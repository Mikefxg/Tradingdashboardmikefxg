from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


# =========================
# Page + Theme (PRO+++)
# =========================
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRO_CSS = """
<style>
:root{
  --bg:#0B1220;
  --panel:#0F1A2B;
  --panel2:#0C1626;
  --text:#EAF0FF;
  --muted:#9FB0D0;
  --border:rgba(255,255,255,.08);
  --good:#22C55E;
  --bad:#EF4444;
  --warn:#F59E0B;
  --blue:#60A5FA;
  --chip:#111C2E;
}
html, body, [class*="css"]  { background: var(--bg) !important; color: var(--text) !important; }
section[data-testid="stSidebar"] { background: #07101D !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stToolbar"] { display:none; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px; }
h1,h2,h3 { letter-spacing: -0.02em; }
.small { color: var(--muted); font-size: 0.92rem; }
.hr { height:1px; background:var(--border); margin:14px 0 18px 0; }
.card{
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,.25);
}
.card2{
  background: rgba(255,255,255,.02);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
}
.badge{
  display:inline-flex; align-items:center; gap:10px;
  background: rgba(255,255,255,.03);
  border:1px solid var(--border);
  border-radius: 999px;
  padding: 8px 12px;
  font-weight: 700;
}
.dot{ width:10px; height:10px; border-radius:99px; display:inline-block; }
.dot.good{ background: var(--good); box-shadow: 0 0 16px rgba(34,197,94,.55); }
.dot.bad{ background: var(--bad); box-shadow: 0 0 16px rgba(239,68,68,.55); }
.dot.neu{ background: #94A3B8; box-shadow: 0 0 16px rgba(148,163,184,.35); }
.kpiGrid{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap:10px;
}
.kpi{
  background: rgba(255,255,255,.02);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
}
.kpi .label{ color: var(--muted); font-size: .82rem; }
.kpi .value{ font-size: 1.15rem; font-weight: 800; margin-top: 4px; }
.pill{
  display:inline-block;
  background: var(--chip);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: .82rem;
  color: var(--muted);
}
.bigTitle{
  font-size: 2.6rem;
  font-weight: 900;
  margin-bottom: 0.2rem;
}
.subTitle{
  color: var(--muted);
  margin-top: 0;
}
.marketRow{
  display:grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 14px;
  align-items: start;
}
@media (max-width: 1100px){
  .marketRow{ grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)


# =========================
# Client-side auto refresh
# (no extra dependency)
# =========================
def auto_refresh_every(ms: int):
    components.html(
        f"""
        <script>
          const ms = {ms};
          setTimeout(() => {{
            window.parent.location.reload();
          }}, ms);
        </script>
        """,
        height=0,
    )


# =========================
# TradingView embed (big)
# =========================
def tradingview_widget(symbol: str, interval: str = "15", height: int = 520):
    # interval: "1","5","15","60","240","D"
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tv_{symbol.replace(':','_')}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{interval}",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#0B1220",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": false,
          "container_id": "tv_{symbol.replace(':','_')}"
        }});
      </script>
    </div>
    """
    components.html(html, height=height + 12)


# =========================
# Capital.com API (IG-style)
# =========================
@dataclass
class CapitalSession:
    cst: str
    xst: str
    created_at: float


class CapitalClient:
    def __init__(self, base_url: str, api_key: str, identifier: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.identifier = identifier
        self.password = password

    def _headers(self, session: Optional[CapitalSession] = None) -> Dict[str, str]:
        h = {
            "X-IG-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
        }
        if session:
            h["CST"] = session.cst
            h["X-SECURITY-TOKEN"] = session.xst
        return h

    def login(self) -> CapitalSession:
        # Cache in session_state (valid long enough for our use)
        ss_key = "_capital_session"
        if ss_key in st.session_state:
            s: CapitalSession = st.session_state[ss_key]
            # refresh token every ~30min
            if time.time() - s.created_at < 30 * 60:
                return s

        url = f"{self.base_url}/session"
        payload = {"identifier": self.identifier, "password": self.password}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=25)

        if r.status_code >= 400:
            raise RuntimeError(f"Capital login failed ({r.status_code}): {r.text[:250]}")

        cst = r.headers.get("CST")
        xst = r.headers.get("X-SECURITY-TOKEN")
        if not cst or not xst:
            raise RuntimeError("Capital login succeeded but CST/X-SECURITY-TOKEN missing in headers.")

        sess = CapitalSession(cst=cst, xst=xst, created_at=time.time())
        st.session_state[ss_key] = sess
        return sess

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        sess = self.login()
        url = f"{self.base_url}{path}"
        r = requests.get(url, headers=self._headers(sess), params=params, timeout=25)
        if r.status_code >= 400:
            raise RuntimeError(f"Capital GET {path} failed ({r.status_code}): {r.text[:250]}")
        return r.json()

    def search_markets(self, term: str) -> List[dict]:
        # IG-style search endpoint
        data = self.get("/markets", params={"searchTerm": term})
        # Expected: {"markets":[...]}
        return data.get("markets", []) if isinstance(data, dict) else []

    def get_prices(self, epic: str, resolution: str, max_points: int = 250) -> pd.DataFrame:
        data = self.get(f"/prices/{epic}", params={"resolution": resolution, "max": max_points})
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()

        rows = []
        for p in prices:
            # time in UTC like "2026-02-15T12:30:00"
            t = p.get("snapshotTimeUTC") or p.get("snapshotTime")
            close = (p.get("closePrice") or {}).get("bid")
            high = (p.get("highPrice") or {}).get("bid")
            low = (p.get("lowPrice") or {}).get("bid")
            open_ = (p.get("openPrice") or {}).get("bid")
            if t and close is not None:
                rows.append((t, float(open_ or close), float(high or close), float(low or close), float(close)))

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        return df


# =========================
# Indicators (pure pandas)
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pct_change_last(close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

def format_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.5f}"


# =========================
# Outlook / Score Model
# (institutional-ish, simple)
# =========================
def outlook_from_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 60:
        return {
            "label": "NEUTRAL",
            "bias": "WAIT",
            "score": 0.0,
            "rsi": None,
            "ema20": None,
            "ema50": None,
            "atr_pct": None,
            "last": None,
            "delta_pct": None,
            "reason": "Not enough candle data.",
        }

    close = df["close"]
    last = float(close.iloc[-1])
    delta_pct = pct_change_last(close)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close, 14)
    a = atr(df, 14)
    atr_pct = float((a.iloc[-1] / last) * 100) if not np.isnan(a.iloc[-1]) else None

    # scoring
    score = 0.0
    # trend
    if e20.iloc[-1] > e50.iloc[-1]:
        score += 1.0
    else:
        score -= 1.0
    # momentum (price vs EMA20)
    if last > e20.iloc[-1]:
        score += 0.6
    else:
        score -= 0.6
    # RSI regime
    if r.iloc[-1] >= 55:
        score += 0.6
    elif r.iloc[-1] <= 45:
        score -= 0.6

    # volatility penalty (too wild = less confidence)
    if atr_pct is not None:
        if atr_pct > 2.0:
            score *= 0.85
        if atr_pct > 4.0:
            score *= 0.75

    # label
    if score >= 0.8:
        label = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.8:
        label = "BEARISH"
        bias = "SELL BIAS"
    else:
        label = "NEUTRAL"
        bias = "WAIT"

    reason = f"EMA20/EMA50={'UP' if e20.iloc[-1] > e50.iloc[-1] else 'DOWN'}, RSI={r.iloc[-1]:.1f}, Δ={delta_pct:+.2f}%"
    return {
        "label": label,
        "bias": bias,
        "score": float(score),
        "rsi": float(r.iloc[-1]),
        "ema20": float(e20.iloc[-1]),
        "ema50": float(e50.iloc[-1]),
        "atr_pct": atr_pct,
        "last": last,
        "delta_pct": delta_pct,
        "reason": reason,
    }

def mtf_votes(client: CapitalClient, epic: str) -> dict:
    # Resolution mapping: MINUTE_15, HOUR, HOUR_4, DAY
    frames = [
        ("15m", "MINUTE_15"),
        ("1h", "HOUR"),
        ("4h", "HOUR_4"),
        ("1D", "DAY"),
    ]
    votes = []
    details = []
    for name, res in frames:
        df = client.get_prices(epic, res, max_points=200)
        o = outlook_from_df(df)
        v = 0
        if o["label"] == "BULLISH":
            v = 1
        elif o["label"] == "BEARISH":
            v = -1
        votes.append(v)
        details.append((name, o["label"], o["score"]))

    total = sum(votes)
    if total >= 2:
        verdict = "BULLISH BIAS"
    elif total <= -2:
        verdict = "BEARISH BIAS"
    else:
        verdict = "MIXED / WAIT"

    return {"verdict": verdict, "details": details, "total": total}


# =========================
# Plotly sparkline (dark)
# =========================
def sparkline(df: pd.DataFrame, title: str = ""):
    if df is None or df.empty:
        st.caption("No data for sparkline.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["close"], mode="lines", name=title))
    fig.update_layout(
        margin=dict(l=0, r=0, t=18, b=0),
        height=120,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Markets config (TradingView symbols)
# NOTE: EPIC must be set via sidebar finder.
# =========================
DEFAULT_MARKETS = [
    {"key": "US100", "name": "US100 (Nasdaq)", "tv": "CAPITALCOM:US100", "search": "US 100"},
    {"key": "US500", "name": "US500 (S&P 500)", "tv": "CAPITALCOM:US500", "search": "US 500"},
    {"key": "XAUUSD", "name": "Gold (XAUUSD)", "tv": "CAPITALCOM:GOLD", "search": "Gold"},
    {"key": "EURUSD", "name": "EURUSD", "tv": "CAPITALCOM:EURUSD", "search": "EUR/USD"},
    {"key": "DXY", "name": "US Dollar Index (DXY)", "tv": "CAPITALCOM:DXY", "search": "Dollar Index"},
]


# =========================
# Sidebar: settings + EPIC finder
# =========================
st.markdown('<div class="bigTitle">🚀 UnknownFX Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subTitle">Institutional view • Capital.com data (candles) • TradingView charts • MTF confluence • Auto refresh</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    refresh_minutes = st.slider("Auto refresh (minutes)", 1, 10, 2)
    tv_interval = st.selectbox("TradingView interval", ["1", "5", "15", "60", "240", "D"], index=2)
    st.caption("Tip: 15m is nice for intraday bias. 60m/240m for cleaner trend.")

    st.markdown("---")
    st.markdown("### 🔑 Capital.com (required)")
    missing = []
    for k in ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD", "CAPITAL_API_BASE"]:
        if k not in st.secrets:
            missing.append(k)
    if missing:
        st.error("Missing secrets: " + ", ".join(missing))
        st.stop()

    client = CapitalClient(
        base_url=st.secrets["CAPITAL_API_BASE"],
        api_key=st.secrets["CAPITAL_API_KEY"],
        identifier=st.secrets["CAPITAL_IDENTIFIER"],
        password=st.secrets["CAPITAL_PASSWORD"],
    )

    st.markdown("---")
    st.markdown("### 🧭 EPIC finder (1x instellen)")
    st.caption("Capital gebruikt EPICs. Zoek je market → kies → klik **Use this EPIC**.")

    if "epic_map" not in st.session_state:
        st.session_state["epic_map"] = {}  # key -> epic

    chosen_market_key = st.selectbox("Market to set EPIC for", [m["key"] for m in DEFAULT_MARKETS], index=0)
    search_term_default = next(m["search"] for m in DEFAULT_MARKETS if m["key"] == chosen_market_key)
    search_term = st.text_input("Search term", value=search_term_default)

    colA, colB = st.columns([1, 1])
    with colA:
        do_search = st.button("Search EPICs", use_container_width=True)
    with colB:
        st.button("Clear EPICs", use_container_width=True, on_click=lambda: st.session_state["epic_map"].clear())

    if do_search:
        try:
            results = client.search_markets(search_term)
            st.session_state["_last_search_results"] = results
        except Exception as e:
            st.error(str(e))

    results = st.session_state.get("_last_search_results", [])
    if results:
        options = []
        for r in results[:30]:
            # r fields: epic, instrumentName, instrumentType, marketStatus, ...
            epic = r.get("epic", "")
            name = r.get("instrumentName", "")
            typ = r.get("instrumentType", "")
            options.append((f"{name} • {typ} • {epic}", epic))

        label = st.selectbox("Pick result", [o[0] for o in options])
        epic = dict(options)[label]
        st.code(epic, language="text")
        if st.button("Use this EPIC", use_container_width=True):
            st.session_state["epic_map"][chosen_market_key] = epic
            st.success(f"Saved EPIC for {chosen_market_key}")

    st.markdown("---")
    st.markdown("### ✅ Current EPIC map")
    if st.session_state["epic_map"]:
        st.json(st.session_state["epic_map"])
    else:
        st.info("Nog leeg. Stel EPICs in met de finder hierboven.")


# Auto refresh
auto_refresh_every(refresh_minutes * 60 * 1000)


# =========================
# Top: Global Risk Meter
# =========================
def risk_meter(scores: List[float]) -> Tuple[str, float]:
    if not scores:
        return "NEUTRAL", 0.0
    s = float(np.mean(scores))
    if s >= 0.6:
        return "RISK-ON", s
    if s <= -0.6:
        return "RISK-OFF", s
    return "NEUTRAL", s


# =========================
# Main render: 1 per row
# =========================
scores_for_global = []

for m in DEFAULT_MARKETS:
    key = m["key"]
    name = m["name"]
    tv_symbol = m["tv"]
    epic = st.session_state["epic_map"].get(key)

    st.markdown(f"## {name}")
    st.markdown('<div class="marketRow">', unsafe_allow_html=True)

    # LEFT: big chart
    left = st.container()
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tradingview_widget(tv_symbol, interval=tv_interval, height=560)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT: stats + outlook
    right = st.container()
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if not epic:
            st.warning(f"EPIC ontbreekt voor {key}. Ga naar sidebar → EPIC finder → stel hem 1x in.")
            st.markdown(f"<span class='pill'>TradingView: {tv_symbol}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            continue

        try:
            df_15m = client.get_prices(epic, "MINUTE_15", max_points=260)
            out = outlook_from_df(df_15m)
            mtf = mtf_votes(client, epic)
        except Exception as e:
            st.error(f"Capital data error for {key}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            continue

        label = out["label"]
        dot_class = "neu"
        if label == "BULLISH":
            dot_class = "good"
        elif label == "BEARISH":
            dot_class = "bad"

        scores_for_global.append(out["score"])

        st.markdown(
            f"""
            <div class="badge">
              <span class="dot {dot_class}"></span>
              <span>{label}</span>
              <span class="pill">{out["bias"]}</span>
              <span class="pill">score {out["score"]:+.2f}</span>
            </div>
            <div class="small" style="margin-top:10px;">{out["reason"]}</div>
            """,
            unsafe_allow_html=True,
        )

        # KPIs
        last = out["last"]
        delta_pct = out["delta_pct"]
        rsi_v = out["rsi"]
        atr_pct_v = out["atr_pct"]

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="kpiGrid">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="kpi">
              <div class="label">Last (Capital)</div>
              <div class="value">{format_price(last) if last is not None else "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi">
              <div class="label">Δ last candle</div>
              <div class="value">{delta_pct:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi">
              <div class="label">RSI(14)</div>
              <div class="value">{rsi_v:.1f if rsi_v is not None else "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # EMA + ATR line
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card2">
              <div class="small"><b>Trend / Volatility</b></div>
              <div class="small">EMA20: {format_price(out["ema20"]) if out["ema20"] else "—"} • EMA50: {format_price(out["ema50"]) if out["ema50"] else "—"}</div>
              <div class="small">ATR% (proxy): {atr_pct_v:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sparkline
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown("<div class='card2'><div class='small'><b>Intraday movement (15m)</b></div>", unsafe_allow_html=True)
        sparkline(df_15m, title=key)
        st.markdown("</div>", unsafe_allow_html=True)

        # MTF confluence
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card2">
              <div class="small"><b>MTF Confluence (15m → 1D)</b></div>
              <div style="margin-top:6px;">
                <span class="pill">{mtf["verdict"]}</span>
                <span class="pill">votes {mtf["total"]:+d}</span>
              </div>
              <div class="small" style="margin-top:10px;">
                {" • ".join([f"{t}:{lab}" for (t, lab, sc) in mtf["details"]])}
              </div>
              <div class="small" style="margin-top:8px; color: var(--muted);">
                Rule of thumb: agressief traden alleen als 4h & 1D dezelfde kant op wijzen.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<div class='small' style='margin-top:10px;'><span class='pill'>EPIC</span> {epic}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # marketRow
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# Global banner at bottom/top feel
risk_label, risk_score = risk_meter(scores_for_global)
st.markdown("### 🌍 Global Risk Regime")
dot = "neu"
if risk_label == "RISK-ON":
    dot = "good"
elif risk_label == "RISK-OFF":
    dot = "bad"

st.markdown(
    f"""
    <div class="card">
      <div class="badge">
        <span class="dot {dot}"></span>
        <span>{risk_label}</span>
        <span class="pill">avg score {risk_score:+.2f}</span>
        <span class="pill">refresh {int(refresh_minutes)}m</span>
      </div>
      <div class="small" style="margin-top:10px;">
        Dit is de gemiddelde bias van je watchlist. Gebruik dit als “context” voor entries, niet als single signal.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
import requests

def capital_login(base: str, api_key: str, identifier: str, password: str, timeout=15):
    """
    Returns: (ok: bool, data: dict)
    On success: data has 'cst' and 'x_security_token' + json body
    """
    base = base.rstrip("/")
    url = f"{base}/api/v1/session"

    headers = {
        "X-CAP-API-KEY": api_key,          # <- BELANGRIJK: exact deze header
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "identifier": identifier,
        "password": password,
        "encryptedPassword": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)

    # Capital tokens zitten vaak in headers
    cst = r.headers.get("CST")
    xst = r.headers.get("X-SECURITY-TOKEN")

    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}

    if r.status_code >= 200 and r.status_code < 300 and cst and xst:
        return True, {"status": r.status_code, "body": body, "cst": cst, "x_security_token": xst}
    else:
        return False, {
            "status": r.status_code,
            "body": body,
            "got_CST": bool(cst),
            "got_X_SECURITY_TOKEN": bool(xst),
            "resp_headers_keys": list(r.headers.keys()),
        
