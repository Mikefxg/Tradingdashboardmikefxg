import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "🚀 UnknownFX Dashboard"
APP_SUBTITLE = "Institutional view • Capital.com candles • TradingView charts • MTF confluence • Auto refresh"


# ----------------------------
# Styling (institutional look)
# ----------------------------
st.markdown(
    """
<style>
/* Layout polish */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
.small-muted { opacity: 0.78; font-size: 0.95rem; }

/* Card look */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 16px 16px;
}
.card-title { font-size: 0.95rem; opacity: 0.85; margin-bottom: 6px; }
.card-big { font-size: 1.8rem; font-weight: 700; }
.card-sub { font-size: 0.95rem; opacity: 0.8; }

/* Badges */
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  font-weight: 700; font-size: 0.85rem; letter-spacing: 0.3px;
  border: 1px solid rgba(255,255,255,0.10);
}
.badge-bull { background: rgba(34,197,94,0.15); color: rgb(134,239,172); }
.badge-bear { background: rgba(239,68,68,0.15); color: rgb(252,165,165); }
.badge-neutral { background: rgba(234,179,8,0.15); color: rgb(253,230,138); }

/* Divider */
.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }

/* TradingView wrapper */
.tv-wrap {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(0,0,0,0.10);
  border-radius: 16px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Auto refresh without extra packages (no streamlit_autorefresh)
# ----------------------------
def inject_auto_refresh(enabled: bool, seconds: int) -> None:
    if enabled and seconds >= 10:
        st.markdown(
            f"""<meta http-equiv="refresh" content="{int(seconds)}">""",
            unsafe_allow_html=True,
        )


# ----------------------------
# Market config
# ----------------------------
@dataclass
class Market:
    key: str
    label: str
    tv_symbol: str  # TradingView symbol
    default_epic_hint: str  # shown to user
    asset_class: str


MARKETS: List[Market] = [
    Market("US100", "US 100 (Nasdaq)", "CAPITALCOM:US100", "Zoek: US 100 / US100", "Index"),
    Market("US500", "US 500 (S&P 500)", "CAPITALCOM:US500", "Zoek: US 500 / US500", "Index"),
    Market("GOLD", "Gold (XAUUSD)", "CAPITALCOM:GOLD", "Zoek: Gold / XAUUSD", "Commodity"),
    Market("DXY", "US Dollar Index (DXY)", "CAPITALCOM:DXY", "Zoek: DXY / Dollar Index", "Index"),
    Market("EURUSD", "EURUSD", "CAPITALCOM:EURUSD", "Zoek: EURUSD", "FX"),
]

TV_INTERVALS = {
    "1": "1",
    "5": "5",
    "15": "15",
    "60": "60",
    "240": "240",
    "1D": "D",
}

CAPITAL_RESOLUTION_MAP = {
    "1": "MINUTE",
    "5": "MINUTE_5",
    "15": "MINUTE_15",
    "60": "HOUR",
    "240": "HOUR_4",
    "1D": "DAY",
}


# ----------------------------
# Secrets helpers
# ----------------------------
def get_secret(name: str) -> Optional[str]:
    # Works on Streamlit Cloud + local (if you provide .streamlit/secrets.toml)
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def required_capital_secrets_present() -> Tuple[bool, List[str]]:
    missing = []
    for k in ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD", "CAPITAL_API_BASE"]:
        if not get_secret(k):
            missing.append(k)
    return (len(missing) == 0), missing


# ----------------------------
# Capital.com API
# ----------------------------
def capital_login(force: bool = False) -> Tuple[bool, str]:
    """
    Logs in once and stores CST + X-SECURITY-TOKEN in st.session_state.
    Avoids repeated login to prevent 429.
    """
    ok, missing = required_capital_secrets_present()
    if not ok:
        return False, f"Missing secrets: {', '.join(missing)}"

    # Reuse existing token (55 minutes)
    token = st.session_state.get("capital_token", {})
    age = time.time() - float(token.get("ts", 0) or 0)
    token_ok = bool(token.get("cst")) and bool(token.get("xst")) and age < 55 * 60

    if token_ok and not force:
        return True, "Already logged in (cached token)."

    base = get_secret("CAPITAL_API_BASE").rstrip("/")
    url = f"{base}/api/v1/session"

    headers = {
        "X-CAP-API-KEY": get_secret("CAPITAL_API_KEY"),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "identifier": get_secret("CAPITAL_IDENTIFIER"),
        "password": get_secret("CAPITAL_PASSWORD"),
        "encryptedPassword": False,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
    except Exception as e:
        return False, f"Network error during login: {e}"

    # Rate limit / auth handling
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}

    cst = r.headers.get("CST")
    xst = r.headers.get("X-SECURITY-TOKEN")

    if 200 <= r.status_code < 300 and cst and xst:
        st.session_state["capital_token"] = {"cst": cst, "xst": xst, "ts": time.time(), "base": base}
        return True, "Capital login OK ✅"

    # Common errors: 401, 400, 429
    if r.status_code == 429:
        return False, "Capital login blocked (429 too many requests). Wacht 1–2 minuten en klik opnieuw."
    return False, f"Capital login failed ({r.status_code}): {body}"


def capital_headers() -> Tuple[bool, Dict[str, str], str]:
    token = st.session_state.get("capital_token", {})
    if not token.get("cst") or not token.get("xst"):
        return False, {}, "No active Capital session. Click 'Login / Refresh'."

    headers = {
        "X-CAP-API-KEY": get_secret("CAPITAL_API_KEY"),
        "CST": token["cst"],
        "X-SECURITY-TOKEN": token["xst"],
        "Accept": "application/json",
    }
    return True, headers, "OK"


@st.cache_data(ttl=25)  # light caching to reduce calls
def capital_search_markets(base: str, headers: Dict[str, str], term: str) -> Dict:
    url = f"{base}/api/v1/markets"
    r = requests.get(url, headers=headers, params={"searchTerm": term}, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=20)
def capital_get_prices(base: str, headers: Dict[str, str], epic: str, resolution: str, max_points: int = 200) -> pd.DataFrame:
    """
    Gets candles from Capital. Returns DataFrame with datetime index and OHLC.
    """
    url = f"{base}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": max_points}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    r.raise_for_status()
    j = r.json()

    prices = j.get("prices", [])
    if not prices:
        return pd.DataFrame()

    rows = []
    for p in prices:
        snap = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        o = p.get("openPrice", {})
        h = p.get("highPrice", {})
        l = p.get("lowPrice", {})
        c = p.get("closePrice", {})

        def mid(x: dict) -> Optional[float]:
            # Capital sometimes gives bid/ask
            if "mid" in x and x["mid"] is not None:
                return float(x["mid"])
            if "bid" in x and "ask" in x and x["bid"] is not None and x["ask"] is not None:
                return (float(x["bid"]) + float(x["ask"])) / 2.0
            return None

        row = {
            "time": snap,
            "open": mid(o),
            "high": mid(h),
            "low": mid(l),
            "close": mid(c),
        }
        if row["close"] is not None and row["open"] is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    df = df.set_index("time")
    return df


# ----------------------------
# Indicators
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bias_from_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns RSI, SMA20, SMA50, ATR% and a simple bias score.
    """
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    r = rsi(close, 14)
    a = atr(df, 14)
    atr_pct = (a / close) * 100.0

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_sma20 = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else np.nan
    last_sma50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else np.nan
    last_rsi = float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else np.nan
    last_atr_pct = float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else np.nan

    score = 0
    if not np.isnan(last_sma20) and last_close > last_sma20:
        score += 1
    if not np.isnan(last_sma50) and last_close > last_sma50:
        score += 1
    if not np.isnan(last_rsi) and last_rsi >= 55:
        score += 1
    if not np.isnan(last_rsi) and last_rsi <= 45:
        score -= 1
    if not np.isnan(last_sma20) and not np.isnan(last_sma50) and last_sma20 < last_sma50:
        score -= 1

    return {
        "close": last_close,
        "rsi": last_rsi,
        "sma20": last_sma20,
        "sma50": last_sma50,
        "atr_pct": last_atr_pct,
        "score": float(score),
    }


def score_to_label(score: float) -> Tuple[str, str]:
    if score >= 2:
        return "BULLISH", "badge badge-bull"
    if score <= -2:
        return "BEARISH", "badge badge-bear"
    return "NEUTRAL", "badge badge-neutral"


def mtf_verdict(scores: Dict[str, float]) -> Tuple[str, str]:
    """
    scores keys: "15", "60", "240", "1D" -> score
    """
    bull = sum(1 for s in scores.values() if s >= 2)
    bear = sum(1 for s in scores.values() if s <= -2)
    if bull >= 3:
        return "MTF: BULLISH BIAS", "badge badge-bull"
    if bear >= 3:
        return "MTF: BEARISH BIAS", "badge badge-bear"
    return "MTF: MIXED / WAIT", "badge badge-neutral"


# ----------------------------
# TradingView embed
# ----------------------------
def tradingview_advanced_chart(symbol: str, interval: str, height: int = 620) -> str:
    # interval must be "1", "5", "15", "60", "240" or "D"
    interval_tv = TV_INTERVALS.get(interval, "15")
    return f"""
<div class="tv-wrap">
  <div class="tradingview-widget-container">
    <div id="tv_{symbol.replace(':','_')}"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "{interval_tv}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "withdateranges": true,
        "allow_symbol_change": false,
        "container_id": "tv_{symbol.replace(':','_')}",
        "height": {height}
      }});
    </script>
  </div>
</div>
"""


# ----------------------------
# Sidebar (settings & capital)
# ----------------------------
st.markdown(f"# {APP_TITLE}")
st.markdown(f"<div class='small-muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    auto = st.toggle("Auto refresh", value=True)
    refresh_min = st.slider("Refresh interval (min)", 1, 30, 2)
    inject_auto_refresh(auto, int(refresh_min * 60))

    tv_interval = st.selectbox("TradingView interval (chart)", options=["1", "5", "15", "60", "240", "1D"], index=2)

    st.markdown("---")
    st.markdown("### 🔑 Capital.com (required voor candles & indicators)")

    ok, missing = required_capital_secrets_present()
    if not ok:
        st.error("Missing secrets:\n\n- " + "\n- ".join(missing))
        st.info("Ga naar Streamlit Cloud → App → Settings → Secrets en zet ze in TOML formaat.")
    else:
        colA, colB = st.columns(2)
        with colA:
            if st.button("Login / Refresh", use_container_width=True):
                success, msg = capital_login(force=True)
                (st.success(msg) if success else st.error(msg))
        with colB:
            if st.button("Clear session", use_container_width=True):
                st.session_state.pop("capital_token", None)
                st.success("Session cleared.")

        # Attempt a safe reuse (no force)
        success, msg = capital_login(force=False)
        if success:
            st.success("Capital login OK ✅")
        else:
            st.warning(msg)

    st.markdown("---")
    st.markdown("### 🧭 EPIC map (1x instellen)")
    st.caption("Plak jouw EPICs hier (van Capital). Je zei dat je ze al hebt gevonden via 'Use this EPIC'.")

    # --- EPIC storage persistent ---
import json, os

EPIC_FILE = "epics.json"

def load_epics():
    if os.path.exists(EPIC_FILE):
        with open(EPIC_FILE, "r") as f:
            return json.load(f)
    return {}

def save_epics(data):
    with open(EPIC_FILE, "w") as f:
        json.dump(data, f)

if "epics" not in st.session_state:
    st.session_state["epics"] = load_epics()

for m in MARKETS:
    val = st.text_input(
        f"{m.key} EPIC",
        value=st.session_state["epics"].get(m.key, ""),
        placeholder=m.default_epic_hint,
        key=f"epic_{m.key}"
    )
    st.session_state["epics"][m.key] = val

save_epics(st.session_state["epics"])

    st.markdown("---")
    st.caption("Tip: als je 429 krijgt → je logt te vaak in. Wacht 1–2 min en klik 1x op Login/Refresh.")


# ----------------------------
# Main content
# ----------------------------
# Market selector
market_key = st.selectbox("Select market", options=[m.key for m in MARKETS], index=0)
market = next(m for m in MARKETS if m.key == market_key)
epic = st.session_state.get("epics", {}).get(market.key, "").strip()

# Top row: Overview cards
left, mid, right = st.columns([1.2, 1.2, 1.6], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>Market</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-big'>{market.key}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>{market.label} • {market.asset_class}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with mid:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>TradingView Symbol</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-big'>{market.tv_symbol}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>Interval: {tv_interval}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Capital EPIC</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-big'>{epic if epic else '—'}</div>", unsafe_allow_html=True)
    if not epic:
        st.markdown("<div class='card-sub'>Vul EPIC in via sidebar → EPIC map</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-sub'>Wordt gebruikt voor candles/indicators</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# Capital candles + indicators (safe)
capital_ok, cap_headers, cap_msg = capital_headers()
base = (st.session_state.get("capital_token", {}) or {}).get("base") or (get_secret("CAPITAL_API_BASE") or "").rstrip("/")

ind_col, mtf_col = st.columns([1.4, 1.0], gap="large")

with ind_col:
    st.subheader("📌 Capital.com Snapshot (candles → indicators)")
    if not capital_ok:
        st.warning(cap_msg)
    elif not epic:
        st.info("Geen EPIC ingevuld. Vul EPIC in (sidebar) om candles & indicators te zien.")
    else:
        try:
            df = capital_get_prices(base, cap_headers, epic, CAPITAL_RESOLUTION_MAP.get(tv_interval, "MINUTE_15"), max_points=250)
            if df.empty or len(df) < 60:
                st.warning("Te weinig candle data ontvangen voor goede indicators. Probeer andere interval of check EPIC.")
            else:
                info = bias_from_indicators(df)
                label, badge_cls = score_to_label(info["score"])

                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<div class='card'><div class='card-title'>Price</div><div class='card-big'>{info['close']:.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='card'><div class='card-title'>RSI(14)</div><div class='card-big'>{info['rsi']:.1f}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='card'><div class='card-title'>SMA20 / SMA50</div><div class='card-big'>{info['sma20']:.2f} / {info['sma50']:.2f}</div></div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='card'><div class='card-title'>Volatility (ATR%)</div><div class='card-big'>{info['atr_pct']:.2f}%</div></div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div class='{badge_cls}'>{label}</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Bias score is een simpele institutional-style rule-set op SMA/RSI/structure. Geen financieel advies.")
        except requests.HTTPError as e:
            st.error(f"Capital request failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with mtf_col:
    st.subheader("🧠 MTF Confluence (15m → 1D)")
    if not capital_ok:
        st.warning("Login eerst op Capital.")
    elif not epic:
        st.info("Vul EPIC in voor MTF.")
    else:
        intervals = [("15", "15m"), ("60", "1H"), ("240", "4H"), ("1D", "1D")]
        scores = {}
        lines = []
        for iv, label_iv in intervals:
            try:
                df_iv = capital_get_prices(base, cap_headers, epic, CAPITAL_RESOLUTION_MAP[iv], max_points=250)
                if df_iv.empty or len(df_iv) < 60:
                    lines.append(f"- {label_iv}: onvoldoende data")
                    continue
                info_iv = bias_from_indicators(df_iv)
                scores[iv] = info_iv["score"]
                lbl, _ = score_to_label(info_iv["score"])
                lines.append(f"- {label_iv}: **{lbl}** (RSI {info_iv['rsi']:.1f})")
            except Exception:
                lines.append(f"- {label_iv}: error")

        if scores:
            verdict, cls = mtf_verdict(scores)
            st.markdown(f"<div class='{cls}'>{verdict}</div>", unsafe_allow_html=True)
        st.markdown("\n".join(lines))
        st.caption("Rule of thumb: agressiever handelen als 3+ timeframes hetzelfde zeggen.")


st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# TradingView big chart (1 per row)
st.subheader("📈 TradingView Chart (groot)")
st.components.v1.html(tradingview_advanced_chart(market.tv_symbol, tv_interval, height=700), height=740)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# Watchlist section (all markets, 1 per row, quick view)
st.subheader("🧾 Watchlist (Quick Outlook)")
st.caption("Per market: TradingView chart + (als EPIC ingevuld) Capital bias.")

for m in MARKETS:
    ep = st.session_state.get("epics", {}).get(m.key, "").strip()
    st.markdown(f"### {m.key} — {m.label}")
    row_left, row_right = st.columns([1.6, 1.0], gap="large")

    with row_left:
        st.components.v1.html(tradingview_advanced_chart(m.tv_symbol, tv_interval, height=520), height=560)

    with row_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Outlook</div>", unsafe_allow_html=True)

        if not capital_ok:
            st.markdown("<div class='card-sub'>Capital not logged in</div>", unsafe_allow_html=True)
        elif not ep:
            st.markdown("<div class='card-sub'>No EPIC set</div>", unsafe_allow_html=True)
            st.caption("Zet EPIC in sidebar.")
        else:
            try:
                df_w = capital_get_prices(base, cap_headers, ep, CAPITAL_RESOLUTION_MAP.get(tv_interval, "MINUTE_15"), max_points=250)
                if df_w.empty or len(df_w) < 60:
                    st.markdown("<div class='card-sub'>Not enough data</div>", unsafe_allow_html=True)
                else:
                    info_w = bias_from_indicators(df_w)
                    lbl, cls = score_to_label(info_w["score"])
                    st.markdown(f"<div class='{cls}'>{lbl}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-sub'>Price: <b>{info_w['close']:.2f}</b><br>"
                        f"RSI(14): <b>{info_w['rsi']:.1f}</b><br>"
                        f"ATR%: <b>{info_w['atr_pct']:.2f}%</b></div>",
                        unsafe_allow_html=True,
                    )
            except requests.HTTPError as e:
                st.error(f"Capital error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

st.caption("© UnknownFX • Built with Streamlit • Data: TradingView embeds + Capital.com API")
