import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparser

# TradingView TA (unofficial but commonly works)
from tradingview_ta import TA_Handler, Interval

st.set_page_config(page_title="Trading Outlook Dashboard", layout="wide")

# ---------------------------
# CONFIG
# ---------------------------
REFRESH_SECONDS = 60

# CFD/SPOT symbols on TradingView (pas aan als jij andere broker-prefix gebruikt)
MARKETS = {
    "US100 (Nasdaq CFD)": {"tv": "OANDA:NAS100USD", "exchange": "OANDA"},
    "US30 (Dow CFD)": {"tv": "OANDA:US30USD", "exchange": "OANDA"},
    "SPX500 (S&P CFD)": {"tv": "OANDA:SPX500USD", "exchange": "OANDA"},
    "XAUUSD (Gold Spot)": {"tv": "OANDA:XAUUSD", "exchange": "OANDA"},
    "EURUSD": {"tv": "OANDA:EURUSD", "exchange": "OANDA"},
    "DXY (Dollar Index)": {"tv": "TVC:DXY", "exchange": "TVC"},
}

# Myfxbook calendar URL
MYFXBOOK_CAL_URL = "https://www.myfxbook.com/forex-economic-calendar"

# We tonen alleen Medium/High (orange/red)
ALLOWED_IMPACTS = {"Medium", "High"}

# ---------------------------
# HELPERS
# ---------------------------

def _tv_split_symbol(tv: str):
    # "OANDA:XAUUSD" -> ("OANDA", "XAUUSD")
    if ":" not in tv:
        return "TVC", tv
    ex, sym = tv.split(":", 1)
    return ex, sym

@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_tradingview_ta(tv_symbol: str):
    ex, sym = _tv_split_symbol(tv_symbol)
    handler = TA_Handler(
        symbol=sym,
        exchange=ex,
        screener="forex" if ex in ["OANDA", "FX", "FOREXCOM"] else "america",
        interval=Interval.INTERVAL_1_HOUR,
    )
    analysis = handler.get_analysis()

    # summary: BUY/SELL/NEUTRAL counts + overall recommendation
    rec = analysis.summary.get("RECOMMENDATION", "NEUTRAL")

    # indicators we use for scoring/volatility
    ind = analysis.indicators
    # Some symbols may miss fields; use safe get
    rsi = float(ind.get("RSI", 50))
    adx = float(ind.get("ADX", 20))
    atr = ind.get("ATR", None)  # absolute ATR
    close = ind.get("close", None) or ind.get("Close", None)

    # compute ATR% if possible
    atrp = None
    if atr is not None and close is not None and float(close) != 0:
        atrp = (float(atr) / float(close)) * 100.0

    return {
        "recommendation": rec,
        "rsi": rsi,
        "adx": adx,
        "atr": atr,
        "close": close,
        "atr_pct": atrp,
        "raw": analysis.summary,
    }

def outlook_from_ta(ta: dict):
    """
    Maak een simpele maar duidelijke Outlook:
    - Trend bias: TradingView recommendation
    - Sterkte: ADX
    - Volatiliteit: ATR%
    Score: -100 .. +100
    """
    rec = ta["recommendation"]
    rsi = ta["rsi"]
    adx = ta["adx"]
    atrp = ta["atr_pct"]

    # base score from recommendation
    base = {
        "STRONG_BUY": 70,
        "BUY": 40,
        "NEUTRAL": 0,
        "SELL": -40,
        "STRONG_SELL": -70,
    }.get(rec, 0)

    # ADX boosts conviction (trend strength)
    # ADX ~ 10 weak, 25 trend, 40 strong
    strength_boost = 0
    if adx >= 40:
        strength_boost = 20
    elif adx >= 25:
        strength_boost = 10
    elif adx <= 15:
        strength_boost = -10

    # RSI extremes add caution
    caution = 0
    if rsi >= 75:
        caution = -10
    elif rsi <= 25:
        caution = -10

    # Volatility label
    vol_label = "—"
    if atrp is not None:
        if atrp >= 1.2:
            vol_label = "High vol"
        elif atrp >= 0.6:
            vol_label = "Medium vol"
        else:
            vol_label = "Low vol"

    score = int(max(-100, min(100, base + strength_boost + caution)))

    if score >= 30:
        label = "BULLISH"
    elif score <= -30:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "score": score,
        "label": label,
        "vol_label": vol_label,
    }

@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_myfxbook_calendar():
    """
    Scrape Myfxbook economic calendar page.
    We filter later to only Medium/High.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitDashboard/1.0)"
    }
    r = requests.get(MYFXBOOK_CAL_URL, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # Myfxbook page contains event blocks in html text.
    # Easiest robust way: use pandas read_html and pick tables that look like calendar.
    tables = pd.read_html(r.text)

    # Heuristic: pick the biggest table
    df = max(tables, key=lambda x: len(x)) if tables else pd.DataFrame()

    # Normalize columns if possible
    # The exact column names can vary. We try to detect common fields.
    cols = [c.lower() for c in df.columns.astype(str).tolist()]
    df.columns = cols

    # Common expected fields in some form:
    # date/time, currency, event, impact, actual, forecast, previous
    # If the table already has these, great; otherwise we still display minimal.
    return df

def clean_calendar(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    # Try to map columns
    # We attempt to find: "date" or "time", "currency", "event", "impact"
    col_map = {}
    for c in df.columns:
        if "date" in c or "time" in c:
            col_map["time"] = c
        elif "currency" in c or c in ["ccy", "cur"]:
            col_map["currency"] = c
        elif "event" in c:
            col_map["event"] = c
        elif "impact" in c or "volatility" in c:
            col_map["impact"] = c
        elif "actual" in c:
            col_map["actual"] = c
        elif "forecast" in c:
            col_map["forecast"] = c
        elif "previous" in c:
            col_map["previous"] = c

    # If critical fields missing, just return something
    if "currency" not in col_map or "event" not in col_map:
        # best effort: show raw
        out = df.copy()
        return out

    out = pd.DataFrame()
    out["currency"] = df[col_map["currency"]].astype(str)
    out["event"] = df[col_map["event"]].astype(str)

    if "impact" in col_map:
        out["impact"] = df[col_map["impact"]].astype(str)
    else:
        out["impact"] = "Unknown"

    if "time" in col_map:
        out["time_raw"] = df[col_map["time"]].astype(str)
    else:
        out["time_raw"] = ""

    for k in ["actual", "forecast", "previous"]:
        if k in col_map:
            out[k] = df[col_map[k]].astype(str)
        else:
            out[k] = ""

    # Filter Medium/High only
    out["impact_clean"] = out["impact"].str.extract(r"(High|Medium|Low)", expand=False)
    out = out[out["impact_clean"].isin(ALLOWED_IMPACTS)].copy()
    out.drop(columns=["impact"], inplace=True, errors="ignore")
    out.rename(columns={"impact_clean": "impact"}, inplace=True)

    # Sort-ish: leave as-is if parsing time is messy
    return out.reset_index(drop=True)

def tv_widget(symbol: str, height: int = 420):
    """
    TradingView Advanced Chart widget (iframe).
    """
    # TradingView widget uses symbol like "OANDA:XAUUSD"
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "allow_symbol_change": true,
        "save_image": false,
        "studies": ["RSI@tv-basicstudies","ATR@tv-basicstudies"],
        "container_id": "tv_chart"
      }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=height, scrolling=False)

# ---------------------------
# UI
# ---------------------------
st.title("📊 Trading Outlook Dashboard")
st.caption(
    f"Auto refresh ~ elke {REFRESH_SECONDS}s • CFD/Spot via TradingView • Outlook: TA + volatility • Calendar: Myfxbook (High/Medium)"
)

# Auto refresh
st.markdown(
    f"""
    <script>
    setTimeout(function() {{
        window.location.reload();
    }}, {REFRESH_SECONDS * 1000});
    </script>
    """,
    unsafe_allow_html=True,
)

# Top cards
top_keys = ["US100 (Nasdaq CFD)", "US30 (Dow CFD)", "SPX500 (S&P CFD)", "XAUUSD (Gold Spot)", "EURUSD"]
cols = st.columns(len(top_keys))

for i, k in enumerate(top_keys):
    sym = MARKETS[k]["tv"]
    with cols[i]:
        st.subheader(k)

        try:
            ta = fetch_tradingview_ta(sym)
            out = outlook_from_ta(ta)

            st.metric("Outlook", out["label"], delta=f"score {out['score']}")
            st.write(f"**Volatility:** {out['vol_label']}")
            st.write(f"**TV Rec:** `{ta['recommendation']}`")
            st.write(f"RSI: {ta['rsi']:.1f} • ADX: {ta['adx']:.1f}")
            if ta["close"] is not None:
                st.write(f"Last (TV): {float(ta['close']):,.5f}" if "USD" in sym else f"Last (TV): {float(ta['close']):,.2f}")
        except Exception as e:
            st.warning(f"TA error: {e}")

st.divider()

# Calendar + DXY
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🗓️ Economic Calendar (High / Medium only) — Myfxbook")
    try:
        raw = fetch_myfxbook_calendar()
        cal = clean_calendar(raw)
        if cal.empty:
            st.info("Geen calendar data kunnen ophalen (mogelijk layout gewijzigd / rate limit).")
        else:
            # Optional filter currencies
            ccy_filter = st.multiselect(
                "Filter currencies (optioneel)",
                options=sorted(cal["currency"].unique().tolist()),
                default=["USD", "EUR"] if set(["USD", "EUR"]).issubset(set(cal["currency"].unique())) else None,
            )
            view = cal.copy()
            if ccy_filter:
                view = view[view["currency"].isin(ccy_filter)]
            st.dataframe(view, use_container_width=True, height=380)
    except Exception as e:
        st.error(f"Myfxbook calendar error: {e}")

with right:
    st.subheader("💵 DXY (Dollar Index) outlook")
    try:
        ta = fetch_tradingview_ta(MARKETS["DXY (Dollar Index)"]["tv"])
        out = outlook_from_ta(ta)
        st.metric("DXY Outlook", out["label"], delta=f"score {out['score']}")
        st.write(f"**Volatility:** {out['vol_label']}")
        st.write(f"**TV Rec:** `{ta['recommendation']}`")
        st.write(f"RSI: {ta['rsi']:.1f} • ADX: {ta['adx']:.1f}")
        if ta["close"] is not None:
            st.write(f"Last (TV): {float(ta['close']):,.2f}")
    except Exception as e:
        st.warning(f"DXY TA error: {e}")

st.divider()

# Chart section
st.subheader("📈 Chart (TradingView)")
chart_market = st.selectbox("Select market", options=list(MARKETS.keys()), index=0)
tv_widget(MARKETS[chart_market]["tv"], height=520)

st.caption("Tip: als een symbool niet werkt met OANDA, probeer bv FOREXCOM:SPXUSD / OANDA:SPX500USD / TVC:DXY etc.")
