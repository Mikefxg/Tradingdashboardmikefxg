import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser

from tradingview_ta import TA_Handler, Interval

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trading Outlook Dashboard", layout="wide")

REFRESH_SECONDS = 60
LOOKAHEAD_HOURS = 24          # upcoming news window
NEWS_SOON_HOURS = 6           # bigger penalty if big news soon
TIMEZONE_NAME = "UTC"         # Myfxbook times are tricky; keep everything UTC-like

# OANDA symbols that work with tradingview_ta
MARKETS = {
    "US100 (Nasdaq CFD)": {"symbol": "NAS100USD", "exchange": "OANDA", "screener": "forex", "currencies": ["USD"]},
    "US30 (Dow CFD)": {"symbol": "US30USD", "exchange": "OANDA", "screener": "forex", "currencies": ["USD"]},
    "SPX500 (S&P CFD)": {"symbol": "SPX500USD", "exchange": "OANDA", "screener": "forex", "currencies": ["USD"]},
    "XAUUSD (Gold Spot)": {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex", "currencies": ["USD"]},
    "EURUSD": {"symbol": "EURUSD", "exchange": "OANDA", "screener": "forex", "currencies": ["EUR", "USD"]},

    # DXY: OANDA "USDOLLAR" sometimes works, but TVC:DXY is more reliable.
    "DXY (Dollar Index)": {"symbol": "DXY", "exchange": "TVC", "screener": "america", "currencies": ["USD"]},
}

# Weighting (tweak later)
WEIGHTS = {
    "ta": 0.65,
    "vol": 0.20,
    "news": 0.15,
}

TA_SCORE_MAP = {
    "STRONG_BUY":  +2,
    "BUY":         +1,
    "NEUTRAL":      0,
    "SELL":        -1,
    "STRONG_SELL": -2,
}

# -----------------------------
# HELPERS
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def now_utc():
    return datetime.now(timezone.utc)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rec_to_score(rec: str) -> int:
    return TA_SCORE_MAP.get((rec or "").upper(), 0)

def score_to_label(score: float):
    if score >= 0.75:
        return "BULLISH"
    if score <= -0.75:
        return "BEARISH"
    return "NEUTRAL"

def label_color(label: str):
    if label == "BULLISH":
        return "🟢"
    if label == "BEARISH":
        return "🔴"
    return "⚪️"

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_ta(symbol: str, exchange: str, screener: str, interval: Interval):
    handler = TA_Handler(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        interval=interval
    )
    analysis = handler.get_analysis()
    return {
        "recommendation": analysis.summary.get("RECOMMENDATION"),
        "indicators": analysis.indicators,
        "summary": analysis.summary,
    }

def volatility_proxy(indicators: dict):
    """
    "Volatility proxy" (0..1):
    - Higher ADX -> more directional volatility
    - RSI far from 50 -> more momentum/expansion
    """
    adx = safe_float(indicators.get("ADX"))
    rsi = safe_float(indicators.get("RSI"))

    if adx is None and rsi is None:
        return None

    # Normalize ADX: 10..40 (rough range)
    adx_n = None
    if adx is not None:
        adx_n = clamp((adx - 10) / (40 - 10), 0, 1)

    # RSI distance from 50: 0..25 becomes 0..1
    rsi_n = None
    if rsi is not None:
        rsi_n = clamp(abs(rsi - 50) / 25, 0, 1)

    parts = [p for p in [adx_n, rsi_n] if p is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)

@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_myfxbook_calendar():
    """
    Best-effort scrape Myfxbook calendar.
    Returns list of dicts: time_utc (datetime), currency, impact ("High"/"Medium"/etc), title
    Myfxbook markup can change; we handle failures gracefully.
    """
    url = "https://www.myfxbook.com/forex-economic-calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TradingOutlookDashboard/1.0)"
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # Myfxbook often renders a table with rows containing date/time/currency/impact/event
    # We’ll collect rows by looking for the main calendar table.
    events = []

    table = soup.find("table")
    if not table:
        return events

    rows = table.find_all("tr")
    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        text_cells = [td.get_text(" ", strip=True) for td in tds]
        joined = " | ".join(text_cells).lower()

        # heuristic: skip headers / empty
        if "currency" in joined and "event" in joined:
            continue

        # attempt extract:
        # time often in first/second cell; currency in a cell; impact indicated by words or icons
        raw_time = text_cells[0] or ""
        currency = None
        impact = None
        title = None

        # find currency: usually 3-letter like USD/EUR etc.
        for cell in text_cells:
            c = cell.strip().upper()
            if c in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]:
                currency = c
                break

        # impact: look for High/Medium/Low text
        for cell in text_cells:
            c = cell.strip().lower()
            if "high" in c:
                impact = "High"
                break
            if "medium" in c:
                impact = "Medium"
                break
            if "low" in c:
                impact = "Low"
                break

        # title: pick the longest "event-like" cell
        title = max(text_cells, key=lambda s: len(s)) if text_cells else None

        # parse time: Myfxbook time formatting is inconsistent.
        # We’ll try parse with dateutil. If no date is present, assume "today" in UTC.
        time_utc = None
        try:
            # If raw_time contains just "HH:MM", add today's date
            raw = raw_time.strip()
            if raw and (":" in raw) and all(ch.isdigit() or ch in [":", " "] for ch in raw):
                today = now_utc().date().isoformat()
                time_utc = dtparser.parse(f"{today} {raw}", dayfirst=False).replace(tzinfo=timezone.utc)
            else:
                # try parse direct
                time_utc = dtparser.parse(raw_time)
                if time_utc.tzinfo is None:
                    time_utc = time_utc.replace(tzinfo=timezone.utc)
                else:
                    time_utc = time_utc.astimezone(timezone.utc)
        except Exception:
            time_utc = None

        if currency and impact and title:
            events.append({
                "time_utc": time_utc,
                "currency": currency,
                "impact": impact,
                "title": title,
                "source": "Myfxbook",
            })

    # Deduplicate rough
    uniq = []
    seen = set()
    for e in events:
        key = (e["currency"], e["impact"], e["title"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    return uniq

def filter_upcoming_events(events, currencies, lookahead_hours=24):
    start = now_utc() - timedelta(minutes=1)
    end = now_utc() + timedelta(hours=lookahead_hours)

    out = []
    for e in events:
        if e["impact"] not in ["High", "Medium"]:
            continue
        if e["currency"] not in currencies:
            continue
        t = e["time_utc"]
        # If we couldn't parse time, still show but don't score heavy
        if t is None:
            out.append(e)
            continue
        if start <= t <= end:
            out.append(e)

    # sort by time (unknown at end)
    out.sort(key=lambda x: x["time_utc"] if x["time_utc"] else datetime.max.replace(tzinfo=timezone.utc))
    return out

def news_pressure_score(upcoming_events):
    """
    Return -1..0 (pressure / uncertainty), more negative if High impact soon.
    """
    if not upcoming_events:
        return 0.0

    pressure = 0.0
    for e in upcoming_events:
        impact = e["impact"]
        t = e["time_utc"]

        base = -0.20 if impact == "Medium" else -0.40  # High impact more negative
        if t is None:
            # unknown time -> small penalty
            pressure += base * 0.25
            continue

        hours = (t - now_utc()).total_seconds() / 3600.0
        if hours < 0:
            continue

        # closer news -> stronger penalty
        if hours <= NEWS_SOON_HOURS:
            mult = 1.0
        elif hours <= 12:
            mult = 0.6
        else:
            mult = 0.35

        pressure += base * mult

    # clamp: don't overdo
    return clamp(pressure, -1.0, 0.0)

def market_outlook(ta_rec: str, vol: float, news_pressure: float):
    # TA score: -2..+2 -> normalize to -1..+1
    ta_s = rec_to_score(ta_rec)
    ta_n = ta_s / 2.0

    # Vol: 0..1 -> map to -0.1..+0.1 "confidence boost" depending on direction strength
    # If TA is strong direction, volatility helps; if neutral, volatility is just noise.
    if vol is None:
        vol_term = 0.0
    else:
        vol_term = (vol - 0.5) * 0.2  # -0.1..+0.1

    # news_pressure is negative (0 to -1)
    # We'll blend: more upcoming important news => slightly reduce bullish/bearish conviction
    # So we push score toward 0 by subtracting magnitude from absolute
    # We'll implement as direct weighted add:
    score = (WEIGHTS["ta"] * ta_n) + (WEIGHTS["vol"] * vol_term) + (WEIGHTS["news"] * news_pressure)

    # clamp
    score = clamp(score, -1.0, 1.0)
    return score

def fmt_dt(dt):
    if dt is None:
        return "Time: ?"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# -----------------------------
# UI HEADER
# -----------------------------
st.title("📊 Trading Outlook Dashboard")
st.caption(f"Auto refresh ~ elke {REFRESH_SECONDS}s • CFD/Spot via TradingView (OANDA) • Outlook: TA + volatility + news • Calendar: Myfxbook (High/Medium)")

# Auto refresh
st.markdown(
    f"""
    <script>
    setTimeout(function(){{
        window.location.reload();
    }}, {REFRESH_SECONDS * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# NEWS FETCH
# -----------------------------
with st.spinner("News calendar laden (Myfxbook)..."):
    try:
        all_events = fetch_myfxbook_calendar()
        calendar_ok = True
    except Exception as e:
        all_events = []
        calendar_ok = False
        st.warning(f"Myfxbook calendar niet beschikbaar (tijdelijk). Error: {e}")

# -----------------------------
# MARKET CARDS
# -----------------------------
cols = st.columns(len(MARKETS))

market_results = {}

for i, (name, cfg) in enumerate(MARKETS.items()):
    with cols[i]:
        st.subheader(name)

        # TA fetch
        ta_rec = None
        indicators = {}
        last = None
        rsi = None
        adx = None

        try:
            ta = fetch_ta(cfg["symbol"], cfg["exchange"], cfg["screener"], Interval.INTERVAL_1_HOUR)
            ta_rec = ta["recommendation"] or "NEUTRAL"
            indicators = ta["indicators"] or {}
            last = safe_float(indicators.get("close"))
            rsi = safe_float(indicators.get("RSI"))
            adx = safe_float(indicators.get("ADX"))
        except Exception as e:
            st.error(f"TA error: {e}")
            continue

        # volatility proxy
        vol = volatility_proxy(indicators)

        # news
        currencies = cfg.get("currencies", [])
        upcoming = filter_upcoming_events(all_events, currencies, LOOKAHEAD_HOURS) if calendar_ok else []
        pressure = news_pressure_score(upcoming)

        # outlook score
        score = market_outlook(ta_rec, vol, pressure)
        label = score_to_label(score)

        # Show
        st.markdown(f"### {label_color(label)} {label}")
        st.metric("Outlook score", f"{score:+.2f}")

        st.write(f"**TV Rec:** `{ta_rec}`")
        if last is not None:
            st.write(f"**Last (TV):** `{last}`")
        if rsi is not None and adx is not None:
            st.write(f"**RSI:** `{rsi:.1f}` • **ADX:** `{adx:.1f}`")
        if vol is not None:
            st.write(f"**Volatility (proxy):** `{vol:.2f}`")

        if calendar_ok:
            if upcoming:
                st.write("**Upcoming High/Medium news:**")
                for e in upcoming[:3]:
                    st.caption(f"• {fmt_dt(e['time_utc'])} • {e['currency']} • {e['impact']} • {e['title'][:60]}")
            else:
                st.caption("Geen High/Medium events in komende 24h (voor deze currencies).")

        market_results[name] = {
            "label": label,
            "score": score,
            "ta": ta_rec,
            "vol": vol,
            "news_pressure": pressure,
            "last": last,
        }

st.divider()

# -----------------------------
# NEWS TABLE (GLOBAL FILTER: RED/ORANGE = High/Medium)
# -----------------------------
st.subheader("🗓️ Upcoming High/Medium Events (Myfxbook)")
if not calendar_ok:
    st.info("Calendar kon niet geladen worden. Probeer later opnieuw.")
else:
    # show everything high/medium next 24h
    start = now_utc() - timedelta(minutes=1)
    end = now_utc() + timedelta(hours=LOOKAHEAD_HOURS)

    rows = []
    for e in all_events:
        if e["impact"] not in ["High", "Medium"]:
            continue
        t = e["time_utc"]
        if t is not None and not (start <= t <= end):
            continue
        rows.append({
            "Time (UTC)": fmt_dt(t).replace("Time: ", ""),
            "Currency": e["currency"],
            "Impact": e["impact"],
            "Event": e["title"],
            "Source": e["source"],
        })

    if rows:
        df_news = pd.DataFrame(rows)
        st.dataframe(df_news, use_container_width=True, height=320)
    else:
        st.caption("Geen High/Medium events gevonden in komende 24h (of parsing is leeg).")

st.divider()

# -----------------------------
# DETAILS / EXPLANATION (simple)
# -----------------------------
with st.expander("ℹ️ Hoe wordt Bullish/Bearish bepaald?"):
    st.write(
        """
- **TA (TradingView)** geeft de basis richting (STRONG BUY → BUY → NEUTRAL → SELL → STRONG SELL).
- **Volatility proxy** (ADX + RSI distance) verhoogt of verlaagt de “confidence” licht.
- **News pressure** (High/Medium events) drukt de score richting NEUTRAL, vooral als het binnen ~6 uur is.
        """
    )
