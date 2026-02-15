import os
import math
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trading Dashboard", layout="wide")

REFRESH_SECONDS = 60
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# Markets (spot/CFD intent) - we use TradingView symbols for charts
MARKETS = {
    "US100 (Nasdaq CFD)": {"tv": "OANDA:NAS100USD", "class": "index"},
    "US30 (Dow CFD)": {"tv": "OANDA:US30USD", "class": "index"},
    "SPX500 (S&P CFD)": {"tv": "OANDA:SPX500USD", "class": "index"},
    "XAUUSD (Spot)": {"tv": "OANDA:XAUUSD", "class": "metal"},
    "EURUSD (Spot)": {"tv": "OANDA:EURUSD", "class": "fx"},
}

# DXY proxy (TradingView) – echte DXY kan via TV: "TVC:DXY"
DXY_TV = "TVC:DXY"  # als deze niet werkt in embed, gebruik "ICEUS:DXY" of "TVC:DXY"
# Alternatief: UUP ETF is proxy, maar jij wilde spot -> daarom DXY direct via TV.

# ForexFactory: alleen high impact (RED/ORANGE)
# FF heeft "calendar.php". Scrape is best-effort.
FF_URL = "https://www.forexfactory.com/calendar"

# -----------------------------
# HELPERS: general
# -----------------------------
def clamp(x, lo=-3, hi=3):
    return max(lo, min(hi, x))

def safe_get(url, params=None, headers=None, timeout=15):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params, headers=h, timeout=timeout)
    r.raise_for_status()
    return r

def now_utc():
    return datetime.now(timezone.utc)

# -----------------------------
# TradingView embed
# -----------------------------
def tv_widget(symbol: str, height: int = 360):
    # Lightweight widget (single symbol)
    # NOTE: Streamlit HTML embed
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_{symbol.replace(':','_').replace('/','_')}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "15",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": false,
          "container_id": "tv_{symbol.replace(':','_').replace('/','_')}"
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=height+30)

# -----------------------------
# ForexFactory calendar (best-effort scrape)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_ff_calendar():
    """
    Returns dataframe with: time_utc, currency, impact, title
    impact: 'red' or 'orange' only
    """
    try:
        r = safe_get(FF_URL, timeout=20)
        html = r.text

        # SUPER simple parse that works often enough:
        # We look for rows containing "calendar__row" and extract bits.
        # If FF changes HTML, this may fail; we'll handle gracefully.

        rows = []
        # crude split
        for chunk in html.split('calendar__row'):
            # impact
            impact = None
            if "impact--high" in chunk:
                impact = "red"
            elif "impact--medium" in chunk:
                impact = "orange"
            else:
                continue  # only red/orange

            # currency (e.g. "USD")
            currency = None
            if 'calendar__currency' in chunk:
                try:
                    currency = chunk.split('calendar__currency')[1].split('>')[1].split('<')[0].strip()
                except Exception:
                    currency = None

            # title
            title = None
            if 'calendar__event-title' in chunk:
                try:
                    title = chunk.split('calendar__event-title')[1].split('>')[1].split('<')[0].strip()
                except Exception:
                    title = None

            # time (FF is local / site setting; we assume it's shown like "All Day" or "10:00am")
            # We'll store raw_time; convert to UTC is not reliable without timezone setting.
            raw_time = None
            if 'calendar__time' in chunk:
                try:
                    raw_time = chunk.split('calendar__time')[1].split('>')[1].split('<')[0].strip()
                except Exception:
                    raw_time = None

            if title and currency:
                rows.append({
                    "raw_time": raw_time or "",
                    "currency": currency,
                    "impact": impact,
                    "title": title
                })

        df = pd.DataFrame(rows).drop_duplicates()
        return df
    except Exception:
        return pd.DataFrame(columns=["raw_time", "currency", "impact", "title"])

def ff_risk_score(df_ff: pd.DataFrame) -> int:
    """
    Score event-risk based on count of red/orange items in the feed.
    More events => more risk => more bearish for indices, mixed for FX/metals.
    We'll return negative values (risk = bearish tilt).
    """
    if df_ff is None or df_ff.empty:
        return 0
    reds = (df_ff["impact"] == "red").sum()
    oranges = (df_ff["impact"] == "orange").sum()
    # weight red more
    x = -(2 * reds + 1 * oranges)
    # map to -3..0
    if x <= -8:
        return -3
    if x <= -5:
        return -2
    if x <= -2:
        return -1
    return 0

# -----------------------------
# Price / momentum / volatility signals
# We use TradingView charts for visuals; for scoring we do "proxy scoring" without paid APIs.
# We'll use a simple synthetic approach:
# - momentum: based on last refresh direction of a small free endpoint (exchangerate.host for EURUSD)
# - for indices & XAU we fallback to neutral unless you add a real data API.
# This keeps it stable and avoids 401/403 issues.
# -----------------------------
@st.cache_data(ttl=120)
def fetch_eurusd():
    try:
        # exchangerate.host (free)
        r = safe_get("https://api.exchangerate.host/latest", params={"base": "EUR", "symbols": "USD"}, timeout=15)
        j = r.json()
        rate = float(j["rates"]["USD"])
        return rate
    except Exception:
        return None

@st.cache_data(ttl=120)
def fetch_dxy_proxy_move():
    """
    Without paid feeds, we approximate DXY move using a free USD basket proxy:
    Use EURUSD + USDJPY if available; here we do a minimal approach using EURUSD only:
    - If EURUSD down -> USD stronger -> DXY up (bullish DXY).
    """
    eurusd = fetch_eurusd()
    if eurusd is None:
        return 0
    # compare with value 2 hours ago using timeseries (free)
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=7)
        r = safe_get(
            "https://api.exchangerate.host/timeseries",
            params={"start_date": start.isoformat(), "end_date": end.isoformat(), "base": "EUR", "symbols": "USD"},
            timeout=20,
        )
        j = r.json()
        rates = j.get("rates", {})
        # pick latest 2 available dates
        keys = sorted(rates.keys())
        if len(keys) < 2:
            return 0
        last = float(rates[keys[-1]]["USD"])
        prev = float(rates[keys[-2]]["USD"])
        # EURUSD up => USD weaker => DXY down (negative)
        pct = (last - prev) / prev
        if pct > 0.002:
            return -2
        if pct > 0.0005:
            return -1
        if pct < -0.002:
            return 2
        if pct < -0.0005:
            return 1
        return 0
    except Exception:
        return 0

def momentum_score(market: str) -> int:
    """
    Minimal momentum scoring.
    - EURUSD: uses exchangerate change (today vs previous day in timeseries)
    - Others: neutral (0) unless you add a price API later
    """
    if "EURUSD" in market:
        # reuse dxy proxy logic but inverted: EURUSD up => bullish EURUSD
        try:
            end = datetime.utcnow().date()
            start = end - timedelta(days=7)
            r = safe_get(
                "https://api.exchangerate.host/timeseries",
                params={"start_date": start.isoformat(), "end_date": end.isoformat(), "base": "EUR", "symbols": "USD"},
                timeout=20,
            )
            j = r.json()
            rates = j.get("rates", {})
            keys = sorted(rates.keys())
            if len(keys) < 2:
                return 0
            last = float(rates[keys[-1]]["USD"])
            prev = float(rates[keys[-2]]["USD"])
            pct = (last - prev) / prev
            if pct > 0.002:
                return 2
            if pct > 0.0005:
                return 1
            if pct < -0.002:
                return -2
            if pct < -0.0005:
                return -1
            return 0
        except Exception:
            return 0

    # TODO: add real CFD spot feeds later
    return 0

def volatility_score(market: str, ff_score: int) -> int:
    """
    Use FF event risk as a volatility proxy:
    more red/orange => higher vol => bearish for indices
    return:
      calm => +1
      elevated => 0
      high => -1 or -2
    """
    if ff_score <= -3:
        return -2
    if ff_score <= -2:
        return -1
    if ff_score <= -1:
        return 0
    return 1

def dxy_effect_for_market(market: str, dxy_move: int) -> int:
    """
    Positive dxy_move = USD stronger.
    - EURUSD: USD stronger => bearish EURUSD
    - XAUUSD: USD stronger often bearish gold
    - Indices: strong USD can be mixed; keep mild negative if risk-off
    """
    if "EURUSD" in market:
        return clamp(-dxy_move, -2, 2)
    if "XAUUSD" in market:
        return clamp(-dxy_move, -2, 2)
    if "US" in market or "SPX" in market:
        return clamp(-1 if dxy_move >= 2 else 0, -2, 2)
    return 0

def label_from_total(total: int) -> str:
    if total >= 3:
        return "BULLISH"
    if total == 2:
        return "BULLISH"
    if total == 1:
        return "SLIGHT BULL"
    if total == 0:
        return "NEUTRAL"
    if total == -1:
        return "SLIGHT BEAR"
    if total <= -2:
        return "BEARISH"
    return "NEUTRAL"

def badge(label: str) -> str:
    if "BULL" in label:
        return f"<span style='padding:6px 10px;border-radius:999px;background:#E8F7EE;color:#0A6B2D;font-weight:700;font-size:12px'>{label}</span>"
    if "BEAR" in label:
        return f"<span style='padding:6px 10px;border-radius:999px;background:#FDECEC;color:#9B1C1C;font-weight:700;font-size:12px'>{label}</span>"
    return f"<span style='padding:6px 10px;border-radius:999px;background:#EEF2F7;color:#334155;font-weight:700;font-size:12px'>{label}</span>"

def outlook_text(market: str, mom: int, vol: int, ff_score: int, dxy_component: int, total: int) -> str:
    parts = []

    if total >= 2:
        regime = "Risk-on / bullish bias"
    elif total <= -2:
        regime = "Risk-off / bearish bias"
    else:
        regime = "Mixed / neutral bias"

    if mom >= 2:
        parts.append("momentum stijgt")
    elif mom <= -2:
        parts.append("momentum daalt")
    else:
        parts.append("momentum is gemengd")

    if vol <= -2:
        parts.append("volatiliteit is hoog")
    elif vol == -1:
        parts.append("volatiliteit is verhoogd")
    else:
        parts.append("volatiliteit is rustig")

    if ff_score <= -2:
        parts.append("veel RED/ORANGE nieuws (event-risk)")
    elif ff_score == -1:
        parts.append("RED/ORANGE nieuws in de buurt")
    else:
        parts.append("weinig high-impact nieuws")

    if dxy_component > 0:
        parts.append("DXY werkt mee")
    elif dxy_component < 0:
        parts.append("DXY werkt tegen")
    else:
        parts.append("DXY neutraal")

    if "XAUUSD" in market:
        hint = " (Goud reageert sterk op USD + risk-off.)"
    elif "EURUSD" in market:
        hint = " (EURUSD is vaak inverse van USD-strength.)"
    else:
        hint = " (Indices houden niet van hoge vol + heavy news.)"

    return f"**{regime}** — " + ", ".join(parts) + "." + hint

def global_outlook(market_scores: dict, ff_score: int, dxy_move: int) -> tuple[str, str]:
    """
    Combine everything into a single global regime.
    """
    # average market totals
    totals = [v["total"] for v in market_scores.values()]
    avg = sum(totals) / max(1, len(totals))

    # global adjustments
    # more event risk => more risk-off
    adj = 0
    adj += ff_score  # ff_score is negative when risky
    # strong USD can lean risk-off a bit
    if dxy_move >= 2:
        adj -= 1
    elif dxy_move <= -2:
        adj += 1

    g = clamp(round(avg + 0.6 * adj), -3, 3)
    lbl = label_from_total(g)

    explanation = []
    explanation.append(f"Gemiddelde marketscore: **{avg:+.2f}**")
    explanation.append(f"Event-risk (FF RED/ORANGE): **{ff_score:+d}**")
    explanation.append(f"DXY move proxy: **{dxy_move:+d}**")

    if g >= 2:
        exp2 = "Globaal: **Risk-on** (bullish bias). Focus op trend-follow, dips buy (met risk management rond nieuws)."
    elif g <= -2:
        exp2 = "Globaal: **Risk-off** (bearish bias). Focus op defensief, mean reversion/short rallies, en smaller size rond nieuws."
    else:
        exp2 = "Globaal: **Mixed**. Kies selectief: trade alleen A+ setups, pas size aan, en respecteer nieuws windows."

    return lbl, " • ".join(explanation) + "\n\n" + exp2

# -----------------------------
# UI
# -----------------------------
st.title("📊 Trading Dashboard")
st.caption(
    f"Outlook score = momentum + volatility + news(event-risk) + DXY-effect. "
    f"Auto refresh elke {REFRESH_SECONDS}s. Charts via TradingView. Calendar: alleen RED/ORANGE."
)

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

# Pull FF events and DXY proxy move
df_ff = fetch_ff_calendar()
ff_score = ff_risk_score(df_ff)
dxy_move = fetch_dxy_proxy_move()

# Build per-market scores
market_scores = {}
for m in MARKETS.keys():
    mom = momentum_score(m)
    vol = volatility_score(m, ff_score)
    dxy_component = dxy_effect_for_market(m, dxy_move)

    # News component: treat high event risk as bearish bias across all, strongest for indices
    if MARKETS[m]["class"] == "index":
        news_component = clamp(ff_score, -3, 0)
    else:
        # FX/metals are more two-sided; keep milder
        news_component = clamp(math.ceil(ff_score / 2), -2, 0)

    total = clamp(mom + vol + news_component + dxy_component, -3, 3)
    lbl = label_from_total(total)

    market_scores[m] = {
        "mom": mom,
        "vol": vol,
        "news": news_component,
        "dxy": dxy_component,
        "total": total,
        "label": lbl,
    }

# GLOBAL OUTLOOK
glbl, gtext = global_outlook(market_scores, ff_score, dxy_move)
st.markdown(f"### Global Outlook {badge(glbl)}", unsafe_allow_html=True)
st.write(gtext)

st.divider()

# MARKET CARDS
cols = st.columns(len(MARKETS))
for i, (m, meta) in enumerate(MARKETS.items()):
    s = market_scores[m]
    with cols[i]:
        st.subheader(m)
        st.markdown(badge(s["label"]), unsafe_allow_html=True)

        st.write(f"Score: **{s['total']:+d}**  "
                 f"(mom {s['mom']:+d} | vol {s['vol']:+d} | news {s['news']:+d} | dxy {s['dxy']:+d})")

        st.write(outlook_text(m, s["mom"], s["vol"], ff_score, s["dxy"], s["total"]))

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 Charts", "🗞️ News (RED/ORANGE)", "💵 DXY"])

with tab1:
    pick = st.selectbox("Select market", list(MARKETS.keys()), index=0)
    st.caption("Tip: als een OANDA symbool niet laadt, zeg het, dan pak ik een andere TradingView ticker.")
    tv_widget(MARKETS[pick]["tv"], height=520)

with tab2:
    st.subheader("ForexFactory Calendar (alleen RED/ORANGE)")
    if df_ff is None or df_ff.empty:
        st.warning("Geen events gevonden. ForexFactory kan scraping blokkeren (rate-limit/403). "
                   "Als dit blijft, zet ik je om naar een alternatief met stabiele API.")
    else:
        # Show only USD/EUR high impact first for your instruments
        prio = df_ff[df_ff["currency"].isin(["USD", "EUR"])]
        rest = df_ff[~df_ff["currency"].isin(["USD", "EUR"])]

        st.markdown("**Priority (USD/EUR):**")
        st.dataframe(prio.reset_index(drop=True), use_container_width=True)

        st.markdown("**Others:**")
        st.dataframe(rest.reset_index(drop=True), use_container_width=True)

        st.caption("Impact: red = high, orange = medium (ForexFactory).")

with tab3:
    st.subheader("DXY")
    st.write(f"DXY proxy move score: **{dxy_move:+d}** (positief = USD sterker, negatief = USD zwakker)")
    tv_widget(DXY_TV, height=520)
