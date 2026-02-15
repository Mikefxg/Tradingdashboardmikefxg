import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from tradingview_ta import TA_Handler, Interval, Exchange, TradingViewTAError

# =========================
# Settings
# =========================
REFRESH_SECONDS = 600  # 10 min
INTERVAL = Interval.INTERVAL_15_MINUTES  # bias op 15m (kan je aanpassen)

st.set_page_config(page_title="Market Sentiment (TradingView x Capital.com)", layout="wide")

# Auto-refresh de pagina elke 10 min (werkt op Streamlit Cloud)
st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
    unsafe_allow_html=True
)

# =========================
# Candidate tickers
# =========================
# TradingView symbol format: "EXCHANGE:SYMBOL"
# Capital.com is vaak: CAPITALCOM:US100 etc, maar kan per account/regio variëren.
CANDIDATES: Dict[str, List[str]] = {
    "US100 (Nasdaq CFD)": [
        "CAPITALCOM:US100",
        "CAPITALCOM:NAS100",
        "CAPITALCOM:USTEC",
        "CAPITALCOM:NAS100USD",
        "OANDA:NAS100USD",
        "TVC:NDX",
    ],
    "US500 (S&P CFD)": [
        "CAPITALCOM:US500",
        "CAPITALCOM:SPX500",
        "CAPITALCOM:USSPX500",
        "OANDA:SPX500USD",
        "TVC:SPX",
    ],
    "US30 (Dow CFD)": [
        "CAPITALCOM:US30",
        "CAPITALCOM:DJI",
        "CAPITALCOM:USA30",
        "OANDA:US30USD",
        "TVC:DJI",
    ],
    "XAUUSD (Gold Spot)": [
        "CAPITALCOM:XAUUSD",
        "CAPITALCOM:GOLD",
        "OANDA:XAUUSD",
        "TVC:GOLD",
    ],
    "EURUSD": [
        "CAPITALCOM:EURUSD",
        "OANDA:EURUSD",
        "FX:EURUSD",
    ],
    "DXY (Dollar Index)": [
        # DXY staat vaak NIET bij brokers, wél bij TVC / ICEUS.
        "TVC:DXY",
        "ICEUS:DXY",
        "CAPITALCOM:DXY",  # soms bestaat dit, vaak niet
    ],
}

# =========================
# Helpers
# =========================
@dataclass
class MarketResult:
    name: str
    symbol: str
    ok: bool
    error: Optional[str]
    rec: Optional[str]
    indicators: Optional[dict]
    score: float
    sentiment: str
    bias: str


def split_symbol(full: str) -> Tuple[str, str]:
    if ":" not in full:
        return "", full
    ex, sym = full.split(":", 1)
    return ex, sym


def compute_score(rec: str, ind: dict) -> Tuple[float, str, str]:
    """
    Score op basis van simpele, robuuste regels:
    - SMA20 vs SMA50 (trend)
    - RSI (momentum)
    - MACD histogram (momentum)
    - TradingView Recommendation (consensus)
    """
    sma20 = ind.get("SMA20")
    sma50 = ind.get("SMA50")
    rsi = ind.get("RSI")
    macd_hist = ind.get("MACD.hist")

    score = 0.0

    # Trend
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 1.0
        elif sma20 < sma50:
            score -= 1.0

    # RSI
    if rsi is not None:
        if rsi >= 55:
            score += 0.6
        elif rsi <= 45:
            score -= 0.6

    # MACD hist
    if macd_hist is not None:
        if macd_hist > 0:
            score += 0.5
        elif macd_hist < 0:
            score -= 0.5

    # TradingView recommendation
    rec = (rec or "").upper()
    if rec in ("STRONG_BUY", "BUY"):
        score += 0.8
    elif rec in ("STRONG_SELL", "SELL"):
        score -= 0.8

    # Map naar sentiment/bias
    if score >= 1.2:
        sentiment = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -1.2:
        sentiment = "BEARISH"
        bias = "SELL BIAS"
    else:
        sentiment = "NEUTRAL"
        bias = "NEUTRAL"

    return score, sentiment, bias


def fetch_ta(full_symbol: str, interval: Interval) -> Tuple[bool, Optional[str], Optional[str], Optional[dict]]:
    """
    Haal TradingView TA op. Retourneert: ok, error, recommendation, indicators
    """
    exchange, symbol = split_symbol(full_symbol)
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener="forex" if "USD" in symbol or symbol.endswith("USD") else "america",
            interval=interval,
            timeout=10,
        )
        analysis = handler.get_analysis()
        rec = analysis.summary.get("RECOMMENDATION")
        indicators = analysis.indicators
        return True, None, rec, indicators
    except Exception as e:
        return False, str(e), None, None


@st.cache_data(ttl=REFRESH_SECONDS)
def resolve_market(name: str, candidates: List[str], interval: Interval, override: str = "") -> MarketResult:
    # 1) Override eerst (als user iets invult)
    if override.strip():
        ok, err, rec, ind = fetch_ta(override.strip(), interval)
        if ok and ind:
            score, sentiment, bias = compute_score(rec, ind)
            return MarketResult(name, override.strip(), True, None, rec, ind, score, sentiment, bias)
        return MarketResult(name, override.strip(), False, err or "Unknown error", None, None, 0.0, "ERROR", "ERROR")

    # 2) Anders: probeer kandidaten in volgorde
    last_err = None
    for sym in candidates:
        ok, err, rec, ind = fetch_ta(sym, interval)
        if ok and ind:
            score, sentiment, bias = compute_score(rec, ind)
            return MarketResult(name, sym, True, None, rec, ind, score, sentiment, bias)
        last_err = err

    return MarketResult(name, candidates[0] if candidates else "—", False, last_err or "No candidates", None, None, 0.0, "ERROR", "ERROR")


def tv_widget(symbol: str, height: int = 360):
    # TradingView embed (werkt zonder API keys)
    # Let op: sommige broker-feeds renderen prima als widget, ook als TA faalt — en andersom.
    widget = f"""
    <div class="tradingview-widget-container">
      <div id="tv_{symbol.replace(':','_')}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "15",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "container_id": "tv_{symbol.replace(':','_')}"
      }});
      </script>
    </div>
    """
    st.components.v1.html(widget, height=height, scrolling=False)


# =========================
# UI
# =========================
st.title("📈 Market Sentiment (TradingView × Capital.com)")
st.caption(f"Auto-refresh elke {REFRESH_SECONDS//60} min • Interval: 15m • Feed: Capital.com waar mogelijk (fallbacks actief)")

with st.sidebar:
    st.header("⚙️ Ticker overrides (100% instelbaar)")
    st.write("Als een market niet goed resolvet, plak hier het exacte TradingView symbool, bv `CAPITALCOM:US100`.")
    overrides: Dict[str, str] = {}
    for market in CANDIDATES.keys():
        overrides[market] = st.text_input(market, value="", placeholder="Laat leeg = auto-detect")
    st.divider()
    st.write("Tip: op TradingView → klik op symbool → kopieer exact `EXCHANGE:SYMBOL` (linksboven).")

# Resolve alle markets
results: List[MarketResult] = []
for market, symbols in CANDIDATES.items():
    results.append(resolve_market(market, symbols, INTERVAL, overrides.get(market, "")))

# Cards row
cols = st.columns(6)
for i, r in enumerate(results):
    c = cols[i % 6]
    with c:
        st.subheader(r.name.split(" (")[0] if "(" in r.name else r.name)

        if not r.ok:
            st.error("Feed/TA error")
            st.caption(f"Probeerde: `{r.symbol}`")
            if r.error:
                st.caption(f"Error: {r.error[:160]}")
            st.caption("➡️ Zet een override in de sidebar als je de exacte ticker weet.")
            continue

        # Sentiment badge
        if r.sentiment == "BULLISH":
            st.success(f"🟢 {r.sentiment}")
        elif r.sentiment == "BEARISH":
            st.error(f"🔴 {r.sentiment}")
        else:
            st.info(f"⚪ {r.sentiment}")

        st.markdown(f"**{r.bias}**")
        st.caption(f"score: {r.score:+.2f}")
        st.caption(f"symbol: `{r.symbol}`")

        ind = r.indicators or {}
        last = ind.get("close") or ind.get("Close") or ind.get("close[1]")
        rsi = ind.get("RSI")
        sma20 = ind.get("SMA20")
        sma50 = ind.get("SMA50")
        macd_hist = ind.get("MACD.hist")
        tvrec = r.rec

        if last is not None:
            st.write(f"Last: **{last:.5f}**" if isinstance(last, (int, float)) else f"Last: **{last}**")
        if rsi is not None:
            st.write(f"RSI: **{rsi:.1f}**")
        if sma20 is not None and sma50 is not None:
            st.write(f"SMA20/SMA50: **{sma20:.2f} / {sma50:.2f}**")
        if macd_hist is not None:
            st.write(f"MACD hist: **{macd_hist:.4f}**")
        if tvrec:
            st.caption(f"TV Rec: {tvrec}")

st.divider()

# Charts section
st.subheader("📊 Charts (TradingView)")
st.caption("Als een chart leeg blijft: ticker bestaat niet op die feed → zet override in sidebar.")

chart_cols = st.columns(2)
for idx, r in enumerate(results):
    with chart_cols[idx % 2]:
        st.markdown(f"### {r.name}")
        if r.ok:
            tv_widget(r.symbol, height=420)
        else:
            st.warning("Geen chart: ticker/TA mismatch. Zet override in sidebar.")

st.divider()
st.caption("⚠️ Dit is een indicatie (geen financieel advies). Gebruik altijd eigen risk management.")
st.caption(f"Laatste render: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
