# streamlit_app.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import streamlit as st
from tradingview_ta import TA_Handler, Interval, Exchange

# =========================
# SETTINGS (pas hier aan)
# =========================

REFRESH_SECONDS = 600  # 10 minuten
INTERVAL = Interval.INTERVAL_15_MINUTES  # 15m werkt fijn voor sentiment

# 100% instelbaar: probeer eerst Capital.com-achtige symbols.
# Vul hier EXACT in wat jij in TradingView ziet (symbool + exchange/screener context).
# Als een symbool niet bestaat, probeert het script fallbacks.
SYMBOLS: Dict[str, Dict[str, str]] = {
    "US100 (Nasdaq 100)": {
        "symbol": "US100",
        "screener": "cfd",
        "exchange": "CAPITALCOM",
        "fallbacks": "NDX, NAS100, US100USD, US100IDX, QQQ",
    },
    "US500 (S&P 500)": {
        "symbol": "US500",
        "screener": "cfd",
        "exchange": "CAPITALCOM",
        "fallbacks": "SPX, SP500, US500USD, US500IDX, SPY",
    },
    "US30 (Dow Jones)": {
        "symbol": "US30",
        "screener": "cfd",
        "exchange": "CAPITALCOM",
        "fallbacks": "DJI, DJIA, US30USD, US30IDX, DIA",
    },
    "XAUUSD (Gold Spot)": {
        "symbol": "XAUUSD",
        "screener": "cfd",
        "exchange": "CAPITALCOM",
        "fallbacks": "XAUUSD, GOLD, XAUUSDUSD",
    },
    "EURUSD": {
        "symbol": "EURUSD",
        "screener": "forex",
        "exchange": "OANDA",
        "fallbacks": "EURUSD, FX:EURUSD, OANDA:EURUSD, FOREXCOM:EURUSD",
    },
    "DXY (Dollar Index)": {
        "symbol": "DXY",
        "screener": "forex",
        "exchange": "TVC",
        "fallbacks": "DXY, TVC:DXY, ICEUS:DXY",
    },
}

# =========================
# UI helpers
# =========================

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .big-title { font-size: 56px; font-weight: 800; margin-bottom: 0.2rem; }
      .subtle { color: #777; margin-top: 0; }
      .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 700; }
      .bull { background: #e7f7ed; color: #177245; }
      .bear { background: #fde8e8; color: #a61b1b; }
      .neut { background: #f1f3f5; color: #444; }
      .card { border: 1px solid #eee; border-radius: 16px; padding: 16px; }
      .muted { color: #666; font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TA logic
# =========================

@dataclass
class MarketResult:
    ok: bool
    used: str
    sentiment: str
    bias: str
    score: float
    last: Optional[float]
    rsi: Optional[float]
    sma20: Optional[float]
    sma50: Optional[float]
    macd: Optional[float]
    rec: Optional[str]
    error: Optional[str]


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def calc_score(ind: Dict) -> Tuple[float, str, str]:
    """
    Score = combinatie van:
    - SMA20 vs SMA50 (trend)
    - RSI (momentum)
    - MACD.hist (momentum)
    - TradingView recommendation (bias)
    Output: (score -2..+2), sentiment label, buy/sell bias text
    """
    score = 0.0

    rsi = safe_float(ind.get("RSI"))
    sma20 = safe_float(ind.get("SMA20"))
    sma50 = safe_float(ind.get("SMA50"))
    macd = safe_float(ind.get("MACD.macd"))
    macds = safe_float(ind.get("MACD.signal"))

    # Trend
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 0.6
        elif sma20 < sma50:
            score -= 0.6

    # RSI
    if rsi is not None:
        if rsi >= 55:
            score += 0.4
        elif rsi <= 45:
            score -= 0.4

    # MACD approx (macd - signal)
    if macd is not None and macds is not None:
        hist = macd - macds
        if hist > 0:
            score += 0.3
        elif hist < 0:
            score -= 0.3

    # Recommendation
    rec = (ind.get("Recommend.All") or "").upper()
    # tradingview-ta geeft vaak "BUY"/"SELL"/"NEUTRAL"/"STRONG_BUY"/"STRONG_SELL"
    if "STRONG_BUY" in rec:
        score += 0.7
    elif rec == "BUY":
        score += 0.4
    elif "STRONG_SELL" in rec:
        score -= 0.7
    elif rec == "SELL":
        score -= 0.4

    # Label
    if score >= 0.6:
        sentiment = "BULLISH"
        bias = "BUY BIAS"
    elif score <= -0.6:
        sentiment = "BEARISH"
        bias = "SELL BIAS"
    else:
        sentiment = "NEUTRAL"
        bias = "WAIT / NEUTRAL"

    return score, sentiment, bias


def build_handler(symbol: str, screener: str, exchange: str) -> TA_Handler:
    h = TA_Handler(
        symbol=symbol,
        screener=screener,
        exchange=exchange,
        interval=INTERVAL,
    )
    return h


def try_fetch_ta(name: str, cfg: Dict[str, str]) -> MarketResult:
    primary = (cfg.get("symbol", ""), cfg.get("screener", ""), cfg.get("exchange", ""))
    fallbacks = [s.strip() for s in (cfg.get("fallbacks", "") or "").split(",") if s.strip()]

    candidates: List[Tuple[str, str, str]] = [primary]

    # fallback logic:
    # - Als fallback een "EXCHANGE:SYMBOL" bevat, splitsen we
    # - Anders proberen we zelfde screener/exchange
    for fb in fallbacks:
        if ":" in fb:
            ex, sym = fb.split(":", 1)
            candidates.append((sym.strip(), cfg.get("screener", ""), ex.strip()))
        else:
            candidates.append((fb, cfg.get("screener", ""), cfg.get("exchange", "")))

    last_error = None

    for sym, screener, exch in candidates:
        if not sym or not screener or not exch:
            continue
        try:
            handler = build_handler(sym, screener, exch)
            analysis = handler.get_analysis()
            ind = analysis.indicators or {}

            score, sentiment, bias = calc_score(ind)

            last = safe_float(ind.get("close"))
            rsi = safe_float(ind.get("RSI"))
            sma20 = safe_float(ind.get("SMA20"))
            sma50 = safe_float(ind.get("SMA50"))

            macd = None
            m = safe_float(ind.get("MACD.macd"))
            s = safe_float(ind.get("MACD.signal"))
            if m is not None and s is not None:
                macd = m - s

            rec = (analysis.summary or {}).get("RECOMMENDATION")
            used = f"{exch}:{sym} ({screener})"

            return MarketResult(
                ok=True,
                used=used,
                sentiment=sentiment,
                bias=bias,
                score=score,
                last=last,
                rsi=rsi,
                sma20=sma20,
                sma50=sma50,
                macd=macd,
                rec=rec,
                error=None,
            )
        except Exception as e:
            last_error = str(e)

    return MarketResult(
        ok=False,
        used=f"{cfg.get('exchange')}:{cfg.get('symbol')} ({cfg.get('screener')})",
        sentiment="NEUTRAL",
        bias="NO DATA",
        score=0.0,
        last=None,
        rsi=None,
        sma20=None,
        sma50=None,
        macd=None,
        rec=None,
        error=last_error or "Unknown error",
    )


# =========================
# Auto-refresh
# =========================

def auto_refresh(seconds: int):
    # simpele auto-refresh zonder extra packages
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Page
# =========================

st.markdown('<div class="big-title">📈 Market Sentiment (TradingView)</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtle">Refresh elke 10 min • Interval: 15m • Bron: TradingView Technicals via tradingview-ta</p>',
    unsafe_allow_html=True
)

auto_refresh(REFRESH_SECONDS)

cols = st.columns(len(SYMBOLS))

for col, (market_name, cfg) in zip(cols, SYMBOLS.items()):
    with col:
        res = try_fetch_ta(market_name, cfg)

        pill_class = "neut"
        if res.sentiment == "BULLISH":
            pill_class = "bull"
        elif res.sentiment == "BEARISH":
            pill_class = "bear"

        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {market_name}")

        st.markdown(
            f'<span class="pill {pill_class}">{res.sentiment}</span> '
            f'<span class="muted"> &nbsp; {res.bias} &nbsp; • score {res.score:+.2f}</span>',
            unsafe_allow_html=True
        )

        if not res.ok:
            st.warning("Feed error / ticker niet gevonden op deze exchange.")
            st.caption(f"Probeerde: {res.used}")
            if res.error:
                st.caption(f"Error: {res.error}")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        # Metrics
        st.caption(f"Used: {res.used}")
        if res.last is not None:
            st.write(f"**Last:** {res.last}")
        st.write(
            f"**RSI:** {res.rsi if res.rsi is not None else '—'}  \n"
            f"**SMA20 / SMA50:** {res.sma20 if res.sma20 is not None else '—'} / {res.sma50 if res.sma50 is not None else '—'}  \n"
            f"**MACD hist:** {res.macd if res.macd is not None else '—'}  \n"
            f"**TV Rec:** {res.rec if res.rec else '—'}"
        )

        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown("### Wat past hier nog goed bij?")
st.markdown(
    """
- **Timeframe switcher (15m / 1h / 4h)** zodat je bias robuuster is  
- **Confluence score**: meerdere timeframes samen → 1 eind-score  
- **Risk note**: ATR/volatility indicator → “high vol / low vol”  
- **Session filter**: London / NY open highlight  
"""
)
