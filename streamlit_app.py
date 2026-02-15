import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import streamlit as st
from tradingview_ta import TA_Handler, Interval

# =========================
# Page / App config
# =========================
st.set_page_config(
    page_title="UnknownFX Dashboard",
    page_icon="🚀",
    layout="wide",
)

REFRESH_SECONDS = 120  # 2 minutes

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .subtle { color: rgba(250,250,250,0.65); font-size: 0.95rem; }
      .card {
        padding: 18px 18px 14px 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
      }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
      }
      .bull { color: #22c55e; }
      .bear { color: #ef4444; }
      .neut { color: #eab308; }
      .big {
        font-size: 2.1rem;
        font-weight: 900;
        margin: 6px 0 2px 0;
      }
      .metric {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.75);
        margin-top: 8px;
      }
      iframe { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 UnknownFX Dashboard")
st.markdown(
    '<div class="subtle">Pro++ • 1 chart per rij • Outlook op basis van TradingView TA • Auto refresh elke 2 minuten</div>',
    unsafe_allow_html=True,
)

# =========================
# Market config
# =========================
@dataclass
class Market:
    key: str
    name: str
    # Chart (Capital.com symbols as you showed)
    chart_symbol: str
    # TA (use stable sources for analysis; Capital.com often fails for TA)
    ta_symbol: str
    ta_screener: str
    ta_exchange: str

MARKETS: List[Market] = [
    Market("US100", "US100 (Nasdaq CFD)", "CAPITALCOM:US100", "NASDAQ:NDX", "america", "NASDAQ"),
    Market("US500", "US500 (S&P CFD)", "CAPITALCOM:US500", "SP:SPX", "america", "SP"),
    Market("XAUUSD", "GOLD (XAUUSD)", "CAPITALCOM:GOLD", "OANDA:XAUUSD", "forex", "OANDA"),
    Market("EURUSD", "EURUSD", "CAPITALCOM:EURUSD", "OANDA:EURUSD", "forex", "OANDA"),
    Market("DXY", "DXY (Dollar Index)", "CAPITALCOM:DXY", "TVC:DXY", "america", "TVC"),
]

# =========================
# Helpers
# =========================
def tv_chart_embed(symbol: str, interval: str = "15", height: int = 640) -> str:
    """
    TradingView Advanced Chart widget
    interval: "1", "5", "15", "60", "240", "D"
    """
    # NOTE: widget config uses JSON. Keep it simple & stable.
    return f"""
    <iframe
        src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol.replace(':','_')}"
        style="width: 100%; height: {height}px;"
        frameborder="0"
        allowtransparency="true"
        scrolling="no"
        allowfullscreen="true">
    </iframe>
    <script type="text/javascript">
      (function() {{
        var s = document.createElement('script');
        s.type = 'text/javascript';
        s.async = true;
        s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
        s.innerHTML = {{}};
      }})();
    </script>
    """

def embed_advanced_chart(symbol: str, interval: str = "15", height: int = 640):
    # Use official embed script version (more reliable than URL-only iframe).
    config = {
        "symbol": symbol,
        "interval": interval,
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#0b1220",
        "enable_publishing": False,
        "allow_symbol_change": False,
        "hide_side_toolbar": False,
        "withdateranges": True,
        "details": True,
        "hotlist": False,
        "calendar": False,
        "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
        "support_host": "https://www.tradingview.com"
    }

    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_{symbol.replace(':','_')}" style="height:{height}px; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{config['symbol']}",
          "interval": "{config['interval']}",
          "timezone": "{config['timezone']}",
          "theme": "{config['theme']}",
          "style": "{config['style']}",
          "locale": "{config['locale']}",
          "toolbar_bg": "{config['toolbar_bg']}",
          "enable_publishing": {str(config['enable_publishing']).lower()},
          "allow_symbol_change": {str(config['allow_symbol_change']).lower()},
          "hide_side_toolbar": {str(config['hide_side_toolbar']).lower()},
          "withdateranges": {str(config['withdateranges']).lower()},
          "details": {str(config['details']).lower()},
          "hotlist": {str(config['hotlist']).lower()},
          "calendar": {str(config['calendar']).lower()},
          "studies": {config['studies']},
          "container_id": "tv_{symbol.replace(':','_')}",
          "support_host": "{config['support_host']}"
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=height + 40, scrolling=False)

@st.cache_data(ttl=REFRESH_SECONDS)
def get_ta(exchange: str, screener: str, symbol: str, interval: Interval) -> Dict[str, Any]:
    handler = TA_Handler(
        symbol=symbol.split(":")[-1] if ":" in symbol else symbol,
        exchange=exchange,
        screener=screener,
        interval=interval,
    )
    analysis = handler.get_analysis()
    return {
        "summary": analysis.summary,
        "indicators": analysis.indicators,
        "oscillators": analysis.oscillators,
        "moving_averages": analysis.moving_averages,
    }

def outlook_label(summary: Dict[str, Any]) -> str:
    rec = (summary or {}).get("RECOMMENDATION", "NEUTRAL")
    # Normalize
    if rec in ("STRONG_BUY", "BUY"):
        return "BULLISH"
    if rec in ("STRONG_SELL", "SELL"):
        return "BEARISH"
    return "NEUTRAL"

def outlook_score(summary: Dict[str, Any]) -> float:
    # Use BUY/SELL counts for a simple score
    buy = float(summary.get("BUY", 0) + summary.get("STRONG_BUY", 0))
    sell = float(summary.get("SELL", 0) + summary.get("STRONG_SELL", 0))
    neut = float(summary.get("NEUTRAL", 0))
    denom = max(buy + sell + neut, 1.0)
    return (buy - sell) / denom  # -1..+1

def pill_html(label: str, score: float) -> str:
    if label == "BULLISH":
        cls = "bull"
        icon = "🟢"
    elif label == "BEARISH":
        cls = "bear"
        icon = "🔴"
    else:
        cls = "neut"
        icon = "🟡"
    return f"""
    <div class="card">
      <div class="pill {cls}">{icon} {label}</div>
      <div class="big">{'BUY BIAS' if label=='BULLISH' else ('SELL BIAS' if label=='BEARISH' else 'WAIT / NEUTRAL')}</div>
      <div class="metric">Outlook score: <b>{score:+.2f}</b> ( -1 bearish → +1 bullish )</div>
    </div>
    """

# =========================
# Controls
# =========================
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    interval_choice = st.selectbox("TA interval", ["5m", "15m", "1h", "4h", "1D"], index=1)
with c2:
    chart_interval = st.selectbox("Chart interval", ["1", "5", "15", "60", "240", "D"], index=2)
with c3:
    st.caption("Tip: TA interval bepaalt bias, chart interval bepaalt visualisatie.")

interval_map = {
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1D": Interval.INTERVAL_1_DAY,
}

st.divider()

# =========================
# Main layout (1 per row)
# =========================
for m in MARKETS:
    st.subheader(m.name)

    # Left: chart, Right: outlook
    left, right = st.columns([2.2, 1])

    with left:
        # Big chart
        embed_advanced_chart(m.chart_symbol, interval=chart_interval, height=680)

    with right:
        try:
            ta = get_ta(m.ta_exchange, m.ta_screener, m.ta_symbol, interval_map[interval_choice])
            summary = ta["summary"]
            indicators = ta["indicators"]

            label = outlook_label(summary)
            score = outlook_score(summary)

            st.markdown(pill_html(label, score), unsafe_allow_html=True)

            # Extra quick metrics
            rsi = indicators.get("RSI")
            macd = indicators.get("MACD.macd")
            macd_sig = indicators.get("MACD.signal")
            sma20 = indicators.get("SMA20")
            sma50 = indicators.get("SMA50")
            last = indicators.get("close")

            st.markdown(
                f"""
                <div class="card" style="margin-top:12px;">
                  <div class="metric"><b>Last:</b> {last if last is not None else "—"}</div>
                  <div class="metric"><b>RSI:</b> {rsi if rsi is not None else "—"}</div>
                  <div class="metric"><b>SMA20 / SMA50:</b> {sma20 if sma20 is not None else "—"} / {sma50 if sma50 is not None else "—"}</div>
                  <div class="metric"><b>MACD:</b> {macd if macd is not None else "—"} | <b>Signal:</b> {macd_sig if macd_sig is not None else "—"}</div>
                  <div class="metric"><b>TV Rec:</b> {summary.get("RECOMMENDATION","—")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"TA/Outlook error voor {m.key}: {e}")

    st.divider()

# =========================
# Auto refresh
# =========================
st.caption(f"Auto refresh: elke {REFRESH_SECONDS//60} min • (Streamlit herlaadt automatisch via cache TTL)")
time.sleep(0.1)
