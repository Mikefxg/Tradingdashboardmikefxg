import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# =========================
# UnknownFX Pro Dashboard
# =========================

APP_TITLE = "🚀 UnknownFX Pro Dashboard"
DEFAULT_REFRESH_SECONDS = 120  # 2 minutes
DEFAULT_CHART_INTERVAL = "15"  # "1","5","15","60","240","D","W"

# ✅ Capital.com (TradingView) symbols (edit here if needed)
MARKETS = [
    {"name": "US100 (Nasdaq 100)", "symbol": "CAPITALCOM:US100", "asset": "index"},
    {"name": "US500 (S&P 500)", "symbol": "CAPITALCOM:US500", "asset": "index"},
    {"name": "US30 (Dow Jones)", "symbol": "CAPITALCOM:US30", "asset": "index"},
    {"name": "GOLD (XAUUSD)", "symbol": "CAPITALCOM:GOLD", "asset": "commodity"},
    {"name": "EURUSD", "symbol": "CAPITALCOM:EURUSD", "asset": "fx"},
    {"name": "DXY (US Dollar Index)", "symbol": "CAPITALCOM:DXY", "asset": "index"},
]

# TA intervals for MTF confluence (TradingView widget format)
TA_INTERVALS = [
    ("5m", "5m"),
    ("15m", "15m"),
    ("1h", "1h"),
    ("4h", "4h"),
    ("1D", "1D"),
]

TV_INTERVAL_MAP = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
    "D": "1D",
    "W": "1W",
}

# =========================
# PAGE CONFIG + STYLING
# =========================
st.set_page_config(page_title="UnknownFX Pro", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }
      .topbar {
        display:flex; justify-content:space-between; align-items:flex-end;
        gap: 20px; margin-bottom: 10px;
      }
      .title {
        font-size: 42px; font-weight: 900; letter-spacing: -0.8px;
        margin: 0; line-height: 1.05;
      }
      .subtitle {
        color: #6b7280; font-size: 14px; margin-top: 8px;
      }
      .badge {
        display:inline-block; padding: 6px 10px; border-radius: 999px;
        border: 1px solid #e5e7eb; background: #fafafa;
        font-size: 12px; color:#111827;
      }
      .section-title {
        font-size: 18px; font-weight: 800; margin: 8px 0 10px 0;
      }
      .market-title {
        font-size: 30px; font-weight: 900; margin: 14px 0 8px 0;
      }
      .small { color:#6b7280; font-size: 12px; margin-bottom: 12px; }
      .panel {
        padding: 14px 16px; border-radius: 16px;
        border: 1px solid #e5e7eb; background: #ffffff;
      }
      .panel-soft {
        padding: 14px 16px; border-radius: 16px;
        border: 1px solid #e5e7eb; background: #fafafa;
      }
      .divider { margin: 22px 0; border-top: 1px solid #eee; }
      .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      .grid3 { display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
      .mtf-row { display:grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
      @media(max-width: 1100px){
        .mtf-row { grid-template-columns: 1fr 1fr; }
        .grid2, .grid3 { grid-template-columns: 1fr; }
      }
      .hint { font-size: 12px; color:#6b7280; margin-top: 6px; }
      .footer { color:#6b7280; font-size: 12px; margin-top: 20px; }
      iframe { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.markdown("## ⚙️ Settings")

refresh_seconds = st.sidebar.slider(
    "Auto-refresh (seconds)",
    min_value=30,
    max_value=600,
    value=DEFAULT_REFRESH_SECONDS,
    step=10,
)

chart_interval = st.sidebar.selectbox(
    "Chart timeframe",
    options=["1", "5", "15", "60", "240", "D", "W"],
    index=["1", "5", "15", "60", "240", "D", "W"].index(DEFAULT_CHART_INTERVAL),
)

theme = st.sidebar.selectbox("Theme", options=["light", "dark"], index=0)
show_calendar = st.sidebar.toggle("Show Economic Calendar", value=True)
show_news = st.sidebar.toggle("Show Market News", value=False)
show_symbol_info = st.sidebar.toggle("Show Symbol Info cards", value=True)
show_mtf = st.sidebar.toggle("Show MTF Confluence (TA)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ✅ Watchlist")
for m in MARKETS:
    st.sidebar.write(f"• {m['name']}")

# =========================
# AUTO REFRESH (no extra libs)
# =========================
components.html(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {refresh_seconds * 1000});
    </script>
    """,
    height=0,
)

# =========================
# TRADINGVIEW WIDGET HELPERS
# =========================
def tv_symbol_info(symbol: str, height: int = 160):
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
      {{
        "symbol": "{symbol}",
        "width": "100%",
        "locale": "en",
        "colorTheme": "{theme}",
        "isTransparent": false
      }}
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

def tv_chart(symbol: str, interval_val: str, height: int = 760):
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tv_chart_{symbol.replace(':','_')}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{interval_val}",
          "timezone": "Etc/UTC",
          "theme": "{theme}",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "{'#0b1220' if theme=='dark' else '#f1f3f6'}",
          "enable_publishing": false,
          "allow_symbol_change": false,
          "hide_top_toolbar": false,
          "hide_side_toolbar": false,
          "withdateranges": true,
          "save_image": false,
          "details": true,
          "container_id": "tv_chart_{symbol.replace(':','_')}"
        }});
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

def tv_ta(symbol: str, ta_interval: str, height: int = 430, tabs: bool = True):
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "{ta_interval}",
        "width": "100%",
        "isTransparent": false,
        "height": "{height}",
        "symbol": "{symbol}",
        "showIntervalTabs": {str(tabs).lower()},
        "locale": "en",
        "colorTheme": "{theme}"
      }}
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

def tv_calendar(height: int = 760):
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-economic-calendar.js" async>
      {{
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "colorTheme": "{theme}",
        "isTransparent": false,
        "importanceFilter": "1,2,3",
        "currencyFilter": "USD,EUR"
      }}
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

def tv_news(height: int = 760):
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-news.js" async>
      {{
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "colorTheme": "{theme}",
        "isTransparent": false
      }}
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

# =========================
# HEADER
# =========================
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <div class="title">{APP_TITLE}</div>
        <div class="subtitle">
          Capital.com feeds via TradingView • MTF Confluence • Outlook • Calendar • Refresh: {refresh_seconds}s
        </div>
      </div>
      <div>
        <span class="badge">Last refresh: <b>{datetime.now().strftime('%H:%M:%S')}</b></span>
        <span class="badge">Timeframe: <b>{chart_interval}</b></span>
        <span class="badge">Theme: <b>{theme}</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =========================
# TOP TABS
# =========================
tabs = st.tabs(["📊 Dashboard", "🗓️ Calendar", "📰 News", "ℹ️ How to use"])

# -------------------------
# TAB: DASHBOARD
# -------------------------
with tabs[0]:
    st.markdown("<div class='section-title'>Markets</div>", unsafe_allow_html=True)
    st.caption("Elke market: 1 grote chart + Outlook + (optioneel) MTF confluence. Alles draait op officiële TradingView widgets (stabiel op Streamlit Cloud).")

    for m in MARKETS:
        st.markdown(f"<div class='market-title'>{m['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>Symbol: <b>{m['symbol']}</b> • Source: <b>Capital.com (TradingView)</b></div>", unsafe_allow_html=True)

        if show_symbol_info:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            tv_symbol_info(m["symbol"], height=170)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

        # Outlook block (single TA on selected timeframe)
        st.markdown("<div class='panel-soft'>", unsafe_allow_html=True)
        st.markdown("### 📌 Outlook (BUY / SELL / NEUTRAL)")
        st.caption("Dit is de snelste en meest betrouwbare “Bullish/Bearish/Neutral” indicator zonder API keys.")
        tv_ta(m["symbol"], TV_INTERVAL_MAP.get(chart_interval, "15m"), height=430, tabs=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # MTF Confluence (five TA widgets in a row)
        if show_mtf:
            st.markdown("")
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("### 🧠 MTF Confluence (5m → 1D)")
            st.caption("Check of meerdere timeframes dezelfde richting geven. Als 3+ timeframes BUY = bullish bias, 3+ SELL = bearish bias.")
            st.markdown("<div class='mtf-row'>", unsafe_allow_html=True)
            for label, ta_int in TA_INTERVALS:
                # each TA widget
                components.html(
                    f"""
                    <div style="border:1px solid #e5e7eb; border-radius:14px; padding:10px; background:{'#0b1220' if theme=='dark' else '#ffffff'};">
                      <div style="font-weight:800; margin-bottom:8px; color:{'#e5e7eb' if theme=='dark' else '#111827'};">{label}</div>
                      <div class="tradingview-widget-container">
                        <div class="tradingview-widget-container__widget"></div>
                        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
                        {{
                          "interval": "{ta_int}",
                          "width": "100%",
                          "isTransparent": false,
                          "height": "310",
                          "symbol": "{m["symbol"]}",
                          "showIntervalTabs": false,
                          "locale": "en",
                          "colorTheme": "{theme}"
                        }}
                        </script>
                      </div>
                    </div>
                    """,
                    height=360,
                    scrolling=False,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### 📈 Chart (Full width)")
        tv_chart(m["symbol"], chart_interval, height=820)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# -------------------------
# TAB: CALENDAR
# -------------------------
with tabs[1]:
    st.markdown("<div class='section-title'>Economic Calendar</div>", unsafe_allow_html=True)
    if show_calendar:
        st.caption("High/Medium/Low impact filter is enabled. Currency focus: USD + EUR.")
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        tv_calendar(height=820)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Calendar staat uit. Zet ‘Show Economic Calendar’ aan in de sidebar.")

# -------------------------
# TAB: NEWS
# -------------------------
with tabs[2]:
    st.markdown("<div class='section-title'>Market News</div>", unsafe_allow_html=True)
    if show_news:
        st.caption("Optioneel nieuws via TradingView (geen scraping).")
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        tv_news(height=820)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("News staat uit. Zet ‘Show Market News’ aan in de sidebar.")

# -------------------------
# TAB: HOW TO USE
# -------------------------
with tabs[3]:
    st.markdown("<div class='section-title'>How to use (simpel & effectief)</div>", unsafe_allow_html=True)
    st.markdown(
        """
**Snelle workflow (pro):**
1) Check **DXY** eerst → als DXY bullish is, drukt dat vaak op EURUSD en kan het GOLD beïnvloeden.  
2) Bekijk per market de **Outlook** op jouw timeframe (15m / 1h).  
3) Check **MTF Confluence** → als meerdere timeframes dezelfde richting geven, heb je hogere confidence.  
4) Open de chart en kijk naar structuur (highs/lows) + momentum.

**Tip voor “nog ziekere” versie (maar met keys):**
- Echte live prices + alerts + eigen score berekenen (RSI/MACD/ATR) kan 100% betrouwbaar met een price API (bijv. OANDA).
        """
    )

st.markdown(
    "<div class='footer'>UnknownFX Pro • Built on Streamlit + TradingView official widgets • No keys needed • Stable on Streamlit Cloud</div>",
    unsafe_allow_html=True,
)
