import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(layout="wide")

APP_TITLE = "🚀 UnknownFX Pro Dashboard"

REFRESH = 120
INTERVAL = "15"

MARKETS = [
    {"name": "US100 (Nasdaq)", "chart": "CAPITALCOM:US100", "ta": "NASDAQ:NDX"},
    {"name": "US500 (S&P500)", "chart": "CAPITALCOM:US500", "ta": "SP:SPX"},
    {"name": "US30 (Dow)", "chart": "CAPITALCOM:US30", "ta": "DJI"},
    {"name": "GOLD (XAUUSD)", "chart": "CAPITALCOM:GOLD", "ta": "OANDA:XAUUSD"},
    {"name": "EURUSD", "chart": "CAPITALCOM:EURUSD", "ta": "FX:EURUSD"},
    {"name": "DXY", "chart": "CAPITALCOM:DXY", "ta": "TVC:DXY"},
]

TA_INTERVALS = ["5m", "15m", "1h", "4h", "1D"]

components.html(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {REFRESH*1000});
    </script>
    """,
    height=0,
)

st.title(APP_TITLE)
st.caption("Capital.com charts + TradingView TA • Auto refresh 2 min")

# ---------- widgets ----------

def tv_chart(symbol, interval="15"):
    html = f"""
    <div id="chart_{symbol}" style="height:750px;"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "container_id": "chart_{symbol}"
    }});
    </script>
    """
    components.html(html, height=760)

def tv_ta(symbol, interval):
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "{interval}",
        "width": "100%",
        "height": "280",
        "symbol": "{symbol}",
        "locale": "en",
        "colorTheme": "light"
      }}
      </script>
    </div>
    """
    components.html(html, height=300)

# ---------- layout ----------

for m in MARKETS:
    st.subheader(m["name"])

    st.markdown("**Outlook (15m)**")
    tv_ta(m["ta"], "15m")

    st.markdown("**MTF Confluence**")
    cols = st.columns(5)
    for i, tf in enumerate(TA_INTERVALS):
        with cols[i]:
            tv_ta(m["ta"], tf)

    st.markdown("**Chart**")
    tv_chart(m["chart"], INTERVAL)

    st.divider()
