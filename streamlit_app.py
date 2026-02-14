import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Trading Dashboard (TV)", page_icon="📊", layout="wide")

st.title("📊 Trading Dashboard")
st.caption("Free TradingView widgets • CFD/indices-style symbols • Live chart view")

# ---- Symbol presets (pas aan als je wil)
MARKETS = {
    "US100 / Nasdaq (CFD)": "OANDA:NAS100USD",
    "US30 / Dow (CFD)": "OANDA:US30USD",
    "SPX500 / S&P (CFD)": "OANDA:SPX500USD",
    "XAUUSD (Gold Spot)": "OANDA:XAUUSD",
    "EURUSD": "OANDA:EURUSD",
}

# Fallbacks als OANDA symbol bij jou niet laadt
FALLBACKS = {
    "US100 / Nasdaq (CFD)": ["FX:US100", "TVC:NDX", "NASDAQ:QQQ"],
    "US30 / Dow (CFD)": ["TVC:DJI", "AMEX:DIA"],
    "SPX500 / S&P (CFD)": ["TVC:SPX", "AMEX:SPY"],
    "XAUUSD (Gold Spot)": ["TVC:GOLD", "COMEX:GC1!", "AMEX:GLD"],
    "EURUSD": ["FX:EURUSD"],
}

def tv_symbol_info(symbol: str, height: int = 140) -> str:
    # Mini “symbol info” card (laat prijs + change zien in widget zelf)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
      {{
        "symbol": "{symbol}",
        "width": "100%",
        "locale": "en",
        "colorTheme": "light",
        "isTransparent": true
      }}
      </script>
    </div>
    """

def tv_advanced_chart(symbol: str, interval: str = "15", height: int = 520) -> str:
    # Grote interactieve chart
    return f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{interval}",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "save_image": false,
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """

def tv_news(symbol: str, height: int = 520) -> str:
    # News widget (op basis van symbool)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "symbol",
        "symbol": "{symbol}",
        "colorTheme": "light",
        "isTransparent": true,
        "displayMode": "regular",
        "width": "100%",
        "height": {height},
        "locale": "en"
      }}
      </script>
    </div>
    """

# ---- UI
selected_market = st.selectbox("Select market", list(MARKETS.keys()), index=0)
primary_symbol = MARKETS[selected_market]

with st.expander("Symbol not loading? Try a fallback", expanded=False):
    st.write("Als je chart leeg is: kies een fallback symbool hieronder.")
    fallback_list = FALLBACKS.get(selected_market, [])
    fallback_symbol = st.selectbox("Fallback symbol", ["(none)"] + fallback_list)
    if fallback_symbol != "(none)":
        active_symbol = fallback_symbol
    else:
        active_symbol = primary_symbol

# Top row: mini symbol cards
st.subheader("Live snapshot")
cols = st.columns(5)
symbols_order = list(MARKETS.values())

# Als user een fallback activeert, toon die ook als “selected”
for i, (name, sym) in enumerate(MARKETS.items()):
    with cols[i]:
        label = name
        show_sym = sym
        if name == selected_market:
            show_sym = active_symbol
        st.caption(label)
        components.html(tv_symbol_info(show_sym), height=120)

st.divider()

# Tabs: chart / news
tab1, tab2 = st.tabs(["📈 Chart", "📰 News"])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        interval = st.selectbox("Timeframe", ["1", "5", "15", "60", "240", "D"], index=2)
        st.caption(f"Symbol: {active_symbol}")
        components.html(tv_advanced_chart(active_symbol, interval=interval), height=560)
    with right:
        st.caption("Tip: Als een OANDA symbool niet werkt, kies een fallback in de dropdown hierboven.")
        st.markdown(
            """
**Veelgebruikte alternatieven**
- US100: `TVC:NDX` of `NASDAQ:QQQ`
- SPX: `TVC:SPX` of `AMEX:SPY`
- Gold: `TVC:GOLD` of `COMEX:GC1!`
- EURUSD: `FX:EURUSD`
            """
        )

with tab2:
    components.html(tv_news(active_symbol), height=560)
