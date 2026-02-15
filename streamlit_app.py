import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# =========================
# UnknownFX Dashboard (CLEAN)
# =========================

st.set_page_config(page_title="UnknownFX Dashboard", layout="wide")

# Auto refresh elke 2 minuten (120.000 ms)
st_autorefresh(interval=120_000, key="unknownfx_refresh")

st.title("🚀 UnknownFX Dashboard")
st.caption("Charts: TradingView (Capital.com) • Sentiment: price action + trend + momentum + volatility • Refresh: elke 2 min")

# -------------------------
# 1) TradingView embed helper
# -------------------------
def tradingview_embed(symbol: str, title: str, height: int = 420):
    """
    Embed TradingView Advanced Chart widget.
    symbol voorbeeld: "CAPITALCOM:US100", "CAPITALCOM:US500", "CAPITALCOM:US30", "CAPITALCOM:GOLD", "CAPITALCOM:EURUSD", "CAPITALCOM:DXY"
    """
    widget = f"""
    <div class="tradingview-widget-container">
      <div id="tv_{symbol.replace(':','_')}"></div>
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
        "allow_symbol_change": false,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "container_id": "tv_{symbol.replace(':','_')}"
      }});
      </script>
    </div>
    """
    st.subheader(title)
    st.components.v1.html(widget, height=height, scrolling=False)

# -------------------------
# 2) Indicators (sentiment engine)
# -------------------------
def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    val = 100 - (100 / (1 + rs))
    return float(val.iloc[-1]) if len(val) else float("nan")

def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr_val.iloc[-1]) if len(atr_val) else float("nan")

@st.cache_data(ttl=110)  # cache iets korter dan 2 min refresh
def fetch_ohlc(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance geeft soms kolomnamen in MultiIndex; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

def sentiment_from_price(df: pd.DataFrame) -> dict:
    """
    Score op basis van:
    - Trend: EMA20 vs EMA50
    - Momentum: RSI
    - Price position: close vs EMA20
    - Volatility filter: ATR% (voor 'confidence')
    """
    if df.empty or len(df) < 60:
        return {"state": "NEUTRAL", "score": 0.0, "rsi": np.nan, "atr_pct": np.nan, "last": np.nan, "ema20": np.nan, "ema50": np.nan}

    close = df["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    last = float(close.iloc[-1])
    r = rsi(close, 14)
    a = atr(df, 14)
    atr_pct = (a / last) * 100 if (a and last) else np.nan

    score = 0.0

    # Trend
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1.2
    elif ema20.iloc[-1] < ema50.iloc[-1]:
        score -= 1.2

    # Price vs EMA20
    if last > ema20.iloc[-1]:
        score += 0.6
    else:
        score -= 0.6

    # RSI zones
    if r >= 60:
        score += 0.9
    elif r <= 40:
        score -= 0.9
    else:
        # licht momentum in mid-zone
        score += (r - 50) / 50 * 0.4  # -0.4..+0.4

    # Confidence: hoge ATR% = onrustig -> minder confident
    # (dit verandert score niet, maar we tonen het)
    if score >= 1.2:
        state = "BULLISH"
    elif score <= -1.2:
        state = "BEARISH"
    else:
        state = "NEUTRAL"

    return {
        "state": state,
        "score": float(score),
        "rsi": float(r),
        "atr_pct": float(atr_pct),
        "last": last,
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
    }

def badge(state: str) -> str:
    if state == "BULLISH":
        return "🟢 BULLISH"
    if state == "BEARISH":
        return "🔴 BEARISH"
    return "⚪ NEUTRAL"

# -------------------------
# 3) Markets config
# -------------------------
# TradingView symbols (Capital.com)
TV_MARKETS = [
    ("CAPITALCOM:US100", "US100 (Nasdaq CFD)"),
    ("CAPITALCOM:US500", "US500 (S&P CFD)"),
    ("CAPITALCOM:US30",  "US30 (Dow CFD)"),
    ("CAPITALCOM:GOLD",  "GOLD (XAUUSD)"),
    ("CAPITALCOM:EURUSD","EURUSD"),
    ("CAPITALCOM:DXY",   "DXY (US Dollar Index)"),
]

# Data tickers (gratis proxies) voor sentiment engine
# (zodat het zonder keys werkt)
DATA_TICKERS = {
    "US100 (Nasdaq CFD)": "QQQ",
    "US500 (S&P CFD)": "SPY",
    "US30 (Dow CFD)": "DIA",
    "GOLD (XAUUSD)": "GC=F",
    "EURUSD": "EURUSD=X",
    "DXY (US Dollar Index)": "DX=F",
}

# -------------------------
# 4) Top summary cards
# -------------------------
st.markdown("### 📌 Market Outlook (2-min refresh)")

cards = st.columns(6)
results = []

for i, (_, name) in enumerate(TV_MARKETS):
    data_ticker = DATA_TICKERS.get(name)
    df = fetch_ohlc(data_ticker) if data_ticker else pd.DataFrame()
    s = sentiment_from_price(df)
    results.append([name, data_ticker, s["state"], s["score"], s["last"], s["rsi"], s["atr_pct"]])

    with cards[i]:
        st.markdown(f"**{name}**")
        st.markdown(badge(s["state"]))
        st.markdown(f"**Bias:** {'BUY' if s['state']=='BULLISH' else 'SELL' if s['state']=='BEARISH' else 'WAIT'}")
        st.caption(f"score: {s['score']:+.2f}")
        if not math.isnan(s["last"]):
            st.caption(f"last: {s['last']:.5f}" if "EURUSD" in name else f"last: {s['last']:.2f}")
        if not math.isnan(s["rsi"]):
            st.caption(f"RSI: {s['rsi']:.1f}")
        if not math.isnan(s["atr_pct"]):
            st.caption(f"Vol (ATR%): {s['atr_pct']:.2f}%")

st.divider()

# -------------------------
# 5) Details table
# -------------------------
st.markdown("### 📊 Details (sentiment engine)")
df_out = pd.DataFrame(
    results,
    columns=["Market", "Proxy Ticker", "Sentiment", "Score", "Last", "RSI", "ATR%"]
)
st.dataframe(df_out, use_container_width=True, hide_index=True)

st.info(
    "Let op: sentiment wordt berekend met gratis proxy-tickers (yfinance). "
    "Charts zijn wél Capital.com op TradingView. "
    "Wil je sentiment ook 100% op CFD/spot data? Dan heb je een broker/API key nodig."
)

st.divider()

# -------------------------
# 6) Charts section (TradingView Capital.com)
# -------------------------
st.markdown("### 📈 TradingView Charts (Capital.com)")

# 2 rijen van 3 charts
row1 = st.columns(3)
row2 = st.columns(3)

for idx, (tv_symbol, title) in enumerate(TV_MARKETS):
    target = row1[idx] if idx < 3 else row2[idx - 3]
    with target:
        tradingview_embed(tv_symbol, title, height=420)

st.divider()

# -------------------------
# 7) Extra: DXY correlation hint (simple)
# -------------------------
st.markdown("### 🔁 Extra: DXY vs EURUSD (quick bias check)")

dxy = fetch_ohlc("DX=F")
eur = fetch_ohlc("EURUSD=X")

if (not dxy.empty) and (not eur.empty):
    # Align closes
    joined = pd.concat([dxy["Close"].rename("DXY"), eur["Close"].rename("EURUSD")], axis=1).dropna()
    if len(joined) > 50:
        ret = joined.pct_change().dropna()
        corr = float(ret["DXY"].corr(ret["EURUSD"]))
        st.write(f"**Intraday return correlation (DXY vs EURUSD):** `{corr:+.2f}` (meestal negatief)")
    else:
        st.write("Niet genoeg data voor correlatie.")
else:
    st.write("Kon DXY/EURUSD proxy data niet ophalen.")

st.caption("Geen financieel advies. Gebruik dit als indicatie + bevestig altijd met je eigen analysis.")
