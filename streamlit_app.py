import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# =========================
# App settings
# =========================
st.set_page_config(page_title="UnknownFX Dashboard", layout="wide")

REFRESH_SECONDS = 120  # 2 minutes
INTERVAL = "5m"        # yfinance interval
LOOKBACK_DAYS = "5d"   # intraday history window

# TradingView tickers (Capital.com)
TV_TICKERS = {
    "US100 (Nasdaq CFD)": "CAPITALCOM:US100",
    "US500 (S&P CFD)": "CAPITALCOM:US500",
    "US30 (Dow CFD)": "CAPITALCOM:US30",
    "GOLD (XAUUSD)": "CAPITALCOM:GOLD",
    "EURUSD": "CAPITALCOM:EURUSD",
    "DXY (Dollar Index)": "CAPITALCOM:DXY",
}

# Price data tickers (yfinance fallback/source for calculations)
# (These are widely supported and stable on Streamlit Cloud.)
YF_TICKERS = {
    "US100 (Nasdaq CFD)": "^NDX",        # Nasdaq 100 index
    "US500 (S&P CFD)": "^GSPC",          # S&P 500 index
    "US30 (Dow CFD)": "^DJI",            # Dow Jones
    "GOLD (XAUUSD)": "XAUUSD=X",         # Gold spot (may work); if not, it will fallback to GC=F
    "EURUSD": "EURUSD=X",
    "DXY (Dollar Index)": "DX-Y.NYB",
}

# =========================
# Helpers
# =========================
def safe_download(ticker):
    # yfinance intraday can fail for some tickers; try fallback for gold
    try:
        df = yf.download(ticker, period=LOOKBACK_DAYS, interval=INTERVAL, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            # Sometimes yfinance returns multiindex columns
            df.columns = [c[0] for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def sentiment_label(score):
    # score roughly between -3 and +3
    if score >= 1.0:
        return "BULLISH"
    if score <= -1.0:
        return "BEARISH"
    return "NEUTRAL"

def score_market(df):
    """
    Score based on:
    - EMA trend (20/50)
    - RSI
    - last candle direction
    - volatility (ATR%): higher volatility -> reduce confidence
    """
    if df is None or df.empty:
        return None

    # Ensure needed columns
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return None

    df = df.dropna().copy()
    if len(df) < 60:
        return None

    close = df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = compute_rsi(close, 14)
    atr = compute_atr(df, 14)

    last = df.iloc[-1]
    last_close = float(last["Close"])
    prev_close = float(df["Close"].iloc[-2])

    # Trend component
    trend = 0.0
    if ema20.iloc[-1] > ema50.iloc[-1]:
        trend += 1.0
    else:
        trend -= 1.0

    # RSI component
    rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    rsi_score = 0.0
    if rsi_val >= 55:
        rsi_score += 1.0
    elif rsi_val <= 45:
        rsi_score -= 1.0

    # Momentum / candle
    mom = 0.0
    if last_close > prev_close:
        mom += 0.5
    elif last_close < prev_close:
        mom -= 0.5

    # Volatility penalty (ATR%)
    atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
    atr_pct = (atr_val / last_close) * 100 if last_close else 0.0

    # If volatility is high, dampen score a bit (more "uncertain")
    damp = 1.0
    if atr_pct >= 0.8:
        damp = 0.7
    if atr_pct >= 1.2:
        damp = 0.55

    raw_score = (trend + rsi_score + mom) * damp

    return {
        "score": float(raw_score),
        "label": sentiment_label(raw_score),
        "last": last_close,
        "rsi": rsi_val,
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "atr_pct": float(atr_pct),
    }

def tv_widget(symbol, height=380):
    # Lightweight TradingView "symbol overview" widget
    # Works without API keys. Uses TradingView’s embedded widget.
    html = f"""
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
          "hide_side_toolbar": false,
          "allow_symbol_change": false,
          "container_id": "tv_{symbol.replace(':','_')}"
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=height)

def send_telegram(msg):
    # Optional: only works if you set secrets
    token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Telegram secrets not set"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=10)
        if r.status_code == 200:
            return True, "sent"
        return False, f"telegram error {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, str(e)

def maybe_alert(name, label, score, last):
    # Alert when sentiment changes
    key = f"prev_label::{name}"
    prev = st.session_state.get(key)
    if prev is None:
        st.session_state[key] = label
        return

    if prev != label:
        st.session_state[key] = label
        msg = f"UnknownFX Alert ✅\n{name}\nSentiment: {prev} → {label}\nScore: {score:+.2f}\nLast: {last}"
        send_telegram(msg)

@st.cache_data(ttl=REFRESH_SECONDS)
def load_market_calc_data(name):
    t = YF_TICKERS.get(name)
    df = safe_download(t)

    # Gold fallback if XAUUSD=X fails
    if name.startswith("GOLD") and (df is None or df.empty):
        df = safe_download("GC=F")

    return df

# =========================
# Header
# =========================
st.title("🚀 UnknownFX Dashboard")
st.caption("MTF Confluence • Key Levels • Confidence % • Alerts • DXY Correlation • Trend Probability")
st.caption(f"Auto refresh: every {REFRESH_SECONDS//60} min • Calc source: yfinance • Charts: TradingView (Capital.com tickers)")

# Auto refresh (simple + stable)
now = int(time.time())
st.session_state["__last_refresh__"] = st.session_state.get("__last_refresh__", now)
if now - st.session_state["__last_refresh__"] >= REFRESH_SECONDS:
    st.session_state["__last_refresh__"] = now
    st.rerun()

# Sidebar controls
st.sidebar.header("Settings")
show_charts = st.sidebar.toggle("Show TradingView charts", value=True)
enable_alerts = st.sidebar.toggle("Enable Telegram alerts", value=False)
st.sidebar.write("Telegram werkt pas als je secrets zet.")

st.sidebar.divider()
st.sidebar.write("Instrument mapping (TradingView / Calc):")
for k in TV_TICKERS:
    st.sidebar.write(f"- {k}: {TV_TICKERS[k]} / {YF_TICKERS[k]}")

# =========================
# Dashboard grid
# =========================
names = list(TV_TICKERS.keys())
cols = st.columns(3)

for i, name in enumerate(names):
    with cols[i % 3]:
        st.subheader(name)

        # 1) Calculate sentiment
        df = load_market_calc_data(name)
        result = score_market(df)

        if result is None:
            st.error("No calc data available (yfinance).")
        else:
            label = result["label"]
            score = result["score"]

            if label == "BULLISH":
                st.success(f"🟢 {label}")
                bias = "BUY BIAS"
            elif label == "BEARISH":
                st.error(f"🔴 {label}")
                bias = "SELL BIAS"
            else:
                st.warning(f"🟡 {label}")
                bias = "NEUTRAL"

            st.markdown(f"### {bias}")
            st.metric("Score", f"{score:+.2f}")
            st.write(f"Last: **{result['last']:.5f}**" if "EURUSD" in name else f"Last: **{result['last']:.2f}**")
            st.write(f"RSI: **{result['rsi']:.1f}**")
            st.write(f"EMA20 / EMA50: **{result['ema20']:.2f} / {result['ema50']:.2f}**")
            st.write(f"Volatility (ATR%): **{result['atr_pct']:.2f}%**")

            # Alerts on sentiment change
            if enable_alerts:
                maybe_alert(name, label, score, result["last"])

        # 2) Charts
        if show_charts:
            with st.expander("Chart (TradingView / Capital.com)", expanded=False):
                tv_widget(TV_TICKERS[name], height=420)

# =========================
# Footer / Telegram help
# =========================
with st.expander("📩 Telegram setup (optioneel)"):
st.title("UnknownFX Dashboard")
**Bot token krijgen:**
1) Open Telegram → zoek **@BotFather**
2) `/newbot` → kies naam + username
3) BotFather geeft jou een **TOKEN** (iets als `123456:ABC...`)

**Chat ID krijgen:**
- Makkelijkste: stuur eerst een bericht naar je bot, en gebruik daarna een tool zoals **@userinfobot** om je chat_id te zien  
  óf gebruik een eenvoudige "getUpdates" call (kan ik je ook geven als je wil).

**Streamlit Cloud secrets zetten:**
- In Streamlit Cloud → App → **Settings → Secrets**
Plaats:
```toml
TELEGRAM_BOT_TOKEN="jouw_token"
TELEGRAM_CHAT_ID="jouw_chat_id"
