import os
import asyncio
import httpx
import pandas as pd
from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load API credentials from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
SYMBOL = "XAU/USD"
EXCHANGE = "OANDA"  # Matches TradingView OANDA Gold Spot

app = FastAPI()

class SignalOutput(BaseModel):
    action: str = Field(description="BUY, SELL, or HOLD")
    confidence: float
    summary: str

def fetch_twelve_data(interval: str, outputsize: int = 50) -> pd.DataFrame:
    """
    Fetches real-time OHLC candles for Spot Gold from Twelve Data (OANDA exchange).
    """
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&exchange={EXCHANGE}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    try:
        res = httpx.get(url, timeout=10.0).json()
    except Exception as e:
        print(f"HTTP Request failed ({interval}): {e}")
        return pd.DataFrame()

    if "values" not in res:
        print(f"Twelve Data API Error ({interval}): {res.get('message', res)}")
        return pd.DataFrame()

    df = pd.DataFrame(res["values"])
    if df.empty:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Rename Twelve Data lower-case keys to Title Case
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

def calculate_key_levels():
    df_daily = fetch_twelve_data(interval="1day", outputsize=5)
    df_1h = fetch_twelve_data(interval="1h", outputsize=30)

    # Early exit if data is missing or insufficient
    if df_daily.empty or df_1h.empty or len(df_daily) < 2 or len(df_1h) < 20:
        return None

    prev_day = df_daily.iloc[-2]
    high = float(prev_day['High'])
    low = float(prev_day['Low'])
    close = float(prev_day['Close'])

    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high

    swing_high = float(df_1h['High'].tail(20).max())
    swing_low = float(df_1h['Low'].tail(20).min())

    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2)
    }

def calculate_indicators(df: pd.DataFrame, stoch_k=14, stoch_d=3, smooth_k=3, ema_period=50):
    if df.empty or len(df) < max(stoch_k + stoch_d + smooth_k, ema_period):
        return None

    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    low_min = df['Low'].rolling(window=stoch_k).min()
    high_max = df['High'].rolling(window=stoch_k).max()
    fast_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%K'] = fast_k.rolling(window=smooth_k).mean()
    df['%D'] = df['%K'].rolling(window=stoch_d).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']),
        "stoch_cross_down": bool(prev['%K'] > prev['%D'] and latest['%K'] < latest['%D'])
    }
def get_mtf_data():
    """
    Fetches multi-timeframe OHLC data via Twelve Data REST API.
    """
    df_5m = fetch_twelve_data(interval="5min", outputsize=60)
    df_15m = fetch_twelve_data(interval="15min", outputsize=60)
    df_1h = fetch_twelve_data(interval="1h", outputsize=60)
    df_4h = fetch_twelve_data(interval="4h", outputsize=60)

    if df_5m.empty or df_15m.empty or df_1h.empty or df_4h.empty:
        return None

    return {
        "5m": calculate_indicators(df_5m),
        "15m": calculate_indicators(df_15m),
        "1h": calculate_indicators(df_1h),
        "4h": calculate_indicators(df_4h)
    }

def send_telegram_message(text: str):
    """
    Dispatches signal alerts directly to Telegram.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    httpx.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def analyze_and_alert():
    """
    Main loop function to check multi-timeframe alignment and ask Gemini.
    """
    print("Checking Twelve Data Multi-Timeframe Spot Gold Data...")
    mtf = get_mtf_data()
    levels = calculate_key_levels()

    if not mtf or not levels or not all([mtf["5m"], mtf["15m"], mtf["1h"], mtf["4h"]]):
        print("Waiting for valid data stream...")
        return

    price = mtf["5m"]["close"]

    prompt = f"""
Analyze this Full Multi-Timeframe (MTF) Alignment setup for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

KEY LEVELS & LIQUIDITY ZONES:
- 1H Swing High (Major Resistance): ${levels['swing_high']}
- Daily Pivot R1: ${levels['r1']}
- Central Pivot: ${levels['pivot']}
- Daily Pivot S1: ${levels['s1']}
- 1H Swing Low (Major Support): ${levels['swing_low']}

TECHNICAL METRICS:
- 4H Timeframe: Above 50 EMA = {mtf['4h']['is_above_ema']}
- 1H Timeframe: Above 50 EMA = {mtf['1h']['is_above_ema']}
- 15M Timeframe: Above 50 EMA = {mtf['15m']['is_above_ema']}
- 5M Timeframe: Above 50 EMA = {mtf['5m']['is_above_ema']}, Stoch %K = {mtf['5m']['stoch_k']:.1f}, Stoch Cross Up = {mtf['5m']['stoch_cross_up']}, Stoch Cross Down = {mtf['5m']['stoch_cross_down']}

Strategy Rules:
- BUY: 4H & 1H strictly Bullish (Price > 50 EMA), 5M pulls back and crosses up on Stochastic (14,3,3) while reclaiming 5M 50 EMA.
- SELL: 4H & 1H strictly Bearish (Price < 50 EMA), 5M pulls back and crosses down on Stochastic (14,3,3) while falling below 5M 50 EMA.
- HOLD: If timeframes conflict or lack 100% alignment.

Instructions for "summary" field:
If decision is BUY or SELL, set action to "BUY" or "SELL" and write the summary EXACTLY in this format:

🚨 TRADE SIGNAL ALERT

Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Entry Price: ${price:.2f}

Stop Loss (SL): $[Price]
Take Profit 1 (TP1): $[Price]
Take Profit 2 (TP2): $[Price]

📍 NEAREST KEY LEVELS:
• Resistance 2 (R2): ${levels['swing_high']} (1H Major High)
• Resistance 1 (R1): ${levels['r1']} (Daily Pivot R1)
---------------------------------------------
• Current Price:     ${price:.2f}
---------------------------------------------
• Support 1 (S1):    ${levels['s1']} (Daily Pivot S1)
• Support 2 (S2):    ${levels['swing_low']} (1H Major Low)

📊 Confluence Reasoning: [2-sentence explanation]

If decision is HOLD, set action to "HOLD" and summary to "HOLD".
"""

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=SignalOutput,
    )

    # Models to attempt in sequence if primary hits 503 capacity limit
    models_to_try = ['gemini-3-flash-preview', 'gemini-2.0-flash']
    response = None

    for model_name in models_to_try:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            break  # Success! Exit retry loop
        except Exception as e:
            print(f"Model {model_name} failed ({e}). Retrying fallback...")
            import time
            time.sleep(2)

    if not response:
        print("All Gemini model attempts failed this cycle.")
        return

    try:
        output = SignalOutput.model_validate_json(response.text)
        print(f"Decision: {output.action} ({output.confidence * 100:.0f}%)")

        if output.action in ["BUY", "SELL"]:
            send_telegram_message(output.summary)

    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
