import os
import json
import asyncio
from contextlib import asynccontextmanager
import httpx
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from openai import OpenAI

# Load API credentials from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Initialize Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

SYMBOL = "XAU/USD"
EXCHANGE = "OANDA"

# Cooldown Tracker to Prevent Signal Spam
LAST_SIGNAL_ACTION = "HOLD"
LAST_SIGNAL_PRICE = 0.0


class SignalOutput(BaseModel):
    action: str = Field(description="BUY, SELL, or HOLD")
    confidence: float
    summary: str


def send_telegram_message(message: str):
    """Sends a formatted notification to your Telegram Channel/Chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = httpx.post(url, json=payload, timeout=10.0)
        if res.status_code == 200:
            print("Telegram alert sent successfully.")
        else:
            print(f"Failed to send Telegram alert: {res.text}")
    except Exception as e:
        print(f"Telegram HTTP Error: {e}")


def fetch_twelve_data(interval: str, outputsize: int = 100) -> pd.DataFrame:
    """Fetches real-time OHLC candles for Spot Gold from Twelve Data."""
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

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def calculate_metrics(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50):
    """Calculates EMA 50, VWAP, Stoch RSI, Swing Levels, and ChoCH conditions."""
    if df.empty or len(df) < (rsi_period + stoch_period + ema_period):
        return None

    # 1. EMA 50
    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # 2. VWAP (Session Approximation)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    # Approximating VWAP over rolling window
    df['VWAP'] = (typical_price * df['Close']).rolling(window=20).sum() / df['Close'].rolling(window=20).sum()

    # 3. RSI & Stochastic RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi_min = df['RSI'].rolling(window=stoch_period).min()
    rsi_max = df['RSI'].rolling(window=stoch_period).max()
    stoch_rsi = (df['RSI'] - rsi_min) / (rsi_max - rsi_min)

    df['%K'] = stoch_rsi.rolling(window=k_period).mean() * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Swing levels (Tight 10-candle window for 5M/15M)
    swing_high = float(df['High'].tail(10).max())
    swing_low = float(df['Low'].tail(10).min())

    # Structural ChoCH condition: Price below BOTH EMA 50 and VWAP
    is_choch_bearish = bool(latest['Close'] < latest['EMA_50'] and latest['Close'] < latest['VWAP'])
    is_choch_bullish = bool(latest['Close'] > latest['EMA_50'] and latest['Close'] > latest['VWAP'])

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "vwap": float(latest['VWAP']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "is_above_vwap": bool(latest['Close'] > latest['VWAP']),
        "choch_bearish": is_choch_bearish,
        "choch_bullish": is_choch_bullish,
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']),
        "stoch_cross_down": bool(prev['%K'] > prev['%D'] and latest['%K'] < latest['%D']),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "is_green_candle": bool(latest['Close'] > latest['Open'])
    }


def get_mtf_data():
    df_5m = fetch_twelve_data("5min", outputsize=100)
    df_15m = fetch_twelve_data("15min", outputsize=100)
    df_1h = fetch_twelve_data("1h", outputsize=100)

    if df_5m.empty or df_15m.empty or df_1h.empty:
        return None

    return {
        "5m": calculate_metrics(df_5m),
        "15m": calculate_metrics(df_15m),
        "1h": calculate_metrics(df_1h)
    }


def analyze_and_alert():
    global LAST_SIGNAL_ACTION, LAST_SIGNAL_PRICE

    print("Checking Twelve Data Multi-Timeframe Spot Gold Data...")
    mtf = get_mtf_data()

    if not mtf or not mtf["5m"] or not mtf["15m"] or not mtf["1h"]:
        print("Waiting for valid data stream...")
        return

    price = mtf["5m"]["close"]

    prompt = f"""
Analyze this 1H / 15M / 5M Multi-Timeframe Strategy for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

5M TIMEFRAME (Primary Execution Timeframe):
- Price: ${mtf['5m']['close']:.2f} | 5M EMA 50: ${mtf['5m']['ema_50']:.2f} | 5M VWAP: ${mtf['5m']['vwap']:.2f}
- Above EMA 50: {mtf['5m']['is_above_ema']} | Above VWAP: {mtf['5m']['is_above_vwap']}
- 5M Bearish ChoCH (Below EMA & VWAP): {mtf['5m']['choch_bearish']}
- 5M Bullish ChoCH (Above EMA & VWAP): {mtf['5m']['choch_bullish']}
- 5M Stoch RSI %K: {mtf['5m']['stoch_k']:.1f} | %D: {mtf['5m']['stoch_d']:.1f}
- 5M Stoch Cross Up: {mtf['5m']['stoch_cross_up']} | Cross Down: {mtf['5m']['stoch_cross_down']}
- 5M Candle Green: {mtf['5m']['is_green_candle']}
- Tight 5M Swing High: ${mtf['5m']['swing_high']:.2f} | Swing Low: ${mtf['5m']['swing_low']:.2f}

15M & 1H CONTEXT:
- 15M Above EMA 50: {mtf['15m']['is_above_ema']} | 15M Swing Low: ${mtf['15m']['swing_low']:.2f}
- 1H Above EMA 50: {mtf['1h']['is_above_ema']}

CHANGE OF CHARACTER (ChoCH) & EXECUTION RULES:
1. STRICT ChoCH RULE:
   - If 5M Bearish ChoCH is TRUE (Price closed below BOTH 5M EMA 50 & VWAP), DO NOT ISSUE ANY BUY SIGNALS! Force action to HOLD or SELL.
   - If 5M Bullish ChoCH is TRUE (Price closed above BOTH 5M EMA 50 & VWAP), DO NOT ISSUE ANY SELL SIGNALS! Force action to HOLD or BUY.

2. 5M BUY ENTRY:
   - 5M Price MUST be ABOVE BOTH 5M EMA 50 AND VWAP.
   - 5M Stoch RSI crosses UP from Oversold (< 25).
   - 5M Candle MUST be Green.

3. 5M SELL ENTRY:
   - 5M Price MUST be BELOW BOTH 5M EMA 50 AND VWAP.
   - 5M Stoch RSI crosses DOWN from Overbought (> 75).
   - 5M Candle MUST be Red.

4. TIGHT STOP LOSS (SL) CALCULATION:
   - BUY SL = Minimum of (5M Swing Low, 15M Swing Low) - $1.20 buffer.
   - SELL SL = Maximum of (5M Swing High, 15M Swing High) + $1.20 buffer.
   - TP1 = Entry Price +/- (1.5 * Risk)
   - TP2 = Entry Price +/- (2.5 * Risk)

OUTPUT REQUIREMENTS:
You MUST respond strictly with valid JSON with these exact key fields:
- "action": "BUY", "SELL", or "HOLD"
- "confidence": float between 0.0 and 1.0
- "summary": string formatted alert or "HOLD"

Format for "summary" when action is BUY or SELL:
STOCH RSI & ChoCH SIGNAL (5M EXECUTION)

Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Entry Price: ${price:.2f}

Stop Loss (SL): $[Calculated Tight SL]
Take Profit 1 (TP1): $[Calculated TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[Calculated TP2] (1:2.5 RRR)

INDICATOR METRICS:
- 5M EMA 50: ${mtf['5m']['ema_50']:.2f}
- 5M VWAP: ${mtf['5m']['vwap']:.2f}
- 5M Stoch RSI: {mtf['5m']['stoch_k']:.1f}

Reasoning: [2-sentence explanation confirming VWAP/EMA respect and ChoCH status]
"""

    output = None

    # Step 1: Groq Primary (Llama 3.3 70B)
    if groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading model. Output JSON strictly matching the requested keys."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw_text = res.choices[0].message.content
            output = SignalOutput.model_validate_json(raw_text)
            print(f"[Groq Llama 3.3] Decision: {output.action} ({output.confidence * 100:.0f}%)")
        except Exception as e:
            print(f"Groq API call failed: {e}. Falling back to Gemini...")

    # Step 2: Gemini Fallback
    if not output and genai_client:
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=SignalOutput,
            )
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )
            output = SignalOutput.model_validate_json(response.text)
            print(f"[Gemini 2.0] Decision: {output.action} ({output.confidence * 100:.0f}%)")
        except Exception as e:
            print(f"Gemini API call failed: {e}")

    if not output:
        print("All AI model attempts failed this cycle.")
        return

    # Spam Prevention Filter
    if output.action in ["BUY", "SELL"]:
        if output.action == LAST_SIGNAL_ACTION and abs(price - LAST_SIGNAL_PRICE) < 4.00:
            print(f"Skipping duplicate {output.action} alert. Price (${price:.2f}) too close to last entry (${LAST_SIGNAL_PRICE:.2f}).")
            return

        LAST_SIGNAL_ACTION = output.action
        LAST_SIGNAL_PRICE = price
        send_telegram_message(output.summary)


async def background_scanning_loop():
    while True:
        try:
            await asyncio.to_thread(analyze_and_alert)
        except Exception as e:
            print(f"Loop Exception caught: {e}")

        await asyncio.sleep(300)  # Runs scan every 5 minutes to match M5 candles


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_scanning_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot background scanner is active!"}
