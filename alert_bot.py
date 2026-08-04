import os
import json
import asyncio
from contextlib import asynccontextmanager
import httpx
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Importing both Google GenAI and OpenAI (for Groq)
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
EXCHANGE = "OANDA"  # Matches TradingView OANDA Gold Spot


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


def calculate_stoch_rsi(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50):
    """Calculates EMA 50, True Stoch RSI (3,3,14,14), and Swing Levels."""
    if df.empty or len(df) < (rsi_period + stoch_period + ema_period):
        return None

    # 1. EMA 50
    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # 2. RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. Stochastic RSI (3,3,14,14)
    rsi_min = df['RSI'].rolling(window=stoch_period).min()
    rsi_max = df['RSI'].rolling(window=stoch_period).max()
    stoch_rsi = (df['RSI'] - rsi_min) / (rsi_max - rsi_min)

    df['%K'] = stoch_rsi.rolling(window=k_period).mean() * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Calculate recent 15M Swing High/Low for SL calculation
    swing_high = float(df['High'].tail(15).max())
    swing_low = float(df['Low'].tail(15).min())

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']),
        "stoch_cross_down": bool(prev['%K'] > prev['%D'] and latest['%K'] < latest['%D']),
        "swing_high": swing_high,
        "swing_low": swing_low
    }


def get_mtf_data():
    df_15m = fetch_twelve_data("15min", outputsize=100)
    df_1h = fetch_twelve_data("1h", outputsize=100)

    if df_15m.empty or df_1h.empty:
        return None

    return {
        "15m": calculate_stoch_rsi(df_15m),
        "1h": calculate_stoch_rsi(df_1h)
    }


def analyze_and_alert():
    print("Checking Twelve Data Multi-Timeframe Spot Gold Data...")
    mtf = get_mtf_data()

    if not mtf or not mtf["15m"] or not mtf["1h"]:
        print("Waiting for valid data stream...")
        return

    price = mtf["15m"]["close"]

    prompt = f"""
Analyze this 1H / 15M Stochastic RSI Strategy for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

1H TIMEFRAME (Higher Timeframe Trend Direction):
- Price: ${mtf['1h']['close']:.2f} | 1H EMA 50: ${mtf['1h']['ema_50']:.2f} (Above EMA: {mtf['1h']['is_above_ema']})
- 1H Stoch RSI %K: {mtf['1h']['stoch_k']:.1f} | %D: {mtf['1h']['stoch_d']:.1f}

15M TIMEFRAME (Execution & Momentum):
- Price: ${mtf['15m']['close']:.2f} | 15M EMA 50: ${mtf['15m']['ema_50']:.2f} (Above EMA: {mtf['15m']['is_above_ema']})
- 15M Stoch RSI %K: {mtf['15m']['stoch_k']:.1f} | %D: {mtf['15m']['stoch_d']:.1f}
- 15M Stoch Cross Up: {mtf['15m']['stoch_cross_up']} | Cross Down: {mtf['15m']['stoch_cross_down']}
- 15M Recent Swing High: ${mtf['15m']['swing_high']:.2f}
- 15M Recent Swing Low: ${mtf['15m']['swing_low']:.2f}

STRATEGY ENTRY RULES:
- BUY ENTRY: 
  * 1H Trend is Bullish (Price > 1H EMA 50 OR 1H Stoch RSI pointing up).
  * BOTH 1H and 15M Stoch RSI show oversold conditions (< 20 or returning from oversold).
  * 15M Stoch RSI confirms a Cross Up (%K > %D).
- SELL ENTRY: 
  * 1H Trend is Bearish (Price < 1H EMA 50 OR 1H Stoch RSI pointing down).
  * BOTH 1H and 15M Stoch RSI show overbought conditions (> 80 or returning from overbought).
  * 15M Stoch RSI confirms a Cross Down (%K < %D).
- HOLD: If 1H and 15M timeframes conflict or lack clear extreme Stoch RSI conditions.

SL/TP CALCULATIONS (IF BUY OR SELL):
- BUY:
  * Stop Loss (SL) = 15M Swing Low - $1.50
  * Risk = Entry Price - SL
  * Take Profit 1 (TP1) = Entry Price + (1.5 * Risk)
  * Take Profit 2 (TP2) = Entry Price + (2.5 * Risk)
- SELL:
  * Stop Loss (SL) = 15M Swing High + $1.50
  * Risk = SL - Entry Price
  * Take Profit 1 (TP1) = Entry Price - (1.5 * Risk)
  * Take Profit 2 (TP2) = Entry Price - (2.5 * Risk)

OUTPUT REQUIREMENTS:
You MUST respond strictly with valid JSON with these exact key fields:
- "action": "BUY", "SELL", or "HOLD"
- "confidence": float between 0.0 and 1.0
- "summary": string formatted alert or "HOLD"

Format for "summary" when action is BUY or SELL:
🚨 STOCH RSI TRADE SIGNAL

Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Entry Price: ${price:.2f}

Stop Loss (SL): $[Calculated SL]
Take Profit 1 (TP1): $[Calculated TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[Calculated TP2] (1:2.5 RRR)

📍 INDICATOR METRICS:
• 1H Stoch RSI: {mtf['1h']['stoch_k']:.1f}
• 15M Stoch RSI: {mtf['15m']['stoch_k']:.1f}
• 15M EMA 50: ${mtf['15m']['ema_50']:.2f}

📊 Reasoning: [2-sentence explanation of 1H alignment and 15M Stoch RSI condition]
"""

    output = None

    # Step 1: Try Groq First (Llama 3.3 70B)
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

    # Step 2: Fallback to Gemini 2.5 Flash if Groq fails
    if not output and genai_client:
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=SignalOutput,
            )
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            output = SignalOutput.model_validate_json(response.text)
            print(f"[Gemini 2.5] Decision: {output.action} ({output.confidence * 100:.0f}%)")
        except Exception as e:
            print(f"Gemini API call failed: {e}")

    if not output:
        print("All AI model attempts failed this cycle.")
        return

    if output.action in ["BUY", "SELL"]:
        send_telegram_message(output.summary)


async def background_scanning_loop():
    """Runs the analysis every 10 minutes in a non-blocking background loop."""
    while True:
        try:
            await asyncio.to_thread(analyze_and_alert)
        except Exception as e:
            print(f"Loop Exception caught: {e}")

        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_scanning_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot background scanner is active!"}
