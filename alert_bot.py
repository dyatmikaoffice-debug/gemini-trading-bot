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


def calculate_stoch_rsi(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50, atr_period=14):
    """Calculates EMA 50, True Stoch RSI (3,3,14,14), Swing Levels, and ATR(14)."""
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

    # 4. True Range (TR) & ATR (14) for Dynamic SL Buffer
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['PrevClose']), abs(x['Low'] - x['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    swing_high = float(df['High'].tail(15).max())
    swing_low = float(df['Low'].tail(15).min())
    atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 3.50

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']),
        "stoch_cross_down": bool(prev['%K'] > prev['%D'] and latest['%K'] < latest['%D']),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "atr": round(atr_val, 2),
        "atr_buffer": round(1.5 * atr_val, 2)
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
Analyze this 1H / 15M Stochastic RSI & Trend-Following Strategy for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

1H TIMEFRAME (Higher Timeframe Trend Context):
- Price: ${mtf['1h']['close']:.2f} | 1H EMA 50: ${mtf['1h']['ema_50']:.2f} (Above EMA: {mtf['1h']['is_above_ema']})
- 1H Stoch RSI %K: {mtf['1h']['stoch_k']:.1f} | %D: {mtf['1h']['stoch_d']:.1f}

15M TIMEFRAME (Execution & Volatility Metrics):
- Price: ${mtf['15m']['close']:.2f} | 15M EMA 50: ${mtf['15m']['ema_50']:.2f} (Above EMA: {mtf['15m']['is_above_ema']})
- 15M Stoch RSI %K: {mtf['15m']['stoch_k']:.1f} | %D: {mtf['15m']['stoch_d']:.1f}
- 15M Stoch Cross Up: {mtf['15m']['stoch_cross_up']} | Cross Down: {mtf['15m']['stoch_cross_down']}
- 15M Swing High: ${mtf['15m']['swing_high']:.2f} | Swing Low: ${mtf['15m']['swing_low']:.2f}
- 15M ATR(14): ${mtf['15m']['atr']:.2f} | Dynamic SL Volatility Buffer: ${mtf['15m']['atr_buffer']:.2f}

STRATEGY ENTRY RULES:

1. TREND-FOLLOWING PULLBACK SIGNALS (PRIMARY):
- BUY ENTRY: Price is ABOVE 15M EMA 50 AND 15M Stoch RSI dips to Oversold (< 20 or < 25) and completes a Cross Up (%K > %D).
- SELL ENTRY: Price is BELOW 15M EMA 50 AND 15M Stoch RSI rallies to Overbought (> 80 or > 75) and completes a Cross Down (%K < %D).

2. REVERSAL SIGNALS (SECONDARY):
- Allowed ONLY if there is clear timeframe alignment or momentum divergence.

3. FILTER / HOLD RULES:
- DO NOT SELL when Price is ABOVE 15M EMA 50 unless clear Bearish Divergence exists.
- DO NOT BUY when Price is BELOW 15M EMA 50 unless clear Bullish Divergence exists.
- If Stoch RSI is floating in the neutral middle zone (30 - 70), output "HOLD".

SL/TP CALCULATION RULES (USING DYNAMIC ATR BUFFER):
- FOR BUY:
  * Stop Loss (SL) = 15M Swing Low - ${mtf['15m']['atr_buffer']:.2f}
  * Risk = Entry Price - SL
  * Take Profit 1 (TP1) = Entry Price + (1.5 * Risk)
  * Take Profit 2 (TP2) = Entry Price + (2.5 * Risk)
- FOR SELL:
  * Stop Loss (SL) = 15M Swing High + ${mtf['15m']['atr_buffer']:.2f}
  * Risk = SL - Entry Price
  * Take Profit 1 (TP1) = Entry Price - (1.5 * Risk)
  * Take Profit 2 (TP2) = Entry Price - (2.5 * Risk)

OUTPUT REQUIREMENTS:
You MUST respond strictly with valid JSON with these exact key fields:
- "action": "BUY", "SELL", or "HOLD"
- "confidence": float between 0.0 and 1.0
- "summary": string formatted alert or "HOLD"

Format for "summary" when action is BUY or SELL:
STOCH RSI TRADE SIGNAL

Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Type: [Trend Pullback / Reversal]
Entry Price: ${price:.2f}

Stop Loss (SL): $[Calculated SL]
Take Profit 1 (TP1): $[Calculated TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[Calculated TP2] (1:2.5 RRR)

INDICATOR METRICS:
- 1H Stoch RSI: {mtf['1h']['stoch_k']:.1f}
- 15M Stoch RSI: {mtf['15m']['stoch_k']:.1f}
- 15M EMA 50: ${mtf['15m']['ema_50']:.2f}

Reasoning: [2-sentence explanation of trend direction and pullback confirmation]
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

    if output.action in ["BUY", "SELL"]:
        send_telegram_message(output.summary)


async def background_scanning_loop():
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
