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

# Cooldown Tracker
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


def calculate_metrics(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50, atr_period=14):
    """Calculates EMA 50, VWAP, Stoch RSI, ATR, Swing Levels, and ChoCH conditions."""
    if df.empty or len(df) < (rsi_period + stoch_period + ema_period):
        return None

    # 1. EMA 50
    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # 2. VWAP Approximation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
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

    # 4. ATR Calculation for Upgrade B
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['PrevClose']), abs(x['Low'] - x['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    swing_high = float(df['High'].tail(10).max())
    swing_low = float(df['Low'].tail(10).min())
    atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 2.50

    is_choch_bearish = bool(latest['Close'] < latest['EMA_50'] and latest['Close'] < latest['VWAP'])
    is_choch_bullish = bool(latest['Close'] > latest['EMA_50'] and latest['Close'] > latest['VWAP'])

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "vwap": float(latest['VWAP']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "is_above_vwap": bool(latest['Close'] > latest['VWAP']),
        "vwap_distance": round(abs(float(latest['Close']) - float(latest['VWAP'])), 2),
        "choch_bearish": is_choch_bearish,
        "choch_bullish": is_choch_bullish,
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']),
        "stoch_cross_down": bool(prev['%K'] > prev['%D'] and latest['%K'] < latest['%D']),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "atr": round(atr_val, 2),
        "atr_buffer": round(1.2 * atr_val, 2), # Upgrade B: Dynamic ATR SL Buffer
        "max_vwap_allowed": round(3.0 * atr_val, 2), # Upgrade A: Dynamic VWAP Stretch Limit
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
Analyze this 1H / 15M / 5M Strategy for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

5M METRICS (Execution):
- Price: ${mtf['5m']['close']:.2f} | 5M EMA 50: ${mtf['5m']['ema_50']:.2f} | 5M VWAP: ${mtf['5m']['vwap']:.2f}
- VWAP Distance: ${mtf['5m']['vwap_distance']:.2f} (Max Allowed Stretch: ${mtf['5m']['max_vwap_allowed']:.2f})
- 5M Bearish ChoCH: {mtf['5m']['choch_bearish']} | Bullish ChoCH: {mtf['5m']['choch_bullish']}
- 5M Stoch RSI %K: {mtf['5m']['stoch_k']:.1f} | Cross Up: {mtf['5m']['stoch_cross_up']} | Cross Down: {mtf['5m']['stoch_cross_down']}
- 5M Green Candle: {mtf['5m']['is_green_candle']}
- 5M Swing High: ${mtf['5m']['swing_high']:.2f} | Swing Low: ${mtf['5m']['swing_low']:.2f}
- Dynamic ATR SL Buffer: ${mtf['5m']['atr_buffer']:.2f}

15M & 1H STRUCTURAL TARGETS (Upgrade D):
- 15M Swing High (Resistance): ${mtf['15m']['swing_high']:.2f} | 15M Swing Low (Support): ${mtf['15m']['swing_low']:.2f}

RULES (UPGRADES A, B, C, D ACTIVE):
1. OVEREXTENSION RULE (UPGRADE A):
   - If 5M VWAP Distance (${mtf['5m']['vwap_distance']:.2f}) is greater than Max Stretch (${mtf['5m']['max_vwap_allowed']:.2f}), DO NOT BUY/SELL. Force "HOLD" to prevent chasing extended moves.

2. DYNAMIC ATR STOP LOSS (UPGRADE B):
   - For BUY: SL = Minimum(5M Swing Low, 15M Swing Low) - ${mtf['5m']['atr_buffer']:.2f}.
   - For SELL: SL = Maximum(5M Swing High, 15M Swing High) + ${mtf['5m']['atr_buffer']:.2f}.

3. STRUCTURE-BASED TARGETS (UPGRADE D):
   - TP1 = Entry Price +/- (1.5 * Risk)
   - TP2 = Set at 1:2.5 RRR, BUT cap TP2 at the 15M Swing High (for BUY) or 15M Swing Low (for SELL) if structural levels are closer.

4. ENTRY CONFIRMATION:
   - BUY: Above EMA 50 & VWAP, Stoch RSI Cross Up (< 25), Green 5M Candle close.
   - SELL: Below EMA 50 & VWAP, Stoch RSI Cross Down (> 75), Red 5M Candle close.

OUTPUT REQUIREMENTS:
Output strictly valid JSON with keys: "action" ("BUY", "SELL", "HOLD"), "confidence", "summary".

"summary" Format based on Signal Type:

1. Trend Pullback:
STOCH RSI PULLBACK SIGNAL (5M EXECUTION)
Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Signal Type: Trend Pullback
Entry Price: $[Price]
Stop Loss (SL): $[Calculated SL with ATR]
Take Profit 1 (TP1): $[TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[TP2 capped at 15M Structure]
INDICATOR METRICS:
- 5M EMA 50: $[EMA]
- 5M VWAP: $[VWAP]
- 5M Stoch RSI: [K_val]
Reasoning: [2-sentence explanation]

2. ChoCH Reversal:
MARKET STRUCTURE SIGNAL (5M ChoCH REVERSAL)
Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Signal Type: Change of Character (ChoCH)
Entry Price: $[Price]
Stop Loss (SL): $[Calculated SL with ATR]
Take Profit 1 (TP1): $[TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[TP2 capped at 15M Structure]
INDICATOR METRICS:
- 5M EMA 50: $[EMA]
- 5M VWAP: $[VWAP]
- 5M Stoch RSI: [K_val]
Reasoning: [2-sentence explanation]

3. Momentum Divergence:
MOMENTUM DIVERGENCE SIGNAL (5M EXECUTION)
Asset: XAUUSD (Gold Spot)
Action: [BUY or SELL]
Signal Type: Momentum Divergence
Entry Price: $[Price]
Stop Loss (SL): $[Calculated SL with ATR]
Take Profit 1 (TP1): $[TP1] (1:1.5 RRR)
Take Profit 2 (TP2): $[TP2 capped at 15M Structure]
INDICATOR METRICS:
- 5M EMA 50: $[EMA]
- 5M VWAP: $[VWAP]
- 5M Stoch RSI: [K_val]
Reasoning: [2-sentence explanation]
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

    # Cooldown Filter (Upgrade C: $6.00 distance lock)
    if output.action in ["BUY", "SELL"]:
        if output.action == LAST_SIGNAL_ACTION and abs(price - LAST_SIGNAL_PRICE) < 6.00:
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

        await asyncio.sleep(360)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_scanning_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot background scanner is active!"}
