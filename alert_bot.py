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
    action: str = Field(default="HOLD", description="BUY, SELL, or HOLD")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(default="Market conditions do not favor entry.", description="2 clean sentences explaining the market analysis.")


def send_telegram_message(message: str):
    """Sends a notification to your Telegram Channel/Chat cleanly without parsing errors."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
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

    # 4. ATR Calculation
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['PrevClose']), abs(x['Low'] - x['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Structural Swing Points (Excludes current forming candle)
    swing_high = float(df['High'].iloc[-11:-1].max())
    swing_low = float(df['Low'].iloc[-11:-1].min())
    atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 2.50

    # UPGRADED CHOCH LOGIC: Requires breaking true structural Swing High/Low
    is_choch_bearish = bool(latest['Close'] < latest['EMA_50'] and latest['Close'] < latest['VWAP'] and latest['Close'] < swing_low)
    is_choch_bullish = bool(latest['Close'] > latest['EMA_50'] and latest['Close'] > latest['VWAP'] and latest['Close'] > swing_high)

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
        "atr_buffer": round(1.2 * atr_val, 2),
        "max_vwap_allowed": round(3.0 * atr_val, 2),
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
    atr_buf = mtf["5m"]["atr_buffer"]

    # --- HIGHER TIMEFRAME TREND ALIGNMENT CHECK ---
    is_15m_bullish = mtf["15m"]["close"] > mtf["15m"]["ema_50"]
    is_15m_bearish = mtf["15m"]["close"] < mtf["15m"]["ema_50"]

    # --- DYNAMIC TRIGGER TYPE DEFINITION ---
    stoch_k_5m = mtf["5m"]["stoch_k"]
    is_choch_bull = mtf["5m"]["choch_bullish"]
    is_choch_bear = mtf["5m"]["choch_bearish"]

    if stoch_k_5m < 25 or stoch_k_5m > 75:
        trigger_type = "Trend Pullback"
    elif is_choch_bull or is_choch_bear:
        trigger_type = "Counter-Trend Reversal"
    else:
        trigger_type = "Breakout Continuation"

    # --- TIGHT 5M EXECUTION SL & TP MATH WITH SIDE-NOTE ---
    # BUY SETUP
    local_buy_sl = round(mtf["5m"]["swing_low"] - atr_buf, 2)
    buy_sl = round(max(local_buy_sl, price - 12.00), 2)
    buy_risk = round(price - buy_sl, 2)
    buy_tp1 = round(price + (1.5 * buy_risk), 2)
    raw_buy_tp2 = round(price + (2.5 * buy_risk), 2)
    swing_high_15m = mtf["15m"]["swing_high"]
    buy_tp2 = round(min(raw_buy_tp2, max(swing_high_15m, buy_tp1 + 2.0)), 2)

    if buy_tp2 < raw_buy_tp2:
        buy_tp2_str = f"${buy_tp2:.2f} (Capped at 15M Swing High; Raw 1:2.5 RRR ${raw_buy_tp2:.2f})"
    else:
        buy_tp2_str = f"${buy_tp2:.2f} (1:2.5 RRR)"

    # SELL SETUP
    local_sell_sl = round(mtf["5m"]["swing_high"] + atr_buf, 2)
    sell_sl = round(min(local_sell_sl, price + 12.00), 2)
    sell_risk = round(sell_sl - price, 2)
    sell_tp1 = round(price - (1.5 * sell_risk), 2)
    raw_sell_tp2 = round(price - (2.5 * sell_risk), 2)
    swing_low_15m = mtf["15m"]["swing_low"]
    sell_tp2 = round(max(raw_sell_tp2, min(swing_low_15m, sell_tp1 - 2.0)), 2)

    if sell_tp2 > raw_sell_tp2:
        sell_tp2_str = f"${sell_tp2:.2f} (Capped at 15M Swing Low; Raw 1:2.5 RRR ${raw_sell_tp2:.2f})"
    else:
        sell_tp2_str = f"${sell_tp2:.2f} (1:2.5 RRR)"

    prompt = f"""
Analyze this 1H / 15M / 5M Strategy for Spot Gold (XAU/USD OANDA):
Current Price: ${price:.2f}

PRE-CALCULATED EXPLICIT LEVELS:
BUY SETUP: Entry ${price:.2f} | SL ${buy_sl:.2f} | TP1 ${buy_tp1:.2f} | TP2 {buy_tp2_str}
SELL SETUP: Entry ${price:.2f} | SL ${sell_sl:.2f} | TP1 ${sell_tp1:.2f} | TP2 {sell_tp2_str}

MULTI-TIMEFRAME METRICS:
- 1H Stoch RSI %K: {mtf['1h']['stoch_k']:.1f}
- 15M Stoch RSI %K: {mtf['15m']['stoch_k']:.1f} | 15M Bullish: {is_15m_bullish} | 15M Bearish: {is_15m_bearish}
- 15M EMA 50: ${mtf['15m']['ema_50']:.2f}
- 5M VWAP Distance: ${mtf['5m']['vwap_distance']:.2f} (Max Stretch: ${mtf['5m']['max_vwap_allowed']:.2f})
- 5M Stoch RSI %K: {mtf['5m']['stoch_k']:.1f} | Cross Up: {mtf['5m']['stoch_cross_up']} | Cross Down: {mtf['5m']['stoch_cross_down']}
- 5M Green Candle: {mtf['5m']['is_green_candle']}

RULES:
1. OVEREXTENSION RULE: If 5M VWAP Distance (${mtf['5m']['vwap_distance']:.2f}) > Max Stretch (${mtf['5m']['max_vwap_allowed']:.2f}), output "action": "HOLD".
2. TIMEFRAME ALIGNMENT FILTER:
   - DO NOT BUY if 15M Trend is Bearish ({is_15m_bearish}). Output "action": "HOLD".
   - DO NOT SELL if 15M Trend is Bullish ({is_15m_bullish}). Output "action": "HOLD".
3. ENTRY CONFIRMATION:
   - BUY: Above EMA 50 & VWAP, Stoch RSI Cross Up (< 25), Green 5M Candle close.
   - SELL: Below EMA 50 & VWAP, Stoch RSI Cross Down (> 75), Red 5M Candle close.

OUTPUT REQUIREMENTS:
Output JSON with schema keys:
- "action": "BUY", "SELL", or "HOLD"
- "confidence": float between 0.0 and 1.0
- "reasoning": "Write 2 clean sentences explaining the setup decision."
"""

    output = None

    # Step 1: Groq Primary (Llama 3.1 8B Instant)
    if groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading model. Output strictly JSON matching schema with action, confidence, and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw_text = res.choices[0].message.content
            output = SignalOutput.model_validate_json(raw_text)
            print(f"[Groq Llama 3.1 8B] Decision: {output.action} ({output.confidence * 100:.0f}%)")
        except Exception as e:
            print(f"Groq API call error: {e}. Falling back to Gemini...")

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
            print(f"Gemini API fallback skipped (Rate Limit or API Error).")

    if not output:
        print("All AI model attempts failed this cycle. Continuing to next loop...")
        return

    # Cooldown & Notification Formatting in Python
    if output.action in ["BUY", "SELL"]:
        if output.action == LAST_SIGNAL_ACTION and abs(price - LAST_SIGNAL_PRICE) < 6.00:
            print(f"Skipping duplicate {output.action} alert. Price (${price:.2f}) too close to last entry (${LAST_SIGNAL_PRICE:.2f}).")
            return

        LAST_SIGNAL_ACTION = output.action
        LAST_SIGNAL_PRICE = price

        sl_val = buy_sl if output.action == "BUY" else sell_sl
        tp1_val = buy_tp1 if output.action == "BUY" else sell_tp1
        tp2_str = buy_tp2_str if output.action == "BUY" else sell_tp2_str

        # Clean Programmatic Telegram Output
        telegram_text = (
            f"STOCH RSI TRADE SIGNAL\n\n"
            f"Asset: XAUUSD (Gold Spot)\n"
            f"Action: {output.action}\n"
            f"Type: {trigger_type}\n"
            f"Entry Price: ${price:.2f}\n\n"
            f"Stop Loss (SL): ${sl_val:.2f}\n"
            f"Take Profit 1 (TP1): ${tp1_val:.2f} (1:1.5 RRR)\n"
            f"Take Profit 2 (TP2): {tp2_str}\n\n"
            f"INDICATOR METRICS:\n"
            f"- 1H Stoch RSI: {mtf['1h']['stoch_k']:.1f}\n"
            f"- 15M Stoch RSI: {mtf['15m']['stoch_k']:.1f}\n"
            f"- 15M EMA 50: ${mtf['15m']['ema_50']:.2f}\n\n"
            f"Reasoning: {output.reasoning}"
        )

        send_telegram_message(telegram_text)


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
