import os
import json
import asyncio
from contextlib import asynccontextmanager
import httpx
import pandas as pd
import numpy as np
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
    reasoning: str = Field(default="Market conditions do not favor entry.", description="2 clean sentences explaining the setup decision or veto reason.")


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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates the Average Directional Index (ADX) to measure trend strength."""
    if df.empty or len(df) < (period * 2):
        return 20.0

    temp_df = df.copy()
    temp_df['prev_close'] = temp_df['Close'].shift(1)
    
    # True Range
    temp_df['tr'] = temp_df[['High', 'Low', 'prev_close']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['prev_close']), abs(x['Low'] - x['prev_close'])), axis=1
    )
    
    # Directional Movement
    temp_df['up_move'] = temp_df['High'] - temp_df['High'].shift(1)
    temp_df['down_move'] = temp_df['Low'].shift(1) - temp_df['Low']

    temp_df['plus_dm'] = temp_df.apply(lambda x: x['up_move'] if (x['up_move'] > x['down_move'] and x['up_move'] > 0) else 0.0, axis=1)
    temp_df['minus_dm'] = temp_df.apply(lambda x: x['down_move'] if (x['down_move'] > x['up_move'] and x['down_move'] > 0) else 0.0, axis=1)

    alpha = 1.0 / period
    temp_df['tr_smoothed'] = temp_df['tr'].ewm(alpha=alpha, adjust=False).mean()
    temp_df['plus_di'] = 100 * (temp_df['plus_dm'].ewm(alpha=alpha, adjust=False).mean() / temp_df['tr_smoothed'].replace(0, 1))
    temp_df['minus_di'] = 100 * (temp_df['minus_dm'].ewm(alpha=alpha, adjust=False).mean() / temp_df['tr_smoothed'].replace(0, 1))

    di_sum = temp_df['plus_di'] + temp_df['minus_di']
    di_diff = (temp_df['plus_di'] - temp_df['minus_di']).abs()
    
    temp_df['dx'] = 100 * (di_diff / di_sum.replace(0, 1))
    temp_df['adx'] = temp_df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return float(temp_df['adx'].iloc[-1])


def calculate_metrics(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50, atr_period=14):
    """Calculates EMA 50, VWAP, Stoch RSI, ATR, ADX, Slope, Swing Levels, ChoCH conditions, and Dynamic Squeeze Bounds."""
    if df.empty or len(df) < (rsi_period + stoch_period + ema_period):
        return None

    # 1. EMA 50 & Slope
    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    ema_50_rising = bool(df['EMA_50'].iloc[-1] > df['EMA_50'].iloc[-2])

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

    # 5. ADX Calculation
    adx_val = calculate_adx(df, period=14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Two-Candle Confirmation Filter (Requires current and previous candle above/below baseline)
    is_confirmed_bullish = bool(
        latest['Close'] > latest['EMA_50'] and prev['Close'] > prev['EMA_50'] and
        latest['Close'] > latest['VWAP'] and prev['Close'] > prev['VWAP'] and
        latest['Close'] > latest['Open']
    )
    is_confirmed_bearish = bool(
        latest['Close'] < latest['EMA_50'] and prev['Close'] < prev['EMA_50'] and
        latest['Close'] < latest['VWAP'] and prev['Close'] < prev['VWAP'] and
        latest['Close'] < latest['Open']
    )

    # Structural Swing Points
    swing_high = float(df['High'].iloc[-11:-1].max())
    swing_low = float(df['Low'].iloc[-11:-1].min())
    atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 2.50

    # DYNAMIC ATR CONSOLIDATION GUARD
    range_high = float(df['High'].iloc[-16:-1].max())
    range_low = float(df['Low'].iloc[-16:-1].min())
    range_width = round(range_high - range_low, 2)
    is_tight_squeeze = bool(range_width < (2.0 * atr_val))

    # Dynamic Breakout Detections
    is_breakout_up = bool(latest['Close'] > range_high and latest['Close'] > latest['Open'])
    is_breakout_down = bool(latest['Close'] < range_low and latest['Close'] < latest['Open'])

    # UPGRADED CHOCH LOGIC
    is_choch_bearish = bool(latest['Close'] < latest['EMA_50'] and latest['Close'] < latest['VWAP'] and latest['Close'] < swing_low)
    is_choch_bullish = bool(latest['Close'] > latest['EMA_50'] and latest['Close'] > latest['VWAP'] and latest['Close'] > swing_high)

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "ema_50_rising": ema_50_rising,
        "vwap": float(latest['VWAP']),
        "is_above_ema": bool(latest['Close'] > latest['EMA_50']),
        "is_above_vwap": bool(latest['Close'] > latest['VWAP']),
        "is_confirmed_bullish": is_confirmed_bullish,
        "is_confirmed_bearish": is_confirmed_bearish,
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
        "is_green_candle": bool(latest['Close'] > latest['Open']),
        "range_width": range_width,
        "is_tight_squeeze": is_tight_squeeze,
        "is_breakout_up": is_breakout_up,
        "is_breakout_down": is_breakout_down,
        "adx": round(adx_val, 1)
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
    adx_15m = mtf["15m"]["adx"]

    # --- 1H RANGE BOUNDARY FILTER ---
    swing_high_1h = mtf["1h"]["swing_high"]
    swing_low_1h = mtf["1h"]["swing_low"]
    range_1h = swing_high_1h - swing_low_1h
    range_pos_1h = (price - swing_low_1h) / range_1h if range_1h > 0 else 0.5

    is_break_up = mtf["5m"]["is_breakout_up"]
    is_break_dn = mtf["5m"]["is_breakout_down"]

    # Prevent buying into 1H Resistance (> 85% range) or selling into 1H Support (< 15% range) without breakout
    at_1h_resistance = bool(range_pos_1h > 0.85 and not is_break_up)
    at_1h_support = bool(range_pos_1h < 0.15 and not is_break_dn)

    # --- HIGHER TIMEFRAME TREND & SLOPE CHECK ---
    is_15m_bullish = mtf["15m"]["close"] > mtf["15m"]["ema_50"]
    is_15m_bearish = mtf["15m"]["close"] < mtf["15m"]["ema_50"]
    is_15m_ema_rising = mtf["15m"]["ema_50_rising"]

    # --- HARD PYTHON-LEVEL CONFIRMATION FILTERS ---
    is_5m_confirmed_bull = mtf["5m"]["is_confirmed_bullish"]
    is_5m_confirmed_bear = mtf["5m"]["is_confirmed_bearish"]
    vwap_dist = mtf["5m"]["vwap_distance"]
    max_vwap = mtf["5m"]["max_vwap_allowed"]
    is_squeeze = mtf["15m"]["is_tight_squeeze"]

    # Strictly require 2-Candle Confirmation, 15M Slope Alignment, AND 1H Range Boundary Compliance
    valid_buy_structure = is_5m_confirmed_bull and is_15m_ema_rising and not is_15m_bearish and mtf["15m"]["stoch_k"] <= 65.0 and not at_1h_resistance
    valid_sell_structure = is_5m_confirmed_bear and not is_15m_ema_rising and not is_15m_bullish and mtf["15m"]["stoch_k"] >= 35.0 and not at_1h_support

    # Overextension Guard & Consolidation Squeeze Guard
    if vwap_dist > max_vwap:
        print(f"Skipping: Price overextended from VWAP (${vwap_dist:.2f} > ${max_vwap:.2f}).")
        return

    if is_squeeze and not is_break_up and not is_break_dn:
        print("Skipping: Market trapped in tight consolidation squeeze.")
        return

    if at_1h_resistance and is_5m_confirmed_bull:
        print(f"Skipping BUY: Price (${price:.2f}) at 1H Resistance Boundary ({range_pos_1h*100:.1f}% of 1H Range) without breakout.")
        return

    if at_1h_support and is_5m_confirmed_bear:
        print(f"Skipping SELL: Price (${price:.2f}) at 1H Support Boundary ({range_pos_1h*100:.1f}% of 1H Range) without breakout.")
        return

    if not valid_buy_structure and not valid_sell_structure:
        print("Skipping: Price structure violates baseline alignment, 2-candle confirmation, or 1H boundary rules.")
        return

    candidate_action = "BUY" if valid_buy_structure else "SELL"

    # --- DYNAMIC ADX RISK-TO-REWARD RATIO (RRR) SCALING ---
    if adx_15m < 20.0:
        tp1_mult, tp2_mult = 1.0, 1.5
        adx_desc = f"15M ADX ({adx_15m:.1f}) Weak Trend"
    elif adx_15m >= 35.0:
        tp1_mult, tp2_mult = 2.0, 4.0
        adx_desc = f"15M ADX ({adx_15m:.1f}) Explosive Momentum"
    else:
        tp1_mult, tp2_mult = 1.5, 2.5
        adx_desc = f"15M ADX ({adx_15m:.1f}) Moderate Trend"

    # --- DYNAMIC TRIGGER TYPE DEFINITION ---
    stoch_k_5m = mtf["5m"]["stoch_k"]
    is_choch_bull = mtf["5m"]["choch_bullish"]
    is_choch_bear = mtf["5m"]["choch_bearish"]

    if is_break_up or is_break_dn:
        trigger_type = "Breakout Continuation"
    elif is_choch_bull or is_choch_bear:
        trigger_type = "Counter-Trend Reversal"
    elif stoch_k_5m < 25 or stoch_k_5m > 75:
        trigger_type = "Trend Pullback"
    else:
        trigger_type = "Breakout Continuation"

    # --- DYNAMIC SL & TP MULTI-LEVEL MATH ---
    # BUY SETUP
    local_buy_sl = round(mtf["5m"]["swing_low"] - atr_buf, 2)
    buy_sl = round(max(local_buy_sl, price - 12.00), 2)
    buy_risk = round(price - buy_sl, 2)
    buy_tp1 = round(price + (tp1_mult * buy_risk), 2)
    raw_buy_tp2 = round(price + (tp2_mult * buy_risk), 2)
    swing_high_15m = mtf["15m"]["swing_high"]
    buy_tp2 = round(min(raw_buy_tp2, max(swing_high_15m, buy_tp1 + 2.0)), 2)

    if buy_tp2 < raw_buy_tp2:
        buy_tp2_str = f"${buy_tp2:.2f} (Capped at 15M Swing High; Raw 1:{tp2_mult:.1f} RRR ${raw_buy_tp2:.2f})"
    else:
        buy_tp2_str = f"${buy_tp2:.2f} (1:{tp2_mult:.1f} RRR)"

    # SELL SETUP
    local_sell_sl = round(mtf["5m"]["swing_high"] + atr_buf, 2)
    sell_sl = round(min(local_sell_sl, price + 12.00), 2)
    sell_risk = round(sell_sl - price, 2)
    sell_tp1 = round(price - (tp1_mult * sell_risk), 2)
    raw_sell_tp2 = round(price - (tp2_mult * sell_risk), 2)
    swing_low_15m = mtf["15m"]["swing_low"]
    sell_tp2 = round(max(raw_sell_tp2, min(swing_low_15m, sell_tp1 - 2.0)), 2)

    if sell_tp2 > raw_sell_tp2:
        sell_tp2_str = f"${sell_tp2:.2f} (Capped at 15M Swing Low; Raw 1:{tp2_mult:.1f} RRR ${raw_sell_tp2:.2f})"
    else:
        sell_tp2_str = f"${sell_tp2:.2f} (1:{tp2_mult:.1f} RRR)"

    # --- HYBRID VETO MODEL PROMPT ---
    prompt = f"""
You are a Senior Quantitative Trading Analyst with VETO POWER.

Python Risk Manager has passed a candidate setup:
PROPOSED CANDIDATE ACTION: {candidate_action}
Current Price: ${price:.2f}

PRE-CALCULATED EXPLICIT LEVELS ({adx_desc}):
BUY SETUP: Entry ${price:.2f} | SL ${buy_sl:.2f} | TP1 ${buy_tp1:.2f} (1:{tp1_mult:.1f} RRR) | TP2 {buy_tp2_str}
SELL SETUP: Entry ${price:.2f} | SL ${sell_sl:.2f} | TP1 ${sell_tp1:.2f} (1:{tp1_mult:.1f} RRR) | TP2 {sell_tp2_str}

MULTI-TIMEFRAME & BOUNDARY METRICS:
- 1H Swing High: ${swing_high_1h:.2f} | 1H Swing Low: ${swing_low_1h:.2f}
- Price Position in 1H Range: {range_pos_1h * 100:.1f}%
- 15M ADX Trend Strength: {adx_15m:.1f}
- 15M EMA 50 Slope Rising: {is_15m_ema_rising}
- 5M Two-Candle Bullish Confirmation: {is_5m_confirmed_bull}
- 5M Two-Candle Bearish Confirmation: {is_5m_confirmed_bear}
- 15M Range Width: ${mtf['15m']['range_width']:.2f} | 15M Tight Squeeze: {mtf['15m']['is_tight_squeeze']}
- 5M Breakout Up: {is_break_up} | 5M Breakout Down: {is_break_dn}
- 1H Stoch RSI %K: {mtf['1h']['stoch_k']:.1f}
- 15M Stoch RSI %K: {mtf['15m']['stoch_k']:.1f} | 15M Bullish: {is_15m_bullish} | 15M Bearish: {is_15m_bearish}
- 15M EMA 50: ${mtf['15m']['ema_50']:.2f}
- 5M VWAP Distance: ${mtf['5m']['vwap_distance']:.2f} (Max Stretch: ${mtf['5m']['max_vwap_allowed']:.2f})
- 5M Stoch RSI %K: {mtf['5m']['stoch_k']:.1f} | Cross Up: {mtf['5m']['stoch_cross_up']} | Cross Down: {mtf['5m']['stoch_cross_down']}
- 5M Green Candle: {mtf['5m']['is_green_candle']}

ANALYST VETO DIRECTIVE:
Review the market context for the proposed setup ({candidate_action}).
Look for structural red flags such as lower-high traps, relief bounces into resistance, fading momentum, or horizontal range chop.
- If you detect chop or trap risk, EXERCISE YOUR VETO POWER and return "action": "HOLD".
- If the trend structure is clean and momentum is genuine, CONFIRM by returning "action": "{candidate_action}".

OUTPUT REQUIREMENTS:
Output JSON with schema keys:
- "action": "{candidate_action}" or "HOLD"
- "confidence": float between 0.0 and 1.0
- "reasoning": "Write 2 clean sentences explaining your decision (confirming entry or exercising veto)."
"""

    output = None

    # Step 1: Groq Primary (Llama 3.1 8B Instant)
    if groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading model with veto power. Output strictly JSON matching schema with action, confidence, and reasoning."},
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

    # Handle AI Veto Execution
    if output.action == "HOLD":
        print(f"[AI VETO EXERCISED] Analyst rejected candidate setup '{candidate_action}'. Reason: {output.reasoning}")
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
            f"Take Profit 1 (TP1): ${tp1_val:.2f} (1:{tp1_mult:.1f} RRR)\n"
            f"Take Profit 2 (TP2): {tp2_str}\n\n"
            f"INDICATOR METRICS:\n"
            f"- 1H Range Position: {range_pos_1h * 100:.1f}%\n"
            f"- 15M ADX Strength: {adx_15m:.1f}\n"
            f"- 15M EMA 50 Slope: {'Rising' if is_15m_ema_rising else 'Falling'}\n"
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
