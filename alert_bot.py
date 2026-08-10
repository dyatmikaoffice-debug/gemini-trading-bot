import os
import json
import asyncio
import psycopg2
import gc
import logging
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from openai import OpenAI

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- ENVIRONMENT VARIABLES & SANITIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
RAW_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()

# Sanitize Telegram Bot Token
CLEAN_BOT_TOKEN = "".join(RAW_BOT_TOKEN.split())
TELEGRAM_CHAT_ID = "".join(RAW_CHAT_ID.split())

if CLEAN_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = CLEAN_BOT_TOKEN[3:]
else:
    TELEGRAM_BOT_TOKEN = CLEAN_BOT_TOKEN

# Initialize AI Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

SYMBOL = "XAU/USD"

class SignalOutput(BaseModel):
    action: str = Field(default="HOLD", description="BUY, SELL, or HOLD")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(default="Market conditions do not favor entry.", description="2 clean sentences explaining the decision")

# --- DATABASE CONNECTION & INITIALIZATION ---
def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing.")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    if not DATABASE_URL:
        logging.warning("[WARNING] DATABASE_URL not set. Database logging disabled.")
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_price REAL,
                tp1_price REAL,
                tp2_price REAL,
                status TEXT NOT NULL,
                outcome TEXT DEFAULT 'PENDING',
                exit_price REAL,
                trigger_type TEXT,
                confidence REAL,
                adx_15m REAL,
                stoch_rsi_15m REAL,
                divergence_type TEXT,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[NEON DATABASE] Table structure verified successfully.")
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to initialize database: {e}")

def log_trade_signal(status: str, action: str, trigger_type: str, price: float, sl: float, tp1: float, tp2: float, confidence: float, adx_15m: float, stoch_rsi_15m: float, divergence_type: str, reasoning: str):
    if not DATABASE_URL:
        return None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        price_val = float(price) if price is not None else 0.0
        sl_val = float(sl) if sl is not None else 0.0
        tp1_val = float(tp1) if tp1 is not None else 0.0
        tp2_val = float(tp2) if tp2 is not None else 0.0
        conf_val = float(confidence) if confidence is not None else 0.0
        adx_val = float(adx_15m) if adx_15m is not None else 0.0
        stoch_val = float(stoch_rsi_15m) if stoch_rsi_15m is not None else 0.0

        cursor.execute("""
            INSERT INTO signals 
            (action, entry_price, sl_price, tp1_price, tp2_price, status, outcome, trigger_type, confidence, adx_15m, stoch_rsi_15m, divergence_type, reasoning, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'PENDING', %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """, (str(action), price_val, sl_val, tp1_val, tp2_val, str(status), str(trigger_type), conf_val, adx_val, stoch_val, str(divergence_type), str(reasoning)))
        
        row = cursor.fetchone()
        new_id = row['id'] if row else None
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"[NEON DB LOGGED] Signal ID #{new_id} | Status: {status} | Action: {action} | Price: ${price_val:.2f}")
        return new_id
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to log signal: {e}")
        return None

# --- TWO-STAGE TP TRACKING FUNCTION ---
def update_open_trades(current_high: float, current_low: float):
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED' AND (outcome = 'PENDING' OR outcome = 'WIN (TP1 HIT)')")
        open_trades = cursor.fetchall()

        if not open_trades:
            cursor.close()
            conn.close()
            return

        c_high = float(current_high)
        c_low = float(current_low)

        for trade in open_trades:
            trade_id = trade['id']
            action = trade['action']
            entry_price = float(trade['entry_price'])
            sl = float(trade['sl_price']) if trade['sl_price'] is not None else None
            tp1 = float(trade['tp1_price']) if trade['tp1_price'] is not None else None
            tp2 = float(trade['tp2_price']) if trade['tp2_price'] is not None else None
            current_outcome = trade['outcome']

            new_outcome = None
            exit_price = None

            if action == "BUY":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 is not None and c_high >= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_low <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1
                elif sl is not None and c_low <= sl:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 is not None and c_high >= tp2:
                    new_outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 is not None and c_high >= tp1:
                    new_outcome = "WIN (TP1 HIT)"
                    exit_price = tp1

            elif action == "SELL":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 is not None and c_low <= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_high >= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1
                elif sl is not None and c_high >= sl:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 is not None and c_low <= tp2:
                    new_outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 is not None and c_low <= tp1:
                    new_outcome = "WIN (TP1 HIT)"
                    exit_price = tp1

            if new_outcome and new_outcome != current_outcome:
                cursor.execute("""
                    UPDATE signals 
                    SET outcome = %s, exit_price = %s 
                    WHERE id = %s
                """, (new_outcome, float(exit_price), trade_id))
                conn.commit()
                logging.info(f"[TRADE UPDATE] Signal ID {trade_id} -> {new_outcome} at ${exit_price:.2f}")

        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to update trade outcomes: {e}")

# --- MARKET DATA FETCHING ---
async def fetch_timeframe_data(client: httpx.AsyncClient, timeframe: str, outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={timeframe}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    res = await client.get(url)
    if res.status_code != 200 or not res.text:
        return None
    try:
        data = res.json()
    except Exception:
        return None

    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df

# --- INDICATORS CALCULATIONS ---
def calculate_metrics(df: pd.DataFrame):
    df = df.tail(100).copy()
    
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["close"]).rolling(window=20).sum() / df["close"].rolling(window=20).sum()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    rsi_min = df["rsi"].rolling(window=14).min()
    rsi_max = df["rsi"].rolling(window=14).max()
    stoch_rsi = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10)
    df["stoch_k"] = stoch_rsi.rolling(window=3).mean() * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ADX Calculation
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]
    
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)

    tr14 = df["tr"].rolling(14).sum()
    plus_di = 100 * (df["plus_dm"].rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (df["minus_dm"].rolling(14).sum() / (tr14 + 1e-10))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df["adx"] = dx.rolling(14).mean()

    # ATR Calculation
    df["atr"] = df["tr"].rolling(window=14).mean()

    return df

def check_divergence(df: pd.DataFrame):
    if len(df) < 15:
        return "None"
    
    p_now = float(df["close"].iloc[-1])
    p_prev = float(df["close"].iloc[-5:-1].min())
    s_now = float(df["stoch_k"].iloc[-1])
    s_prev = float(df["stoch_k"].iloc[-5:-1].min())

    if p_now < p_prev and s_now > s_prev and s_now < 40:
        return "Bullish Divergence (Lower Price Low + Higher Stoch Low)"

    p_prev_max = float(df["close"].iloc[-5:-1].max())
    s_prev_max = float(df["stoch_k"].iloc[-5:-1].max())
    if p_now > p_prev_max and s_now < s_prev_max and s_now > 60:
        return "Bearish Divergence (Higher Price High + Lower Stoch High)"

    return "None"

# --- TELEGRAM NOTIFICATIONS ---
async def send_telegram_alert(client: httpx.AsyncClient, text: str, target_chat_id: str = None):
    chat_id = "".join(str(target_chat_id or TELEGRAM_CHAT_ID).split())
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logging.error("[TELEGRAM ERROR] Missing token or chat_id")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    
    try:
        res = await client.post(url, json=payload)
        if res.status_code != 200:
            payload_plain = {"chat_id": chat_id, "text": text}
            res_plain = await client.post(url, json=payload_plain)
            if res_plain.status_code == 200:
                logging.info(f"[TELEGRAM SENT] Delivered plain text to Chat ID {chat_id}")
            else:
                logging.error(f"[TELEGRAM ERROR] Failed sending message: {res_plain.text}")
        else:
            logging.info(f"[TELEGRAM SENT] Delivered Markdown to Chat ID {chat_id}")
    except Exception as e:
        logging.error(f"[TELEGRAM EXCEPTION] {e}")

# --- AI ANALYST EVALUATION ---
async def analyze_signal_with_ai(proposed_action: str, current_price: float, df_5m: pd.DataFrame, df_15m: pd.DataFrame, divergence: str):
    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD).
A technical trigger suggests a {proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 5-Minute: Close=${float(df_5m['close'].iloc[-1]):.2f}, EMA 50=${float(df_5m['ema_50'].iloc[-1]):.2f}, EMA 200=${float(df_5m['ema_200'].iloc[-1]):.2f}, VWAP=${float(df_5m['vwap'].iloc[-1]):.2f}, Stoch RSI %K=${float(df_5m['stoch_k'].iloc[-1]):.1f}. 2. 15-Minute: Close=${float(df_15m['close'].iloc[-1]):.2f}, EMA 50=${float(df_15m['ema_50'].iloc[-1]):.2f}, EMA 200=${float(df_15m['ema_200'].iloc[-1]):.2f}, ADX Trend Strength=${float(df_15m['adx'].iloc[-1]):.1f}, Stoch RSI \%K=${float(df_15m['stoch_k'].iloc[-1]):.1f}.
3. Divergence State: {divergence}.

CRITICAL VETO RULES:
- VETO if proposed BUY is below 5M EMA 200 or proposed SELL is above 5M EMA 200 (counter-trend trap).
- VETO if 15M ADX is below 15.0 indicating extreme horizontal chop.
- ALLOW BUY entries during strong momentum (15M ADX > 30.0) even if Stoch RSI is above 50, provided price is cleanly aligned above 5M & 15M EMA 200.

Respond strictly in valid JSON matching schema:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, "reasoning": "2 concise sentences explaining decision"}}
"""

    if GROQ_API_KEY:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            return SignalOutput(**data)
        except Exception as e:
            logging.warning(f"[AI WARNING] Groq call failed: {e}. Falling back to Gemini.")

    if GEMINI_API_KEY:
        try:
            res = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SignalOutput,
                    temperature=0.1,
                )
            )
            return SignalOutput.model_validate_json(res.text)
        except Exception as e:
            logging.error(f"[AI ERROR] Gemini call failed: {e}")

    return SignalOutput(action=proposed_action, confidence=0.7, reasoning="Fallback: Executed on pure quantitative indicator alignment.")

# --- BACKGROUND SCANNING LOOP ---
async def background_scanning_loop():
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        while True:
            try:
                logging.info("Checking Twelve Data Multi-Timeframe Spot Gold Data (5m & 15m)...")
                df_5m = await fetch_timeframe_data(client, "5min")
                df_15m = await fetch_timeframe_data(client, "15min")

                if df_5m is None or df_15m is None:
                    logging.warning("Failed to fetch complete timeframe candles. Retrying next cycle.")
                    await asyncio.sleep(300)
                    continue

                df_5m = calculate_metrics(df_5m)
                df_15m = calculate_metrics(df_15m)

                curr_high = float(df_5m["high"].tail(3).max())
                curr_low = float(df_5m["low"].tail(3).min())
                update_open_trades(curr_high, curr_low)

                curr_price = float(df_5m["close"].iloc[-1])
                adx_15m = float(df_15m["adx"].iloc[-1])
                stoch_15m = float(df_15m["stoch_k"].iloc[-1])

                c_5m = float(df_5m["close"].iloc[-1])
                ema_50_5m = float(df_5m["ema_50"].iloc[-1])
                ema_200_5m = float(df_5m["ema_200"].iloc[-1])
                
                c_15m = float(df_15m["close"].iloc[-1])
                ema_200_15m = float(df_15m["ema_200"].iloc[-1])

                is_uptrend = (c_5m > ema_200_5m) and (ema_50_5m > ema_200_5m) and (c_15m > ema_200_15m)
                is_downtrend = (c_5m < ema_200_5m) and (ema_50_5m < ema_200_5m) and (c_15m < ema_200_15m)

                divergence = check_divergence(df_5m)

                stoch_k_curr = float(df_5m["stoch_k"].iloc[-1])
                stoch_k_prev = float(df_5m["stoch_k"].iloc[-2])
                stoch_d_curr = float(df_5m["stoch_d"].iloc[-1])
                stoch_d_prev = float(df_5m["stoch_d"].iloc[-2])

                if adx_15m > 30.0:
                    buy_stoch_limit = 55.0
                    sell_stoch_limit = 45.0
                else:
                    buy_stoch_limit = 40.0
                    sell_stoch_limit = 60.0

                stoch_buy_cross = (stoch_k_prev <= stoch_d_prev) and (stoch_k_curr > stoch_d_curr) and (stoch_k_curr < buy_stoch_limit)
                stoch_sell_cross = (stoch_k_prev >= stoch_d_prev) and (stoch_k_curr < stoch_d_curr) and (stoch_k_curr > sell_stoch_limit)

                proposed_action = "HOLD"
                trigger_type = "None"

                if divergence == "Bullish Divergence (Lower Price Low + Higher Stoch Low)":
                    proposed_action = "BUY"
                    trigger_type = "Divergence Reversal"

                elif divergence == "Bearish Divergence (Higher Price High + Lower Stoch High)":
                    proposed_action = "SELL"
                    trigger_type = "Divergence Reversal"

                elif is_uptrend and stoch_buy_cross:
                    proposed_action = "BUY"
                    trigger_type = "High Trend Continuation" if adx_15m > 30.0 else "Trend Setup"
                elif is_downtrend and stoch_sell_cross:
                    proposed_action = "SELL"
                    trigger_type = "High Trend Continuation" if adx_15m > 30.0 else "Trend Setup"

                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor(cursor_factory=RealDictCursor)
                        cursor.execute("""
                            SELECT entry_price, outcome FROM signals 
                            WHERE status = 'EXECUTED' AND action = %s 
                            ORDER BY id DESC LIMIT 1
                        """, (str(proposed_action),))
                        last_trade = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if last_trade:
                            last_entry_price = float(last_trade['entry_price'])
                            last_outcome = str(last_trade['outcome'])
                            
                            required_distance = 6.0 if last_outcome == "PENDING" else 3.0

                            if abs(curr_price - last_entry_price) < required_distance:
                                logging.info(f"[STATE-AWARE COOLDOWN] Skipping {proposed_action}: Price within ${required_distance:.2f} of previous trade at ${last_entry_price:.2f} (Status: {last_outcome}).")
                                proposed_action = "HOLD"
                    except Exception as cd_err:
                        logging.error(f"[STATE-AWARE COOLDOWN ERROR] {cd_err}")

                if proposed_action == "HOLD":
                    logging.info(f"[MARKET SCAN] Price: ${curr_price:.2f} | 5M EMA 200: ${ema_200_5m:.2f} | 15M ADX: {adx_15m:.1f} | Status: HOLD (No entry setup)")
                else:
                    logging.info(f"[MARKET SCAN] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI Analysis...")
                    ai_decision = await analyze_signal_with_ai(proposed_action, curr_price, df_5m, df_15m, divergence)

                    atr_5m = float(df_5m["atr"].iloc[-1])
                    sl_dist = float(max(4.50, min(8.00, atr_5m * 2.0)))
                    tp1_mult = 1.5
                    tp2_mult = 3.0

                    if proposed_action == "BUY":
                        sl_price = float(curr_price - sl_dist)
                        tp1_price = float(curr_price + (sl_dist * tp1_mult))
                        tp2_price = float(curr_price + (sl_dist * tp2_mult))
                    else:
                        sl_price = float(curr_price + sl_dist)
                        tp1_price = float(curr_price - (sl_dist * tp1_mult))
                        tp2_price = float(curr_price - (sl_dist * tp2_mult))

                    if ai_decision.action == proposed_action:
                        new_id = log_trade_signal("EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_15m, stoch_15m, divergence, ai_decision.reasoning)

                        id_tag = f" #{new_id}" if new_id else ""
                        msg = (
                            f"⚡ *STOCH RSI TRADE SIGNAL{id_tag}*\n\n"
                            f"Asset: *XAUUSD (Gold Spot)*\n"
                            f"Action: *{proposed_action}*\n"
                            f"Type: *{trigger_type}*\n"
                            f"Entry Price: *${curr_price:.2f}*\n\n"
                            f"Stop Loss (SL): *${sl_price:.2f}*\n"
                            f"Take Profit 1 (TP1): *${tp1_price:.2f}* (1:{tp1_mult:.1f} RRR)\n"
                            f"Take Profit 2 (TP2): *${tp2_price:.2f}* (1:{tp2_mult:.1f} RRR)\n\n"
                            f"INDICATOR METRICS:\n"
                            f"- Setup Type: {trigger_type}\n"
                            f"- Divergence Context: {divergence}\n"
                            f"- 15M
