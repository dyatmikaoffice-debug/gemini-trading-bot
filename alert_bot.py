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

# --- DATABASE CONNECTION & AUTO-MIGRATION INITIALIZATION ---
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
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        migrations = [
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS timestamp TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS trigger_type TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS sl REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS sl_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS tp1 REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS tp1_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS tp2 REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS tp2_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS confidence REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_15m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS stoch_rsi_15m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS divergence_type TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS reasoning TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS outcome TEXT DEFAULT 'PENDING';",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS exit_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS outcome_timestamp TEXT;"
        ]
        
        for query in migrations:
            cursor.execute(query)
            
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[NEON DATABASE] Full schema verified and missing columns auto-migrated.")
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to initialize database schema: {e}")

def log_trade_signal(status: str, action: str, trigger_type: str, price: float, sl: float, tp1: float, tp2: float, confidence: float, adx_15m: float, stoch_rsi_15m: float, divergence_type: str, reasoning: str):
    if not DATABASE_URL:
        return None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        wib_time = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        
        price_val = float(price) if price is not None else 0.0
        sl_val = float(sl) if sl is not None else 0.0
        tp1_val = float(tp1) if tp1 is not None else 0.0
        tp2_val = float(tp2) if tp2 is not None else 0.0
        conf_val = float(confidence) if confidence is not None else 0.0
        adx_val = float(adx_15m) if adx_15m is not None else 0.0
        stoch_val = float(stoch_rsi_15m) if stoch_rsi_15m is not None else 0.0

        cursor.execute("""
            INSERT INTO signals (timestamp, status, action, trigger_type, price, entry_price, sl, sl_price, tp1, tp1_price, tp2, tp2_price, confidence, adx_15m, stoch_rsi_15m, divergence_type, reasoning, outcome, outcome_timestamp, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """, (str(wib_time), str(status), str(action), str(trigger_type), price_val, price_val, sl_val, sl_val, tp1_val, tp1_val, tp2_val, tp2_val, conf_val, adx_val, stoch_val, str(divergence_type), str(reasoning), "PENDING", ""))
        
        inserted_row = cursor.fetchone()
        new_id = inserted_row['id'] if inserted_row and 'id' in inserted_row else None
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
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED' AND (outcome = 'PENDING' OR outcome = 'WIN (TP1 HIT)')")
        open_trades = cursor.fetchall()
        
        if not open_trades:
            cursor.close()
            conn.close()
            return

        wib_now = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        c_high = float(current_high)
        c_low = float(current_low)

        for trade in open_trades:
            trade_id = trade['id']
            action = trade['action']
            
            entry_price = float(trade['entry_price'] if trade.get('entry_price') is not None else trade.get('price', 0.0))
            sl = float(trade['sl_price'] if trade.get('sl_price') is not None else trade.get('sl', 0.0))
            tp1 = float(trade['tp1_price'] if trade.get('tp1_price') is not None else trade.get('tp1', 0.0))
            tp2 = float(trade['tp2_price'] if trade.get('tp2_price') is not None else trade.get('tp2', 0.0))
            
            current_outcome = trade['outcome']
            new_outcome = None
            exit_price = None

            if action == "BUY":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 > 0 and c_high >= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_low <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1
                elif sl > 0 and c_low <= sl:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 > 0 and c_high >= tp2:
                    new_outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 > 0 and c_high >= tp1:
                    new_outcome = "WIN (TP1 HIT)"
                    exit_price = tp1

            elif action == "SELL":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 > 0 and c_low <= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_high >= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1
                elif sl > 0 and c_high >= sl:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 > 0 and c_low <= tp2:
                    new_outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 > 0 and c_low <= tp1:
                    new_outcome = "WIN (TP1 HIT)"
                    exit_price = tp1

            if new_outcome and new_outcome != current_outcome:
                cursor.execute("""
                    UPDATE signals 
                    SET outcome = %s, exit_price = %s, outcome_timestamp = %s 
                    WHERE id = %s
                """, (new_outcome, float(exit_price), wib_now, trade_id))
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

# --- INDICATORS CALCULATIONS FOR M1 SCALPING ---
def calculate_metrics_m1(df: pd.DataFrame):
    df = df.tail(100).copy()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["close"]).rolling(window=20).sum() / df["close"].rolling(window=20).sum()
    
    # ATR Calculation on M1
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    df["atr"] = df["tr"].rolling(window=14).mean()
    return df

# --- INDICATORS CALCULATIONS FOR 5M TIMEFRAME (PULLBACK & TREND CONTEXT) ---
def calculate_metrics_5m(df: pd.DataFrame):
    df = df.tail(100).copy()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["close"]).rolling(window=20).sum() / df["close"].rolling(window=20).sum()

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
    return df

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
async def analyze_signal_with_ai(proposed_action: str, trigger_type: str, current_price: float, df_1m: pd.DataFrame, df_5m: pd.DataFrame):
    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD) Scalping.
A technical scalp trigger ({trigger_type}) suggests a {proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 1-Minute: Close=${float(df_1m['close'].iloc[-1]):.2f}, EMA 9=${float(df_1m['ema_9'].iloc[-1]):.2f}, EMA 21=${float(df_1m['ema_21'].iloc[-1]):.2f}, VWAP=${float(df_1m['vwap'].iloc[-1]):.2f}.
2. 5-Minute: Close=${float(df_5m['close'].iloc[-1]):.2f}, 5M EMA 9=${float(df_5m['ema_9'].iloc[-1]):.2f}, 5M VWAP=${float(df_5m['vwap'].iloc[-1]):.2f}, ADX Trend Strength=${float(df_5m['adx'].iloc[-1]):.1f}.

CRITICAL SCALP VETO RULES:
- VETO if proposed BUY is below M1/M5 VWAP or proposed SELL is above M1/M5 VWAP (fighting intraday balance).
- VETO if 5M ADX is below 12.0 indicating complete market freeze / zero volatility.

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

# --- BACKGROUND SCANNING LOOP (WIDE SL BUFFERS & 5M PULLBACK) ---
async def background_scanning_loop():
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        while True:
            try:
                now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
                current_hour_wib = now_wib.hour

                # SESSION WINDOWING: Active London/NY hours (14:00 - 23:00 WIB)
                if not (14 <= current_hour_wib < 23):
                    logging.info(f"[SLEEP MODE] WIB Time: {now_wib.strftime('%H:%M')} | Outside active trading session. Pausing API calls...")
                    await asyncio.sleep(300)
                    continue

                logging.info(f"[ACTIVE SCAN] WIB Time: {now_wib.strftime('%H:%M')} | Scanning setups (5M Pullback / 1M Cross)...")
                df_1m = await fetch_timeframe_data(client, "1min")
                df_5m = await fetch_timeframe_data(client, "5min")

                if df_1m is None or df_5m is None:
                    logging.warning("Failed to fetch M1/M5 candles. Retrying next cycle.")
                    await asyncio.sleep(120)
                    continue

                df_1m = calculate_metrics_m1(df_1m)
                df_5m = calculate_metrics_5m(df_5m)

                curr_high = float(df_1m["high"].tail(3).max())
                curr_low = float(df_1m["low"].tail(3).min())
                update_open_trades(curr_high, curr_low)

                # M1 Current Candle
                curr_price = float(df_1m["close"].iloc[-1])
                curr_open = float(df_1m["open"].iloc[-1])
                vwap_1m = float(df_1m["vwap"].iloc[-1])
                high_1m_prev = float(df_1m["high"].iloc[-2])
                low_1m_prev = float(df_1m["low"].iloc[-2])

                ema9_1m_curr = float(df_1m["ema_9"].iloc[-1])
                ema9_1m_prev = float(df_1m["ema_9"].iloc[-2])
                ema21_1m_curr = float(df_1m["ema_21"].iloc[-1])
                ema21_1m_prev = float(df_1m["ema_21"].iloc[-2])

                # 5M Indicators
                adx_5m = float(df_5m["adx"].iloc[-1])
                ema9_5m_curr = float(df_5m["ema_9"].iloc[-1])
                ema9_5m_prev = float(df_5m["ema_9"].iloc[-2])
                ema21_5m_curr = float(df_5m["ema_21"].iloc[-1])
                vwap_5m_curr = float(df_5m["vwap"].iloc[-1])
                vwap_5m_prev = float(df_5m["vwap"].iloc[-2])

                open_5m_prev = float(df_5m["open"].iloc[-2])
                close_5m_prev = float(df_5m["close"].iloc[-2])
                high_5m_prev = float(df_5m["high"].iloc[-2])
                low_5m_prev = float(df_5m["low"].iloc[-2])

                # --- RULE 1: DETECT CUT/PIERCE THROUGH VWAP ON 5M CHART ---
                bullish_cut_vwap_5m = (open_5m_prev < vwap_5m_prev) and (close_5m_prev > vwap_5m_prev)
                bearish_cut_vwap_5m = (open_5m_prev > vwap_5m_prev) and (close_5m_prev < vwap_5m_prev)

                # --- RULE 2: 5-MINUTE CHART PULLBACK TOUCH & REVERSAL CONDITIONS ---
                bullish_pullback_5m = (
                    (ema9_5m_curr > ema21_5m_curr) and 
                    (low_5m_prev <= max(ema9_5m_prev, vwap_5m_prev)) and 
                    (close_5m_prev >= vwap_5m_prev) and 
                    not bearish_cut_vwap_5m and 
                    (curr_price > curr_open) and 
                    (curr_price > high_1m_prev)
                )

                bearish_pullback_5m = (
                    (ema9_5m_curr < ema21_5m_curr) and 
                    (high_5m_prev >= min(ema9_5m_prev, vwap_5m_prev)) and 
                    (close_5m_prev <= vwap_5m_prev) and 
                    not bullish_cut_vwap_5m and 
                    (curr_price < curr_open) and 
                    (curr_price < low_1m_prev)
                )

                # 1M Crossover Triggers
                bullish_cross_1m = (ema9_1m_prev <= ema21_1m_prev) and (ema9_1m_curr > ema21_1m_curr) and not bearish_cut_vwap_5m
                bearish_cross_1m = (ema9_1m_prev >= ema21_1m_prev) and (ema9_1m_curr < ema21_1m_curr) and not bullish_cut_vwap_5m

                proposed_action = "HOLD"
                trigger_type = "None"

                if (bullish_cross_1m or bullish_pullback_5m) and (curr_price > vwap_1m) and (curr_price > vwap_5m_curr):
                    proposed_action = "BUY"
                    trigger_type = "5M VWAP Pullback" if bullish_pullback_5m else "M1 VWAP Crossover"

                elif (bearish_cross_1m or bearish_pullback_5m) and (curr_price < vwap_1m) and (curr_price < vwap_5m_curr):
                    proposed_action = "SELL"
                    trigger_type = "5M VWAP Pullback" if bearish_pullback_5m else "M1 VWAP Crossover"

                # State-Aware Cooldown Distance Check ($5.00 pending / $3.00 closed)
                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT COALESCE(entry_price, price, 0) as entry_p, outcome FROM signals 
                            WHERE status = 'EXECUTED' AND action = %s 
                            ORDER BY id DESC LIMIT 1
                        """, (str(proposed_action),))
                        last_trade = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if last_trade:
                            last_entry_price = float(last_trade['entry_p'])
                            last_outcome = str(last_trade['outcome']) if last_trade.get('outcome') else "PENDING"
                            required_distance = 5.00 if last_outcome == "PENDING" else 3.00

                            if abs(curr_price - last_entry_price) < required_distance:
                                logging.info(f"[SCALP COOLDOWN] Skipping {proposed_action}: Price within ${required_distance:.2f} of previous trade at ${last_entry_price:.2f}.")
                                proposed_action = "HOLD"
                    except Exception as cd_err:
                        logging.error(f"[COOLDOWN ERROR] {cd_err}")

                if proposed_action == "HOLD":
                    logging.info(f"[MARKET SCAN] Price: ${curr_price:.2f} | 5M VWAP: ${vwap_5m_curr:.2f} | 5M ADX: {adx_5m:.1f} | Status: HOLD")
                else:
                    logging.info(f"[MARKET SCAN] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI Analysis...")
                    ai_decision = await analyze_signal_with_ai(proposed_action, trigger_type, curr_price, df_1m, df_5m)

                    # EXPANDED & WIDER SL/TP DISTANCES FOR XAU/USD (Comfortable Breathing Room)
                    atr_1m = float(df_1m["atr"].iloc[-1])
                    sl_dist = float(max(4.50, min(8.50, atr_1m * 3.5)))
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
                        new_id = log_trade_signal("EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning)
                        id_tag = f" #{new_id}" if new_id else ""
                        msg = (
                            f"⚡ *5M/1M VWAP SCALP SIGNAL{id_tag}*\n\n"
                            f"Asset: *XAUUSD (Gold Spot)*\n"
                            f"Action: *{proposed_action}*\n"
                            f"Type: *{trigger_type}*\n"
                            f"Entry Price: *${curr_price:.2f}*\n\n"
                            f"Stop Loss (SL): *${sl_price:.2f}*\n"
                            f"Take Profit 1 (TP1): *${tp1_price:.2f}* (1:{tp1_mult:.1f} RRR)\n"
                            f"Take Profit 2 (TP2): *${tp2_price:.2f}* (1:{tp2_mult:.1f} RRR)\n\n"
                            f"INDICATOR METRICS:\n"
                            f"- Setup: {trigger_type}\n"
                            f"- 5M VWAP Anchor: ${vwap_5m_curr:.2f}\n"
                            f"- 5M ADX Volatility: {adx_5m:.1f}\n\n"
                            f"Reasoning: {ai_decision.reasoning}"
                        )
                        await send_telegram_alert(client, msg)
                    else:
                        log_trade_signal("VETOED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning)

                del df_1m, df_5m
                gc.collect()

            except Exception as e:
                logging.error(f"[SCAN LOOP ERROR] {e}")

            await asyncio.sleep(120)

# --- FASTAPI LIFESPAN & AUTOMATED WEBHOOK SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if TELEGRAM_BOT_TOKEN and APP_URL:
        try:
            webhook_endpoint = f"{APP_URL.rstrip('/')}/telegram-webhook"
            set_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={webhook_endpoint}"
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                res = await client.get(set_url)
                logging.info(f"[AUTO WEBHOOK SETUP] Response: {res.text}")
        except Exception as e:
            logging.error(f"[AUTO WEBHOOK SETUP ERROR] Failed: {e}")
            
    scan_task = asyncio.create_task(background_scanning_loop())
    yield
    scan_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot scanner and webhook server active."}

# --- WEBHOOK ENDPOINT FOR TELEGRAM COMMANDS ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        message = data.get("message", {})
        raw_text = message.get("text", "").strip().lower()
        sender_chat_id = str(message.get("chat", {}).get("id", ""))

        if not sender_chat_id or not raw_text:
            return {"status": "ignored"}

        logging.info(f"[WEBHOOK RECEIVED] Chat ID: {sender_chat_id} | Command: '{raw_text}'")

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            if raw_text in ["/help", "/start"]:
                reply = (
                    "🤖 *TRADING BOT COMMANDS:*\n\n"
                    "• `/stats` - Comprehensive Win-Rate & Risk Analytics Dashboard\n"
                    "• `/pips` - Detailed Gross/Net Pips & USD Profit Breakdown (0.01 Lot)\n"
                    "• `/logs` - Detailed View of Last 10 Trades & Outcomes\n"
                    "• `/help` - Display Command Menu"
                )
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/stats":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("SELECT COUNT(*) as total FROM signals WHERE status = 'EXECUTED'")
                    total_executed = cur.fetchone()['total'] or 0

                    cur.execute("SELECT COUNT(*) as vetoes FROM signals WHERE status = 'VETOED'")
                    total_vetoes = cur.fetchone()['vetoes'] or 0

                    cur.execute("SELECT COUNT(*) as pending FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING'")
                    total_pending = cur.fetchone()['pending'] or 0

                    cur.execute("SELECT COUNT(*) as tp1_wins FROM signals WHERE outcome LIKE 'WIN (TP1%' OR outcome LIKE 'CLOSED%'")
                    tp1_wins = cur.fetchone()['tp1_wins'] or 0

                    cur.execute("SELECT COUNT(*) as tp2_wins FROM signals WHERE outcome LIKE 'WIN (TP2%'")
                    tp2_wins = cur.fetchone()['tp2_wins'] or 0

                    cur.execute("SELECT COUNT(*) as losses FROM signals WHERE outcome LIKE 'LOSS%'")
                    losses = cur.fetchone()['losses'] or 0

                    cur.execute("SELECT action, COALESCE(entry_price, price, 0) as entry_p, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                    closed_trades = cur.fetchall()

                    total_pips = 0.0
                    win_pips = 0.0
                    loss_pips = 0.0
                    total_wins_count = tp1_wins + tp2_wins

                    for t in closed_trades:
                        entry = float(t['entry_p'])
                        exit_p = float(t['exit_price'])
                        action = t['action']
                        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
                        pips = diff * 10.0
                        total_pips += pips
                        if pips > 0:
                            win_pips += pips
                        else:
                            loss_pips += abs(pips)

                    win_rate = (total_wins_count / total_executed * 100) if total_executed > 0 else 0.0
                    est_dollar = total_pips * 0.10
                    avg_win = (win_pips / total_wins_count) if total_wins_count > 0 else 0.0
                    avg_loss = (loss_pips / losses) if losses > 0 else 0.0
                    profit_factor = (win_pips / loss_pips) if loss_pips > 0 else (win_pips if win_pips > 0 else 0.0)

                    cur.close()
                    conn.close()

                    reply = (
                        f"📊 *PERFORMANCE ANALYTICS DASHBOARD*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💰 *NET PIPS & PROFIT:*\n"
                        f"• Net Pips: *{total_pips:+.1f} pips*\n"
                        f"• Est. Profit (0.01 Lot): *${est_dollar:+.2f}*\n\n"
                        f"📈 *WIN / LOSS BREAKDOWN:*\n"
                        f"• Total Executed: *{total_executed}*\n"
                        f"• Total Wins: *{total_wins_count} ({win_rate:.1f}%)*\n"
                        f"  └─ Hit TP1 (BE Runner): *{tp1_wins}*\n"
                        f"  └─ Hit TP2 (Full Target): *{tp2_wins}*\n"
                        f"• Total Losses (SL Hit): *{losses}*\n"
                        f"• Active Pending: *{total_pending}*\n\n"
                        f"⚡ *SYSTEM & AI EFFICIENCY:*\n"
                        f"• Total Signals: *{total_executed + total_vetoes}*\n"
                        f"• AI Vetoed Signals: *{total_vetoes}*\n\n"
                        f"🎯 *RISK & TRADE METRICS:*\n"
                        f"• Avg Win: *+{avg_win:.1f} pips* | Avg Loss: *-{avg_loss:.1f} pips*\n"
                        f"• Profit Factor: *{profit_factor:.2f}*\n"
                        f"• Win Rate: *{win_rate:.1f}%*"
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

                except Exception as db_err:
                    logging.error(f"[WEBHOOK ERROR /stats] {db_err}")
                    await send_telegram_alert(client, f"⚠️ Error querying stats: {db_err}", target_chat_id=sender_chat_id)

            elif raw_text == "/pips":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT action, COALESCE(entry_price, price, 0) as entry_p, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                    trades = cur.fetchall()

                    total_pips = 0.0
                    gross_win_pips = 0.0
                    gross_loss_pips = 0.0
                    winning_trades_count = 0
                    losing_trades_count = 0

                    for t in trades:
                        entry = float(t['entry_p'])
                        exit_p = float(t['exit_price'])
                        action = t['action']
                        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
                        pips = diff * 10.0
                        total_pips += pips

                        if pips > 0:
                            gross_win_pips += pips
                            winning_trades_count += 1
                        elif pips < 0:
                            gross_loss_pips += abs(pips)
                            losing_trades_count += 1

                    avg_win_pips = (gross_win_pips / winning_trades_count) if winning_trades_count > 0 else 0.0
                    avg_loss_pips = (gross_loss_pips / losing_trades_count) if losing_trades_count > 0 else 0.0
                    est_profit_usd = total_pips * 0.10

                    cur.close()
                    conn.close()

                    reply = (
                        f"💵 *DETAILED PIPS & EARNINGS REPORT*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 *SUMMARY:*\n"
                        f"• Total Net Pips: *{total_pips:+.1f} pips*\n"
                        f"• Net Profit (0.01 Lot): *${est_profit_usd:+.2f}*\n\n"
                        f"📈 *PIPS BREAKDOWN:*\n"
                        f"• Gross Gain: *+{gross_win_pips:.1f} pips*\n"
                        f"• Gross Loss: *-{gross_loss_pips:.1f} pips*\n\n"
                        f"🎯 *AVERAGE METRICS:*\n"
                        f"• Avg Win Trade: *+{avg_win_pips:.1f} pips*\n"
                        f"• Avg Loss Trade: *-{avg_loss_pips:.1f} pips*\n"
                        f"• Pip Efficiency Ratio: *{(gross_win_pips / (gross_loss_pips + 1e-5)):.2f}*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💡 *Catatan:* Dihitung pada $0.10/pip (0.01 lot XAU/USD)."
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

                except Exception as err:
                    logging.error(f"[WEBHOOK ERROR /pips] {err}")
                    await send_telegram_alert(client, f"⚠️ Error calculating pips: {err}", target_chat_id=sender_chat_id)

            elif raw_text == "/logs":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT id, action, 
                               COALESCE(entry_price, price, 0) as entry_p, 
                               exit_price, 
                               COALESCE(outcome, 'PENDING') as outcome_val, 
                               COALESCE(timestamp, created_at::text, 'N/A') as log_time 
                        FROM signals 
                        WHERE status = 'EXECUTED' 
                        ORDER BY id DESC LIMIT 10
                    """)
                    logs = cur.fetchall()
                    cur.close()
                    conn.close()

                    if not logs:
                        reply = "📜 *LAST 10 TRADE LOGS:*\n\n_Belum ada transaksi yang tereksekusi di database._"
                    else:
                        reply = "📜 *LAST 10 DETAILED TRADE LOGS:*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        for l in logs:
                            trade_id = l['id']
                            action = l['action']
                            entry = float(l['entry_p'])
                            exit_p = float(l['exit_price']) if l.get('exit_price') is not None else None
                            outcome = l['outcome_val']
                            date_str = str(l['log_time'])

                            if exit_p is not None:
                                diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
                                pips = diff * 10.0
                                pip_str = f"*{pips:+.1f} pips*"
                            else:
                                pip_str = "*ACTIVE / IN PROGRESS*"

                            if "WIN" in outcome or "CLOSED" in outcome:
                                icon = "🟢"
                            elif "LOSS" in outcome:
                                icon = "🔴"
                            else:
                                icon = "🟡"

                            reply += (
                                f"{icon} *ID #{trade_id} | {action} XAU/USD*\n"
                                f"• Entry: ${entry:.2f} → Exit: *${(exit_p if exit_p else 0.0):.2f}*\n"
                                f"• Outcome: *{outcome}*\n"
                                f"• Result: {pip_str} | Time: {date_str}\n"
                                f"──────────────────────────\n"
                            )

                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

                except Exception as log_err:
                    logging.error(f"[WEBHOOK ERROR /logs] {log_err}")
                    await send_telegram_alert(client, f"⚠️ Error querying logs: {log_err}", target_chat_id=sender_chat_id)

    except Exception as e:
        logging.error(f"[WEBHOOK ERROR] {e}")

    return {"status": "ok"}
