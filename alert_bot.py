import os
import json
import asyncio
import sqlite3
import psycopg2
import gc
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from openai import OpenAI

# --- ENVIRONMENT VARIABLES & SANITIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAW_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Clean Telegram token (strip 'bot' if user included it in Env Vars to prevent double prefix)
if RAW_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = RAW_BOT_TOKEN[3:]
else:
    TELEGRAM_BOT_TOKEN = RAW_BOT_TOKEN

# Initialize AI Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

SYMBOL = "XAU/USD"
EXCHANGE = "TDEX"

# Cooldown Tracker & State
LAST_SIGNAL_ACTION = "HOLD"
LAST_SIGNAL_PRICE = 0.0

class SignalOutput(BaseModel):
    action: str = Field(default="HOLD", description="BUY, SELL, or HOLD")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(default="Market conditions do not favor entry.", description="2 clean sentences explaining the decision")

# --- DATABASE CONNECTION & INITIALIZATION ---
def get_db_connection():
    """Returns a connection to Neon PostgreSQL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing.")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Initializes PostgreSQL database table for trade tracking."""
    if not DATABASE_URL:
        print("[WARNING] DATABASE_URL not set. Database logging disabled.")
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                action TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                price REAL NOT NULL,
                sl REAL,
                tp1 REAL,
                tp2 REAL,
                confidence REAL,
                adx_15m REAL,
                stoch_rsi_15m REAL,
                divergence_type TEXT,
                reasoning TEXT,
                outcome TEXT DEFAULT 'PENDING',
                exit_price REAL,
                outcome_timestamp TEXT
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("[NEON DATABASE] Table structure verified successfully.")
    except Exception as e:
        print(f"[NEON DB ERROR] Failed to initialize database: {e}")

def log_trade_signal(status: str, action: str, trigger_type: str, price: float, sl: float, tp1: float, tp2: float, confidence: float, adx_15m: float, stoch_rsi_15m: float, divergence_type: str, reasoning: str):
    """Logs trade executions and AI veto decisions to Neon database using native Python types."""
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        wib_time = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        
        # Cast NumPy types explicitly to standard Python floats
        price_val = float(price) if price is not None else 0.0
        sl_val = float(sl) if sl is not None else 0.0
        tp1_val = float(tp1) if tp1 is not None else 0.0
        tp2_val = float(tp2) if tp2 is not None else 0.0
        conf_val = float(confidence) if confidence is not None else 0.0
        adx_val = float(adx_15m) if adx_15m is not None else 0.0
        stoch_val = float(stoch_rsi_15m) if stoch_rsi_15m is not None else 0.0

        cursor.execute("""
            INSERT INTO signals 
            (timestamp, status, action, trigger_type, price, sl, tp1, tp2, confidence, adx_15m, stoch_rsi_15m, divergence_type, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (wib_time, status, action, trigger_type, price_val, sl_val, tp1_val, tp2_val, conf_val, adx_val, stoch_val, str(divergence_type), str(reasoning)))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[NEON DB LOGGED] Status: {status} | Action: {action} | Price: ${price_val:.2f}")
    except Exception as e:
        print(f"[NEON DB ERROR] Failed to log signal: {e}")

def update_open_trades(current_high: float, current_low: float):
    """Checks open pending trades in Neon against live candle high/low for TP/SL hits."""
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING'")
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
            sl = float(trade['sl']) if trade['sl'] is not None else None
            tp1 = float(trade['tp1']) if trade['tp1'] is not None else None
            tp2 = float(trade['tp2']) if trade['tp2'] is not None else None

            outcome = None
            exit_price = None

            if action == "BUY":
                if sl is not None and c_low <= sl:
                    outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 is not None and c_high >= tp2:
                    outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 is not None and c_high >= tp1:
                    outcome = "WIN (TP1 HIT)"
                    exit_price = tp1
            elif action == "SELL":
                if sl is not None and c_high >= sl:
                    outcome = "LOSS (SL HIT)"
                    exit_price = sl
                elif tp2 is not None and c_low <= tp2:
                    outcome = "WIN (TP2 HIT)"
                    exit_price = tp2
                elif tp1 is not None and c_low <= tp1:
                    outcome = "WIN (TP1 HIT)"
                    exit_price = tp1

            if outcome:
                cursor.execute("""
                    UPDATE signals 
                    SET outcome = %s, exit_price = %s, outcome_timestamp = %s
                    WHERE id = %s
                """, (outcome, float(exit_price), wib_now, trade_id))
                conn.commit()
                print(f"[TRADE CLOSED] Signal ID {trade_id} -> {outcome} at ${exit_price:.2f}")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[NEON DB ERROR] Failed to update trade outcomes: {e}")

# --- MARKET DATA FETCHING ---
async def fetch_timeframe_data(client: httpx.AsyncClient, timeframe: str, outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={timeframe}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    res = await client.get(url)
    if res.status_code != 200 or not res.text:
        return None
    data = res.json()
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
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["close"]).rolling(window=20).sum() / df["close"].rolling(window=20).sum()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    df["rsi"] = 100 - (100 / (1 + rs))

    rsi_min = df["rsi"].rolling(window=14).min()
    rsi_max = df["rsi"].rolling(window=14).max()
    stoch_rsi = (df["rsi"] - rsi_min) / (rsi_max - rsi_min)
    df["stoch_k"] = stoch_rsi.rolling(window=3).mean() * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ADX Calculation
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]
    
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    tr14 = df["tr"].rolling(14).sum()
    plus_di = 100 * (df["plus_dm"].rolling(14).sum() / tr14)
    minus_di = 100 * (df["minus_dm"].rolling(14).sum() / tr14)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
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

    if p_now < p_prev and s_now > s_prev and s_now < 30:
        return "Bullish Divergence (Lower Price Low + Higher Stoch Low)"

    p_prev_max = float(df["close"].iloc[-5:-1].max())
    s_prev_max = float(df["stoch_k"].iloc[-5:-1].max())
    if p_now > p_prev_max and s_now < s_prev_max and s_now > 70:
        return "Bearish Divergence (Higher Price High + Lower Stoch High)"

    return "None"

# --- TELEGRAM NOTIFICATIONS ---
async def send_telegram_alert(client: httpx.AsyncClient, text: str, target_chat_id: str = None):
    chat_id = target_chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print("[TELEGRAM ERROR] Missing token or chat_id")
        return
    url = f"https://api.telegram.com/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": str(chat_id), "text": text, "parse_mode": "Markdown"}
    
    try:
        res = await client.post(url, json=payload)
        if res.status_code != 200:
            print(f"[TELEGRAM API WARNING] Code {res.status_code}: {res.text}. Retrying plain text.")
            payload_plain = {"chat_id": str(chat_id), "text": text}
            res_plain = await client.post(url, json=payload_plain)
            if res_plain.status_code != 200:
                print(f"[TELEGRAM API ERROR] Failed plain text: {res_plain.text}")
            else:
                print(f"[TELEGRAM SENT] Delivered plain text to Chat ID {chat_id}")
        else:
            print(f"[TELEGRAM SENT] Delivered Markdown to Chat ID {chat_id}")
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")

# --- AI ANALYST EVALUATION ---
async def analyze_signal_with_ai(proposed_action: str, current_price: float, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, divergence: str):
    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD).
A technical trigger suggests a {proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 5-Minute: Close=${float(df_5m['close'].iloc[-1]):.2f}, EMA 50=${float(df_5m['ema_50'].iloc[-1]):.2f}, VWAP=${float(df_5m['vwap'].iloc[-1]):.2f}, Stoch RSI %K=${float(df_5m['stoch_k'].iloc[-1]):.1f}.
2. 15-Minute: Close=${float(df_15m['close'].iloc[-1]):.2f}, ADX Trend Strength=${float(df_15m['adx'].iloc[-1]):.1f}, Stoch RSI %K=${float(df_15m['stoch_k'].iloc[-1]):.1f}.
3. 1-Hour: Close=${float(df_1h['close'].iloc[-1]):.2f}, EMA 50=${float(df_1h['ema_50'].iloc[-1]):.2f}, VWAP=${float(df_1h['vwap'].iloc[-1]):.2f}.
4. Divergence State: {divergence}.

CRITICAL VETO RULES:
- VETO if price is directly buying into 1H resistance or selling into 1H support.
- VETO if 15M ADX is below 18.0 indicating horizontal chop.
- VETO if 5M Stoch RSI is exhausted (>85 for BUY, <15 for SELL) without bullish/bearish divergence.

Respond strictly in valid JSON matching schema:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, "reasoning": "2 concise sentences explaining decision"}}
"""

    if GROQ_API_KEY:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            return SignalOutput(**data)
        except Exception as e:
            print(f"[AI WARNING] Groq call failed: {e}. Falling back to Gemini.")

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
            print(f"[AI ERROR] Gemini call failed: {e}")

    return SignalOutput(action=proposed_action, confidence=0.7, reasoning="Fallback: Executed on pure quantitative indicator alignment.")

# --- BACKGROUND SCANNING LOOP ---
async def background_scanning_loop():
    global LAST_SIGNAL_ACTION, LAST_SIGNAL_PRICE
    
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        while True:
            try:
                print("Checking Twelve Data Multi-Timeframe Spot Gold Data...")
                df_5m = await fetch_timeframe_data(client, "5min")
                df_15m = await fetch_timeframe_data(client, "15min")
                df_1h = await fetch_timeframe_data(client, "1h")

                if df_5m is None or df_15m is None or df_1h is None:
                    print("Failed to fetch complete timeframe candles. Retrying next cycle.")
                    await asyncio.sleep(360)
                    continue

                df_5m = calculate_metrics(df_5m)
                df_15m = calculate_metrics(df_15m)
                df_1h = calculate_metrics(df_1h)

                curr_high = float(df_5m["high"].iloc[-1])
                curr_low = float(df_5m["low"].iloc[-1])
                update_open_trades(curr_high, curr_low)

                curr_price = float(df_5m["close"].iloc[-1])
                adx_15m = float(df_15m["adx"].iloc[-1])
                stoch_15m = float(df_15m["stoch_k"].iloc[-1])
                
                # ADX Filter
                if adx_15m < 18.0:
                    print(f"Skipping: Low ADX ({adx_15m:.1f}) indicates horizontal chop.")
                    del df_5m, df_15m, df_1h
                    gc.collect()
                    await asyncio.sleep(360)
                    continue

                # Signal Logic Setup
                c_5m, ema_5m, vwap_5m = float(df_5m["close"].iloc[-1]), float(df_5m["ema_50"].iloc[-1]), float(df_5m["vwap"].iloc[-1])
                c_5m_prev, o_5m = float(df_5m["close"].iloc[-2]), float(df_5m["open"].iloc[-1])
                
                is_5m_bull = (c_5m > ema_5m) and (c_5m > vwap_5m) and (c_5m > o_5m)
                is_5m_bear = (c_5m < ema_5m) and (c_5m < vwap_5m) and (c_5m < o_5m)

                divergence = check_divergence(df_5m)

                stoch_k_curr = float(df_5m["stoch_k"].iloc[-1])
                stoch_k_prev = float(df_5m["stoch_k"].iloc[-2])
                stoch_buy_cross = (stoch_k_prev < 20) and (stoch_k_curr >= 20)
                stoch_sell_cross = (stoch_k_prev > 80) and (stoch_k_curr <= 80)

                proposed_action = "HOLD"
                trigger_type = "None"

                if divergence == "Bullish Divergence (Lower Price Low + Higher Stoch Low)":
                    proposed_action = "BUY"
                    trigger_type = "Divergence Reversal"
                elif divergence == "Bearish Divergence (Higher Price High + Lower Stoch High)":
                    proposed_action = "SELL"
                    trigger_type = "Divergence Reversal"
                elif is_5m_bull and stoch_buy_cross:
                    proposed_action = "BUY"
                    trigger_type = "Trend Setup"
                elif is_5m_bear and stoch_sell_cross:
                    proposed_action = "SELL"
                    trigger_type = "Trend Setup"

                # Distance & Cooldown Guard ($6.00 distance)
                if proposed_action != "HOLD":
                    if proposed_action == LAST_SIGNAL_ACTION and abs(curr_price - LAST_SIGNAL_PRICE) < 6.0:
                        print(f"Skipping: Price within $6.00 of previous {proposed_action} alert.")
                        proposed_action = "HOLD"

                if proposed_action != "HOLD":
                    ai_decision = await analyze_signal_with_ai(proposed_action, curr_price, df_5m, df_15m, df_1h, divergence)

                    atr_5m = float(df_5m["atr"].iloc[-1])
                    sl_dist = min(atr_5m * 1.5, 6.0)
                    tp1_mult = 1.5
                    tp2_mult = 4.0 if adx_15m > 35.0 else 2.5

                    if proposed_action == "BUY":
                        sl_price = curr_price - sl_dist
                        tp1_price = curr_price + (sl_dist * tp1_mult)
                        tp2_price = curr_price + (sl_dist * tp2_mult)
                    else:
                        sl_price = curr_price + sl_dist
                        tp1_price = curr_price - (sl_dist * tp1_mult)
                        tp2_price = curr_price - (sl_dist * tp2_mult)

                    if ai_decision.action == proposed_action:
                        LAST_SIGNAL_ACTION = proposed_action
                        LAST_SIGNAL_PRICE = curr_price

                        log_trade_signal("EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, ai_decision.confidence, adx_15m, stoch_15m, divergence, ai_decision.reasoning)

                        msg = (
                            f"⚡ *STOCH RSI TRADE SIGNAL*\n\n"
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
                            f"- 15M ADX Strength: {adx_15m:.1f}\n"
                            f"- 15M Stoch RSI: {stoch_15m:.1f}\n\n"
                            f"Reasoning: {ai_decision.reasoning}"
                        )
                        await send_telegram_alert(client, msg)
                    else:
                        log_trade_signal("VETOED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, ai_decision.confidence, adx_15m, stoch_15m, divergence, ai_decision.reasoning)

                del df_5m, df_15m, df_1h
                gc.collect()

            except Exception as e:
                print(f"Error in scanning loop: {e}")

            await asyncio.sleep(360)

# --- TELEGRAM POLLING LOOP ---
async def telegram_polling_loop():
    if not TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM ERROR] TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            del_res = await client.get(f"https://api.telegram.com/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook")
            print(f"[TELEGRAM] Webhook clear status: {del_res.status_code}")
        except Exception as e:
            print(f"[TELEGRAM WARNING] Webhook clear failed: {e}")

        offset = 0
        while True:
            try:
                res = await client.get(f"https://api.telegram.com/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=5")
                if res.status_code == 200 and res.text:
                    data = res.json()

                    if data.get("ok") and "result" in data:
                        for update in data["result"]:
                            offset = update["update_id"] + 1
                            message = update.get("message", {})
                            raw_text = message.get("text", "").strip().lower()
                            sender_chat_id = str(message.get("chat", {}).get("id", ""))

                            if not sender_chat_id or not raw_text:
                                continue

                            if raw_text.startswith("/help"):
                                help_msg = (
                                    "🤖 *TRADING BOT COMMANDS:*\n\n"
                                    "• `/stats` - View live win rate, TP/SL hits, and veto counts\n"
                                    "• `/logs` - View details of the last 5 signals\n"
                                    "• `/help` - Display command menu"
                                )
                                await send_telegram_alert(client, help_msg, target_chat_id=sender_chat_id)

                            elif raw_text.startswith("/logs"):
                                try:
                                    conn = get_db_connection()
                                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                                    cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 5")
                                    rows = cursor.fetchall()
                                    cursor.close()
                                    conn.close()

                                    if not rows:
                                        await send_telegram_alert(client, "No trade logs recorded yet.", target_chat_id=sender_chat_id)
                                    else:
                                        log_text = "📜 *LAST 5 TRADE LOGS:*\n\n"
                                        for r in rows:
                                            log_text += f"• *ID {r['id']}* | {r['action']} @ ${float(r['price']):.2f} | Status: `{r['status']}` ({r['outcome']})\n"
                                        await send_telegram_alert(client, log_text, target_chat_id=sender_chat_id)
                                except Exception as log_err:
                                    await send_telegram_alert(client, f"⚠️ Error querying logs: {log_err}", target_chat_id=sender_chat_id)

                            elif raw_text.startswith("/stats"):
                                try:
                                    conn = get_db_connection()
                                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                                    cursor.execute("SELECT * FROM signals")
                                    rows = cursor.fetchall()

                                    executed = [r for r in rows if r['status'] == 'EXECUTED']
                                    vetoed = [r for r in rows if r['status'] == 'VETOED']
                                    pending = [r for r in executed if r['outcome'] == 'PENDING']
                                    closed = [r for r in executed if r['outcome'] != 'PENDING']

                                    wins_tp1 = len([r for r in closed if 'TP1' in str(r['outcome'])])
                                    wins_tp2 = len([r for r in closed if 'TP2' in str(r['outcome'])])
                                    losses = len([r for r in closed if 'LOSS' in str(r['outcome'])])

                                    total_closed = len(closed)
                                    win_rate = ((wins_tp1 + wins_tp2) / total_closed * 100) if total_closed > 0 else 0.0

                                    cursor.close()
                                    conn.close()

                                    stats_msg = (
                                        f"📊 *SYSTEM PERFORMANCE STATS (NEON CLOUD)*\n\n"
                                        f"• Executed Signals: {len(executed)}\n"
                                        f"• AI Vetoes: {len(vetoed)}\n"
                                        f"• Pending Trades: {len(pending)}\n"
                                        f"• Closed Trades: {total_closed}\n"
                                        f"• Total Wins: {wins_tp1 + wins_tp2} (TP1: {wins_tp1} | TP2: {wins_tp2})\n"
                                        f"• Total Losses: {losses}\n"
                                        f"• Win Rate: *{win_rate:.1f}%*"
                                    )
                                    await send_telegram_alert(client, stats_msg, target_chat_id=sender_chat_id)
                                except Exception as db_err:
                                    await send_telegram_alert(client, f"⚠️ Error querying Neon DB stats: {db_err}", target_chat_id=sender_chat_id)
                else:
                    print(f"[TELEGRAM POLLING WARNING] Code {res.status_code}: {res.text}")

            except Exception as e:
                print(f"Error in Telegram polling loop: {e}")

            await asyncio.sleep(3)

# --- FASTAPI LIFESPAN & ROUTING ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scan_task = asyncio.create_task(background_scanning_loop())
    poll_task = asyncio.create_task(telegram_polling_loop())
    yield
    scan_task.cancel()
    poll_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot background scanner running on Neon PostgreSQL."}

@app.get("/logs")
def get_trade_logs(limit: int = 50):
    """Retrieves paper trading signal history from Neon."""
    if not DATABASE_URL:
        return {"error": "DATABASE_URL environment variable is missing."}
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"count": len(rows), "logs": rows}
    except Exception as e:
        return {"error": str(e)}

@app.get("/stats")
def get_trade_stats():
    """Retrieves live win/loss performance stats from Neon."""
    if not DATABASE_URL:
        return {"error": "DATABASE_URL environment variable is missing."}
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED'")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        closed = [r for r in rows if r['outcome'] != 'PENDING']
        wins = [r for r in closed if 'WIN' in str(r['outcome'])]
        losses = [r for r in closed if 'LOSS' in str(r['outcome'])]

        return {
            "total_executed": len(rows),
            "pending_trades": len(rows) - len(closed),
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": f"{(len(wins) / len(closed) * 100):.1f}%" if closed else "0.0%"
        }
    except Exception as e:
        return {"error": str(e)}
