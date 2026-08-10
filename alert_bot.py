import os
import json
import asyncio
import psycopg2
import gc
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

# --- ENVIRONMENT VARIABLES & SANITIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
RAW_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()

# Clean hidden newlines and spaces aggressively
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
    if not DATABASE_URL:
        return
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

# --- TWO-STAGE TP TRACKING FUNCTION (PRIORITY SL CHECK FIRST) ---
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
            entry_price = float(trade['price'])
            sl = float(trade['sl']) if trade['sl'] is not None else None
            tp1 = float(trade['tp1']) if trade['tp1'] is not None else None
            tp2 = float(trade['tp2']) if trade['tp2'] is not None else None
            current_outcome = trade['outcome']

            new_outcome = None
            exit_price = None

            if action == "BUY":
                # Stage 2: Trade already hit TP1 and is tracking to TP2 / BE
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 is not None and c_high >= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_low <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1  # Retain TP1 price so TP1 pips are preserved
                # Stage 1: PENDING trade — Priority 1 is checking Stop Loss BEFORE TP targets
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
                # Stage 2: Trade already hit TP1 and is tracking to TP2 / BE
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 is not None and c_low <= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_high >= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = tp1  # Retain TP1 price so TP1 pips are preserved
                # Stage 1: PENDING trade — Priority 1 is checking Stop Loss BEFORE TP targets
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
                    SET outcome = %s, exit_price = %s, outcome_timestamp = %s
                    WHERE id = %s
                """, (new_outcome, float(exit_price), wib_now, trade_id))
                conn.commit()
                print(f"[TRADE UPDATE] Signal ID {trade_id} -> {new_outcome} at ${exit_price:.2f}")

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
        print("[TELEGRAM ERROR] Missing token or chat_id")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    
    try:
        res = await client.post(url, json=payload)
        if res.status_code != 200:
            payload_plain = {"chat_id": chat_id, "text": text}
            res_plain = await client.post(url, json=payload_plain)
            if res_plain.status_code == 200:
                print(f"[TELEGRAM SENT] Delivered plain text to Chat ID {chat_id}")
            else:
                print(f"[TELEGRAM ERROR] Failed sending message: {res_plain.text}")
        else:
            print(f"[TELEGRAM SENT] Delivered Markdown to Chat ID {chat_id}")
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")

# --- AI ANALYST EVALUATION (REVISED FOR 5M & 15M ONLY) ---
async def analyze_signal_with_ai(proposed_action: str, current_price: float, df_5m: pd.DataFrame, df_15m: pd.DataFrame, divergence: str):
    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD).
A technical trigger suggests a {proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 5-Minute: Close=${float(df_5m['close'].iloc[-1]):.2f}, EMA 50=${float(df_5m['ema_50'].iloc[-1]):.2f}, EMA 200=${float(df_5m['ema_200'].iloc[-1]):.2f}, VWAP=${float(df_5m['vwap'].iloc[-1]):.2f}, Stoch RSI %K=${float(df_5m['stoch_k'].iloc[-1]):.1f}.
2. 15-Minute: Close=${float(df_15m['close'].iloc[-1]):.2f}, EMA 50=${float(df_15m['ema_50'].iloc[-1]):.2f}, EMA 200=${float(df_15m['ema_200'].iloc[-1]):.2f}, ADX Trend Strength=${float(df_15m['adx'].iloc[-1]):.1f}, Stoch RSI %K=${float(df_15m['stoch_k'].iloc[-1]):.1f}.
3. Divergence State: {divergence}.

CRITICAL VETO RULES:
- VETO if 5M Stoch RSI is exhausted (>85 for BUY, <15 for SELL) without bullish/bearish divergence.
- VETO if proposed BUY is below 5M EMA 200 or proposed SELL is above 5M EMA 200 (counter-trend trap).
- VETO if 15M ADX is below 15.0 indicating extreme horizontal chop.

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

# --- BACKGROUND SCANNING LOOP (5M & 15M ONLY, SAFE 300s CYCLE) ---
async def background_scanning_loop():
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        while True:
            try:
                print("Checking Twelve Data Multi-Timeframe Spot Gold Data (5m & 15m)...")
                df_5m = await fetch_timeframe_data(client, "5min")
                df_15m = await fetch_timeframe_data(client, "15min")

                if df_5m is None or df_15m is None:
                    print("Failed to fetch complete timeframe candles. Retrying next cycle.")
                    await asyncio.sleep(300)
                    continue

                df_5m = calculate_metrics(df_5m)
                df_15m = calculate_metrics(df_15m)

                # Look across the last 3 candles to ensure wicks are covered
                curr_high = float(df_5m["high"].tail(3).max())
                curr_low = float(df_5m["low"].tail(3).min())
                update_open_trades(curr_high, curr_low)

                curr_price = float(df_5m["close"].iloc[-1])
                adx_15m = float(df_15m["adx"].iloc[-1])
                stoch_15m = float(df_15m["stoch_k"].iloc[-1])

                # Signal Logic Setup
                c_5m = float(df_5m["close"].iloc[-1])
                ema_50_5m = float(df_5m["ema_50"].iloc[-1])
                ema_200_5m = float(df_5m["ema_200"].iloc[-1])
                
                # Multi-Timeframe Trend Lock (5m & 15m)
                c_15m = float(df_15m["close"].iloc[-1])
                ema_200_15m = float(df_15m["ema_200"].iloc[-1])

                is_uptrend = (c_5m > ema_200_5m) and (ema_50_5m > ema_200_5m) and (c_15m > ema_200_15m)
                is_downtrend = (c_5m < ema_200_5m) and (ema_50_5m < ema_200_5m) and (c_15m < ema_200_15m)

                divergence = check_divergence(df_5m)

                stoch_k_curr = float(df_5m["stoch_k"].iloc[-1])
                stoch_k_prev = float(df_5m["stoch_k"].iloc[-2])
                stoch_d_curr = float(df_5m["stoch_d"].iloc[-1])
                stoch_d_prev = float(df_5m["stoch_d"].iloc[-2])

                # Widened Stochastic RSI crossover boundary (< 40.0 / > 60.0) for increased frequency
                stoch_buy_cross = (stoch_k_prev <= stoch_d_prev) and (stoch_k_curr > stoch_d_curr) and (stoch_k_curr < 40.0)
                stoch_sell_cross = (stoch_k_prev >= stoch_d_prev) and (stoch_k_curr < stoch_d_curr) and (stoch_k_curr > 60.0)

                proposed_action = "HOLD"
                trigger_type = "None"

                # --- TREND-FILTERED SIGNAL EVALUATION ---
                if divergence == "Bullish Divergence (Lower Price Low + Higher Stoch Low)":
                    proposed_action = "BUY"
                    trigger_type = "Divergence Reversal"

                elif divergence == "Bearish Divergence (Higher Price High + Lower Stoch High)":
                    proposed_action = "SELL"
                    trigger_type = "Divergence Reversal"

                elif is_uptrend and stoch_buy_cross:
                    proposed_action = "BUY"
                    trigger_type = "Trend Setup"
                elif is_downtrend and stoch_sell_cross:
                    proposed_action = "SELL"
                    trigger_type = "Trend Setup"

                # --- STATE-AWARE COOLDOWN & DISTANCE GUARD ---
                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT price, outcome FROM signals 
                            WHERE status = 'EXECUTED' AND action = %s 
                            ORDER BY id DESC LIMIT 1
                        """, (proposed_action,))
                        last_trade = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if last_trade:
                            last_entry_price = float(last_trade['price'])
                            last_outcome = str(last_trade['outcome'])
                            
                            required_distance = 6.0 if last_outcome == "PENDING" else 3.0

                            if abs(curr_price - last_entry_price) < required_distance:
                                print(f"[STATE-AWARE COOLDOWN] Skipping {proposed_action}: Price within ${required_distance:.2f} of previous trade at ${last_entry_price:.2f} (Status: {last_outcome}).")
                                proposed_action = "HOLD"
                    except Exception as cd_err:
                        print(f"[STATE-AWARE COOLDOWN ERROR] {cd_err}")

                # --- VISIBLE SCAN STATUS LOGS ---
                if proposed_action == "HOLD":
                    print(f"[MARKET SCAN] Price: ${curr_price:.2f} | 5M EMA 200: ${ema_200_5m:.2f} | 15M ADX: {adx_15m:.1f} | Status: HOLD (No entry setup)")
                else:
                    print(f"[MARKET SCAN] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI Analysis...")
                    ai_decision = await analyze_signal_with_ai(proposed_action, curr_price, df_5m, df_15m, divergence)

                    atr_5m = float(df_5m["atr"].iloc[-1])
                    sl_dist = max(4.50, min(8.00, atr_5m * 2.0))
                    tp1_mult = 1.5
                    tp2_mult = 3.0

                    if proposed_action == "BUY":
                        sl_price = curr_price - sl_dist
                        tp1_price = curr_price + (sl_dist * tp1_mult)
                        tp2_price = curr_price + (sl_dist * tp2_mult)
                    else:
                        sl_price = curr_price + sl_dist
                        tp1_price = curr_price - (sl_dist * tp1_mult)
                        tp2_price = curr_price - (sl_dist * tp2_mult)

                    if ai_decision.action == proposed_action:
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

                del df_5m, df_15m
                gc.collect()

            except Exception as e:
                print(f"[SCAN LOOP ERROR] {e}")

            # 300 seconds (5 minutes) sleep = 576 requests/day (well under Twelve Data's 800 free cap)
            await asyncio.sleep(300)

# --- FASTAPI LIFESPAN & AUTOMATED WEBHOOK SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Automatically register Webhook with Telegram on server startup
    if TELEGRAM_BOT_TOKEN and APP_URL:
        try:
            webhook_endpoint = f"{APP_URL.rstrip('/')}/telegram-webhook"
            set_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={webhook_endpoint}"
            
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                res = await client.get(set_url)
                print(f"[AUTO WEBHOOK SETUP] Response: {res.text}")
        except Exception as e:
            print(f"[AUTO WEBHOOK SETUP ERROR] Failed: {e}")

    scan_task = asyncio.create_task(background_scanning_loop())
    yield
    scan_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot scanner and webhook server active."}

# --- WEBHOOK ENDPOINT FOR TELEGRAM COMMANDS ($0.01 LOT CALIBRATED) ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        message = data.get("message", {})
        raw_text = message.get("text", "").strip().lower()
        sender_chat_id = str(message.get("chat", {}).get("id", ""))

        if not sender_chat_id or not raw_text:
            return {"status": "ignored"}

        print(f"[WEBHOOK RECEIVED] Chat ID: {sender_chat_id} | Command: '{raw_text}'")

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            if raw_text in ["/help", "/start"]:
                help_msg = (
                    "🤖 *TRADING BOT COMMANDS:*\n\n"
                    "• `/stats` - Comprehensive Win-Rate & Risk Analytics Dashboard\n"
                    "• `/pips` - Detailed Gross/Net Pips & USD Profit Breakdown (0.01 Lot)\n"
                    "• `/logs` - Detailed View of Last 10 Trades & Outcomes\n"
                    "• `/help` - Display Command Menu"
                )
                await send_telegram_alert(client, help_msg, target_chat_id=sender_chat_id)

            elif raw_text == "/stats":
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) as total FROM signals WHERE status = 'EXECUTED'")
                    total_executed = cursor.fetchone()['total'] or 0

                    cursor.execute("SELECT COUNT(*) as vetoes FROM signals WHERE status = 'VETOED'")
                    total_vetoes = cursor.fetchone()['vetoes'] or 0

                    cursor.execute("SELECT COUNT(*) as pending FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING'")
                    total_pending = cursor.fetchone()['pending'] or 0

                    cursor.execute("SELECT COUNT(*) as tp1_wins FROM signals WHERE outcome LIKE 'WIN (TP1%' OR outcome LIKE 'CLOSED%'")
                    tp1_wins = cursor.fetchone()['tp1_wins'] or 0

                    cursor.execute("SELECT COUNT(*) as tp2_wins FROM signals WHERE outcome LIKE 'WIN (TP2%'")
                    tp2_wins = cursor.fetchone()['tp2_wins'] or 0

                    cursor.execute("SELECT COUNT(*) as losses FROM signals WHERE outcome LIKE 'LOSS%'")
                    losses = cursor.fetchone()['losses'] or 0

                    cursor.execute("SELECT action, price, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                    closed_trades = cursor.fetchall()

                    total_pips = 0.0
                    win_pips = 0.0
                    loss_pips = 0.0
                    total_wins_count = tp1_wins + tp2_wins

                    for t in closed_trades:
                        entry = float(t['price'])
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
                    est_dollar = total_pips * 0.10  # $0.10 per pip for 0.01 Lot ($1.00 move = $1.00 USD)
                    avg_win = (win_pips / total_wins_count) if total_wins_count > 0 else 0.0
                    avg_loss = (loss_pips / losses) if losses > 0 else 0.0
                    profit_factor = (win_pips / loss_pips) if loss_pips > 0 else (win_pips if win_pips > 0 else 0.0)

                    cursor.close()
                    conn.close()

                    reply = (
                        f"📊 *PERFORMANCE ANALYTICS DASHBOARD*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💰 *NET PIPS & PROFIT:*\n"
                        f"• Net Pips: *{total_pips:+.1f} pips*\n"
                        f"• Est. Profit (0.01 Lot): *${est_dollar:+.2f} USD*\n\n"
                        f"📈 *WIN / LOSS BREAKDOWN:*\n"
                        f"• Total Executed: {total_executed}\n"
                        f"• Total Wins: {total_wins_count} (*{win_rate:.1f}%*)\n"
                        f"  └─ Hit TP1 (BE Runner): {tp1_wins}\n"
                        f"  └─ Hit TP2 (Full Target): {tp2_wins}\n"
                        f"• Total Losses (SL Hit): {losses}\n"
                        f"• Active Pending: {total_pending}\n\n"
                        f"⚡ *SYSTEM & AI EFFICIENCY:*\n"
                        f"• Total Signals Generated: {total_executed + total_vetoes}\n"
                        f"• AI Vetoed Signals: {total_vetoes}\n\n"
                        f"🎯 *RISK & METRICS:*\n"
                        f"• Avg Win: +{avg_win:.1f} pips | Avg Loss: -{avg_loss:.1f} pips\n"
                        f"• Profit Factor: *{profit_factor:.2f}*\n"
                        f"• Win Rate: *{win_rate:.1f}%*"
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as db_err:
                    await send_telegram_alert(client, f"⚠️ Error querying stats: {db_err}", target_chat_id=sender_chat_id)

            elif raw_text == "/pips":
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT action, price, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                    trades = cursor.fetchall()

                    total_pips = 0.0
                    gross_win_pips = 0.0
                    gross_loss_pips = 0.0
                    winning_trades_count = 0
                    losing_trades_count = 0

                    for t in trades:
                        entry = float(t['price'])
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

                    cursor.close()
                    conn.close()

                    reply = (
                        f"💵 *DETAILED PIPS & EARNINGS REPORT*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 *SUMMARY:*\n"
                        f"• Total Net Pips: *{total_pips:+.1f} pips*\n"
                        f"• Net Profit (0.01 Lot): *${est_profit_usd:+.2f} USD*\n\n"
                        f"📈 *PIPS BREAKDOWN:*\n"
                        f"• Gross Gain: +{gross_win_pips:.1f} pips\n"
                        f"• Gross Loss: -{gross_loss_pips:.1f} pips\n\n"
                        f"🎯 *AVERAGE METRICS:*\n"
                        f"• Avg Win Trade: +{avg_win_pips:.1f} pips\n"
                        f"• Avg Loss Trade: -{avg_loss_pips:.1f} pips\n"
                        f"• Pip Efficiency Ratio: {(gross_win_pips / (gross_loss_pips + 1e-5)):.2f}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💡 Note: Calibrated for 0.01 lot XAU/USD ($1.00 move = $1.00 USD)."
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as err:
                    await send_telegram_alert(client, f"⚠️ Error calculating pips: {err}", target_chat_id=sender_chat_id)

            elif raw_text == "/logs":
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, action, price, exit_price, outcome, timestamp 
                        FROM signals 
                        WHERE status = 'EXECUTED' 
                        ORDER BY id DESC 
                        LIMIT 10
                    """)
                    logs = cursor.fetchall()
                    cursor.close()
                    conn.close()

                    if not logs:
                        reply = "📜 *LAST 10 TRADE LOGS:*\n\n_No executed trades found in database._"
                    else:
                        reply = "📜 *LAST 10 DETAILED TRADE LOGS:*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        for l in logs:
                            trade_id = l['id']
                            action = l['action']
                            entry = float(l['price'])
                            exit_p = float(l['exit_price']) if l['exit_price'] else None
                            outcome = l['outcome']
                            date_str = str(l['timestamp']) if l['timestamp'] else "N/A"

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
                                f"{icon} *ID #{trade_id}* | *{action} XAU/USD*\n"
                                f"• Entry: ${entry:.2f} → Exit: ${exit_p if exit_p else 0.0:.2f}\n"
                                f"• Outcome: `{outcome}`\n"
                                f"• Result: {pip_str} | Time: {date_str}\n"
                                f"──────────────────────────\n"
                            )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as log_err:
                    await send_telegram_alert(client, f"⚠️ Error querying logs: {log_err}", target_chat_id=sender_chat_id)

    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")

    return {"status": "ok"}
