import os
import json
import asyncio
import psycopg2
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

# Load API credentials from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

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


# --- NEON POSTGRESQL DATABASE SYSTEM ---
def get_db_connection():
    """Establishes connection to Neon PostgreSQL instance."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing.")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Initializes PostgreSQL table structure on Neon if not exists."""
    if not DATABASE_URL:
        print("[DATABASE WARNING] DATABASE_URL missing. Skipping PostgreSQL init.")
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
        print(f"[NEON DATABASE ERROR] Initialization failed: {e}")


def log_trade_signal(status: str, action: str, trigger_type: str, price: float, sl: float, tp1: float, tp2: float, confidence: float, adx_15m: float, stoch_15m: float, divergence: str, reasoning: str):
    """Logs executed trades and AI veto decisions to Neon PostgreSQL."""
    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        initial_outcome = "PENDING" if status == "EXECUTED" else "N/A"
        cursor.execute("""
            INSERT INTO signals (timestamp, status, action, trigger_type, price, sl, tp1, tp2, confidence, adx_15m, stoch_rsi_15m, divergence_type, reasoning, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.now(timezone.utc).isoformat(),
            status,
            action,
            trigger_type,
            price,
            sl,
            tp1,
            tp2,
            confidence,
            adx_15m,
            stoch_15m,
            divergence,
            reasoning,
            initial_outcome
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[NEON LOG] Signal recorded successfully with status: {status}")
    except Exception as e:
        print(f"Failed to log trade to Neon: {e}")


def update_trade_outcomes():
    """Evaluates pending trades against recent candle Highs/Lows to mark TP1, TP2, or SL hits."""
    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING'")
        pending_signals = cursor.fetchall()

        if not pending_signals:
            cursor.close()
            conn.close()
            return

        df_5m = fetch_twelve_data("5min", outputsize=100)
        if df_5m.empty:
            cursor.close()
            conn.close()
            return

        now_utc = datetime.now(timezone.utc)

        for sig in pending_signals:
            sig_id = sig["id"]
            sig_time = datetime.fromisoformat(sig["timestamp"])
            action = sig["action"]
            sl = sig["sl"]
            tp1 = sig["tp1"]
            tp2 = sig["tp2"]

            candles_after = df_5m[df_5m["datetime"] >= sig_time]
            if candles_after.empty:
                continue

            outcome = "PENDING"
            exit_price = None

            for _, candle in candles_after.iterrows():
                high = candle["High"]
                low = candle["Low"]

                if action == "BUY":
                    if low <= sl:
                        outcome = "HIT_SL"
                        exit_price = sl
                        break
                    elif high >= tp2:
                        outcome = "HIT_TP2"
                        exit_price = tp2
                        break
                    elif high >= tp1 and outcome != "HIT_TP1":
                        outcome = "HIT_TP1"
                        exit_price = tp1

                elif action == "SELL":
                    if high >= sl:
                        outcome = "HIT_SL"
                        exit_price = sl
                        break
                    elif low <= tp2:
                        outcome = "HIT_TP2"
                        exit_price = tp2
                        break
                    elif low <= tp1 and outcome != "HIT_TP1":
                        outcome = "HIT_TP1"
                        exit_price = tp1

            if outcome == "PENDING" and (now_utc - sig_time) > timedelta(hours=24):
                outcome = "EXPIRED"
                exit_price = candles_after.iloc[-1]["Close"]

            if outcome != "PENDING":
                cursor.execute("""
                    UPDATE signals 
                    SET outcome = %s, exit_price = %s, outcome_timestamp = %s 
                    WHERE id = %s
                """, (outcome, exit_price, datetime.now(timezone.utc).isoformat(), sig_id))
                conn.commit()
                print(f"[OUTCOME TRACKER] Signal #{sig_id} ({action}) updated to: {outcome} at ${exit_price}")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Outcome evaluation error: {e}")


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


def format_stats_message() -> str:
    """Formats system statistics for Telegram direct response."""
    if not DATABASE_URL:
        return "Database not configured."

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'EXECUTED'")
    total_executed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'VETOED'")
    total_vetoed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome IN ('HIT_TP1', 'HIT_TP2')")
    wins = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_SL'")
    losses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_TP1'")
    tp1_hits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_TP2'")
    tp2_hits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'PENDING'")
    pending = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()

    closed_trades = wins + losses
    win_rate = round((wins / closed_trades * 100), 1) if closed_trades > 0 else 0.0

    return (
        f"📊 SYSTEM PERFORMANCE STATS (NEON CLOUD)\n\n"
        f"• Executed Signals: {total_executed}\n"
        f"• AI Vetoes: {total_vetoed}\n"
        f"• Pending Trades: {pending}\n"
        f"• Closed Trades: {closed_trades}\n"
        f"• Total Wins: {wins} (TP1: {tp1_hits} | TP2: {tp2_hits})\n"
        f"• Total Losses: {losses}\n"
        f"• Win Rate: {win_rate}%\n"
    )


def format_recent_logs_message(limit: int = 5) -> str:
    """Formats the latest logged signals for Telegram direct response."""
    if not DATABASE_URL:
        return "Database not configured."

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return "No trade logs recorded yet."

    msg = f"📜 RECENT {len(rows)} SIGNAL LOGS:\n\n"
    for r in rows:
        status_icon = "✅" if r["status"] == "EXECUTED" else "🛑"
        outcome_str = f" | Outcome: {r['outcome']}" if r["status"] == "EXECUTED" else ""
        msg += f"{status_icon} #{r['id']} {r['action']} @ ${r['price']:.2f}\n"
        msg += f"Status: {r['status']}{outcome_str}\n"
        msg += f"Type: {r['trigger_type']}\n"
        msg += f"Reasoning: {r['reasoning']}\n"
        msg += "-" * 25 + "\n"
    return msg


async def telegram_polling_loop():
    """Asynchronously polls Telegram for commands like /stats, /logs, or /help."""
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram token missing. Polling listener disabled.")
        return

    offset = 0
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"

    while True:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(url, params={"offset": offset, "timeout": 10}, timeout=15.0)
                if res.status_code == 200:
                    data = res.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        message = update.get("message", {})
                        text = message.get("text", "").strip()

                        cmd = text.split("@")[0].lower()

                        if cmd == "/stats":
                            stats_msg = format_stats_message()
                            send_telegram_message(stats_msg)
                        elif cmd == "/logs":
                            logs_msg = format_recent_logs_message(limit=5)
                            send_telegram_message(logs_msg)
                        elif cmd == "/help":
                            help_msg = (
                                "🤖 TRADING BOT COMMANDS:\n\n"
                                "/stats - View live win rate, TP/SL hits, and veto counts\n"
                                "/logs - View details of the last 5 signals\n"
                                "/help - Display command menu"
                            )
                            send_telegram_message(help_msg)

        except Exception as e:
            print(f"Telegram Polling Exception: {e}")

        await asyncio.sleep(3)


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

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(timezone.utc)
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates ADX to measure overall trend strength."""
    if df.empty or len(df) < (period * 2):
        return 20.0

    temp_df = df.copy()
    temp_df['prev_close'] = temp_df['Close'].shift(1)
    
    temp_df['tr'] = temp_df[['High', 'Low', 'prev_close']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['prev_close']), abs(x['Low'] - x['prev_close'])), axis=1
    )
    
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


def detect_divergence(df: pd.DataFrame):
    """Detects Bullish or Bearish Divergence on Stochastic RSI."""
    if df.empty or len(df) < 25:
        return False, False, ""

    recent_df = df.iloc[-20:].copy().reset_index(drop=True)
    lows = recent_df['Low'].values
    stoch_k = recent_df['%K'].values
    
    latest_k = stoch_k[-1]
    prev_k = stoch_k[-2]
    cross_up = bool(prev_k < recent_df['%D'].iloc[-2] and latest_k > recent_df['%D'].iloc[-1])
    cross_dn = bool(prev_k > recent_df['%D'].iloc[-2] and latest_k < recent_df['%D'].iloc[-1])

    p_low_curr = lows[-1]
    p_low_prev = np.min(lows[:-5])
    p_low_prev_idx = np.argmin(lows[:-5])
    k_low_curr = latest_k
    k_low_prev = stoch_k[p_low_prev_idx]

    highs = recent_df['High'].values
    p_high_curr = highs[-1]
    p_high_prev = np.max(highs[:-5])
    p_high_prev_idx = np.argmax(highs[:-5])
    k_high_curr = latest_k
    k_high_prev = stoch_k[p_high_prev_idx]

    is_bull_div = bool(p_low_curr < p_low_prev and k_low_curr > k_low_prev and k_low_curr <= 30.0 and cross_up)
    is_bear_div = bool(p_high_curr > p_high_prev and k_high_curr < k_high_prev and k_high_curr >= 70.0 and cross_dn)

    div_desc = ""
    if is_bull_div:
        div_desc = "Bullish Divergence (Lower Price Low + Higher Stoch Low)"
    elif is_bear_div:
        div_desc = "Bearish Divergence (Higher Price High + Lower Stoch High)"

    return is_bull_div, is_bear_div, div_desc


def calculate_metrics(df: pd.DataFrame, rsi_period=14, stoch_period=14, k_period=3, d_period=3, ema_period=50, atr_period=14):
    """Calculates core indicators: EMA 50, VWAP, Stoch RSI, ATR, ADX, and Divergence."""
    if df.empty or len(df) < (rsi_period + stoch_period + ema_period):
        return None

    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Close']).rolling(window=20).sum() / df['Close'].rolling(window=20).sum()

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

    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['PrevClose']), abs(x['Low'] - x['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    adx_val = calculate_adx(df, period=14)
    is_bull_div, is_bear_div, div_desc = detect_divergence(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

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

    swing_high = float(df['High'].iloc[-11:-1].max())
    swing_low = float(df['Low'].iloc[-11:-1].min())
    atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 2.50

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "vwap": float(latest['VWAP']),
        "is_confirmed_bullish": is_confirmed_bullish,
        "is_confirmed_bearish": is_confirmed_bearish,
        "is_bull_div": is_bull_div,
        "is_bear_div": is_bear_div,
        "div_desc": div_desc,
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "atr": round(atr_val, 2),
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

    update_trade_outcomes()

    print("Checking Twelve Data Multi-Timeframe Spot Gold Data...")
    mtf = get_mtf_data()

    if not mtf or not mtf["5m"] or not mtf["15m"] or not mtf["1h"]:
        print("Waiting for valid data stream...")
        return

    price = mtf["5m"]["close"]
    adx_15m = mtf["15m"]["adx"]

    # --- PILLAR 1: ADX CHOP GUARD (15M ADX > 18.0) ---
    is_bull_div = mtf["5m"]["is_bull_div"] or mtf["15m"]["is_bull_div"]
    is_bear_div = mtf["5m"]["is_bear_div"] or mtf["15m"]["is_bear_div"]

    if adx_15m < 18.0 and not (is_bull_div or is_bear_div):
        print(f"Skipping: Low ADX ({adx_15m:.1f}) indicates horizontal chop.")
        return

    # --- PILLAR 2: STREAMLINED ENTRY EVALUATION ---
    is_5m_bull = mtf["5m"]["is_confirmed_bullish"]
    is_5m_bear = mtf["5m"]["is_confirmed_bearish"]

    valid_buy_structure = is_5m_bull or is_bull_div
    valid_sell_structure = is_5m_bear or is_bear_div

    if not valid_buy_structure and not valid_sell_structure:
        print("Skipping: Price structure does not meet trend confirmation or divergence setup.")
        return

    candidate_action = "BUY" if valid_buy_structure else "SELL"

    if adx_15m >= 35.0:
        tp1_mult, tp2_mult = 2.0, 4.0
        adx_desc = f"15M ADX ({adx_15m:.1f}) Explosive Momentum"
    else:
        tp1_mult, tp2_mult = 1.5, 2.5
        adx_desc = f"15M ADX ({adx_15m:.1f}) Active Trend"

    trigger_type = "Divergence Reversal" if (is_bull_div or is_bear_div) else "Trend Setup"

    active_atr = mtf["5m"]["atr"]
    max_allowed_risk = round(2.0 * active_atr, 2)

    if candidate_action == "BUY":
        raw_risk = price - mtf["5m"]["swing_low"]
        actual_risk = min(max(1.2 * active_atr, raw_risk), max_allowed_risk)
        buy_sl = round(price - actual_risk, 2)
        buy_tp1 = round(price + (tp1_mult * actual_risk), 2)
        buy_tp2 = round(price + (tp2_mult * actual_risk), 2)
        sl_val, tp1_val, tp2_val = buy_sl, buy_tp1, buy_tp2
    else:
        raw_risk = mtf["5m"]["swing_high"] - price
        actual_risk = min(max(1.2 * active_atr, raw_risk), max_allowed_risk)
        sell_sl = round(price + actual_risk, 2)
        sell_tp1 = round(price - (tp1_mult * actual_risk), 2)
        sell_tp2 = round(price - (tp2_mult * actual_risk), 2)
        sl_val, tp1_val, tp2_val = sell_sl, sell_tp1, sell_tp2

    # --- PILLAR 3: HYBRID AI VETO MODEL PROMPT ---
    div_note = mtf["5m"]["div_desc"] or mtf["15m"]["div_desc"] or "None"
    prompt = f"""
You are a Senior Quantitative Trading Analyst with VETO POWER.

Python Risk Manager has passed a candidate setup:
PROPOSED CANDIDATE ACTION: {candidate_action}
TRIGGER TYPE: {trigger_type}
DETECTED DIVERGENCE: {div_note}
Current Price: ${price:.2f}

PRE-CALCULATED LEVELS ({adx_desc}):
Entry: ${price:.2f} | SL: ${sl_val:.2f} | TP1: ${tp1_val:.2f} | TP2: ${tp2_val:.2f}

MULTI-TIMEFRAME METRICS:
- 1H Range: High ${mtf['1h']['swing_high']:.2f} | Low ${mtf['1h']['swing_low']:.2f}
- 15M ADX Trend Strength: {adx_15m:.1f}
- 15M Stoch RSI %K: {mtf['15m']['stoch_k']:.1f}
- 5M Stoch RSI %K: {mtf['5m']['stoch_k']:.1f}
- 5M EMA 50: ${mtf['5m']['ema_50']:.2f} | 5M VWAP: ${mtf['5m']['vwap']:.2f}

ANALYST VETO DIRECTIVE:
Review the proposed {candidate_action} setup against higher timeframe context.
- Check if price is buying directly into 1H resistance or shorting into 1H support.
- If you detect chop, trap risk, or exhausted momentum, EXERCISE YOUR VETO POWER and return "action": "HOLD".
- Otherwise, return "action": "{candidate_action}".

OUTPUT REQUIREMENTS:
Output JSON with schema keys:
- "action": "{candidate_action}" or "HOLD"
- "confidence": float between 0.0 and 1.0
- "reasoning": "Write 2 clean sentences explaining your decision."
"""

    output = None

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
            print(f"Gemini API fallback error: {e}")

    if not output:
        print("All AI model attempts failed this cycle. Continuing...")
        return

    if output.action == "HOLD":
        print(f"[AI VETO EXERCISED] Analyst rejected setup '{candidate_action}'. Reason: {output.reasoning}")
        log_trade_signal(
            status="VETOED",
            action=candidate_action,
            trigger_type=trigger_type,
            price=price,
            sl=sl_val,
            tp1=tp1_val,
            tp2=tp2_val,
            confidence=output.confidence,
            adx_15m=adx_15m,
            stoch_15m=mtf["15m"]["stoch_k"],
            divergence=div_note,
            reasoning=output.reasoning
        )
        return

    if output.action in ["BUY", "SELL"]:
        if output.action == LAST_SIGNAL_ACTION and abs(price - LAST_SIGNAL_PRICE) < 6.00:
            print(f"Skipping duplicate {output.action} alert. Price (${price:.2f}) too close to last entry.")
            return

        LAST_SIGNAL_ACTION = output.action
        LAST_SIGNAL_PRICE = price

        log_trade_signal(
            status="EXECUTED",
            action=output.action,
            trigger_type=trigger_type,
            price=price,
            sl=sl_val,
            tp1=tp1_val,
            tp2=tp2_val,
            confidence=output.confidence,
            adx_15m=adx_15m,
            stoch_15m=mtf["15m"]["stoch_k"],
            divergence=div_note,
            reasoning=output.reasoning
        )

        telegram_text = (
            f"STOCH RSI TRADE SIGNAL\n\n"
            f"Asset: XAUUSD (Gold Spot)\n"
            f"Action: {output.action}\n"
            f"Type: {trigger_type}\n"
            f"Entry Price: ${price:.2f}\n\n"
            f"Stop Loss (SL): ${sl_val:.2f}\n"
            f"Take Profit 1 (TP1): ${tp1_val:.2f} (1:{tp1_mult:.1f} RRR)\n"
            f"Take Profit 2 (TP2): ${tp2_val:.2f} (1:{tp2_mult:.1f} RRR)\n\n"
            f"INDICATOR METRICS:\n"
            f"- Setup Type: {trigger_type}\n"
            f"- Divergence Context: {div_note}\n"
            f"- 15M ADX Strength: {adx_15m:.1f}\n"
            f"- 15M Stoch RSI: {mtf['15m']['stoch_k']:.1f}\n\n"
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
    init_db()
    scan_task = asyncio.create_task(background_scanning_loop())
    poll_task = asyncio.create_task(telegram_polling_loop())
    yield
    scan_task.cancel()
    poll_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "ok", "message": "Trading bot background scanner, outcome tracker, and Telegram listener are active on Neon PostgreSQL!"}


@app.get("/logs")
def get_trade_logs(limit: int = 50):
    """Retrieves paper trading signal history from Neon."""
    if not DATABASE_URL:
        return {"error": "DATABASE_URL environment variable missing"}

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"count": len(rows), "logs": rows}


@app.get("/stats")
def get_trade_stats():
    """Retrieves live win/loss performance stats from Neon."""
    if not DATABASE_URL:
        return {"error": "DATABASE_URL environment variable missing"}

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'EXECUTED'")
    total_executed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'VETOED'")
    total_vetoed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome IN ('HIT_TP1', 'HIT_TP2')")
    wins = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_SL'")
    losses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_TP1'")
    tp1_hits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'HIT_TP2'")
    tp2_hits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'PENDING'")
    pending = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()

    closed_trades = wins + losses
    win_rate = round((wins / closed_trades * 100), 1) if closed_trades > 0 else 0.0
    
    return {
        "executed_signals": total_executed,
        "ai_vetoes": total_vetoed,
        "pending_signals": pending,
        "closed_trades": closed_trades,
        "wins": wins,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "losses": losses,
        "win_rate_percentage": f"{win_rate}%"
    }
