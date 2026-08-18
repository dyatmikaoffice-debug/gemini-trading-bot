import os
import json
import asyncio
import psycopg2
import gc
import logging
import uuid
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

# --- LOGGING CONFIGURATION & UVICORN FILTER ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress noisy polling requests from flooding the console
        return "/get-latest-signal" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# --- ENVIRONMENT VARIABLES & SANITIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
RAW_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()

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

# --- GLOBAL SYSTEM STATE & HEARTBEAT ---
SYSTEM_TRADING_ENABLED = True
CURRENT_SCAN_CYCLE_ID = None
LAST_MT5_PING_TIME = None  # Tracks live MT5 connection timestamp

# --- 15M CONFLUENCE CACHE ---
cached_15m = {"df": None, "fetched_at": None}

# --- SCAN SCHEDULE & BUDGET ---
SCAN_INTERVAL_SECONDS = 300  # 5-minute candle boundary
ACTIVE_SESSION_START_HOUR = 0  # 24/7 all-day trading
ACTIVE_SESSION_END_HOUR = 24
TWELVE_DATA_DAILY_LIMIT = 800

# ==========================================================
# EMA 9/15 STRATEGY PARAMETERS
# ==========================================================
EMA_FAST = 9
EMA_SLOW = 15
EMA_TOUCH_TOLERANCE = 0.0
EMA_REQUIRE_CLOSE_IN_TREND = True
EMA_15M_REQUIRE_CLOSE_SIDE = False  # Bypasses API lag timing discrepancies

# --- RISK FILTERS ---
LOSS_COOLDOWN_MINUTES = 10


class SignalOutput(BaseModel):
    action: str = Field(default="HOLD", description="BUY, SELL, or HOLD")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(
        default="Market conditions do not favor entry.",
        description="2 clean sentences explaining the decision"
    )


# --- DATABASE CONNECTION & SCHEMA AUTO-MIGRATION ---
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_events (
                id BIGSERIAL PRIMARY KEY,
                event_time TIMESTAMP DEFAULT NOW(),
                timestamp TEXT,
                cycle_id TEXT,
                event_type TEXT,
                stage TEXT,
                action TEXT,
                trigger_type TEXT,
                price REAL,
                recent_high REAL,
                recent_low REAL,
                adx_5m REAL,
                adx_15m REAL,
                trend_15m TEXT,
                pending_action TEXT,
                pending_trigger TEXT,
                pending_trigger_price REAL,
                pending_zone_lower REAL,
                pending_zone_upper REAL,
                pending_expires TEXT,
                extension_atr REAL,
                body_atr REAL,
                close_location REAL,
                decision TEXT,
                reason TEXT,
                details JSONB
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_events_time ON bot_events(event_time DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_events_cycle ON bot_events(cycle_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_events_type ON bot_events(event_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_events_action ON bot_events(action);")

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
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS outcome_timestamp TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS trend_15m TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_15m_true REAL;",
            "ALTER TABLE bot_events ADD COLUMN IF NOT EXISTS extension_atr REAL;",
            "ALTER TABLE bot_events ADD COLUMN IF NOT EXISTS body_atr REAL;",
            "ALTER TABLE bot_events ADD COLUMN IF NOT EXISTS close_location REAL;"
        ]

        for query in migrations:
            cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[NEON DATABASE] Full schema verified and missing columns auto-migrated.")

    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to initialize database schema: {e}")


def log_bot_event(
    event_type: str,
    stage: str = None,
    action: str = None,
    trigger_type: str = None,
    price: float = None,
    recent_high: float = None,
    recent_low: float = None,
    adx_5m: float = None,
    adx_15m: float = None,
    trend_15m: str = None,
    pending: dict = None,
    extension_atr: float = None,
    body_atr: float = None,
    close_location: float = None,
    decision: str = None,
    reason: str = None,
    details: dict = None,
    cycle_id: str = None
):
    if not DATABASE_URL:
        return None

    conn = None
    cursor = None
    try:
        now_utc = datetime.now(timezone.utc)
        wib_time = (now_utc + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        pending = pending or {}

        def f(value):
            try:
                return float(value) if value is not None else None
            except Exception:
                return None

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bot_events (
                timestamp, cycle_id, event_type, stage, action, trigger_type,
                price, recent_high, recent_low, adx_5m, adx_15m, trend_15m,
                pending_action, pending_trigger, pending_trigger_price,
                pending_zone_lower, pending_zone_upper, pending_expires,
                extension_atr, body_atr, close_location,
                decision, reason, details
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
        """, (
            wib_time,
            cycle_id or CURRENT_SCAN_CYCLE_ID,
            str(event_type),
            str(stage) if stage is not None else None,
            str(action) if action is not None else None,
            str(trigger_type) if trigger_type is not None else None,
            f(price), f(recent_high), f(recent_low), f(adx_5m), f(adx_15m),
            str(trend_15m) if trend_15m is not None else None,
            str(pending.get("action")) if pending.get("action") is not None else None,
            str(pending.get("trigger_type")) if pending.get("trigger_type") is not None else None,
            f(pending.get("trigger_price")),
            f(pending.get("zone_lower")),
            f(pending.get("zone_upper")),
            str(pending.get("expires")) if pending.get("expires") is not None else None,
            f(extension_atr),
            f(body_atr),
            f(close_location),
            str(decision) if decision is not None else None,
            str(reason) if reason is not None else None,
            json.dumps(details or {}, default=str)
        ))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"[BOT EVENT LOG ERROR] {event_type}: {e}")
        return None
    finally:
        try:
            if cursor: cursor.close()
            if conn: conn.close()
        except Exception:
            pass


def log_scan_event(event_type: str, **kwargs):
    return log_bot_event(event_type=event_type, **kwargs)


def log_trade_signal(
    status: str,
    action: str,
    trigger_type: str,
    price: float,
    sl: float,
    tp1: float,
    tp2: float,
    confidence: float,
    adx_15m: float,
    stoch_rsi_15m: float,
    divergence_type: str,
    reasoning: str,
    trend_15m: str = None,
    adx_15m_true: float = None
):
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

        trend_15m_val = str(trend_15m) if trend_15m is not None else None
        adx_15m_true_val = float(adx_15m_true) if adx_15m_true is not None else None

        cursor.execute("""
            INSERT INTO signals (
                timestamp, status, action, trigger_type, price, entry_price, sl, sl_price,
                tp1, tp1_price, tp2, tp2_price, confidence, adx_15m, stoch_rsi_15m,
                divergence_type, reasoning, outcome, outcome_timestamp, trend_15m, adx_15m_true, created_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, NOW()
            )
            RETURNING id;
        """, (
            str(wib_time), str(status), str(action), str(trigger_type), price_val, price_val,
            sl_val, sl_val, tp1_val, tp1_val, tp2_val, tp2_val, conf_val, adx_val, stoch_val,
            str(divergence_type), str(reasoning), "PENDING", "", trend_15m_val, adx_15m_true_val
        ))

        inserted_row = cursor.fetchone()
        new_id = inserted_row["id"] if inserted_row and "id" in inserted_row else None

        conn.commit()
        cursor.close()
        conn.close()

        logging.info(f"[NEON DB LOGGED] Signal ID #{new_id} | Status: {status} | Action: {action} | Price: ${price_val:.2f}")
        return new_id

    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to log signal: {e}")
        return None


# --- TWO-STAGE TP TRACKING FUNCTION ---
def update_open_trades(current_high: float, current_low: float, current_close: float):
    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM signals
            WHERE status = 'EXECUTED'
              AND (outcome = 'PENDING' OR outcome = 'WIN (TP1 HIT)')
        """)

        open_trades = cursor.fetchall()
        if not open_trades:
            cursor.close()
            conn.close()
            return

        wib_now = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        c_high = float(current_high)
        c_low = float(current_low)
        c_close = float(current_close)

        for trade in open_trades:
            trade_id = trade["id"]
            action = trade["action"]

            entry_price = float(trade["entry_price"] if trade.get("entry_price") is not None else trade.get("price", 0.0))
            sl = float(trade["sl_price"] if trade.get("sl_price") is not None else trade.get("sl", 0.0))
            tp1 = float(trade["tp1_price"] if trade.get("tp1_price") is not None else trade.get("tp1", 0.0))
            tp2 = float(trade["tp2_price"] if trade.get("tp2_price") is not None else trade.get("tp2", 0.0))

            current_outcome = trade["outcome"]
            new_outcome = None
            exit_price = None

            if action == "BUY":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 > 0 and c_high >= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2
                    elif c_close <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = entry_price
                elif sl > 0 and c_close <= sl:
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
                    elif c_close >= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = entry_price
                elif sl > 0 and c_close >= sl:
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

                trade_for_calc = {
                    "action": action, "entry_price": entry_price, "sl_price": sl,
                    "tp1_price": tp1, "tp2_price": tp2, "exit_price": float(exit_price), "outcome": new_outcome
                }
                result_pips, result_usd = compute_trade_pips(trade_for_calc)
                result_r = compute_r_multiple(action, entry_price, float(exit_price), sl, tp1, tp2, new_outcome)

                log_bot_event(
                    "TRADE_OUTCOME", stage="TRADE_MANAGEMENT", action=action, price=float(exit_price), decision=new_outcome,
                    reason="Two-stage TP/SL outcome detected",
                    details={
                        "signal_id": trade_id, "entry": entry_price, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "exit": float(exit_price), "outcome": new_outcome, "pips": result_pips,
                        "profit_usd": result_usd, "r_multiple": result_r, "legacy_fallback_used": not bool(sl > 0 and tp1 > 0)
                    }
                )

                logging.info(f"[TRADE UPDATE] Signal ID {trade_id} -> {new_outcome} at ${exit_price:.2f} | Result: {result_pips:+.1f} pips | {result_r:+.2f}R | ${result_usd:+.2f}")

        cursor.close()
        conn.close()

    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to update trade outcomes: {e}")


# --- MARKET DATA FETCHING ---
async def fetch_timeframe_data(client: httpx.AsyncClient, timeframe: str, outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={timeframe}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    try:
        res = await client.get(url)
        if res.status_code != 200 or not res.text:
            return None
        data = res.json()
    except Exception as e:
        logging.error(f"[DATA FETCH ERROR] {timeframe}: {e}")
        return None

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    if df.empty:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


# --- EMA / ATR CALCULATIONS ---
def calculate_ema_metrics(df: pd.DataFrame):
    df = df.tail(150).copy()
    df["ema9"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema15"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    df["atr"] = df["tr"].rolling(window=14).mean()
    return df


def calculate_metrics_5m(df: pd.DataFrame):
    return calculate_ema_metrics(df)


def compute_ema_trend(df: pd.DataFrame, fast: int = EMA_FAST, slow: int = EMA_SLOW):
    if df is None or len(df) < slow + 1:
        return "NEUTRAL", 0.0
    if "ema9" not in df.columns or "ema15" not in df.columns:
        df = calculate_ema_metrics(df)

    last_fast = float(df["ema9"].iloc[-1])
    last_slow = float(df["ema15"].iloc[-1])
    if last_slow == 0:
        return "NEUTRAL", 0.0

    separation_pct = abs(last_fast - last_slow) / abs(last_slow) * 100.0
    if last_fast > last_slow:
        return "BULLISH", separation_pct
    if last_fast < last_slow:
        return "BEARISH", separation_pct
    return "NEUTRAL", separation_pct


# --- STRATEGY ENGINE: CROSSOVERS + PULLBACKS ---
def detect_ema_touch_signal(df_5m: pd.DataFrame, df_15m: pd.DataFrame):
    if df_5m is None or df_15m is None or len(df_5m) < 20 or len(df_15m) < 20:
        return "HOLD", "EMA setup rejected: insufficient 5M/15M data", 0.0, 0.0

    m5 = df_5m.iloc[-1]
    p5 = df_5m.iloc[-2]
    m15 = df_15m.iloc[-1]

    ema9_5 = float(m5["ema9"])
    ema15_5 = float(m5["ema15"])
    ema9_15 = float(m15["ema9"])
    ema15_15 = float(m15["ema15"])

    close5 = float(m5["close"])
    open5 = float(m5["open"])
    high5 = float(m5["high"])
    low5 = float(m5["low"])

    trend_5m = "BULLISH" if ema9_5 > ema15_5 else "BEARISH" if ema9_5 < ema15_5 else "NEUTRAL"
    trend_15m = "BULLISH" if ema9_15 > ema15_15 else "BEARISH" if ema9_15 < ema15_15 else "NEUTRAL"

    p_high = float(p5["high"])
    p_low = float(p5["low"])
    p_ema9 = float(p5["ema9"])
    p_ema15 = float(p5["ema15"])

    bullish_cross = (p_ema9 <= p_ema15) and (ema9_5 > ema15_5)
    bearish_cross = (p_ema9 >= p_ema15) and (ema9_5 < ema15_5)

    touched_ema15 = (low5 - EMA_TOUCH_TOLERANCE <= ema15_5 <= high5 + EMA_TOUCH_TOLERANCE)
    touched_ema9 = (low5 - EMA_TOUCH_TOLERANCE <= ema9_5 <= high5 + EMA_TOUCH_TOLERANCE)
    prev_touched_ema15 = (p_low <= p_ema15 <= p_high)
    prev_touched_ema9 = (p_low <= p_ema9 <= p_high)

    # 1. CROSSOVER TRIGGER (Independent of 15M Trend Lag)
    if bullish_cross and close5 > ema9_5:
        return "BUY", "5M EMA Crossover - Bullish Momentum", ema9_5, ema15_5

    if bearish_cross and close5 < ema9_5:
        return "SELL", "5M EMA Crossover - Bearish Momentum", ema9_5, ema15_5

    # 2. TOUCH / PULLBACK TRIGGER (Requires 15M Confluence)
    if trend_5m == "BULLISH" and trend_15m == "BULLISH":
        close_side_ok = close5 > ema15_5 if EMA_REQUIRE_CLOSE_IN_TREND else True
        m15_side_ok = (float(m15["close"]) > ema15_15) if EMA_15M_REQUIRE_CLOSE_SIDE else True

        if m15_side_ok and close_side_ok:
            if touched_ema15:
                trigger = "5M EMA15 Touch - Bullish Trend"
                if prev_touched_ema15: trigger += " (Continuation Touch)"
                return "BUY", trigger, ema9_5, ema15_5

            if touched_ema9:
                trigger = "5M EMA9 Touch - Bullish Trend"
                if prev_touched_ema9: trigger += " (Continuation Touch)"
                return "BUY", trigger, ema9_5, ema15_5

    if trend_5m == "BEARISH" and trend_15m == "BEARISH":
        close_side_ok = close5 < ema15_5 if EMA_REQUIRE_CLOSE_IN_TREND else True
        m15_side_ok = (float(m15["close"]) < ema15_15) if EMA_15M_REQUIRE_CLOSE_SIDE else True

        if m15_side_ok and close_side_ok:
            if touched_ema15:
                trigger = "5M EMA15 Touch - Bearish Trend"
                if prev_touched_ema15: trigger += " (Continuation Touch)"
                return "SELL", trigger, ema9_5, ema15_5

            if touched_ema9:
                trigger = "5M EMA9 Touch - Bearish Trend"
                if prev_touched_ema9: trigger += " (Continuation Touch)"
                return "SELL", trigger, ema9_5, ema15_5

    reason = f"5M {trend_5m}, 15M {trend_15m}; EMA9/15=${ema9_5:.2f}/${ema15_5:.2f}; no aligned EMA touch or cross"
    return "HOLD", reason, ema9_5, ema15_5


# --- FORWARD-TEST ANALYTICS HELPERS ---
def compute_trade_pips(trade: dict) -> tuple[float, float]:
    action = str(trade.get("action") or "BUY").upper()
    entry = float(trade.get("entry_price") or trade.get("entry_p") or trade.get("price") or 0.0)
    sl = float(trade.get("sl_price") or trade.get("sl") or 0.0)
    tp1 = float(trade.get("tp1_price") or trade.get("tp1") or 0.0)
    tp2 = float(trade.get("tp2_price") or trade.get("tp2") or 0.0)
    exit_p = float(trade.get("exit_price") or entry)
    outcome = str(trade.get("outcome") or trade.get("outcome_val") or "PENDING")

    sl_dist = abs(entry - sl) if sl > 0 else abs(entry - exit_p)
    if sl_dist == 0: sl_dist = 2.5
    tp1_dist = abs(tp1 - entry) if tp1 > 0 else sl_dist * 1.5
    tp2_dist = abs(tp2 - entry) if tp2 > 0 else sl_dist * 3.0

    if outcome == "CLOSED (TP1 HIT / SL BE)": total_pips = tp1_dist * 10.0
    elif outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]: total_pips = (tp1_dist + tp2_dist) * 10.0
    elif outcome == "WIN (TP1 HIT)": total_pips = tp1_dist * 10.0
    elif "LOSS" in outcome: total_pips = -(sl_dist * 10.0 * 2.0)
    else:
        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
        total_pips = diff * 10.0 * 2.0

    profit_usd = total_pips * 0.10
    return total_pips, profit_usd


def compute_r_multiple(action: str, entry: float, exit_price: float, sl: float, tp1: float = 0.0, tp2: float = 0.0, outcome: str = "PENDING") -> float:
    risk_dist = abs(entry - sl)
    if risk_dist <= 0:
        risk_dist = abs(entry - exit_price) if "LOSS" in outcome else 2.5
        if risk_dist == 0: risk_dist = 2.5

    if outcome == "CLOSED (TP1 HIT / SL BE)":
        tp1_dist = abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5
        return tp1_dist / risk_dist
    if outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        tp1_dist = abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5
        tp2_dist = abs(tp2 - entry) if tp2 > 0 else risk_dist * 3.0
        return (tp1_dist + tp2_dist) / risk_dist
    if outcome == "WIN (TP1 HIT)":
        tp1_dist = abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5
        return tp1_dist / risk_dist
    if "LOSS" in outcome:
        return -2.0

    single_r = ((exit_price - entry) / risk_dist if action == "BUY" else (entry - exit_price) / risk_dist)
    return single_r * 2.0


def bucket_adx(adx: float) -> str:
    if adx < 20: return "ADX <20"
    if adx < 30: return "ADX 20-30"
    if adx < 40: return "ADX 30-40"
    if adx < 50: return "ADX 40-50"
    return "ADX 50+"


def bucket_strategy(trigger_type: str) -> str:
    t = trigger_type or ""
    if "EMA15" in t: return "EMA15 Trend Pullback"
    if "EMA9" in t: return "EMA9 Trend Pullback"
    if "Crossover" in t: return "EMA Crossover Momentum"
    return "EMA 9/15 Trend Pullback"


def bucket_session(timestamp_str: str) -> str:
    try:
        hour = int(str(timestamp_str).split(" ")[1].split(":")[0])
    except Exception:
        return "Unknown"
    if 9 <= hour < 14: return "Early (09-14 WIB)"
    if 14 <= hour < 18: return "Mid (14-18 WIB)"
    if 18 <= hour < 22: return "Late (18-22 WIB)"
    return "Outside session"


def bucket_confluence(action: str, trend_15m: str) -> str:
    t = (trend_15m or "").upper()
    if not t or t == "NEUTRAL": return "15m Neutral"
    if ((action == "BUY" and t == "BULLISH") or (action == "SELL" and t == "BEARISH")): return "15m Aligned"
    return "15m Disagreed"


def format_performance_segment(dim_name: str, buckets: dict, min_sample_to_flag: int = 8) -> str:
    lines = [f"*{dim_name}:*"]
    for label, r_values in sorted(buckets.items(), key=lambda item: -len(item[1])):
        n = len(r_values)
        wins = sum(1 for r in r_values if r > 0)
        win_rate = (wins / n * 100) if n else 0.0
        avg_r = (sum(r_values) / n) if n else 0.0
        flag = ""
        if n >= min_sample_to_flag:
            if win_rate < 35: flag = " ⚠️ underperforming"
            elif win_rate > 65: flag = " ✅ strong"
        lines.append(f"• {label}: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}")
    return "\n".join(lines)


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
async def analyze_signal_with_ai(
    proposed_action: str, trigger_type: str, current_price: float, df_5m: pd.DataFrame,
    df_15m: pd.DataFrame, ema9_5m: float, ema15_5m: float, trend_15m: str = "NEUTRAL", adx_15m_true: float = 0.0
):
    m5 = df_5m.iloc[-1]
    m15 = df_15m.iloc[-1]

    prompt = f"""
Act as a conservative risk manager for Spot Gold (XAU/USD).

ONLY strategy in use: 5-minute EMA9/EMA15 trend-following pullback & crossover.
The deterministic strategy has already produced a {proposed_action} trigger.
Your job is ONLY to approve or veto it; never reverse the direction.

5M:
- Trend: {('BULLISH' if ema9_5m > ema15_5m else 'BEARISH')}
- EMA9: ${ema9_5m:.2f}
- EMA15: ${ema15_5m:.2f}
- Open: ${float(m5['open']):.2f}
- High: ${float(m5['high']):.2f}
- Low: ${float(m5['low']):.2f}
- Close: ${float(m5['close']):.2f}

15M CONFLUENCE:
- Trend: {trend_15m}
- EMA9: ${float(m15['ema9']):.2f}
- EMA15: ${float(m15['ema15']):.2f}
- Close: ${float(m15['close']):.2f}
- ADX: {adx_15m_true:.1f}

Trigger: {trigger_type}
Entry: ${current_price:.2f}

Approve only if the technical reaction aligns cleanly with the trigger and there is no obvious immediate structural contradiction.
Return strict JSON:
{{"action":"{proposed_action}" or "HOLD", "confidence":0.0-1.0, "reasoning":"2 concise sentences"}}
"""

    if GROQ_API_KEY:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return SignalOutput(**json.loads(res.choices[0].message.content))
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

    return SignalOutput(action=proposed_action, confidence=0.70, reasoning="Fallback approval: EMA9/EMA15 strategy conditions satisfied.")


# --- BACKGROUND SCANNING LOOP ---
async def background_scanning_loop():
    global SYSTEM_TRADING_ENABLED, CURRENT_SCAN_CYCLE_ID, cached_15m

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        last_5m_candle_time = None

        while True:
            try:
                now_wib = datetime.now(timezone.utc) + timedelta(hours=7)

                if not SYSTEM_TRADING_ENABLED:
                    log_scan_event("SYSTEM_PAUSED", stage="SYSTEM", decision="SKIPPED", reason="Kill switch active")
                    await asyncio.sleep(30)
                    continue

                active_session = ACTIVE_SESSION_START_HOUR <= now_wib.hour < ACTIVE_SESSION_END_HOUR
                if not active_session:
                    log_scan_event("OUTSIDE_SESSION", stage="SESSION", decision="SKIPPED", reason="Outside active session", details={"wib_time": now_wib.strftime('%H:%M')})
                    await asyncio.sleep(60)
                    continue

                # Widened window to prevent MT5 polling delays from skipping scans
                if now_wib.minute % 5 != 0 or now_wib.second > 45:
                    await asyncio.sleep(1)
                    continue

                CURRENT_SCAN_CYCLE_ID = f"{now_wib.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
                log_scan_event("SCAN_START", stage="SCAN", decision="STARTED", reason="5M EMA scan started", details={"wib_time": now_wib.strftime('%Y-%m-%d %H:%M:%S')})

                df_5m_raw = await fetch_timeframe_data(client, "5min", outputsize=100)
                if df_5m_raw is None or len(df_5m_raw) < 20:
                    log_scan_event("DATA_FETCH_FAILED", stage="DATA", decision="RETRY", reason="Failed to fetch sufficient 5M candles")
                    await asyncio.sleep(15)
                    continue

                df_5m = calculate_ema_metrics(df_5m_raw)
                if len(df_5m) >= 2:
                    df_5m = df_5m.iloc[:-1].copy()
                if len(df_5m) < 20:
                    log_scan_event("DATA_INSUFFICIENT", stage="DATA", decision="RETRY", reason="No completed 5M candle available yet")
                    await asyncio.sleep(10)
                    continue
                
                candle_time_5m = df_5m["datetime"].iloc[-1]
                if last_5m_candle_time is not None and candle_time_5m == last_5m_candle_time:
                    await asyncio.sleep(2)
                    continue
                last_5m_candle_time = candle_time_5m

                # Refresh 15M data alongside 5M data
                df_15m_raw = await fetch_timeframe_data(client, "15min", outputsize=100)
                if df_15m_raw is not None and len(df_15m_raw) >= 20:
                    df_15m_new = calculate_ema_metrics(df_15m_raw)
                    if len(df_15m_new) >= 2:
                        df_15m_new = df_15m_new.iloc[:-1].copy()
                    if len(df_15m_new) >= 20:
                        cached_15m["df"] = df_15m_new
                        cached_15m["fetched_at"] = datetime.now(timezone.utc)
                        logging.info(f"[15M REFRESH] New completed 15M EMA snapshot cached at {now_wib.strftime('%H:%M:%S')} WIB")

                df_15m = cached_15m["df"]
                if df_15m is None or len(df_15m) < 20:
                    log_scan_event("DATA_INSUFFICIENT", stage="DATA", decision="HOLD", reason="No valid 15M confluence cache")
                    await asyncio.sleep(2)
                    continue

                trend_15m, trend_15m_sep = compute_ema_trend(df_15m)
                adx_15m_true = 0.0

                update_open_trades(
                    float(df_5m["high"].iloc[-1]), 
                    float(df_5m["low"].iloc[-1]), 
                    float(df_5m["close"].iloc[-1])
                )

                curr_price = float(df_5m["close"].iloc[-1])
                proposed_action, trigger_type, ema9_5m, ema15_5m = detect_ema_touch_signal(df_5m, df_15m)

                log_scan_event(
                    "EMA_EVALUATION", stage="EMA_STRATEGY", action=proposed_action, trigger_type=trigger_type, price=curr_price,
                    adx_15m=adx_15m_true, trend_15m=trend_15m, decision="ENTRY_CANDIDATE" if proposed_action != "HOLD" else "HOLD",
                    reason=trigger_type,
                    details={
                        "5m_candle_time": str(candle_time_5m), "ema9_5m": ema9_5m, "ema15_5m": ema15_5m,
                        "ema9_15m": float(df_15m["ema9"].iloc[-1]), "ema15_15m": float(df_15m["ema15"].iloc[-1]),
                        "trend_15m": trend_15m, "15m_cached": True,
                    }
                )

                if proposed_action == "HOLD":
                    logging.info(f"[EMA SCAN] {now_wib.strftime('%H:%M')} WIB | Price ${curr_price:.2f} | 5M EMA9 ${ema9_5m:.2f} / EMA15 ${ema15_5m:.2f} | 15M {trend_15m} | HOLD ({trigger_type})")
                    log_scan_event("SCAN_END", stage="SCAN", action="HOLD", trigger_type=trigger_type, price=curr_price, adx_15m=adx_15m_true, trend_15m=trend_15m, decision="COMPLETE", reason="No aligned EMA touch or crossover")
                    await asyncio.sleep(2)
                    continue

                # Loss Cooldown Check
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT outcome_timestamp FROM signals
                        WHERE status='EXECUTED' AND outcome='LOSS (SL HIT)'
                        AND outcome_timestamp IS NOT NULL AND outcome_timestamp != ''
                        ORDER BY id DESC LIMIT 1
                    """)
                    last_loss = cursor.fetchone()
                    cursor.close(); conn.close()

                    if last_loss and last_loss.get("outcome_timestamp"):
                        ts_str = str(last_loss["outcome_timestamp"]).replace(" WIB", "")
                        last_loss_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        minutes_since_loss = (now_wib.replace(tzinfo=None) - last_loss_time).total_seconds() / 60.0
                        if 0 <= minutes_since_loss < LOSS_COOLDOWN_MINUTES:
                            log_scan_event("LOSS_COOLDOWN", stage="RISK_FILTER", action=proposed_action, trigger_type=trigger_type, price=curr_price, trend_15m=trend_15m, decision="BLOCKED", reason=f"{minutes_since_loss:.1f} minutes since last SL hit")
                            logging.info(f"[RISK FILTER] Valid {proposed_action} blocked by LOSS COOLDOWN. {minutes_since_loss:.1f} mins since last SL hit.")
                            await asyncio.sleep(2)
                            continue
                except Exception as e:
                    logging.error(f"[LOSS COOLDOWN ERROR] {e}")

                # Distance Cooldown Check
                try:
                    conn = get_db_connection(); cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COALESCE(entry_price, price, 0) AS entry_p, outcome
                        FROM signals WHERE status='EXECUTED' AND action=%s
                        ORDER BY id DESC LIMIT 1
                    """, (str(proposed_action),))
                    last_trade = cursor.fetchone()
                    cursor.close(); conn.close()
                    
                    if last_trade:
                        last_entry_price = float(last_trade["entry_p"] or 0.0)
                        required_distance = 2.00 if str(last_trade.get("outcome") or "PENDING") == "PENDING" else 1.50
                        if last_entry_price > 0 and abs(curr_price - last_entry_price) < required_distance:
                            log_scan_event("DISTANCE_COOLDOWN", stage="RISK_FILTER", action=proposed_action, trigger_type=trigger_type, price=curr_price, trend_15m=trend_15m, decision="BLOCKED", reason=f"Price within ${required_distance:.2f} of previous {proposed_action} trade")
                            logging.info(f"[RISK FILTER] Valid {proposed_action} blocked by DISTANCE COOLDOWN. Price ${curr_price:.2f} is within ${required_distance:.2f} of previous entry.")
                            await asyncio.sleep(2)
                            continue
                except Exception as e:
                    logging.error(f"[DISTANCE COOLDOWN ERROR] {e}")

                ai_decision = await analyze_signal_with_ai(
                    proposed_action, trigger_type, curr_price,
                    df_5m, df_15m, ema9_5m, ema15_5m,
                    trend_15m, adx_15m_true
                )

                log_scan_event(
                    "AI_DECISION", stage="AI", action=proposed_action, trigger_type=trigger_type,
                    price=curr_price, trend_15m=trend_15m, decision=ai_decision.action,
                    reason=ai_decision.reasoning, details={"confidence": float(ai_decision.confidence)}
                )

                raw_atr = df_5m["atr"].iloc[-1]
                atr_5m = float(raw_atr) if not pd.isna(raw_atr) and float(raw_atr) > 0 else 2.5
                risk = max(2.5, atr_5m * 1.2)
                tp1_r_mult = 1.5
                tp2_r_mult = 2.5

                if proposed_action == "BUY":
                    sl_price = curr_price - risk
                    tp1_price = curr_price + risk * tp1_r_mult
                    tp2_price = curr_price + risk * tp2_r_mult
                else:
                    sl_price = curr_price + risk
                    tp1_price = curr_price - risk * tp1_r_mult
                    tp2_price = curr_price - risk * tp2_r_mult

                if ai_decision.action == proposed_action:
                    new_id = log_trade_signal(
                        "EXECUTED", proposed_action, trigger_type, curr_price,
                        sl_price, tp1_price, tp2_price, float(ai_decision.confidence),
                        0.0, 0.0, "None", ai_decision.reasoning, trend_15m, adx_15m_true
                    )

                    log_scan_event(
                        "SIGNAL_EXECUTED", stage="TRADE_SIGNAL", action=proposed_action,
                        trigger_type=trigger_type, price=curr_price, trend_15m=trend_15m,
                        decision="EXECUTED", reason=ai_decision.reasoning,
                        details={"signal_id": new_id, "ema9_5m": ema9_5m, "ema15_5m": ema15_5m,
                                 "ema9_15m": float(df_15m["ema9"].iloc[-1]),
                                 "ema15_15m": float(df_15m["ema15"].iloc[-1]),
                                 "risk": risk, "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price}
                    )

                    msg = (
                        f"📈 *EMA 9/15 TREND SIGNAL #{new_id}*\n\n"
                        f"Asset: *XAUUSD (Gold Spot)*\n"
                        f"Action: *{proposed_action}*\n"
                        f"Type: *{trigger_type}*\n"
                        f"Entry Price: *${curr_price:.2f}*\n\n"
                        f"Stop Loss (SL): *${sl_price:.2f}*\n"
                        f"Take Profit 1 (1.5R): *${tp1_price:.2f}*\n"
                        f"Take Profit 2 (2.5R): *${tp2_price:.2f}*\n\n"
                        f"INDICATOR METRICS:\n"
                        f"- 5M EMA9: *${ema9_5m:.2f}*\n"
                        f"- 5M EMA15: *${ema15_5m:.2f}*\n"
                        f"- 15M Trend: *{trend_15m}*\n"
                        f"- 15M EMA9: *${float(df_15m['ema9'].iloc[-1]):.2f}*\n"
                        f"- 15M EMA15: *${float(df_15m['ema15'].iloc[-1]):.2f}*\n"
                        f"- 5M ATR (risk unit): *${atr_5m:.2f}*\n\n"
                        f"Reasoning: {ai_decision.reasoning}"
                    )
                    await send_telegram_alert(client, msg)
                else:
                    log_trade_signal(
                        "VETOED", proposed_action, trigger_type, curr_price,
                        sl_price, tp1_price, tp2_price, float(ai_decision.confidence),
                        0.0, 0.0, "None", ai_decision.reasoning, trend_15m, adx_15m_true
                    )
                    log_scan_event("AI_VETO", stage="AI", action=proposed_action, trigger_type=trigger_type, price=curr_price, trend_15m=trend_15m, decision="VETOED", reason=ai_decision.reasoning)

                log_scan_event("SCAN_END", stage="SCAN", action=proposed_action, trigger_type=trigger_type, price=curr_price, trend_15m=trend_15m, decision="COMPLETE", reason="EMA strategy cycle completed")
                await asyncio.sleep(2)

            except Exception as e:
                log_scan_event("SCAN_ERROR", stage="SCAN", decision="ERROR", reason=str(e), details={"exception_type": type(e).__name__})
                logging.error(f"[SCAN LOOP ERROR] {e}")
                await asyncio.sleep(10)


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


# =====================================================================
# MT5 COPIER BRIDGE API ENDPOINT
# =====================================================================
@app.get("/get-latest-signal")
async def get_latest_signal():
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME

    # Heartbeat: Record the exact time MT5 polled the server
    LAST_MT5_PING_TIME = datetime.now(timezone.utc) + timedelta(hours=7)

    if not SYSTEM_TRADING_ENABLED:
        return {"signal": None, "trading_enabled": False, "status": "PAUSED"}
    if not DATABASE_URL:
        return {"signal": None, "error": "DATABASE_URL not set", "trading_enabled": SYSTEM_TRADING_ENABLED}

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                   COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, COALESCE(timestamp, created_at::text, '') AS log_time
            FROM signals WHERE status = 'EXECUTED' ORDER BY id DESC LIMIT 1;
        """)

        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            return {
                "id": int(row["id"]), "action": str(row["action"]).upper(), "entry": float(row["entry_p"]),
                "sl": float(row["sl_p"]), "tp1": float(row["tp1_p"]), "tp2": float(row["tp2_p"]),
                "timestamp": str(row["log_time"]), "trading_enabled": True
            }
        return {"signal": None, "trading_enabled": True}
    except Exception as e:
        logging.error(f"[MT5 BRIDGE ERROR /get-latest-signal] {e}")
        return {"error": str(e), "trading_enabled": SYSTEM_TRADING_ENABLED}


# --- WEBHOOK ENDPOINT FOR TELEGRAM COMMANDS ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME
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
                    "• `/status` - 📡 Real-Time MT5 & Server Heartbeat Status\n"
                    "• `/stats` - Comprehensive Win-Rate & Risk Analytics Dashboard\n"
                    "• `/pips` - Detailed Gross/Net Pips & USD Profit Breakdown (0.01 Lot)\n"
                    "• `/logs` - Detailed View of Last 10 Trades & Outcomes\n"
                    "• `/analyze` - Forward-Test Breakdown by Strategy, ADX Regime & Session\n"
                    "• `/pause` - 🛑 *EMERGENCY KILL SWITCH* (Stop Bot & MT5 auto-trade)\n"
                    "• `/resume` - 🟢 Re-enable Auto-Trading Execution\n"
                    "• `/help` - Display Command Menu\n\n"
                    "📐 Active strategy: 5M EMA9/EMA15 trend pullback and crossover.\n"
                    f"⏱️ Loss cooldown: {LOSS_COOLDOWN_MINUTES} min after any SL hit."
                )
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/status":
                if LAST_MT5_PING_TIME:
                    now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
                    seconds_ago = (now_wib.replace(tzinfo=None) - LAST_MT5_PING_TIME.replace(tzinfo=None)).total_seconds()
                    
                    if seconds_ago < 60:
                        status_icon = "🟢"
                        conn_msg = f"Connected and Active\n• Last ping: *{seconds_ago:.0f}s ago*"
                    elif seconds_ago < 180:
                        status_icon = "🟡"
                        conn_msg = f"Slight Lag\n• Last ping: *{seconds_ago:.0f}s ago*"
                    else:
                        status_icon = "🔴"
                        conn_msg = f"DISCONNECTED\n• Last ping was *{seconds_ago:.0f}s ago*! Please check your MT5 terminal."

                    reply = (
                        f"{status_icon} *SYSTEM & BRIDGE STATUS*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• Trading State: *{'ACTIVE' if SYSTEM_TRADING_ENABLED else 'PAUSED (Kill-Switch)'}*\n"
                        f"• MT5 Bridge: *{conn_msg}*\n"
                        f"• Server Time: `{now_wib.strftime('%Y-%m-%d %H:%M:%S WIB')}`"
                    )
                else:
                    reply = (
                        "🔴 *MT5 DISCONNECTED*\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "The server is running, but MT5 has not sent any pings since the last reboot."
                    )
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/pause":
                SYSTEM_TRADING_ENABLED = False
                reply = "🛑 *EMERGENCY KILL SWITCH ACTIVATED*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Market scanner loop has been **PAUSED**.\n• MT5 Signal Copier will **IGNORE** all new signals.\n\n👉 Send `/resume` to reactivate trading."
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/resume":
                SYSTEM_TRADING_ENABLED = True
                reply = "🟢 *AUTO-TRADING SYSTEM RESUMED*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n• Scanner loop is now **ACTIVE**.\n• MT5 bridge is listening for live setups."
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/stats":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("SELECT COUNT(*) AS total FROM signals WHERE status = 'EXECUTED'")
                    total_executed = cur.fetchone()["total"] or 0

                    cur.execute("SELECT COUNT(*) AS vetoes FROM signals WHERE status = 'VETOED'")
                    total_vetoes = cur.fetchone()["vetoes"] or 0

                    cur.execute("SELECT COUNT(*) AS pending FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING'")
                    total_pending = cur.fetchone()["pending"] or 0

                    cur.execute("SELECT COUNT(*) AS tp1_wins FROM signals WHERE outcome LIKE 'WIN (TP1%' OR outcome LIKE 'CLOSED%'")
                    tp1_wins = cur.fetchone()["tp1_wins"] or 0

                    cur.execute("SELECT COUNT(*) AS tp2_wins FROM signals WHERE outcome LIKE 'WIN (TP2%'")
                    tp2_wins = cur.fetchone()["tp2_wins"] or 0

                    cur.execute("SELECT COUNT(*) AS losses FROM signals WHERE outcome LIKE 'LOSS%'")
                    losses = cur.fetchone()["losses"] or 0

                    cur.execute("""
                        SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    closed_trades = cur.fetchall()
                    total_pips = win_pips = loss_pips = 0.0
                    total_wins_count = tp1_wins + tp2_wins

                    for t in closed_trades:
                        trade_pips, _trade_usd = compute_trade_pips({
                            "action": t["action"], "entry_price": t["entry_p"], "sl_price": t["sl_p"],
                            "tp1_price": t["tp1_p"], "tp2_price": t["tp2_p"], "exit_price": t["exit_price"], "outcome": t["outcome_val"]
                        })
                        total_pips += trade_pips
                        if trade_pips > 0: win_pips += trade_pips
                        elif trade_pips < 0: loss_pips += abs(trade_pips)

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
                    cur.execute("""
                        SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price, COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    trades = cur.fetchall()
                    
                    total_pips = gross_win_pips = gross_loss_pips = 0.0
                    winning_trades_count = losing_trades_count = 0

                    for t in trades:
                        pips, _profit_usd = compute_trade_pips({
                            "action": t["action"], "entry_price": t["entry_p"], "sl_price": t["sl_p"],
                            "tp1_price": t["tp1_p"], "tp2_price": t["tp2_p"], "exit_price": t["exit_price"], "outcome": t["outcome_val"]
                        })
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
                        SELECT id, action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val, COALESCE(timestamp, created_at::text, 'N/A') AS log_time
                        FROM signals WHERE status = 'EXECUTED' ORDER BY id DESC LIMIT 10
                    """)
                    logs = cur.fetchall()
                    cur.close()
                    conn.close()

                    if not logs:
                        reply = "📜 *LAST 10 TRADE LOGS:*\n\n_Belum ada transaksi yang tereksekusi di database._"
                    else:
                        reply = "📜 *LAST 10 DETAILED TRADE LOGS:*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        for l in logs:
                            trade_id = l["id"]
                            action = l["action"]
                            entry = float(l["entry_p"])
                            exit_p = float(l["exit_price"]) if l.get("exit_price") is not None else None
                            outcome = l["outcome_val"]
                            date_str = str(l["log_time"])

                            if exit_p is not None:
                                pips, profit_usd = compute_trade_pips({
                                    "action": action, "entry_price": entry, "sl_price": l["sl_p"],
                                    "tp1_price": l["tp1_p"], "tp2_price": l["tp2_p"], "exit_price": exit_p, "outcome": outcome
                                })
                                r_multiple = compute_r_multiple(
                                    action, entry, exit_p, float(l["sl_p"] or 0.0), float(l["tp1_p"] or 0.0), float(l["tp2_p"] or 0.0), outcome
                                )
                                pip_str = f"*{pips:+.1f} pips* | {r_multiple:+.2f}R | ${profit_usd:+.2f}"
                            else:
                                pip_str = "*ACTIVE / IN PROGRESS*"

                            icon = "🟢" if ("WIN" in outcome or "CLOSED" in outcome) else "🔴" if "LOSS" in outcome else "🟡"
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

            elif raw_text == "/analyze":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val, COALESCE(adx_15m, 0) AS adx_val, trend_15m, COALESCE(timestamp, created_at::text, '') AS ts
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    rows = cur.fetchall()
                    cur.close()
                    conn.close()

                    if not rows:
                        reply = "📐 *STRATEGY FORWARD-TEST ANALYSIS*\n\n_Not enough closed trades yet to analyze._"
                    else:
                        segments = {"Strategy": {}, "ADX Regime": {}, "Session": {}, "15m Confluence": {}}
                        overall_r = []

                        for r in rows:
                            entry = float(r["entry_p"])
                            exit_p = float(r["exit_price"])
                            sl = float(r["sl_p"] or 0.0)
                            tp1 = float(r["tp1_p"] or 0.0)
                            tp2 = float(r["tp2_p"] or 0.0)
                            outcome = str(r["outcome_val"] or "PENDING")
                            action = r["action"]
                            adx_val = float(r["adx_val"])
                            trigger = r["trigger_type"] or ""
                            trend_15m_val = r.get("trend_15m")
                            ts = r["ts"] or ""

                            r_mult = compute_r_multiple(action, entry, exit_p, sl, tp1, tp2, outcome)
                            overall_r.append(r_mult)
                            segments["Strategy"].setdefault(bucket_strategy(trigger), []).append(r_mult)
                            segments["ADX Regime"].setdefault(bucket_adx(adx_val), []).append(r_mult)
                            segments["Session"].setdefault(bucket_session(ts), []).append(r_mult)
                            segments["15m Confluence"].setdefault(bucket_confluence(action, trend_15m_val), []).append(r_mult)

                        n_total = len(overall_r)
                        overall_wr = (sum(1 for x in overall_r if x > 0) / n_total * 100) if n_total else 0.0
                        overall_avg_r = (sum(overall_r) / n_total) if n_total else 0.0

                        reply_parts = [
                            "📐 *STRATEGY FORWARD-TEST ANALYSIS*",
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | Avg R: *{overall_avg_r:+.2f}*", ""
                        ]

                        for dim in ["Strategy", "ADX Regime", "Session", "15m Confluence"]:
                            reply_parts.append(format_performance_segment(dim, segments[dim]))
                            reply_parts.append("")

                        reply_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        reply_parts.append("💡 Segments need n≥8 to be flagged ⚠️/✅.")
                        reply = "\n".join(reply_parts)

                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as an_err:
                    logging.error(f"[WEBHOOK ERROR /analyze] {an_err}")
                    await send_telegram_alert(client, f"⚠️ Error running analysis: {an_err}", target_chat_id=sender_chat_id)

    except Exception as e:
        logging.error(f"[WEBHOOK ERROR] {e}")

    return {"status": "ok"}
