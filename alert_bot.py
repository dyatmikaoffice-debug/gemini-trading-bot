# V9 AGGRESSIVE EMA 5+9 IMPULSE CATCHER: 24/7 SCANNER, RATE-LIMIT PROTECTED, MT5 HEARTBEAT
# 
# CHANGES FROM V8.1:
# 1. Sped up EMAs from 9/15 to 5/9 for earlier entries on sudden momentum shifts.
# 2. Reduced TREND_15M_MIN_SEPARATION_PCT to 0.01 to allow earlier 15M trend confirmation.
# 3. Added "Aggressive Price Impulse" trigger logic to catch massive candles that 
#    cross both EMAs before the moving averages have time to untangle.
# 4. Made EMA column mapping dynamic so logging automatically updates if EMA speeds change.

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

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

SYMBOL = "XAU/USD"

# --- GLOBAL EMERGENCY KILL SWITCH STATE & HEARTBEAT ---
SYSTEM_TRADING_ENABLED = True
CURRENT_SCAN_CYCLE_ID = None
LAST_MT5_PING_TIME = None

# --- 15M CONFLUENCE CACHE ---
cached_15m = {"df": None, "fetched_at": None}
FIFTEEN_M_REFRESH_MINUTES = 15

# --- EMA EXECUTION SIGNAL (5M chart, fast settings for early impulse capture) ---
EMA_TREND_FAST = 5
EMA_TREND_SLOW = 9

# FIXED: the 15M confluence filter previously reused the same 5/9 EMA as the 5M
# execution signal, so it reacted almost as fast as the thing it was supposed to
# be filtering -- the confluence check added far less protection than intended.
# Decoupled to a slower, dedicated pair so it behaves like an actual higher-
# timeframe trend read.
TREND_15M_EMA_FAST = 9
TREND_15M_EMA_SLOW = 20
TREND_15M_MIN_SEPARATION_PCT = 0.02

# --- SCAN SCHEDULE / TWELVE DATA BUDGET ---
ACTIVE_SESSION_START_HOUR = 0
ACTIVE_SESSION_END_HOUR = 24
TWELVE_DATA_DAILY_LIMIT = 800
TWELVE_DATA_SAFETY_MARGIN = 40 

_twelve_data_call_count = 0
_twelve_data_budget_date = None  
_twelve_data_calls_by_tf = {"5min": 0, "15min": 0}


def _reset_budget_if_new_day(now_wib: datetime):
    global _twelve_data_call_count, _twelve_data_budget_date, _twelve_data_calls_by_tf
    today = now_wib.date()
    if _twelve_data_budget_date != today:
        if _twelve_data_budget_date is not None:
            logging.info(f"[TWELVE DATA BUDGET] New WIB day. Resetting counter.")
        _twelve_data_budget_date = today
        _twelve_data_call_count = 0
        _twelve_data_calls_by_tf = {"5min": 0, "15min": 0}


def twelve_data_budget_ok(now_wib: datetime) -> bool:
    _reset_budget_if_new_day(now_wib)
    remaining = TWELVE_DATA_DAILY_LIMIT - _twelve_data_call_count
    if remaining <= TWELVE_DATA_SAFETY_MARGIN:
        logging.warning(f"[TWELVE DATA BUDGET] Only {remaining} calls left today. Throttling.")
        return False
    return True


def note_twelve_data_call(timeframe: str = None):
    global _twelve_data_call_count
    _twelve_data_call_count += 1
    if timeframe in _twelve_data_calls_by_tf:
        _twelve_data_calls_by_tf[timeframe] += 1
    if _twelve_data_call_count % 100 == 0:
        logging.info(f"[TWELVE DATA BUDGET] {_twelve_data_call_count}/{TWELVE_DATA_DAILY_LIMIT} calls used today.")


# ==========================================================
# STAT-BASED VETOES & COOLDOWNS
# ==========================================================
LOSS_COOLDOWN_MINUTES = 10

def check_stat_veto(adx_5m: float, current_hour_wib: int):
    # REMOVED: mid-session (14-18 WIB) veto. It was carried over from the old
    # liquidity-sweep strategy's forward-test and never actually validated for
    # this EMA system -- removed per request rather than kept on an unverified
    # assumption. current_hour_wib is kept as a parameter (unused for now) so the
    # call sites and signature don't need to change if a real, EMA-specific
    # session filter gets added later based on actual /analyze data.
    if adx_5m < 20.0:
        return True, (
            f"Stat-veto: 5M ADX {adx_5m:.1f} < 20.0 indicates chopping/ranging market. "
            f"EMA strategies are auto-vetoed to prevent false signals."
        )

    return False, ""


class SignalOutput(BaseModel):
    action: str = Field(default="HOLD", description="BUY, SELL, or HOLD")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(
        default="Market conditions do not favor entry.",
        description="2 clean sentences explaining the decision"
    )


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
                adx_5m REAL,
                adx_15m REAL,
                trend_15m TEXT,
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
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_15m_true REAL;"
        ]

        for query in migrations:
            cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[NEON DATABASE] Full schema verified and auto-migrated.")
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to initialize database schema: {e}")


def log_bot_event(
    event_type: str, stage: str = None, action: str = None, trigger_type: str = None,
    price: float = None, adx_5m: float = None, adx_15m: float = None, trend_15m: str = None,
    decision: str = None, reason: str = None, details: dict = None, cycle_id: str = None
):
    if not DATABASE_URL:
        return None

    conn = None
    cursor = None
    try:
        now_utc = datetime.now(timezone.utc)
        wib_time = (now_utc + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")

        def f(value):
            try: return float(value) if value is not None else None
            except Exception: return None

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bot_events (
                timestamp, cycle_id, event_type, stage, action, trigger_type,
                price, adx_5m, adx_15m, trend_15m, decision, reason, details
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
        """, (
            wib_time, cycle_id or CURRENT_SCAN_CYCLE_ID, str(event_type), str(stage) if stage is not None else None,
            str(action) if action is not None else None, str(trigger_type) if trigger_type is not None else None,
            f(price), f(adx_5m), f(adx_15m), str(trend_15m) if trend_15m is not None else None,
            str(decision) if decision is not None else None, str(reason) if reason is not None else None, 
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
    status: str, action: str, trigger_type: str, price: float, sl: float, tp1: float, tp2: float,
    confidence: float, adx_15m: float, stoch_rsi_15m: float, divergence_type: str, reasoning: str,
    trend_15m: str = None, adx_15m_true: float = None
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


def update_open_trades(current_high: float, current_low: float):
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE status = 'EXECUTED' AND (outcome = 'PENDING' OR outcome = 'WIN (TP1 HIT)')
        """)
        open_trades = cursor.fetchall()

        if not open_trades:
            cursor.close()
            conn.close()
            return

        wib_now = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
        c_high = float(current_high)
        c_low = float(current_low)

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
                        new_outcome = "WIN (TP2 HIT)"; exit_price = tp2
                    elif c_low <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"; exit_price = entry_price
                elif sl > 0 and c_low <= sl:
                    new_outcome = "LOSS (SL HIT)"; exit_price = sl
                elif tp2 > 0 and c_high >= tp2:
                    new_outcome = "WIN (TP2 HIT)"; exit_price = tp2
                elif tp1 > 0 and c_high >= tp1:
                    new_outcome = "WIN (TP1 HIT)"; exit_price = tp1

            elif action == "SELL":
                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 > 0 and c_low <= tp2:
                        new_outcome = "WIN (TP2 HIT)"; exit_price = tp2
                    elif c_high >= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"; exit_price = entry_price
                elif sl > 0 and c_high >= sl:
                    new_outcome = "LOSS (SL HIT)"; exit_price = sl
                elif tp2 > 0 and c_low <= tp2:
                    new_outcome = "WIN (TP2 HIT)"; exit_price = tp2
                elif tp1 > 0 and c_low <= tp1:
                    new_outcome = "WIN (TP1 HIT)"; exit_price = tp1

            if new_outcome and new_outcome != current_outcome:
                cursor.execute("""
                    UPDATE signals SET outcome = %s, exit_price = %s, outcome_timestamp = %s WHERE id = %s
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
                    details={"signal_id": trade_id, "pips": result_pips, "profit_usd": result_usd, "r_multiple": result_r}
                )
                logging.info(f"[TRADE UPDATE] Signal ID {trade_id} -> {new_outcome} at ${exit_price:.2f} | {result_pips:+.1f} pips | {result_r:+.2f}R")

        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to update trade outcomes: {e}")


async def fetch_timeframe_data(client: httpx.AsyncClient, timeframe: str, outputsize: int = 100, now_wib: datetime = None):
    now_wib = now_wib or (datetime.now(timezone.utc) + timedelta(hours=7))
    if not twelve_data_budget_ok(now_wib):
        logging.warning(f"[TWELVE DATA BUDGET] Skipping {timeframe} fetch - daily budget nearly exhausted.")
        return None

    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={timeframe}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    res = await client.get(url)
    note_twelve_data_call(timeframe)
    if res.status_code != 200 or not res.text: return None
    try: data = res.json()
    except Exception: return None
    if "values" not in data: return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns: df[col] = df[col].astype(float)
    return df


def calculate_metrics_tf(df: pd.DataFrame):
    df = df.tail(100).copy()
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
    tr14 = df["tr"].rolling(14).sum()
    plus_di = 100 * (df["plus_dm"].rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (df["minus_dm"].rolling(14).sum() / (tr14 + 1e-10))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.rolling(14).mean()
    df["atr"] = df["tr"].rolling(window=14).mean()
    
    # 5M execution EMAs (fast -- used for entry signals on df_5m)
    df["ema_fast"] = df["close"].ewm(span=EMA_TREND_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_TREND_SLOW, adjust=False).mean()

    # FIXED: separate, slower EMAs for the 15M confluence read. These are computed
    # on every call (cheap) so the same helper works for both the 5M and 15M frames,
    # but compute_ema_trend() below now reads THESE columns, not the fast ones.
    df["trend_ema_fast"] = df["close"].ewm(span=TREND_15M_EMA_FAST, adjust=False).mean()
    df["trend_ema_slow"] = df["close"].ewm(span=TREND_15M_EMA_SLOW, adjust=False).mean()

    return df


def compute_ema_trend(df: pd.DataFrame):
    # FIXED: was reading df["ema_fast"]/df["ema_slow"] -- the same 5/9 pair used
    # for 5M execution -- which made the "15M confluence filter" flip almost as
    # fast as the signal it was supposed to be filtering. Now reads the dedicated,
    # slower trend_ema_fast/trend_ema_slow (9/20) columns instead.
    if df is None or len(df) < TREND_15M_EMA_SLOW + 1: return "NEUTRAL", 0.0
    last_fast = float(df["trend_ema_fast"].iloc[-1])
    last_slow = float(df["trend_ema_slow"].iloc[-1])
    if last_slow == 0: return "NEUTRAL", 0.0
    separation_pct = abs(last_fast - last_slow) / last_slow * 100
    if separation_pct < TREND_15M_MIN_SEPARATION_PCT: return "NEUTRAL", separation_pct
    return "BULLISH" if last_fast > last_slow else "BEARISH", separation_pct


TOUCH_MIN_BODY_ATR_MULT = 0.15  # FIXED: touch signals previously had zero quality filter

def detect_ema_signal(df_5m: pd.DataFrame, trend_15m: str):
    if len(df_5m) < 2: return "HOLD", "Insufficient data"
    
    curr = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    
    c_ema_fast = curr["ema_fast"]
    c_ema_slow = curr["ema_slow"]
    p_ema_fast = prev["ema_fast"]
    p_ema_slow = prev["ema_slow"]
    
    # 1. PRICE IMPULSE CROSS (Aggressive Early Entry)
    # Catches massive candles that explode through both EMAs instantly
    bullish_impulse = prev["close"] <= p_ema_slow and curr["close"] > c_ema_fast and curr["close"] > c_ema_slow and curr["close"] > curr["open"]
    bearish_impulse = prev["close"] >= p_ema_slow and curr["close"] < c_ema_fast and curr["close"] < c_ema_slow and curr["close"] < curr["open"]
    
    # 2. EMA CROSSOVER (The Standard Cross)
    bullish_cross = p_ema_fast <= p_ema_slow and c_ema_fast > c_ema_slow
    bearish_cross = p_ema_fast >= p_ema_slow and c_ema_fast < c_ema_slow

    # 3. EMA TOUCH (Trend Continuation)
    # FIXED: this trigger had no candle-quality check at all -- a tiny indecisive
    # doji sitting on the EMA line qualified exactly the same as a strong reclaim
    # candle, unlike every other trigger in this file which normalizes body size
    # against ATR. Require the bounce candle to show at least modest conviction.
    raw_atr = curr["atr"] if "atr" in curr and not pd.isna(curr["atr"]) else None
    atr_val = float(raw_atr) if raw_atr is not None and raw_atr > 0 else None
    candle_body = abs(float(curr["close"]) - float(curr["open"]))
    touch_body_ok = (atr_val is None) or (candle_body / atr_val >= TOUCH_MIN_BODY_ATR_MULT)

    touch_bullish = c_ema_fast > c_ema_slow and curr["low"] <= c_ema_fast and curr["close"] > c_ema_fast and touch_body_ok
    touch_bearish = c_ema_fast < c_ema_slow and curr["high"] >= c_ema_fast and curr["close"] < c_ema_fast and touch_body_ok
    
    # Evaluate Hierarchy: Impulse > Crossover > Touch
    if bullish_impulse and trend_15m == "BULLISH":
        return "BUY", "Aggressive Price Impulse (Bullish)"
    if bearish_impulse and trend_15m == "BEARISH":
        return "SELL", "Aggressive Price Impulse (Bearish)"
        
    if bullish_cross and trend_15m == "BULLISH":
        return "BUY", f"EMA {EMA_TREND_FAST}/{EMA_TREND_SLOW} Bullish Cross"
    if bearish_cross and trend_15m == "BEARISH":
        return "SELL", f"EMA {EMA_TREND_FAST}/{EMA_TREND_SLOW} Bearish Cross"
        
    if touch_bullish and trend_15m == "BULLISH":
        return "BUY", f"EMA {EMA_TREND_FAST} Line Touch (Bullish)"
    if touch_bearish and trend_15m == "BEARISH":
        return "SELL", f"EMA {EMA_TREND_FAST} Line Touch (Bearish)"
        
    return "HOLD", "No EMA Setup"


# --- FORWARD-TEST ANALYTICS HELPERS ---
TP1_PARTIAL_CLOSE_RATIO = 0.5  # Fraction of the lot closed at TP1; rest rides to TP2/BE.
                                # CONFIRM this matches your MT5 EA's actual split -- everything
                                # below assumes 50/50. If your EA uses a different ratio, change
                                # only this constant.

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
    tp2_dist = abs(tp2 - entry) if tp2 > 0 else sl_dist * 2.5
    r = TP1_PARTIAL_CLOSE_RATIO

    # FIXED (partial-close accounting): the old formulas applied the FULL 0.01-lot
    # rate to both the TP1 leg and the TP2 leg, which double-counts profit under a
    # partial-close model -- only `r` of the lot actually captured tp1_dist, and
    # only `(1-r)` of the lot captured tp2_dist. The old flat "LOSS x2" and
    # "generic x2" multipliers are also gone: a straight SL hit (never reached
    # TP1) stops the FULL lot at exactly 1R, not 2R -- there's no partial-close
    # event to justify doubling it.
    if outcome == "CLOSED (TP1 HIT / SL BE)":
        total_pips = tp1_dist * r * 10.0
    elif outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        total_pips = (tp1_dist * r + tp2_dist * (1 - r)) * 10.0
    elif outcome == "WIN (TP1 HIT)":
        # Interim state: only the TP1 leg has actually closed so far.
        total_pips = tp1_dist * r * 10.0
    elif "LOSS" in outcome:
        total_pips = -(sl_dist * 10.0)
    else:
        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
        total_pips = diff * 10.0
    profit_usd = total_pips * 0.10
    return total_pips, profit_usd

def compute_r_multiple(action: str, entry: float, exit_price: float, sl: float, tp1: float = 0.0, tp2: float = 0.0, outcome: str = "PENDING") -> float:
    risk_dist = abs(entry - sl)
    if risk_dist <= 0:
        risk_dist = abs(entry - exit_price) if "LOSS" in outcome else 2.5
        if risk_dist == 0: risk_dist = 2.5
    r = TP1_PARTIAL_CLOSE_RATIO
    # FIXED: same partial-close accounting as compute_trade_pips above -- TP1/TP2
    # legs are weighted by the actual lot fraction each one closes, and a straight
    # loss (SL hit before TP1) is exactly -1.0R by definition (risk_dist IS 1R),
    # not the old hardcoded -2.0.
    if outcome == "CLOSED (TP1 HIT / SL BE)":
        return (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5) * r / risk_dist
    if outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        tp1_leg = (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5) * r
        tp2_leg = (abs(tp2 - entry) if tp2 > 0 else risk_dist * 2.5) * (1 - r)
        return (tp1_leg + tp2_leg) / risk_dist
    if outcome == "WIN (TP1 HIT)":
        return (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5) * r / risk_dist
    if "LOSS" in outcome: return -1.0
    return ((exit_price - entry) / risk_dist if action == "BUY" else (entry - exit_price) / risk_dist)

def bucket_adx(adx: float) -> str:
    if adx < 20: return "ADX < 20 (Chop)"
    if adx < 25: return "ADX 20-25 (Weak Trend)"
    if adx < 35: return "ADX 25-35 (Solid Trend)"
    if adx < 45: return "ADX 35-45 (Strong Trend)"
    return "ADX 45+ (Overextended)"

def bucket_strategy(trigger_type: str) -> str:
    t = trigger_type or ""
    if "Impulse" in t: return "Aggressive Price Impulse"
    if "Cross" in t: return "EMA Crossovers"
    if "Touch" in t: return "EMA Pullback/Touches"
    return "Other EMA Setup"

def bucket_session(timestamp_str: str) -> str:
    try: hour = int(str(timestamp_str).split(" ")[1].split(":")[0])
    except Exception: return "Unknown"
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
            if win_rate < 35: flag = " \u26a0\ufe0f underperforming"
            elif win_rate > 65: flag = " \u2705 strong"
        lines.append(f"\u2022 {label}: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}")
    return "\n".join(lines)


# --- TELEGRAM NOTIFICATIONS ---
async def send_telegram_alert(client: httpx.AsyncClient, text: str, target_chat_id: str = None):
    chat_id = "".join(str(target_chat_id or TELEGRAM_CHAT_ID).split())
    if not TELEGRAM_BOT_TOKEN or not chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        res = await client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        if res.status_code != 200: await client.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e: logging.error(f"[TELEGRAM EXCEPTION] {e}")


# --- AI ANALYST EVALUATION ---
async def analyze_signal_with_ai(
    proposed_action: str, trigger_type: str, current_price: float, df_5m: pd.DataFrame, 
    trend_15m: str = "NEUTRAL", adx_15m_true: float = 0.0
):
    adx_5m = float(df_5m['adx'].iloc[-1])
    plus_di_5m = float(df_5m['plus_di'].iloc[-1])
    minus_di_5m = float(df_5m['minus_di'].iloc[-1])
    di_text = f"+DI={plus_di_5m:.1f}, -DI={minus_di_5m:.1f}"
    
    c_ema_fast = float(df_5m["ema_fast"].iloc[-1])
    c_ema_slow = float(df_5m["ema_slow"].iloc[-1])

    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD) intraday scalping.
Strategy in play: EMA {EMA_TREND_FAST}+{EMA_TREND_SLOW} Trend Follower (5M Execution + 15M Confluence).
Trigger ({trigger_type}): {proposed_action} at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 5M Close=${float(df_5m['close'].iloc[-1]):.2f}
2. 5M EMAs: EMA{EMA_TREND_FAST}=${c_ema_fast:.2f}, EMA{EMA_TREND_SLOW}=${c_ema_slow:.2f}.
3. 5M ADX={adx_5m:.1f}; {di_text}.
4. 15M Trend Filter: {trend_15m}

CRITICAL SCALP VETO RULES:
- VETO if 5M ADX < 20.0 (Choppy/Ranging market, EMA setups will fail).
- Ensure price action agrees with momentum.

Respond strictly in valid JSON matching schema:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, "reasoning": "2 concise sentences explaining decision"}}
"""

    if GROQ_API_KEY:
        try:
            res = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, response_format={"type": "json_object"}
            )
            return SignalOutput(**json.loads(res.choices[0].message.content))
        except Exception as e:
            logging.warning(f"[AI WARNING] Groq call failed: {e}. Falling back to Gemini.")

    if GEMINI_API_KEY:
        try:
            res = genai_client.models.generate_content(
                model="gemini-3.7-flash",
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

    return SignalOutput(action=proposed_action, confidence=0.7, reasoning="Fallback: Executed on pure EMA structural alignment.")


# --- BACKGROUND SCANNING LOOP (RATE-LIMIT PROTECTED) ---
async def background_scanning_loop():
    global SYSTEM_TRADING_ENABLED, CURRENT_SCAN_CYCLE_ID, cached_15m

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        last_processed_candle_time = None
        last_claimed_bucket = None

        while True:
            try:
                now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
                current_hour_wib = now_wib.hour

                if not SYSTEM_TRADING_ENABLED:
                    if now_wib.minute % 5 == 0 and now_wib.second < 5:
                        logging.info("[SLEEP STATUS] Bot is PAUSED via kill switch. Waiting...")
                    await asyncio.sleep(60)
                    continue

                active_session = ACTIVE_SESSION_START_HOUR <= current_hour_wib < ACTIVE_SESSION_END_HOUR
                if not active_session:
                    if now_wib.minute % 5 == 0 and now_wib.second < 5:
                        logging.info(f"[SLEEP STATUS] Out of session ({ACTIVE_SESSION_START_HOUR}:00 - {ACTIVE_SESSION_END_HOUR}:00 WIB). Waiting...")
                    await asyncio.sleep(60)
                    continue

                if now_wib.minute % 5 != 0 or now_wib.second > 45:
                    await asyncio.sleep(2)
                    continue

                current_bucket = now_wib.strftime("%Y-%m-%d %H:%M")
                if current_bucket == last_claimed_bucket:
                    await asyncio.sleep(20)
                    continue

                last_claimed_bucket = current_bucket

                if not twelve_data_budget_ok(now_wib):
                    await asyncio.sleep(20)
                    continue

                df_5m = await fetch_timeframe_data(client, "5min", now_wib=now_wib)
                if df_5m is None or len(df_5m) < 6:
                    logging.warning("[SCAN LOOP] 5M fetch failed or insufficient data this window; will retry next candle.")
                    await asyncio.sleep(15)
                    continue

                df_5m = calculate_metrics_tf(df_5m)
                candle_time_5m = df_5m["datetime"].iloc[-1]

                if last_processed_candle_time is not None and candle_time_5m == last_processed_candle_time:
                    await asyncio.sleep(5)
                    continue

                last_processed_candle_time = candle_time_5m

                CURRENT_SCAN_CYCLE_ID = f"{now_wib.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
                log_scan_event("SCAN_START", stage="SCAN", decision="STARTED", reason="Scanner cycle started (5M Execution)")

                curr_high = float(df_5m["high"].iloc[-1])
                curr_low = float(df_5m["low"].iloc[-1])
                update_open_trades(curr_high, curr_low)

                curr_price = float(df_5m["close"].iloc[-1])
                curr_ema_fast = float(df_5m["ema_fast"].iloc[-1])
                curr_ema_slow = float(df_5m["ema_slow"].iloc[-1])
                adx_5m = float(df_5m["adx"].iloc[-1]) if not pd.isna(df_5m["adx"].iloc[-1]) else 0.0

                # REMOVED: this used to skip the 15M refresh entirely during the
                # mid-session veto window, since trades were blocked there anyway
                # so refreshing was "wasted." Now that the veto is gone, trades can
                # execute during those hours too -- skipping the refresh would mean
                # confluence checks run on stale (potentially hours-old) 15M data
                # exactly when it matters. Refresh logic is now the same 24/7.
                need_refresh = (cached_15m["df"] is None or cached_15m["fetched_at"] is None or (datetime.now(timezone.utc) - cached_15m["fetched_at"] >= timedelta(minutes=FIFTEEN_M_REFRESH_MINUTES)))

                if need_refresh and twelve_data_budget_ok(now_wib):
                    df_15m_raw = await fetch_timeframe_data(client, "15min", now_wib=now_wib)
                    if df_15m_raw is not None and len(df_15m_raw) >= 21:
                        cached_15m["df"] = calculate_metrics_tf(df_15m_raw)
                        cached_15m["fetched_at"] = datetime.now(timezone.utc)

                trend_15m, trend_15m_sep = compute_ema_trend(cached_15m["df"]) if cached_15m["df"] is not None else ("NEUTRAL", 0.0)
                adx_15m_true = float(cached_15m["df"]["adx"].iloc[-1]) if cached_15m["df"] is not None and not pd.isna(cached_15m["df"]["adx"].iloc[-1]) else 0.0

                proposed_action, trigger_type = detect_ema_signal(df_5m, trend_15m)

                # --- VETO CHECKS ---
                if proposed_action != "HOLD":
                    stat_vetoed, stat_veto_reason = check_stat_veto(adx_5m, current_hour_wib)
                    if stat_vetoed:
                        logging.info(f"[STAT VETO] {stat_veto_reason}")
                        proposed_action = "HOLD"

                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection(); cursor = conn.cursor()
                        cursor.execute("SELECT outcome_timestamp FROM signals WHERE status = 'EXECUTED' AND outcome = 'LOSS (SL HIT)' AND outcome_timestamp IS NOT NULL AND outcome_timestamp != '' ORDER BY id DESC LIMIT 1")
                        last_loss = cursor.fetchone()
                        cursor.close(); conn.close()
                        if last_loss and last_loss.get("outcome_timestamp"):
                            last_loss_time = datetime.strptime(str(last_loss["outcome_timestamp"]).replace(" WIB", ""), "%Y-%m-%d %H:%M:%S")
                            if 0 <= (now_wib.replace(tzinfo=None) - last_loss_time).total_seconds() / 60.0 < LOSS_COOLDOWN_MINUTES:
                                logging.info(f"[LOSS COOLDOWN] Skipping {proposed_action}.")
                                proposed_action = "HOLD"
                    except Exception as e: logging.error(e)

                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection(); cursor = conn.cursor()
                        cursor.execute("SELECT COALESCE(entry_price, price, 0) AS entry_p, outcome FROM signals WHERE status = 'EXECUTED' AND action = %s ORDER BY id DESC LIMIT 1", (str(proposed_action),))
                        last_trade = cursor.fetchone()
                        cursor.close(); conn.close()
                        if last_trade:
                            required_distance = 2.00 if str(last_trade.get("outcome") or "PENDING") == "PENDING" else 1.50
                            if abs(curr_price - float(last_trade["entry_p"])) < required_distance:
                                logging.info(f"[DISTANCE COOLDOWN] Skipping {proposed_action}: Price too close.")
                                proposed_action = "HOLD"
                    except Exception as e: logging.error(e)

                # --- AI EVALUATION & FINAL LOGGING ---
                if proposed_action == "HOLD":
                    logging.info(f"[MARKET SCAN] Price: ${curr_price:.2f} | EMA{EMA_TREND_FAST}: ${curr_ema_fast:.2f} | EMA{EMA_TREND_SLOW}: ${curr_ema_slow:.2f} | ADX5m: {adx_5m:.1f} | 15mTrend: {trend_15m} | Status: HOLD | Cycle End")
                else:
                    logging.info(f"[MARKET SCAN] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI...")
                    ai_decision = await analyze_signal_with_ai(proposed_action, trigger_type, curr_price, df_5m, trend_15m, adx_15m_true)

                    atr_5m = float(df_5m["atr"].iloc[-1]) if not pd.isna(df_5m["atr"].iloc[-1]) else 3.0
                    risk = max(2.5, atr_5m * 1.0)
                    tp1_r_mult = 1.5
                    tp2_r_mult = 2.5 

                    sl_price = curr_price - risk if proposed_action == "BUY" else curr_price + risk
                    tp1_price = curr_price + risk * tp1_r_mult if proposed_action == "BUY" else curr_price - risk * tp1_r_mult
                    tp2_price = curr_price + risk * tp2_r_mult if proposed_action == "BUY" else curr_price - risk * tp2_r_mult

                    if ai_decision.action == proposed_action:
                        new_id = log_trade_signal("EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning, trend_15m, adx_15m_true)
                        msg = (f"\U0001f680 *SIGNAL #{new_id}*\n\nAsset: *XAUUSD*\nAction: *{proposed_action}*\nType: *{trigger_type}*\nEntry Price: *${curr_price:.2f}*\n\nStop Loss: *${sl_price:.2f}*\nTP1 ({tp1_r_mult:.1f}R): *${tp1_price:.2f}*\nTP2 ({tp2_r_mult:.1f}R): *${tp2_price:.2f}*\n\nReasoning: {ai_decision.reasoning}")
                        await send_telegram_alert(client, msg)
                    else:
                        log_trade_signal("VETOED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning, trend_15m, adx_15m_true)

                del df_5m; gc.collect()
                await asyncio.sleep(5)

            except Exception as e:
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
    return {"status": "ok", "message": f"EMA {EMA_TREND_FAST}+{EMA_TREND_SLOW} Trading bot scanner active."}


# =====================================================================
# MT5 COPIER BRIDGE API ENDPOINT
# =====================================================================
@app.get("/get-latest-signal")
async def get_latest_signal():
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME

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
            return {"id": int(row["id"]), "action": str(row["action"]).upper(), "entry": float(row["entry_p"]), "sl": float(row["sl_p"]), "tp1": float(row["tp1_p"]), "tp2": float(row["tp2_p"]), "timestamp": str(row["log_time"]), "trading_enabled": True}
        return {"signal": None, "trading_enabled": True}
    except Exception as e:
        logging.error(f"[MT5 BRIDGE ERROR] {e}")
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

        if not sender_chat_id or not raw_text: return {"status": "ignored"}

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            if raw_text in ["/help", "/start"]:
                reply = (
                    f"\U0001f916 *EMA {EMA_TREND_FAST}+{EMA_TREND_SLOW} BOT COMMANDS:*\n\n"
                    "\u2022 `/status` - \U0001f4e1 Real-Time MT5, Server & API Budget Status\n"
                    "\u2022 `/stats` - Comprehensive Win-Rate & Risk Analytics Dashboard\n"
                    "\u2022 `/pips` - Detailed Gross/Net Pips & USD Profit Breakdown (0.01 Lot)\n"
                    "\u2022 `/logs` - Detailed View of Last 10 Trades & Outcomes\n"
                    "\u2022 `/analyze` - Forward-Test Breakdown by Strategy, ADX Regime & Session\n"
                    "\u2022 `/pause` - \U0001f6d1 *EMERGENCY KILL SWITCH* (Stop Bot & MT5 auto-trade)\n"
                    "\u2022 `/resume` - \U0001f7e2 Re-enable Auto-Trading Execution\n"
                    "\u2022 `/help` - Display Command Menu\n\n"
                    f"\u26a0\ufe0f Active stat-vetoes: ADX < 20.0 (Choppy limits).\n"
                    f"\u23f1\ufe0f Loss cooldown: {LOSS_COOLDOWN_MINUTES} min after any SL hit "
                    f"(any direction) before a new signal can execute."
                )
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/status":
                if LAST_MT5_PING_TIME:
                    now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
                    seconds_ago = (now_wib.replace(tzinfo=None) - LAST_MT5_PING_TIME.replace(tzinfo=None)).total_seconds()

                    if seconds_ago < 60:
                        status_icon = "\U0001f7e2"
                        conn_msg = f"Connected and Active\n\u2022 Last ping: *{seconds_ago:.0f}s ago*"
                    elif seconds_ago < 180:
                        status_icon = "\U0001f7e1"
                        conn_msg = f"Slight Lag\n\u2022 Last ping: *{seconds_ago:.0f}s ago*"
                    else:
                        status_icon = "\U0001f534"
                        conn_msg = f"DISCONNECTED\n\u2022 Last ping was *{seconds_ago:.0f}s ago*! Please check your MT5 terminal."

                    remaining = TWELVE_DATA_DAILY_LIMIT - _twelve_data_call_count
                    budget_pct = (_twelve_data_call_count / TWELVE_DATA_DAILY_LIMIT * 100) if TWELVE_DATA_DAILY_LIMIT else 0.0
                    budget_icon = "\U0001f7e2" if budget_pct < 70 else ("\U0001f7e1" if budget_pct < 90 else "\U0001f534")
                    reply = (
                        f"{status_icon} *SYSTEM & BRIDGE STATUS*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\u2022 Trading State: *{'ACTIVE' if SYSTEM_TRADING_ENABLED else 'PAUSED (Kill-Switch)'}*\n"
                        f"\u2022 MT5 Bridge: *{conn_msg}*\n"
                        f"\u2022 Server Time: `{now_wib.strftime('%Y-%m-%d %H:%M:%S WIB')}`\n\n"
                        f"{budget_icon} *TWELVEDATA API BUDGET:*\n"
                        f"\u2022 Used Today: *{_twelve_data_call_count}/{TWELVE_DATA_DAILY_LIMIT}* ({budget_pct:.0f}%) | Remaining: *{remaining}*\n"
                        f"  \u2514\u2500 5M: {_twelve_data_calls_by_tf['5min']} | 15M: {_twelve_data_calls_by_tf['15min']}\n\n"
                        f"\U0001f4c8 *STRATEGY:*\n"
                        f"\u2022 Execution (5M): *EMA {EMA_TREND_FAST}/{EMA_TREND_SLOW}*\n"
                        f"\u2022 Confluence (15M): *EMA {TREND_15M_EMA_FAST}/{TREND_15M_EMA_SLOW}*"
                    )
                else:
                    reply = "\U0001f534 *MT5 DISCONNECTED*\n\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\nThe server is running, but MT5 has not sent any pings since the last reboot."
                await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

            elif raw_text == "/pause":
                SYSTEM_TRADING_ENABLED = False
                await send_telegram_alert(client, "\U0001f6d1 *EMERGENCY KILL SWITCH ACTIVATED*\nMarket scanner paused. Send `/resume` to reactivate.", target_chat_id=sender_chat_id)

            elif raw_text == "/resume":
                SYSTEM_TRADING_ENABLED = True
                await send_telegram_alert(client, "\U0001f7e2 *AUTO-TRADING SYSTEM RESUMED*\nScanner loop is now active.", target_chat_id=sender_chat_id)

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

                    cur.execute("SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price, COALESCE(outcome, 'PENDING') AS outcome_val FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                    closed_trades = cur.fetchall()
                    total_pips = win_pips = loss_pips = 0.0
                    total_wins_count = tp1_wins + tp2_wins

                    for t in closed_trades:
                        trade_pips, _ = compute_trade_pips({"action": t["action"], "entry_price": t["entry_p"], "sl_price": t["sl_p"], "tp1_price": t["tp1_p"], "tp2_price": t["tp2_p"], "exit_price": t["exit_price"], "outcome": t["outcome_val"]})
                        total_pips += trade_pips
                        if trade_pips > 0: win_pips += trade_pips
                        elif trade_pips < 0: loss_pips += abs(trade_pips)

                    win_rate = (total_wins_count / total_executed * 100) if total_executed > 0 else 0.0
                    est_dollar = total_pips * 0.10
                    avg_win = (win_pips / total_wins_count) if total_wins_count > 0 else 0.0
                    avg_loss = (loss_pips / losses) if losses > 0 else 0.0
                    profit_factor = (win_pips / loss_pips) if loss_pips > 0 else (win_pips if win_pips > 0 else 0.0)
                    cur.close(); conn.close()

                    reply = (
                        f"\U0001f4ca *PERFORMANCE ANALYTICS DASHBOARD*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\U0001f4b0 *NET PIPS & PROFIT:*\n"
                        f"\u2022 Net Pips: *{total_pips:+.1f} pips*\n"
                        f"\u2022 Est. Profit (0.01 Lot): *${est_dollar:+.2f}*\n\n"
                        f"\U0001f4c8 *WIN / LOSS BREAKDOWN:*\n"
                        f"\u2022 Total Executed: *{total_executed}*\n"
                        f"\u2022 Total Wins: *{total_wins_count} ({win_rate:.1f}%)*\n"
                        f"  \u2514\u2500 Hit TP1 (BE Runner): *{tp1_wins}*\n"
                        f"  \u2514\u2500 Hit TP2 (Full Target): *{tp2_wins}*\n"
                        f"\u2022 Total Losses (SL Hit): *{losses}*\n"
                        f"\u2022 Active Pending: *{total_pending}*\n\n"
                        f"\u26a1 *SYSTEM & AI EFFICIENCY:*\n"
                        f"\u2022 Total Signals: *{total_executed + total_vetoes}*\n"
                        f"\u2022 AI Vetoed Signals: *{total_vetoes}*\n\n"
                        f"\U0001f3af *RISK & TRADE METRICS:*\n"
                        f"\u2022 Avg Win: *+{avg_win:.1f} pips* | Avg Loss: *-{avg_loss:.1f} pips*\n"
                        f"\u2022 Profit Factor: *{profit_factor:.2f}*\n"
                        f"\u2022 Win Rate: *{win_rate:.1f}%*"
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as e: await send_telegram_alert(client, f"\u26a0\ufe0f Error querying stats: {e}", target_chat_id=sender_chat_id)

            elif raw_text == "/pips":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p,
                               exit_price, COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    trades = cur.fetchall()
                    cur.close(); conn.close()

                    total_pips = gross_win_pips = gross_loss_pips = 0.0
                    winning_trades_count = losing_trades_count = 0

                    for t in trades:
                        pips, _usd = compute_trade_pips({
                            "action": t["action"], "entry_price": t["entry_p"], "sl_price": t["sl_p"],
                            "tp1_price": t["tp1_p"], "tp2_price": t["tp2_p"], "exit_price": t["exit_price"],
                            "outcome": t["outcome_val"]
                        })
                        total_pips += pips
                        if pips > 0:
                            gross_win_pips += pips; winning_trades_count += 1
                        elif pips < 0:
                            gross_loss_pips += abs(pips); losing_trades_count += 1

                    avg_win_pips = (gross_win_pips / winning_trades_count) if winning_trades_count > 0 else 0.0
                    avg_loss_pips = (gross_loss_pips / losing_trades_count) if losing_trades_count > 0 else 0.0
                    est_profit_usd = total_pips * 0.10
                    pip_efficiency = gross_win_pips / (gross_loss_pips + 1e-5)

                    reply = (
                        f"\U0001f4b5 *DETAILED PIPS & EARNINGS REPORT*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\U0001f4ca *SUMMARY:*\n"
                        f"\u2022 Total Net Pips: *{total_pips:+.1f} pips*\n"
                        f"\u2022 Net Profit (0.01 Lot): *${est_profit_usd:+.2f}*\n\n"
                        f"\U0001f4c8 *PIPS BREAKDOWN:*\n"
                        f"\u2022 Gross Gain: *+{gross_win_pips:.1f} pips*\n"
                        f"\u2022 Gross Loss: *-{gross_loss_pips:.1f} pips*\n\n"
                        f"\U0001f3af *AVERAGE METRICS:*\n"
                        f"\u2022 Avg Win Trade: *+{avg_win_pips:.1f} pips*\n"
                        f"\u2022 Avg Loss Trade: *-{avg_loss_pips:.1f} pips*\n"
                        f"\u2022 Pip Efficiency Ratio: *{pip_efficiency:.2f}*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\U0001f4a1 *Note:* $0.10/pip = full 0.01 lot. TP1/TP2 legs weighted "
                        f"{int(TP1_PARTIAL_CLOSE_RATIO*100)}/{int((1-TP1_PARTIAL_CLOSE_RATIO)*100)} "
                        f"per your partial-close split."
                    )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as e:
                    await send_telegram_alert(client, f"\u26a0\ufe0f Error calculating pips: {e}", target_chat_id=sender_chat_id)

            elif raw_text == "/logs":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p,
                               COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                               COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val,
                               COALESCE(timestamp, created_at::text, 'N/A') AS log_time
                        FROM signals WHERE status = 'EXECUTED' ORDER BY id DESC LIMIT 10
                    """)
                    logs = cur.fetchall()
                    cur.close(); conn.close()

                    if not logs:
                        reply = "\U0001f4dc *LAST 10 TRADE LOGS:*\n\n_No executed trades in the database yet._"
                    else:
                        reply = "\U0001f4dc *LAST 10 DETAILED TRADE LOGS:*\n\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n\n"
                        for l in logs:
                            trade_id = l["id"]; action = l["action"]
                            entry = float(l["entry_p"])
                            exit_p = float(l["exit_price"]) if l.get("exit_price") is not None else None
                            outcome = l["outcome_val"]
                            date_str = str(l["log_time"])

                            if exit_p is not None:
                                pips, profit_usd = compute_trade_pips({
                                    "action": action, "entry_price": entry, "sl_price": l["sl_p"],
                                    "tp1_price": l["tp1_p"], "tp2_price": l["tp2_p"], "exit_price": exit_p,
                                    "outcome": outcome
                                })
                                r_multiple = compute_r_multiple(
                                    action, entry, exit_p, float(l["sl_p"] or 0.0),
                                    float(l["tp1_p"] or 0.0), float(l["tp2_p"] or 0.0), outcome
                                )
                                pip_str = f"*{pips:+.1f} pips* | {r_multiple:+.2f}R | ${profit_usd:+.2f}"
                            else:
                                pip_str = "*ACTIVE / IN PROGRESS*"

                            if "WIN" in outcome or "CLOSED" in outcome:
                                icon = "\U0001f7e2"
                            elif "LOSS" in outcome:
                                icon = "\U0001f534"
                            else:
                                icon = "\U0001f7e1"

                            reply += (
                                f"{icon} *ID #{trade_id} | {action} XAU/USD*\n"
                                f"\u2022 Entry: ${entry:.2f} \u2192 Exit: *${(exit_p if exit_p else 0.0):.2f}*\n"
                                f"\u2022 Outcome: *{outcome}*\n"
                                f"\u2022 Result: {pip_str} | Time: {date_str}\n"
                                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                            )
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as e:
                    await send_telegram_alert(client, f"\u26a0\ufe0f Error querying logs: {e}", target_chat_id=sender_chat_id)

            elif raw_text == "/analyze":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p,
                               COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                               COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val, adx_15m,
                               COALESCE(timestamp, created_at::text, '') AS log_time,
                               trend_15m
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    rows = cur.fetchall()
                    cur.close(); conn.close()

                    if not rows:
                        reply = (
                            "\U0001f4d0 *EMA STRATEGY FORWARD-TEST ANALYSIS*\n\n"
                            "_Not enough closed trades yet to analyze. Check back after more signals complete._"
                        )
                    else:
                        segments = {"Strategy": {}, "ADX Regime": {}, "Session": {}, "15m Confluence": {}}
                        overall_r = []
                        for r in rows:
                            r_mult = compute_r_multiple(
                                r["action"], float(r["entry_p"]), float(r["exit_price"]), float(r["sl_p"]),
                                float(r["tp1_p"]), float(r["tp2_p"]), r["outcome_val"]
                            )
                            overall_r.append(r_mult)
                            adx_val = float(r["adx_15m"]) if r["adx_15m"] is not None else 0.0
                            segments["Strategy"].setdefault(bucket_strategy(r["trigger_type"]), []).append(r_mult)
                            segments["ADX Regime"].setdefault(bucket_adx(adx_val), []).append(r_mult)
                            segments["Session"].setdefault(bucket_session(r["log_time"]), []).append(r_mult)
                            segments["15m Confluence"].setdefault(bucket_confluence(r["action"], r["trend_15m"]), []).append(r_mult)

                        n_total = len(overall_r)
                        overall_wr = (sum(1 for x in overall_r if x > 0) / n_total * 100) if n_total else 0.0
                        overall_avg_r = (sum(overall_r) / n_total) if n_total else 0.0

                        reply_parts = [
                            "\U0001f4d0 *EMA STRATEGY FORWARD-TEST ANALYSIS*",
                            "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | Avg R: *{overall_avg_r:+.2f}*",
                            "",
                        ]
                        for dim in ["Strategy", "ADX Regime", "Session", "15m Confluence"]:
                            reply_parts.append(format_performance_segment(dim, segments[dim]))
                            reply_parts.append("")

                        reply_parts.append("\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015")
                        reply_parts.append(
                            "\U0001f4a1 Segments need n\u22658 to be flagged \u26a0\ufe0f/\u2705 (smaller samples are shown "
                            "but noisy).\n\n"
                            f"\U0001f6ab Active stat-vetoes: ADX < 20 (Choppy limits) only.\n"
                            f"\u23f1\ufe0f Loss cooldown ({LOSS_COOLDOWN_MINUTES} min, any direction) is also active."
                        )
                        reply = "\n".join(reply_parts)
                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)
                except Exception as e:
                    await send_telegram_alert(client, f"\u26a0\ufe0f Error: {e}", target_chat_id=sender_chat_id)

    except Exception as e: logging.error(f"[WEBHOOK ERROR] {e}")
    return {"status": "ok"}
