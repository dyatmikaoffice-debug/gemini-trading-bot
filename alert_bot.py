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

# --- LOGGING CONFIGURATION ---
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

# --- GLOBAL EMERGENCY KILL SWITCH STATE ---
SYSTEM_TRADING_ENABLED = True
CURRENT_SCAN_CYCLE_ID = None

# --- PULLBACK STATE (Stage 2 of the liquidity sweep strategy) ---
pending_setup = None
PULLBACK_EXPIRY_MINUTES = 10
FIB_RETRACE_MIN = 0.382
FIB_RETRACE_MAX = 0.618

# --- 15M CONFLUENCE CACHE ---
cached_15m = {"df": None, "fetched_at": None}
FIFTEEN_M_REFRESH_MINUTES = 15

# --- SCAN SCHEDULE / TWELVE DATA BUDGET ---
SCAN_INTERVAL_SECONDS = 120  # M1 + M5 refresh every 2 minutes
ACTIVE_SESSION_START_HOUR = 10  # WIB
ACTIVE_SESSION_END_HOUR = 22    # WIB
MID_SESSION_START_HOUR = 14     # WIB
MID_SESSION_END_HOUR = 18       # WIB

TWELVE_DATA_DAILY_LIMIT = 800
LEVEL_LOOKBACK_CANDLES = 4

# ==========================================================
# MOMENTUM BREAKOUT QUALITY FILTERS
# ==========================================================
BREAKOUT_MIN_BODY_ATR_MULT = 0.50
BREAKOUT_MIN_CLOSE_LOCATION = 0.65
BREAKOUT_MAX_EXTENSION_ATR_MULT = 1.20

# ----------------------------------------------------------
# MOMENTUM CONTINUATION MODE
# ----------------------------------------------------------
MOMENTUM_CONTINUATION_MAX_EXTENSION_ATR_MULT = 1.80
MOMENTUM_CONTINUATION_MIN_BODY_ATR_MULT = 0.35
MOMENTUM_CONTINUATION_MIN_CLOSE_LOCATION = 0.60
MOMENTUM_CONTINUATION_MIN_ADX = 20.0
BREAKOUT_REQUIRE_BODY_DIRECTION = True

# --- STAT-BASED VETOES ---
STAT_VETO_ADX_THRESHOLD = 50.0
STAT_VETO_MID_SESSION_START_HOUR = 14  # WIB
STAT_VETO_MID_SESSION_END_HOUR = 18    # WIB

# --- LOSS-COOLDOWN ---
LOSS_COOLDOWN_MINUTES = 10


def check_stat_veto(adx_5m: float, current_hour_wib: int):
    """Hard pre-AI veto based on forward-tested underperforming segments."""
    if adx_5m >= STAT_VETO_ADX_THRESHOLD:
        return True, (
            f"Stat-veto: 5M ADX {adx_5m:.1f} >= {STAT_VETO_ADX_THRESHOLD:.1f} "
            f"(forward-test: ADX 50+ regime n=15, WR=27%, AvgR=-0.27 -- likely exhaustion/blow-off, not sustained trend)"
        )

    if STAT_VETO_MID_SESSION_START_HOUR <= current_hour_wib < STAT_VETO_MID_SESSION_END_HOUR:
        return True, (
            f"Stat-veto: Mid session ({STAT_VETO_MID_SESSION_START_HOUR}-{STAT_VETO_MID_SESSION_END_HOUR} WIB) "
            f"forward-tested as underperforming (n=17, WR=29%, AvgR=-0.18)"
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

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )


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

        logging.info(
            "[NEON DATABASE] Full schema verified and missing columns auto-migrated."
        )

    except Exception as e:
        logging.error(
            f"[NEON DB ERROR] Failed to initialize database schema: {e}"
        )


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
    """Best-effort structured audit logging. Never stops the scanner."""
    if not DATABASE_URL:
        return None

    conn = None
    cursor = None
    try:
        now_utc = datetime.now(timezone.utc)
        wib_time = (now_utc + timedelta(hours=7)).strftime(
            "%Y-%m-%d %H:%M:%S WIB"
        )
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


def logged_breakout_check(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    allow_momentum_continuation: bool = False
):
    """Run the existing detector unchanged, then persist its result."""
    result = detect_breakout_continuation(
        df_1m,
        df_5m,
        allow_momentum_continuation=allow_momentum_continuation
    )
    action, trigger_type, recent_high, recent_low = result

    curr = df_1m.iloc[-1]
    curr_open = float(curr["open"])
    curr_high = float(curr["high"])
    curr_low = float(curr["low"])
    curr_close = float(curr["close"])
    raw_atr = df_1m["atr"].iloc[-1] if "atr" in df_1m.columns else None
    atr = float(raw_atr) if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0 else None
    body_atr = abs(curr_close - curr_open) / atr if atr else None
    candle_range = curr_high - curr_low
    close_location = ((curr_close - curr_low) / candle_range) if candle_range > 0 else 0.5
    extension_atr = None
    if atr:
        if action == "BUY":
            extension_atr = (curr_close - recent_high) / atr
        elif action == "SELL":
            extension_atr = (recent_low - curr_close) / atr
        else:
            extension_atr = max(
                (curr_close - recent_high) / atr,
                (recent_low - curr_close) / atr
            )

    log_bot_event(
        event_type="BREAKOUT_EVALUATION",
        stage="MOMENTUM_CONTINUATION" if allow_momentum_continuation else "FRESH_BREAKOUT",
        action=action,
        trigger_type=trigger_type,
        price=float(df_1m["close"].iloc[-1]),
        recent_high=recent_high,
        recent_low=recent_low,
        extension_atr=extension_atr,
        body_atr=body_atr,
        close_location=close_location,
        decision="VALID" if action != "HOLD" else "REJECTED_OR_NONE",
        reason=trigger_type,
        details={"continuation_mode": allow_momentum_continuation}
    )
    return result


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

        wib_time = (
            datetime.now(timezone.utc) + timedelta(hours=7)
        ).strftime("%Y-%m-%d %H:%M:%S WIB")

        price_val = float(price) if price is not None else 0.0
        sl_val = float(sl) if sl is not None else 0.0
        tp1_val = float(tp1) if tp1 is not None else 0.0
        tp2_val = float(tp2) if tp2 is not None else 0.0
        conf_val = float(confidence) if confidence is not None else 0.0
        adx_val = float(adx_15m) if adx_15m is not None else 0.0
        stoch_val = float(stoch_rsi_15m) if stoch_rsi_15m is not None else 0.0

        trend_15m_val = (
            str(trend_15m)
            if trend_15m is not None
            else None
        )

        adx_15m_true_val = (
            float(adx_15m_true)
            if adx_15m_true is not None
            else None
        )

        cursor.execute("""
            INSERT INTO signals (
                timestamp,
                status,
                action,
                trigger_type,
                price,
                entry_price,
                sl,
                sl_price,
                tp1,
                tp1_price,
                tp2,
                tp2_price,
                confidence,
                adx_15m,
                stoch_rsi_15m,
                divergence_type,
                reasoning,
                outcome,
                outcome_timestamp,
                trend_15m,
                adx_15m_true,
                created_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, NOW()
            )
            RETURNING id;
        """, (
            str(wib_time),
            str(status),
            str(action),
            str(trigger_type),
            price_val,
            price_val,
            sl_val,
            sl_val,
            tp1_val,
            tp1_val,
            tp2_val,
            tp2_val,
            conf_val,
            adx_val,
            stoch_val,
            str(divergence_type),
            str(reasoning),
            "PENDING",
            "",
            trend_15m_val,
            adx_15m_true_val
        ))

        inserted_row = cursor.fetchone()
        new_id = (
            inserted_row["id"]
            if inserted_row and "id" in inserted_row
            else None
        )

        conn.commit()
        cursor.close()
        conn.close()

        logging.info(
            f"[NEON DB LOGGED] Signal ID #{new_id} | "
            f"Status: {status} | Action: {action} | Price: ${price_val:.2f}"
        )

        return new_id

    except Exception as e:
        logging.error(
            f"[NEON DB ERROR] Failed to log signal: {e}"
        )
        return None


# --- TWO-STAGE TP TRACKING FUNCTION ---
def update_open_trades(current_high: float, current_low: float):
    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM signals
            WHERE status = 'EXECUTED'
              AND (
                  outcome = 'PENDING'
                  OR outcome = 'WIN (TP1 HIT)'
              )
        """)

        open_trades = cursor.fetchall()

        if not open_trades:
            cursor.close()
            conn.close()
            return

        wib_now = (
            datetime.now(timezone.utc) + timedelta(hours=7)
        ).strftime("%Y-%m-%d %H:%M:%S WIB")

        c_high = float(current_high)
        c_low = float(current_low)

        for trade in open_trades:
            trade_id = trade["id"]
            action = trade["action"]

            entry_price = float(
                trade["entry_price"]
                if trade.get("entry_price") is not None
                else trade.get("price", 0.0)
            )

            sl = float(
                trade["sl_price"]
                if trade.get("sl_price") is not None
                else trade.get("sl", 0.0)
            )

            tp1 = float(
                trade["tp1_price"]
                if trade.get("tp1_price") is not None
                else trade.get("tp1", 0.0)
            )

            tp2 = float(
                trade["tp2_price"]
                if trade.get("tp2_price") is not None
                else trade.get("tp2", 0.0)
            )

            current_outcome = trade["outcome"]
            new_outcome = None
            exit_price = None

            if action == "BUY":

                if current_outcome == "WIN (TP1 HIT)":
                    if tp2 > 0 and c_high >= tp2:
                        new_outcome = "WIN (TP2 HIT)"
                        exit_price = tp2

                    elif c_low <= entry_price:
                        new_outcome = "CLOSED (TP1 HIT / SL BE)"
                        exit_price = entry_price

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
                        exit_price = entry_price

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
                    SET outcome = %s,
                        exit_price = %s,
                        outcome_timestamp = %s
                    WHERE id = %s
                """, (
                    new_outcome,
                    float(exit_price),
                    wib_now,
                    trade_id
                ))

                conn.commit()

                trade_for_calc = {
                    "action": action,
                    "entry_price": entry_price,
                    "sl_price": sl,
                    "tp1_price": tp1,
                    "tp2_price": tp2,
                    "exit_price": float(exit_price),
                    "outcome": new_outcome,
                }
                result_pips, result_usd = compute_trade_pips(trade_for_calc)
                result_r = compute_r_multiple(
                    action, entry_price, float(exit_price), sl,
                    tp1, tp2, new_outcome
                )

                log_bot_event(
                    "TRADE_OUTCOME",
                    stage="TRADE_MANAGEMENT",
                    action=action,
                    price=float(exit_price),
                    decision=new_outcome,
                    reason="Two-stage TP/SL outcome detected",
                    details={
                        "signal_id": trade_id,
                        "entry": entry_price,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "exit": float(exit_price),
                        "outcome": new_outcome,
                        "pips": result_pips,
                        "profit_usd": result_usd,
                        "r_multiple": result_r,
                        "legacy_fallback_used": not bool(sl > 0 and tp1 > 0)
                    }
                )

                logging.info(
                    f"[TRADE UPDATE] Signal ID {trade_id} "
                    f"-> {new_outcome} at ${exit_price:.2f} | "
                    f"Result: {result_pips:+.1f} pips | "
                    f"{result_r:+.2f}R | ${result_usd:+.2f}"
                )

        cursor.close()
        conn.close()

    except Exception as e:
        logging.error(
            f"[NEON DB ERROR] Failed to update trade outcomes: {e}"
        )


# --- MARKET DATA FETCHING ---
async def fetch_timeframe_data(
    client: httpx.AsyncClient,
    timeframe: str,
    outputsize: int = 100
):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={SYMBOL}"
        f"&interval={timeframe}"
        f"&outputsize={outputsize}"
        f"&apikey={TWELVE_DATA_API_KEY}"
    )

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

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


# --- INDICATOR CALCULATIONS ---
def calculate_metrics_m1(df: pd.DataFrame):
    df = df.tail(100).copy()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )

    df["atr"] = df["tr"].rolling(window=14).mean()

    return df


def calculate_metrics_5m(df: pd.DataFrame):
    df = df.tail(100).copy()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )

    df["up_move"] = (
        df["high"] - df["high"].shift(1)
    )

    df["down_move"] = (
        df["low"].shift(1) - df["low"]
    )

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"])
        & (df["up_move"] > 0),
        df["up_move"],
        0.0
    )

    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"])
        & (df["down_move"] > 0),
        df["down_move"],
        0.0
    )

    tr14 = df["tr"].rolling(14).sum()

    plus_di = (
        100
        * (
            df["plus_dm"].rolling(14).sum()
            / (tr14 + 1e-10)
        )
    )

    minus_di = (
        100
        * (
            df["minus_dm"].rolling(14).sum()
            / (tr14 + 1e-10)
        )
    )

    dx = (
        100
        * (
            abs(plus_di - minus_di)
            / (plus_di + minus_di + 1e-10)
        )
    )

    df["adx"] = dx.rolling(14).mean()

    df["atr"] = df["tr"].rolling(window=14).mean()

    return df


TREND_15M_MIN_SEPARATION_PCT = 0.02


def compute_ema_trend(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 20
):
    if df is None or len(df) < slow + 1:
        return "NEUTRAL", 0.0

    ema_fast = df["close"].ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = df["close"].ewm(
        span=slow,
        adjust=False
    ).mean()

    last_fast = float(ema_fast.iloc[-1])
    last_slow = float(ema_slow.iloc[-1])

    if last_slow == 0:
        return "NEUTRAL", 0.0

    separation_pct = (
        abs(last_fast - last_slow)
        / last_slow
        * 100
    )

    if separation_pct < TREND_15M_MIN_SEPARATION_PCT:
        return "NEUTRAL", separation_pct

    return (
        "BULLISH" if last_fast > last_slow else "BEARISH",
        separation_pct
    )


# --- STRATEGIES ---
def detect_liquidity_sweep_structure(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame
):
    recent_high = float(
        df_5m[
            "high"
        ].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].max()
    )

    recent_low = float(
        df_5m[
            "low"
        ].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].min()
    )

    prev = df_1m.iloc[-2]
    curr = df_1m.iloc[-1]

    sweep_high = bool(
        prev["high"] > recent_high
        and curr["close"] < recent_high
    )

    sweep_low = bool(
        prev["low"] < recent_low
        and curr["close"] > recent_low
    )

    bearish_bos = bool(
        curr["close"] < prev["low"]
    )

    bullish_bos = bool(
        curr["close"] > prev["high"]
    )

    if sweep_high and bearish_bos:
        return (
            "SELL",
            "Liquidity Sweep + Bearish BOS",
            recent_high,
            recent_low
        )

    if sweep_low and bullish_bos:
        return (
            "BUY",
            "Liquidity Sweep + Bullish BOS",
            recent_high,
            recent_low
        )

    return (
        "HOLD",
        "No setup",
        recent_high,
        recent_low
    )


# ==========================================================
# IMPROVED MOMENTUM BREAKOUT DETECTION
# ==========================================================
def detect_breakout_continuation(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    allow_momentum_continuation: bool = False
):
    recent_high = float(
        df_5m[
            "high"
        ].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].max()
    )

    recent_low = float(
        df_5m[
            "low"
        ].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].min()
    )

    prev = df_1m.iloc[-2]
    curr = df_1m.iloc[-1]

    raw_atr = (
        df_1m["atr"].iloc[-1]
        if "atr" in df_1m.columns
        else np.nan
    )

    if pd.isna(raw_atr) or float(raw_atr) <= 0:
        return (
            "HOLD",
            "Breakout rejected: M1 ATR unavailable",
            recent_high,
            recent_low
        )

    atr_1m = float(raw_atr)

    curr_open = float(curr["open"])
    curr_high = float(curr["high"])
    curr_low = float(curr["low"])
    curr_close = float(curr["close"])

    prev_close = float(prev["close"])

    candle_range = curr_high - curr_low
    candle_body = abs(curr_close - curr_open)

    if candle_range > 0:
        close_location = (
            (curr_close - curr_low)
            / candle_range
        )
    else:
        close_location = 0.5

    bullish_breakout_structure = bool(
        prev_close > recent_high
        and curr_close > recent_high
        and curr_close > prev_close
    )

    bearish_breakout_structure = bool(
        prev_close < recent_low
        and curr_close < recent_low
        and curr_close < prev_close
    )

    if (
        not bullish_breakout_structure
        and not bearish_breakout_structure
    ):
        return (
            "HOLD",
            "No setup",
            recent_high,
            recent_low
        )

    if allow_momentum_continuation:
        min_body_atr = MOMENTUM_CONTINUATION_MIN_BODY_ATR_MULT
        min_close_location = MOMENTUM_CONTINUATION_MIN_CLOSE_LOCATION
        max_extension_atr = MOMENTUM_CONTINUATION_MAX_EXTENSION_ATR_MULT
    else:
        min_body_atr = BREAKOUT_MIN_BODY_ATR_MULT
        min_close_location = BREAKOUT_MIN_CLOSE_LOCATION
        max_extension_atr = BREAKOUT_MAX_EXTENSION_ATR_MULT

    if bullish_breakout_structure:

        body_direction_ok = (
            curr_close > curr_open
            if BREAKOUT_REQUIRE_BODY_DIRECTION
            else True
        )

        if not body_direction_ok:
            return (
                "HOLD",
                "Breakout rejected: bullish candle direction weak",
                recent_high,
                recent_low
            )

        body_ratio = candle_body / atr_1m
        body_strength_ok = (body_ratio >= min_body_atr)
        close_quality_ok = (close_location >= min_close_location)
        extension_from_level = (curr_close - recent_high)
        extension_atr = (extension_from_level / atr_1m)

        if not allow_momentum_continuation:
            if not body_strength_ok:
                return (
                    "HOLD",
                    "Breakout rejected: insufficient displacement",
                    recent_high,
                    recent_low
                )

            if not close_quality_ok:
                return (
                    "HOLD",
                    "Breakout rejected: weak candle close",
                    recent_high,
                    recent_low
                )

            if extension_from_level > (
                BREAKOUT_MAX_EXTENSION_ATR_MULT * atr_1m
            ):
                return (
                    "HOLD",
                    "Breakout overextended - continuation check required",
                    recent_high,
                    recent_low
                )

            return (
                "BUY",
                "Momentum Breakout Continuation (Bullish)",
                recent_high,
                recent_low
            )

        if extension_atr <= BREAKOUT_MAX_EXTENSION_ATR_MULT:
            return (
                "HOLD",
                "Continuation not active: extension <= normal breakout limit",
                recent_high,
                recent_low
            )

        if extension_atr > MOMENTUM_CONTINUATION_MAX_EXTENSION_ATR_MULT:
            return (
                "HOLD",
                "Continuation rejected: excessively extended",
                recent_high,
                recent_low
            )

        if not body_strength_ok:
            return (
                "HOLD",
                "Continuation rejected: momentum weakening",
                recent_high,
                recent_low
            )

        if not close_quality_ok:
            return (
                "HOLD",
                "Continuation rejected: weak close",
                recent_high,
                recent_low
            )

        return (
            "BUY",
            "Momentum Continuation - No Pullback (Bullish)",
            recent_high,
            recent_low
        )

    if bearish_breakout_structure:

        body_direction_ok = (
            curr_close < curr_open
            if BREAKOUT_REQUIRE_BODY_DIRECTION
            else True
        )

        if not body_direction_ok:
            return (
                "HOLD",
                "Breakout rejected: bearish candle direction weak",
                recent_high,
                recent_low
            )

        body_ratio = candle_body / atr_1m
        body_strength_ok = (body_ratio >= min_body_atr)
        close_quality_ok = (close_location <= (1.0 - min_close_location))
        extension_from_level = (recent_low - curr_close)
        extension_atr = (extension_from_level / atr_1m)

        if not allow_momentum_continuation:
            if not body_strength_ok:
                return (
                    "HOLD",
                    "Breakout rejected: insufficient displacement",
                    recent_high,
                    recent_low
                )

            if not close_quality_ok:
                return (
                    "HOLD",
                    "Breakout rejected: weak candle close",
                    recent_high,
                    recent_low
                )

            if extension_from_level > (
                BREAKOUT_MAX_EXTENSION_ATR_MULT * atr_1m
            ):
                return (
                    "HOLD",
                    "Breakout overextended - continuation check required",
                    recent_high,
                    recent_low
                )

            return (
                "SELL",
                "Momentum Breakout Continuation (Bearish)",
                recent_high,
                recent_low
            )

        if extension_atr <= BREAKOUT_MAX_EXTENSION_ATR_MULT:
            return (
                "HOLD",
                "Continuation not active: extension <= normal breakout limit",
                recent_high,
                recent_low
            )

        if extension_atr > MOMENTUM_CONTINUATION_MAX_EXTENSION_ATR_MULT:
            return (
                "HOLD",
                "Continuation rejected: excessively extended",
                recent_high,
                recent_low
            )

        if not body_strength_ok:
            return (
                "HOLD",
                "Continuation rejected: momentum weakening",
                recent_high,
                recent_low
            )

        if not close_quality_ok:
            return (
                "HOLD",
                "Continuation rejected: weak close",
                recent_high,
                recent_low
            )

        return (
            "SELL",
            "Momentum Continuation - No Pullback (Bearish)",
            recent_high,
            recent_low
        )

    return (
        "HOLD",
        "No setup",
        recent_high,
        recent_low
    )


CONSOLIDATION_LOOKBACK_5M = 8
CONSOLIDATION_MAX_RANGE_ATR_MULT = 1.5


def detect_consolidation_breakout(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame
):
    if (
        len(df_5m) < CONSOLIDATION_LOOKBACK_5M + 1
        or "atr" not in df_5m.columns
    ):
        return "HOLD", "No setup", 0.0, 0.0

    bracket = df_5m.iloc[
        -(CONSOLIDATION_LOOKBACK_5M + 1):-1
    ]

    bracket_high = float(bracket["high"].max())
    bracket_low = float(bracket["low"].min())

    bracket_range = (
        bracket_high - bracket_low
    )

    atr_5m = df_5m["atr"].iloc[-1]

    if pd.isna(atr_5m) or atr_5m <= 0:
        return (
            "HOLD",
            "No setup",
            bracket_high,
            bracket_low
        )

    is_consolidating = (
        bracket_range
        <= (
            CONSOLIDATION_MAX_RANGE_ATR_MULT
            * atr_5m
        )
    )

    if not is_consolidating:
        return (
            "HOLD",
            "No setup",
            bracket_high,
            bracket_low
        )

    curr = df_1m.iloc[-1]

    candle_body = abs(
        float(curr["close"])
        - float(curr["open"])
    )

    atr_1m = (
        df_1m["atr"].iloc[-1]
        if "atr" in df_1m.columns
        else None
    )

    strong_candle = (
        bool(
            candle_body
            >= BREAKOUT_MIN_BODY_ATR_MULT * atr_1m
        )
        if atr_1m and not pd.isna(atr_1m)
        else True
    )

    bullish_break = bool(
        curr["close"] > bracket_high
        and curr["close"] > curr["open"]
        and strong_candle
    )

    bearish_break = bool(
        curr["close"] < bracket_low
        and curr["close"] < curr["open"]
        and strong_candle
    )

    if bullish_break:
        return (
            "BUY",
            "Consolidation Bracket Breakout (Bullish)",
            bracket_high,
            bracket_low
        )

    if bearish_break:
        return (
            "SELL",
            "Consolidation Bracket Breakout (Bearish)",
            bracket_high,
            bracket_low
        )

    return (
        "HOLD",
        "No setup",
        bracket_high,
        bracket_low
    )


# --- FORWARD-TEST ANALYTICS HELPERS ---
def compute_trade_pips(trade: dict) -> tuple[float, float]:
    """
    Calculate the trade result using the bot's two-stage TP model.
    """
    action = str(trade.get("action") or "BUY").upper()

    entry = float(
        trade.get("entry_price")
        or trade.get("entry_p")
        or trade.get("price")
        or 0.0
    )

    sl = float(
        trade.get("sl_price")
        or trade.get("sl")
        or 0.0
    )

    tp1 = float(
        trade.get("tp1_price")
        or trade.get("tp1")
        or 0.0
    )

    tp2 = float(
        trade.get("tp2_price")
        or trade.get("tp2")
        or 0.0
    )

    exit_p = float(
        trade.get("exit_price")
        or entry
    )

    outcome = str(
        trade.get("outcome")
        or trade.get("outcome_val")
        or "PENDING"
    )

    sl_dist = abs(entry - sl) if sl > 0 else abs(entry - exit_p)
    if sl_dist == 0:
        sl_dist = 2.5

    tp1_dist = abs(tp1 - entry) if tp1 > 0 else sl_dist * 1.5
    tp2_dist = abs(tp2 - entry) if tp2 > 0 else sl_dist * 3.0

    if outcome == "CLOSED (TP1 HIT / SL BE)":
        total_pips = tp1_dist * 10.0
    elif outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        total_pips = (tp1_dist + tp2_dist) * 10.0
    elif outcome == "WIN (TP1 HIT)":
        total_pips = tp1_dist * 10.0
    elif "LOSS" in outcome:
        total_pips = -(sl_dist * 10.0 * 2.0)
    else:
        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
        total_pips = diff * 10.0 * 2.0

    profit_usd = total_pips * 0.10
    return total_pips, profit_usd


def compute_r_multiple(
    action: str,
    entry: float,
    exit_price: float,
    sl: float,
    tp1: float = 0.0,
    tp2: float = 0.0,
    outcome: str = "PENDING"
) -> float:
    """Calculate R using the same two-stage TP accounting as pips."""
    risk_dist = abs(entry - sl)

    if risk_dist <= 0:
        risk_dist = abs(entry - exit_price) if "LOSS" in outcome else 2.5
        if risk_dist == 0:
            risk_dist = 2.5

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

    single_r = (
        (exit_price - entry) / risk_dist
        if action == "BUY"
        else (entry - exit_price) / risk_dist
    )
    return single_r * 2.0


def bucket_adx(adx: float) -> str:
    if adx < 20:
        return "ADX <20"

    if adx < 30:
        return "ADX 20-30"

    if adx < 40:
        return "ADX 30-40"

    if adx < 50:
        return "ADX 40-50"

    return "ADX 50+"


def bucket_strategy(trigger_type: str) -> str:
    t = trigger_type or ""

    if "Converted" in t:
        return "No-Pullback Conversion"

    if "Consolidation" in t:
        return "Consolidation Bracket Breakout"

    if "Breakout" in t:
        return "Momentum Breakout"

    return "Sweep + BOS"


def bucket_session(timestamp_str: str) -> str:
    try:
        hour = int(
            str(timestamp_str)
            .split(" ")[1]
            .split(":")[0]
        )

    except Exception:
        return "Unknown"

    if 9 <= hour < 14:
        return "Early (09-14 WIB)"

    if 14 <= hour < 18:
        return "Mid (14-18 WIB)"

    if 18 <= hour < 22:
        return "Late (18-22 WIB)"

    return "Outside session"


def bucket_confluence(
    action: str,
    trend_15m: str
) -> str:
    t = (trend_15m or "").upper()

    if not t or t == "NEUTRAL":
        return "15m Neutral"

    if (
        (action == "BUY" and t == "BULLISH")
        or
        (action == "SELL" and t == "BEARISH")
    ):
        return "15m Aligned"

    return "15m Disagreed"


def format_performance_segment(
    dim_name: str,
    buckets: dict,
    min_sample_to_flag: int = 8
) -> str:
    lines = [f"*{dim_name}:*"]

    for label, r_values in sorted(
        buckets.items(),
        key=lambda item: -len(item[1])
    ):
        n = len(r_values)

        wins = sum(
            1 for r in r_values if r > 0
        )

        win_rate = (
            wins / n * 100
        ) if n else 0.0

        avg_r = (
            sum(r_values) / n
        ) if n else 0.0

        flag = ""

        if n >= min_sample_to_flag:
            if win_rate < 35:
                flag = " ⚠️ underperforming"

            elif win_rate > 65:
                flag = " ✅ strong"

        lines.append(
            f"• {label}: n={n}, "
            f"WR={win_rate:.0f}%, "
            f"AvgR={avg_r:+.2f}{flag}"
        )

    return "\n".join(lines)


def compute_pullback_zone(
    action: str,
    swing_high: float,
    swing_low: float
):
    rng = swing_high - swing_low

    if rng <= 0:
        return None, None

    if action == "BUY":
        zone_upper = (
            swing_high
            - (FIB_RETRACE_MIN * rng)
        )

        zone_lower = (
            swing_high
            - (FIB_RETRACE_MAX * rng)
        )

    else:
        zone_lower = (
            swing_low
            + (FIB_RETRACE_MIN * rng)
        )

        zone_upper = (
            swing_low
            + (FIB_RETRACE_MAX * rng)
        )

    return (
        float(zone_lower),
        float(zone_upper)
    )


def check_pullback_entry(
    action: str,
    zone_lower: float,
    zone_upper: float,
    curr_candle
) -> bool:

    candle_low = float(curr_candle["low"])
    candle_high = float(curr_candle["high"])
    candle_open = float(curr_candle["open"])
    candle_close = float(curr_candle["close"])

    overlaps_zone = (
        candle_high >= zone_lower
        and candle_low <= zone_upper
    )

    if not overlaps_zone:
        return False

    if action == "BUY":
        return candle_close > candle_open

    return candle_close < candle_open


# --- TELEGRAM NOTIFICATIONS ---
async def send_telegram_alert(
    client: httpx.AsyncClient,
    text: str,
    target_chat_id: str = None
):
    chat_id = "".join(
        str(
            target_chat_id or TELEGRAM_CHAT_ID
        ).split()
    )

    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logging.error(
            "[TELEGRAM ERROR] Missing token or chat_id"
        )
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        res = await client.post(
            url,
            json=payload
        )

        if res.status_code != 200:

            payload_plain = {
                "chat_id": chat_id,
                "text": text
            }

            res_plain = await client.post(
                url,
                json=payload_plain
            )

            if res_plain.status_code == 200:
                logging.info(
                    f"[TELEGRAM SENT] Delivered plain text "
                    f"to Chat ID {chat_id}"
                )

            else:
                logging.error(
                    f"[TELEGRAM ERROR] "
                    f"Failed sending message: {res_plain.text}"
                )

        else:
            logging.info(
                f"[TELEGRAM SENT] Delivered Markdown "
                f"to Chat ID {chat_id}"
            )

    except Exception as e:
        logging.error(
            f"[TELEGRAM EXCEPTION] {e}"
        )


# --- AI ANALYST EVALUATION ---
async def analyze_signal_with_ai(
    proposed_action: str,
    trigger_type: str,
    current_price: float,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    recent_high: float,
    recent_low: float,
    trend_15m: str = "NEUTRAL",
    adx_15m_true: float = 0.0
):
    is_breakout = "Breakout" in trigger_type

    if is_breakout:
        strategy_label = (
            "Breakout Continuation "
            "(trend-following momentum entry)"
        )

        veto_rules_text = (
            "- VETO if 5M ADX is below 20.0 "
            "(trend too weak to sustain a breakout; "
            "high risk of a false break/fakeout).\n"
            "- VETO if 5M ADX is above 70.0 "
            "(parabolic/overextended move, "
            "high risk of an imminent sharp pullback)."
        )

    else:
        strategy_label = (
            "Liquidity Sweep + Break-of-Structure "
            "(reversal entry)"
        )

        veto_rules_text = (
            "- VETO if 5M ADX is above 45.0 "
            "(indicating an extreme unstoppable runaway trend "
            "that destroys mean-reversion setups).\n"
            "- VETO if 5M ADX is below 12.0 "
            "(indicating dead market liquidity)."
        )

    confluence_text = (
        f"4. 15-Minute Trend: {trend_15m} "
        f"(EMA9/EMA20, ADX={adx_15m_true:.1f}). "
        f"This is a SOFT input, not a hard rule -- "
        f"weigh it as confluence, don't auto-veto on it alone."
    )

    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD) intraday scalping.

Strategy in play: {strategy_label}.

A technical scalp trigger ({trigger_type}) suggests a
{proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:

1. 1-Minute:
Close=${float(df_1m['close'].iloc[-1]):.2f},
Prior Candle High=${float(df_1m['high'].iloc[-2]):.2f},
Prior Candle Low=${float(df_1m['low'].iloc[-2]):.2f}.

2. 5-Minute Reference Levels:
Recent High=${recent_high:.2f},
Recent Low=${recent_low:.2f}.

3. 5-Minute:
ADX Trend Strength={float(df_5m['adx'].iloc[-1]):.1f}.

{confluence_text}

CRITICAL SCALP VETO RULES:

{veto_rules_text}

Respond strictly in valid JSON matching schema:

{{"action": "BUY" | "SELL" | "HOLD",
"confidence": 0.0-1.0,
"reasoning": "2 concise sentences explaining decision"}}
"""

    if GROQ_API_KEY:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={
                    "type": "json_object"
                }
            )

            data = json.loads(
                res.choices[0].message.content
            )

            return SignalOutput(**data)

        except Exception as e:
            logging.warning(
                f"[AI WARNING] Groq call failed: {e}. "
                f"Falling back to Gemini."
            )

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

            return SignalOutput.model_validate_json(
                res.text
            )

        except Exception as e:
            logging.error(
                f"[AI ERROR] Gemini call failed: {e}"
            )

    return SignalOutput(
        action=proposed_action,
        confidence=0.7,
        reasoning=(
            "Fallback: Executed on pure liquidity sweep "
            "+ structure break alignment."
        )
    )


# --- BACKGROUND SCANNING LOOP ---
async def background_scanning_loop():
    global SYSTEM_TRADING_ENABLED, CURRENT_SCAN_CYCLE_ID

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True
    ) as client:

        while True:
            try:
                if not SYSTEM_TRADING_ENABLED:
                    log_scan_event("SYSTEM_PAUSED", stage="SYSTEM", decision="SKIPPED", reason="Kill switch active")
                    logging.info(
                        "[PAUSED] System trading currently paused "
                        "via kill-switch. Skipping scan."
                    )

                    await asyncio.sleep(60)
                    continue

                now_wib = (
                    datetime.now(timezone.utc)
                    + timedelta(hours=7)
                )

                current_hour_wib = now_wib.hour

                CURRENT_SCAN_CYCLE_ID = (
                    f"{now_wib.strftime('%Y%m%d%H%M%S')}-"
                    f"{uuid.uuid4().hex[:8]}"
                )

                log_scan_event(
                    "SCAN_START",
                    stage="SCAN",
                    decision="STARTED",
                    reason="Scanner cycle started",
                    details={"wib_time": now_wib.strftime('%Y-%m-%d %H:%M:%S')}
                )

                active_session = (
                    ACTIVE_SESSION_START_HOUR <= current_hour_wib < ACTIVE_SESSION_END_HOUR
                )

                if not active_session:
                    log_scan_event(
                        "OUTSIDE_SESSION", stage="SESSION", decision="SKIPPED",
                        reason="Outside active trading session",
                        details={"wib_hour": current_hour_wib}
                    )
                    logging.info(
                        f"[SLEEP MODE] WIB Time: "
                        f"{now_wib.strftime('%H:%M')} | "
                        f"Outside active trading session. "
                        f"Pausing API calls..."
                    )

                    await asyncio.sleep(300)
                    continue

                logging.info(
                    f"[ACTIVE SCAN] WIB Time: "
                    f"{now_wib.strftime('%H:%M')} | "
                    f"Scanning Sweep/BOS + Breakout + "
                    f"No-Pullback Momentum Continuation..."
                )

                df_1m = await fetch_timeframe_data(
                    client,
                    "1min"
                )

                df_5m = await fetch_timeframe_data(
                    client,
                    "5min"
                )

                if df_1m is None or df_5m is None:
                    log_scan_event(
                        "DATA_FETCH_FAILED", stage="DATA", decision="RETRY",
                        reason="Failed to fetch M1/M5 candles"
                    )
                    logging.warning(
                        "Failed to fetch M1/M5 candles. "
                        "Retrying next cycle."
                    )

                    await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                    continue

                if len(df_1m) < 3 or len(df_5m) < 6:
                    log_scan_event(
                        "DATA_INSUFFICIENT", stage="DATA", decision="RETRY",
                        reason="Insufficient M1/M5 candle history",
                        details={"m1_len": len(df_1m), "m5_len": len(df_5m)}
                    )
                    logging.warning(
                        "Insufficient candle history for "
                        "sweep/BOS detection. Retrying next cycle."
                    )

                    await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                    continue

                df_1m = calculate_metrics_m1(df_1m)
                df_5m = calculate_metrics_5m(df_5m)

                curr_high = float(
                    df_1m["high"].tail(3).max()
                )

                curr_low = float(
                    df_1m["low"].tail(3).min()
                )

                update_open_trades(
                    curr_high,
                    curr_low
                )

                curr_price = float(
                    df_1m["close"].iloc[-1]
                )

                adx_5m = (
                    float(df_5m["adx"].iloc[-1])
                    if not pd.isna(
                        df_5m["adx"].iloc[-1]
                    )
                    else 0.0
                )

                global cached_15m

                need_refresh = (
                    cached_15m["df"] is None
                    or cached_15m["fetched_at"] is None
                    or (
                        datetime.now(timezone.utc)
                        - cached_15m["fetched_at"]
                        >= timedelta(
                            minutes=FIFTEEN_M_REFRESH_MINUTES
                        )
                    )
                )

                in_mid_session_veto = (
                    MID_SESSION_START_HOUR <= current_hour_wib < MID_SESSION_END_HOUR
                )

                if in_mid_session_veto:
                    if cached_15m["df"] is not None:
                        logging.info(
                            f"[MID-SESSION VETO] WIB {now_wib.strftime('%H:%M')} | "
                            "Scanning M1/M5 normally; 15M API refresh skipped and cached."
                        )
                    else:
                        logging.info(
                            f"[MID-SESSION VETO] WIB {now_wib.strftime('%H:%M')} | "
                            "Scanning M1/M5; no 15M cache available."
                        )
                elif need_refresh:
                    df_15m_raw = await fetch_timeframe_data(
                        client,
                        "15min"
                    )

                    if (
                        df_15m_raw is not None
                        and len(df_15m_raw) >= 21
                    ):
                        cached_15m["df"] = (
                            calculate_metrics_5m(
                                df_15m_raw
                            )
                        )

                        cached_15m["fetched_at"] = (
                            datetime.now(timezone.utc)
                        )

                    else:
                        logging.warning(
                            "[15M CONFLUENCE] Fetch failed "
                            "or insufficient candles -- "
                            "using stale/neutral cache this cycle."
                        )

                trend_15m, trend_15m_sep = (
                    compute_ema_trend(
                        cached_15m["df"]
                    )
                    if cached_15m["df"] is not None
                    else ("NEUTRAL", 0.0)
                )

                adx_15m_true = (
                    float(
                        cached_15m["df"]["adx"].iloc[-1]
                    )
                    if (
                        cached_15m["df"] is not None
                        and not pd.isna(
                            cached_15m["df"]["adx"].iloc[-1]
                        )
                    )
                    else 0.0
                )

                global pending_setup

                proposed_action = "HOLD"
                trigger_type = "None"

                recent_high = 0.0
                recent_low = 0.0
                status_note = ""

                # ==================================================
                # STAGE 2: Pullback confirmation check
                # ==================================================
                if pending_setup is not None:

                    if (
                        datetime.now(timezone.utc)
                        > pending_setup["expires"]
                    ):

                        expired_action = pending_setup["action"]
                        expired_trigger = pending_setup["trigger_type"]

                        logging.info(
                            f"[PULLBACK EXPIRED] "
                            f"{expired_action} setup "
                            f"from ${pending_setup['trigger_price']:.2f} "
                            f"expired without a pullback entry. "
                            f"Checking final momentum continuation."
                        )

                        (
                            b_action,
                            b_trigger_type,
                            b_recent_high,
                            b_recent_low
                        ) = logged_breakout_check(
                            df_1m,
                            df_5m,
                            allow_momentum_continuation=True
                        )

                        if (
                            b_action == expired_action
                            and adx_5m >= MOMENTUM_CONTINUATION_MIN_ADX
                        ):

                            proposed_action = b_action

                            trigger_type = (
                                f"No Pullback -> "
                                f"Momentum Continuation "
                                f"({b_trigger_type})"
                            )

                            recent_high = b_recent_high
                            recent_low = b_recent_low

                            log_scan_event(
                                "NO_PULLBACK_CONVERSION", stage="MOMENTUM_CONTINUATION",
                                action=b_action, trigger_type=trigger_type, price=curr_price,
                                recent_high=b_recent_high, recent_low=b_recent_low, adx_5m=adx_5m,
                                adx_15m=adx_15m_true, trend_15m=trend_15m, pending=pending_setup,
                                decision="ENTRY_CANDIDATE",
                                reason="Momentum continuation passed while pullback was unavailable",
                                details={"min_adx": MOMENTUM_CONTINUATION_MIN_ADX}
                            )

                            logging.info(
                                f"[NO-PULLBACK CONVERSION] "
                                f"Expired {expired_action} setup "
                                f"converted into momentum continuation. "
                                f"ADX={adx_5m:.1f}"
                            )

                        else:
                            log_scan_event(
                                "PULLBACK_EXPIRED", stage="PULLBACK", action=expired_action,
                                trigger_type=expired_trigger, price=curr_price,
                                recent_high=b_recent_high, recent_low=b_recent_low, adx_5m=adx_5m,
                                adx_15m=adx_15m_true, trend_15m=trend_15m, pending=pending_setup,
                                decision="DISCARDED",
                                reason="10-minute pullback expired with no valid continuation",
                                details={"final_action": b_action}
                            )

                            logging.info(
                                f"[PULLBACK EXPIRED] "
                                f"No valid momentum continuation. "
                                f"Setup discarded. "
                                f"Final action={b_action}, "
                                f"ADX={adx_5m:.1f}, "
                                f"Original={expired_action}."
                            )

                        pending_setup = None

                    else:
                        curr_candle = df_1m.iloc[-1]

                        confirmed = check_pullback_entry(
                            pending_setup["action"],
                            pending_setup["zone_lower"],
                            pending_setup["zone_upper"],
                            curr_candle
                        )

                        recent_high = pending_setup["recent_high"]
                        recent_low = pending_setup["recent_low"]

                        if confirmed:
                            proposed_action = pending_setup["action"]

                            trigger_type = (
                                pending_setup["trigger_type"]
                                + " + Pullback Confirmed"
                            )

                            log_scan_event(
                                "PULLBACK_CONFIRMED", stage="PULLBACK",
                                action=proposed_action, trigger_type=trigger_type, price=curr_price,
                                recent_high=recent_high, recent_low=recent_low, adx_5m=adx_5m,
                                adx_15m=adx_15m_true, trend_15m=trend_15m, pending=pending_setup,
                                decision="ENTRY_CANDIDATE",
                                reason="Pullback candle overlapped zone and confirmed direction"
                            )

                            logging.info(
                                f"[PULLBACK CONFIRMED] "
                                f"{proposed_action} setup entered "
                                f"inside ${pending_setup['zone_lower']:.2f}-"
                                f"${pending_setup['zone_upper']:.2f}."
                            )

                            pending_setup = None

                        else:
                            (
                                b_action,
                                b_trigger_type,
                                b_recent_high,
                                b_recent_low
                            ) = logged_breakout_check(
                                df_1m,
                                df_5m,
                                allow_momentum_continuation=True
                            )

                            if (
                                b_action != "HOLD"
                                and adx_5m >= MOMENTUM_CONTINUATION_MIN_ADX
                            ):

                                if b_action == pending_setup["action"]:
                                    proposed_action = b_action

                                    trigger_type = (
                                        f"No Pullback -> "
                                        f"Converted to "
                                        f"{b_trigger_type}"
                                    )

                                    recent_high = b_recent_high
                                    recent_low = b_recent_low

                                    log_scan_event(
                                        "NO_PULLBACK_CONVERSION", stage="MOMENTUM_CONTINUATION",
                                        action=b_action, trigger_type=trigger_type, price=curr_price,
                                        recent_high=b_recent_high, recent_low=b_recent_low, adx_5m=adx_5m,
                                        adx_15m=adx_15m_true, trend_15m=trend_15m, pending=pending_setup,
                                        decision="ENTRY_CANDIDATE",
                                        reason="Pending setup converted to controlled momentum continuation",
                                        details={"min_adx": MOMENTUM_CONTINUATION_MIN_ADX}
                                    )

                                    logging.info(
                                        f"[NO-PULLBACK CONVERSION] "
                                        f"{b_action} setup converted to "
                                        f"{b_trigger_type}. "
                                        f"ADX={adx_5m:.1f}"
                                    )

                                    pending_setup = None

                                else:
                                    invalidated_action = pending_setup["action"]
                                    invalidated_trigger = pending_setup["trigger_type"]

                                    logging.info(
                                        f"[SETUP INVALIDATED] "
                                        f"{invalidated_action} setup "
                                        f"({invalidated_trigger}) from "
                                        f"${pending_setup['trigger_price']:.2f} "
                                        f"invalidated by opposite "
                                        f"{b_action} breakout continuation."
                                    )

                                    proposed_action = b_action

                                    trigger_type = (
                                        f"{b_trigger_type} "
                                        f"(Invalidated "
                                        f"{invalidated_action} "
                                        f"Sweep Setup)"
                                    )

                                    recent_high = b_recent_high
                                    recent_low = b_recent_low

                                    pending_setup = None

                            else:
                                status_note = (
                                    f" | Awaiting pullback into "
                                    f"${pending_setup['zone_lower']:.2f}-"
                                    f"${pending_setup['zone_upper']:.2f}"
                                )

                # ==================================================
                # STAGE 1: Fresh Sweep + BOS Check
                # ==================================================
                if (
                    pending_setup is None
                    and proposed_action == "HOLD"
                    and not status_note
                ):

                    new_action, new_trigger_type, new_recent_high, new_recent_low = (
                        detect_liquidity_sweep_structure(
                            df_1m,
                            df_5m
                        )
                    )

                    recent_high = new_recent_high
                    recent_low = new_recent_low

                    if new_action != "HOLD":

                        prev_candle = df_1m.iloc[-2]
                        curr_candle = df_1m.iloc[-1]

                        if new_action == "BUY":
                            swing_low = float(
                                prev_candle["low"]
                            )

                            swing_high = float(
                                curr_candle["close"]
                            )

                        else:
                            swing_high = float(
                                prev_candle["high"]
                            )

                            swing_low = float(
                                curr_candle["close"]
                            )

                        zone_lower, zone_upper = (
                            compute_pullback_zone(
                                new_action,
                                swing_high,
                                swing_low
                            )
                        )

                        if zone_lower is not None:
                            pending_setup = {
                                "action": new_action,
                                "trigger_type": new_trigger_type,
                                "trigger_price": float(curr_price),
                                "recent_high": new_recent_high,
                                "recent_low": new_recent_low,
                                "zone_lower": zone_lower,
                                "zone_upper": zone_upper,
                                "expires": (
                                    datetime.now(timezone.utc)
                                    + timedelta(
                                        minutes=PULLBACK_EXPIRY_MINUTES
                                    )
                                ),
                            }

                            log_scan_event(
                                "PENDING_ARMED", stage="PULLBACK", action=new_action,
                                trigger_type=new_trigger_type, price=curr_price,
                                recent_high=new_recent_high, recent_low=new_recent_low,
                                adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
                                pending=pending_setup, decision="ARMED",
                                reason="Sweep + BOS setup armed for pullback",
                                details={"expiry_minutes": PULLBACK_EXPIRY_MINUTES}
                            )

                            status_note = (
                                f" | New {new_action} setup "
                                f"({new_trigger_type}) armed, "
                                f"awaiting pullback into "
                                f"${zone_lower:.2f}-"
                                f"${zone_upper:.2f}"
                            )

                # ==================================================
                # STAGE 1b: Consolidation bracket breakout
                # ==================================================
                if (
                    pending_setup is None
                    and proposed_action == "HOLD"
                    and not status_note
                ):

                    c_action, c_trigger_type, c_bracket_high, c_bracket_low = (
                        detect_consolidation_breakout(
                            df_1m,
                            df_5m
                        )
                    )

                    if c_action != "HOLD":

                        if adx_5m >= 20.0:
                            proposed_action = c_action
                            trigger_type = c_trigger_type

                            recent_high = (
                                c_bracket_high
                            )

                            recent_low = (
                                c_bracket_low
                            )

                        else:
                            status_note = (
                                f" | Bracket breakout seen "
                                f"but 5M ADX {adx_5m:.1f} "
                                f"too weak to confirm"
                            )

                # ==================================================
                # STAGE 1c: Improved Momentum breakout continuation
                # ==================================================
                if (
                    pending_setup is None
                    and proposed_action == "HOLD"
                    and not status_note
                ):

                    b_action, b_trigger_type, b_recent_high, b_recent_low = (
                        logged_breakout_check(
                            df_1m,
                            df_5m,
                            allow_momentum_continuation=False
                        )
                    )

                    if b_action != "HOLD":

                        if adx_5m >= 20.0:
                            proposed_action = b_action
                            trigger_type = b_trigger_type

                            recent_high = (
                                b_recent_high
                            )

                            recent_low = (
                                b_recent_low
                            )

                        else:
                            status_note = (
                                f" | Breakout seen but "
                                f"5M ADX {adx_5m:.1f} "
                                f"too weak to confirm continuation"
                            )

                # ==================================================
                # Stat-Veto Check
                # ==================================================
                if proposed_action != "HOLD":

                    stat_vetoed, stat_veto_reason = (
                        check_stat_veto(
                            adx_5m,
                            current_hour_wib
                        )
                    )

                    if stat_vetoed:

                        logging.info(
                            f"[STAT VETO] "
                            f"{stat_veto_reason}"
                        )

                        log_trade_signal(
                            "VETOED",
                            proposed_action,
                            f"{trigger_type} [STAT-VETO]",
                            curr_price,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            adx_5m,
                            0.0,
                            "None",
                            stat_veto_reason
                        )

                        log_scan_event(
                            "STAT_VETO", stage="STAT_VETO", action=proposed_action,
                            trigger_type=trigger_type, price=curr_price, recent_high=recent_high,
                            recent_low=recent_low, adx_5m=adx_5m, adx_15m=adx_15m_true,
                            trend_15m=trend_15m, decision="VETOED", reason=stat_veto_reason
                        )

                        proposed_action = "HOLD"

                # ==================================================
                # Loss-cooldown Check
                # ==================================================
                if proposed_action != "HOLD":

                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()

                        cursor.execute("""
                            SELECT outcome_timestamp
                            FROM signals
                            WHERE status = 'EXECUTED'
                              AND outcome = 'LOSS (SL HIT)'
                              AND outcome_timestamp IS NOT NULL
                              AND outcome_timestamp != ''
                            ORDER BY id DESC
                            LIMIT 1
                        """)

                        last_loss = cursor.fetchone()

                        cursor.close()
                        conn.close()

                        if (
                            last_loss
                            and last_loss.get(
                                "outcome_timestamp"
                            )
                        ):

                            ts_str = str(
                                last_loss[
                                    "outcome_timestamp"
                                ]
                            ).replace(
                                " WIB",
                                ""
                            )

                            last_loss_time = (
                                datetime.strptime(
                                    ts_str,
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            )

                            minutes_since_loss = (
                                now_wib.replace(
                                    tzinfo=None
                                )
                                - last_loss_time
                            ).total_seconds() / 60.0

                            if (
                                0 <= minutes_since_loss
                                < LOSS_COOLDOWN_MINUTES
                            ):

                                log_scan_event(
                                    "LOSS_COOLDOWN", stage="RISK_FILTER", action=proposed_action,
                                    trigger_type=trigger_type, price=curr_price, adx_5m=adx_5m,
                                    adx_15m=adx_15m_true, trend_15m=trend_15m, decision="BLOCKED",
                                    reason=f"{minutes_since_loss:.1f} minutes since last SL hit",
                                    details={"cooldown_minutes": LOSS_COOLDOWN_MINUTES}
                                )

                                logging.info(
                                    f"[LOSS COOLDOWN] "
                                    f"Skipping "
                                    f"{proposed_action}: "
                                    f"{minutes_since_loss:.1f} "
                                    f"min since last SL hit "
                                    f"(< {LOSS_COOLDOWN_MINUTES} "
                                    f"min cooldown)."
                                )

                                proposed_action = "HOLD"

                    except Exception as lc_err:
                        logging.error(
                            f"[LOSS COOLDOWN ERROR] "
                            f"{lc_err}"
                        )

                # ==================================================
                # Cooldown Distance Check
                # ==================================================
                if proposed_action != "HOLD":

                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()

                        cursor.execute("""
                            SELECT
                                COALESCE(
                                    entry_price,
                                    price,
                                    0
                                ) AS entry_p,
                                outcome
                            FROM signals
                            WHERE status = 'EXECUTED'
                              AND action = %s
                            ORDER BY id DESC
                            LIMIT 1
                        """, (
                            str(proposed_action),
                        ))

                        last_trade = cursor.fetchone()

                        cursor.close()
                        conn.close()

                        if last_trade:

                            last_entry_price = float(
                                last_trade["entry_p"]
                            )

                            last_outcome = (
                                str(
                                    last_trade["outcome"]
                                )
                                if last_trade.get("outcome")
                                else "PENDING"
                            )

                            required_distance = (
                                2.00
                                if last_outcome == "PENDING"
                                else 1.50
                            )

                            if (
                                abs(
                                    curr_price
                                    - last_entry_price
                                )
                                < required_distance
                            ):

                                log_scan_event(
                                    "DISTANCE_COOLDOWN", stage="RISK_FILTER", action=proposed_action,
                                    trigger_type=trigger_type, price=curr_price, adx_5m=adx_5m,
                                    adx_15m=adx_15m_true, trend_15m=trend_15m, decision="BLOCKED",
                                    reason=f"Price within ${required_distance:.2f} of previous trade",
                                    details={"last_entry_price": last_entry_price,
                                             "required_distance": required_distance,
                                             "last_outcome": last_outcome}
                                )

                                logging.info(
                                    f"[SCALP COOLDOWN] "
                                    f"Skipping "
                                    f"{proposed_action}: "
                                    f"Price within "
                                    f"${required_distance:.2f} "
                                    f"of previous trade at "
                                    f"${last_entry_price:.2f}."
                                )

                                proposed_action = "HOLD"

                    except Exception as cd_err:
                        logging.error(
                            f"[COOLDOWN ERROR] "
                            f"{cd_err}"
                        )

                # ==================================================
                # FINAL DECISION
                # ==================================================
                log_scan_event(
                    "SCAN_DECISION", stage="FINAL_DECISION", action=proposed_action,
                    trigger_type=trigger_type, price=curr_price, recent_high=recent_high,
                    recent_low=recent_low, adx_5m=adx_5m, adx_15m=adx_15m_true,
                    trend_15m=trend_15m, pending=pending_setup,
                    decision="HOLD" if proposed_action == "HOLD" else "AI_REVIEW",
                    reason=status_note.strip(" |") if status_note else (
                        trigger_type if proposed_action != "HOLD" else "No entry trigger"
                    ),
                    details={"trend_15m_separation_pct": trend_15m_sep}
                )

                if proposed_action == "HOLD":

                    logging.info(
                        f"[MARKET SCAN] "
                        f"Price: ${curr_price:.2f} | "
                        f"5M Swept High: ${recent_high:.2f} | "
                        f"5M Swept Low: ${recent_low:.2f} | "
                        f"Status: HOLD{status_note}"
                    )

                else:

                    logging.info(
                        f"[MARKET SCAN] Triggered "
                        f"{proposed_action} "
                        f"({trigger_type}) at "
                        f"${curr_price:.2f}. "
                        f"Running AI Analysis..."
                    )

                    ai_decision = (
                        await analyze_signal_with_ai(
                            proposed_action,
                            trigger_type,
                            curr_price,
                            df_1m,
                            df_5m,
                            recent_high,
                            recent_low,
                            trend_15m,
                            adx_15m_true
                        )
                    )

                    log_scan_event(
                        "AI_DECISION", stage="AI", action=proposed_action,
                        trigger_type=trigger_type, price=curr_price, recent_high=recent_high,
                        recent_low=recent_low, adx_5m=adx_5m, adx_15m=adx_15m_true,
                        trend_15m=trend_15m, decision=ai_decision.action,
                        reason=ai_decision.reasoning,
                        details={"confidence": float(ai_decision.confidence)}
                    )

                    raw_atr = (
                        df_1m["atr"].iloc[-1]
                    )

                    atr_1m = (
                        float(raw_atr)
                        if not pd.isna(raw_atr)
                        else 3.0
                    )

                    risk = max(
                        2.5,
                        atr_1m * 1.2
                    )

                    is_breakout_trigger = (
                        "Breakout" in trigger_type
                    )

                    tp1_r_mult = 1.5

                    tp2_r_mult = (
                        3.0
                        if is_breakout_trigger
                        else 2.5
                    )

                    if proposed_action == "BUY":

                        sl_price = (
                            float(curr_price - risk)
                        )

                        tp1_price = (
                            float(
                                curr_price
                                + risk * tp1_r_mult
                            )
                        )

                        tp2_price = (
                            float(
                                curr_price
                                + risk * tp2_r_mult
                            )
                        )

                    else:

                        sl_price = (
                            float(curr_price + risk)
                        )

                        tp1_price = (
                            float(
                                curr_price
                                - risk * tp1_r_mult
                            )
                        )

                        tp2_price = (
                            float(
                                curr_price
                                - risk * tp2_r_mult
                            )
                        )

                    if ai_decision.action == proposed_action:

                        new_id = log_trade_signal(
                            "EXECUTED",
                            proposed_action,
                            trigger_type,
                            curr_price,
                            sl_price,
                            tp1_price,
                            tp2_price,
                            float(
                                ai_decision.confidence
                            ),
                            adx_5m,
                            0.0,
                            "None",
                            ai_decision.reasoning,
                            trend_15m,
                            adx_15m_true
                        )

                        log_scan_event(
                            "SIGNAL_EXECUTED", stage="TRADE_SIGNAL", action=proposed_action,
                            trigger_type=trigger_type, price=curr_price, recent_high=recent_high,
                            recent_low=recent_low, adx_5m=adx_5m, adx_15m=adx_15m_true,
                            trend_15m=trend_15m, decision="EXECUTED", reason=ai_decision.reasoning,
                            details={"signal_id": new_id, "confidence": float(ai_decision.confidence),
                                     "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price,
                                     "risk": risk, "tp1_r": tp1_r_mult, "tp2_r": tp2_r_mult}
                        )

                        id_tag = (
                            f" #{new_id}"
                            if new_id
                            else ""
                        )

                        header_icon = (
                            "🚀"
                            if is_breakout_trigger
                            else "⚡"
                        )

                        header_label = (
                            "BREAKOUT CONTINUATION SIGNAL"
                            if is_breakout_trigger
                            else
                            "LIQUIDITY SWEEP + BOS SIGNAL"
                        )

                        model_label = (
                            "5M/1M Momentum Breakout Continuation"
                            if is_breakout_trigger
                            else
                            "5M Liquidity Sweep + "
                            "1M Break of Structure"
                        )

                        msg = (
                            f"{header_icon} "
                            f"*{header_label}{id_tag}*\n\n"

                            f"Asset: *XAUUSD (Gold Spot)*\n"
                            f"Action: *{proposed_action}*\n"
                            f"Type: *{trigger_type}*\n"
                            f"Entry Price: *${curr_price:.2f}*\n\n"

                            f"Stop Loss (SL): "
                            f"*${sl_price:.2f}*\n"

                            f"Take Profit 1 "
                            f"({tp1_r_mult:.1f}R): "
                            f"*${tp1_price:.2f}*\n"

                            f"Take Profit 2 "
                            f"({tp2_r_mult:.1f}R): "
                            f"*${tp2_price:.2f}*\n\n"

                            f"INDICATOR METRICS:\n"

                            f"- Model: {model_label}\n"

                            f"- Reference 5M High: "
                            f"${recent_high:.2f}\n"

                            f"- Reference 5M Low: "
                            f"${recent_low:.2f}\n"

                            f"- 1M ATR (risk unit): "
                            f"${atr_1m:.2f}\n"

                            f"- 5M ADX Trend Strength: "
                            f"{adx_5m:.1f}\n"

                            f"- 15M Confluence: "
                            f"{trend_15m} "
                            f"(ADX {adx_15m_true:.1f})\n\n"

                            f"Reasoning: "
                            f"{ai_decision.reasoning}"
                        )

                        await send_telegram_alert(
                            client,
                            msg
                        )

                    else:

                        log_trade_signal(
                            "VETOED",
                            proposed_action,
                            trigger_type,
                            curr_price,
                            sl_price,
                            tp1_price,
                            tp2_price,
                            float(
                                ai_decision.confidence
                            ),
                            adx_5m,
                            0.0,
                            "None",
                            ai_decision.reasoning,
                            trend_15m,
                            adx_15m_true
                        )

                        log_scan_event(
                            "AI_VETO", stage="AI", action=proposed_action,
                            trigger_type=trigger_type, price=curr_price, recent_high=recent_high,
                            recent_low=recent_low, adx_5m=adx_5m, adx_15m=adx_15m_true,
                            trend_15m=trend_15m, decision="VETOED", reason=ai_decision.reasoning,
                            details={"confidence": float(ai_decision.confidence)}
                        )

                log_scan_event(
                    "SCAN_END", stage="SCAN", action=proposed_action, trigger_type=trigger_type,
                    price=curr_price, recent_high=recent_high, recent_low=recent_low, adx_5m=adx_5m,
                    adx_15m=adx_15m_true, trend_15m=trend_15m, pending=pending_setup,
                    decision="COMPLETE", reason="Scanner cycle completed",
                    details={"status_note": status_note}
                )

                del df_1m, df_5m
                gc.collect()

            except Exception as e:
                log_scan_event(
                    "SCAN_ERROR", stage="SCAN", decision="ERROR", reason=str(e),
                    details={"exception_type": type(e).__name__}
                )
                logging.error(
                    f"[SCAN LOOP ERROR] {e}"
                )

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# --- FASTAPI LIFESPAN & AUTOMATED WEBHOOK SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()

    if TELEGRAM_BOT_TOKEN and APP_URL:

        try:
            webhook_endpoint = (
                f"{APP_URL.rstrip('/')}"
                f"/telegram-webhook"
            )

            set_url = (
                f"https://api.telegram.org/"
                f"bot{TELEGRAM_BOT_TOKEN}"
                f"/setWebhook?url={webhook_endpoint}"
            )

            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True
            ) as client:

                res = await client.get(
                    set_url
                )

                logging.info(
                    f"[AUTO WEBHOOK SETUP] "
                    f"Response: {res.text}"
                )

        except Exception as e:
            logging.error(
                f"[AUTO WEBHOOK SETUP ERROR] "
                f"Failed: {e}"
            )

    scan_task = asyncio.create_task(
        background_scanning_loop()
    )

    yield

    scan_task.cancel()


app = FastAPI(
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": (
            "Trading bot scanner and webhook "
            "server active."
        )
    }


# =====================================================================
# MT5 COPIER BRIDGE API ENDPOINT
# =====================================================================
@app.get("/get-latest-signal")
async def get_latest_signal():

    global SYSTEM_TRADING_ENABLED

    if not SYSTEM_TRADING_ENABLED:
        return {
            "signal": None,
            "trading_enabled": False,
            "status": "PAUSED"
        }

    if not DATABASE_URL:
        return {
            "signal": None,
            "error": "DATABASE_URL not set",
            "trading_enabled": SYSTEM_TRADING_ENABLED
        }

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                id,
                action,
                COALESCE(
                    entry_price,
                    price,
                    0
                ) AS entry_p,
                COALESCE(
                    sl_price,
                    sl,
                    0
                ) AS sl_p,
                COALESCE(
                    tp1_price,
                    tp1,
                    0
                ) AS tp1_p,
                COALESCE(
                    tp2_price,
                    tp2,
                    0
                ) AS tp2_p,
                COALESCE(
                    timestamp,
                    created_at::text,
                    ''
                ) AS log_time
            FROM signals
            WHERE status = 'EXECUTED'
            ORDER BY id DESC
            LIMIT 1;
        """)

        row = cur.fetchone()

        cur.close()
        conn.close()

        if row:
            return {
                "id": int(row["id"]),
                "action": str(
                    row["action"]
                ).upper(),
                "entry": float(
                    row["entry_p"]
                ),
                "sl": float(
                    row["sl_p"]
                ),
                "tp1": float(
                    row["tp1_p"]
                ),
                "tp2": float(
                    row["tp2_p"]
                ),
                "timestamp": str(
                    row["log_time"]
                ),
                "trading_enabled": True
            }

        return {
            "signal": None,
            "trading_enabled": True
        }

    except Exception as e:

        logging.error(
            f"[MT5 BRIDGE ERROR /get-latest-signal] "
            f"{e}"
        )

        return {
            "error": str(e),
            "trading_enabled":
                SYSTEM_TRADING_ENABLED
        }


# --- WEBHOOK ENDPOINT FOR TELEGRAM COMMANDS ---
@app.post("/telegram-webhook")
async def telegram_webhook(
    request: Request
):

    global SYSTEM_TRADING_ENABLED

    try:
        data = await request.json()

        message = data.get(
            "message",
            {}
        )

        raw_text = (
            message
            .get("text", "")
            .strip()
            .lower()
        )

        sender_chat_id = str(
            message
            .get("chat", {})
            .get("id", "")
        )

        if not sender_chat_id or not raw_text:
            return {
                "status": "ignored"
            }

        logging.info(
            f"[WEBHOOK RECEIVED] "
            f"Chat ID: {sender_chat_id} | "
            f"Command: '{raw_text}'"
        )

        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True
        ) as client:

            if raw_text in [
                "/help",
                "/start"
            ]:

                reply = (
                    "🤖 *TRADING BOT COMMANDS:*\n\n"

                    "• `/stats` - Comprehensive "
                    "Win-Rate & Risk Analytics Dashboard\n"

                    "• `/pips` - Detailed Gross/Net "
                    "Pips & USD Profit Breakdown (0.01 Lot)\n"

                    "• `/logs` - Detailed View "
                    "of Last 10 Trades & Outcomes\n"

                    "• `/analyze` - Forward-Test "
                    "Breakdown by Strategy, ADX Regime & Session\n"

                    "• `/pause` - 🛑 *EMERGENCY KILL SWITCH* "
                    "(Stop Bot & MT5 auto-trade)\n"

                    "• `/resume` - 🟢 Re-enable "
                    "Auto-Trading Execution\n"

                    "• `/help` - Display Command Menu\n\n"

                    f"⚠️ Active stat-vetoes: "
                    f"ADX ≥ {STAT_VETO_ADX_THRESHOLD:.0f} "
                    f"and Mid session "
                    f"({STAT_VETO_MID_SESSION_START_HOUR}-"
                    f"{STAT_VETO_MID_SESSION_END_HOUR} WIB) "
                    f"are auto-skipped based on "
                    f"forward-test underperformance.\n"

                    f"⏱️ Loss cooldown: "
                    f"{LOSS_COOLDOWN_MINUTES} min "
                    f"after any SL hit "
                    f"(any direction) before a new signal can execute."
                )

                await send_telegram_alert(
                    client,
                    reply,
                    target_chat_id=sender_chat_id
                )

            elif raw_text == "/pause":

                SYSTEM_TRADING_ENABLED = False

                reply = (
                    "🛑 *EMERGENCY KILL SWITCH ACTIVATED*\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                    "• Market scanner loop has been "
                    "**PAUSED**.\n"

                    "• MT5 Signal Copier will "
                    "**IGNORE** all new signals.\n\n"

                    "👉 Send `/resume` to reactivate trading."
                )

                await send_telegram_alert(
                    client,
                    reply,
                    target_chat_id=sender_chat_id
                )

            elif raw_text == "/resume":

                SYSTEM_TRADING_ENABLED = True

                reply = (
                    "🟢 *AUTO-TRADING SYSTEM RESUMED*\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                    "• Scanner loop is now **ACTIVE**.\n"

                    "• MT5 bridge is listening "
                    "for live setups."
                )

                await send_telegram_alert(
                    client,
                    reply,
                    target_chat_id=sender_chat_id
                )

            elif raw_text == "/stats":

                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT COUNT(*) AS total
                        FROM signals
                        WHERE status = 'EXECUTED'
                    """)

                    total_executed = (
                        cur.fetchone()["total"]
                        or 0
                    )

                    cur.execute("""
                        SELECT COUNT(*) AS vetoes
                        FROM signals
                        WHERE status = 'VETOED'
                    """)

                    total_vetoes = (
                        cur.fetchone()["vetoes"]
                        or 0
                    )

                    cur.execute("""
                        SELECT COUNT(*) AS pending
                        FROM signals
                        WHERE status = 'EXECUTED'
                          AND outcome = 'PENDING'
                    """)

                    total_pending = (
                        cur.fetchone()["pending"]
                        or 0
                    )

                    cur.execute("""
                        SELECT COUNT(*) AS tp1_wins
                        FROM signals
                        WHERE outcome LIKE 'WIN (TP1%'
                           OR outcome LIKE 'CLOSED%'
                    """)

                    tp1_wins = (
                        cur.fetchone()["tp1_wins"]
                        or 0
                    )

                    cur.execute("""
                        SELECT COUNT(*) AS tp2_wins
                        FROM signals
                        WHERE outcome LIKE 'WIN (TP2%'
                    """)

                    tp2_wins = (
                        cur.fetchone()["tp2_wins"]
                        or 0
                    )

                    cur.execute("""
                        SELECT COUNT(*) AS losses
                        FROM signals
                        WHERE outcome LIKE 'LOSS%'
                    """)

                    losses = (
                        cur.fetchone()["losses"]
                        or 0
                    )

                    cur.execute("""
                        SELECT
                            action,
                            COALESCE(entry_price, price, 0) AS entry_p,
                            COALESCE(sl_price, sl, 0) AS sl_p,
                            COALESCE(tp1_price, tp1, 0) AS tp1_p,
                            COALESCE(tp2_price, tp2, 0) AS tp2_p,
                            exit_price,
                            COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals
                        WHERE status = 'EXECUTED'
                          AND exit_price IS NOT NULL
                    """)

                    closed_trades = cur.fetchall()

                    total_pips = 0.0
                    win_pips = 0.0
                    loss_pips = 0.0

                    total_wins_count = tp1_wins + tp2_wins

                    for t in closed_trades:
                        trade_pips, _trade_usd = compute_trade_pips({
                            "action": t["action"],
                            "entry_price": t["entry_p"],
                            "sl_price": t["sl_p"],
                            "tp1_price": t["tp1_p"],
                            "tp2_price": t["tp2_p"],
                            "exit_price": t["exit_price"],
                            "outcome": t["outcome_val"]
                        })

                        total_pips += trade_pips

                        if trade_pips > 0:
                            win_pips += trade_pips
                        elif trade_pips < 0:
                            loss_pips += abs(trade_pips)

                    win_rate = (
                        total_wins_count
                        / total_executed
                        * 100
                        if total_executed > 0
                        else 0.0
                    )

                    est_dollar = (
                        total_pips * 0.10
                    )

                    avg_win = (
                        win_pips
                        / total_wins_count
                        if total_wins_count > 0
                        else 0.0
                    )

                    avg_loss = (
                        loss_pips
                        / losses
                        if losses > 0
                        else 0.0
                    )

                    profit_factor = (
                        win_pips / loss_pips
                        if loss_pips > 0
                        else (
                            win_pips
                            if win_pips > 0
                            else 0.0
                        )
                    )

                    cur.close()
                    conn.close()

                    reply = (
                        f"📊 *PERFORMANCE ANALYTICS DASHBOARD*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                        f"💰 *NET PIPS & PROFIT:*\n"

                        f"• Net Pips: "
                        f"*{total_pips:+.1f} pips*\n"

                        f"• Est. Profit (0.01 Lot): "
                        f"*${est_dollar:+.2f}*\n\n"

                        f"📈 *WIN / LOSS BREAKDOWN:*\n"

                        f"• Total Executed: "
                        f"*{total_executed}*\n"

                        f"• Total Wins: "
                        f"*{total_wins_count} "
                        f"({win_rate:.1f}%)*\n"

                        f"  └─ Hit TP1 (BE Runner): "
                        f"*{tp1_wins}*\n"

                        f"  └─ Hit TP2 (Full Target): "
                        f"*{tp2_wins}*\n"

                        f"• Total Losses (SL Hit): "
                        f"*{losses}*\n"

                        f"• Active Pending: "
                        f"*{total_pending}*\n\n"

                        f"⚡ *SYSTEM & AI EFFICIENCY:*\n"

                        f"• Total Signals: "
                        f"*{total_executed + total_vetoes}*\n"

                        f"• AI Vetoed Signals: "
                        f"*{total_vetoes}*\n\n"

                        f"🎯 *RISK & TRADE METRICS:*\n"

                        f"• Avg Win: "
                        f"*+{avg_win:.1f} pips* | "

                        f"Avg Loss: "
                        f"*-{avg_loss:.1f} pips*\n"

                        f"• Profit Factor: "
                        f"*{profit_factor:.2f}*\n"

                        f"• Win Rate: "
                        f"*{win_rate:.1f}%*"
                    )

                    await send_telegram_alert(
                        client,
                        reply,
                        target_chat_id=sender_chat_id
                    )

                except Exception as db_err:

                    logging.error(
                        f"[WEBHOOK ERROR /stats] "
                        f"{db_err}"
                    )

                    await send_telegram_alert(
                        client,
                        f"⚠️ Error querying stats: "
                        f"{db_err}",
                        target_chat_id=sender_chat_id
                    )

            elif raw_text == "/pips":

                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT
                            action,
                            COALESCE(entry_price, price, 0) AS entry_p,
                            COALESCE(sl_price, sl, 0) AS sl_p,
                            COALESCE(tp1_price, tp1, 0) AS tp1_p,
                            COALESCE(tp2_price, tp2, 0) AS tp2_p,
                            exit_price,
                            COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals
                        WHERE status = 'EXECUTED'
                          AND exit_price IS NOT NULL
                    """)

                    trades = cur.fetchall()

                    total_pips = 0.0
                    gross_win_pips = 0.0
                    gross_loss_pips = 0.0

                    winning_trades_count = 0
                    losing_trades_count = 0

                    for t in trades:
                        pips, _profit_usd = compute_trade_pips({
                            "action": t["action"],
                            "entry_price": t["entry_p"],
                            "sl_price": t["sl_p"],
                            "tp1_price": t["tp1_p"],
                            "tp2_price": t["tp2_p"],
                            "exit_price": t["exit_price"],
                            "outcome": t["outcome_val"]
                        })

                        total_pips += pips

                        if pips > 0:
                            gross_win_pips += pips
                            winning_trades_count += 1
                        elif pips < 0:
                            gross_loss_pips += abs(pips)
                            losing_trades_count += 1

                    avg_win_pips = (
                        gross_win_pips
                        / winning_trades_count
                        if winning_trades_count > 0
                        else 0.0
                    )

                    avg_loss_pips = (
                        gross_loss_pips
                        / losing_trades_count
                        if losing_trades_count > 0
                        else 0.0
                    )

                    est_profit_usd = (
                        total_pips * 0.10
                    )

                    cur.close()
                    conn.close()

                    reply = (
                        f"💵 *DETAILED PIPS & EARNINGS REPORT*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                        f"📊 *SUMMARY:*\n"

                        f"• Total Net Pips: "
                        f"*{total_pips:+.1f} pips*\n"

                        f"• Net Profit (0.01 Lot): "
                        f"*${est_profit_usd:+.2f}*\n\n"

                        f"📈 *PIPS BREAKDOWN:*\n"

                        f"• Gross Gain: "
                        f"*+{gross_win_pips:.1f} pips*\n"

                        f"• Gross Loss: "
                        f"*-{gross_loss_pips:.1f} pips*\n\n"

                        f"🎯 *AVERAGE METRICS:*\n"

                        f"• Avg Win Trade: "
                        f"*+{avg_win_pips:.1f} pips*\n"

                        f"• Avg Loss Trade: "
                        f"*-{avg_loss_pips:.1f} pips*\n"

                        f"• Pip Efficiency Ratio: "
                        f"*{(
                            gross_win_pips
                            / (gross_loss_pips + 1e-5)
                        ):.2f}*\n"

                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                        f"💡 *Catatan:* "
                        f"Dihitung pada $0.10/pip "
                        f"(0.01 lot XAU/USD)."
                    )

                    await send_telegram_alert(
                        client,
                        reply,
                        target_chat_id=sender_chat_id
                    )

                except Exception as err:

                    logging.error(
                        f"[WEBHOOK ERROR /pips] "
                        f"{err}"
                    )

                    await send_telegram_alert(
                        client,
                        f"⚠️ Error calculating pips: "
                        f"{err}",
                        target_chat_id=sender_chat_id
                    )

            elif raw_text == "/logs":

                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT
                            id,
                            action,
                            COALESCE(entry_price, price, 0) AS entry_p,
                            COALESCE(sl_price, sl, 0) AS sl_p,
                            COALESCE(tp1_price, tp1, 0) AS tp1_p,
                            COALESCE(tp2_price, tp2, 0) AS tp2_p,
                            exit_price,
                            COALESCE(outcome, 'PENDING') AS outcome_val,
                            COALESCE(timestamp, created_at::text, 'N/A') AS log_time
                        FROM signals
                        WHERE status = 'EXECUTED'
                        ORDER BY id DESC
                        LIMIT 10
                    """)

                    logs = cur.fetchall()

                    cur.close()
                    conn.close()

                    if not logs:

                        reply = (
                            "📜 *LAST 10 TRADE LOGS:*\n\n"
                            "_Belum ada transaksi yang "
                            "tereksekusi di database._"
                        )

                    else:

                        reply = (
                            "📜 *LAST 10 DETAILED TRADE LOGS:*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        )

                        for l in logs:

                            trade_id = l["id"]
                            action = l["action"]

                            entry = float(
                                l["entry_p"]
                            )

                            exit_p = (
                                float(
                                    l["exit_price"]
                                )
                                if l.get("exit_price")
                                is not None
                                else None
                            )

                            outcome = l["outcome_val"]
                            date_str = str(
                                l["log_time"]
                            )

                            if exit_p is not None:

                                pips, profit_usd = compute_trade_pips({
                                    "action": action,
                                    "entry_price": entry,
                                    "sl_price": l["sl_p"],
                                    "tp1_price": l["tp1_p"],
                                    "tp2_price": l["tp2_p"],
                                    "exit_price": exit_p,
                                    "outcome": outcome
                                })

                                r_multiple = compute_r_multiple(
                                    action,
                                    entry,
                                    exit_p,
                                    float(l["sl_p"] or 0.0),
                                    float(l["tp1_p"] or 0.0),
                                    float(l["tp2_p"] or 0.0),
                                    outcome
                                )

                                pip_str = (
                                    f"*{pips:+.1f} pips* | "
                                    f"{r_multiple:+.2f}R | "
                                    f"${profit_usd:+.2f}"
                                )

                            else:
                                pip_str = (
                                    "*ACTIVE / IN PROGRESS*"
                                )

                            if (
                                "WIN" in outcome
                                or "CLOSED" in outcome
                            ):
                                icon = "🟢"

                            elif "LOSS" in outcome:
                                icon = "🔴"

                            else:
                                icon = "🟡"

                            reply += (
                                f"{icon} "
                                f"*ID #{trade_id} | "
                                f"{action} XAU/USD*\n"

                                f"• Entry: "
                                f"${entry:.2f} → Exit: "
                                f"*${(
                                    exit_p
                                    if exit_p
                                    else 0.0
                                ):.2f}*\n"

                                f"• Outcome: "
                                f"*{outcome}*\n"

                                f"• Result: "
                                f"{pip_str} | "
                                f"Time: {date_str}\n"

                                f"──────────────────────────\n"
                            )

                    await send_telegram_alert(
                        client,
                        reply,
                        target_chat_id=sender_chat_id
                    )

                except Exception as log_err:

                    logging.error(
                        f"[WEBHOOK ERROR /logs] "
                        f"{log_err}"
                    )

                    await send_telegram_alert(
                        client,
                        f"⚠️ Error querying logs: "
                        f"{log_err}",
                        target_chat_id=sender_chat_id
                    )

            elif raw_text == "/analyze":

                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT
                            action,
                            trigger_type,
                            COALESCE(entry_price, price, 0) AS entry_p,
                            COALESCE(sl_price, sl, 0) AS sl_p,
                            COALESCE(tp1_price, tp1, 0) AS tp1_p,
                            COALESCE(tp2_price, tp2, 0) AS tp2_p,
                            exit_price,
                            COALESCE(outcome, 'PENDING') AS outcome_val,
                            COALESCE(adx_15m, 0) AS adx_val,
                            trend_15m,
                            COALESCE(timestamp, created_at::text, '') AS ts
                        FROM signals
                        WHERE status = 'EXECUTED'
                          AND exit_price IS NOT NULL
                    """)

                    rows = cur.fetchall()

                    cur.close()
                    conn.close()

                    if not rows:

                        reply = (
                            "📐 *STRATEGY FORWARD-TEST ANALYSIS*\n\n"
                            "_Not enough closed trades yet "
                            "to analyze. Check back after "
                            "more signals complete._"
                        )

                    else:

                        segments = {
                            "Strategy": {},
                            "ADX Regime": {},
                            "Session": {},
                            "15m Confluence": {}
                        }

                        overall_r = []

                        for r in rows:

                            entry = float(
                                r["entry_p"]
                            )

                            exit_p = float(
                                r["exit_price"]
                            )

                            sl = float(r["sl_p"] or 0.0)
                            tp1 = float(r["tp1_p"] or 0.0)
                            tp2 = float(r["tp2_p"] or 0.0)
                            outcome = str(r["outcome_val"] or "PENDING")

                            action = r["action"]

                            adx_val = float(r["adx_val"])

                            trigger = r["trigger_type"] or ""

                            trend_15m_val = r.get("trend_15m")
                            ts = r["ts"] or ""

                            r_mult = compute_r_multiple(
                                action,
                                entry,
                                exit_p,
                                sl,
                                tp1,
                                tp2,
                                outcome
                            )

                            overall_r.append(
                                r_mult
                            )

                            segments[
                                "Strategy"
                            ].setdefault(
                                bucket_strategy(
                                    trigger
                                ),
                                []
                            ).append(
                                r_mult
                            )

                            segments[
                                "ADX Regime"
                            ].setdefault(
                                bucket_adx(
                                    adx_val
                                ),
                                []
                            ).append(
                                r_mult
                            )

                            segments[
                                "Session"
                            ].setdefault(
                                bucket_session(
                                    ts
                                ),
                                []
                            ).append(
                                r_mult
                            )

                            segments[
                                "15m Confluence"
                            ].setdefault(
                                bucket_confluence(
                                    action,
                                    trend_15m_val
                                ),
                                []
                            ).append(
                                r_mult
                            )

                        n_total = len(
                            overall_r
                        )

                        overall_wr = (
                            sum(
                                1
                                for x in overall_r
                                if x > 0
                            )
                            / n_total
                            * 100
                            if n_total
                            else 0.0
                        )

                        overall_avg_r = (
                            sum(overall_r)
                            / n_total
                            if n_total
                            else 0.0
                        )

                        reply_parts = [
                            "📐 *STRATEGY FORWARD-TEST ANALYSIS*",
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | "
                            f"Avg R: *{overall_avg_r:+.2f}*",
                            "",
                        ]

                        for dim in [
                            "Strategy",
                            "ADX Regime",
                            "Session",
                            "15m Confluence"
                        ]:

                            reply_parts.append(
                                format_performance_segment(
                                    dim,
                                    segments[dim]
                                )
                            )

                            reply_parts.append("")

                        reply_parts.append(
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        )

                        reply_parts.append(
                            "💡 Segments need n≥8 to be flagged "
                            "⚠️/✅ (smaller samples are shown "
                            "but noisy). This does not auto-adjust "
                            "the bot -- use it to decide whether "
                            "to retune the ADX veto thresholds, "
                            "fib pullback zone, or session windows "
                            "in code.\n\n"

                            f"🚫 Active stat-vetoes: "
                            f"ADX ≥ "
                            f"{STAT_VETO_ADX_THRESHOLD:.0f} "
                            f"and Mid session "
                            f"({STAT_VETO_MID_SESSION_START_HOUR}-"
                            f"{STAT_VETO_MID_SESSION_END_HOUR} WIB) "
                            f"are being auto-skipped -- re-check "
                            f"this report after ~30-40 more trades "
                            f"to confirm they're actually lifting AvgR.\n"

                            f"⏱️ Loss cooldown "
                            f"({LOSS_COOLDOWN_MINUTES} min, "
                            f"any direction) is also active -- "
                            f"added after spotting clustered "
                            f"same-session losing sequences "
                            f"in the raw trade list."
                        )

                        reply = "\n".join(
                            reply_parts
                        )

                    await send_telegram_alert(
                        client,
                        reply,
                        target_chat_id=sender_chat_id
                    )

                except Exception as an_err:

                    logging.error(
                        f"[WEBHOOK ERROR /analyze] "
                        f"{an_err}"
                    )

                    await send_telegram_alert(
                        client,
                        f"⚠️ Error running analysis: "
                        f"{an_err}",
                        target_chat_id=sender_chat_id
                    )

    except Exception as e:

        logging.error(
            f"[WEBHOOK ERROR] {e}"
        )

    return {
        "status": "ok"
    }
