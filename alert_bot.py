# FINAL MAIN TRADING BOT: EMA 5/9 CONTROL vs EMA 5/15 LIVE EXPERIMENT
# Shared market data, shared DB, isolated strategy state/results, separate Telegram alerts.
# A = EMA 5/9 PAPER control | B = EMA 5/15 LIVE | C = Range Breakout OCO PAPER.
# IMPORTANT: Only Strategy B is eligible for the MT5 live-signal bridge.
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
RAW_EXPERIMENTAL_BOT_TOKEN = os.getenv("EXPERIMENTAL_TELEGRAM_BOT_TOKEN", "").strip()
RAW_EXPERIMENTAL_CHAT_ID = os.getenv("EXPERIMENTAL_TELEGRAM_CHAT_ID", "").strip()
RAW_BREAKOUT_BOT_TOKEN = os.getenv("BREAKOUT_TELEGRAM_BOT_TOKEN", "").strip()
RAW_BREAKOUT_CHAT_ID = os.getenv("BREAKOUT_TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()
# Experimental Telegram credentials MUST be supplied through environment variables. Do not hardcode bot tokens.

CLEAN_BOT_TOKEN = "".join(RAW_BOT_TOKEN.split())
TELEGRAM_CHAT_ID = "".join(RAW_CHAT_ID.split())
EXPERIMENTAL_TELEGRAM_CHAT_ID = "".join(RAW_EXPERIMENTAL_CHAT_ID.split())
BREAKOUT_TELEGRAM_CHAT_ID = "".join(RAW_BREAKOUT_CHAT_ID.split())

CLEAN_EXPERIMENTAL_BOT_TOKEN = "".join(RAW_EXPERIMENTAL_BOT_TOKEN.split())
if CLEAN_EXPERIMENTAL_BOT_TOKEN.startswith("bot"):
    EXPERIMENTAL_TELEGRAM_BOT_TOKEN = CLEAN_EXPERIMENTAL_BOT_TOKEN[3:]
else:
    EXPERIMENTAL_TELEGRAM_BOT_TOKEN = CLEAN_EXPERIMENTAL_BOT_TOKEN

CLEAN_BREAKOUT_BOT_TOKEN = "".join(RAW_BREAKOUT_BOT_TOKEN.split())
if CLEAN_BREAKOUT_BOT_TOKEN.startswith("bot"):
    BREAKOUT_TELEGRAM_BOT_TOKEN = CLEAN_BREAKOUT_BOT_TOKEN[3:]
else:
    BREAKOUT_TELEGRAM_BOT_TOKEN = CLEAN_BREAKOUT_BOT_TOKEN

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

# --- 4H directional bias (seasonality-style regime filter) ---
# Backtest on real trade data found SELL signals losing broadly on both
# strategies while gold's higher-timeframe trend ran up -- BUY kept working,
# SELL didn't, across nearly every trigger type. Rather than hard-block one
# direction permanently, this reads the 4H EMA50 (same method requested) so
# the block flips automatically if/when the broader trend reverses. A 60-min
# refresh is once every 4H candle's worth of movement at most -- cheap
# relative to the 800/day TwelveData budget (5M costs ~288/day, 15M ~96/day
# on their own; this adds at most 24/day more).
cached_1h = {"df": None, "fetched_at": None}
ONE_H_REFRESH_MINUTES = 15   # 1H candles close 4x more often than 4H did -- refresh more often to catch it
ONE_H_EMA_PERIOD = 200
ONE_H_OUTPUTSIZE = 250       # need 200+ candles for the EMA itself, so pull well past the default 100
# Buffer before the BULLISH/BEARISH label is allowed to flip at all -- stops
# a bare EMA touch (pure noise) from flip-flopping the bias label back and
# forth. Separate from the ranging-regime threshold below.
ONE_H_BIAS_FLIP_BUFFER_PCT = 0.18
# Separate, wider band: how close price sits to the 1H EMA200 before we
# treat the market as "ranging near the trendline" -- shown in /status as
# informational context only. SL sizing is NOT adjusted by this (plain
# ATR-based risk applies to every trade, same as before the $6 cap ever
# existed) -- the 1H bias filter above is what actually guards against a
# losing streak on a sudden trend reversal, by blocking entries against the
# new direction rather than resizing the stop.
RANGING_REGIME_PCT_THRESHOLD = 0.30

# --- EMA EXECUTION SIGNAL (5M chart, fast settings for early impulse capture) ---
CONTROL_STRATEGY = "CONTROL_5_9"
EXPERIMENTAL_STRATEGY = "EXPERIMENTAL_5_15"
CONTROL_EXECUTION_MODE = "PAPER"
EXPERIMENTAL_EXECUTION_MODE = "LIVE"

BREAKOUT_STRATEGY = "RANGE_BREAKOUT_OCO"
BREAKOUT_EXECUTION_MODE = "PAPER"

# Safety: the MT5 bridge serves ONLY the designated live strategy.
LIVE_BRIDGE_STRATEGY = EXPERIMENTAL_STRATEGY
# A live signal must be recent enough to be actionable. This prevents an old
# signal from being executed after a server/EA outage or after pause/resume.
LIVE_SIGNAL_MAX_AGE_MINUTES = 7

# --- STRATEGY C: RANGE BREAKOUT + OCO PENDING ORDERS ---
# This strategy is intentionally PAPER ONLY in this build. It models two
# pending stop orders (BUY STOP above range / SELL STOP below range) and applies
# one-cancels-other behavior in the Python scanner. No live MT5 pending order is
# placed by Strategy C yet.
BREAKOUT_RANGE_LOOKBACK_5M = 10
# Multiple window sizes to scan, shortest first. A single fixed 10-candle
# window misses both shorter, sharper pauses and longer, slower ones -- price
# doesn't consolidate for the same duration every time. We now test each
# length and use the tightest/most recent one that qualifies, instead of
# only ever looking at one fixed-size box.
BREAKOUT_RANGE_LOOKBACKS = (6, 8, 10, 14, 20)
# Loosened from 0.80-2.00 ATR / ADX 20/25 -- those thresholds produced ZERO
# triggered signals across multiple days of live scanning. This strategy is
# PAPER-only, so the priority is volume for data-gathering (target: 10+
# triggered signals/day) over the tightest possible setup quality -- more
# signals means the /compare stats become trustworthy sooner rather than
# staying stuck at n=0 indefinitely.
BREAKOUT_MIN_WIDTH_ATR = 0.50
BREAKOUT_MAX_WIDTH_ATR = 3.00
BREAKOUT_MAX_5M_ADX = 28.0
BREAKOUT_MAX_15M_ADX = 32.0
# "Flag" consolidations (a brief pause mid-trend, right after an impulsive
# move, before continuation) form precisely while ADX is still elevated from
# that prior leg -- the quiet-market ADX gate above was rejecting exactly
# this pattern. If the range itself is unusually tight, allow it through as
# a distinct flag-type setup even when ADX doesn't clear the quiet-market gate.
BREAKOUT_FLAG_MAX_WIDTH_ATR = 0.75
BREAKOUT_BUFFER_ATR = 0.15
BREAKOUT_MIN_BUFFER_PRICE = 0.15
BREAKOUT_SL_BUFFER_ATR = 0.30
BREAKOUT_TP1_R = 1.50
BREAKOUT_TP2_R = 2.50
BREAKOUT_MAX_PENDING_MINUTES = 45  # was 20 -- more time for either stop to trigger before the setup expires unfired
BREAKOUT_FAKE_WICK_ATR = 0.10
BREAKOUT_ACTIVE_PENDING = None

EMA_TREND_FAST = 5
EMA_TREND_SLOW = 9
EXPERIMENTAL_EMA_FAST = 5
EXPERIMENTAL_EMA_SLOW = 15

# FIXED: the 15M confluence filter previously reused the same 5/9 EMA as the 5M
# execution signal, so it reacted almost as fast as the thing it was supposed to
# be filtering -- the confluence check added far less protection than intended.
# Decoupled to a slower, dedicated pair so it behaves like an actual higher-
# timeframe trend read.
TREND_15M_EMA_FAST = 9
TREND_15M_EMA_SLOW = 20
TREND_15M_MIN_SEPARATION_PCT = 0.02

# --- RANGE / CONSOLIDATION MODE ---
# Trend mode (EMA cross/touch/impulse) and range mode are mutually exclusive,
# selected purely by 5M ADX: >= RANGE_MODE_ADX_MAX runs the trend engine,
# < RANGE_MODE_ADX_MAX runs this fade-the-edges engine instead of going silent.
RANGE_MODE_ADX_MAX = 20.0
RANGE_LOOKBACK_5M = 10             # candles defining the current range bracket (~50 min on 5M)
RANGE_MAX_WIDTH_ATR_MULT = 2.0     # bracket must be no wider than this (in ATR) to count as a real range
RANGE_MIN_WIDTH_ATR_MULT = 0.8     # bracket must be at least this wide -- too tight isn't tradeable (spread/slippage)
RANGE_EDGE_ZONE_PCT = 0.20         # price must be within this fraction of the range width from an edge to fade it
RANGE_SL_BUFFER_ATR_MULT = 0.3     # stop placed this many ATR beyond the bracket edge being faded
RANGE_MODE_MAX_15M_ADX = 25.0      # skip the fade if the 15M chart itself shows a real trend (ADX >= this)

# --- TREND EXHAUSTION / CHOP GUARD v1 ---
# Purpose: keep the original high-frequency EMA engine intact, but detect when
# a previously strong trend is losing directional power and turning into chop.
# Unlike V10, these are NOT hard entry filters. The guard stays dormant during
# healthy trends and only becomes restrictive when several exhaustion signals
# agree.
EXHAUSTION_LOOKBACK = 12                 # 60 minutes of 5M candles
EXHAUSTION_CROSS_LOOKBACK = 6            # 30 minutes
EXHAUSTION_MIN_PEAK_ADX = 30.0            # only guard a trend that was meaningful
EXHAUSTION_ADX_DROP = 5.0                 # peak ADX -> current ADX
EXHAUSTION_DI_DROP = 8.0                  # peak directional DI gap -> current
EXHAUSTION_EMA_CONTRACTION = 0.40         # spread contracted >=40% from recent max
EXHAUSTION_MIN_SLOPE_ATR = 0.10           # weak current directional EMA movement
EXHAUSTION_SCORE_CAUTION = 2              # monitor, but continue trading
EXHAUSTION_SCORE_BLOCK_DIRECTION = 3      # block only the weakening direction
EXHAUSTION_SCORE_CHOP = 5                 # full temporary chop guard
EXHAUSTION_HARD_LOSS_LOCK = 3             # 3 same-direction SLs = hard directional lock
EXHAUSTION_RESET_LOOKBACK = 3             # fresh expansion window
EXHAUSTION_RESET_MIN_DI_GAP = 4.0
EXHAUSTION_RESET_PRICE_LOOKBACK = 6

# --- SAME-DIRECTION LOSS PROTECTION ---
# Two losses do NOT immediately shut the strategy down. They make the
# exhaustion guard more sensitive. Three consecutive SLs in one direction
# hard-lock that direction until a fresh expansion is confirmed.
CONSEC_LOSS_BREAKER_THRESHOLD = 2

# --- SCAN SCHEDULE / TWELVE DATA BUDGET ---
ACTIVE_SESSION_START_HOUR = 0
ACTIVE_SESSION_END_HOUR = 24

# V10.1: forex/gold is closed on weekends, but nothing above ever checked for
# that -- ACTIVE_SESSION_START/END = 0/24 covers every HOUR but not every DAY.
# Confirmed live: three "Range Fade - Bottom Rejection" signals fired Sat
# Aug 22 16:45-17:25 WIB on a dead weekend feed (price frozen at ~4608.27,
# moving <0.01 across 40 minutes) -- the AI reviewer correctly vetoed all
# three, but the scanner should never have evaluated them in the first
# place. In WIB (UTC+7), NY's Sun 17:00 EST reopen lands at ~05:00 WIB
# MONDAY -- so the whole calendar Saturday AND Sunday are closed in WIB,
# not just Saturday.
FOREX_MONDAY_OPEN_HOUR_WIB = 5   # approx NY Sunday 17:00 EST reopen, in WIB


def is_forex_market_open(now_wib: datetime) -> bool:
    weekday = now_wib.weekday()  # Monday=0 ... Sunday=6
    if weekday == 5:  # Saturday: closed all day in WIB
        return False
    if weekday == 6:  # Sunday: closed all day in WIB (reopen lands on Monday)
        return False
    if weekday == 0 and now_wib.hour < FOREX_MONDAY_OPEN_HOUR_WIB:
        return False  # Monday, before the weekend reopen has actually happened
    return True

TWELVE_DATA_DAILY_LIMIT = 800
TWELVE_DATA_SAFETY_MARGIN = 40 

_twelve_data_call_count = 0
_twelve_data_budget_date = None  
_twelve_data_calls_by_tf = {"5min": 0, "15min": 0, "1h": 0}


def _reset_budget_if_new_day(now_wib: datetime):
    global _twelve_data_call_count, _twelve_data_budget_date, _twelve_data_calls_by_tf
    today = now_wib.date()
    if _twelve_data_budget_date != today:
        if _twelve_data_budget_date is not None:
            logging.info(f"[TWELVE DATA BUDGET] New WIB day. Resetting counter.")
        _twelve_data_budget_date = today
        _twelve_data_call_count = 0
        _twelve_data_calls_by_tf = {"5min": 0, "15min": 0, "1h": 0}


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
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_15m_true REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_extension_atr REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_climax_ratio REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_pips REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_usd REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_r REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS regime TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_sep_atr_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_slope_atr_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_slope_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_cross_count_5m INTEGER;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS strategy TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS execution_mode TEXT DEFAULT 'PAPER';",
            "ALTER TABLE signals ALTER COLUMN execution_mode SET DEFAULT 'PAPER';",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pending_buy_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pending_sell_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS order_state TEXT;",
        ]

        for query in migrations:
            cursor.execute(query)

        cursor.execute("UPDATE signals SET strategy = %s, execution_mode = %s WHERE strategy IS NULL", (CONTROL_STRATEGY, CONTROL_EXECUTION_MODE))
        cursor.execute("UPDATE signals SET execution_mode = %s WHERE execution_mode IS NULL", (CONTROL_EXECUTION_MODE,))
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[NEON DATABASE] Full schema verified and auto-migrated.")

        backfill_dual_lot_accounting()
    except Exception as e:
        logging.error(f"[NEON DB ERROR] Failed to initialize database schema: {e}")


def backfill_dual_lot_accounting():
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, action, COALESCE(entry_price, price, 0) AS entry_p,
                   COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                   COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                   COALESCE(outcome, 'PENDING') AS outcome_val
            FROM signals
            WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND result_pips IS NULL
        """)
        rows = cursor.fetchall()
        if not rows:
            cursor.close(); conn.close()
            return

        updated = 0
        for r in rows:
            trade = {
                "action": r["action"], "entry_price": r["entry_p"], "sl_price": r["sl_p"],
                "tp1_price": r["tp1_p"], "tp2_price": r["tp2_p"], "exit_price": r["exit_price"],
                "outcome": r["outcome_val"],
            }
            pips, usd = compute_trade_pips(trade)
            r_mult = compute_r_multiple(
                r["action"], float(r["entry_p"]), float(r["exit_price"]), float(r["sl_p"]),
                float(r["tp1_p"]), float(r["tp2_p"]), r["outcome_val"]
            )
            cursor.execute(
                "UPDATE signals SET result_pips = %s, result_usd = %s, result_r = %s WHERE id = %s",
                (pips, usd, r_mult, r["id"])
            )
            updated += 1

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"[V10 BACKFILL] result_pips/result_usd/result_r populated for {updated} historical signal(s).")
    except Exception as e:
        logging.error(f"[V10 BACKFILL ERROR] {e}")


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
    trend_15m: str = None, adx_15m_true: float = None, entry_extension_atr: float = None,
    entry_climax_ratio: float = None, regime: str = None, regime_metrics: dict = None,
    strategy: str = CONTROL_STRATEGY, execution_mode: str = CONTROL_EXECUTION_MODE
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
        extension_val = float(entry_extension_atr) if entry_extension_atr is not None else None
        climax_val = float(entry_climax_ratio) if entry_climax_ratio is not None else None
        regime_val = str(regime) if regime is not None else None
        rm = regime_metrics or {}
        ema_sep_val = rm.get("ema_sep_atr")
        ema_slope_val = rm.get("ema_slope_atr")
        adx_slope_val = rm.get("adx_slope")
        cross_count_val = rm.get("cross_count")

        cursor.execute("""
            INSERT INTO signals (
                timestamp, status, action, trigger_type, price, entry_price, sl, sl_price,
                tp1, tp1_price, tp2, tp2_price, confidence, adx_15m, stoch_rsi_15m,
                divergence_type, reasoning, outcome, outcome_timestamp, trend_15m, adx_15m_true,
                entry_extension_atr, entry_climax_ratio, regime, ema_sep_atr_5m, ema_slope_atr_5m,
                adx_slope_5m, ema_cross_count_5m, strategy, execution_mode, created_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, NOW()
            )
            RETURNING id;
        """, (
            str(wib_time), str(status), str(action), str(trigger_type), price_val, price_val,
            sl_val, sl_val, tp1_val, tp1_val, tp2_val, tp2_val, conf_val, adx_val, stoch_val,
            str(divergence_type), str(reasoning), "PENDING", "", trend_15m_val, adx_15m_true_val,
            extension_val, climax_val, regime_val, ema_sep_val, ema_slope_val,
            adx_slope_val, cross_count_val, str(strategy), str(execution_mode)
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
                trade_for_calc = {
                    "action": action, "entry_price": entry_price, "sl_price": sl,
                    "tp1_price": tp1, "tp2_price": tp2, "exit_price": float(exit_price), "outcome": new_outcome
                }
                result_pips, result_usd = compute_trade_pips(trade_for_calc)
                result_r = compute_r_multiple(action, entry_price, float(exit_price), sl, tp1, tp2, new_outcome)

                cursor.execute("""
                    UPDATE signals
                    SET outcome = %s, exit_price = %s, outcome_timestamp = %s,
                        result_pips = %s, result_usd = %s, result_r = %s
                    WHERE id = %s
                """, (new_outcome, float(exit_price), wib_now, result_pips, result_usd, result_r, trade_id))
                conn.commit()

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
    
    df["ema_fast"] = df["close"].ewm(span=EMA_TREND_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_TREND_SLOW, adjust=False).mean()

    df["trend_ema_fast"] = df["close"].ewm(span=TREND_15M_EMA_FAST, adjust=False).mean()
    df["trend_ema_slow"] = df["close"].ewm(span=TREND_15M_EMA_SLOW, adjust=False).mean()

    return df


def compute_ema_trend(df: pd.DataFrame):
    if df is None or len(df) < TREND_15M_EMA_SLOW + 1: return "NEUTRAL", 0.0
    last_fast = float(df["trend_ema_fast"].iloc[-1])
    last_slow = float(df["trend_ema_slow"].iloc[-1])
    if last_slow == 0: return "NEUTRAL", 0.0
    separation_pct = abs(last_fast - last_slow) / last_slow * 100
    if separation_pct < TREND_15M_MIN_SEPARATION_PCT: return "NEUTRAL", separation_pct
    return "BULLISH" if last_fast > last_slow else "BEARISH", separation_pct


def compute_1h_directional_bias(df_1h: pd.DataFrame):
    if df_1h is None or len(df_1h) < ONE_H_EMA_PERIOD + 1:
        return "NEUTRAL", 0.0
    ema200 = df_1h["close"].ewm(span=ONE_H_EMA_PERIOD, adjust=False).mean()
    last_close = float(df_1h["close"].iloc[-1])
    last_ema = float(ema200.iloc[-1])
    if last_ema == 0:
        return "NEUTRAL", 0.0
    separation_pct = (last_close - last_ema) / last_ema * 100
    if abs(separation_pct) < ONE_H_BIAS_FLIP_BUFFER_PCT:
        return "NEUTRAL", separation_pct
    return ("BULLISH" if separation_pct > 0 else "BEARISH"), separation_pct


def compute_ranging_regime(df_1h: pd.DataFrame):
    if df_1h is None or len(df_1h) < ONE_H_EMA_PERIOD + 1:
        return False, 0.0
    ema200 = df_1h["close"].ewm(span=ONE_H_EMA_PERIOD, adjust=False).mean()
    last_close = float(df_1h["close"].iloc[-1])
    last_ema = float(ema200.iloc[-1])
    if last_ema == 0:
        return False, 0.0
    abs_sep_pct = abs(last_close - last_ema) / last_ema * 100
    return (abs_sep_pct <= RANGING_REGIME_PCT_THRESHOLD), abs_sep_pct


TOUCH_MIN_BODY_ATR_MULT = 0.15

def detect_ema_signal(df_5m: pd.DataFrame, trend_15m: str, ema_fast: int = EMA_TREND_FAST, ema_slow: int = EMA_TREND_SLOW):
    if len(df_5m) < 2: return "HOLD", "Insufficient data"
    
    curr = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    
    c_ema_fast = curr["ema_fast"]
    c_ema_slow = curr["ema_slow"]
    p_ema_fast = prev["ema_fast"]
    p_ema_slow = prev["ema_slow"]
    
    bullish_impulse = prev["close"] <= p_ema_slow and curr["close"] > c_ema_fast and curr["close"] > c_ema_slow and curr["close"] > curr["open"]
    bearish_impulse = prev["close"] >= p_ema_slow and curr["close"] < c_ema_fast and curr["close"] < c_ema_slow and curr["close"] < curr["open"]
    
    bullish_cross = p_ema_fast <= p_ema_slow and c_ema_fast > c_ema_slow
    bearish_cross = p_ema_fast >= p_ema_slow and c_ema_fast < c_ema_slow

    raw_atr = curr["atr"] if "atr" in curr and not pd.isna(curr["atr"]) else None
    atr_val = float(raw_atr) if raw_atr is not None and raw_atr > 0 else None
    candle_body = abs(float(curr["close"]) - float(curr["open"]))
    touch_body_ok = (atr_val is None) or (candle_body / atr_val >= TOUCH_MIN_BODY_ATR_MULT)

    touch_bullish = c_ema_fast > c_ema_slow and curr["low"] <= c_ema_fast and curr["close"] > c_ema_fast and touch_body_ok
    touch_bearish = c_ema_fast < c_ema_slow and curr["high"] >= c_ema_fast and curr["close"] < c_ema_fast and touch_body_ok
    
    if bullish_impulse and trend_15m == "BULLISH":
        return "BUY", "Aggressive Price Impulse (Bullish)"
    if bearish_impulse and trend_15m == "BEARISH":
        return "SELL", "Aggressive Price Impulse (Bearish)"
        
    if bullish_cross and trend_15m == "BULLISH":
        return "BUY", f"EMA {ema_fast}/{ema_slow} Bullish Cross"
    if bearish_cross and trend_15m == "BEARISH":
        return "SELL", f"EMA {ema_fast}/{ema_slow} Bearish Cross"
        
    if touch_bullish and trend_15m == "BULLISH":
        return "BUY", f"EMA {ema_fast} Line Touch (Bullish)"
    if touch_bearish and trend_15m == "BEARISH":
        return "SELL", f"EMA {ema_fast} Line Touch (Bearish)"
        
    return "HOLD", "No EMA Setup"


def detect_range_reversal(df_5m: pd.DataFrame, adx_15m_true: float):
    if len(df_5m) < RANGE_LOOKBACK_5M + 1:
        return "HOLD", "Insufficient data for range detection", None, None

    bracket = df_5m.iloc[-(RANGE_LOOKBACK_5M + 1):-1]
    bracket_high = float(bracket["high"].max())
    bracket_low = float(bracket["low"].min())
    width = bracket_high - bracket_low

    raw_atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if raw_atr is None or pd.isna(raw_atr) or float(raw_atr) <= 0:
        return "HOLD", "ATR unavailable", bracket_high, bracket_low
    atr_5m = float(raw_atr)

    if width > RANGE_MAX_WIDTH_ATR_MULT * atr_5m:
        return "HOLD", "Range too wide -- likely a drift, not consolidation", bracket_high, bracket_low
    if width < RANGE_MIN_WIDTH_ATR_MULT * atr_5m:
        return "HOLD", "Range too tight -- inside normal noise/spread", bracket_high, bracket_low

    if adx_15m_true >= RANGE_MODE_MAX_15M_ADX:
        return "HOLD", f"15M ADX {adx_15m_true:.1f} still shows a real trend -- skipping fade to avoid trading against it", bracket_high, bracket_low

    curr = df_5m.iloc[-1]
    curr_open = float(curr["open"]); curr_close = float(curr["close"])
    curr_high = float(curr["high"]); curr_low = float(curr["low"])
    edge_zone = width * RANGE_EDGE_ZONE_PCT

    near_bottom = curr_low <= bracket_low + edge_zone
    near_top = curr_high >= bracket_high - edge_zone
    bullish_rejection = curr_close > curr_open and curr_close > (bracket_low + edge_zone)
    bearish_rejection = curr_close < curr_open and curr_close < (bracket_high - edge_zone)

    if near_bottom and bullish_rejection and not near_top:
        return "BUY", "Range Fade - Bottom Rejection", bracket_high, bracket_low
    if near_top and bearish_rejection and not near_bottom:
        return "SELL", "Range Fade - Top Rejection", bracket_high, bracket_low

    return "HOLD", "No range edge rejection", bracket_high, bracket_low


def get_recent_signals_for_direction(action: str, limit: int = 3, strategy: str = CONTROL_STRATEGY):
    if not DATABASE_URL:
        return []
    try:
        conn = get_db_connection(); cursor = conn.cursor()
        cursor.execute("""
            SELECT id, outcome FROM signals
            WHERE status = 'EXECUTED' AND action = %s AND strategy = %s
              AND outcome IS NOT NULL AND outcome NOT IN ('PENDING', 'WIN (TP1 HIT)')
            ORDER BY id DESC LIMIT %s
        """, (str(action), str(strategy), int(limit)))
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return rows or []
    except Exception as e:
        logging.error(f"[EXHAUSTION DB ERROR] {e}")
        return []


def consecutive_loss_count(action: str, limit: int = 3, strategy: str = CONTROL_STRATEGY) -> int:
    recent = get_recent_signals_for_direction(action, limit, strategy)
    count = 0
    for row in recent:
        if str(row.get("outcome")) == "LOSS (SL HIT)":
            count += 1
        else:
            break
    return count


def _directional_di_gap(df_5m: pd.DataFrame, action: str, idx: int) -> float:
    plus = float(df_5m["plus_di"].iloc[idx]) if not pd.isna(df_5m["plus_di"].iloc[idx]) else 0.0
    minus = float(df_5m["minus_di"].iloc[idx]) if not pd.isna(df_5m["minus_di"].iloc[idx]) else 0.0
    return (plus - minus) if action == "BUY" else (minus - plus)


def trend_exhaustion_guard(action: str, df_5m: pd.DataFrame, strategy: str = CONTROL_STRATEGY):
    metrics = {"score": 0, "peak_adx": None, "adx_drop": 0.0, "peak_di_gap": None,
               "di_drop": 0.0, "ema_spread_atr": None, "spread_contraction": 0.0,
               "ema_slope_atr": 0.0, "cross_count": 0, "loss_count": 0,
               "status": "NORMAL", "reason": ""}

    if action not in ("BUY", "SELL") or len(df_5m) < EXHAUSTION_LOOKBACK + 3:
        return False, metrics

    atr_now = float(df_5m["atr"].iloc[-1]) if not pd.isna(df_5m["atr"].iloc[-1]) else 0.0
    if atr_now <= 0:
        return False, metrics

    adx = df_5m["adx"].astype(float)
    ema_fast = df_5m["ema_fast"].astype(float)
    ema_slow = df_5m["ema_slow"].astype(float)

    window = df_5m.iloc[-EXHAUSTION_LOOKBACK:]
    peak_adx = float(window["adx"].max())
    adx_now = float(adx.iloc[-1])
    adx_drop = max(0.0, peak_adx - adx_now)

    di_gaps = [max(0.0, _directional_di_gap(df_5m, action, i))
               for i in range(len(df_5m) - EXHAUSTION_LOOKBACK, len(df_5m))]
    peak_di_gap = max(di_gaps) if di_gaps else 0.0
    current_di_gap = di_gaps[-1] if di_gaps else 0.0
    di_drop = max(0.0, peak_di_gap - current_di_gap)

    spreads = [abs(float(ema_fast.iloc[i]) - float(ema_slow.iloc[i])) / atr_now
               for i in range(len(df_5m) - EXHAUSTION_LOOKBACK, len(df_5m))]
    peak_spread = max(spreads) if spreads else 0.0
    current_spread = spreads[-1] if spreads else 0.0
    contraction = ((peak_spread - current_spread) / peak_spread) if peak_spread > 0 else 0.0

    slope = (float(ema_fast.iloc[-1]) - float(ema_fast.iloc[-1 - EXHAUSTION_RESET_LOOKBACK])) / atr_now
    directional_slope = slope if action == "BUY" else -slope

    diff = ema_fast - ema_slow
    recent_diff = diff.iloc[-(EXHAUSTION_CROSS_LOOKBACK + 1):]
    cross_count = int((np.sign(recent_diff).diff().fillna(0) != 0).sum())

    score = 0
    reasons = []

    strong_trend_context = peak_adx >= EXHAUSTION_MIN_PEAK_ADX and peak_di_gap >= EXHAUSTION_RESET_MIN_DI_GAP
    if strong_trend_context and adx_drop >= EXHAUSTION_ADX_DROP:
        score += 1
        reasons.append(f"ADX peak {peak_adx:.1f}->now {adx_now:.1f}")
    if strong_trend_context and di_drop >= EXHAUSTION_DI_DROP:
        score += 1
        reasons.append(f"DI gap contracted {peak_di_gap:.1f}->{current_di_gap:.1f}")
    if strong_trend_context and contraction >= EXHAUSTION_EMA_CONTRACTION:
        score += 1
        reasons.append(f"EMA spread contracted {contraction*100:.0f}%")
    if strong_trend_context and directional_slope < EXHAUSTION_MIN_SLOPE_ATR:
        score += 1
        reasons.append(f"EMA directional slope weak {directional_slope:+.2f} ATR")

    if cross_count >= 3:
        score += 2
        reasons.append(f"{cross_count} EMA crosses")
    elif cross_count >= 2:
        score += 1
        reasons.append(f"{cross_count} EMA crosses")

    loss_count = consecutive_loss_count(action, EXHAUSTION_HARD_LOSS_LOCK, strategy)
    if loss_count >= EXHAUSTION_HARD_LOSS_LOCK:
        reasons.append(f"{loss_count} consecutive {action} SLs")

    block_direction = score >= EXHAUSTION_SCORE_BLOCK_DIRECTION
    full_chop = score >= EXHAUSTION_SCORE_CHOP

    if full_chop:
        status = "CHOP"
    elif block_direction:
        status = "EXHAUSTION"
    elif score >= EXHAUSTION_SCORE_CAUTION:
        status = "CAUTION"
    else:
        status = "NORMAL"

    metrics.update({
        "score": score, "peak_adx": round(peak_adx, 1), "adx_drop": round(adx_drop, 1),
        "peak_di_gap": round(peak_di_gap, 1), "di_gap": round(current_di_gap, 1),
        "di_drop": round(di_drop, 1), "ema_spread_atr": round(current_spread, 3),
        "spread_contraction": round(contraction, 3), "ema_slope_atr": round(directional_slope, 3),
        "cross_count": cross_count, "loss_count": loss_count, "status": status,
        "reason": "; ".join(reasons) if reasons else "No meaningful exhaustion evidence"
    })

    return block_direction or full_chop, metrics


def fresh_directional_expansion_confirmed(action: str, df_5m: pd.DataFrame, trend_15m: str) -> tuple[bool, str]:
    if df_5m is None or len(df_5m) < max(EXHAUSTION_RESET_PRICE_LOOKBACK + 2, 8):
        return False, "Insufficient 5M history"

    wanted_trend = "BULLISH" if action == "BUY" else "BEARISH"
    if trend_15m != wanted_trend:
        return False, f"15M trend {trend_15m}, need {wanted_trend}"

    atr = float(df_5m["atr"].iloc[-1]) if not pd.isna(df_5m["atr"].iloc[-1]) else 0.0
    if atr <= 0:
        return False, "ATR unavailable"

    ef = float(df_5m["ema_fast"].iloc[-1]); es = float(df_5m["ema_slow"].iloc[-1])
    ef_prev = float(df_5m["ema_fast"].iloc[-1-EXHAUSTION_RESET_LOOKBACK])
    es_prev = float(df_5m["ema_slow"].iloc[-1-EXHAUSTION_RESET_LOOKBACK])
    spread_now = abs(ef-es) / atr
    spread_prev = abs(ef_prev-es_prev) / atr
    spread_expanding = spread_now > spread_prev

    gap_now = _directional_di_gap(df_5m, action, -1)
    gap_prev = _directional_di_gap(df_5m, action, -1-EXHAUSTION_RESET_LOOKBACK)
    di_expanding = gap_now > gap_prev and gap_now >= EXHAUSTION_RESET_MIN_DI_GAP

    recent = df_5m.iloc[-EXHAUSTION_RESET_PRICE_LOOKBACK-1:-1]
    last_close = float(df_5m["close"].iloc[-1])
    if action == "BUY":
        price_break = last_close > float(recent["high"].max())
        ema_aligned = ef > es
    else:
        price_break = last_close < float(recent["low"].min())
        ema_aligned = ef < es

    if ema_aligned and spread_expanding and di_expanding and price_break:
        return True, f"Fresh {action} expansion: EMA spread expanding, DI gap expanding, recent price extreme broken"
    return False, f"Expansion incomplete (EMA {'ok' if ema_aligned else 'bad'}, spread {'up' if spread_expanding else 'flat/down'}, DI {'up' if di_expanding else 'flat/down'}, price {'break' if price_break else 'inside range'})"


def compute_entry_extension(df_5m: pd.DataFrame, action: str, lookback: int = RANGE_LOOKBACK_5M):
    if len(df_5m) < lookback + 1:
        return None, None

    bracket = df_5m.iloc[-(lookback + 1):-1]
    bracket_high = float(bracket["high"].max())
    bracket_low = float(bracket["low"].min())

    curr = df_5m.iloc[-1]
    curr_close = float(curr["close"])
    curr_high = float(curr["high"])
    curr_low = float(curr["low"])

    raw_atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if raw_atr is None or pd.isna(raw_atr) or float(raw_atr) <= 0:
        return None, None
    atr_5m = float(raw_atr)

    if action == "BUY":
        extension_atr = (curr_close - bracket_high) / atr_5m
    elif action == "SELL":
        extension_atr = (bracket_low - curr_close) / atr_5m
    else:
        extension_atr = None

    candle_range = curr_high - curr_low
    climax_ratio = candle_range / atr_5m

    return extension_atr, climax_ratio


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

    if "LOSS" in outcome:
        total_pips = -(sl_dist * 10.0) * 2.0
    elif outcome == "CLOSED (TP1 HIT / SL BE)":
        total_pips = tp1_dist * 10.0
    elif outcome == "WIN (TP1 HIT)":
        total_pips = tp1_dist * 10.0
    elif outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        total_pips = (tp1_dist + tp2_dist) * 10.0
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

    if "LOSS" in outcome:
        return -2.0
    if outcome == "CLOSED (TP1 HIT / SL BE)":
        return (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5) / risk_dist
    if outcome == "WIN (TP1 HIT)":
        return (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5) / risk_dist
    if outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        tp1_leg = (abs(tp1 - entry) if tp1 > 0 else risk_dist * 1.5)
        tp2_leg = (abs(tp2 - entry) if tp2 > 0 else risk_dist * 2.5)
        return (tp1_leg + tp2_leg) / risk_dist
    return 2.0 * ((exit_price - entry) / risk_dist if action == "BUY" else (entry - exit_price) / risk_dist)

def bucket_adx(adx: float) -> str:
    if adx < 20: return "ADX < 20 (Chop)"
    if adx < 25: return "ADX 20-25 (Weak Trend)"
    if adx < 35: return "ADX 25-35 (Solid Trend)"
    if adx < 45: return "ADX 35-45 (Strong Trend)"
    return "ADX 45+ (Overextended)"

def bucket_extension(extension_atr) -> str:
    if extension_atr is None: return "Extension N/A"
    e = float(extension_atr)
    if e < 0.5: return "Extension <0.5 ATR (Early)"
    if e < 1.0: return "Extension 0.5-1.0 ATR"
    if e < 1.5: return "Extension 1.0-1.5 ATR"
    if e < 2.0: return "Extension 1.5-2.0 ATR"
    return "Extension 2.0+ ATR (Chasing)"

def bucket_strategy(trigger_type: str) -> str:
    t = trigger_type or ""
    if "Range Fade" in t: return "Range Fade (Consolidation)"
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
            if win_rate < 35: flag = " ⚠️ underperforming"
            elif win_rate > 65: flag = " ✅ strong"
        lines.append(f"• {label}: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}")
    return "\n".join(lines)


# --- TELEGRAM NOTIFICATIONS ---
async def send_telegram_alert(client: httpx.AsyncClient, text: str, target_chat_id: str = None, target_bot_token: str = None):
    bot_token = target_bot_token or TELEGRAM_BOT_TOKEN
    default_chat = EXPERIMENTAL_TELEGRAM_CHAT_ID if target_bot_token == EXPERIMENTAL_TELEGRAM_BOT_TOKEN else TELEGRAM_CHAT_ID
    chat_id = "".join(str(target_chat_id or default_chat).split())
    if not bot_token or not chat_id: return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        res = await client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        if res.status_code != 200: await client.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e: logging.error(f"[TELEGRAM EXCEPTION] {e}")


def set_execution_ema_columns(df_5m: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df_5m = df_5m.copy()
    df_5m["ema_fast"] = df_5m["close"].ewm(span=fast, adjust=False).mean()
    df_5m["ema_slow"] = df_5m["close"].ewm(span=slow, adjust=False).mean()
    return df_5m


# =====================================================================
# STRATEGY C: RANGE BREAKOUT + SUPPORT / RESISTANCE (PAPER)
# =====================================================================
BREAKOUT_SR_LOOKBACK_5M = 72
BREAKOUT_SR_PIVOT_LEFT = 2
BREAKOUT_SR_PIVOT_RIGHT = 2
BREAKOUT_SR_CLUSTER_ATR = 0.25
BREAKOUT_SR_MIN_TOUCHES = 2
BREAKOUT_SR_MIN_STRENGTH = 1.0
BREAKOUT_SR_MAX_TP_ATR = 14.0
BREAKOUT_SR_MIN_ROOM_R = 0.75
BREAKOUT_SR_SL_BUFFER_ATR = 0.20
BREAKOUT_SR_TP_BUFFER_ATR = 0.10


def _sr_pivot_levels(df_5m: pd.DataFrame, atr: float):
    if atr <= 0 or len(df_5m) < 15:
        return [], []
    look = df_5m.iloc[-BREAKOUT_SR_LOOKBACK_5M:].copy()
    left = BREAKOUT_SR_PIVOT_LEFT; right = BREAKOUT_SR_PIVOT_RIGHT
    highs, lows = [], []
    h = look['high'].astype(float).to_numpy(); l = look['low'].astype(float).to_numpy()
    for i in range(left, len(look) - right):
        if h[i] >= max(h[i-left:i+right+1]): highs.append((h[i], i))
        if l[i] <= min(l[i-left:i+right+1]): lows.append((l[i], i))
    tol = max(0.05, BREAKOUT_SR_CLUSTER_ATR * atr)

    def cluster(points):
        zones = []
        for price, idx in sorted(points, key=lambda x: x[0]):
            hit = next((z for z in zones if abs(price-z['price']) <= tol), None)
            if hit is None:
                zones.append({'price': price, 'touches': 1, 'last_idx': idx, 'prices': [price]})
            else:
                hit['prices'].append(price); hit['touches'] += 1
                hit['last_idx'] = max(hit['last_idx'], idx)
                hit['price'] = sum(hit['prices']) / len(hit['prices'])
        n=len(look)
        for z in zones:
            recency = 1.0 if n-1-z['last_idx'] <= 18 else 0.0
            z['strength'] = float(z['touches']) + recency
        return zones
    return cluster(lows), cluster(highs)


def _sr_nearest(levels, price, direction, max_distance):
    candidates=[]
    for z in levels:
        d=z['price']-price if direction=='above' else price-z['price']
        if d>0 and d<=max_distance and z['strength']>=BREAKOUT_SR_MIN_STRENGTH:
            candidates.append((d,z))
    return min(candidates,key=lambda x:x[0])[1] if candidates else None


def _sr_best_below(levels, price):
    c=[z for z in levels if z['price']<price and z['strength']>=BREAKOUT_SR_MIN_STRENGTH]
    return max(c,key=lambda z:z['price']) if c else None


def _sr_best_above(levels, price):
    c=[z for z in levels if z['price']>price and z['strength']>=BREAKOUT_SR_MIN_STRENGTH]
    return min(c,key=lambda z:z['price']) if c else None


def _breakout_sr_plan(df_5m: pd.DataFrame, range_high: float, range_low: float, atr: float, action: str):
    supports,resistances=_sr_pivot_levels(df_5m,atr)
    buffer=max(BREAKOUT_MIN_BUFFER_PRICE,BREAKOUT_BUFFER_ATR*atr)
    entry=range_high+buffer if action=='BUY' else range_low-buffer
    max_dist=BREAKOUT_SR_MAX_TP_ATR*atr
    if action=='BUY':
        support=_sr_best_below(supports,entry)
        stop_anchor=min(range_low,support['price']) if support else range_low
        sl=stop_anchor-BREAKOUT_SR_SL_BUFFER_ATR*atr
        risk=max(entry-sl,0.01)
        target=_sr_nearest(resistances,entry,'above',max_dist)
        if target is not None and target['price'] <= range_high+0.15*atr: target=None
        if target is None: return None,{'reason':'NO_USABLE_RESISTANCE_ABOVE_BREAKOUT'}
        tp1=target['price']-BREAKOUT_SR_TP_BUFFER_ATR*atr
        room_r=(tp1-entry)/risk
        if room_r<BREAKOUT_SR_MIN_ROOM_R:
            return None,{'reason':f'NEXT_RESISTANCE_TOO_CLOSE_{room_r:.2f}R','sr':target['price']}
        upper=[z for z in resistances if z['price']>target['price'] and z['price']-entry<=max_dist and z['strength']>=BREAKOUT_SR_MIN_STRENGTH]
        second=min(upper,key=lambda z:z['price']) if upper else None
        tp2=second['price']-BREAKOUT_SR_TP_BUFFER_ATR*atr if second else tp1
        return (sl,tp1,tp2),{'support':stop_anchor,'resistance':target['price'],'resistance_strength':target['strength'],'room_r':room_r}
    resistance=_sr_best_above(resistances,entry)
    stop_anchor=max(range_high,resistance['price']) if resistance else range_high
    sl=stop_anchor+BREAKOUT_SR_SL_BUFFER_ATR*atr
    risk=max(sl-entry,0.01)
    target=_sr_nearest(supports,entry,'below',max_dist)
    if target is not None and target['price'] >= range_low-0.15*atr: target=None
    if target is None: return None,{'reason':'NO_USABLE_SUPPORT_BELOW_BREAKOUT'}
    tp1=target['price']+BREAKOUT_SR_TP_BUFFER_ATR*atr
    room_r=(entry-tp1)/risk
    if room_r<BREAKOUT_SR_MIN_ROOM_R:
        return None,{'reason':f'NEXT_SUPPORT_TOO_CLOSE_{room_r:.2f}R','sr':target['price']}
    lower=[z for z in supports if z['price']<target['price'] and entry-z['price']<=max_dist and z['strength']>=BREAKOUT_SR_MIN_STRENGTH]
    second=max(lower,key=lambda z:z['price']) if lower else None
    tp2=second['price']+BREAKOUT_SR_TP_BUFFER_ATR*atr if second else tp1
    return (sl,tp1,tp2),{'resistance':stop_anchor,'support':target['price'],'support_strength':target['strength'],'room_r':room_r}


def detect_range_breakout_setup(df_5m: pd.DataFrame, adx_15m_true: float):
    atr = float(df_5m['atr'].iloc[-1]) if not pd.isna(df_5m['atr'].iloc[-1]) else 0.0
    adx5 = float(df_5m['adx'].iloc[-1]) if not pd.isna(df_5m['adx'].iloc[-1]) else 0.0
    if atr <= 0 or len(df_5m) < max(BREAKOUT_RANGE_LOOKBACKS) + 1:
        return None

    quiet_market = adx5 < BREAKOUT_MAX_5M_ADX and adx_15m_true < BREAKOUT_MAX_15M_ADX
    curr = df_5m.iloc[-1]
    c_close = float(curr['close']); c_high = float(curr['high']); c_low = float(curr['low'])

    best = None
    rejections = []
    fallback_bracket = df_5m.iloc[-(BREAKOUT_RANGE_LOOKBACKS[0] + 1):-1]
    fallback_high = float(fallback_bracket['high'].max()); fallback_low = float(fallback_bracket['low'].min())

    for lookback in BREAKOUT_RANGE_LOOKBACKS:
        bracket = df_5m.iloc[-(lookback + 1):-1]
        high = float(bracket['high'].max()); low = float(bracket['low'].min())
        width = high - low; width_atr = width / atr

        if width_atr < BREAKOUT_MIN_WIDTH_ATR:
            rejections.append(f'{lookback}c too tight ({width_atr:.2f} ATR)'); continue
        if width_atr > BREAKOUT_MAX_WIDTH_ATR:
            rejections.append(f'{lookback}c too wide ({width_atr:.2f} ATR)'); continue

        if quiet_market:
            range_type = 'QUIET_CONSOLIDATION'
        elif width_atr <= BREAKOUT_FLAG_MAX_WIDTH_ATR:
            range_type = 'FLAG_CONTINUATION'
        else:
            rejections.append(f'{lookback}c ADX still elevated (5m {adx5:.1f}, 15m {adx_15m_true:.1f}) and range not tight enough for a flag ({width_atr:.2f} ATR > {BREAKOUT_FLAG_MAX_WIDTH_ATR:.2f})')
            continue

        if best is None or width_atr < best['width_atr']:
            best = {'lookback': lookback, 'high': high, 'low': low, 'width': width,
                    'width_atr': width_atr, 'range_type': range_type}

    base = {'valid': False, 'atr': atr, 'adx5': adx5, 'fake': None}
    if best is None:
        base['reason'] = '; '.join(rejections[-3:]) if rejections else 'No qualifying range in any scanned window'
        base.update({'high': fallback_high, 'low': fallback_low, 'width': fallback_high - fallback_low, 'lookback': BREAKOUT_RANGE_LOOKBACKS[0], 'range_type': 'NONE'})
        return base

    high, low = best['high'], best['low']
    base.update({'high': high, 'low': low, 'width': best['width'], 'lookback': best['lookback'], 'range_type': best['range_type']})

    fake_up = c_high > high and c_close <= high and (c_high - high) >= BREAKOUT_FAKE_WICK_ATR * atr
    fake_down = c_low < low and c_close >= low and (low - c_low) >= BREAKOUT_FAKE_WICK_ATR * atr
    if c_high >= high or c_low <= low:
        if not (fake_up or fake_down):
            base['reason'] = 'Range boundary already breached on current candle -- no late pending order'
            return base
    if fake_up and fake_down: base['fake'] = 'AMBIGUOUS_TWO_SIDED_BREAK'
    elif fake_up: base['fake'] = 'UPSIDE_FAKE_BREAKOUT'
    elif fake_down: base['fake'] = 'DOWNSIDE_FAKE_BREAKOUT'
    base.update({'valid': True, 'reason': f"Qualified {best['range_type'].lower().replace('_',' ')} ({best['lookback']}c, {best['width_atr']:.2f} ATR)"})
    return base


def _breakout_pending_from_db():
    if not DATABASE_URL: return None
    try:
        conn=get_db_connection(); cur=conn.cursor()
        cur.execute("""SELECT id, action, entry_price, sl_price, tp1_price, tp2_price, pending_buy_price, pending_sell_price, created_at FROM signals WHERE strategy=%s AND status='PENDING_ORDER' AND order_state='OCO_ACTIVE' ORDER BY id DESC LIMIT 1""",(BREAKOUT_STRATEGY,))
        row=cur.fetchone(); cur.close(); conn.close(); return row
    except Exception as e: logging.error(f'[BREAKOUT DB ERROR] pending load: {e}'); return None


def _set_breakout_pending_state(row_id:int,state:str,outcome:str):
    if not DATABASE_URL: return
    conn=None; cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); cur.execute("UPDATE signals SET status='CANCELLED', order_state=%s, outcome=%s, outcome_timestamp=%s WHERE id=%s",(state,outcome,(datetime.now(timezone.utc)+timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S WIB'),int(row_id))); conn.commit()
    except Exception as e: logging.error(f'[BREAKOUT DB ERROR] state update: {e}')
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception: pass


def create_breakout_pending_signal(df_5m:pd.DataFrame,range_high:float,range_low:float,atr:float,reasoning:str):
    buffer=max(BREAKOUT_MIN_BUFFER_PRICE,BREAKOUT_BUFFER_ATR*atr); buy_stop=range_high+buffer; sell_stop=range_low-buffer
    buy_plan,buy_sr=_breakout_sr_plan(df_5m,range_high,range_low,atr,'BUY'); sell_plan,sell_sr=_breakout_sr_plan(df_5m,range_high,range_low,atr,'SELL')
    buy_sl=buy_plan[0] if buy_plan else range_low-BREAKOUT_SL_BUFFER_ATR*atr; sell_sl=sell_plan[0] if sell_plan else range_high+BREAKOUT_SL_BUFFER_ATR*atr
    buy_tp1=buy_plan[1] if buy_plan else buy_stop+max(buy_stop-buy_sl,.01)*BREAKOUT_TP1_R; buy_tp2=buy_plan[2] if buy_plan else buy_tp1
    sell_tp1=sell_plan[1] if sell_plan else sell_stop-max(sell_sl-sell_stop,.01)*BREAKOUT_TP1_R; sell_tp2=sell_plan[2] if sell_plan else sell_tp1
    if buy_plan is None: buy_stop=0.0
    if sell_plan is None: sell_stop=0.0
    if buy_plan is None and sell_plan is None: return None,buy_stop,sell_stop,{'buy':buy_sr,'sell':sell_sr}
    if not DATABASE_URL: return None,buy_stop,sell_stop,{'buy':buy_sr,'sell':sell_sr}
    conn=None; cur=None
    try:
        conn=get_db_connection(); cur=conn.cursor(); cur.execute("""INSERT INTO signals (timestamp,status,action,trigger_type,price,entry_price,sl,sl_price,tp1,tp1_price,tp2,tp2_price,confidence,adx_15m,stoch_rsi_15m,divergence_type,reasoning,outcome,outcome_timestamp,trend_15m,adx_15m_true,regime,strategy,execution_mode,created_at,pending_buy_price,pending_sell_price,order_state) VALUES (%s,'PENDING_ORDER','OCO','BUY_STOP/SELL_STOP',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,'None',%s,'PENDING_ORDER','',%s,%s,%s,%s,%s,NOW(),%s,%s,'OCO_ACTIVE') RETURNING id""",((datetime.now(timezone.utc)+timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S WIB'),buy_stop,buy_stop,buy_sl,buy_sl,buy_tp1,buy_tp1,buy_tp2,buy_tp2,.80,0.0,reasoning,'NEUTRAL',0.0,'RANGE_BREAKOUT',BREAKOUT_STRATEGY,BREAKOUT_EXECUTION_MODE,buy_stop,sell_stop)); row=cur.fetchone(); sid=int(row['id']) if row else None; conn.commit(); return sid,buy_stop,sell_stop,{'buy':buy_sr,'sell':sell_sr}
    except Exception as e: logging.error(f'[BREAKOUT DB ERROR] create pending: {e}'); return None,buy_stop,sell_stop,{'buy':buy_sr,'sell':sell_sr}
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception: pass


async def evaluate_breakout_strategy(client:httpx.AsyncClient,market_df_5m:pd.DataFrame,adx_15m_true:float,now_wib:datetime):
    global BREAKOUT_ACTIVE_PENDING
    setup=detect_range_breakout_setup(market_df_5m,adx_15m_true); curr=market_df_5m.iloc[-1]; c_high=float(curr['high']); c_low=float(curr['low'])
    if BREAKOUT_ACTIVE_PENDING is None:
        dbrow=_breakout_pending_from_db()
        if dbrow: BREAKOUT_ACTIVE_PENDING=dict(dbrow)
    if BREAKOUT_ACTIVE_PENDING:
        p=BREAKOUT_ACTIVE_PENDING; buy_stop=float(p.get('pending_buy_price') or 0); sell_stop=float(p.get('pending_sell_price') or 0); created=p.get('created_at'); age_min=0.0
        if created:
            try: age_min=max(0.0,(datetime.now(timezone.utc).replace(tzinfo=None)-created).total_seconds()/60.0)
            except Exception: pass
        buy_hit=buy_stop>0 and c_high>=buy_stop; sell_hit=sell_stop>0 and c_low<=sell_stop
        if buy_hit and sell_hit:
            _set_breakout_pending_state(p['id'],'CANCELLED','CANCELLED_TWO_SIDED'); await send_telegram_alert(client,f'⚠️ *{BREAKOUT_STRATEGY}* #{p["id"]} cancelled — both stops touched in one 5M candle; order sequence is ambiguous.',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN); BREAKOUT_ACTIVE_PENDING=None; return
        if buy_hit or sell_hit:
            action='BUY' if buy_hit else 'SELL'; entry=buy_stop if buy_hit else sell_stop; atr=float(market_df_5m['atr'].iloc[-1]) if not pd.isna(market_df_5m['atr'].iloc[-1]) else 3.0
            range_high=buy_stop-max(BREAKOUT_MIN_BUFFER_PRICE,BREAKOUT_BUFFER_ATR*atr); range_low=sell_stop+max(BREAKOUT_MIN_BUFFER_PRICE,BREAKOUT_BUFFER_ATR*atr)
            plan,sr=_breakout_sr_plan(market_df_5m,range_high,range_low,atr,action)
            if plan is None:
                _set_breakout_pending_state(p['id'],'CANCELLED','CANCELLED_SR_GEOMETRY'); await send_telegram_alert(client,f'⚠️ *{BREAKOUT_STRATEGY}* #{p["id"]} cancelled — S/R geometry no longer provides sufficient room.',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN); BREAKOUT_ACTIVE_PENDING=None; return
            sl,tp1,tp2=plan
            if DATABASE_URL:
                conn=get_db_connection(); cur=conn.cursor(); cur.execute("UPDATE signals SET status='EXECUTED',action=%s,price=%s,entry_price=%s,sl=%s,sl_price=%s,tp1=%s,tp1_price=%s,tp2=%s,tp2_price=%s,outcome='PENDING',order_state='TRIGGERED' WHERE id=%s",(action,entry,entry,sl,sl,tp1,tp1,tp2,tp2,int(p['id']))); conn.commit(); cur.close(); conn.close()
            sr_label=f'Next R: ${sr.get("resistance",0):.2f}' if action=='BUY' else f'Next S: ${sr.get("support",0):.2f}'
            await send_telegram_alert(client,f'🚀 *{BREAKOUT_STRATEGY} — {action} TRIGGERED* #{p["id"]}\n\nEntry: *${entry:.2f}*\nSL: *${sl:.2f}*\nTP1: *${tp1:.2f}*\nTP2: *${tp2:.2f}*\n{sr_label}\nS/R room: *{sr.get("room_r",0):.2f}R*\n\nOCO: *{("SELL STOP cancelled" if action=="BUY" else "BUY STOP cancelled")}*',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN)
            BREAKOUT_ACTIVE_PENDING=None; return
        if age_min>=BREAKOUT_MAX_PENDING_MINUTES:
            _set_breakout_pending_state(p['id'],'EXPIRED','CANCELLED_EXPIRED'); await send_telegram_alert(client,f'⏱️ *{BREAKOUT_STRATEGY}* #{p["id"]} expired after {BREAKOUT_MAX_PENDING_MINUTES} minutes — both stops cancelled.',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN); BREAKOUT_ACTIVE_PENDING=None; return
        return
    if not setup or not setup.get('valid'): return
    if setup.get('fake'):
        await send_telegram_alert(client,f'🧨 *FAKE BREAKOUT DETECTED*\n{setup["fake"]}\nRange: ${setup["low"]:.2f} — ${setup["high"]:.2f}\nNo OCO orders armed.',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN); return
    sid,buy_stop,sell_stop,sr_info=create_breakout_pending_signal(market_df_5m,setup['high'],setup['low'],setup['atr'],f"Qualified {setup['range_type'].lower().replace('_',' ')} ({setup['lookback']}c) + structural Support/Resistance clearance; OCO stops armed only after S/R geometry passed.")
    if sid is None: return
    BREAKOUT_ACTIVE_PENDING={'id':sid,'pending_buy_price':buy_stop,'pending_sell_price':sell_stop,'created_at':datetime.now(timezone.utc).replace(tzinfo=None)}
    buy_desc='valid' if sr_info['buy'].get('resistance') else sr_info['buy'].get('reason','blocked'); sell_desc='valid' if sr_info['sell'].get('support') else sr_info['sell'].get('reason','blocked')
    await send_telegram_alert(client,f'📦 *{BREAKOUT_STRATEGY} — OCO ARMED #{sid}*\n\nRange: *${setup["low"]:.2f} — ${setup["high"]:.2f}* ({setup["width"]/setup["atr"]:.2f} ATR)\n🟢 BUY STOP: *${buy_stop:.2f}* | S/R: *{buy_desc}*\n🔴 SELL STOP: *${sell_stop:.2f}* | S/R: *{sell_desc}*\n\n5M ADX: *{setup["adx5"]:.1f}* | 15M ADX: *{adx_15m_true:.1f}*\nFake-breakout filter: *PASSED*\nS/R structural filter: *ACTIVE*\n⏱️ Expiry: *{BREAKOUT_MAX_PENDING_MINUTES} min*\n⚖️ First trigger cancels the opposite side.\n\n⚠️ PAPER ONLY',BREAKOUT_TELEGRAM_CHAT_ID,BREAKOUT_TELEGRAM_BOT_TOKEN)

# --- AI ANALYST EVALUATION ---
async def analyze_signal_with_ai(
    proposed_action: str, trigger_type: str, current_price: float, df_5m: pd.DataFrame, 
    trend_15m: str = "NEUTRAL", adx_15m_true: float = 0.0, strategy_mode: str = "TREND",
    range_high: float = None, range_low: float = None,
    ema_fast: int = EMA_TREND_FAST, ema_slow: int = EMA_TREND_SLOW
):
    adx_5m = float(df_5m['adx'].iloc[-1])
    plus_di_5m = float(df_5m['plus_di'].iloc[-1])
    minus_di_5m = float(df_5m['minus_di'].iloc[-1])
    di_text = f"+DI={plus_di_5m:.1f}, -DI={minus_di_5m:.1f}"
    
    c_ema_fast = float(df_5m["ema_fast"].iloc[-1])
    c_ema_slow = float(df_5m["ema_slow"].iloc[-1])

    if strategy_mode == "RANGE":
        strategy_desc = f"Range Fade / Consolidation (5M Execution, active only when ADX < {RANGE_MODE_ADX_MAX:.0f})"
        range_text = f"5. Range Bracket: High=${range_high:.2f}, Low=${range_low:.2f}" if range_high else ""
        veto_rules_text = (
            f"- VETO if 5M ADX >= {RANGE_MODE_ADX_MAX:.0f} (a real trend has resumed -- fading it is wrong).\n"
            f"- VETO if the entry isn't clearly near a range edge with a genuine rejection candle, not just noise.\n"
            f"- This is a mean-reversion fade, not a breakout -- do NOT expect trend-style follow-through."
        )
    else:
        strategy_desc = f"EMA {ema_fast}+{ema_slow} Trend Follower (5M Execution + 15M Confluence)"
        range_text = ""
        veto_rules_text = (
            f"- VETO if 5M ADX < {RANGE_MODE_ADX_MAX:.0f} (Choppy/Ranging market, EMA setups will fail).\n"
            f"- Ensure price action agrees with momentum."
        )

    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD) intraday scalping.
Strategy in play: {strategy_desc}.
Trigger ({trigger_type}): {proposed_action} at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 5M Close=${float(df_5m['close'].iloc[-1]):.2f}
2. 5M EMAs: EMA{ema_fast}=${c_ema_fast:.2f}, EMA{ema_slow}=${c_ema_slow:.2f}.
3. 5M ADX={adx_5m:.1f}; {di_text}.
4. 15M Trend Filter: {trend_15m}
{range_text}

CRITICAL SCALP VETO RULES:
{veto_rules_text}

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

    return SignalOutput(action="HOLD", confidence=0.0, reasoning="AI unavailable: fail-safe HOLD. No trade is authorized without AI review.")


# --- SIDE-BY-SIDE BACKGROUND SCANNING LOOP ---
async def evaluate_strategy_cycle(
    client: httpx.AsyncClient,
    market_df_5m: pd.DataFrame,
    trend_15m: str,
    adx_15m_true: float,
    now_wib: datetime,
    strategy: str,
    ema_fast: int,
    ema_slow: int,
    execution_mode: str,
    alert_bot_token: str,
    alert_chat_id: str,
    directional_bias: str = "NEUTRAL",
):
    df_5m = set_execution_ema_columns(market_df_5m, ema_fast, ema_slow)

    curr_price = float(df_5m["close"].iloc[-1])
    curr_ema_fast = float(df_5m["ema_fast"].iloc[-1])
    curr_ema_slow = float(df_5m["ema_slow"].iloc[-1])
    adx_5m = float(df_5m["adx"].iloc[-1]) if not pd.isna(df_5m["adx"].iloc[-1]) else 0.0

    range_high = range_low = None
    regime_metrics = {}

    if adx_5m >= RANGE_MODE_ADX_MAX:
        strategy_mode = "TREND"
        proposed_action, trigger_type = detect_ema_signal(df_5m, trend_15m, ema_fast, ema_slow)
    else:
        strategy_mode = "RANGE"
        proposed_action, trigger_type, range_high, range_low = detect_range_reversal(df_5m, adx_15m_true)

    if proposed_action == "HOLD":
        log_scan_event(
            "SCAN_IDLE", stage="SIGNAL_DETECTION", decision="HOLD", reason=trigger_type,
            price=curr_price, adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
            details={"strategy": strategy, "mode": strategy_mode, "ema_fast": curr_ema_fast, "ema_slow": curr_ema_slow}
        )
        return

    # Check 1H Directional Bias filter
    if directional_bias != "NEUTRAL" and proposed_action != directional_bias:
        reason = f"Blocked by 1H Directional Bias filter ({directional_bias} bias active)"
        log_scan_event(
            "SCAN_VETO", stage="BIAS_FILTER", action=proposed_action, trigger_type=trigger_type,
            price=curr_price, adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
            decision="VETOED", reason=reason, details={"strategy": strategy, "directional_bias": directional_bias}
        )
        return

    # Check Trend Exhaustion / Chop Guard
    if strategy_mode == "TREND":
        blocked, metrics = trend_exhaustion_guard(proposed_action, df_5m, strategy=strategy)
        regime_metrics = metrics
        if blocked:
            confirmed, exp_reason = fresh_directional_expansion_confirmed(proposed_action, df_5m, trend_15m)
            if not confirmed:
                reason = f"Blocked by Trend Exhaustion Guard ({metrics.get('status')} state, score {metrics.get('score')}): {metrics.get('reason')} | Expansion check: {exp_reason}"
                log_scan_event(
                    "SCAN_VETO", stage="EXHAUSTION_GUARD", action=proposed_action, trigger_type=trigger_type,
                    price=curr_price, adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
                    decision="VETOED", reason=reason, details={"strategy": strategy, "exhaustion": metrics}
                )
                return

    # Pass to AI Analyst for final approval
    ai_output = await analyze_signal_with_ai(
        proposed_action=proposed_action,
        trigger_type=trigger_type,
        current_price=curr_price,
        df_5m=df_5m,
        trend_15m=trend_15m,
        adx_15m_true=adx_15m_true,
        strategy_mode=strategy_mode,
        range_high=range_high,
        range_low=range_low,
        ema_fast=ema_fast,
        ema_slow=ema_slow
    )

    if ai_output.action == "HOLD":
        log_scan_event(
            "SCAN_VETO", stage="AI_REVIEW", action=proposed_action, trigger_type=trigger_type,
            price=curr_price, adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
            decision="VETOED", reason=f"AI Veto: {ai_output.reasoning}",
            details={"strategy": strategy, "ai_confidence": ai_output.confidence}
        )
        return

    # Compute Risk Parameters
    atr_val = float(df_5m["atr"].iloc[-1]) if "atr" in df_5m.columns and not pd.isna(df_5m["atr"].iloc[-1]) else 2.5
    if strategy_mode == "RANGE" and range_high is not None and range_low is not None:
        if proposed_action == "BUY":
            sl_price = range_low - (RANGE_SL_BUFFER_ATR_MULT * atr_val)
            tp1_price = curr_price + (abs(curr_price - sl_price) * 1.5)
            tp2_price = range_high
        else:
            sl_price = range_high + (RANGE_SL_BUFFER_ATR_MULT * atr_val)
            tp1_price = curr_price - (abs(sl_price - curr_price) * 1.5)
            tp2_price = range_low
    else:
        risk_dist = atr_val * 1.5
        if proposed_action == "BUY":
            sl_price = curr_price - risk_dist
            tp1_price = curr_price + (risk_dist * 1.5)
            tp2_price = curr_price + (risk_dist * 2.5)
        else:
            sl_price = curr_price + risk_dist
            tp1_price = curr_price - (risk_dist * 1.5)
            tp2_price = curr_price - (risk_dist * 2.5)

    ext_atr, climax_r = compute_entry_extension(df_5m, proposed_action)

    signal_id = log_trade_signal(
        status="EXECUTED",
        action=proposed_action,
        trigger_type=trigger_type,
        price=curr_price,
        sl=sl_price,
        tp1=tp1_price,
        tp2=tp2_price,
        confidence=ai_output.confidence,
        adx_15m=adx_5m,
        stoch_rsi_15m=0.0,
        divergence_type="None",
        reasoning=ai_output.reasoning,
        trend_15m=trend_15m,
        adx_15m_true=adx_15m_true,
        entry_extension_atr=ext_atr,
        entry_climax_ratio=climax_r,
        regime=strategy_mode,
        regime_metrics=regime_metrics,
        strategy=strategy,
        execution_mode=execution_mode
    )

    mode_tag = "🔴 LIVE EXECUTED" if execution_mode == "LIVE" else "⚠️ PAPER ONLY"
    alert_text = (
        f"🚨 *SIGNAL GENERATED [{strategy}]*\n\n"
        f"Action: *{proposed_action}*\n"
        f"Trigger: *{trigger_type}*\n"
        f"Entry: *${curr_price:.2f}*\n"
        f"SL: *${sl_price:.2f}*\n"
        f"TP1: *${tp1_price:.2f}*\n"
        f"TP2: *${tp2_price:.2f}*\n\n"
        f"Mode: *{strategy_mode}* | 15M Trend: *{trend_15m}*\n"
        f"AI Confidence: *{ai_output.confidence:.2f}*\n"
        f"Reasoning: {ai_output.reasoning}\n\n"
        f"Execution: *{mode_tag}*"
    )
    await send_telegram_alert(client, alert_text, alert_chat_id, alert_bot_token)
