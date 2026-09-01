# A/B FORWARD-TEST BOT: EMA 5/9 CONTROL vs EMA 5/15 EXPERIMENTAL
# BASE: alert_bot_exhaustion_guard_v1.py
# Shared market data, shared DB, isolated strategy state/results, separate Telegram alerts.
# EXPERIMENTAL_5_15 is currently the only MT5-live strategy; CONTROL_5_9 is PAPER only.
# (Swapped from the original CONTROL=LIVE/EXPERIMENTAL=PAPER config -- the
# get-latest-signal MT5 bridge follows whichever strategy has
# execution_mode='LIVE', so this stays correct automatically if swapped again.)
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

# STRATEGY C: Extreme-frequency M5 impulse/re-entry scalper
BREAKOUT_STRATEGY = "EXTREME_M5"
BREAKOUT_EXECUTION_MODE = "PAPER"

# A/B directional mode:
# BUY_ONLY is the safe/default replacement for the old dynamic 1H one-direction gate.
# /oneway_on  -> DYNAMIC (1H EMA200 decides which side is allowed)
# /oneway_off -> BUY_ONLY
# /both       -> BOTH (no one-direction gate)
ONE_DIRECTION_MODE = "BUY_ONLY"
ONE_DIRECTION_MODES = {"BUY_ONLY", "DYNAMIC", "BOTH"}

# Optional real-MT5 market-data ingress for Strategy C.
# The MT5 EA can POST fresh M5 bars/ticks to /mt5-market-data. C prefers this cache.
MT5_DATA_SECRET = os.getenv("MT5_DATA_SECRET", "").strip()
MT5_DATA_CACHE_TTL_SECONDS = 20
mt5_market_cache = {"df": None, "updated_at": None, "source": None}

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
# Three consecutive SLs in one direction hard-lock that direction until a
# fresh expansion is confirmed. (A previous "2 losses = more sensitive"
# constant existed here as a comment but was never actually wired into any
# decision logic -- removed rather than left as misleading documentation.
# EXHAUSTION_HARD_LOSS_LOCK above is the only threshold that's real.)

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
    # REMOVED (mid-session): was carried over from the old liquidity-sweep
    # strategy's forward-test and never validated for this EMA system.
    #
    # REMOVED (ADX < 20 chop veto): this veto is now structurally impossible to
    # trigger and would be dead code if left in. Trend mode and range mode are
    # selected by ADX BEFORE either detector even runs (see
    # background_scanning_loop) -- trend mode only ever calls this with
    # adx_5m >= RANGE_MODE_ADX_MAX, and range mode (which specifically WANTS
    # low ADX) never calls this at all. Kept as a shell for any future
    # stat-based veto that isn't already handled by mode selection.
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
            # V10: real dual-0.01-lot accounting, stored per-row (not just
            # computed on the fly in /stats etc). See compute_trade_pips().
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_pips REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_usd REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_r REAL;",
            # V10: regime/quality instrumentation at signal time, for the
            # TRANSITION classifier and future analysis.
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS regime TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_sep_atr_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_slope_atr_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx_slope_5m REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_cross_count_5m INTEGER;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS strategy TEXT;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS execution_mode TEXT DEFAULT 'LIVE';",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pending_buy_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pending_sell_price REAL;",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS order_state TEXT;",
        ]

        for query in migrations:
            cursor.execute(query)

        # Historical rows predate A/B tagging; preserve them as the existing 5/9 control.
        cursor.execute("UPDATE signals SET strategy = %s, execution_mode = %s WHERE strategy IS NULL", (CONTROL_STRATEGY, CONTROL_EXECUTION_MODE))
        cursor.execute("UPDATE signals SET execution_mode = %s WHERE execution_mode IS NULL", (CONTROL_EXECUTION_MODE,))
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("[DATABASE] Full schema verified and auto-migrated.")

        # V10: one-time (idempotent) backfill of result_pips/result_usd/result_r
        # for every historical closed trade that predates these columns. Safe
        # to run on every boot -- it only ever touches rows where result_pips
        # IS NULL, so already-backfilled rows are skipped and this stays cheap.
        backfill_dual_lot_accounting()
    except Exception as e:
        logging.error(f"[DATABASE ERROR] Failed to initialize database schema: {e}")


def backfill_dual_lot_accounting():
    """
    V10: fills result_pips / result_usd / result_r for any EXECUTED, closed
    (exit_price IS NOT NULL) signal that doesn't have them yet -- covers every
    trade logged before this migration, using the SAME compute_trade_pips /
    compute_r_multiple functions the live bot now uses, so historical and
    future numbers are computed identically. Idempotent: only ever updates
    rows where result_pips IS NULL, so re-running on every boot is cheap and
    harmless once the backfill has completed.
    """
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
    # NOTE: despite the name, callers pass adx_5m (the mode-gating value) into the
    # `adx_15m` parameter/column -- inherited from earlier versions. The genuine
    # 15M ADX lives in `adx_15m_true`. /analyze's "5M ADX Regime" bucket reads
    # this column and is labeled accordingly; don't rename the DB column without
    # a migration, but don't assume it holds a real 15M value either.
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

        logging.info(f"[DB LOGGED] Signal ID #{new_id} | Status: {status} | Action: {action} | Price: ${price_val:.2f}")
        return new_id

    except Exception as e:
        logging.error(f"[DATABASE ERROR] Failed to log signal: {e}")
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

                # V10: result_pips/result_usd/result_r are only "final" once the
                # trade is fully closed (LOSS, CLOSED (TP1 HIT / SL BE), or WIN
                # (TP2 HIT)) -- the interim "WIN (TP1 HIT)" state still has an
                # open runner leg, so its stored numbers are a running mark, not
                # yet a settled result. They get overwritten again once the
                # runner actually closes.
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
        logging.error(f"[DATABASE ERROR] Failed to update trade outcomes: {e}")


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


def compute_1h_directional_bias(df_1h: pd.DataFrame):
    """1H EMA200 regime read (swapped from 4H EMA50 per user's validated
    reasoning: 4H is too slow for a 5M-execution system, 15M is too noisy --
    1H is the balance point). Price sustainably above EMA200 -> BULLISH bias
    (blocks SELL for A/B this cycle), below -> BEARISH (blocks BUY).
    Buffer rule: price must clear ONE_H_BIAS_FLIP_BUFFER_PCT beyond the EMA
    before the label is allowed to flip at all -- stops a bare EMA touch
    from flip-flopping the bias back and forth intraday.
    """
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
    """Separate, wider check from the bias-flip buffer above: is price
    currently sitting close enough to the 1H EMA200 to call this a
    ranging/consolidating regime? Purely informational (surfaced in
    /status) -- does NOT adjust SL sizing. Independent of the
    BULLISH/BEARISH label -- a signal can carry a bias label and still be
    inside the ranging band.
    """
    if df_1h is None or len(df_1h) < ONE_H_EMA_PERIOD + 1:
        return False, 0.0
    ema200 = df_1h["close"].ewm(span=ONE_H_EMA_PERIOD, adjust=False).mean()
    last_close = float(df_1h["close"].iloc[-1])
    last_ema = float(ema200.iloc[-1])
    if last_ema == 0:
        return False, 0.0
    abs_sep_pct = abs(last_close - last_ema) / last_ema * 100
    return (abs_sep_pct <= RANGING_REGIME_PCT_THRESHOLD), abs_sep_pct


TOUCH_MIN_BODY_ATR_MULT = 0.15  # FIXED: touch signals previously had zero quality filter

def detect_ema_signal(df_5m: pd.DataFrame, trend_15m: str, ema_fast: int = EMA_TREND_FAST, ema_slow: int = EMA_TREND_SLOW):
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
        return "BUY", f"EMA {ema_fast}/{ema_slow} Bullish Cross"
    if bearish_cross and trend_15m == "BEARISH":
        return "SELL", f"EMA {ema_fast}/{ema_slow} Bearish Cross"
        
    if touch_bullish and trend_15m == "BULLISH":
        return "BUY", f"EMA {ema_fast} Line Touch (Bullish)"
    if touch_bearish and trend_15m == "BEARISH":
        return "SELL", f"EMA {ema_fast} Line Touch (Bearish)"
        
    return "HOLD", "No EMA Setup"


def detect_range_reversal(df_5m: pd.DataFrame, adx_15m_true: float):
    """
    Fade-the-edges consolidation strategy. Only ever called when 5M ADX is
    below RANGE_MODE_ADX_MAX (see background_scanning_loop) -- this is the
    counterpart to detect_ema_signal(), not a supplement to it. The two never
    run in the same cycle.
    """
    if len(df_5m) < RANGE_LOOKBACK_5M + 1:
        return "HOLD", "Insufficient data for range detection", None, None

    # Bracket is defined by the N candles BEFORE the current one, same pattern
    # as the old V7 consolidation-breakout detector -- but here we fade INSIDE
    # the bracket instead of trading a breakout beyond it.
    bracket = df_5m.iloc[-(RANGE_LOOKBACK_5M + 1):-1]
    bracket_high = float(bracket["high"].max())
    bracket_low = float(bracket["low"].min())
    width = bracket_high - bracket_low

    raw_atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if raw_atr is None or pd.isna(raw_atr) or float(raw_atr) <= 0:
        return "HOLD", "ATR unavailable", bracket_high, bracket_low
    atr_5m = float(raw_atr)

    # Reject anything that isn't a genuine tight range: too wide means this is
    # actually a slow drift/pullback, not consolidation; too tight means the
    # edges are inside normal noise/spread and not worth trading.
    if width > RANGE_MAX_WIDTH_ATR_MULT * atr_5m:
        return "HOLD", "Range too wide -- likely a drift, not consolidation", bracket_high, bracket_low
    if width < RANGE_MIN_WIDTH_ATR_MULT * atr_5m:
        return "HOLD", "Range too tight -- inside normal noise/spread", bracket_high, bracket_low

    # FIXED: this used to require compute_ema_trend() to read exactly NEUTRAL,
    # which needs the 15M 9/20 EMA pair within TREND_15M_MIN_SEPARATION_PCT
    # (0.02%, roughly $0.90 at $4490 gold) of each other -- a bar so tight it
    # was almost never met, silently blocking range mode nearly 100% of the
    # time (confirmed: 0 of 63 closed trades were Range Fade). Gated on 15M ADX
    # instead -- a properly calibrated "is there a real higher-timeframe trend"
    # check, using the same ADX language as the 5M gate rather than a brittle
    # EMA-separation threshold.
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
    """Return recent closed EXECUTED trades in one direction, newest first."""
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
    """
    Soft trend-health guard for the original EMA engine.

    Score components:
      +1 ADX dropped materially from a recent strong peak
      +1 directional DI advantage has deteriorated materially
      +1 EMA5/EMA9 spread contracted materially from its recent maximum
      +1 EMA5 directional slope is currently weak
      +1/+2 repeated EMA crosses indicate developing chop
      +2 three consecutive same-direction SLs (hard lock)

    The guard does NOT reject a signal for one weak metric. It only blocks a
    direction when multiple pieces of evidence agree.
    """
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

    # Only score ADX/DI deterioration if there really was a strong directional move.
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

    # Losses are handled by the explicit directional-lock path below so a
    # potential fresh expansion can actually release the lock.
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
    """Confirm that a genuinely new directional expansion is underway."""
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
    """
    Instrumentation only -- does NOT affect entry/veto decisions. Measures two
    things at the moment a signal fires, for both TREND and RANGE signals:

    1. extension_atr: how far price has already traveled beyond the edge of
       its own recent N-candle bracket (the same bracket concept range mode
       uses), in ATR units. A large value means the move is already well away
       from its last consolidation zone -- a proxy for "chasing a move that's
       already run" rather than catching it at the start.

    2. climax_ratio: the current candle's own range (high-low) relative to
       ATR. A candle several times the normal ATR is a classic "climax" shape
       often followed by a shakeout/retracement even when the larger move is
       intact -- this is the pattern behind stops getting tagged mid-impulse
       before the move continues.

    Logged to signals.entry_extension_atr / entry_climax_ratio so /analyze can
    bucket outcomes by these values once enough trades accumulate. Nothing
    here changes what fires or when -- purely for building the evidence base
    before any logic changes, per the "track first" approach.
    """
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


# --- FORWARD-TEST ANALYTICS HELPERS ---
# V10 (REPLACES partial-close model): your MT5 EA does not run a single
# 0.01 lot with a 50/50 partial close. It opens TWO separate 0.01-lot
# positions per signal -- lot 1 targets TP1 and closes there in full, lot 2
# ("the runner") either rides to TP2 or has its SL moved to breakeven once
# lot 1 hits TP1. If price never reaches TP1 at all, BOTH lots are still
# live and BOTH get stopped out at the original SL. Every dollar figure
# below now reflects that two-position reality:
#
#   Outcome                        | pips (2x0.01 lot)          | R
#   --------------------------------------------------------------------
#   LOSS (SL HIT, before TP1)      | -2 x sl_dist                | -2.0
#   CLOSED (TP1 HIT / SL BE)       | +tp1_dist (lot2 nets 0 @BE) | +tp1_r_mult
#   WIN (TP1 HIT) [interim/open]   | +tp1_dist (lot2 still open) | +tp1_r_mult
#   WIN (TP2 HIT)                  | +tp1_dist +tp2_dist         | +tp1_r_mult+tp2_r_mult
#
# 1 pip = $0.10 on a single 0.01 lot (confirmed against your own bot's SL
# alert messages, e.g. ID#79: 43.5 pips SL = $4.35 on one 0.01 lot). Two
# lots at $0.10/pip each is $0.20/pip combined -- captured below by simply
# not halving the distances the way the old TP1_PARTIAL_CLOSE_RATIO did.

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
        # Neither lot ever reached TP1 -- both close at SL. Two 0.01 lots,
        # each risking sl_dist, so the combined loss is 2x a single-lot SL.
        total_pips = -(sl_dist * 10.0) * 2.0
    elif outcome == "CLOSED (TP1 HIT / SL BE)":
        # Lot 1 banked tp1_dist in full. Lot 2 (the runner) was stopped at
        # breakeven -- zero pips, not a loss and not additional profit.
        total_pips = tp1_dist * 10.0
    elif outcome == "WIN (TP1 HIT)":
        # Interim state: lot 1 has closed at TP1; lot 2 is still open and
        # not yet resolved, so only lot 1's pips are realized so far.
        total_pips = tp1_dist * 10.0
    elif outcome in ["WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"]:
        # Lot 1 closed at TP1, lot 2 (the runner) continued on to TP2 --
        # both legs are realized profit, so both are counted in full.
        total_pips = (tp1_dist + tp2_dist) * 10.0
    else:
        diff = (exit_p - entry) if action == "BUY" else (entry - exit_p)
        total_pips = diff * 10.0 * 2.0  # PENDING/unclassified fallback: treat as 2-lot mark-to-market
    profit_usd = total_pips * 0.10
    return total_pips, profit_usd

def compute_r_multiple(action: str, entry: float, exit_price: float, sl: float, tp1: float = 0.0, tp2: float = 0.0, outcome: str = "PENDING") -> float:
    risk_dist = abs(entry - sl)
    if risk_dist <= 0:
        risk_dist = abs(entry - exit_price) if "LOSS" in outcome else 2.5
        if risk_dist == 0: risk_dist = 2.5

    # V10: same dual-0.01-lot model as compute_trade_pips. R is expressed
    # per unit of SINGLE-LOT risk (risk_dist), so a full loss on both lots
    # is exactly -2.0R, matching "every signal risks 1R per lot, two lots
    # per signal" rather than the old hardcoded -1.0.
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
    # Instrumentation bucket -- how far price had already moved beyond its
    # recent consolidation bracket (in ATR) at the moment a signal fired.
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
            if win_rate < 35: flag = " \u26a0\ufe0f underperforming"
            elif win_rate > 65: flag = " \u2705 strong"
        lines.append(f"\u2022 {label}: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}")
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
    """Apply strategy-specific 5M EMA columns without refetching market data."""
    df_5m = df_5m.copy()
    df_5m["ema_fast"] = df_5m["close"].ewm(span=fast, adjust=False).mean()
    df_5m["ema_slow"] = df_5m["close"].ewm(span=slow, adjust=False).mean()
    return df_5m


# =====================================================================
# STRATEGY C: EXTREME-FREQUENCY M5 IMPULSE / PULLBACK / BREAKOUT SCALPER
# =====================================================================
# No 15M or 1H data is required by Strategy C.
# It uses M5 price action + EMA9/EMA21 + session VWAP + ATR.
# It can consume real MT5 M5 bars via /mt5-market-data; otherwise it uses
# the shared Twelve Data M5 snapshot as a paper/backtest fallback.
#
# IMPORTANT: C remains PAPER by default. It is deliberately not promoted to
# live execution merely by replacing the old paper breakout logic.

EXTREME_EMA_FAST = 9
EXTREME_EMA_SLOW = 21
EXTREME_ATR_PERIOD = 14
EXTREME_IMPULSE_MIN_BODY_ATR = 0.30
EXTREME_PULLBACK_MAX_DISTANCE_ATR = 0.35
EXTREME_BREAKOUT_LOOKBACK = 6
EXTREME_MIN_RANGE_ATR = 0.20
EXTREME_MAX_SPREAD_ATR = 0.25
EXTREME_SL_ATR = 0.70
EXTREME_TP1_R = 0.70
EXTREME_TP2_R = 1.20
EXTREME_MAX_HOLD_CANDLES = 6
EXTREME_MIN_REENTRY_DISTANCE_ATR = 0.20
EXTREME_COOLDOWN_SECONDS = 10
EXTREME_REQUIRE_VWAP_ALIGNMENT = True
EXTREME_ALLOW_BOTH_DIRECTIONS = True
EXTREME_MAX_TRADES_PER_DAY = 50
EXTREME_SESSION_START_HOUR = 7
EXTREME_SESSION_END_HOUR = 23

_extreme_state = {
    "last_signal_time": None,
    "last_entry_price": None,
    "last_action": None,
    "trades_today": 0,
    "trade_day": None,
}

def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0

def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP. Uses MT5 tick volume/real volume when supplied.
    Twelve Data often has no usable FX volume, so it falls back to a
    price-only cumulative typical-price proxy rather than inventing volume."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    work = df.copy()
    tp = _typical_price(work)
    if "volume" in work.columns:
        vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    elif "tick_volume" in work.columns:
        vol = pd.to_numeric(work["tick_volume"], errors="coerce").fillna(0.0)
    else:
        vol = pd.Series(1.0, index=work.index)
    # Reset at each UTC date; MT5 payload may include a datetime column.
    if "datetime" in work.columns:
        dt = pd.to_datetime(work["datetime"], errors="coerce")
        day = dt.dt.date
        pv = tp * vol
        return pv.groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)

def _extreme_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    prev_close = d["close"].shift(1)
    tr = np.maximum(
        d["high"] - d["low"],
        np.maximum((d["high"] - prev_close).abs(), (d["low"] - prev_close).abs())
    )
    d["tr"] = tr
    d["atr"] = tr.rolling(EXTREME_ATR_PERIOD).mean()
    d["ema9"] = d["close"].ewm(span=EXTREME_EMA_FAST, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=EXTREME_EMA_SLOW, adjust=False).mean()
    d["vwap"] = _session_vwap(d)
    d["body"] = (d["close"] - d["open"]).abs()
    d["range"] = d["high"] - d["low"]
    return d

def _extreme_roll_day(now_wib: datetime):
    day = now_wib.date()
    if _extreme_state["trade_day"] != day:
        _extreme_state["trade_day"] = day
        _extreme_state["trades_today"] = 0

def _extreme_session_ok(now_wib: datetime) -> bool:
    return EXTREME_SESSION_START_HOUR <= now_wib.hour < EXTREME_SESSION_END_HOUR and is_forex_market_open(now_wib)

def _extreme_vwap_aligned(action: str, row) -> bool:
    if not EXTREME_REQUIRE_VWAP_ALIGNMENT:
        return True
    vwap = row.get("vwap")
    if vwap is None or pd.isna(vwap):
        return True
    return float(row["close"]) > float(vwap) if action == "BUY" else float(row["close"]) < float(vwap)

def detect_extreme_m5_signal(df_5m: pd.DataFrame):
    """High-frequency local engine. Returns (action, trigger, metrics).
    It intentionally permits repeated continuation entries rather than waiting
    for a fresh EMA cross on every trade."""
    if df_5m is None or len(df_5m) < max(EXTREME_ATR_PERIOD + 3, EXTREME_BREAKOUT_LOOKBACK + 2):
        return "HOLD", "Insufficient M5 history", {}
    d = _extreme_indicators(df_5m)
    cur = d.iloc[-1]
    prev = d.iloc[-2]
    atr = float(cur["atr"]) if not pd.isna(cur["atr"]) else 0.0
    if atr <= 0:
        return "HOLD", "ATR unavailable", {}

    close = float(cur["close"]); op = float(cur["open"])
    high = float(cur["high"]); low = float(cur["low"])
    ema9 = float(cur["ema9"]); ema21 = float(cur["ema21"])
    body_atr = abs(close - op) / atr

    long_regime = close > ema9 > ema21 and _extreme_vwap_aligned("BUY", cur)
    short_regime = close < ema9 < ema21 and _extreme_vwap_aligned("SELL", cur)

    prior_high = float(d["high"].iloc[-EXTREME_BREAKOUT_LOOKBACK-1:-1].max())
    prior_low = float(d["low"].iloc[-EXTREME_BREAKOUT_LOOKBACK-1:-1].min())

    bullish_impulse = (
        long_regime and close > prior_high and close > op and
        body_atr >= EXTREME_IMPULSE_MIN_BODY_ATR
    )
    bearish_impulse = (
        short_regime and close < prior_low and close < op and
        body_atr >= EXTREME_IMPULSE_MIN_BODY_ATR
    )

    pullback_long = (
        long_regime and
        float(cur["low"]) <= ema9 + EXTREME_PULLBACK_MAX_DISTANCE_ATR * atr and
        close > ema9 and close > op and
        float(prev["close"]) >= float(prev["ema9"])
    )
    pullback_short = (
        short_regime and
        float(cur["high"]) >= ema9 - EXTREME_PULLBACK_MAX_DISTANCE_ATR * atr and
        close < ema9 and close < op and
        float(prev["close"]) <= float(prev["ema9"])
    )

    # Momentum re-entry: previous candle was aligned and current candle
    # continues through its high/low without requiring a new EMA crossover.
    reentry_long = (
        long_regime and close > float(prev["high"]) and close > op and
        body_atr >= EXTREME_IMPULSE_MIN_BODY_ATR * 0.75
    )
    reentry_short = (
        short_regime and close < float(prev["low"]) and close < op and
        body_atr >= EXTREME_IMPULSE_MIN_BODY_ATR * 0.75
    )

    metrics = {
        "atr": atr, "ema9": ema9, "ema21": ema21,
        "vwap": float(cur["vwap"]) if not pd.isna(cur["vwap"]) else None,
        "body_atr": body_atr,
        "prior_high": prior_high, "prior_low": prior_low,
    }

    if bullish_impulse:
        return "BUY", "Impulse Breakout", metrics
    if bearish_impulse:
        return "SELL", "Impulse Breakout", metrics
    if pullback_long:
        return "BUY", "EMA9 Micro Pullback", metrics
    if pullback_short:
        return "SELL", "EMA9 Micro Pullback", metrics
    if reentry_long:
        return "BUY", "Momentum Re-entry", metrics
    if reentry_short:
        return "SELL", "Momentum Re-entry", metrics
    return "HOLD", "No Extreme M5 Setup", metrics

def _extreme_trade_plan(action: str, price: float, atr: float, df: pd.DataFrame):
    """Dynamic ATR stop/targets; targets remain small enough for high frequency."""
    risk = max(0.01, atr * EXTREME_SL_ATR)
    if action == "BUY":
        sl = price - risk
        tp1 = price + risk * EXTREME_TP1_R
        tp2 = price + risk * EXTREME_TP2_R
    else:
        sl = price + risk
        tp1 = price - risk * EXTREME_TP1_R
        tp2 = price - risk * EXTREME_TP2_R
    return sl, tp1, tp2, risk

def _mt5_cache_fresh() -> bool:
    ts = mt5_market_cache.get("updated_at")
    if mt5_market_cache.get("df") is None or ts is None:
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() <= MT5_DATA_CACHE_TTL_SECONDS

def _df_from_mt5_payload(payload: dict):
    """Accept either {'bars':[...]} or {'candles':[...]} with OHLC and time."""
    rows = payload.get("bars") or payload.get("candles") or []
    if not isinstance(rows, list) or not rows:
        return None
    out = pd.DataFrame(rows)
    time_col = next((c for c in ("datetime", "time", "timestamp") if c in out.columns), None)
    if time_col is None:
        return None
    out["datetime"] = pd.to_datetime(out[time_col], unit="s", errors="coerce")
    # If timestamps were already ISO strings, the unit='s' conversion can fail;
    # retry those rows as normal datetimes.
    bad = out["datetime"].isna()
    if bad.any():
        out.loc[bad, "datetime"] = pd.to_datetime(out.loc[bad, time_col], errors="coerce")
    required = {"open", "high", "low", "close"}
    if not required.issubset(out.columns):
        return None
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out["volume"] = pd.to_numeric(out["tick_volume"], errors="coerce")
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    return out.sort_values("datetime").drop_duplicates("datetime").tail(150).reset_index(drop=True)

async def evaluate_extreme_strategy(client: httpx.AsyncClient, market_df_5m: pd.DataFrame, now_wib: datetime):
    """Strategy C paper engine. Prefers fresh MT5 data; falls back to shared TD M5."""
    _extreme_roll_day(now_wib)
    if not _extreme_session_ok(now_wib):
        return

    if _extreme_state["trades_today"] >= EXTREME_MAX_TRADES_PER_DAY:
        return

    # Prefer the EA-fed MT5 market snapshot when available.
    source = "MT5"
    df = mt5_market_cache["df"] if _mt5_cache_fresh() else market_df_5m
    if df is market_df_5m:
        source = "TWELVE_DATA_FALLBACK"

    action, trigger, metrics = detect_extreme_m5_signal(df)
    if action == "HOLD":
        return

    # Optional repeated-entry spacing. This prevents duplicate signals from
    # repeated POSTs of the same MT5 candle/tick.
    now_utc = datetime.now(timezone.utc)
    last_ts = _extreme_state.get("last_signal_time")
    if last_ts and (now_utc - last_ts).total_seconds() < EXTREME_COOLDOWN_SECONDS:
        return

    price = float(df["close"].iloc[-1])
    atr = float(metrics.get("atr") or 0.0)
    if atr <= 0:
        return

    last_price = _extreme_state.get("last_entry_price")
    if last_price is not None and abs(price - float(last_price)) < EXTREME_MIN_REENTRY_DISTANCE_ATR * atr:
        return

    # For the extreme-frequency strategy, no LLM veto is used. This is a
    # deterministic price-action engine; AI calls would become the bottleneck
    # and destroy the intended frequency.
    sl, tp1, tp2, risk = _extreme_trade_plan(action, price, atr, df)

    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection(); cur = conn.cursor()
        reasoning = (
            f"{trigger}; M5 EMA9/EMA21 aligned, session VWAP aligned, "
            f"ATR={atr:.3f}, body={metrics.get('body_atr', 0):.2f} ATR, data={source}."
        )
        cur.execute("""
            INSERT INTO signals (
                timestamp,status,action,trigger_type,price,entry_price,sl,sl_price,
                tp1,tp1_price,tp2,tp2_price,confidence,adx_15m,stoch_rsi_15m,
                divergence_type,reasoning,outcome,outcome_timestamp,trend_15m,
                adx_15m_true,regime,strategy,execution_mode,created_at
            )
            VALUES (%s,'EXECUTED',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,0,
                    'None',%s,'PENDING','',%s,0,%s,%s,%s,NOW())
            RETURNING id
        """, (
            (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB"),
            action, trigger, price, price, sl, sl, tp1, tp1, tp2, tp2,
            0.90, reasoning, "EXTREME_M5", BREAKOUT_STRATEGY, BREAKOUT_EXECUTION_MODE
        ))
        row = cur.fetchone(); sid = int(row["id"]) if row else None
        conn.commit(); cur.close(); conn.close()
        _extreme_state["last_signal_time"] = now_utc
        _extreme_state["last_entry_price"] = price
        _extreme_state["last_action"] = action
        _extreme_state["trades_today"] += 1

        await send_telegram_alert(
            client,
            f"⚡ *EXTREME M5 {BREAKOUT_STRATEGY} [PAPER] SIGNAL #{sid}*\n\n"
            f"Asset: *XAU/USD*\nAction: *{action}*\nTrigger: *{trigger}*\n"
            f"Entry: *${price:.2f}*\nSL: *${sl:.2f}* ({EXTREME_SL_ATR:.2f} ATR)\n"
            f"TP1: *${tp1:.2f}* ({EXTREME_TP1_R:.2f}R)\n"
            f"TP2: *${tp2:.2f}* ({EXTREME_TP2_R:.2f}R)\n"
            f"EMA: *9/21* | VWAP: *aligned* | ATR: *{atr:.3f}*\n"
            f"Data source: *{source}*\n"
            f"Daily C signals: *{_extreme_state['trades_today']}/{EXTREME_MAX_TRADES_PER_DAY}*\n\n"
            f"⚠️ PAPER ONLY",
            BREAKOUT_TELEGRAM_CHAT_ID, BREAKOUT_TELEGRAM_BOT_TOKEN
        )
    except Exception as e:
        logging.error(f"[EXTREME C DB ERROR] {e}")

# Backward-compatible alias so any old internal status/command text can still
# refer to Strategy C without resurrecting the old OCO engine.
async def evaluate_breakout_strategy(client, market_df_5m, adx_15m_true, now_wib):
    await evaluate_extreme_strategy(client, market_df_5m, now_wib)

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

    # FIXED: the veto-rule text is now mode-aware. Range mode ONLY ever fires
    # when 5M ADX < RANGE_MODE_ADX_MAX by design -- if this prompt still told
    # the AI reviewer "VETO if ADX < 20", every single range signal would get
    # auto-vetoed regardless of quality, silently defeating the whole feature.
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
            # Offloaded to a thread: groq_client is the sync OpenAI-compatible
            # SDK, and calling it directly here would block this async
            # function's event loop for the full round-trip -- delaying
            # Telegram alerts, the other strategy's evaluation, and the MT5
            # bridge's HTTP responses if the AI API is slow.
            res = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, response_format={"type": "json_object"}
            )
            return SignalOutput(**json.loads(res.choices[0].message.content))
        except Exception as e:
            logging.warning(f"[AI WARNING] Groq call failed: {e}. Falling back to Gemini.")

    if GEMINI_API_KEY:
        try:
            res = await asyncio.to_thread(
                genai_client.models.generate_content,
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

    # FIXED: this used to return action=proposed_action here, meaning if BOTH
    # Groq and Gemini failed, the trade executed anyway on raw EMA structure
    # alone -- the AI review was never actually a mandatory gate, just an
    # optional second opinion that silently no-opped on any outage. Now with
    # Strategy B live on real money, an AI outage should mean the system
    # holds, not that it trades blind. Both API failures now HOLD instead.
    logging.warning("[AI FAIL-SAFE] Both Groq and Gemini failed or are unconfigured -- holding this signal rather than executing blind.")
    return SignalOutput(action="HOLD", confidence=0.0, reasoning="AI fail-safe: both Groq and Gemini were unavailable, so this signal was held rather than executed without review.")


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
    """Evaluate one strategy on the SAME market snapshot used by the other strategy."""
    # Strategy-specific EMA pair is passed explicitly; no global EMA state is mutated.
    # This keeps Control A and Experimental B fully independent.
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

    # A/B one-direction controller.
    # Default is BUY_ONLY: the old dynamic 1H one-direction system is OFF.
    # /oneway_on switches back to DYNAMIC; /oneway_off returns to BUY_ONLY.
    # /both disables the one-direction restriction entirely.
    if strategy in (CONTROL_STRATEGY, EXPERIMENTAL_STRATEGY):
        if ONE_DIRECTION_MODE == "BUY_ONLY" and proposed_action == "SELL":
            logging.info(f"[{strategy}] [BUY-ONLY] SELL blocked.")
            proposed_action = "HOLD"
        elif ONE_DIRECTION_MODE == "DYNAMIC":
            if proposed_action == "SELL" and directional_bias == "BULLISH":
                logging.info(f"[{strategy}] [DYNAMIC 1H BLOCK] SELL blocked -- 1H bias BULLISH.")
                proposed_action = "HOLD"
            elif proposed_action == "BUY" and directional_bias == "BEARISH":
                logging.info(f"[{strategy}] [DYNAMIC 1H BLOCK] BUY blocked -- 1H bias BEARISH.")
                proposed_action = "HOLD"

    # Same exhaustion/chop guard for both strategies, isolated by strategy history.
    if proposed_action in ("BUY", "SELL") and strategy_mode == "TREND":
        guard_block, guard_metrics = trend_exhaustion_guard(proposed_action, df_5m, strategy)
        regime_metrics["exhaustion_guard"] = guard_metrics
        if guard_metrics.get("score", 0) >= EXHAUSTION_SCORE_CAUTION:
            log_scan_event(
                "EXHAUSTION_GUARD", stage="RISK", action=proposed_action, price=curr_price,
                adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
                decision="HOLD" if guard_block else "WATCH",
                reason=f"{strategy}: {guard_metrics.get('reason', '')}", details=guard_metrics
            )
        if guard_block:
            logging.info(f"[{strategy}] [EXHAUSTION GUARD] Blocking {proposed_action}: score={guard_metrics.get('score')} reason={guard_metrics.get('reason')}")
            proposed_action = "HOLD"

    # Isolated three-loss directional lock.
    for guarded_direction in ("BUY", "SELL"):
        if proposed_action != guarded_direction:
            continue
        if consecutive_loss_count(guarded_direction, EXHAUSTION_HARD_LOSS_LOCK, strategy) >= EXHAUSTION_HARD_LOSS_LOCK:
            reset_ok, reset_reason = fresh_directional_expansion_confirmed(guarded_direction, df_5m, trend_15m)
            if not reset_ok:
                log_scan_event(
                    "DIRECTION_LOCKED", stage="RISK", action=guarded_direction, price=curr_price,
                    adx_5m=adx_5m, adx_15m=adx_15m_true, trend_15m=trend_15m,
                    decision="HOLD", reason=f"{strategy}: 3-loss directional lock: {reset_reason}"
                )
                proposed_action = "HOLD"
            else:
                logging.info(f"[{strategy}] [DIRECTION LOCK] {guarded_direction} released: {reset_reason}")

    # Strategy-isolated global loss cooldown.
    if proposed_action != "HOLD":
        try:
            conn = get_db_connection(); cursor = conn.cursor()
            cursor.execute("""
                SELECT outcome_timestamp FROM signals
                WHERE status = 'EXECUTED' AND outcome = 'LOSS (SL HIT)'
                  AND strategy = %s AND outcome_timestamp IS NOT NULL AND outcome_timestamp != ''
                ORDER BY id DESC LIMIT 1
            """, (strategy,))
            last_loss = cursor.fetchone(); cursor.close(); conn.close()
            if last_loss and last_loss.get("outcome_timestamp"):
                last_loss_time = datetime.strptime(str(last_loss["outcome_timestamp"]).replace(" WIB", ""), "%Y-%m-%d %H:%M:%S")
                if 0 <= (now_wib.replace(tzinfo=None) - last_loss_time).total_seconds() / 60.0 < LOSS_COOLDOWN_MINUTES:
                    logging.info(f"[{strategy}] [LOSS COOLDOWN] Skipping {proposed_action}.")
                    proposed_action = "HOLD"
        except Exception as e:
            logging.error(f"[{strategy}] loss cooldown check: {e}")

    # Strategy-isolated distance cooldown.
    if proposed_action != "HOLD":
        try:
            conn = get_db_connection(); cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(entry_price, price, 0) AS entry_p, outcome
                FROM signals
                WHERE status = 'EXECUTED' AND action = %s AND strategy = %s
                ORDER BY id DESC LIMIT 1
            """, (str(proposed_action), strategy))
            last_trade = cursor.fetchone(); cursor.close(); conn.close()
            if last_trade:
                required_distance = 2.00 if str(last_trade.get("outcome") or "PENDING") == "PENDING" else 1.50
                if abs(curr_price - float(last_trade["entry_p"])) < required_distance:
                    logging.info(f"[{strategy}] [DISTANCE COOLDOWN] Skipping {proposed_action}: Price too close.")
                    proposed_action = "HOLD"
        except Exception as e:
            logging.error(f"[{strategy}] distance cooldown check: {e}")

    if proposed_action == "HOLD":
        logging.info(
            f"[{strategy}] [MARKET SCAN] Price: ${curr_price:.2f} | EMA{ema_fast}: ${curr_ema_fast:.2f} | "
            f"EMA{ema_slow}: ${curr_ema_slow:.2f} | ADX5m: {adx_5m:.1f} | Mode: {strategy_mode} | "
            f"15mTrend: {trend_15m} | Status: HOLD"
        )
        return

    logging.info(f"[{strategy}] [{strategy_mode}] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI...")
    ai_decision = await analyze_signal_with_ai(
        proposed_action, trigger_type, curr_price, df_5m, trend_15m,
        adx_15m_true, strategy_mode, range_high, range_low, ema_fast, ema_slow
    )

    atr_5m = float(df_5m["atr"].iloc[-1]) if not pd.isna(df_5m["atr"].iloc[-1]) else 3.0
    entry_extension_atr, entry_climax_ratio = compute_entry_extension(df_5m, proposed_action)

    if strategy_mode == "RANGE":
        if proposed_action == "BUY":
            sl_price = range_low - RANGE_SL_BUFFER_ATR_MULT * atr_5m
            tp2_price = range_high
            tp1_price = curr_price + (tp2_price - curr_price) * 0.5
        else:
            sl_price = range_high + RANGE_SL_BUFFER_ATR_MULT * atr_5m
            tp2_price = range_low
            tp1_price = curr_price - (curr_price - tp2_price) * 0.5
        tp1_r_mult = abs(tp1_price - curr_price) / max(abs(curr_price - sl_price), 0.01)
        tp2_r_mult = abs(tp2_price - curr_price) / max(abs(curr_price - sl_price), 0.01)
    else:
        risk = max(2.5, atr_5m * 1.0)
        tp1_r_mult = 1.5
        tp2_r_mult = 2.5
        sl_price = curr_price - risk if proposed_action == "BUY" else curr_price + risk
        tp1_price = curr_price + risk * tp1_r_mult if proposed_action == "BUY" else curr_price - risk * tp1_r_mult
        tp2_price = curr_price + risk * tp2_r_mult if proposed_action == "BUY" else curr_price - risk * tp2_r_mult
        # No SL compression, no dollar cap, no dynamic lot sizing -- plain
        # ATR-based risk for every trade, same as before the $6 cap was ever
        # introduced. The 1H EMA200 bias filter above (kept) is what guards
        # against a losing streak on a sudden trend reversal -- it blocks
        # entries against the new direction rather than resizing the stop.

    if ai_decision.action == proposed_action:
        new_id = log_trade_signal(
            "EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price,
            float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning,
            trend_15m, adx_15m_true, entry_extension_atr, entry_climax_ratio,
            strategy_mode, regime_metrics, strategy, execution_mode
        )
        mode_tag = "📊 RANGE FADE" if strategy_mode == "RANGE" else "🚀 TREND"
        paper_tag = " [PAPER]" if execution_mode == "PAPER" else " [LIVE]"
        msg = (
            f"{mode_tag} *{strategy}{paper_tag} SIGNAL #{new_id}*\n\n"
            f"Asset: *XAUUSD*\nAction: *{proposed_action}*\nType: *{trigger_type}*\n"
            f"Entry Price: *${curr_price:.2f}*\n\n"
            f"Stop Loss: *${sl_price:.2f}*\n"
            f"TP1 ({tp1_r_mult:.1f}R): *${tp1_price:.2f}*\n"
            f"TP2 ({tp2_r_mult:.1f}R): *${tp2_price:.2f}*\n\n"
            f"Execution: *{execution_mode}*\nReasoning: {ai_decision.reasoning}"
        )
        await send_telegram_alert(client, msg, target_chat_id=alert_chat_id, target_bot_token=alert_bot_token)
    else:
        log_trade_signal(
            "VETOED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price,
            float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning,
            trend_15m, adx_15m_true, entry_extension_atr, entry_climax_ratio,
            strategy_mode, regime_metrics, strategy, execution_mode
        )


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

                if not is_forex_market_open(now_wib):
                    if now_wib.minute % 30 == 0 and now_wib.second < 5:
                        logging.info("[SLEEP STATUS] Market closed (weekend, WIB). Waiting for reopen...")
                    await asyncio.sleep(120)
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

                # ONE shared 5M request for both strategies.
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
                log_scan_event("SCAN_START", stage="SCAN", decision="STARTED", reason="Shared 5M market snapshot for A/B experiment")

                curr_high = float(df_5m["high"].iloc[-1])
                curr_low = float(df_5m["low"].iloc[-1])
                update_open_trades(curr_high, curr_low)

                # Build the 15M confluence frame LOCALLY from the shared 5M
                # snapshot. This removes a second Twelve Data feed entirely.
                # 100 M5 candles provide ~33 completed 15M candles, enough for
                # the existing EMA9/20 confluence and ADX14 calculation.
                try:
                    df_15m_local = (
                        df_5m.set_index("datetime")[["open","high","low","close"]]
                        .resample("15min", label="right", closed="right")
                        .agg({"open":"first","high":"max","low":"min","close":"last"})
                        .dropna()
                        .reset_index()
                    )
                    if len(df_15m_local) >= 21:
                        cached_15m["df"] = calculate_metrics_tf(df_15m_local)
                        cached_15m["fetched_at"] = datetime.now(timezone.utc)
                except Exception as e:
                    logging.warning(f"[15M LOCAL RESAMPLE] {e}")

                trend_15m, trend_15m_sep = compute_ema_trend(cached_15m["df"]) if cached_15m["df"] is not None else ("NEUTRAL", 0.0)
                adx_15m_true = float(cached_15m["df"]["adx"].iloc[-1]) if cached_15m["df"] is not None and not pd.isna(cached_15m["df"]["adx"].iloc[-1]) else 0.0

                # 1H data is fetched ONLY when the dynamic one-direction mode
                # is enabled. BUY_ONLY/BOTH therefore spend zero Twelve Data
                # credits on the old directional system.
                directional_bias, bias_sep = "NEUTRAL", 0.0
                if ONE_DIRECTION_MODE == "DYNAMIC":
                    need_1h_refresh = (
                        cached_1h["df"] is None or cached_1h["fetched_at"] is None or
                        (datetime.now(timezone.utc) - cached_1h["fetched_at"] >= timedelta(minutes=ONE_H_REFRESH_MINUTES))
                    )
                    if need_1h_refresh and twelve_data_budget_ok(now_wib):
                        df_1h_raw = await fetch_timeframe_data(client, "1h", outputsize=ONE_H_OUTPUTSIZE, now_wib=now_wib)
                        if df_1h_raw is not None and len(df_1h_raw) >= ONE_H_EMA_PERIOD + 1:
                            cached_1h["df"] = df_1h_raw
                            cached_1h["fetched_at"] = datetime.now(timezone.utc)
                    directional_bias, bias_sep = compute_1h_directional_bias(cached_1h["df"])

                # CONTROL A: existing Exhaustion Guard v1, EMA 5/9, PAPER.
                await evaluate_strategy_cycle(
                    client, df_5m, trend_15m, adx_15m_true, now_wib,
                    CONTROL_STRATEGY, 5, 9, CONTROL_EXECUTION_MODE,
                    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, directional_bias
                )

                # EXPERIMENT B: same system + same exhaustion guard, EMA 5/15, LIVE.
                # It receives the exact same candles and 15M confluence snapshot.
                await evaluate_strategy_cycle(
                    client, df_5m, trend_15m, adx_15m_true, now_wib,
                    EXPERIMENTAL_STRATEGY, EXPERIMENTAL_EMA_FAST, EXPERIMENTAL_EMA_SLOW,
                    EXPERIMENTAL_EXECUTION_MODE, EXPERIMENTAL_TELEGRAM_BOT_TOKEN,
                    EXPERIMENTAL_TELEGRAM_CHAT_ID, directional_bias
                )

                # STRATEGY C: Extreme-frequency M5 engine. It prefers real MT5
                # bars received through /mt5-market-data, with shared Twelve Data
                # M5 as a fallback. It does not request another Twelve Data feed.
                if BREAKOUT_TELEGRAM_BOT_TOKEN and BREAKOUT_TELEGRAM_CHAT_ID:
                    await evaluate_extreme_strategy(client, df_5m, now_wib)

                del df_5m
                gc.collect()
                await asyncio.sleep(5)

            except Exception as e:
                logging.error(f"[SCAN LOOP ERROR] {e}")
                await asyncio.sleep(10)


# --- FASTAPI LIFESPAN & AUTOMATED WEBHOOK SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not EXPERIMENTAL_TELEGRAM_BOT_TOKEN or not EXPERIMENTAL_TELEGRAM_CHAT_ID:
        logging.warning("[A/B] Experimental Telegram credentials are not configured; experimental signals will still be logged to DB but Telegram alerts will be skipped.")
    if APP_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                if TELEGRAM_BOT_TOKEN:
                    webhook_a = f"{APP_URL.rstrip('/')}/telegram-webhook"
                    set_a = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
                    res_a = await client.post(set_a, data={"url": webhook_a})
                    logging.info(f"[CONTROL WEBHOOK SETUP] {res_a.text}")
                    await client.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands", json={"commands":[
                        {"command":"start","description":"Show Control bot commands"},
                        {"command":"help","description":"Show Control command menu"},
                        {"command":"status","description":"MT5/server/API status"},
                        {"command":"stats","description":"Live EMA 5/9 performance"},
                        {"command":"pips","description":"Pips and USD breakdown"},
                        {"command":"logs","description":"Last 10 live trades"},
                        {"command":"analyze","description":"Forward-test analysis"},
                        {"command":"pause","description":"Emergency kill switch"},
                        {"command":"resume","description":"Resume auto-trading"},
                        {"command":"oneway_on","description":"Enable dynamic 1H one-direction"},
                        {"command":"oneway_off","description":"Disable dynamic mode; BUY only"},
                        {"command":"both","description":"Allow BUY and SELL"}
                    ]})
                if EXPERIMENTAL_TELEGRAM_BOT_TOKEN:
                    webhook_b = f"{APP_URL.rstrip('/')}/telegram-webhook-b"
                    set_b = f"https://api.telegram.org/bot{EXPERIMENTAL_TELEGRAM_BOT_TOKEN}/setWebhook"
                    res_b = await client.post(set_b, data={"url": webhook_b})
                    logging.info(f"[EXPERIMENTAL WEBHOOK SETUP] {res_b.text}")
                    await client.post(f"https://api.telegram.org/bot{EXPERIMENTAL_TELEGRAM_BOT_TOKEN}/setMyCommands", json={"commands":[
                        {"command":"start","description":"Show A/B bot commands"},
                        {"command":"help","description":"Show A/B command menu"},
                        {"command":"stats","description":"A/B performance dashboard"},
                        {"command":"compare","description":"Compare EMA 5/9 vs 5/15"},
                        {"command":"status","description":"Read-only system status"},
                        {"command":"last","description":"Last 10 paper trades"},
                        {"command":"oneway_on","description":"Enable dynamic 1H one-direction"},
                        {"command":"oneway_off","description":"Disable dynamic mode; BUY only"},
                        {"command":"both","description":"Allow BUY and SELL"}
                    ]})
                if BREAKOUT_TELEGRAM_BOT_TOKEN:
                    webhook_c = f"{APP_URL.rstrip('/')}/telegram-webhook-c"
                    set_c = f"https://api.telegram.org/bot{BREAKOUT_TELEGRAM_BOT_TOKEN}/setWebhook"
                    res_c = await client.post(set_c, data={"url": webhook_c})
                    logging.info(f"[BREAKOUT WEBHOOK SETUP] {res_c.text}")
                    await client.post(f"https://api.telegram.org/bot{BREAKOUT_TELEGRAM_BOT_TOKEN}/setMyCommands", json={"commands":[
                        {"command":"start","description":"Show breakout bot commands"},
                        {"command":"help","description":"Show extreme M5 command menu"},
                        {"command":"status","description":"Extreme M5 status"},
                        {"command":"stats","description":"Extreme M5 performance"},
                        {"command":"range","description":"Current Extreme M5 setup"},
                        {"command":"last","description":"Last Extreme M5 signals"},
                        {"command":"cancel","description":"No active OCO"}
                    ]})
        except Exception as e:
            logging.error(f"[AUTO WEBHOOK SETUP ERROR] Failed: {e}")

    scan_task = asyncio.create_task(background_scanning_loop())
    yield
    scan_task.cancel()


app = FastAPI(lifespan=lifespan)

# Real-MT5 market-data ingress. The EA should POST a small rolling M5 history.
@app.post("/mt5-market-data")
async def mt5_market_data(request: Request):
    global mt5_market_cache
    try:
        if MT5_DATA_SECRET:
            supplied = request.headers.get("X-MT5-SECRET", "")
            if supplied != MT5_DATA_SECRET:
                return {"ok": False, "error": "unauthorized"}
        payload = await request.json()
        df = _df_from_mt5_payload(payload)
        if df is None or len(df) < EXTREME_ATR_PERIOD + 3:
            return {"ok": False, "error": "invalid_or_insufficient_m5_bars"}
        mt5_market_cache["df"] = df
        mt5_market_cache["updated_at"] = datetime.now(timezone.utc)
        mt5_market_cache["source"] = "MT5_EA"
        return {
            "ok": True,
            "source": "MT5_EA",
            "bars": len(df),
            "last_bar": str(df["datetime"].iloc[-1]),
            "updated_at": mt5_market_cache["updated_at"].isoformat(),
        }
    except Exception as e:
        logging.error(f"[MT5 DATA INGEST ERROR] {e}")
        return {"ok": False, "error": str(e)}

@app.get("/")
def home():
    return {"status": "ok", "message": "A/B/C scanner active: A=EMA 5/9 PAPER + B=EMA 5/15 LIVE + C=EXTREME M5 PAPER.", "comparison": "/ab-comparison"}


# =====================================================================
# MT5 COPIER BRIDGE API ENDPOINT
# =====================================================================
_latest_signal_cache = {"response": None, "cached_at": None}
LATEST_SIGNAL_CACHE_TTL_SECONDS = 12
# A new LIVE signal can only ever appear once per 5-minute scan cycle at the
# absolute fastest -- so caching this endpoint's DB read for a few seconds
# costs nothing in responsiveness (worst case: the EA sees a new trade up to
# ~12s later than instant) but cuts Postgres connections from one per EA
# poll (every 10s by default = ~8,640/day) down to one per ~12s regardless
# of how often or how many EAs poll. This was the dominant driver behind
# the database's compute/connection limits being hit: constant fresh connections never let
# the compute endpoint go idle long enough to auto-suspend.

@app.get("/get-latest-signal")
async def get_latest_signal():
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME

    LAST_MT5_PING_TIME = datetime.now(timezone.utc) + timedelta(hours=7)

    if not SYSTEM_TRADING_ENABLED:
        return {"signal": None, "trading_enabled": False, "status": "PAUSED"}
    if not DATABASE_URL:
        return {"signal": None, "error": "DATABASE_URL not set", "trading_enabled": SYSTEM_TRADING_ENABLED}

    now = datetime.now(timezone.utc)
    cached = _latest_signal_cache
    if cached["response"] is not None and cached["cached_at"] is not None:
        age = (now - cached["cached_at"]).total_seconds()
        if age < LATEST_SIGNAL_CACHE_TTL_SECONDS:
            return cached["response"]

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                   COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, COALESCE(timestamp, created_at::text, '') AS log_time
            FROM signals
            WHERE status = 'EXECUTED' AND execution_mode = 'LIVE'
            ORDER BY id DESC LIMIT 1;
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            result = {"id": int(row["id"]), "action": str(row["action"]).upper(), "entry": float(row["entry_p"]), "sl": float(row["sl_p"]), "tp1": float(row["tp1_p"]), "tp2": float(row["tp2_p"]), "timestamp": str(row["log_time"]), "trading_enabled": True}
        else:
            result = {"signal": None, "trading_enabled": True}
        _latest_signal_cache["response"] = result
        _latest_signal_cache["cached_at"] = now
        return result
    except Exception as e:
        logging.error(f"[MT5 BRIDGE ERROR] {e}")
        return {"error": str(e), "trading_enabled": SYSTEM_TRADING_ENABLED}


# =====================================================================
# A/B COMPARISON ENDPOINT (READ-ONLY)
# =====================================================================
@app.get("/ab-comparison")
async def ab_comparison():
    if not DATABASE_URL:
        return {"error": "DATABASE_URL not set"}
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("""
            SELECT strategy, execution_mode,
                   COUNT(*) FILTER (WHERE status='EXECUTED') AS executed,
                   COUNT(*) FILTER (WHERE status='VETOED') AS vetoed,
                   COUNT(*) FILTER (WHERE status='EXECUTED' AND (outcome LIKE 'WIN%%' OR outcome LIKE 'CLOSED%%')) AS wins,
                   COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome LIKE 'LOSS%%') AS losses,
                   COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome='PENDING') AS pending,
                   COALESCE(SUM(result_pips) FILTER (WHERE status='EXECUTED' AND result_pips IS NOT NULL),0) AS net_pips,
                   COALESCE(SUM(result_usd) FILTER (WHERE status='EXECUTED' AND result_usd IS NOT NULL),0) AS net_usd,
                   COALESCE(AVG(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS avg_r
            FROM signals
            WHERE strategy IN (%s, %s, %s)
            GROUP BY strategy, execution_mode
            ORDER BY strategy;
        """, (CONTROL_STRATEGY, EXPERIMENTAL_STRATEGY, BREAKOUT_STRATEGY))
        rows=cur.fetchall(); cur.close(); conn.close()
        result={}
        for r in rows:
            executed=int(r["executed"] or 0); wins=int(r["wins"] or 0)
            result[str(r["strategy"])] = {
                "execution_mode": r["execution_mode"], "executed": executed,
                "vetoed": int(r["vetoed"] or 0), "wins": wins,
                "losses": int(r["losses"] or 0), "pending": int(r["pending"] or 0),
                "win_rate_pct": round((wins/executed*100) if executed else 0, 2),
                "net_pips": round(float(r["net_pips"] or 0), 2),
                "net_usd": round(float(r["net_usd"] or 0), 2),
                "avg_r": round(float(r["avg_r"] or 0), 3),
            }
        return {"control": result.get(CONTROL_STRATEGY, {}), "experimental": result.get(EXPERIMENTAL_STRATEGY, {}), "breakout": result.get(BREAKOUT_STRATEGY, {})}
    except Exception as e:
        logging.error(f"[A/B COMPARISON ERROR] {e}")
        return {"error": str(e)}


# --- WEBHOOK ENDPOINT FOR TELEGRAM COMMANDS ---
async def _handle_telegram_webhook(request: Request, bot_role: str):
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME, ONE_DIRECTION_MODE
    try:
        data = await request.json()
        message = data.get("message", {})
        raw_text = message.get("text", "").strip().lower()
        sender_chat_id = str(message.get("chat", {}).get("id", ""))

        if not sender_chat_id or not raw_text: return {"status": "ignored"}

        # Each Telegram bot has its own command surface. Bot B is read-only/paper-only.
        CONTROL_COMMANDS = {"/start", "/help", "/status", "/stats", "/pips", "/logs", "/analyze", "/pause", "/resume", "/oneway_on", "/oneway_off", "/both"}
        EXPERIMENTAL_COMMANDS = {"/start", "/help", "/status", "/stats", "/compare", "/last", "/oneway_on", "/oneway_off", "/both"}
        BREAKOUT_COMMANDS = {"/start", "/help", "/status", "/stats", "/last"}
        allowed = BREAKOUT_COMMANDS if bot_role == "breakout" else (EXPERIMENTAL_COMMANDS if bot_role == "experimental" else CONTROL_COMMANDS)
        if raw_text not in allowed:
            return {"status": "ignored", "reason": "command_not_available_for_this_bot"}

        active_token = BREAKOUT_TELEGRAM_BOT_TOKEN if bot_role == "breakout" else (EXPERIMENTAL_TELEGRAM_BOT_TOKEN if bot_role == "experimental" else TELEGRAM_BOT_TOKEN)
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async def send_reply(text: str):
                await send_telegram_alert(client, text, target_chat_id=sender_chat_id, target_bot_token=active_token)

            if raw_text in ["/help", "/start"]:
                if bot_role == "breakout":
                    reply = (
                        "📦 *EXTREME M5 BOT COMMANDS:*\n\n"
                        "• `/status` - Extreme M5 status\n"
                        "• `/stats` - Extreme M5 performance\n"
                        "• `/last` - Last Extreme M5 signals\n"
                        "• `/help` - Display this menu\n\n"
                        "🟣 Strategy: *Extreme M5 Impulse/Pullback/Re-entry*\n"
                        "⚠️ PAPER ONLY — Strategy C uses MT5-fed market data when available."
                    )
                elif bot_role == "experimental":
                    reply = (
                        "🔬 *EXPERIMENTAL A/B BOT COMMANDS:*\n\n"
                        "• `/stats` - A/B performance dashboard\n"
                        "• `/compare` - EMA 5/9 vs EMA 5/15 comparison\n"
                        "• `/status` - Read-only system status\n"
                        "• `/last` - Last 10 experimental trades\n"
                         "• `/oneway_on` - Dynamic 1H direction ON\n"
                         "• `/oneway_off` - Dynamic direction OFF → BUY ONLY\n"
                         "• `/both` - Allow BUY + SELL\n"
                        "• `/help` - Display this command menu\n\n"
                        "🔴 Strategy: *EMA 5/15 — LIVE (real MT5 trades)*\n"
                        "🚨 This bot IS connected to live MT5 execution. Use /pause on the Control bot for the emergency kill switch.\n"
                    )
                else:
                    reply = (
                        f"🤖 *CONTROL EMA 5/9 BOT COMMANDS:*\n\n"
                        "• `/status` - Real-time MT5, server & API status\n"
                        "• `/stats` - Control performance dashboard (PAPER)\n"
                        "• `/pips` - Gross/net pips & USD breakdown\n"
                        "• `/logs` - Last 10 executed trades\n"
                        "• `/analyze` - Forward-test strategy analysis\n"
                        "• `/pause` - 🚨 Emergency kill switch (stops ALL strategies, including live B)\n"
                        "• `/resume` - 🟢 Re-enable auto-trading\n"
                         "• `/oneway_on` - Dynamic 1H direction ON\n"
                         "• `/oneway_off` - Dynamic direction OFF → BUY ONLY\n"
                         "• `/both` - Allow BUY + SELL\n"
                        "• `/help` - Display this command menu\n\n"
                        f"🟡 Strategy: *EMA {EMA_TREND_FAST}/{EMA_TREND_SLOW} — PAPER ONLY* | 15M confluence: *EMA {TREND_15M_EMA_FAST}/{TREND_15M_EMA_SLOW}*\n"
                        f"\u2139\ufe0f Live MT5 execution is currently on the *Experimental* bot (EMA 5/15), not this one.\n"
                    )
                await send_reply(reply)

            elif raw_text in ("/oneway_on", "/oneway_off", "/both"):
                if raw_text == "/oneway_on":
                    ONE_DIRECTION_MODE = "DYNAMIC"
                    mode_text = "DYNAMIC — 1H EMA200 decides allowed direction"
                elif raw_text == "/oneway_off":
                    ONE_DIRECTION_MODE = "BUY_ONLY"
                    mode_text = "BUY_ONLY — dynamic 1H one-direction system OFF"
                else:
                    ONE_DIRECTION_MODE = "BOTH"
                    mode_text = "BOTH — BUY and SELL allowed; one-direction restriction OFF"
                await send_reply(
                    f"🎛️ *A/B DIRECTION MODE UPDATED*\n\n"
                    f"Mode: *{ONE_DIRECTION_MODE}*\n"
                    f"{mode_text}\n\n"
                    f"Applies to *Strategy A + B*. Strategy C keeps its own extreme-frequency engine."
                )

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
                    bias_val, bias_sep_val = compute_1h_directional_bias(cached_1h["df"])
                    ranging_now, ranging_sep_val = compute_ranging_regime(cached_1h["df"])
                    bias_icon = {"BULLISH": "\U0001f7e2\U0001f4c8", "BEARISH": "\U0001f534\U0001f4c9", "NEUTRAL": "\u26aa"}.get(bias_val, "\u26aa")
                    if ONE_DIRECTION_MODE == "BUY_ONLY":
                        bias_note = "Dynamic 1H system OFF — SELL blocked; BUY only"
                    elif ONE_DIRECTION_MODE == "DYNAMIC":
                        bias_note = {
                            "BULLISH": "SELL blocked on A/B this cycle",
                            "BEARISH": "BUY blocked on A/B this cycle",
                            "NEUTRAL": "Neither direction blocked",
                        }.get(bias_val, "")
                    else:
                        bias_note = "One-direction restriction OFF — BUY + SELL allowed"
                    ranging_note = f"\U0001f7e0 Near EMA200 (ranging zone) -- informational only, SL unaffected" if ranging_now else "\U0001f7e2 Trending -- outside ranging zone"
                    reply = (
                        f"{status_icon} *SYSTEM & BRIDGE STATUS*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\u2022 Trading State: *{'ACTIVE' if SYSTEM_TRADING_ENABLED else 'PAUSED (Kill-Switch)'}*\n"
                        f"\u2022 MT5 Bridge: *{conn_msg}*\n"
                        f"\u2022 Server Time: `{now_wib.strftime('%Y-%m-%d %H:%M:%S WIB')}`\n\n"
                        f"{bias_icon} *1H DIRECTIONAL BIAS (EMA{ONE_H_EMA_PERIOD}):* *{bias_val}* ({bias_sep_val:+.3f}% sep)\n"
                        f"\u2022 {bias_note}\n"
                        f"\u2022 {ranging_note} ({ranging_sep_val:.3f}% from EMA)\n\n"
                        f"{budget_icon} *TWELVEDATA API BUDGET:*\n"
                        f"\u2022 Used Today: *{_twelve_data_call_count}/{TWELVE_DATA_DAILY_LIMIT}* ({budget_pct:.0f}%) | Remaining: *{remaining}*\n"
                        f"  \u2514\u2500 5M: {_twelve_data_calls_by_tf['5min']} | 15M: {_twelve_data_calls_by_tf['15min']} | 1H: {_twelve_data_calls_by_tf['1h']}\n\n"
                         f"⚡ *C DATA SOURCE:* {'MT5 EA (fresh)' if _mt5_cache_fresh() else 'Twelve Data M5 fallback'}\n\n"
                        f"\U0001f4c8 *STRATEGY:*\n"
                        f"\u2022 A/B Direction Mode: *{ONE_DIRECTION_MODE}*\n"
                        f"\u2022 A/B Execution (5M): *EMA {(EXPERIMENTAL_EMA_FAST if bot_role == 'experimental' else EMA_TREND_FAST)}/{(EXPERIMENTAL_EMA_SLOW if bot_role == 'experimental' else EMA_TREND_SLOW)}*\n"
                        f"\u2022 Confluence (15M): *EMA {TREND_15M_EMA_FAST}/{TREND_15M_EMA_SLOW}* (derived locally from M5)\n"
                        f"\u2022 Strategy C: *EXTREME M5 EMA9/21 + VWAP + ATR* | Max {EXTREME_MAX_TRADES_PER_DAY}/day"
                    )
                else:
                    reply = "\U0001f534 *MT5 DISCONNECTED*\n\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\nThe server is running, but MT5 has not sent any pings since the last reboot."
                await send_reply(reply)

            elif bot_role == "control" and raw_text == "/pause":
                SYSTEM_TRADING_ENABLED = False
                await send_reply("\U0001f6d1 *EMERGENCY KILL SWITCH ACTIVATED*\nMarket scanner paused. Send `/resume` to reactivate.")

            elif bot_role == "control" and raw_text == "/resume":
                SYSTEM_TRADING_ENABLED = True
                await send_reply("\U0001f7e2 *AUTO-TRADING SYSTEM RESUMED*\nScanner loop is now active.")

            elif bot_role == "breakout" and raw_text == "/status":
                _extreme_roll_day(datetime.now(timezone.utc) + timedelta(hours=7))
                mt5_src = "MT5 EA" if _mt5_cache_fresh() else "Twelve Data fallback"
                reply = (
                    "⚡ *EXTREME M5 STRATEGY C STATUS*\n\n"
                    f"Engine: *EMA9/EMA21 + Session VWAP + ATR*\n"
                    f"Data source: *{mt5_src}*\n"
                    f"Signals today: *{_extreme_state['trades_today']}/{EXTREME_MAX_TRADES_PER_DAY}*\n"
                    f"Execution: *{BREAKOUT_EXECUTION_MODE}*\n"
                    f"MT5 feed cache: *{'FRESH' if _mt5_cache_fresh() else 'NOT FRESH'}*"
                )
                await send_reply(reply)

            elif bot_role == "breakout" and raw_text == "/stats":
                try:
                    conn=get_db_connection(); cur=conn.cursor()
                    cur.execute("""SELECT COUNT(*) FILTER (WHERE status='EXECUTED') AS executed, COUNT(*) FILTER (WHERE status='CANCELLED') AS cancelled, COUNT(*) FILTER (WHERE status='EXECUTED' AND (outcome LIKE 'WIN%%' OR outcome LIKE 'CLOSED%%')) AS wins, COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome LIKE 'LOSS%%') AS losses, COALESCE(SUM(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS total_r, COALESCE(AVG(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS avg_r FROM signals WHERE strategy=%s""",(BREAKOUT_STRATEGY,))
                    s=cur.fetchone(); cur.close(); conn.close(); ex=int(s['executed'] or 0); wins=int(s['wins'] or 0); losses=int(s['losses'] or 0)
                    await send_reply(f"⚡ *EXTREME M5 PERFORMANCE*\n━━━━━━━━━━━━━━━━━━━━\nExecuted: *{ex}*\nWins/Losses: *{wins}/{losses}*\nWin Rate: *{(wins/ex*100 if ex else 0):.1f}%*\nTotal R: *{float(s['total_r'] or 0):+.2f}R* | Avg R: *{float(s['avg_r'] or 0):+.3f}R*\nMode: *{BREAKOUT_EXECUTION_MODE}*\nEngine: *EMA9/21 + VWAP + ATR*")
                except Exception as e: await send_reply(f"⚠️ Error querying breakout stats: {e}")

            elif bot_role == "breakout" and raw_text == "/last":
                try:
                    conn=get_db_connection(); cur=conn.cursor(); cur.execute("SELECT id,action,trigger_type,entry_price,outcome,created_at FROM signals WHERE strategy=%s ORDER BY id DESC LIMIT 10",(BREAKOUT_STRATEGY,)); rows=cur.fetchall(); cur.close(); conn.close()
                    if not rows: reply="📋 *LAST EXTREME M5 SIGNALS*\n\n_No Extreme M5 signals yet._"
                    else: reply="📋 *LAST EXTREME M5 SIGNALS*\n\n"+"\n".join(f"#{r['id']} | {r['action']} | {r['trigger_type']} | ${float(r['entry_price'] or 0):.2f} | {r['outcome'] or 'N/A'}" for r in rows)
                    await send_reply(reply)
                except Exception as e: await send_reply(f"⚠️ Error querying breakout logs: {e}")

            elif bot_role == "control" and raw_text == "/stats":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) AS total FROM signals WHERE status = 'EXECUTED' AND strategy = %s", (CONTROL_STRATEGY,))
                    total_executed = cur.fetchone()["total"] or 0
                    cur.execute("SELECT COUNT(*) AS vetoes FROM signals WHERE status = 'VETOED' AND strategy = %s", (CONTROL_STRATEGY,))
                    total_vetoes = cur.fetchone()["vetoes"] or 0
                    cur.execute("SELECT COUNT(*) AS pending FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING' AND strategy = %s", (CONTROL_STRATEGY,))
                    total_pending = cur.fetchone()["pending"] or 0
                    cur.execute("SELECT COUNT(*) AS tp1_wins FROM signals WHERE strategy = %s AND (outcome LIKE 'WIN (TP1%%' OR outcome LIKE 'CLOSED%%')", (CONTROL_STRATEGY,))
                    tp1_wins = cur.fetchone()["tp1_wins"] or 0
                    cur.execute("SELECT COUNT(*) AS tp2_wins FROM signals WHERE strategy = %s AND outcome LIKE 'WIN (TP2%%'", (CONTROL_STRATEGY,))
                    tp2_wins = cur.fetchone()["tp2_wins"] or 0
                    cur.execute("SELECT COUNT(*) AS losses FROM signals WHERE strategy = %s AND outcome LIKE 'LOSS%%'", (CONTROL_STRATEGY,))
                    losses = cur.fetchone()["losses"] or 0

                    cur.execute("SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price, COALESCE(outcome, 'PENDING') AS outcome_val FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s", (CONTROL_STRATEGY,))
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
                        f"\u2022 Net Pips (2x0.01 lot): *{total_pips:+.1f} pips*\n"
                        f"\u2022 Net Profit (2x0.01 Lot, actual MT5 exposure): *${est_dollar:+.2f}*\n\n"
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
                    await send_reply(reply)
                except Exception as e: await send_reply(f"\u26a0\ufe0f Error querying stats: {e}")

            elif (bot_role == "experimental" and raw_text in ("/stats", "/compare")):
                try:
                    conn = get_db_connection(); cur = conn.cursor()
                    cur.execute("""
                        SELECT strategy, execution_mode,
                               COUNT(*) FILTER (WHERE status='EXECUTED') AS executed,
                               COUNT(*) FILTER (WHERE status='VETOED') AS vetoed,
                               COUNT(*) FILTER (WHERE status='EXECUTED' AND (outcome LIKE 'WIN%%' OR outcome LIKE 'CLOSED%%')) AS wins,
                               COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome LIKE 'LOSS%%') AS losses,
                               COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome='PENDING') AS pending,
                               COALESCE(SUM(result_pips) FILTER (WHERE status='EXECUTED' AND result_pips IS NOT NULL),0) AS net_pips,
                               COALESCE(SUM(result_usd) FILTER (WHERE status='EXECUTED' AND result_usd IS NOT NULL),0) AS net_usd,
                               COALESCE(SUM(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS total_r,
                               COALESCE(AVG(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS avg_r
                        FROM signals
                        WHERE strategy IN (%s, %s, %s)
                        GROUP BY strategy, execution_mode
                        ORDER BY strategy;
                    """, (CONTROL_STRATEGY, EXPERIMENTAL_STRATEGY, BREAKOUT_STRATEGY))
                    rows = cur.fetchall()
                    stats = {}
                    for r in rows:
                        executed = int(r["executed"] or 0); wins = int(r["wins"] or 0); losses = int(r["losses"] or 0)
                        net_pips = float(r["net_pips"] or 0); net_usd = float(r["net_usd"] or 0)
                        total_r = float(r["total_r"] or 0); avg_r = float(r["avg_r"] or 0)
                        # Profit factor from stored result_pips, falling back to R when needed.
                        cur.execute("""
                            SELECT COALESCE(SUM(result_pips) FILTER (WHERE result_pips > 0),0) AS gross_win,
                                   COALESCE(SUM(ABS(result_pips)) FILTER (WHERE result_pips < 0),0) AS gross_loss
                            FROM signals WHERE status='EXECUTED' AND strategy=%s AND result_pips IS NOT NULL
                        """, (r["strategy"],))
                        pfrow = cur.fetchone(); gross_win = float(pfrow["gross_win"] or 0); gross_loss = float(pfrow["gross_loss"] or 0)
                        pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
                        stats[str(r["strategy"])] = {
                            "mode": str(r["execution_mode"]), "executed": executed, "vetoed": int(r["vetoed"] or 0),
                            "wins": wins, "losses": losses, "pending": int(r["pending"] or 0),
                            "wr": (wins / executed * 100) if executed else 0.0, "pips": net_pips,
                            "usd": net_usd, "total_r": total_r, "avg_r": avg_r, "pf": pf
                        }

                    # Calculate current maximum consecutive SL streak per strategy.
                    for strategy in (CONTROL_STRATEGY, EXPERIMENTAL_STRATEGY, BREAKOUT_STRATEGY):
                        cur.execute("""
                            SELECT outcome FROM signals
                            WHERE status='EXECUTED' AND strategy=%s AND outcome IS NOT NULL
                            ORDER BY id ASC
                        """, (strategy,))
                        streak = best = 0
                        for rr in cur.fetchall():
                            if str(rr["outcome"]).startswith("LOSS"):
                                streak += 1; best = max(best, streak)
                            else:
                                streak = 0
                        stats.setdefault(strategy, {})["max_loss_streak"] = best

                    cur.close(); conn.close()
                    a = stats.get(CONTROL_STRATEGY, {})
                    b = stats.get(EXPERIMENTAL_STRATEGY, {})
                    c = stats.get(BREAKOUT_STRATEGY, {})
                    leader = "Not enough data"
                    if a.get("executed", 0) or b.get("executed", 0) or c.get("executed", 0):
                        candidates = [
                            ("🟢 EMA 5/9 (CONTROL)", a.get("total_r", 0)),
                            ("🔵 EMA 5/15 (EXPERIMENT)", b.get("total_r", 0)),
                            ("🟠 Range Breakout (C)", c.get("total_r", 0)),
                        ]
                        best_label, best_r = max(candidates, key=lambda x: x[1])
                        tied = [lbl for lbl, val in candidates if val == best_r]
                        leader = best_label if len(tied) == 1 else "🤝 Tied"

                    def block(label, d):
                        if not d:
                            return f"{label}\nNo data yet."
                        return (
                            f"{label} — *{d.get('mode','UNKNOWN')}*\n"
                            f"• Executed: *{d.get('executed',0)}* | Vetoed: *{d.get('vetoed',0)}*\n"
                            f"• Wins/Losses: *{d.get('wins',0)}/{d.get('losses',0)}* | Pending: *{d.get('pending',0)}*\n"
                            f"• Win Rate: *{d.get('wr',0):.1f}%*\n"
                            f"• Net Pips: *{d.get('pips',0):+.1f}*\n"
                            f"• Net USD: *${d.get('usd',0):+.2f}*\n"
                            f"• Total R: *{d.get('total_r',0):+.2f}R* | Avg R: *{d.get('avg_r',0):+.3f}R*\n"
                            f"• Profit Factor: *{d.get('pf',0):.2f}*\n"
                            f"• Max SL Streak: *{d.get('max_loss_streak',0)}*"
                        )

                    reply = (
                        "🔬 *A/B/C STRATEGY DASHBOARD*\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "XAU/USD • Same market snapshot • Same risk framework\n\n"
                        f"🟢 *A — CONTROL (EMA 5/9)*\n{block('', a)}\n\n"
                        f"🔵 *B — EXPERIMENT (EMA 5/15)*\n{block('', b)}\n\n"
                        f"⚡ *C — EXTREME M5 IMPULSE SCALPER*\n{block('', c)}\n\n"
                        f"🏆 *CURRENT LEADER:* {leader}\n"
                        "\n_Compare again after more trades; early samples are not statistically meaningful._"
                    )
                    await send_reply(reply)
                except Exception as e:
                    await send_reply(f"⚠️ Error querying A/B/C dashboard: {e}")

            elif bot_role == "control" and raw_text == "/pips":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p,
                               exit_price, COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s
                    """, (CONTROL_STRATEGY,))
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
                        f"\U0001f4a1 *Note:* Reflects your actual 2x0.01 lot execution -- "
                        f"SL (before TP1) = both lots @ SL, TP1/BE = lot1 @ TP1 + lot2 @ BE, "
                        f"TP2 = lot1 @ TP1 + lot2 @ TP2."
                    )
                    await send_reply(reply)
                except Exception as e:
                    await send_reply(f"\u26a0\ufe0f Error calculating pips: {e}")

            elif ((bot_role == "control" and raw_text == "/logs") or (bot_role == "experimental" and raw_text == "/last")):
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p,
                               COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                               COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val,
                               COALESCE(timestamp, created_at::text, 'N/A') AS log_time
                        FROM signals WHERE status = 'EXECUTED' AND strategy = %s
                        ORDER BY id DESC LIMIT 10
                    """, (CONTROL_STRATEGY if bot_role == 'control' else EXPERIMENTAL_STRATEGY,))
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
                    await send_reply(reply)
                except Exception as e:
                    await send_reply(f"\u26a0\ufe0f Error querying logs: {e}")

            elif bot_role == "control" and raw_text == "/analyze":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p,
                               COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                               COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val, adx_15m,
                               COALESCE(timestamp, created_at::text, '') AS log_time,
                               trend_15m, entry_extension_atr, regime
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s
                    """, (CONTROL_STRATEGY,))
                    rows = cur.fetchall()
                    cur.close(); conn.close()

                    if not rows:
                        reply = (
                            "\U0001f4d0 *STRATEGY FORWARD-TEST ANALYSIS*\n\n"
                            "_Not enough closed trades yet to analyze. Check back after more signals complete._"
                        )
                    else:
                        segments = {"Strategy": {}, "5M ADX Regime": {}, "V10 Regime": {}, "Entry Extension": {}, "Session": {}, "15m Confluence": {}}
                        overall_r = []
                        for r in rows:
                            r_mult = compute_r_multiple(
                                r["action"], float(r["entry_p"]), float(r["exit_price"]), float(r["sl_p"]),
                                float(r["tp1_p"]), float(r["tp2_p"]), r["outcome_val"]
                            )
                            overall_r.append(r_mult)
                            adx_val = float(r["adx_15m"]) if r["adx_15m"] is not None else 0.0
                            segments["Strategy"].setdefault(bucket_strategy(r["trigger_type"]), []).append(r_mult)
                            segments["5M ADX Regime"].setdefault(bucket_adx(adx_val), []).append(r_mult)
                            segments["V10 Regime"].setdefault(r["regime"] or "Pre-V10 (unlabeled)", []).append(r_mult)
                            segments["Entry Extension"].setdefault(bucket_extension(r["entry_extension_atr"]), []).append(r_mult)
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
                        for dim in ["Strategy", "5M ADX Regime", "V10 Regime", "Entry Extension", "Session", "15m Confluence"]:
                            reply_parts.append(format_performance_segment(dim, segments[dim]))
                            reply_parts.append("")

                        reply_parts.append("\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015")
                        reply_parts.append(
                            "\U0001f4a1 Segments need n\u22658 to be flagged \u26a0\ufe0f/\u2705 (smaller samples are shown "
                            "but noisy). Check the Strategy breakdown for \"Range Fade (Consolidation)\" vs the "
                            "EMA buckets to see which regime is actually working.\n\n"
                            f"\u2696\ufe0f Mode is auto-selected by 5M ADX (\u2265{RANGE_MODE_ADX_MAX:.0f} trend, "
                            f"<{RANGE_MODE_ADX_MAX:.0f} range) -- not a veto, both regimes are live.\n"
                            f"\u23f1\ufe0f Loss cooldown ({LOSS_COOLDOWN_MINUTES} min, any direction) is also active."
                        )
                        reply = "\n".join(reply_parts)
                    await send_reply(reply)
                except Exception as e:
                    await send_reply(f"\u26a0\ufe0f Error: {e}")

    except Exception as e: logging.error(f"[WEBHOOK ERROR] {e}")
    return {"status": "ok"}


# =====================================================================
# SEPARATE TELEGRAM WEBHOOKS — CONTROL vs EXPERIMENTAL
# =====================================================================
@app.post("/telegram-webhook")
async def telegram_webhook_control(request: Request):
    return await _handle_telegram_webhook(request, "control")

@app.post("/telegram-webhook-b")
async def telegram_webhook_experimental(request: Request):
    return await _handle_telegram_webhook(request, "experimental")

@app.post("/telegram-webhook-c")
async def telegram_webhook_breakout(request: Request):
    return await _handle_telegram_webhook(request, "breakout")
