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

# --- PULLBACK STATE (Stage 2 of the liquidity sweep strategy) ---
# In-memory pending setup: a sweep+BOS was detected and we are waiting for
# price to retrace into the fib zone and reject before actually entering.
# This lives in process memory (single background task), so it resets on redeploy.
pending_setup = None
PULLBACK_EXPIRY_MINUTES = 10  # widened from 6: more time for a valid pullback to land before the setup is discarded
FIB_RETRACE_MIN = 0.382
FIB_RETRACE_MAX = 0.618
LEVEL_LOOKBACK_CANDLES = 4  # narrowed from 5: fresher 5M liquidity levels reset more often -> more sweep/breakout opportunities

# --- STAT-BASED VETOES (added from /analyze forward-test results, 69-trade sample) ---
# These are hard, non-AI vetoes based on segments that were flagged as underperforming
# with a large-enough sample (n>=15) to be trusted:
#   - ADX 50+:            n=15, WR=27%, AvgR=-0.27  (blow-off/exhaustion regime, not "more trend = better")
#   - Mid session 14-18 WIB: n=17, WR=29%, AvgR=-0.18
# Re-check with /analyze after another ~30-40 trades post-veto to confirm these
# actually lift overall AvgR rather than just cutting volume. If they do, this
# is where to also consider retuning the AI veto rules in analyze_signal_with_ai
# (currently the breakout ADX veto only fires above 70.0, well past this line).
STAT_VETO_ADX_THRESHOLD = 50.0
STAT_VETO_MID_SESSION_START_HOUR = 14  # WIB
STAT_VETO_MID_SESSION_END_HOUR = 18    # WIB

# --- LOSS-COOLDOWN (any direction) ---
# Added after inspecting the real trade list: losing sequences (e.g. BUY loss ->
# BUY loss -> SELL loss -> BUY loss) tended to cluster within 30-60 minutes of
# each other, typical of choppy microstructure rather than independent setups.
# This is direction-agnostic and separate from the existing same-direction/price-
# distance cooldown below -- it fires even on a SELL signal right after a BUY loss.
LOSS_COOLDOWN_MINUTES = 10


def check_stat_veto(adx_5m: float, current_hour_wib: int):
    """
    Hard pre-AI veto based on forward-tested underperforming segments.
    Returns (is_vetoed: bool, reason: str). Checked before calling the AI
    analyst so we also skip the API call entirely on a stat-veto.
    """
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
                        # BLENDED EXIT (bugfix): this outcome means half the conceptual position
                        # banked TP1 and the other half rode the runner back to breakeven (0R).
                        # Recording exit_price = tp1 credited the FULL TP1 gain to the whole
                        # position, overstating every downstream pips/R/PF calculation in
                        # /stats, /pips, and /analyze. The blended price below is the midpoint
                        # between entry and TP1, so (exit_price - entry_price) correctly equals
                        # 0.5R at TP1 + 0.5R at breakeven = 0.5x the full TP1 move.
                        exit_price = entry_price + 0.5 * (tp1 - entry_price)
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
                        # BLENDED EXIT (bugfix): same reasoning as the BUY branch above, mirrored
                        # for a short position (TP1 is below entry for a SELL).
                        exit_price = entry_price - 0.5 * (entry_price - tp1)
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

# --- INDICATOR CALCULATIONS (M1): ATR only (VWAP bands no longer used for entries) ---
def calculate_metrics_m1(df: pd.DataFrame):
    df = df.tail(100).copy()

    # ATR Calculation on M1 (used for volatility-based SL/TP sizing)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
    )
    df["atr"] = df["tr"].rolling(window=14).mean()
    return df

def calculate_metrics_5m(df: pd.DataFrame):
    df = df.tail(100).copy()
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
    df["atr"] = df["tr"].rolling(window=14).mean()
    return df

# --- STRATEGY: LIQUIDITY SWEEP + STRUCTURE BREAK (5M liquidity/structure, 1M trigger) ---
def detect_liquidity_sweep_structure(df_1m: pd.DataFrame, df_5m: pd.DataFrame):
    """
    Looks at the last completed 5 x 5M candles for the swept liquidity level,
    then checks the last two 1M candles for a sweep-and-reject + break of structure.

    Returns: (action, trigger_type, recent_high, recent_low)
    """
    recent_high = float(df_5m["high"].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].max())
    recent_low = float(df_5m["low"].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].min())

    prev = df_1m.iloc[-2]
    curr = df_1m.iloc[-1]

    # SELL setup: price sweeps above the recent 5M high, then closes back below it
    sweep_high = bool(prev["high"] > recent_high and curr["close"] < recent_high)
    # BUY setup: price sweeps below the recent 5M low, then closes back above it
    sweep_low = bool(prev["low"] < recent_low and curr["close"] > recent_low)

    # Structure confirmation on the 1M candle
    bearish_bos = bool(curr["close"] < prev["low"])
    bullish_bos = bool(curr["close"] > prev["high"])

    if sweep_high and bearish_bos:
        return "SELL", "Liquidity Sweep + Bearish BOS", recent_high, recent_low
    if sweep_low and bullish_bos:
        return "BUY", "Liquidity Sweep + Bullish BOS", recent_high, recent_low
    return "HOLD", "No setup", recent_high, recent_low

# --- STRATEGY: BREAKOUT CONTINUATION (trend-following, sits alongside the sweep strategy) ---
def detect_breakout_continuation(df_1m: pd.DataFrame, df_5m: pd.DataFrame):
    """
    Detects a clean momentum breakout of a recent 5M high/low that is NOT a
    sweep-and-reject: both the previous AND current 1M candle close beyond the
    level, and the current close extends further than the previous one. That
    sustained follow-through (rather than a snap-back) is what separates a
    continuation trade from a liquidity sweep fakeout.

    Returns: (action, trigger_type, recent_high, recent_low)
    """
    recent_high = float(df_5m["high"].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].max())
    recent_low = float(df_5m["low"].iloc[-(LEVEL_LOOKBACK_CANDLES + 1):-1].min())

    prev = df_1m.iloc[-2]
    curr = df_1m.iloc[-1]

    bullish_breakout = bool(prev["close"] > recent_high and curr["close"] > recent_high and curr["close"] > prev["close"])
    bearish_breakout = bool(prev["close"] < recent_low and curr["close"] < recent_low and curr["close"] < prev["close"])

    if bullish_breakout:
        return "BUY", "Breakout Continuation (Bullish)", recent_high, recent_low
    if bearish_breakout:
        return "SELL", "Breakout Continuation (Bearish)", recent_high, recent_low
    return "HOLD", "No setup", recent_high, recent_low

# --- STRATEGY: CONSOLIDATION BRACKET BREAKOUT (squeeze first, then a decisive break of it) ---
CONSOLIDATION_LOOKBACK_5M = 8       # 5M candles used to define the bracket (40 min of range)
CONSOLIDATION_MAX_RANGE_ATR_MULT = 1.5  # bracket must be tighter than 1.5x the 5M ATR to count as a real squeeze
BREAKOUT_MIN_BODY_ATR_MULT = 0.5    # the breaking 1M candle's body must be a real push, not a doji poke

def detect_consolidation_breakout(df_1m: pd.DataFrame, df_5m: pd.DataFrame):
    """
    Unlike detect_breakout_continuation (which just breaks a recent high/low),
    this requires price to have actually been RANGING first: the high-low range
    over the last CONSOLIDATION_LOOKBACK_5M candles must be compressed relative
    to average volatility (a "bracket"/squeeze), and only THEN does a decisive,
    strong-bodied close outside that bracket count as a breakout signal.

    Returns: (action, trigger_type, bracket_high, bracket_low)
    """
    if len(df_5m) < CONSOLIDATION_LOOKBACK_5M + 1 or "atr" not in df_5m.columns:
        return "HOLD", "No setup", 0.0, 0.0

    bracket = df_5m.iloc[-(CONSOLIDATION_LOOKBACK_5M + 1):-1]
    bracket_high = float(bracket["high"].max())
    bracket_low = float(bracket["low"].min())
    bracket_range = bracket_high - bracket_low

    atr_5m = df_5m["atr"].iloc[-1]
    if pd.isna(atr_5m) or atr_5m <= 0:
        return "HOLD", "No setup", bracket_high, bracket_low

    is_consolidating = bracket_range <= (CONSOLIDATION_MAX_RANGE_ATR_MULT * atr_5m)
    if not is_consolidating:
        return "HOLD", "No setup", bracket_high, bracket_low

    curr = df_1m.iloc[-1]
    candle_body = abs(float(curr["close"]) - float(curr["open"]))
    atr_1m = df_1m["atr"].iloc[-1] if "atr" in df_1m.columns else None
    strong_candle = bool(candle_body >= (BREAKOUT_MIN_BODY_ATR_MULT * atr_1m)) if atr_1m and not pd.isna(atr_1m) else True

    bullish_break = bool(curr["close"] > bracket_high and curr["close"] > curr["open"] and strong_candle)
    bearish_break = bool(curr["close"] < bracket_low and curr["close"] < curr["open"] and strong_candle)

    if bullish_break:
        return "BUY", "Consolidation Bracket Breakout (Bullish)", bracket_high, bracket_low
    if bearish_break:
        return "SELL", "Consolidation Bracket Breakout (Bearish)", bracket_high, bracket_low
    return "HOLD", "No setup", bracket_high, bracket_low

# --- FORWARD-TEST ANALYTICS: turns closed trades into segment-level performance stats ---
def compute_r_multiple(action: str, entry: float, exit_price: float, sl: float) -> float:
    """
    R multiple = profit/loss expressed in units of the original risk (entry-to-SL distance).
    +1.0R means the trade made exactly as much as it risked; -1.0R is a full stop-out.
    """
    risk_dist = abs(entry - sl)
    if risk_dist <= 0:
        return 0.0
    if action == "BUY":
        return (exit_price - entry) / risk_dist
    return (entry - exit_price) / risk_dist

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
        hour = int(str(timestamp_str).split(" ")[1].split(":")[0])
    except Exception:
        return "Unknown"
    if 9 <= hour < 14:
        return "Early (09-14 WIB)"
    if 14 <= hour < 18:
        return "Mid (14-18 WIB)"
    if 18 <= hour < 22:
        return "Late (18-22 WIB)"
    return "Outside session"

def format_performance_segment(dim_name: str, buckets: dict, min_sample_to_flag: int = 8) -> str:
    lines = [f"*{dim_name}:*"]
    for label, r_values in sorted(buckets.items(), key=lambda item: -len(item[1])):
        n = len(r_values)
        wins = sum(1 for r in r_values if r > 0)
        win_rate = (wins / n * 100) if n else 0.0
        avg_r = (sum(r_values) / n) if n else 0.0
        flag = ""
        if n >= min_sample_to_flag:
            if win_rate < 35:
                flag = " ⚠️ underperforming"
            elif win_rate > 65:
                flag = " ✅ strong"
        lines.append(f"• {label}: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}")
    return "\n".join(lines)

def compute_pullback_zone(action: str, swing_high: float, swing_low: float):
    """
    Given the impulse leg of a sweep+BOS move, returns the (zone_lower, zone_upper)
    fib retracement band price should pull back into before we enter.
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return None, None
    if action == "BUY":
        # Impulse moved up from swing_low to swing_high; pullback retraces down into the zone.
        zone_upper = swing_high - (FIB_RETRACE_MIN * rng)
        zone_lower = swing_high - (FIB_RETRACE_MAX * rng)
    else:
        # Impulse moved down from swing_high to swing_low; pullback retraces up into the zone.
        zone_lower = swing_low + (FIB_RETRACE_MIN * rng)
        zone_upper = swing_low + (FIB_RETRACE_MAX * rng)
    return float(zone_lower), float(zone_upper)

def check_pullback_entry(action: str, zone_lower: float, zone_upper: float, curr_candle) -> bool:
    """
    True if the current 1M candle's range overlaps the fib pullback zone
    AND the candle closes back in the direction of the original trade (a rejection candle).
    """
    candle_low = float(curr_candle["low"])
    candle_high = float(curr_candle["high"])
    candle_open = float(curr_candle["open"])
    candle_close = float(curr_candle["close"])

    overlaps_zone = candle_high >= zone_lower and candle_low <= zone_upper
    if not overlaps_zone:
        return False

    if action == "BUY":
        return candle_close > candle_open
    else:
        return candle_close < candle_open

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
async def analyze_signal_with_ai(proposed_action: str, trigger_type: str, current_price: float, df_1m: pd.DataFrame, df_5m: pd.DataFrame, recent_high: float, recent_low: float):
    is_breakout = "Breakout" in trigger_type

    if is_breakout:
        strategy_label = "Breakout Continuation (trend-following momentum entry)"
        veto_rules_text = (
            "- VETO if 5M ADX is below 20.0 (trend too weak to sustain a breakout; high risk of a false break/fakeout).\n"
            "- VETO if 5M ADX is above 70.0 (parabolic/overextended move, high risk of an imminent sharp pullback)."
        )
    else:
        strategy_label = "Liquidity Sweep + Break-of-Structure (reversal entry)"
        veto_rules_text = (
            "- VETO if 5M ADX is above 45.0 (indicating an extreme unstoppable runaway trend that destroys mean-reversion setups).\n"
            "- VETO if 5M ADX is below 12.0 (indicating dead market liquidity)."
        )

    prompt = f"""
Act as a Senior Institutional Risk Manager for Spot Gold (XAU/USD) intraday scalping.
Strategy in play: {strategy_label}.
A technical scalp trigger ({trigger_type}) suggests a {proposed_action} entry at ${current_price:.2f}.

TECHNICAL CONTEXT:
1. 1-Minute: Close=${float(df_1m['close'].iloc[-1]):.2f}, Prior Candle High=${float(df_1m['high'].iloc[-2]):.2f}, Prior Candle Low=${float(df_1m['low'].iloc[-2]):.2f}.
2. 5-Minute Reference Levels: Recent High=${recent_high:.2f}, Recent Low=${recent_low:.2f}.
3. 5-Minute: ADX Trend Strength={float(df_5m['adx'].iloc[-1]):.1f}.

CRITICAL SCALP VETO RULES:
{veto_rules_text}

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

    return SignalOutput(action=proposed_action, confidence=0.7, reasoning="Fallback: Executed on pure liquidity sweep + structure break alignment.")

# --- BACKGROUND SCANNING LOOP (LIQUIDITY SWEEP + STRUCTURE BREAK) ---
async def background_scanning_loop():
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        while True:
            try:
                now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
                current_hour_wib = now_wib.hour

                # SESSION WINDOWING: 10:00-22:00 WIB (12h active window).
                # At 2 calls/cycle every 120s, 12 active hours = 720 calls/day = 90% of the
                # Twelve Data free-tier 800/day cap, leaving ~80 calls/day (10%) spare for
                # manual testing/redeploys without risking a rate-limit hit.
                active_session = (10 <= current_hour_wib < 22)

                if not active_session:
                    logging.info(f"[SLEEP MODE] WIB Time: {now_wib.strftime('%H:%M')} | Outside active trading session. Pausing API calls...")
                    await asyncio.sleep(300)
                    continue

                logging.info(f"[ACTIVE SCAN] WIB Time: {now_wib.strftime('%H:%M')} | Scanning for Liquidity Sweep + Structure Break...")
                df_1m = await fetch_timeframe_data(client, "1min")
                df_5m = await fetch_timeframe_data(client, "5min")

                if df_1m is None or df_5m is None:
                    logging.warning("Failed to fetch M1/M5 candles. Retrying next cycle.")
                    await asyncio.sleep(120)
                    continue

                if len(df_1m) < 3 or len(df_5m) < 6:
                    logging.warning("Insufficient candle history for sweep/BOS detection. Retrying next cycle.")
                    await asyncio.sleep(120)
                    continue

                df_1m = calculate_metrics_m1(df_1m)
                df_5m = calculate_metrics_5m(df_5m)

                curr_high = float(df_1m["high"].tail(3).max())
                curr_low = float(df_1m["low"].tail(3).min())
                update_open_trades(curr_high, curr_low)

                curr_price = float(df_1m["close"].iloc[-1])
                adx_5m = float(df_5m["adx"].iloc[-1]) if not pd.isna(df_5m["adx"].iloc[-1]) else 0.0

                global pending_setup
                proposed_action = "HOLD"
                trigger_type = "None"
                recent_high = 0.0
                recent_low = 0.0
                status_note = ""

                # STAGE 2: If a setup is already pending, check for pullback confirmation first.
                if pending_setup is not None:
                    if datetime.now(timezone.utc) > pending_setup["expires"]:
                        logging.info(f"[PULLBACK EXPIRED] {pending_setup['action']} setup from ${pending_setup['trigger_price']:.2f} expired without a pullback entry.")
                        pending_setup = None
                    else:
                        curr_candle = df_1m.iloc[-1]
                        confirmed = check_pullback_entry(
                            pending_setup["action"], pending_setup["zone_lower"], pending_setup["zone_upper"], curr_candle
                        )
                        recent_high = pending_setup["recent_high"]
                        recent_low = pending_setup["recent_low"]
                        if confirmed:
                            proposed_action = pending_setup["action"]
                            trigger_type = pending_setup["trigger_type"] + " + Pullback Confirmed"
                            pending_setup = None
                        else:
                            # No pullback yet -- check if price is breaking out strongly in either
                            # direction while we wait.
                            b_action, b_trigger_type, b_recent_high, b_recent_low = detect_breakout_continuation(df_1m, df_5m)
                            if b_action != "HOLD" and adx_5m >= 20.0:
                                if b_action == pending_setup["action"]:
                                    # SAME direction: price kept running straight through instead of
                                    # pulling back. The reversal-via-pullback thesis isn't going to
                                    # happen; convert this into an immediate breakout continuation
                                    # entry instead of leaving it stuck waiting and missing the move.
                                    proposed_action = b_action
                                    trigger_type = f"No Pullback -> Converted to {b_trigger_type}"
                                    recent_high, recent_low = b_recent_high, b_recent_low
                                    pending_setup = None
                                else:
                                    # OPPOSITE direction: a real breakout is running against the
                                    # pending reversal setup (e.g. a SELL sweep+BOS armed mid-uptrend
                                    # while price keeps making new highs). Waiting up to
                                    # PULLBACK_EXPIRY_MINUTES for a pullback that contradicts the
                                    # live trend just means the bot sits blind through the move -- and
                                    # once the setup expires, Stage 1 can immediately re-arm another
                                    # setup in the same losing direction. Cancel the stale setup now
                                    # and take the real breakout instead.
                                    invalidated_action = pending_setup["action"]
                                    invalidated_trigger = pending_setup["trigger_type"]
                                    logging.info(
                                        f"[SETUP INVALIDATED] {invalidated_action} setup ({invalidated_trigger}) "
                                        f"from ${pending_setup['trigger_price']:.2f} invalidated by opposite "
                                        f"{b_action} breakout continuation."
                                    )
                                    proposed_action = b_action
                                    trigger_type = f"{b_trigger_type} (Invalidated {invalidated_action} Sweep Setup)"
                                    recent_high, recent_low = b_recent_high, b_recent_low
                                    pending_setup = None
                            else:
                                status_note = f" | Awaiting pullback into ${pending_setup['zone_lower']:.2f}-${pending_setup['zone_upper']:.2f}"

                # STAGE 1: No pending setup and nothing just confirmed -> look for a fresh sweep+BOS.
                if pending_setup is None and proposed_action == "HOLD" and not status_note:
                    new_action, new_trigger_type, new_recent_high, new_recent_low = detect_liquidity_sweep_structure(df_1m, df_5m)
                    recent_high, recent_low = new_recent_high, new_recent_low

                    if new_action != "HOLD":
                        prev_candle = df_1m.iloc[-2]
                        curr_candle = df_1m.iloc[-1]
                        if new_action == "BUY":
                            swing_low = float(prev_candle["low"])
                            swing_high = float(curr_candle["close"])
                        else:
                            swing_high = float(prev_candle["high"])
                            swing_low = float(curr_candle["close"])

                        zone_lower, zone_upper = compute_pullback_zone(new_action, swing_high, swing_low)
                        if zone_lower is not None:
                            pending_setup = {
                                "action": new_action,
                                "trigger_type": new_trigger_type,
                                "trigger_price": float(curr_price),
                                "recent_high": new_recent_high,
                                "recent_low": new_recent_low,
                                "zone_lower": zone_lower,
                                "zone_upper": zone_upper,
                                "expires": datetime.now(timezone.utc) + timedelta(minutes=PULLBACK_EXPIRY_MINUTES),
                            }
                            status_note = f" | New {new_action} setup ({new_trigger_type}) armed, awaiting pullback into ${zone_lower:.2f}-${zone_upper:.2f}"

                # STAGE 1b: No sweep setup found -> check for a consolidation bracket breakout.
                # This is the more selective breakout flavor: it requires price to have actually
                # been ranging tightly first (a real squeeze), so it's checked before the looser
                # momentum-breakout fallback below.
                if pending_setup is None and proposed_action == "HOLD" and not status_note:
                    c_action, c_trigger_type, c_bracket_high, c_bracket_low = detect_consolidation_breakout(df_1m, df_5m)
                    if c_action != "HOLD":
                        if adx_5m >= 20.0:
                            proposed_action = c_action
                            trigger_type = c_trigger_type
                            recent_high, recent_low = c_bracket_high, c_bracket_low
                        else:
                            status_note = f" | Bracket breakout seen but 5M ADX {adx_5m:.1f} too weak to confirm"

                # STAGE 1c: No sweep or bracket breakout -> check for a plain momentum breakout continuation.
                # This is a trend-following entry (not a reversal), so it enters immediately rather
                # than waiting for a pullback -- a strong continuation move may not retrace at all.
                if pending_setup is None and proposed_action == "HOLD" and not status_note:
                    b_action, b_trigger_type, b_recent_high, b_recent_low = detect_breakout_continuation(df_1m, df_5m)
                    if b_action != "HOLD":
                        if adx_5m >= 20.0:
                            proposed_action = b_action
                            trigger_type = b_trigger_type
                            recent_high, recent_low = b_recent_high, b_recent_low
                        else:
                            status_note = f" | Breakout seen but 5M ADX {adx_5m:.1f} too weak to confirm continuation"

                # --- STAT VETO CHECK (new) ---
                # Hard veto based on the /analyze forward-test results, checked BEFORE the AI
                # call so we also skip the (paid) AI request entirely on a stat-veto. Applies
                # regardless of which strategy fired, since both flagged segments (ADX 50+ and
                # Mid session) were underperforming across the whole trade sample, not just one
                # strategy.
                if proposed_action != "HOLD":
                    stat_vetoed, stat_veto_reason = check_stat_veto(adx_5m, current_hour_wib)
                    if stat_vetoed:
                        logging.info(f"[STAT VETO] {stat_veto_reason}")
                        log_trade_signal(
                            "VETOED", proposed_action, f"{trigger_type} [STAT-VETO]",
                            curr_price, 0.0, 0.0, 0.0, 1.0, adx_5m, 0.0, "None", stat_veto_reason
                        )
                        proposed_action = "HOLD"

                # Loss-cooldown Check (any direction, {LOSS_COOLDOWN_MINUTES} min after any SL hit)
                if proposed_action != "HOLD":
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT outcome_timestamp FROM signals
                            WHERE status = 'EXECUTED' AND outcome = 'LOSS (SL HIT)' AND outcome_timestamp IS NOT NULL AND outcome_timestamp != ''
                            ORDER BY id DESC LIMIT 1
                        """)
                        last_loss = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if last_loss and last_loss.get('outcome_timestamp'):
                            # outcome_timestamp is stored as "YYYY-MM-DD HH:MM:SS WIB"
                            ts_str = str(last_loss['outcome_timestamp']).replace(" WIB", "")
                            last_loss_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                            minutes_since_loss = (now_wib.replace(tzinfo=None) - last_loss_time).total_seconds() / 60.0

                            if 0 <= minutes_since_loss < LOSS_COOLDOWN_MINUTES:
                                logging.info(f"[LOSS COOLDOWN] Skipping {proposed_action}: {minutes_since_loss:.1f} min since last SL hit (< {LOSS_COOLDOWN_MINUTES} min cooldown).")
                                proposed_action = "HOLD"
                    except Exception as lc_err:
                        logging.error(f"[LOSS COOLDOWN ERROR] {lc_err}")

                # Cooldown Distance Check ($3.00 pending / $2.00 closed)
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
                            required_distance = 2.00 if last_outcome == "PENDING" else 1.50

                            if abs(curr_price - last_entry_price) < required_distance:
                                logging.info(f"[SCALP COOLDOWN] Skipping {proposed_action}: Price within ${required_distance:.2f} of previous trade at ${last_entry_price:.2f}.")
                                proposed_action = "HOLD"
                    except Exception as cd_err:
                        logging.error(f"[COOLDOWN ERROR] {cd_err}")

                if proposed_action == "HOLD":
                    logging.info(f"[MARKET SCAN] Price: ${curr_price:.2f} | 5M Swept High: ${recent_high:.2f} | 5M Swept Low: ${recent_low:.2f} | Status: HOLD{status_note}")
                else:
                    logging.info(f"[MARKET SCAN] Triggered {proposed_action} ({trigger_type}) at ${curr_price:.2f}. Running AI Analysis...")
                    ai_decision = await analyze_signal_with_ai(proposed_action, trigger_type, curr_price, df_1m, df_5m, recent_high, recent_low)

                    # ATR-BASED RISK SIZING (replaces fixed $2.5-5.5 VWAP-distance TP)
                    raw_atr = df_1m["atr"].iloc[-1]
                    atr_1m = float(raw_atr) if not pd.isna(raw_atr) else 3.0
                    risk = max(2.5, atr_1m * 1.2)

                    is_breakout_trigger = "Breakout" in trigger_type
                    tp1_r_mult = 1.5
                    tp2_r_mult = 3.0 if is_breakout_trigger else 2.5  # let continuation trades run further

                    if proposed_action == "BUY":
                        sl_price = float(curr_price - risk)
                        tp1_price = float(curr_price + risk * tp1_r_mult)
                        tp2_price = float(curr_price + risk * tp2_r_mult)
                    else:
                        sl_price = float(curr_price + risk)
                        tp1_price = float(curr_price - risk * tp1_r_mult)
                        tp2_price = float(curr_price - risk * tp2_r_mult)

                    if ai_decision.action == proposed_action:
                        new_id = log_trade_signal("EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price, float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning)
                        id_tag = f" #{new_id}" if new_id else ""
                        header_icon = "🚀" if is_breakout_trigger else "⚡"
                        header_label = "BREAKOUT CONTINUATION SIGNAL" if is_breakout_trigger else "LIQUIDITY SWEEP + BOS SIGNAL"
                        model_label = "5M/1M Momentum Breakout Continuation" if is_breakout_trigger else "5M Liquidity Sweep + 1M Break of Structure"
                        msg = (
                            f"{header_icon} *{header_label}{id_tag}*\n\n"
                            f"Asset: *XAUUSD (Gold Spot)*\n"
                            f"Action: *{proposed_action}*\n"
                            f"Type: *{trigger_type}*\n"
                            f"Entry Price: *${curr_price:.2f}*\n\n"
                            f"Stop Loss (SL): *${sl_price:.2f}*\n"
                            f"Take Profit 1 ({tp1_r_mult:.1f}R): *${tp1_price:.2f}*\n"
                            f"Take Profit 2 ({tp2_r_mult:.1f}R): *${tp2_price:.2f}*\n\n"
                            f"INDICATOR METRICS:\n"
                            f"- Model: {model_label}\n"
                            f"- Reference 5M High: ${recent_high:.2f}\n"
                            f"- Reference 5M Low: ${recent_low:.2f}\n"
                            f"- 1M ATR (risk unit): ${atr_1m:.2f}\n"
                            f"- 5M ADX Trend Strength: {adx_5m:.1f}\n\n"
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
                    "• `/analyze` - Forward-Test Breakdown by Strategy, ADX Regime & Session\n"
                    "• `/help` - Display Command Menu\n\n"
                    f"⚠️ Active stat-vetoes: ADX ≥ {STAT_VETO_ADX_THRESHOLD:.0f} and Mid session "
                    f"({STAT_VETO_MID_SESSION_START_HOUR}-{STAT_VETO_MID_SESSION_END_HOUR} WIB) are auto-skipped "
                    "based on forward-test underperformance.\n"
                    f"⏱️ Loss cooldown: {LOSS_COOLDOWN_MINUTES} min after any SL hit (any direction) before a new signal can execute."
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

            elif raw_text == "/analyze":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, trigger_type,
                               COALESCE(entry_price, price, 0) as entry_p,
                               sl_price, exit_price,
                               COALESCE(adx_15m, 0) as adx_val,
                               COALESCE(timestamp, created_at::text, '') as ts
                        FROM signals
                        WHERE status = 'EXECUTED' AND exit_price IS NOT NULL
                    """)
                    rows = cur.fetchall()
                    cur.close()
                    conn.close()

                    if not rows:
                        reply = "📐 *STRATEGY FORWARD-TEST ANALYSIS*\n\n_Not enough closed trades yet to analyze. Check back after more signals complete._"
                    else:
                        segments = {"Strategy": {}, "ADX Regime": {}, "Session": {}}
                        overall_r = []

                        for r in rows:
                            entry = float(r['entry_p'])
                            exit_p = float(r['exit_price'])
                            sl = float(r['sl_price']) if r.get('sl_price') is not None else 0.0
                            action = r['action']
                            adx_val = float(r['adx_val'])
                            trigger = r['trigger_type'] or ""
                            ts = r['ts'] or ""

                            r_mult = compute_r_multiple(action, entry, exit_p, sl)
                            overall_r.append(r_mult)

                            segments["Strategy"].setdefault(bucket_strategy(trigger), []).append(r_mult)
                            segments["ADX Regime"].setdefault(bucket_adx(adx_val), []).append(r_mult)
                            segments["Session"].setdefault(bucket_session(ts), []).append(r_mult)

                        n_total = len(overall_r)
                        overall_wr = (sum(1 for x in overall_r if x > 0) / n_total * 100) if n_total else 0.0
                        overall_avg_r = (sum(overall_r) / n_total) if n_total else 0.0

                        reply_parts = [
                            "📐 *STRATEGY FORWARD-TEST ANALYSIS*",
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | Avg R: *{overall_avg_r:+.2f}*",
                            "",
                        ]
                        for dim in ["Strategy", "ADX Regime", "Session"]:
                            reply_parts.append(format_performance_segment(dim, segments[dim]))
                            reply_parts.append("")

                        reply_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        reply_parts.append(
                            "💡 Segments need n≥8 to be flagged ⚠️/✅ (smaller samples are shown but noisy). "
                            "This does not auto-adjust the bot -- use it to decide whether to retune the ADX veto "
                            "thresholds, fib pullback zone, or session windows in code.\n\n"
                            f"🚫 Active stat-vetoes: ADX ≥ {STAT_VETO_ADX_THRESHOLD:.0f} and Mid session "
                            f"({STAT_VETO_MID_SESSION_START_HOUR}-{STAT_VETO_MID_SESSION_END_HOUR} WIB) are being "
                            "auto-skipped -- re-check this report after ~30-40 more trades to confirm they're "
                            "actually lifting AvgR.\n"
                            f"⏱️ Loss cooldown ({LOSS_COOLDOWN_MINUTES} min, any direction) is also active -- "
                            "added after spotting clustered same-session losing sequences in the raw trade list."
                        )
                        reply = "\n".join(reply_parts)

                    await send_telegram_alert(client, reply, target_chat_id=sender_chat_id)

                except Exception as an_err:
                    logging.error(f"[WEBHOOK ERROR /analyze] {an_err}")
                    await send_telegram_alert(client, f"⚠️ Error running analysis: {an_err}", target_chat_id=sender_chat_id)

    except Exception as e:
        logging.error(f"[WEBHOOK ERROR] {e}")

    return {"status": "ok"}
