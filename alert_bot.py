# A/B FORWARD-TEST BOT: EMA 5/9 CONTROL vs EMA 5/15 EXPERIMENTAL
# BASE: alert_bot_exhaustion_guard_v1.py
# Shared market data, shared DB, isolated strategy state/results, separate Telegram alerts.
# MOTHER_BAR_MICRO (Strategy C) is now the only MT5-live strategy as of
# 2026-09-19; CONTROL_5_9 and EXPERIMENTAL_5_15 are both PAPER only.
# (Swapped from EXPERIMENTAL=LIVE after B's live forward-test showed a
# real loss while C's forward-test log showed +0.38 avg R across 44 trades --
# see the MOTHER BAR C entry-condition analysis. The get-latest-signal MT5
# bridge follows whichever strategy has execution_mode='LIVE', so this stays
# correct automatically if swapped again -- just flip the two
# *_EXECUTION_MODE constants below, nothing else needs to change.)
# 
# CHANGES FROM V8.1:
# 1. Sped up EMAs from 9/15 to 5/9 for earlier entries on sudden momentum shifts.
# 2. Reduced TREND_15M_MIN_SEPARATION_PCT to 0.01 to allow earlier 15M trend confirmation.
# 3. Added "Aggressive Price Impulse" trigger logic to catch massive candles that 
#    cross both EMAs before the moving averages have time to untangle.
# 4. Made EMA column mapping dynamic so logging automatically updates if EMA speeds change.
#
# STRATEGY A REPLACEMENT (this revision): CONTROL_5_9's EMA engine, then its
# range-bracket-breakout successor, are both retired. A now trades classic
# XABCD harmonic reversal patterns (Gartley/Bat/Butterfly/Crab) built from a
# 5M zigzag of swing pivots -- see detect_harmonic_signal() and the
# HARMONIC_* constants. Still PAPER-only; B (EXPERIMENTAL_5_15) remains the
# only LIVE strategy, untouched by this change.

import os
import re
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
        msg = record.getMessage()
        return "/get-latest-signal" not in msg and "/get-pending-signals" not in msg

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
# informational context only. This threshold itself doesn't affect SL sizing;
# the 1H bias filter above is what guards against a losing streak on a sudden
# trend reversal, by blocking entries against the new direction rather than
# resizing the stop. (A separate, unrelated risk cap on B's stop DISTANCE was
# reintroduced later -- see EXPERIMENTAL_MAX_RISK_PRICE below.)
RANGING_REGIME_PCT_THRESHOLD = 0.30

# --- EMA EXECUTION SIGNAL (5M chart, fast settings for early impulse capture) ---
CONTROL_STRATEGY = "CONTROL_MB_V2"  # renamed 2026-09-23: A is now Mother Bar V2 (M1), not EMA 5/9 -- new name keeps its DB rows from mixing with the old harmonic/EMA-5-9 history
EXPERIMENTAL_STRATEGY = "EXPERIMENTAL_5_15"
CONTROL_EXECUTION_MODE = "LIVE"  # switched 2026-09-23: A is now the sole MT5-live strategy
EXPERIMENTAL_EXECUTION_MODE = "PAPER"  # stays PAPER -- B's V3 patch needs more forward-test time before going live

# STRATEGY C: Extreme-frequency M5 impulse/re-entry scalper
BREAKOUT_STRATEGY = "MOTHER_BAR_MICRO"
BREAKOUT_EXECUTION_MODE = "PAPER"  # switched 2026-09-23: back to PAPER now that A holds MT5-live

# A/B directional mode:
# BUY_ONLY is the safe/default replacement for the old dynamic 1H one-direction gate.
# /oneway_on  -> DYNAMIC (1H EMA200 decides which side is allowed)
# /oneway_off -> BUY_ONLY
# /both       -> BOTH (no one-direction gate)
CONTROL_DIRECTION_MODE = "BUY_ONLY"       # A's own switch -- independent of B. NOTE: left at its prior default;
                                           # your S3/V2 mother-bar backtest numbers weren't confirmed as BUY_ONLY vs BOTH,
                                           # toggle via the existing /a-direction command if you ran them as BOTH.
EXPERIMENTAL_DIRECTION_MODE = "BUY_ONLY"  # B's own switch -- independent of A
ONE_DIRECTION_MODES = {"BUY_ONLY", "DYNAMIC", "BOTH"}

# --- STRATEGY A REPLACEMENT: RANGE BRACKET BREAKOUT (5M) ---
# CONTROL_5_9's EMA logic is retired. This is a straddle/OCO-style breakout
# system built from a consolidation range on the 5M chart: a virtual
# buy-stop sits above range high and a virtual sell-stop below range low;
# whichever side actually breaks with a confirmed candle CLOSE (not a wick)
# is taken, and the other side is treated as auto-cancelled for that range.
# Stays PAPER-only (CONTROL_EXECUTION_MODE above is unchanged).
BRACKET_LOOKBACK_BARS = 30          # ~2.5h of 5M bars used to define the range
BRACKET_MIN_RANGE_ATR = 1.0         # range must be at least this many ATRs wide -- skip noise-tight ranges
BRACKET_MAX_RANGE_ATR = 6.0         # range wider than this isn't a real consolidation anymore (likely a drift)
BRACKET_BREAKOUT_BUFFER_ATR = 0.15  # close must clear the level by this much -- filters marginal pokes
BRACKET_SL_BUFFER_ATR = 0.25        # stop sits just back inside the broken level, not across the whole range
BRACKET_MIN_RISK_ATR = 0.5          # floor on stop distance so spread/noise can't produce an unrealistically tight stop
BRACKET_TP1_RANGE_MULT = 0.5        # TP1 = 0.5x the range height, projected from the breakout (measured-move target)
BRACKET_TP2_RANGE_MULT = 1.0        # TP2 = 1.0x the range height, projected from the breakout
BRACKET_STOCH_PERIOD = 14
BRACKET_STOCH_OVERBOUGHT = 95       # skip a BUY breakout if stoch is already this extreme -- likely exhausted, not fresh
BRACKET_STOCH_OVERSOLD = 5          # skip a SELL breakout if stoch is already this extreme

# --- STRATEGY A REPLACEMENT v2: HARMONIC PATTERN (XABCD), 5M CHART ---
# detect_bracket_breakout_signal() below is retired for CONTROL_STRATEGY (kept
# in place, unused, same as this file's convention elsewhere). Bot A now
# trades classic XABCD harmonic reversal patterns -- Gartley, Bat, Butterfly,
# Crab -- built off a zigzag of 5M swing pivots. X-A-B-C are confirmed swing
# points; D is never a confirmed pivot, it's the *live* price testing a
# computed Potential Reversal Zone (PRZ). A signal only fires when:
#   1. Four alternating pivots (X,A,B,C) exist with a big-enough XA leg.
#   2. The AB/XA and BC/AB ratios fall inside a known pattern's tolerance --
#      that pattern's PRZ for D is computed from its AD/XA Fibonacci range.
#   3. Current price is actually inside that PRZ right now.
#   4. The current candle shows a reversal in the completion direction (not
#      just a wick poke through the zone).
#   5. Stochastic confirms exhaustion (oversold for a bullish D, overbought
#      for a bearish D) -- momentum confluence, same spirit as the stoch
#      filter already used on the bracket breakout.
# Stays PAPER-only (CONTROL_EXECUTION_MODE above is unchanged).
HARMONIC_ZIGZAG_ATR_MULT = 0.8     # min swing size (in ATR) to register a new zigzag pivot -- tuned smaller
                                    # than a typical 1.2-1.5x zigzag so more, smaller patterns qualify.
HARMONIC_MIN_XA_ATR = 2.0          # XA leg must be at least this many ATRs -- lowered from 3.0 for more
                                    # frequent, smaller-scale patterns; still enough to filter pure noise.
HARMONIC_MAX_PIVOTS_TRACKED = 8    # how many recent zigzag pivots to keep around
HARMONIC_SL_BUFFER_ATR = 0.25      # stop sits just beyond X, not a fixed distance from entry
HARMONIC_MIN_RISK_ATR = 0.5        # floor on stop distance, same purpose as BRACKET_MIN_RISK_ATR
HARMONIC_TP1_AD_FRACTION = 0.382   # TP1 = 38.2% of the way from D back toward A (first partial)
HARMONIC_TP2_R_FALLBACK = 2.5      # used only if point A doesn't make sense as TP2 (e.g. already tagged)
HARMONIC_STOCH_PERIOD = 14
HARMONIC_STOCH_OVERSOLD_MAX = 35   # bullish D completion needs stoch at/under this
HARMONIC_STOCH_OVERBOUGHT_MIN = 65 # bearish D completion needs stoch at/over this

# Fibonacci tolerance bands per pattern: (ab_xa_range, bc_ab_range, ad_xa_range).
# ad_xa is measured from A, i.e. D = A - ad_ratio * (A - X) -- values <1.0
# retrace back toward X (Gartley/Bat), values >1.0 extend past X (Butterfly/Crab).
HARMONIC_PATTERNS = {
    "Gartley":   {"ab_xa": (0.55, 0.68), "bc_ab": (0.38, 0.90), "ad_xa": (0.75, 0.82)},
    "Bat":       {"ab_xa": (0.35, 0.52), "bc_ab": (0.38, 0.90), "ad_xa": (0.84, 0.90)},
    "Butterfly": {"ab_xa": (0.74, 0.82), "bc_ab": (0.38, 0.90), "ad_xa": (1.24, 1.65)},
    "Crab":      {"ab_xa": (0.32, 0.68), "bc_ab": (0.38, 0.90), "ad_xa": (1.55, 1.70)},
}

# Optional real-MT5 market-data ingress for Strategy C.
# The MT5 EA can POST fresh M5 bars/ticks to /mt5-market-data. C prefers this cache.
MT5_DATA_SECRET = os.getenv("MT5_DATA_SECRET", "").strip()
MT5_DATA_CACHE_TTL_SECONDS = 90  # widened 2026-09-24: was 20s, too tight for the M1 feed --
                                  # the EA only detects+posts a new M1 bar once per
                                  # InpPollInterval (10s default) after it closes, and the
                                  # server's A-loop checks within the first 20s of each new
                                  # minute, so a 20s TTL could force a fallback on nothing
                                  # more than ordinary latency. 90s still means "this data is
                                  # at most 1 closed bar old", which is fine for M1 execution.
mt5_market_cache = {"df": None, "updated_at": None, "source": None}
mt5_market_cache_m1 = {"df": None, "updated_at": None, "source": None}  # Bot A's free M1 feed, pushed by the MT5 EA

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
CONTROL_TREND_ADX_MIN = 27.0  # A-only distinctive filter: shared RANGE_MODE_ADX_MAX=20 lets weak-trend
                              # entries (adx 20-27) through to A same as B. Forward test showed A's
                              # 75 executed trades were 100% classified TREND -- this raises the bar so
                              # A only takes the stronger-trend subset. B is untouched.
RANGE_LOOKBACK_5M = 10             # candles defining the current range bracket (~50 min on 5M)
RANGE_MAX_WIDTH_ATR_MULT = 2.0     # bracket must be no wider than this (in ATR) to count as a real range
RANGE_MIN_WIDTH_ATR_MULT = 0.8     # bracket must be at least this wide -- too tight isn't tradeable (spread/slippage)
RANGE_EDGE_ZONE_PCT = 0.20         # price must be within this fraction of the range width from an edge to fade it
RANGE_SL_BUFFER_ATR_MULT = 0.3     # stop placed this many ATR beyond the bracket edge being faded
RANGE_MODE_MAX_15M_ADX = 25.0      # skip the fade if the 15M chart itself shows a real trend (ADX >= this)

# --- EXPERIMENTAL-ONLY (Bot B): CHOP/EFFICIENCY REGIME OVERRIDE ---
# Live data check (Aug 30 - Sep 10): 0 of 99 executed EXPERIMENTAL trades
# were ever classified RANGE, because 5M ADX never once read below
# RANGE_MODE_ADX_MAX=20 on an executed trade (median ~37). ADX can read high
# on big-but-directionless candles, so a genuinely choppy stretch (lots of
# movement, no net progress) still gets waved through as TREND. This adds a
# second, independent check -- an efficiency ratio (net displacement / total
# path length) -- so real chop gets faded even when ADX disagrees. Control A
# is left untouched so it still serves as the unmodified baseline.
EXPERIMENTAL_EFFICIENCY_LOOKBACK = 12   # ~60 min on 5M
EXPERIMENTAL_EFFICIENCY_MAX = 0.50  # V3: only trade materially directional 5M structure      # B trend gate: below this = too much back-and-forth

# --- EXPERIMENTAL-ONLY (Bot B): TREND MODE RISK/TARGET OVERRIDE ---
# Your MT5 EA already opens two 0.01-lot positions per signal: lot 1 closes
# at TP1, lot 2 ("the runner") has its SL moved to breakeven once TP1 hits
# and then either rides to TP2 or stops at BE. Because the runner is already
# risk-free past TP1, there is no reason to cap it at only 2.5R -- widening
# TP2 costs nothing extra in risk and lets it actually capture trend moves.
# TP1 is pulled in a bit so the short/quick leg banks faster and more often.
# SL is widened slightly (1.0 -> 1.15x ATR) so ordinary noise survives.
EXPERIMENTAL_RISK_ATR_MULT = 1.15       # was 1.0 (shared with Control A)
EXPERIMENTAL_TP1_R = 1.2                # was 1.5 (shared)
EXPERIMENTAL_TP2_R = 3.0  # V3: backtest showed 3.0R captures more of the runner edge                # was 2.5 (shared) -- runner leg, already risk-free at BE
# BUG FIX (B's R-vs-$ mismatch): your MT5 EA always opens two FIXED 0.01-lot
# positions per signal -- lot size never scales with stop distance. That
# means the real $ risked on a loss is directly proportional to how wide the
# ATR-based stop happened to be that trade, while result_r normalizes every
# loss to a flat -2.0R regardless of width. Those two only agree if stop
# width stays roughly constant. It doesn't: pulling the historical B trades,
# sl_dist was usually 2.5-7 (median ~5, ~$10 combined risk) but spiked to
# 15-21 during a few volatile stretches -- $30-42 lost on ONE trade, same
# -2.0R as a routine $10 loss. That's what produced +5.11R alongside -$82 on
# the same trade set: the R ledger was blind to the fixed-lot $ reality.
# Capping the ATR-based risk distance directly shrinks the real stop sent to
# MT5 on those outlier-volatility trades, so both ledgers describe the same
# risk again. $18 combined (9.0 price-distance on a single 0.01 lot) only
# clamps the top ~7% widest-stop trades historically -- the routine ones are
# untouched.
EXPERIMENTAL_MAX_RISK_PRICE = 9.0
# B V2 trend-quality gates. These are intentionally separate from the shared
# EMA trigger so A is unaffected.
EXPERIMENTAL_MIN_ADX = 35.0  # V3: ADX 20-35 was a persistent low-quality bucket in the CSV
EXPERIMENTAL_MIN_EMA_SPREAD_ATR = 0.60
EXPERIMENTAL_MIN_EMA_SLOPE_ATR = 0.00
EXPERIMENTAL_MAX_ENTRY_EXTENSION_ATR = 1.00
EXPERIMENTAL_TOUCH_ONLY = True

# --- EXPERIMENTAL-ONLY (Bot B): STRUCTURAL S/R BRACKET STRATEGY ---
# Real support/resistance from swing pivots over a wider lookback, not just
# the last 10 candles' high/low (that bracket reacts to noise, not structure).
# Two uses of the same levels, both fixed-risk -- no position-size scaling
# anywhere in this file, ever:
#   RANGE:  fade a level that's actually been touched more than once
#   TREND:  broken level retested and held -> trade the continuation
SR_PIVOT_LOOKBACK = 60           # 5M candles (~5 hours) scanned for swing pivots
SR_PIVOT_WING = 3                # candles each side that must be lower/higher to count as a pivot
SR_LEVEL_CLUSTER_ATR_MULT = 0.5  # pivots within this many ATR of each other = one level
SR_MIN_TOUCHES = 2               # a level needs at least this many pivot touches to count as real S/R
SR_EDGE_ZONE_ATR_MULT = 0.4      # price must be within this many ATR of a level to act on it
SR_RETEST_MAX_BARS = 6           # a breakout must retest the broken level within this many candles
SR_RANGE_SL_BUFFER_ATR = 0.3     # fade-mode SL beyond the level
SR_TREND_SL_BUFFER_ATR = 0.5     # retest-mode SL beyond the retested level (needs room to survive the retest wick)
SR_RANGE_TP1_R = 1.0
SR_RANGE_TP2_R = 1.8
SR_TREND_TP1_R = 1.2
SR_TREND_TP2_R = 3.5             # runner leg -- same already-risk-free-past-TP1 logic as the EMA trend override

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
# place. In WIB (UTC+7), NY's Fri 17:00 EST close lands at ~04:00 WIB
# SATURDAY, and NY's Sun 17:00 EST reopen lands at ~05:00 WIB MONDAY --
# so the market is actually open the first few hours of WIB Saturday
# before the real close, and all of Sunday is closed.
FOREX_SATURDAY_CLOSE_HOUR_WIB = 4  # approx NY Friday 17:00 EST close, in WIB
FOREX_MONDAY_OPEN_HOUR_WIB = 5   # approx NY Sunday 17:00 EST reopen, in WIB


def is_forex_market_open(now_wib: datetime) -> bool:
    weekday = now_wib.weekday()  # Monday=0 ... Sunday=6
    if weekday == 5 and now_wib.hour >= FOREX_SATURDAY_CLOSE_HOUR_WIB:
        return False  # Saturday, after the Friday session has actually closed
    if weekday == 6:  # Sunday: closed all day in WIB (reopen lands on Monday)
        return False
    if weekday == 0 and now_wib.hour < FOREX_MONDAY_OPEN_HOUR_WIB:
        return False  # Monday, before the weekend reopen has actually happened
    return True

TWELVE_DATA_DAILY_LIMIT = 800
TWELVE_DATA_SAFETY_MARGIN = 40 

_twelve_data_call_count = 0
_twelve_data_budget_date = None  
_twelve_data_calls_by_tf = {"5min": 0, "15min": 0, "1h": 0, "1min": 0}


def _reset_budget_if_new_day(now_wib: datetime):
    global _twelve_data_call_count, _twelve_data_budget_date, _twelve_data_calls_by_tf
    today = now_wib.date()
    if _twelve_data_budget_date != today:
        if _twelve_data_budget_date is not None:
            logging.info(f"[TWELVE DATA BUDGET] New WIB day. Resetting counter.")
        _twelve_data_budget_date = today
        _twelve_data_call_count = 0
        _twelve_data_calls_by_tf = {"5min": 0, "15min": 0, "1h": 0, "1min": 0}


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


def get_setting(key: str, default: str) -> str:
    """Read a persisted runtime setting (e.g. direction mode) from bot_settings.
    Falls back to `default` if unset, table missing, or DB unreachable -- so a
    fresh DB or a DB hiccup never crashes startup, it just uses the hardcoded
    default like before."""
    if not DATABASE_URL:
        return default
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM bot_settings WHERE key = %s", (key,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row["value"] if row else default
    except Exception as e:
        logging.warning(f"[SETTINGS] Failed to read '{key}', using default '{default}': {e}")
        return default


def set_setting(key: str, value: str):
    """Persist a runtime setting so it survives restarts/redeploys -- without
    this, toggles like /oneway_on only live in memory and silently revert to
    the hardcoded default the next time the service restarts."""
    if not DATABASE_URL:
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO bot_settings (key, value, updated_at) VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
            """,
            (key, value),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logging.warning(f"[SETTINGS] Failed to persist '{key}'={value}: {e}")


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
            CREATE TABLE IF NOT EXISTS bot_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
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


# =============================================================================
# MACRO CONTEXT MODULE
# Read-only, informational only -- does NOT gate or filter any A/B/C signal.
# Shared across all three bots via /macro and appended to each bot's /status.
#
# Three public, traceable inputs (no insider/secret-network framing -- this
# is literally what macro/gold traders watch):
#   1. Real yields (US 10Y TIPS, FRED series DFII10). Falling real yields =
#      bullish for non-yielding gold; rising = bearish. This is usually the
#      single strongest fundamental driver of gold.
#   2. Dollar strength, via a EUR/USD proxy (EUR is ~57% of the real ICE
#      Dollar Index by weight, so its inverse is a reasonable single-pair
#      stand-in -- NOT the real DXY). Gold is priced in USD, so USD
#      strength (EUR/USD falling) is mechanically bearish for gold in other
#      currencies' terms; USD weakness is bullish.
#   3. Speculative positioning, via CFTC's weekly Commitment of Traders
#      report for COMEX gold. Rising non-commercial (large speculator) net
#      length = bullish confirmation; falling = bearish. NOTE: at historic
#      extremes this flips to a contrarian crowding signal in practice --
#      not implemented here, flagged as a known simplification.
#
# Each fetch is independently wrapped so one dead source degrades to "N/A"
# rather than breaking the whole feature. Refreshed at most once per WIB day
# (all three sources update daily or slower anyway).
# =============================================================================

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
CFTC_APP_TOKEN = os.getenv("CFTC_APP_TOKEN", "").strip()  # optional, raises Socrata rate limit if set

_macro_cache = {
    "date": None, "score": 0, "label": "NEUTRAL",
    "components": [], "text": "Macro context not yet refreshed today."
}


async def _fetch_real_yield_signal(client: httpx.AsyncClient):
    if not FRED_API_KEY:
        return {"name": "Real Yields (10Y TIPS)", "direction": 0, "detail": "FRED_API_KEY not set -- skipped"}
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id=DFII10"
               f"&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=10")
        res = await client.get(url, timeout=15)
        data = res.json()
        obs = [o for o in data.get("observations", []) if o.get("value") not in (None, ".")]
        if len(obs) < 6:
            return {"name": "Real Yields (10Y TIPS)", "direction": 0, "detail": "Insufficient FRED data"}
        latest = float(obs[0]["value"])
        prior = float(obs[5]["value"])  # ~5 business days back
        change = latest - prior
        direction = -1 if change > 0.02 else (1 if change < -0.02 else 0)
        arrow = "falling" if direction == 1 else ("rising" if direction == -1 else "flat")
        return {
            "name": "Real Yields (10Y TIPS)", "direction": direction,
            "detail": f"{latest:.2f}% ({arrow}, {change:+.2f} pts / ~5 sessions) -- {'bullish' if direction == 1 else ('bearish' if direction == -1 else 'neutral')} for gold"
        }
    except Exception as e:
        return {"name": "Real Yields (10Y TIPS)", "direction": 0, "detail": f"Fetch failed: {e}"}


async def _fetch_dollar_proxy_signal(client: httpx.AsyncClient):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1day&outputsize=6&apikey={TWELVE_DATA_API_KEY}"
        res = await client.get(url, timeout=15)
        note_twelve_data_call("1day")
        data = res.json()
        if "values" not in data or len(data["values"]) < 6:
            return {"name": "Dollar Strength (EUR/USD proxy)", "direction": 0, "detail": "Insufficient Twelve Data"}
        closes = [float(v["close"]) for v in data["values"]]
        latest, prior = closes[0], closes[-1]
        pct_change = (latest - prior) / prior * 100
        direction = 1 if pct_change > 0.15 else (-1 if pct_change < -0.15 else 0)
        usd_move = "weakening" if direction == 1 else ("strengthening" if direction == -1 else "flat")
        return {
            "name": "Dollar Strength (EUR/USD proxy)", "direction": direction,
            "detail": f"EUR/USD {pct_change:+.2f}% / 5 sessions -- USD {usd_move}, {'bullish' if direction == 1 else ('bearish' if direction == -1 else 'neutral')} for gold. NOTE: proxy only, not the real DXY."
        }
    except Exception as e:
        return {"name": "Dollar Strength (EUR/USD proxy)", "direction": 0, "detail": f"Fetch failed: {e}"}


async def _fetch_cot_positioning_signal(client: httpx.AsyncClient):
    try:
        url = ("https://publicreporting.cftc.gov/resource/6dca-aqww.json"
               "?$where=market_and_exchange_names like '%25GOLD - COMMODITY EXCHANGE%25'"
               "&$order=report_date_as_yyyy_mm_dd DESC&$limit=2")
        headers = {"X-App-Token": CFTC_APP_TOKEN} if CFTC_APP_TOKEN else {}
        res = await client.get(url, headers=headers, timeout=15)
        rows = res.json()
        if not isinstance(rows, list) or len(rows) < 2:
            return {"name": "Speculative Positioning (CFTC COT)", "direction": 0, "detail": "Insufficient CFTC data"}
        def net_long(row):
            return float(row.get("noncomm_positions_long_all", 0)) - float(row.get("noncomm_positions_short_all", 0))
        latest_net, prior_net = net_long(rows[0]), net_long(rows[1])
        change = latest_net - prior_net
        direction = 1 if change > 0 else (-1 if change < 0 else 0)
        report_date = rows[0].get("report_date_as_yyyy_mm_dd", "")[:10]
        return {
            "name": "Speculative Positioning (CFTC COT)", "direction": direction,
            "detail": f"Net large-spec longs {change:+,.0f} contracts vs prior week (as of {report_date}) -- "
                      f"{'bullish confirmation' if direction == 1 else ('bearish' if direction == -1 else 'flat')}. "
                      f"NOTE: at multi-year-extreme crowding this can flip contrarian -- not modeled here."
        }
    except Exception as e:
        return {"name": "Speculative Positioning (CFTC COT)", "direction": 0, "detail": f"Fetch failed: {e}"}


async def refresh_macro_context(client: httpx.AsyncClient, now_wib: datetime):
    """Refreshes at most once per WIB day. Never raises -- worst case leaves
    yesterday's cached text in place with its own date still visible."""
    global _macro_cache
    today = now_wib.date()
    if _macro_cache["date"] == today:
        return
    try:
        yields, dollar, cot = await asyncio.gather(
            _fetch_real_yield_signal(client),
            _fetch_dollar_proxy_signal(client),
            _fetch_cot_positioning_signal(client),
        )
        components = [yields, dollar, cot]
        score = sum(c["direction"] for c in components)
        label = {3: "STRONG BULLISH", 2: "BULLISH", 1: "BULLISH", 0: "NEUTRAL",
                  -1: "BEARISH", -2: "BEARISH", -3: "STRONG BEARISH"}[score]
        lines = [f"🌐 *MACRO CONTEXT FOR GOLD* (score {score:+d}/3 -> *{label}*)"]
        lines.append("Informational only -- does not filter or gate any A/B/C signal.\n")
        for c in components:
            arrow = "🟢" if c["direction"] == 1 else ("🔴" if c["direction"] == -1 else "⚪")
            lines.append(f"{arrow} *{c['name']}*\n   {c['detail']}")
        text = "\n".join(lines)
        _macro_cache = {"date": today, "score": score, "label": label, "components": components, "text": text}
    except Exception as e:
        logging.warning(f"[MACRO CONTEXT] Refresh failed, keeping stale cache: {e}")


def get_macro_status_line() -> str:
    """Short one-liner for embedding inside /status -- full detail lives in /macro."""
    return f"🌐 Macro (gold): *{_macro_cache['label']}* ({_macro_cache['score']:+d}/3) -- see /macro for detail"


def compute_ema_trend(df: pd.DataFrame):
    """15M EMA9/20 confluence read, used only as a report/label -- range mode
    now gates on 15M ADX instead (see RANGE_MODE_MAX_15M_ADX)."""
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

def _compute_stochastic_k(df: pd.DataFrame, period: int = BRACKET_STOCH_PERIOD) -> float:
    """Plain %K stochastic over `period` bars. Returns 50.0 (neutral) if there
    isn't enough history yet -- callers treat that as "no opinion", not a veto."""
    if df is None or len(df) < period:
        return 50.0
    window = df.iloc[-period:]
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    if hi - lo <= 0:
        return 50.0
    close = float(df["close"].iloc[-1])
    return max(0.0, min(100.0, (close - lo) / (hi - lo) * 100.0))


def detect_bracket_breakout_signal(df_5m: pd.DataFrame):
    """STRATEGY A: Range Bracket Breakout (OCO-style), 5M chart.

    Defines a consolidation range from the BRACKET_LOOKBACK_BARS candles
    before the current one (same "prior N candles, not including current"
    pattern used by detect_range_reversal), then treats the range edges as
    a virtual buy-stop / sell-stop pair. A signal only fires when:
      1. The range itself is a genuine consolidation (width bounded in ATR
         terms -- not noise-tight, not a slow drift).
      2. The PREVIOUS candle closed back inside the range (so this is the
         actual break bar, not something already several bars past it).
      3. The CURRENT candle CLOSES beyond the level by at least the buffer
         (a wick poke through the line does not count -- this is the main
         defense against M1/M5 false breakouts).
      4. The breakout candle's own body agrees with the direction (bullish
         body for a BUY break, bearish body for a SELL break).
    Only one side can ever qualify on a given candle, which is the OCO
    property: whichever level actually gets taken out cancels the other.
    A soft stochastic filter (not a hard structural gate) skips breakouts
    that are already at an extreme reading, since those are more likely to
    be the tail end of a move than a fresh one.
    """
    if len(df_5m) < BRACKET_LOOKBACK_BARS + 2:
        return "HOLD", "Insufficient data for bracket range", None, None

    raw_atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if raw_atr is None or pd.isna(raw_atr) or float(raw_atr) <= 0:
        return "HOLD", "ATR unavailable", None, None
    atr_5m = float(raw_atr)

    bracket = df_5m.iloc[-(BRACKET_LOOKBACK_BARS + 1):-1]
    range_high = float(bracket["high"].max())
    range_low = float(bracket["low"].min())
    width = range_high - range_low

    if width < BRACKET_MIN_RANGE_ATR * atr_5m:
        return "HOLD", "Bracket range too tight -- inside normal noise/spread", range_high, range_low
    if width > BRACKET_MAX_RANGE_ATR * atr_5m:
        return "HOLD", "Bracket range too wide -- likely a drift, not consolidation", range_high, range_low

    prev = df_5m.iloc[-2]
    curr = df_5m.iloc[-1]
    prev_close = float(prev["close"])
    curr_open = float(curr["open"])
    curr_close = float(curr["close"])

    # Require the break to be fresh: the bar right before this one must
    # still have closed inside the range (with a little slack for the
    # buffer itself), otherwise we're catching a move several bars late.
    prev_was_inside = (range_low - BRACKET_BREAKOUT_BUFFER_ATR * atr_5m) <= prev_close <= (range_high + BRACKET_BREAKOUT_BUFFER_ATR * atr_5m)
    if not prev_was_inside:
        return "HOLD", "No fresh break -- previous candle already outside range", range_high, range_low

    buy_trigger = range_high + BRACKET_BREAKOUT_BUFFER_ATR * atr_5m
    sell_trigger = range_low - BRACKET_BREAKOUT_BUFFER_ATR * atr_5m

    bullish_break = curr_close > buy_trigger and curr_close > curr_open
    bearish_break = curr_close < sell_trigger and curr_close < curr_open

    if bullish_break and bearish_break:
        # Shouldn't happen given the trigger math, but stay OCO-safe if it ever does.
        return "HOLD", "Ambiguous break -- both sides tagged", range_high, range_low

    stoch_k = _compute_stochastic_k(df_5m)

    if bullish_break:
        if stoch_k >= BRACKET_STOCH_OVERBOUGHT:
            return "HOLD", f"Buy-stop hit but Stoch {stoch_k:.0f} already extreme -- skipping", range_high, range_low
        return "BUY", "Bracket Breakout (Buy-Stop, close-confirmed)", range_high, range_low

    if bearish_break:
        if stoch_k <= BRACKET_STOCH_OVERSOLD:
            return "HOLD", f"Sell-stop hit but Stoch {stoch_k:.0f} already extreme -- skipping", range_high, range_low
        return "SELL", "Bracket Breakout (Sell-Stop, close-confirmed)", range_high, range_low

    return "HOLD", "Price still inside bracket", range_high, range_low


def find_zigzag_pivots(df_5m: pd.DataFrame, deviation_atr_mult: float = HARMONIC_ZIGZAG_ATR_MULT,
                        max_pivots: int = HARMONIC_MAX_PIVOTS_TRACKED):
    """ATR-deviation zigzag. Tracks the running extreme (high in an up leg,
    low in a down leg); once price retraces from that extreme by at least
    `deviation_atr_mult` x ATR, the extreme is locked in as a confirmed pivot
    and the zigzag flips direction. Returns a list of (bar_index, price,
    'H'/'L') pivots, oldest first, alternating type by construction -- the
    live/unconfirmed price action after the last pivot is NOT included
    (that's the D-completion zone the caller tests against)."""
    if df_5m is None or len(df_5m) < 20 or "atr" not in df_5m.columns:
        return []
    raw_atr = df_5m["atr"].iloc[-1]
    if pd.isna(raw_atr) or raw_atr <= 0:
        return []
    atr = float(raw_atr)
    threshold = deviation_atr_mult * atr

    highs = df_5m["high"].astype(float).values
    lows = df_5m["low"].astype(float).values
    closes = df_5m["close"].astype(float).values
    n = len(df_5m)

    pivots = []
    trend = None  # None until the first leg is established, then 'up'/'down'
    extreme_idx = 0
    extreme_price = closes[0]

    for i in range(1, n):
        if trend is None:
            move = closes[i] - extreme_price
            if abs(move) >= threshold:
                trend = "up" if move > 0 else "down"
                extreme_idx, extreme_price = i, closes[i]
            continue

        if trend == "up":
            if highs[i] > extreme_price:
                extreme_price, extreme_idx = highs[i], i
            elif extreme_price - lows[i] >= threshold:
                pivots.append((extreme_idx, float(extreme_price), "H"))
                trend = "down"
                extreme_price, extreme_idx = lows[i], i
        else:
            if lows[i] < extreme_price:
                extreme_price, extreme_idx = lows[i], i
            elif highs[i] - extreme_price >= threshold:
                pivots.append((extreme_idx, float(extreme_price), "L"))
                trend = "up"
                extreme_price, extreme_idx = highs[i], i

    return pivots[-max_pivots:]


def detect_harmonic_signal(df_5m: pd.DataFrame):
    """STRATEGY A (v2): XABCD Harmonic Pattern completion, 5M chart.
    See the HARMONIC_* constants block above for the full rationale. Returns
    the same (action, trigger_type, level_1, level_2) shape as the other
    detectors -- level_1 is point X (used for SL placement beyond X),
    level_2 is point A (used as the classic TP2 projection target)."""
    if len(df_5m) < HARMONIC_MIN_XA_ATR + 20:
        return "HOLD", "Insufficient data for harmonic structure", None, None

    raw_atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if raw_atr is None or pd.isna(raw_atr) or float(raw_atr) <= 0:
        return "HOLD", "ATR unavailable", None, None
    atr = float(raw_atr)

    pivots = find_zigzag_pivots(df_5m)
    if len(pivots) < 4:
        return "HOLD", "Not enough confirmed swing structure yet", None, None

    (_, X, x_type), (_, A, a_type), (_, B, b_type), (_, C, c_type) = pivots[-4:]

    # Zigzag construction already guarantees strict alternation, but stay
    # defensive -- a malformed/duplicated pivot list should never fire.
    if x_type == a_type or a_type == b_type or b_type == c_type:
        return "HOLD", "Pivot sequence not alternating -- skipping", None, None

    XA = A - X
    AB = B - A
    BC = C - B
    if abs(XA) < HARMONIC_MIN_XA_ATR * atr or AB == 0:
        return "HOLD", "XA leg too small for a valid harmonic pattern", X, A

    ab_xa = abs(AB / XA)
    bc_ab = abs(BC / AB)

    candidates = []
    for name, spec in HARMONIC_PATTERNS.items():
        ab_lo, ab_hi = spec["ab_xa"]
        bc_lo, bc_hi = spec["bc_ab"]
        if not (ab_lo <= ab_xa <= ab_hi and bc_lo <= bc_ab <= bc_hi):
            continue
        ad_lo, ad_hi = spec["ad_xa"]
        d1 = A - ad_lo * (A - X)
        d2 = A - ad_hi * (A - X)
        prz_low, prz_high = (d1, d2) if d1 <= d2 else (d2, d1)
        candidates.append((name, prz_low, prz_high))

    if not candidates:
        return "HOLD", "No harmonic pattern ratios matched this swing", X, A

    curr = df_5m.iloc[-1]
    price = float(curr["close"])
    in_zone = [c for c in candidates if c[1] <= price <= c[2]]
    if not in_zone:
        names = ", ".join(c[0] for c in candidates)
        return "HOLD", f"Forming ({names}) -- price hasn't reached the PRZ yet", X, A

    # Prefer the narrowest matching PRZ -- the most precisely-defined pattern.
    in_zone.sort(key=lambda c: c[2] - c[1])
    name, prz_low, prz_high = in_zone[0]

    bullish_completion = (c_type == "H")  # C is a high -> D completes as a low -> BUY
    stoch_k = _compute_stochastic_k(df_5m, HARMONIC_STOCH_PERIOD)

    if bullish_completion:
        reversal_candle = curr["close"] > curr["open"]
        if not reversal_candle:
            return "HOLD", f"{name} PRZ reached, waiting for a bullish reversal candle", X, A
        if stoch_k > HARMONIC_STOCH_OVERSOLD_MAX:
            return "HOLD", f"{name} PRZ reached but Stoch {stoch_k:.0f} not oversold yet", X, A
        return "BUY", f"Harmonic {name} (Bullish D Completion)", X, A
    else:
        reversal_candle = curr["close"] < curr["open"]
        if not reversal_candle:
            return "HOLD", f"{name} PRZ reached, waiting for a bearish reversal candle", X, A
        if stoch_k < HARMONIC_STOCH_OVERBOUGHT_MIN:
            return "HOLD", f"{name} PRZ reached but Stoch {stoch_k:.0f} not overbought yet", X, A
        return "SELL", f"Harmonic {name} (Bearish D Completion)", X, A


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


def compute_sr_levels(df_5m: pd.DataFrame, lookback: int = SR_PIVOT_LOOKBACK, wing: int = SR_PIVOT_WING,
                       cluster_atr_mult: float = SR_LEVEL_CLUSTER_ATR_MULT, min_touches: int = SR_MIN_TOUCHES):
    """Structural support/resistance from swing pivots, not just the last N
    candles' high/low. A pivot is a local high/low with `wing` candles lower/
    higher on both sides. Nearby pivots (within cluster_atr_mult x ATR) are
    merged into one level; a level only counts once it's been touched at
    least min_touches times -- one wick doesn't make a level."""
    if df_5m is None or len(df_5m) < lookback:
        return []
    window = df_5m.iloc[-lookback:].reset_index(drop=True)
    raw_atr = window["atr"].iloc[-1] if "atr" in window.columns else None
    atr = float(raw_atr) if raw_atr is not None and not pd.isna(raw_atr) and raw_atr > 0 else None
    if not atr:
        return []
    highs = window["high"].astype(float).values
    lows = window["low"].astype(float).values
    pivots = []
    for i in range(wing, len(window) - wing):
        if highs[i] == max(highs[i - wing:i + wing + 1]):
            pivots.append(float(highs[i]))
        if lows[i] == min(lows[i - wing:i + wing + 1]):
            pivots.append(float(lows[i]))
    if not pivots:
        return []
    pivots.sort()
    clusters = [[pivots[0]]]
    for p in pivots[1:]:
        if p - clusters[-1][-1] <= cluster_atr_mult * atr:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(sum(c) / len(c), len(c)) for c in clusters if len(c) >= min_touches]


def _nearest_sr_levels(levels, price: float):
    """Split levels into nearest support (below price) and resistance (above price)."""
    below = [lv for lv in levels if lv[0] < price]
    above = [lv for lv in levels if lv[0] > price]
    support = max(below, key=lambda x: x[0]) if below else None
    resistance = min(above, key=lambda x: x[0]) if above else None
    return support, resistance


def detect_sr_bracket_signal(df_5m: pd.DataFrame, strategy_mode: str, trend_15m: str):
    """Fixed-risk structural strategy. RANGE: fade a real (multi-touch) S/R
    level. TREND: broken level retested and held -> trade the continuation.
    No position-size scaling anywhere -- same fixed per-trade risk as every
    other signal in this file, just placed against real structure."""
    if df_5m is None or len(df_5m) < SR_PIVOT_LOOKBACK:
        return "HOLD", "Insufficient data for S/R", None, None
    curr = df_5m.iloc[-1]
    price = float(curr["close"])
    raw_atr = curr["atr"] if "atr" in curr and not pd.isna(curr["atr"]) else None
    atr = float(raw_atr) if raw_atr is not None and raw_atr > 0 else None
    if not atr:
        return "HOLD", "ATR unavailable", None, None

    levels = compute_sr_levels(df_5m)
    support, resistance = _nearest_sr_levels(levels, price)
    edge_zone = SR_EDGE_ZONE_ATR_MULT * atr

    if strategy_mode == "RANGE":
        if support and abs(price - support[0]) <= edge_zone and curr["close"] > curr["open"]:
            return "BUY", f"S/R Fade - Support ({support[1]}x touched)", (resistance[0] if resistance else None), support[0]
        if resistance and abs(price - resistance[0]) <= edge_zone and curr["close"] < curr["open"]:
            return "SELL", f"S/R Fade - Resistance ({resistance[1]}x touched)", resistance[0], (support[0] if support else None)
        return "HOLD", "No S/R edge reaction", (resistance[0] if resistance else None), (support[0] if support else None)

    # TREND mode: break-and-retest continuation -- trade WITH the trend off a
    # level that was just broken and is now being retested and held, not
    # against it. Requires 15M trend agreement, same as the EMA trigger does.
    # Note: once price breaks above a resistance level, that level now sits
    # BELOW price -- it shows up as `support`, not `resistance`, in the
    # nearest-level split above. Same in reverse for a bearish breakdown.
    recent = df_5m.iloc[-(SR_RETEST_MAX_BARS + 1):-1]
    if support and trend_15m == "BULLISH":
        broke_above = bool((recent["close"] < support[0]).any())
        retesting = abs(price - support[0]) <= edge_zone and price >= support[0]
        if broke_above and retesting and curr["close"] > curr["open"]:
            return "BUY", "S/R Retest - Broken Resistance Held as Support", support[0], None
    if resistance and trend_15m == "BEARISH":
        broke_below = bool((recent["close"] > resistance[0]).any())
        retesting = abs(price - resistance[0]) <= edge_zone and price <= resistance[0]
        if broke_below and retesting and curr["close"] < curr["open"]:
            return "SELL", "S/R Retest - Broken Support Held as Resistance", None, resistance[0]

    return "HOLD", "No S/R retest setup", (resistance[0] if resistance else None), (support[0] if support else None)


def compute_efficiency_ratio(df_5m: pd.DataFrame, lookback: int = EXPERIMENTAL_EFFICIENCY_LOOKBACK) -> float:
    """Kaufman-style efficiency ratio: net displacement / total path length
    over the lookback window. 1.0 = a straight-line move (a real trend).
    Near 0 = lots of back-and-forth movement that nets out to nowhere (chop),
    even when ADX reads high because individual candles are large.
    Bot B (EXPERIMENTAL) uses this to catch range conditions ADX misses."""
    if df_5m is None or len(df_5m) < lookback + 1:
        return 1.0  # not enough data -- default to "trending", don't force RANGE mode blind
    window = df_5m["close"].iloc[-(lookback + 1):]
    net_displacement = abs(float(window.iloc[-1]) - float(window.iloc[0]))
    path_length = float(window.diff().abs().sum())
    if path_length <= 0:
        return 1.0
    return net_displacement / path_length


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

def bucket_adx_gate_shadow(divergence_type_val: str) -> str:
    """Buckets the ADX-gate shadow verdict logged into `divergence_type`
    for Mother Bar signals (e.g. 'ADX_GATE_SHADOW=ALLOW_RANGING'). The gate
    is shadow-only right now (MB_ADX_GATE_SHADOW_MODE=True) -- it never
    actually blocks a trade -- so every signal fires regardless of the
    verdict here. This bucket exists purely to build the comparison data
    needed to decide whether to flip the gate live later."""
    v = divergence_type_val or ""
    m = re.match(r"ADX_GATE_SHADOW=(ALLOW|BLOCK)_(\w+)", v)
    if not m:
        return "N/A (pre-gate)"
    decision, zone = m.group(1), m.group(2).title()
    return f"{zone} \u2014 gate would {decision}"

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
    if "Harmonic" in t:
        for pattern_name in ("Gartley", "Bat", "Butterfly", "Crab"):
            if pattern_name in t:
                return f"Harmonic ({pattern_name})"
        return "Harmonic (Other)"
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

def format_entry_condition_segment(buckets: dict, min_sample_to_flag: int = 5) -> str:
    """Same idea as format_performance_segment, but per entry-condition
    category (trigger_type) and carrying pips/$ alongside R, not just R --
    for bot C's /analyze, where 'which entry condition is actually paying
    for itself' is the whole point of the report. buckets maps
    category_name -> list of (r_mult, pips, usd) tuples."""
    lines = []
    for label, rows in sorted(buckets.items(), key=lambda item: -len(item[1])):
        n = len(rows)
        r_vals = [r for r, _, _ in rows]
        pips_vals = [p for _, p, _ in rows]
        usd_vals = [u for _, _, u in rows]
        wins = sum(1 for r in r_vals if r > 0)
        win_rate = (wins / n * 100) if n else 0.0
        avg_r = (sum(r_vals) / n) if n else 0.0
        total_pips = sum(pips_vals)
        total_usd = sum(usd_vals)
        flag = ""
        if n >= min_sample_to_flag:
            if win_rate < 35: flag = " \u26a0\ufe0f underperforming"
            elif win_rate > 65: flag = " \u2705 strong"
        lines.append(
            f"\u2022 *{label}*: n={n}, WR={win_rate:.0f}%, AvgR={avg_r:+.2f}{flag}\n"
            f"   Pips: {total_pips:+.1f} | $: {total_usd:+.2f}"
        )
    return "\n".join(lines)

def extract_mb_data_source(regime_val: str, reasoning: str) -> str:
    """C's signal rows logged after this fix store the real feed used for
    that bar ('MT5' or 'TWELVE_DATA_FALLBACK') directly in the `regime`
    column -- see evaluate_extreme_strategy. Rows logged before this fix
    still have regime='MOTHER_BAR_MICRO' (a constant, not actually useful)
    and only recorded the source inside the free-text `reasoning` field
    (e.g. '...ATR=5.267, data=TWELVE_DATA_FALLBACK.') -- fall back to
    parsing that so older trades still show up in the breakdown instead of
    silently disappearing into 'Unknown'."""
    if regime_val in ("MT5", "TWELVE_DATA_FALLBACK"):
        return regime_val
    if reasoning:
        m = re.search(r"data=([A-Z0-9_]+)\.", reasoning)
        if m:
            return m.group(1)
    return "Unknown (pre-tracking)"


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
# STRATEGY C: MOTHER BAR MICRO-BREAKOUT (MBMB) -- EXTREME-FREQUENCY SCALPER
# =====================================================================
# Replaces the old EMA9/21+VWAP pullback/re-entry engine with a
# structure-based engine derived from the "Teknik MB Super Simpel"
# (Mother Bar / inside-bar breakout) method:
#   1. A "Mother Bar" (MB) is a candle with real body/range (not a doji)
#      that is large relative to local volatility.
#   2. Smaller "inside bars" then consolidate fully inside the MB's
#      High/Low -- more inside bars = a tighter squeeze = higher-quality
#      breakout when it finally comes.
#   3. A valid break requires a CLOSE beyond the MB High/Low, not just a
#      wick poke (wick-only touches are explicitly invalid per the
#      source method).
#   4. Three entry models, all implemented here:
#        Model 1 - Close Break:     enter on the close of the candle that
#                   closes beyond MB High/Low.
#        Model 2 - Momentum Break:  same as Model 1 but requires a bigger
#                   penetration through the level (a more committed,
#                   higher-conviction breakout candle).
#        Model 3 - Fib Retracement: after a confirmed break, wait for
#                   price to pull back to the 25/50/61.8/75% level of the MB
#                   range and re-confirm direction -- cheaper entry,
#                   same target, so a better realized R:R.
#   5. Stop-loss sits at the opposite extreme of the MB. Targets are
#      NOT a fixed ATR multiple -- they are sized directly off the MB's
#      own high-low range (1.0x for TP1, 1.5x for TP2), so target
#      distance automatically follows the size of the actual candle
#      structure that formed it, exactly as the source method (and the
#      requested design change) specifies. On a fast, low timeframe an
#      MB's range is small, so targets are small and get hit fast --
#      that's what makes this an extreme-frequency engine rather than a
#      swing one; the same logic on a higher timeframe would just as
#      naturally produce wider, slower targets.
#
# Everything is recomputed fresh from OHLC history on every call
# (stateless pattern detection) -- only trade counters/cooldown/last
# price are kept in _extreme_state, same as before, so nothing else in
# the file that reads _extreme_state / EXTREME_* needs to change.

EXTREME_EMA_FAST = 9
EXTREME_EMA_SLOW = 21
EXTREME_ATR_PERIOD = 14
EXTREME_DAILY_LOSS_LIMIT_R = -6.0   # UNUSED as of 2026-09-24: C's daily loss breaker was removed (your request), same as A's earlier. Value kept here only as a reference/reintroduction point.
EXTREME_MAX_TRADES_PER_DAY = 60     # raised vs the old 40 -- MB structures on a fast timeframe are smaller/quicker, so more clean setups/day are expected
EXTREME_COOLDOWN_SECONDS = 10
EXTREME_DIRECTION_MODE = "BOTH"     # BOTH | BUY_ONLY | SELL_ONLY -- toggle via /c_both, /c_buyonly, /c_sellonly on Bot C
EXTREME_DIRECTION_MODES = {"BOTH", "BUY_ONLY", "SELL_ONLY"}
EXTREME_MAX_TRADES_PER_DAY_LABEL = "Unlimited" if EXTREME_MAX_TRADES_PER_DAY is None else str(EXTREME_MAX_TRADES_PER_DAY)
EXTREME_MIN_REENTRY_DISTANCE_ATR = 0.20
EXTREME_SESSION_START_HOUR = 0      # No time gate: runs all day
EXTREME_SESSION_END_HOUR = 24

# --- Mother Bar detection parameters ---
MB_LOOKBACK_BARS = 40            # how far back to search for the most recent qualifying Mother Bar
MB_MIN_BODY_ATR = 1.30           # MB candle body must be this many ATRs -- excludes dojis/small bars (source rule #1)
MB_MIN_RANGE_ATR = 1.50          # MB total high-low range must be this many ATRs -- proxy for the source method's "min 100-200 pip" size rule, scaled to whatever timeframe/instrument is fed in
MB_MIN_RANGE_ABS = 0.15          # hard floor in price units (XAUUSD $) so a technically-qualifying MB isn't so small that spread/slippage eats the whole target on M1
MB_MAX_AGE_BARS = 25             # an MB structure is ignored once this many bars have passed with no clean break (stale structure -- source method implies MBs decay if nothing happens)
MB_MIN_INSIDE_BARS = 1           # require at least 1 inside bar of consolidation before a break counts (source rule: more inside bars = higher probability; 0 inside bars = avoid, per source rule about single-inside-bar MBs shown as a bad example)
MB_MAX_INSIDE_BARS = 12          # beyond this the "squeeze" has gone stale / turned into a range, not a coiled MB
MB_INSIDE_TOLERANCE_ATR = 0.05   # small wick allowance so one bar's minor wick poke doesn't wrongly disqualify it as "inside"
MB_BREAK_CONFIRM_ATR = 0.05      # Model 1: minimum close-through distance beyond MB High/Low to count as a valid (non-wick) break
MB_MOMENTUM_BREAK_ATR = 0.20     # Model 2: bigger penetration required to tag a break as a stronger "momentum" break vs a bare Model 1 close-break
MB_RETRACE_LEVELS = (0.25, 0.50, 0.618, 0.75)  # Model 3 Fibonacci retracement zones measured back into the MB range from the broken level. 0.618 (golden ratio) added 2026-09-25 -- applies to both A (always all levels) and C (toggleable, see /c_fib618_on/off)
MB_RETRACE_TOLERANCE_ATR = 0.08
MB_SL_BUFFER_ATR = 0.05          # small buffer placed beyond the opposite MB extreme so SL isn't sitting exactly on the level
MB_TP1_MULT = 1.0                # TP1 = 1x the MB's own range (source method's baseline 1:1 RR rule)
MB_TP2_MULT = 1.5                # TP2 = 1.5x the MB's own range (source method's stated alternative RR)

# Trigger-specific override (2026-09-19): both the offline backtest (5M
# reconstruction) and the live forward-test log agree "MB Momentum Break"
# has a fine win rate (62-73%) but its reward is too small relative to its
# risk (avg winner roughly half the avg loser) -- a reward-sizing problem,
# not an entry-quality problem. Widening just this trigger's targets tested
# net positive (+$163 over 112 backtested 5M trades, WR only dropping from
# 62.5% to 53.6%) without touching any other trigger's math.
MB_MOMENTUM_BREAK_TP1_MULT = 1.5
MB_MOMENTUM_BREAK_TP2_MULT = 3.0
# BUG FIX (trade #551, FOMC night): C's SL is "opposite MB extreme" -- risk
# is literally the Mother Bar's own candle range, with no ceiling. During
# the FOMC spike, three consecutive Mother Bars ballooned to 5-10x their
# normal size (typical MB range here is ~4-15; that stretch hit 29.3, 39.7,
# then 48.6) and, under the fixed 2x0.01-lot PAPER model, produced real
# losses of -$41.52 / -$45.01 / -$116.32 on trades that R-multiple still
# just called -2.0R each -- same blind spot as B's fix earlier. #551 alone
# gave back more than half of the cumulative $ the strategy had built up
# ($211.90 -> $95.58 in one trade). TP1/TP2 are left untouched -- a bigger
# real range earning a bigger real target is legitimate; it's the loss side
# that can't be allowed to scale unbounded under fixed lot size. Capping at
# 20.0 catches exactly that FOMC cluster (and two elevated post-event
# trades right after) while leaving the other ~84% of C's history untouched
# (next-highest risk outside that cluster was 16.65).
MB_MAX_RISK_PRICE = 20.0
# COMPLEMENTARY FIX: the cap above bounds the DAMAGE if a trade fires during
# a shock; this stops it from firing during/right after one at all. Purely
# price-derived (no news calendar dependency) -- see
# _mb_volatility_cooldown_active()'s docstring for the full reasoning.
MB_VOL_SPIKE_BASELINE_PERIOD = 60   # slow ATR baseline (~5 hours on M5) -- long enough not to inflate within the first few bars of a shock, unlike the fast 14-period ATR
MB_VOL_SPIKE_ATR_MULT = 3.0         # a single bar's true range at 3x+ the slow baseline = shock bar (FOMC print, surprise headline, flash move)
MB_VOL_SPIKE_COOLDOWN_BARS = 6      # ~30 minutes on M5 -- sit out new MB entries for this long after the most recent shock bar
MB_REQUIRE_TREND_ALIGNMENT = True  # soft filter: only take breaks in the direction the EMA9/21 stack already leans, to cut down false breakouts at extreme frequency
# ADX-GATED TREND FILTER (added per your call): ranging markets are the
# documented failure mode for every inside-bar/mother-bar breakout strategy
# -- research consistently splits performance at ADX ~20-25 (below = skip,
# above = the breakout has real momentum behind it), not by which direction
# a short lagging EMA9/21 stack happens to be leaning. So:
#   ADX <= MB_RANGING_ADX_MAX        -> ranging, HOLD regardless of direction
#   MB_RANGING_ADX_MAX < ADX < MB_STRONG_TREND_ADX_MIN
#                                     -> ambiguous zone, still require the
#                                        EMA9/21 alignment as a precaution
#   ADX >= MB_STRONG_TREND_ADX_MIN   -> strong confirmed trend; EMA9/21
#                                        alignment is waived entirely, so a
#                                        Mother Bar break AGAINST the local
#                                        EMA lean is allowed -- ADX itself
#                                        is already confirming real
#                                        directional energy exists, which is
#                                        the actual thing the alignment
#                                        filter was a (lagging) proxy for.
MB_RANGING_ADX_MAX = 20.0
MB_STRONG_TREND_ADX_MIN = 25.0
# SHADOW MODE (per your call): this gate has ZERO historical validation --
# ADX was never even computed for C before this session (every historical
# row logged adx_15m=0.0), and C's Fib-Retrace entries in particular are a
# pullback style that the "avoid ranging markets" research was never
# specifically tested against -- it's mostly established for breakout-style
# continuation entries. So for now the gate does NOT block any real trade:
# every signal fires exactly as it would with only the EMA9/21 check (the
# pre-ADX-gate behavior). It only RECORDS what it would have decided (ADX
# reading, zone, allow/block) into metrics/reasoning/the alert, so real
# forward data builds up a comparison instead of guessing from literature.
# Flip this to False once there's a few weeks of shadow data to justify it.
MB_ADX_GATE_SHADOW_MODE = True

# MOMENTUM BREAK ENTRY -- DISABLED (2026-09 backtest finding): across a
# 1,265-trade / 3-month offline backtest AND a 39-trade live paper sample,
# "MB Momentum Break" was the one trigger that was flat-to-negative while
# every other Mother Bar trigger (Fib Retrace 25/50/75%, Close Break) had
# a positive edge. Re-tested by widening its TP1/TP2 multiples from
# 1.5x/3.0x up to 4.0x/8.0x -- it never turned net positive at any width,
# which means the problem is where this entry fires (chasing an
# already-extended breakout), not how far its targets are set. Confirmed
# again on the full Jan-Sep 2026 dataset (10,314 raw candidates): dropping
# this trigger alone took the strategy from +$717 to +$1,996 over the same
# 9 months on a $200/0.01-lot backtest, while also cutting max drawdown
# from -87.0% to -53.1%. Set to True only after a fresh out-of-sample
# stretch shows it's stopped being the weak link.
MB_MOMENTUM_BREAK_ENABLED = False

_extreme_state = {
    "last_signal_time": None,
    "last_entry_price": None,
    "last_action": None,
    "trades_today": 0,
    "trade_day": None,
}

# =====================================================================
# MOTHER BAR CONFIG OBJECTS (2026-09 A/C split)
#
# The detection/trade-plan functions below now take a `cfg` dict instead of
# reading the MB_* globals directly, so the exact same engine can run twice
# concurrently with different parameters: C keeps every current default
# (nothing above this comment changed), A runs a separate "V2" tune on M1.
# =====================================================================
MB_CONFIG_C = {
    "min_body_atr": MB_MIN_BODY_ATR, "min_range_atr": MB_MIN_RANGE_ATR, "min_range_abs": MB_MIN_RANGE_ABS,
    "max_age_bars": MB_MAX_AGE_BARS, "min_inside_bars": MB_MIN_INSIDE_BARS, "max_inside_bars": MB_MAX_INSIDE_BARS,
    "inside_tolerance_atr": MB_INSIDE_TOLERANCE_ATR, "break_confirm_atr": MB_BREAK_CONFIRM_ATR,
    "momentum_break_atr": MB_MOMENTUM_BREAK_ATR, "retrace_levels": MB_RETRACE_LEVELS,
    "retrace_tolerance_atr": MB_RETRACE_TOLERANCE_ATR, "sl_buffer_atr": MB_SL_BUFFER_ATR,
    "tp1_mult": MB_TP1_MULT, "tp2_mult": MB_TP2_MULT,
    "momentum_tp1_mult": MB_MOMENTUM_BREAK_TP1_MULT, "momentum_tp2_mult": MB_MOMENTUM_BREAK_TP2_MULT,
    "momentum_break_enabled": MB_MOMENTUM_BREAK_ENABLED,
    "close_break_enabled": True,                  # toggle via /c_closebreak_on, /c_closebreak_off on Bot C
    "max_risk_price": MB_MAX_RISK_PRICE,          # cap-and-shrink-stop (unchanged C behavior)
    "reject_risk_floor": None, "reject_risk_atr_mult": None,   # C does not reject on risk, only caps
    "vol_spike_baseline_period": MB_VOL_SPIKE_BASELINE_PERIOD, "vol_spike_atr_mult": MB_VOL_SPIKE_ATR_MULT,
    "vol_spike_cooldown_bars": MB_VOL_SPIKE_COOLDOWN_BARS, "require_trend_alignment": MB_REQUIRE_TREND_ALIGNMENT,
    "ranging_adx_max": MB_RANGING_ADX_MAX, "strong_trend_adx_min": MB_STRONG_TREND_ADX_MIN,
    "adx_gate_shadow_mode": MB_ADX_GATE_SHADOW_MODE,
    "atr_period": EXTREME_ATR_PERIOD, "ema_fast": EXTREME_EMA_FAST, "ema_slow": EXTREME_EMA_SLOW,
}

# Bot A "V2" Mother Bar tune, per your spec: same pattern-detection rules
# (all 6 triggers, same MB qualification/inside-bar/retrace geometry, same
# 0.20 ATR re-entry spacing, same -6R daily breaker) but on M1, wider TP2s,
# and a hard risk REJECT instead of C's cap-and-shrink.
MB_CONFIG_A_V2 = dict(MB_CONFIG_C)
MB_CONFIG_A_V2.update({
    "tp1_mult": 1.25, "tp2_mult": 8.0,
    "momentum_tp1_mult": 2.0, "momentum_tp2_mult": 15.0,
    "momentum_break_enabled": True,      # "all 6 triggers" -- re-enabled for A only, C stays disabled
    "close_break_enabled": True,         # A always runs all 6 triggers -- independent of C's per-trigger toggles below
    "retrace_levels": MB_RETRACE_LEVELS, # A always runs all 4 Fib levels (25/50/61.8/75%) -- independent of C's per-trigger toggles below
    "max_risk_price": None,              # A does not cap-and-shrink; it rejects instead (below)
    "reject_risk_floor": 15.0,           # reject if risk > max($15, 2.25 x ATR)
    "reject_risk_atr_mult": 2.25,
})

# --- BOT C PER-TRIGGER TOGGLES (each of C's 5 Mother Bar entry conditions
# can be switched on/off independently via Telegram -- /c_fib25_on/off,
# /c_fib50_on/off, /c_fib618_on/off, /c_fib75_on/off, /c_closebreak_on/off,
# /c_momentum_on/off). Persisted so a toggle survives restarts, same as
# the direction-mode settings. This only ever touches MB_CONFIG_C -- A
# always runs all 6 triggers regardless (see MB_CONFIG_A_V2 above).
_c_trigger_state = {
    "fib25": True, "fib50": True, "fib618": True, "fib75": True,
    "close_break": True, "momentum": MB_MOMENTUM_BREAK_ENABLED,
}
_C_FIB_LEVELS = {"fib25": 0.25, "fib50": 0.50, "fib618": 0.618, "fib75": 0.75}

def _apply_c_trigger_state():
    """Rebuilds MB_CONFIG_C's trigger-gating keys from _c_trigger_state.
    Always assigns NEW tuple/bool values (never mutates a shared list/tuple
    in place) so MB_CONFIG_A_V2 -- a separate dict that already pinned its
    own retrace_levels/close_break_enabled/momentum_break_enabled -- is
    never affected by a C-only toggle."""
    MB_CONFIG_C["momentum_break_enabled"] = _c_trigger_state["momentum"]
    MB_CONFIG_C["close_break_enabled"] = _c_trigger_state["close_break"]
    MB_CONFIG_C["retrace_levels"] = tuple(
        lvl for key, lvl in _C_FIB_LEVELS.items() if _c_trigger_state[key]
    )

# Free/no-payment M1 data path for Bot A: prefer bars pushed by your own MT5
# EA (zero API cost, no rate limit -- see /mt5-market-data below); Twelve
# Data 1min is only a fallback, and is capped separately from the shared
# 800/day budget so a stalled MT5 push can't starve A/B/C's core 5-minute
# cycle of credits.
TD_1MIN_FALLBACK_DAILY_CAP = 200
TD_1MIN_FALLBACK_MIN_INTERVAL_SECONDS = 55
_td_1min_fallback_state = {"date": None, "count": 0, "last_call": None}

def _td_1min_fallback_ok(now_wib: datetime) -> bool:
    today = now_wib.date()
    if _td_1min_fallback_state["date"] != today:
        _td_1min_fallback_state["date"] = today
        _td_1min_fallback_state["count"] = 0
    if _td_1min_fallback_state["count"] >= TD_1MIN_FALLBACK_DAILY_CAP:
        return False
    last = _td_1min_fallback_state["last_call"]
    if last is not None and (datetime.now(timezone.utc) - last).total_seconds() < TD_1MIN_FALLBACK_MIN_INTERVAL_SECONDS:
        return False
    return True

def _note_td_1min_fallback_call():
    _td_1min_fallback_state["count"] += 1
    _td_1min_fallback_state["last_call"] = datetime.now(timezone.utc)

def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0

def _session_vwap(df: pd.DataFrame) -> pd.Series:
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
    if "datetime" in work.columns:
        dt = pd.to_datetime(work["datetime"], errors="coerce")
        day = dt.dt.date
        pv = tp * vol
        return pv.groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)

def _extreme_indicators(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or MB_CONFIG_C
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    prev_close = d["close"].shift(1)
    tr = np.maximum(
        d["high"] - d["low"],
        np.maximum((d["high"] - prev_close).abs(), (d["low"] - prev_close).abs())
    )
    d["tr"] = tr
    d["atr"] = tr.rolling(cfg["atr_period"]).mean()
    d["atr_slow"] = tr.rolling(cfg["vol_spike_baseline_period"]).mean()
    d["ema9"] = d["close"].ewm(span=cfg["ema_fast"], adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=cfg["ema_slow"], adjust=False).mean()
    d["vwap"] = _session_vwap(d)
    d["body"] = (d["close"] - d["open"]).abs()
    d["range"] = d["high"] - d["low"]
    # ADX (same rolling-sum DI/DX formula used for A/B's 5M ADX, period 14,
    # so C's readings are directly comparable to A/B's on /status). Feeds
    # the ranging-market gate in _mb_trend_aligned() below.
    up_move = d["high"] - d["high"].shift(1)
    down_move = d["low"].shift(1) - d["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=d.index).rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm, index=d.index).rolling(14).sum() / (tr14 + 1e-10))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    d["plus_di"] = plus_di
    d["minus_di"] = minus_di
    d["adx"] = dx.rolling(14).mean()
    return d

def _extreme_roll_day(now_wib: datetime, state: dict = None):
    state = state if state is not None else _extreme_state
    day = now_wib.date()
    if state["trade_day"] != day:
        state["trade_day"] = day
        state["trades_today"] = 0

def _extreme_session_ok(now_wib: datetime) -> bool:
    return EXTREME_SESSION_START_HOUR <= now_wib.hour < EXTREME_SESSION_END_HOUR and is_forex_market_open(now_wib)

def _mb_trend_aligned(action: str, cur, metrics: dict = None, cfg: dict = None) -> bool:
    cfg = cfg or MB_CONFIG_C
    ema9, ema21 = float(cur["ema9"]), float(cur["ema21"])
    ema_aligned = ema9 >= ema21 if action == "BUY" else ema9 <= ema21

    adx = float(cur["adx"]) if not pd.isna(cur["adx"]) else 0.0
    if adx <= cfg["ranging_adx_max"]:
        gate_would_allow, zone = False, "ranging"
    elif adx >= cfg["strong_trend_adx_min"]:
        gate_would_allow, zone = True, "strong"
    else:
        gate_would_allow = ema_aligned if cfg["require_trend_alignment"] else True
        zone = "ambiguous"
    if metrics is not None:
        metrics["adx_gate_would_allow"] = gate_would_allow
        metrics["adx_gate_zone"] = zone

    if cfg["adx_gate_shadow_mode"]:
        # Shadow mode: real trading still gates on plain EMA9/21 alignment
        # only, same as before this gate existed. The tiered ADX verdict
        # above is recorded but never blocks anything yet.
        return ema_aligned if cfg["require_trend_alignment"] else True
    return gate_would_allow

def _mb_volatility_cooldown_active(d: pd.DataFrame, last_idx: int, cfg: dict = None) -> bool:
    cfg = cfg or MB_CONFIG_C
    """Purely price-derived 'we just had a volatility shock' cooldown -- no
    external news calendar needed, and no dependency on a third-party feed
    being reachable. A 'shock bar' is one whose OWN true range blew past
    MB_VOL_SPIKE_ATR_MULT x a SLOW baseline ATR (MB_VOL_SPIKE_BASELINE_PERIOD
    bars, ~5 hours on M5) -- deliberately a much longer window than the fast
    14-period ATR the rest of the strategy sizes off of, so this baseline
    doesn't itself inflate within the first few bars of the event the way
    the fast ATR does. If any of the last MB_VOL_SPIKE_COOLDOWN_BARS bars
    was a shock bar, new MB entries are held off entirely -- this is the
    complementary layer to MB_MAX_RISK_PRICE: that cap bounds the damage IF
    a trade fires during/right after a shock; this stops it from firing
    there at all. Can't be precisely backtested against #549/550/551
    without the raw OHLC history (only the signal log was available), so
    the thresholds below are reasoned defaults, not curve-fit ones --
    worth watching /analyze's data after a few real news events."""
    if "atr_slow" not in d.columns:
        return False
    lookback = min(cfg["vol_spike_cooldown_bars"], last_idx)
    for i in range(last_idx - lookback + 1, last_idx + 1):
        if i < 0:
            continue
        baseline = d["atr_slow"].iloc[i]
        tr_i = d["tr"].iloc[i]
        if pd.isna(baseline) or baseline <= 0 or pd.isna(tr_i):
            continue
        if tr_i >= cfg["vol_spike_atr_mult"] * baseline:
            return True
    return False

def find_mother_bar_signal(d: pd.DataFrame, cfg: dict = None):
    """Stateless scan of closed-bar OHLC history for an active Mother Bar
    structure, returning a trade signal if the LAST bar is either a fresh
    valid breakout (Model 1/2) or a valid Fib retracement entry into an
    already-broken MB (Model 3). Returns (action, trigger, metrics)."""
    cfg = cfg or MB_CONFIG_C
    n = len(d)
    if n < MB_LOOKBACK_BARS + 3:
        return "HOLD", "Insufficient history for MB scan", {}

    last_idx = n - 1
    if _mb_volatility_cooldown_active(d, last_idx, cfg):
        return "HOLD", "Volatility-spike cooldown active (recent shock bar) -- sitting out", {}

    atr_arr = d["atr"].values
    high_arr = d["high"].values
    low_arr = d["low"].values
    close_arr = d["close"].values
    open_arr = d["open"].values
    body_arr = d["body"].values
    range_arr = d["range"].values

    search_start = max(0, last_idx - 1 - MB_LOOKBACK_BARS)

    # Walk backward from the bar just before the current one, looking for
    # the most recent bar big enough to qualify as a Mother Bar.
    for mb_idx in range(last_idx - 1, search_start - 1, -1):
        atr_mb = atr_arr[mb_idx]
        if not atr_mb or np.isnan(atr_mb) or atr_mb <= 0:
            continue
        body_atr = body_arr[mb_idx] / atr_mb
        rng_atr = range_arr[mb_idx] / atr_mb
        rng_abs = range_arr[mb_idx]
        if body_atr < cfg["min_body_atr"] or rng_atr < cfg["min_range_atr"] or rng_abs < cfg["min_range_abs"]:
            continue

        mb_high = high_arr[mb_idx]
        mb_low = low_arr[mb_idx]
        mb_range = mb_high - mb_low
        age = last_idx - mb_idx
        if age > cfg["max_age_bars"]:
            # Nothing useful further back than this either -- structures
            # only get staler as we keep walking backward.
            break

        # Replay every bar since the MB to find inside-bar count and
        # whether/where a valid break already happened.
        inside_count = 0
        broken = False
        break_idx = None
        break_action = None
        for j in range(mb_idx + 1, n):
            atr_j = atr_arr[j]
            if not atr_j or np.isnan(atr_j) or atr_j <= 0:
                continue
            c, o, h, l = close_arr[j], open_arr[j], high_arr[j], low_arr[j]
            if not broken:
                if c > mb_high + cfg["break_confirm_atr"] * atr_j and c > o:
                    broken, break_idx, break_action = True, j, "BUY"
                elif c < mb_low - cfg["break_confirm_atr"] * atr_j and c < o:
                    broken, break_idx, break_action = True, j, "SELL"
                elif h <= mb_high + cfg["inside_tolerance_atr"] * atr_j and l >= mb_low - cfg["inside_tolerance_atr"] * atr_j:
                    inside_count += 1
                # else: a wick poked outside without a valid close-break --
                # per the source method this is NOT a valid break, so the
                # MB structure just keeps waiting (bar doesn't count as
                # inside, but doesn't invalidate the structure either).

        metrics = {
            "atr": float(atr_arr[last_idx]), "mb_high": float(mb_high), "mb_low": float(mb_low),
            "mb_range": float(mb_range), "mb_age_bars": age, "inside_bars": inside_count,
            "adx": float(d["adx"].iloc[last_idx]) if not pd.isna(d["adx"].iloc[last_idx]) else 0.0,
        }

        if not broken:
            # Still coiling. Only actionable if inside-bar count already
            # qualifies AND the break is happening right now, on the last bar.
            if inside_count < cfg["min_inside_bars"] or inside_count > cfg["max_inside_bars"]:
                continue
            cur = d.iloc[last_idx]
            c, o = float(cur["close"]), float(cur["open"])
            atr_now = float(cur["atr"])
            if atr_now <= 0 or np.isnan(atr_now):
                continue
            if c > mb_high + cfg["break_confirm_atr"] * atr_now and c > o:
                action = "BUY"
            elif c < mb_low - cfg["break_confirm_atr"] * atr_now and c < o:
                action = "SELL"
            else:
                continue  # this MB hasn't broken yet -- keep it as the active structure, nothing to do this bar
            if not _mb_trend_aligned(action, cur, metrics, cfg):
                continue
            penetration_atr = (c - mb_high) / atr_now if action == "BUY" else (mb_low - c) / atr_now
            if penetration_atr >= cfg["momentum_break_atr"]:
                if not cfg["momentum_break_enabled"]:
                    continue  # momentum-break entries disabled -- see cfg["momentum_break_enabled"] comment above
                trigger = "MB Momentum Break"
            else:
                if not cfg.get("close_break_enabled", True):
                    continue  # close-break entries disabled -- see cfg["close_break_enabled"] comment above
                trigger = "MB Close Break"
            return action, trigger, metrics

        # Already broken. If the break happened on THIS bar, that's the
        # same fresh-breakout case as above, just found via the "broken"
        # branch because the loop already recorded it.
        if break_idx == last_idx:
            if inside_count < cfg["min_inside_bars"] or inside_count > cfg["max_inside_bars"]:
                continue
            cur = d.iloc[last_idx]
            if not _mb_trend_aligned(break_action, cur, metrics, cfg):
                continue
            atr_now = float(cur["atr"])
            c = float(cur["close"])
            penetration_atr = (c - mb_high) / atr_now if break_action == "BUY" else (mb_low - c) / atr_now
            if penetration_atr >= cfg["momentum_break_atr"]:
                if not cfg["momentum_break_enabled"]:
                    continue  # momentum-break entries disabled -- see cfg["momentum_break_enabled"] comment above
                trigger = "MB Momentum Break"
            else:
                if not cfg.get("close_break_enabled", True):
                    continue  # close-break entries disabled -- see cfg["close_break_enabled"] comment above
                trigger = "MB Close Break"
            return break_action, trigger, metrics

        # Break happened earlier -- check for a Model 3 retracement entry
        # on the current bar, as long as the structure isn't stale yet.
        bars_since_break = last_idx - break_idx
        if bars_since_break > cfg["max_age_bars"] or inside_count < cfg["min_inside_bars"] or inside_count > cfg["max_inside_bars"]:
            continue
        cur = d.iloc[last_idx]
        atr_now = float(cur["atr"])
        if atr_now <= 0 or np.isnan(atr_now):
            continue
        c, o, h, l = float(cur["close"]), float(cur["open"]), float(cur["high"]), float(cur["low"])
        tol = cfg["retrace_tolerance_atr"] * atr_now
        if break_action == "BUY":
            for lvl in cfg["retrace_levels"]:
                zone_price = mb_high - lvl * mb_range
                if l <= zone_price + tol and c > o and c >= zone_price - tol:
                    if not _mb_trend_aligned("BUY", cur, metrics, cfg):
                        break
                    metrics["retrace_level"] = lvl
                    return "BUY", f"MB Fib Retrace {round(lvl*100)}%", metrics
        else:
            for lvl in cfg["retrace_levels"]:
                zone_price = mb_low + lvl * mb_range
                if h >= zone_price - tol and c < o and c <= zone_price + tol:
                    if not _mb_trend_aligned("SELL", cur, metrics, cfg):
                        break
                    metrics["retrace_level"] = lvl
                    return "SELL", f"MB Fib Retrace {round(lvl*100)}%", metrics
        continue  # this MB gave no clean entry on this bar; keep walking back for an older/other candidate

    return "HOLD", "No active Mother Bar setup", {}

def detect_extreme_m5_signal(df_5m: pd.DataFrame, cfg: dict = None):
    """Entry point kept for compatibility with the rest of the file.
    Runs Mother Bar detection and returns (action, trigger, metrics)."""
    cfg = cfg or MB_CONFIG_C
    if df_5m is None or len(df_5m) < max(EXTREME_ATR_PERIOD + 3, MB_LOOKBACK_BARS + 3):
        return "HOLD", "Insufficient M5 history", {}
    d = _extreme_indicators(df_5m, cfg)
    return find_mother_bar_signal(d, cfg)

def _extreme_trade_plan(action: str, price: float, atr: float, df: pd.DataFrame, metrics: dict, trigger: str = "", cfg: dict = None):
    """Target/stop sizing driven by the detected Mother Bar's OWN range,
    not a fixed ATR multiple -- this is what makes target distance follow
    the size of the actual candle structure that produced the signal.

    `trigger` lets specific entry conditions override the TP multiples --
    see cfg["momentum_tp1_mult"]/cfg["momentum_tp2_mult"] above.

    Returns (sl, tp1, tp2, risk), or (None, None, None, None) if cfg uses a
    hard risk-reject (Bot A) and this trade's risk is too large -- caller
    must treat that as HOLD, not open a position."""
    cfg = cfg or MB_CONFIG_C
    mb_high = metrics.get("mb_high")
    mb_low = metrics.get("mb_low")
    mb_range = metrics.get("mb_range")

    tp1_mult, tp2_mult = cfg["tp1_mult"], cfg["tp2_mult"]
    if trigger == "MB Momentum Break":
        tp1_mult, tp2_mult = cfg["momentum_tp1_mult"], cfg["momentum_tp2_mult"]

    if mb_high is None or mb_low is None or not mb_range or mb_range <= 0:
        # Defensive fallback (should not normally trigger -- every returned
        # signal carries MB metrics) so a malformed call never crashes.
        risk = max(0.01, atr * 0.70)
        if action == "BUY":
            return price - risk, price + risk * 1.0, price + risk * 1.5, risk
        return price + risk, price - risk * 1.0, price - risk * 1.5, risk

    buffer = cfg["sl_buffer_atr"] * atr
    if action == "BUY":
        sl = mb_low - buffer
        tp1 = mb_high + tp1_mult * mb_range
        tp2 = mb_high + tp2_mult * mb_range
    else:
        sl = mb_high + buffer
        tp1 = mb_low - tp1_mult * mb_range
        tp2 = mb_low - tp2_mult * mb_range
    risk = abs(price - sl)

    if cfg.get("reject_risk_floor") is not None:
        # Bot A: hard reject instead of capping -- max($15, 2.25x ATR)
        reject_ceiling = max(cfg["reject_risk_floor"], cfg["reject_risk_atr_mult"] * atr)
        if risk > reject_ceiling:
            return None, None, None, None
    elif cfg.get("max_risk_price") is not None and risk > cfg["max_risk_price"]:
        # Bot C (unchanged): pull the stop in to the cap rather than
        # accepting the MB's full (possibly news-spiked) range as risk.
        # TP1/TP2 stay anchored to the true mb_high/mb_low.
        risk = cfg["max_risk_price"]
        sl = price - risk if action == "BUY" else price + risk

    return sl, tp1, tp2, risk

def _mt5_cache_fresh(cache: dict = None) -> bool:
    cache = cache if cache is not None else mt5_market_cache
    ts = cache.get("updated_at")
    if cache.get("df") is None or ts is None:
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() <= MT5_DATA_CACHE_TTL_SECONDS

def _df_from_mt5_payload(payload: dict, max_bars: int = 150):
    """Accept either {'bars':[...]} or {'candles':[...]} with OHLC and time."""
    rows = payload.get("bars") or payload.get("candles") or []
    if not isinstance(rows, list) or not rows:
        return None
    out = pd.DataFrame(rows)
    time_col = next((c for c in ("datetime", "time", "timestamp") if c in out.columns), None)
    if time_col is None:
        return None
    out["datetime"] = pd.to_datetime(out[time_col], unit="s", errors="coerce")
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
    return out.sort_values("datetime").drop_duplicates("datetime").tail(max_bars).reset_index(drop=True)

async def evaluate_extreme_strategy(client: httpx.AsyncClient, market_df_5m: pd.DataFrame, now_wib: datetime):
    """Strategy C paper engine: Mother Bar micro-breakout. Prefers fresh
    MT5 data (ideally M1 for true extreme frequency -- see status note in
    /status); falls back to shared TD M5."""
    _extreme_roll_day(now_wib)
    if not _extreme_session_ok(now_wib):
        return

    if EXTREME_MAX_TRADES_PER_DAY is not None and _extreme_state["trades_today"] >= EXTREME_MAX_TRADES_PER_DAY:
        return

    # Daily -6R breaker REMOVED for C (2026-09-24, your request) -- C now
    # trades every valid Mother Bar micro setup all session long regardless
    # of how the day's gone so far, same as A. No other execution/risk rule
    # changed (trade cap, cooldown, re-entry spacing all unchanged).

    source = "MT5"
    df = mt5_market_cache["df"] if _mt5_cache_fresh() else market_df_5m
    if df is market_df_5m:
        source = "TWELVE_DATA_FALLBACK"

    action, trigger, metrics = detect_extreme_m5_signal(df)
    if action == "HOLD":
        return
    if EXTREME_DIRECTION_MODE == "BUY_ONLY" and action == "SELL":
        return
    if EXTREME_DIRECTION_MODE == "SELL_ONLY" and action == "BUY":
        return

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

    # Deterministic price-action engine -- no LLM veto, same reasoning as
    # before: an AI call per bar would become the bottleneck and defeat
    # the point of an extreme-frequency engine.
    sl, tp1, tp2, risk = _extreme_trade_plan(action, price, atr, df, metrics, trigger)

    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection(); cur = conn.cursor()
        adx_val = metrics.get('adx', 0.0)
        gate_allow = metrics.get('adx_gate_would_allow')
        gate_zone = metrics.get('adx_gate_zone', 'unknown')
        gate_label = f"ADX_GATE_SHADOW={'ALLOW' if gate_allow else 'BLOCK'}_{gate_zone.upper()}" if gate_allow is not None else "ADX_GATE_SHADOW=N/A"
        reasoning = (
            f"{trigger}; MB range={metrics.get('mb_range', 0):.3f}, "
            f"inside_bars={metrics.get('inside_bars', 0)}, mb_age={metrics.get('mb_age_bars', 0)} bars, "
            f"ATR={atr:.3f}, ADX={adx_val:.1f}, {gate_label}, data={source}."
        )
        cur.execute("""
            INSERT INTO signals (
                timestamp,status,action,trigger_type,price,entry_price,sl,sl_price,
                tp1,tp1_price,tp2,tp2_price,confidence,adx_15m,stoch_rsi_15m,
                divergence_type,reasoning,outcome,outcome_timestamp,trend_15m,
                adx_15m_true,regime,strategy,execution_mode,created_at
            )
            VALUES (%s,'EXECUTED',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,
                    %s,%s,'PENDING','',%s,%s,%s,%s,%s,NOW())
            RETURNING id
        """, (
            (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB"),
            # DATA SOURCE FIX: `regime` used to just re-store the strategy
            # name ("MOTHER_BAR_MICRO" -- identical for every row, so it was
            # dead weight). It now holds the actual feed used for this bar
            # ("MT5" or "TWELVE_DATA_FALLBACK") so /analyze can break
            # performance down by data source. Still also embedded in
            # `reasoning`'s free text as before, for readability in /last.
            # ADX SHADOW FIX: `adx_15m`/`adx_15m_true` used to be hardcoded 0
            # for every C row -- now hold the real ADX reading. `divergence_type`
            # used to be hardcoded 'None' -- now holds the ADX gate's shadow
            # verdict (ALLOW/BLOCK + zone) so a future /analyze breakdown can
            # compare shadow-gated vs actual outcomes without parsing `reasoning`.
            action, trigger, price, price, sl, sl, tp1, tp1, tp2, tp2,
            0.90, adx_val, gate_label, reasoning, None, adx_val, source, BREAKOUT_STRATEGY, BREAKOUT_EXECUTION_MODE
        ))
        row = cur.fetchone(); sid = int(row["id"]) if row else None
        conn.commit(); cur.close(); conn.close()
        _extreme_state["last_signal_time"] = now_utc
        _extreme_state["last_entry_price"] = price
        _extreme_state["last_action"] = action
        _extreme_state["trades_today"] += 1

        risk_pct_of_range = (risk / metrics.get("mb_range", risk)) if metrics.get("mb_range") else 1.0
        mode_tag = "[PAPER]" if BREAKOUT_EXECUTION_MODE == "PAPER" else "[LIVE]"
        mode_footer = "⚠️ PAPER ONLY" if BREAKOUT_EXECUTION_MODE == "PAPER" else "🔴 LIVE -- MT5 will execute this"
        tp1_mult, tp2_mult = (
            (MB_MOMENTUM_BREAK_TP1_MULT, MB_MOMENTUM_BREAK_TP2_MULT)
            if trigger == "MB Momentum Break" else (MB_TP1_MULT, MB_TP2_MULT)
        )
        await send_telegram_alert(
            client,
            f"⚡ *MOTHER BAR C {BREAKOUT_STRATEGY} {mode_tag} SIGNAL #{sid}*\n\n"
            f"Asset: *XAU/USD*\nAction: *{action}*\nTrigger: *{trigger}*\n"
            f"Entry: *${price:.2f}*\nSL: *${sl:.2f}* (opp. MB extreme, {risk:.3f} risk)\n"
            f"TP1: *${tp1:.2f}* ({tp1_mult:.1f}x MB range)\nTP2: *${tp2:.2f}* ({tp2_mult:.1f}x MB range)\n"
            f"MB range: *{metrics.get('mb_range', 0):.3f}* | Inside bars: *{metrics.get('inside_bars', 0)}* | "
            f"MB age: *{metrics.get('mb_age_bars', 0)} bars*\n"
            f"ATR: *{atr:.3f}* | ADX: *{metrics.get('adx', 0):.1f}* "
            f"(gate shadow: {'✅ would allow' if gate_allow else '⛔ would block'} [{gate_zone}])\n"
            f"Data source: *{source}*\n"
            f"Daily C signals: *{_extreme_state['trades_today']}/{EXTREME_MAX_TRADES_PER_DAY_LABEL}*\n\n"
            f"{mode_footer}",
            BREAKOUT_TELEGRAM_CHAT_ID, BREAKOUT_TELEGRAM_BOT_TOKEN
        )
    except Exception as e:
        logging.error(f"[MB C DB ERROR] {e}")

# =====================================================================
# BOT A -- MOTHER BAR V2 (M1), now the sole MT5-live strategy
# =====================================================================
_control_mb_state = {
    "last_signal_time": None,
    "last_entry_price": None,
    "last_action": None,
    "trades_today": 0,
    "trade_day": None,
}

async def get_m1_dataframe(client: httpx.AsyncClient, now_wib: datetime):
    """Free, no-payment M1 source for Bot A: prefer the MT5 EA push
    (mt5_market_cache_m1 -- zero API cost, no rate limit); only fall back to
    a throttled Twelve Data 1min pull, capped well under the shared 800/day
    budget so a stalled MT5 feed can't starve A/B/C's core 5-minute cycle.
    Returns (df, source_label) or (None, reason)."""
    if _mt5_cache_fresh(mt5_market_cache_m1):
        return mt5_market_cache_m1["df"], "MT5"
    if _td_1min_fallback_ok(now_wib):
        df = await fetch_timeframe_data(client, "1min", outputsize=300, now_wib=now_wib)
        if df is not None and len(df) >= MB_LOOKBACK_BARS + 3:
            _note_td_1min_fallback_call()
            return df, "TWELVE_DATA_FALLBACK"
        _note_td_1min_fallback_call()  # still counts against the cap even on a bad/empty response
    return None, "NO_DATA"

async def evaluate_control_mb_strategy(client: httpx.AsyncClient, now_wib: datetime, df: pd.DataFrame = None, source: str = None):
    """Strategy A live engine: Mother Bar V2, M1, all 6 triggers, hard
    risk-reject, no daily trade cap, -6R daily breaker. This replaces the
    old harmonic-pattern A entirely. `df`/`source` may be passed in by the
    caller (the M1 loop branch, which also needs the same bar for
    update_open_trades) to avoid fetching twice."""
    _extreme_roll_day(now_wib, _control_mb_state)
    # No session gate by spec ("no 60/day limit" implies always-on like C);
    # still respects the actual forex market being open.
    if not is_forex_market_open(now_wib):
        return

    # Daily -6R breaker REMOVED for A (2026-09-24, your request) -- A now
    # trades every valid Mother Bar V2 setup all session long regardless of
    # how the day's gone so far. No other execution/risk rule changed: the
    # per-trade reject-risk cap (max($15, 2.25x ATR)), 0.20 ATR re-entry
    # spacing, and no-daily-trade-cap are all still exactly as before.
    # C's -6R breaker was also REMOVED (2026-09-24, your request) -- see
    # evaluate_extreme_strategy(). Both A and C now trade every valid setup
    # all session long regardless of the day's running R.

    df, source = (df, source) if df is not None else await get_m1_dataframe(client, now_wib)
    if df is None:
        logging.warning("[MB A] No M1 data available (MT5 push stale, TD fallback exhausted/cooling) -- holding.")
        return

    cfg = MB_CONFIG_A_V2
    action, trigger, metrics = detect_extreme_m5_signal(df, cfg)  # name kept generic; works on any timeframe fed in
    if action == "HOLD":
        return
    if CONTROL_DIRECTION_MODE == "BUY_ONLY" and action == "SELL":
        return
    if CONTROL_DIRECTION_MODE == "SELL_ONLY" and action == "BUY":
        return

    now_utc = datetime.now(timezone.utc)
    last_ts = _control_mb_state.get("last_signal_time")
    if last_ts and (now_utc - last_ts).total_seconds() < EXTREME_COOLDOWN_SECONDS:
        return

    price = float(df["close"].iloc[-1])
    atr = float(metrics.get("atr") or 0.0)
    if atr <= 0:
        return

    last_price = _control_mb_state.get("last_entry_price")
    if last_price is not None and abs(price - float(last_price)) < EXTREME_MIN_REENTRY_DISTANCE_ATR * atr:
        return

    sl, tp1, tp2, risk = _extreme_trade_plan(action, price, atr, df, metrics, trigger, cfg)
    if sl is None:
        # Hard risk-reject: risk > max($15, 2.25x ATR) -- skip this trade
        # entirely rather than shrinking the stop (that's C's behavior, not A's).
        reject_ceiling = max(cfg["reject_risk_floor"], cfg["reject_risk_atr_mult"] * atr)
        logging.info(f"[MB A] [RISK REJECT] {trigger} risk too large vs ceiling ${reject_ceiling:.2f} -- skipped.")
        return

    if not DATABASE_URL:
        return

    try:
        conn = get_db_connection(); cur = conn.cursor()
        adx_val = metrics.get('adx', 0.0)
        gate_allow = metrics.get('adx_gate_would_allow')
        gate_zone = metrics.get('adx_gate_zone', 'unknown')
        gate_label = f"ADX_GATE_SHADOW={'ALLOW' if gate_allow else 'BLOCK'}_{gate_zone.upper()}" if gate_allow is not None else "ADX_GATE_SHADOW=N/A"
        reasoning = (
            f"{trigger}; MB range={metrics.get('mb_range', 0):.3f}, "
            f"inside_bars={metrics.get('inside_bars', 0)}, mb_age={metrics.get('mb_age_bars', 0)} bars, "
            f"ATR={atr:.3f}, ADX={adx_val:.1f}, {gate_label}, data={source}, risk=${risk:.2f}."
        )
        cur.execute("""
            INSERT INTO signals (
                timestamp,status,action,trigger_type,price,entry_price,sl,sl_price,
                tp1,tp1_price,tp2,tp2_price,confidence,adx_15m,stoch_rsi_15m,
                divergence_type,reasoning,outcome,outcome_timestamp,trend_15m,
                adx_15m_true,regime,strategy,execution_mode,created_at
            )
            VALUES (%s,'EXECUTED',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,
                    %s,%s,'PENDING','',%s,%s,%s,%s,%s,NOW())
            RETURNING id
        """, (
            (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB"),
            action, trigger, price, price, sl, sl, tp1, tp1, tp2, tp2,
            0.90, adx_val, gate_label, reasoning, None, adx_val, source, CONTROL_STRATEGY, CONTROL_EXECUTION_MODE
        ))
        row = cur.fetchone(); sid = int(row["id"]) if row else None
        conn.commit(); cur.close(); conn.close()
        _control_mb_state["last_signal_time"] = now_utc
        _control_mb_state["last_entry_price"] = price
        _control_mb_state["last_action"] = action
        _control_mb_state["trades_today"] += 1

        mode_tag = "[PAPER]" if CONTROL_EXECUTION_MODE == "PAPER" else "[LIVE]"
        mode_footer = "⚠️ PAPER ONLY" if CONTROL_EXECUTION_MODE == "PAPER" else "🔴 LIVE -- MT5 will execute this"
        tp1_mult, tp2_mult = (
            (cfg["momentum_tp1_mult"], cfg["momentum_tp2_mult"])
            if trigger == "MB Momentum Break" else (cfg["tp1_mult"], cfg["tp2_mult"])
        )
        await send_telegram_alert(
            client,
            f"🅰️ *MOTHER BAR A-V2 {CONTROL_STRATEGY} {mode_tag} SIGNAL #{sid}*\n\n"
            f"Asset: *XAU/USD* (M1)\nAction: *{action}*\nTrigger: *{trigger}*\n"
            f"Entry: *${price:.2f}*\nSL: *${sl:.2f}* (risk ${risk:.2f})\n"
            f"TP1: *${tp1:.2f}* ({tp1_mult:.2f}x MB range)\nTP2: *${tp2:.2f}* ({tp2_mult:.2f}x MB range)\n"
            f"MB range: *{metrics.get('mb_range', 0):.3f}* | Inside bars: *{metrics.get('inside_bars', 0)}* | "
            f"MB age: *{metrics.get('mb_age_bars', 0)} bars*\n"
            f"ATR: *{atr:.3f}* | ADX: *{metrics.get('adx', 0):.1f}*\n"
            f"Data source: *{source}*\n"
            f"Signals today: *{_control_mb_state['trades_today']}* (no daily cap)\n\n"
            f"{mode_footer}",
            TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN
        )
    except Exception as e:
        logging.error(f"[MB A DB ERROR] {e}")

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
    elif strategy_mode == "HARMONIC":
        strategy_desc = "XABCD Harmonic Pattern Completion (Gartley/Bat/Butterfly/Crab, 5M Execution)"
        range_text = (
            f"5. Pattern Points: X=${range_high:.2f}, A=${range_low:.2f} "
            f"(D completing now at ${current_price:.2f}; SL sits just beyond X, TP2 targets A)"
        ) if range_high else ""
        veto_rules_text = (
            "- VETO if this looks like a fresh impulsive breakout rather than an exhaustion reversal at "
            "the pattern's completion (D) point -- harmonic trades fade momentum, they don't chase it.\n"
            "- VETO if the reversal candle is weak/indecisive (small body, long opposing wick) given how "
            "far price has already traveled through the XA-AB-BC-CD structure.\n"
            "- This is a reversal trade against the CD leg -- do NOT expect trend-style follow-through "
            "past point A; the primary target IS point A."
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

    if strategy == CONTROL_STRATEGY:
        # A no longer branches on ADX at all -- the harmonic detector defines
        # its own XABCD structure and only cares whether price is currently
        # inside a computed Potential Reversal Zone with a confirming candle.
        # TREND/RANGE ADX split below is B-only now.
        strategy_mode = "HARMONIC"
        proposed_action, trigger_type, range_high, range_low = detect_harmonic_signal(df_5m)
    elif adx_5m >= RANGE_MODE_ADX_MAX:
        strategy_mode = "TREND"
        proposed_action, trigger_type = detect_ema_signal(df_5m, trend_15m, ema_fast, ema_slow)
    else:
        strategy_mode = "RANGE"
        proposed_action, trigger_type, range_high, range_low = detect_range_reversal(df_5m, adx_15m_true)

    # B V2: use a clean trend-only entry regime. The range engine and
    # aggressive EMA impulse/crossover triggers are intentionally not mixed
    # into this experiment. The backtest showed the useful B edge is the
    # confirmed EMA5 pullback/touch, not every EMA event.
    if strategy == EXPERIMENTAL_STRATEGY:
        if adx_5m < EXPERIMENTAL_MIN_ADX:
            strategy_mode = "CHOP"
            proposed_action = "HOLD"
            trigger_type = f"B V3 Trend-Quality Block (ADX {adx_5m:.1f} < {EXPERIMENTAL_MIN_ADX:.0f})"
        elif proposed_action == "BUY":
            spread_atr = abs(curr_ema_fast - curr_ema_slow) / max(float(df_5m["atr"].iloc[-1]), 1e-9)
            slope_atr = (curr_ema_fast - float(df_5m["ema_fast"].iloc[-3])) / max(float(df_5m["atr"].iloc[-1]), 1e-9) if len(df_5m) >= 3 else 0.0
            extension_atr = (curr_price - curr_ema_fast) / max(float(df_5m["atr"].iloc[-1]), 1e-9)
            regime_metrics.update({"ema_spread_atr": round(spread_atr, 4), "ema_slope_atr": round(slope_atr, 4), "entry_extension_atr": round(extension_atr, 4)})
            if EXPERIMENTAL_TOUCH_ONLY and "Line Touch" not in trigger_type:
                proposed_action = "HOLD"
                trigger_type = "B V3 Trigger Quality Block"
            elif spread_atr < EXPERIMENTAL_MIN_EMA_SPREAD_ATR or slope_atr <= EXPERIMENTAL_MIN_EMA_SLOPE_ATR or extension_atr > EXPERIMENTAL_MAX_ENTRY_EXTENSION_ATR:
                proposed_action = "HOLD"
                trigger_type = "B V3 Trend Quality Block"
            elif float(df_5m["close"].iloc[-2]) <= float(df_5m["ema_fast"].iloc[-2]):
                proposed_action = "HOLD"
                trigger_type = "B V3 Pullback-Structure Block"

    # B-only regime fix:
    # Efficiency is a HARD veto, not a request to find an alternative S/R trade.
    # The previous code detected high-ADX chop but then allowed S/R RANGE/TREND
    # signals to replace the EMA setup, which made B progressively less like a
    # clean EMA 5/15 experiment and could add trades exactly when efficiency
    # said the market was choppy.
    #
    # For this controlled patch we intentionally disable both S/R fallbacks.
    # This isolates the effect of the regime fix. S/R can be re-tested later as
    # a separate experiment rather than being mixed into the same result.
    if strategy == EXPERIMENTAL_STRATEGY and strategy_mode == "TREND":
        efficiency = compute_efficiency_ratio(df_5m)
        regime_metrics["efficiency"] = round(float(efficiency), 4)
        if efficiency < EXPERIMENTAL_EFFICIENCY_MAX:
            strategy_mode = "CHOP"
            proposed_action = "HOLD"
            trigger_type = f"Efficiency Chop Block ({efficiency:.2f})"
            log_scan_event(
                "EFFICIENCY_CHOP_BLOCK", stage="REGIME", action="HOLD",
                price=curr_price, adx_5m=adx_5m, adx_15m=adx_15m_true,
                trend_15m=trend_15m, decision="HOLD",
                reason=f"{strategy}: 5M efficiency {efficiency:.2f} < {EXPERIMENTAL_EFFICIENCY_MAX:.2f}",
                details={"efficiency": round(float(efficiency), 4)}
            )

    # Dead for A as of the harmonic-pattern replacement: CONTROL_STRATEGY now
    # always runs strategy_mode == "HARMONIC" (set above), never "TREND", so
    # this condition can no longer be true for A. Left in place rather than
    # deleted since CONTROL_TREND_ADX_MIN and the guard are still referenced
    # in a few status/analytics strings elsewhere; harmless no-op.
    if strategy == CONTROL_STRATEGY and strategy_mode == "TREND" and adx_5m < CONTROL_TREND_ADX_MIN:
        logging.info(f"[{strategy}] [STRICT REGIME FILTER] Holding: adx_5m={adx_5m:.1f} below {CONTROL_TREND_ADX_MIN}")
        proposed_action = "HOLD"

    # A/B one-direction controller -- now fully independent per strategy.
    # Default is BUY_ONLY: the old dynamic 1H one-direction system is OFF.
    # /oneway_on switches back to DYNAMIC; /oneway_off returns to BUY_ONLY.
    # /both disables the one-direction restriction entirely.
    if strategy in (CONTROL_STRATEGY, EXPERIMENTAL_STRATEGY):
        active_direction_mode = CONTROL_DIRECTION_MODE if strategy == CONTROL_STRATEGY else EXPERIMENTAL_DIRECTION_MODE
        if active_direction_mode == "BUY_ONLY" and proposed_action == "SELL":
            logging.info(f"[{strategy}] [BUY-ONLY] SELL blocked.")
            proposed_action = "HOLD"
        elif active_direction_mode == "DYNAMIC":
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
        if strategy_mode == "BRACKET":
            range_str = f"Range: ${range_low:.2f}-${range_high:.2f}" if (range_high is not None and range_low is not None) else "Range: n/a"
            logging.info(
                f"[{strategy}] [MARKET SCAN] Price: ${curr_price:.2f} | {range_str} | "
                f"ADX5m: {adx_5m:.1f} | Mode: {strategy_mode} | Reason: {trigger_type} | Status: HOLD"
            )
        elif strategy_mode == "HARMONIC":
            xa_str = f"X: ${range_high:.2f} A: ${range_low:.2f}" if (range_high is not None and range_low is not None) else "X/A: n/a"
            logging.info(
                f"[{strategy}] [MARKET SCAN] Price: ${curr_price:.2f} | {xa_str} | "
                f"ADX5m: {adx_5m:.1f} | Mode: {strategy_mode} | Reason: {trigger_type} | Status: HOLD"
            )
        else:
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

    if trigger_type and str(trigger_type).startswith("S/R Retest"):
        # TREND-mode break-and-retest: SL goes beyond the actual retested
        # level (structure), not a generic ATR distance from entry price.
        level = range_high if proposed_action == "BUY" else range_low
        if level is None:
            level = curr_price - atr_5m if proposed_action == "BUY" else curr_price + atr_5m
        if proposed_action == "BUY":
            sl_price = level - SR_TREND_SL_BUFFER_ATR * atr_5m
            risk = max(0.5, curr_price - sl_price)
            tp1_price = curr_price + risk * SR_TREND_TP1_R
            tp2_price = curr_price + risk * SR_TREND_TP2_R
        else:
            sl_price = level + SR_TREND_SL_BUFFER_ATR * atr_5m
            risk = max(0.5, sl_price - curr_price)
            tp1_price = curr_price - risk * SR_TREND_TP1_R
            tp2_price = curr_price - risk * SR_TREND_TP2_R
        tp1_r_mult, tp2_r_mult = SR_TREND_TP1_R, SR_TREND_TP2_R
    elif trigger_type and str(trigger_type).startswith("S/R Fade"):
        # RANGE-mode fade: SL beyond the touched level, TP2 targets the
        # opposite structural level when there is one, TP1 is a fixed-R
        # partial toward it. Fixed risk throughout -- no size scaling.
        if proposed_action == "BUY":
            level = range_low  # support just touched
            sl_price = level - SR_RANGE_SL_BUFFER_ATR * atr_5m
            risk = max(0.5, curr_price - sl_price)
            tp1_price = curr_price + risk * SR_RANGE_TP1_R
            tp2_price = range_high if (range_high and range_high > tp1_price) else curr_price + risk * SR_RANGE_TP2_R
        else:
            level = range_high  # resistance just touched
            sl_price = level + SR_RANGE_SL_BUFFER_ATR * atr_5m
            risk = max(0.5, sl_price - curr_price)
            tp1_price = curr_price - risk * SR_RANGE_TP1_R
            tp2_price = range_low if (range_low and range_low < tp1_price) else curr_price - risk * SR_RANGE_TP2_R
        tp1_r_mult = abs(tp1_price - curr_price) / max(abs(curr_price - sl_price), 0.01)
        tp2_r_mult = abs(tp2_price - curr_price) / max(abs(curr_price - sl_price), 0.01)
    elif strategy_mode == "RANGE":
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
    elif strategy_mode == "BRACKET":
        # SL sits just back inside the broken level (not across the whole
        # range) -- a genuine confirmed-close breakout that immediately
        # gives the level back is treated as a fast invalidation, not
        # something to ride out. TP is a measured-move projection: the
        # range's own height, projected forward from the breakout point --
        # the classic range-breakout target, and it scales with whatever
        # size range actually formed instead of a fixed ATR multiple.
        range_width = abs(range_high - range_low) if (range_high is not None and range_low is not None) else atr_5m * 3.0
        if proposed_action == "BUY":
            sl_price = range_high - BRACKET_SL_BUFFER_ATR * atr_5m
        else:
            sl_price = range_low + BRACKET_SL_BUFFER_ATR * atr_5m

        min_risk = max(BRACKET_MIN_RISK_ATR * atr_5m, 1.0)
        risk = abs(curr_price - sl_price)
        if risk < min_risk:
            # Breakout candle barely cleared the level -- widen the stop to a
            # sane floor rather than leaving it inside normal spread/noise.
            sl_price = curr_price - min_risk if proposed_action == "BUY" else curr_price + min_risk
            risk = min_risk

        if proposed_action == "BUY":
            tp1_price = curr_price + range_width * BRACKET_TP1_RANGE_MULT
            tp2_price = curr_price + range_width * BRACKET_TP2_RANGE_MULT
        else:
            tp1_price = curr_price - range_width * BRACKET_TP1_RANGE_MULT
            tp2_price = curr_price - range_width * BRACKET_TP2_RANGE_MULT
        tp1_r_mult = abs(tp1_price - curr_price) / max(risk, 0.01)
        tp2_r_mult = abs(tp2_price - curr_price) / max(risk, 0.01)
    elif strategy_mode == "HARMONIC":
        # SL sits just beyond point X (the pattern's structural invalidation
        # level -- if X gets taken out, this was never really the pattern).
        # TP1 is a partial back toward point A (first Fib retracement of the
        # D->A leg); TP2 is point A itself, the classic harmonic profit
        # target. Both projections fall naturally out of X/A -- no fixed
        # ATR multiple needed, same measured-move spirit as the bracket TPs.
        harmonic_x, harmonic_a = range_high, range_low
        if harmonic_x is None or harmonic_a is None:
            # Shouldn't happen once a signal fires (detect_harmonic_signal
            # always returns X/A alongside BUY/SELL), but fall back to a
            # plain ATR risk rather than crash if it ever does.
            risk = max(2.5, atr_5m * 1.0)
            sl_price = curr_price - risk if proposed_action == "BUY" else curr_price + risk
            tp1_r_mult, tp2_r_mult = 1.5, 2.5
            tp1_price = curr_price + risk * tp1_r_mult if proposed_action == "BUY" else curr_price - risk * tp1_r_mult
            tp2_price = curr_price + risk * tp2_r_mult if proposed_action == "BUY" else curr_price - risk * tp2_r_mult
        else:
            if proposed_action == "BUY":
                sl_price = harmonic_x - HARMONIC_SL_BUFFER_ATR * atr_5m
            else:
                sl_price = harmonic_x + HARMONIC_SL_BUFFER_ATR * atr_5m

            min_risk = max(HARMONIC_MIN_RISK_ATR * atr_5m, 1.0)
            risk = abs(curr_price - sl_price)
            if risk < min_risk:
                sl_price = curr_price - min_risk if proposed_action == "BUY" else curr_price + min_risk
                risk = min_risk

            tp1_price = curr_price + (harmonic_a - curr_price) * HARMONIC_TP1_AD_FRACTION
            tp2_price = harmonic_a
            # Point A can end up on the wrong side of TP1 (or of entry) if
            # the D leg already ran unusually far -- fall back to a plain
            # R-multiple target rather than log a nonsensical TP2.
            wrong_side = (proposed_action == "BUY" and tp2_price <= tp1_price) or \
                         (proposed_action == "SELL" and tp2_price >= tp1_price)
            if wrong_side:
                tp2_price = curr_price + risk * HARMONIC_TP2_R_FALLBACK if proposed_action == "BUY" \
                    else curr_price - risk * HARMONIC_TP2_R_FALLBACK

            tp1_r_mult = abs(tp1_price - curr_price) / max(risk, 0.01)
            tp2_r_mult = abs(tp2_price - curr_price) / max(risk, 0.01)
    else:
        if strategy == EXPERIMENTAL_STRATEGY:
            # Short TP1 to bank the quick leg, wide TP2 since lot 2 is
            # already risk-free at breakeven past TP1, slightly wider SL
            # so ordinary noise doesn't stop it before the idea plays out.
            risk = max(2.5, atr_5m * EXPERIMENTAL_RISK_ATR_MULT)
            risk = min(risk, EXPERIMENTAL_MAX_RISK_PRICE)  # BUG FIX: see constant's comment above
            tp1_r_mult = EXPERIMENTAL_TP1_R
            tp2_r_mult = EXPERIMENTAL_TP2_R
        else:
            risk = max(2.5, atr_5m * 1.0)
            tp1_r_mult = 1.5
            tp2_r_mult = 2.5
        sl_price = curr_price - risk if proposed_action == "BUY" else curr_price + risk
        tp1_price = curr_price + risk * tp1_r_mult if proposed_action == "BUY" else curr_price - risk * tp1_r_mult
        tp2_price = curr_price + risk * tp2_r_mult if proposed_action == "BUY" else curr_price - risk * tp2_r_mult
        # SL is now capped for B (EXPERIMENTAL_MAX_RISK_PRICE) so a volatile-
        # ATR trade can't silently risk 3-4x a normal trade's $ under the
        # fixed 2x0.01-lot execution model. TPs are computed off the same
        # (possibly capped) risk, so R-multiples still line up with the
        # actual stop that gets sent to MT5. The 1H EMA200 bias filter above
        # (kept) still guards against a losing streak on a trend reversal by
        # blocking entries against the new direction, not by resizing stops.

    if ai_decision.action == proposed_action:
        new_id = log_trade_signal(
            "EXECUTED", proposed_action, trigger_type, curr_price, sl_price, tp1_price, tp2_price,
            float(ai_decision.confidence), adx_5m, 0.0, "None", ai_decision.reasoning,
            trend_15m, adx_15m_true, entry_extension_atr, entry_climax_ratio,
            strategy_mode, regime_metrics, strategy, execution_mode
        )
        mode_tag = (
            "📊 RANGE FADE" if strategy_mode == "RANGE" else
            "🧱 BRACKET" if strategy_mode == "BRACKET" else
            "🦋 HARMONIC" if strategy_mode == "HARMONIC" else
            "🚀 TREND"
        )
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
        last_claimed_minute_bucket_a = None

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

                # Bot A runs on its own M1 cadence, independent of B/C's
                # shared 5-minute block below -- it's the only MT5-live
                # strategy now, so it needs to see every 1-minute close.
                #
                # RECEIVER-SIDE FIX: update_open_trades() used to only run
                # once per 5 minutes (below, off the M5 candle), which is
                # fine for B/C but was silently going to smear every A
                # trade's TP1/TP2/SL detection out to 5-minute resolution
                # even though A now opens on M1 closes. It's called here too,
                # off the same M1 bar A itself is trading, so A's own
                # signals get checked every minute like everything else does
                # on its native timeframe. It's harmless to call twice in the
                # same wall-clock minute since it only acts on genuinely new
                # outcome transitions.
                current_minute_bucket = now_wib.strftime("%Y-%m-%d %H:%M")
                if current_minute_bucket != last_claimed_minute_bucket_a and now_wib.second < 20:
                    last_claimed_minute_bucket_a = current_minute_bucket
                    try:
                        df_m1, source_m1 = await get_m1_dataframe(client, now_wib)
                        if df_m1 is not None:
                            update_open_trades(float(df_m1["high"].iloc[-1]), float(df_m1["low"].iloc[-1]))
                            await evaluate_control_mb_strategy(client, now_wib, df_m1, source_m1)
                    except Exception as e:
                        logging.error(f"[MB A LOOP ERROR] {e}")

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

                # 1H data is fetched ONLY when at least one of A/B's dynamic
                # one-direction modes is enabled. If both are BUY_ONLY/BOTH,
                # zero Twelve Data credits are spent on the old directional system.
                directional_bias, bias_sep = "NEUTRAL", 0.0
                if CONTROL_DIRECTION_MODE == "DYNAMIC" or EXPERIMENTAL_DIRECTION_MODE == "DYNAMIC":
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

                # CONTROL A now runs on its own M1 cadence above
                # (evaluate_control_mb_strategy) -- Mother Bar V2, no
                # longer the shared-5M harmonic engine, no longer PAPER.

                # EXPERIMENT B: same system + same exhaustion guard, EMA 5/15, PAPER.
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
    global CONTROL_DIRECTION_MODE, EXPERIMENTAL_DIRECTION_MODE, EXTREME_DIRECTION_MODE
    global CONTROL_EXECUTION_MODE, EXPERIMENTAL_EXECUTION_MODE, BREAKOUT_EXECUTION_MODE
    init_db()
    # Restore direction-mode toggles (/oneway_on, /c_both, etc.) set before the
    # last restart -- these used to live only in memory and silently reset to
    # BUY_ONLY / BOTH on every redeploy or crash.
    CONTROL_DIRECTION_MODE = get_setting("control_direction_mode", CONTROL_DIRECTION_MODE)
    EXPERIMENTAL_DIRECTION_MODE = get_setting("experimental_direction_mode", EXPERIMENTAL_DIRECTION_MODE)
    EXTREME_DIRECTION_MODE = get_setting("extreme_direction_mode", EXTREME_DIRECTION_MODE)
    logging.info(
        f"[SETTINGS] Restored direction modes -- Control: {CONTROL_DIRECTION_MODE}, "
        f"Experimental: {EXPERIMENTAL_DIRECTION_MODE}, Extreme: {EXTREME_DIRECTION_MODE}"
    )
    # Restore which of A/B/C is LIVE (/live_a, /live_b, /live_c) set before the
    # last restart -- same reasoning as direction modes above. Falls back to
    # the hardcoded defaults above (A=LIVE, B/C=PAPER) on a fresh DB, which
    # matches the constants as written so a first deploy behaves exactly as
    # the file says with no settings yet persisted.
    CONTROL_EXECUTION_MODE = get_setting("control_execution_mode", CONTROL_EXECUTION_MODE)
    EXPERIMENTAL_EXECUTION_MODE = get_setting("experimental_execution_mode", EXPERIMENTAL_EXECUTION_MODE)
    BREAKOUT_EXECUTION_MODE = get_setting("breakout_execution_mode", BREAKOUT_EXECUTION_MODE)
    logging.info(
        f"[SETTINGS] Restored execution modes -- A: {CONTROL_EXECUTION_MODE}, "
        f"B: {EXPERIMENTAL_EXECUTION_MODE}, C: {BREAKOUT_EXECUTION_MODE}"
    )
    # Restore Bot C's per-trigger toggles (/c_fib25_on, /c_momentum_off, etc.)
    # set before the last restart, then rebuild MB_CONFIG_C from them. A is
    # unaffected -- MB_CONFIG_A_V2 always runs all 6 triggers regardless.
    for _key in _c_trigger_state:
        _c_trigger_state[_key] = get_setting(f"c_trigger_{_key}", "on" if _c_trigger_state[_key] else "off") == "on"
    _apply_c_trigger_state()
    logging.info(f"[SETTINGS] Restored Bot C trigger toggles -- {_c_trigger_state}")
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
                        {"command":"stats","description":"Mother Bar A (M1) performance"},
                        {"command":"pips","description":"Pips and USD breakdown"},
                        {"command":"logs","description":"Last 10 executed trades"},
                        {"command":"analyze","description":"Entry condition + ADX gate analysis"},
                        {"command":"pause","description":"Emergency kill switch (all bots)"},
                        {"command":"resume","description":"Resume auto-trading"},
                        {"command":"oneway_on","description":"Enable dynamic 1H one-direction"},
                        {"command":"oneway_off","description":"Disable dynamic mode; BUY only"},
                        {"command":"both","description":"Allow BUY and SELL"},
                        {"command":"macro","description":"Gold macro context (yields, USD, COT)"},
                        {"command":"live_a","description":"Send A's signals to live MT5"},
                        {"command":"live_b","description":"Send B's signals to live MT5"},
                        {"command":"live_c","description":"Send C's signals to live MT5"}
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
                        {"command":"compare","description":"Compare Mother Bar A vs EMA B vs C"},
                        {"command":"status","description":"Read-only system status"},
                        {"command":"last","description":"Last 10 trades"},
                        {"command":"oneway_on","description":"Enable dynamic 1H one-direction"},
                        {"command":"oneway_off","description":"Disable dynamic mode; BUY only"},
                        {"command":"both","description":"Allow BUY and SELL"},
                        {"command":"macro","description":"Gold macro context (yields, USD, COT)"},
                        {"command":"live_a","description":"Send A's signals to live MT5"},
                        {"command":"live_b","description":"Send B's signals to live MT5"},
                        {"command":"live_c","description":"Send C's signals to live MT5"}
                    ]})
                if BREAKOUT_TELEGRAM_BOT_TOKEN:
                    webhook_c = f"{APP_URL.rstrip('/')}/telegram-webhook-c"
                    set_c = f"https://api.telegram.org/bot{BREAKOUT_TELEGRAM_BOT_TOKEN}/setWebhook"
                    res_c = await client.post(set_c, data={"url": webhook_c})
                    logging.info(f"[BREAKOUT WEBHOOK SETUP] {res_c.text}")
                    await client.post(f"https://api.telegram.org/bot{BREAKOUT_TELEGRAM_BOT_TOKEN}/setMyCommands", json={"commands":[
                        {"command":"start","description":"Show Mother Bar C bot commands"},
                        {"command":"help","description":"Show Mother Bar C command menu"},
                        {"command":"status","description":"Mother Bar C status"},
                        {"command":"stats","description":"Mother Bar C performance"},
                        {"command":"pips","description":"Pips/earnings report (TP1/TP2/SL)"},
                        {"command":"analyze","description":"Entry condition category breakdown"},
                        {"command":"last","description":"Last Mother Bar signals"},
                        {"command":"c_both","description":"Allow both BUY and SELL"},
                        {"command":"c_buyonly","description":"Restrict C to BUY only"},
                        {"command":"c_sellonly","description":"Restrict C to SELL only"},
                        {"command":"c_fib25_on","description":"Enable Fib Retrace 25% entry"},
                        {"command":"c_fib25_off","description":"Disable Fib Retrace 25% entry"},
                        {"command":"c_fib50_on","description":"Enable Fib Retrace 50% entry"},
                        {"command":"c_fib50_off","description":"Disable Fib Retrace 50% entry"},
                        {"command":"c_fib618_on","description":"Enable Fib Retrace 62% (golden ratio) entry"},
                        {"command":"c_fib618_off","description":"Disable Fib Retrace 62% (golden ratio) entry"},
                        {"command":"c_fib75_on","description":"Enable Fib Retrace 75% entry"},
                        {"command":"c_fib75_off","description":"Disable Fib Retrace 75% entry"},
                        {"command":"c_closebreak_on","description":"Enable Close Break entry"},
                        {"command":"c_closebreak_off","description":"Disable Close Break entry"},
                        {"command":"c_momentum_on","description":"Enable Momentum Break entry"},
                        {"command":"c_momentum_off","description":"Disable Momentum Break entry"},
                        {"command":"macro","description":"Gold macro context (yields, USD, COT)"},
                        {"command":"live_a","description":"Send A's signals to live MT5"},
                        {"command":"live_b","description":"Send B's signals to live MT5"},
                        {"command":"live_c","description":"Send C's signals to live MT5"}
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
    """MT5 EA pushes free, no-rate-limit bars here. Include a "timeframe"
    field ("M1" or "M5") in the JSON body to route to the right cache --
    defaults to "M5" for backward compatibility with EAs that don't send it.
    Bot A (M1, live) needs its own EA timer object pushing M1 bars every
    close; Bot C keeps using the M5 push exactly as before."""
    global mt5_market_cache, mt5_market_cache_m1
    try:
        if MT5_DATA_SECRET:
            supplied = request.headers.get("X-MT5-SECRET", "")
            if supplied != MT5_DATA_SECRET:
                return {"ok": False, "error": "unauthorized"}
        payload = await request.json()
        timeframe = str(payload.get("timeframe", "M5")).upper()
        df = _df_from_mt5_payload(payload, max_bars=300 if timeframe == "M1" else 150)
        min_bars = MB_LOOKBACK_BARS + 3 if timeframe == "M1" else EXTREME_ATR_PERIOD + 3
        if df is None or len(df) < min_bars:
            return {"ok": False, "error": f"invalid_or_insufficient_{timeframe.lower()}_bars"}
        cache = mt5_market_cache_m1 if timeframe == "M1" else mt5_market_cache
        cache["df"] = df
        cache["updated_at"] = datetime.now(timezone.utc)
        cache["source"] = "MT5_EA"
        return {
            "ok": True,
            "source": "MT5_EA",
            "timeframe": timeframe,
            "bars": len(df),
            "last_bar": str(df["datetime"].iloc[-1]),
            "updated_at": cache["updated_at"].isoformat(),
        }
    except Exception as e:
        logging.error(f"[MT5 DATA INGEST ERROR] {e}")
        return {"ok": False, "error": str(e)}

@app.get("/")
def home():
    return {"status": "ok", "message": "A/B/C scanner active: A=Mother Bar V2 (M1) LIVE + B=EMA 5/15 PAPER + C=Mother Bar Micro-Breakout PAPER.", "comparison": "/ab-comparison"}


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
# MULTI-SIGNAL BRIDGE (2026-09 fix): /get-latest-signal above only ever
# returns the single newest row ("ORDER BY id DESC LIMIT 1"), plus a
# 12-second response cache on top of that. Strategy C can fire more than
# one signal inside a single ~10s MT5 poll cycle (that's the whole point
# of "extreme frequency"), and the old endpoint has no memory of what the
# EA has already seen -- so any signal that isn't still "the latest" by
# the time the EA polls again is silently skipped and NEVER executed.
# This endpoint instead returns every unhandled LIVE signal since the
# id the EA tells us it already processed, oldest first, uncached, so a
# single poll can catch up on a whole backlog in one shot instead of
# losing everything but the last one.
# =====================================================================
@app.get("/get-pending-signals")
async def get_pending_signals(since_id: int = 0, limit: int = 20):
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME

    LAST_MT5_PING_TIME = datetime.now(timezone.utc) + timedelta(hours=7)

    if not SYSTEM_TRADING_ENABLED:
        return {"signals": [], "trading_enabled": False, "status": "PAUSED"}
    if not DATABASE_URL:
        return {"signals": [], "error": "DATABASE_URL not set", "trading_enabled": SYSTEM_TRADING_ENABLED}

    limit = max(1, min(limit, 50))

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                   COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, COALESCE(timestamp, created_at::text, '') AS log_time
            FROM signals
            WHERE status = 'EXECUTED' AND execution_mode = 'LIVE' AND id > %s
            ORDER BY id ASC LIMIT %s;
        """, (since_id, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        signals = [
            {"id": int(r["id"]), "action": str(r["action"]).upper(),
             "entry": float(r["entry_p"]), "sl": float(r["sl_p"]),
             "tp1": float(r["tp1_p"]), "tp2": float(r["tp2_p"]),
             "timestamp": str(r["log_time"])}
            for r in rows
        ]
        return {"signals": signals, "trading_enabled": True}
    except Exception as e:
        logging.error(f"[MT5 BRIDGE ERROR - pending signals] {e}")
        return {"signals": [], "error": str(e), "trading_enabled": SYSTEM_TRADING_ENABLED}


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
    global SYSTEM_TRADING_ENABLED, LAST_MT5_PING_TIME, CONTROL_DIRECTION_MODE, EXPERIMENTAL_DIRECTION_MODE, EXTREME_DIRECTION_MODE
    global CONTROL_EXECUTION_MODE, EXPERIMENTAL_EXECUTION_MODE, BREAKOUT_EXECUTION_MODE
    try:
        data = await request.json()
        message = data.get("message", {})
        raw_text = message.get("text", "").strip().lower()
        sender_chat_id = str(message.get("chat", {}).get("id", ""))

        if not sender_chat_id or not raw_text: return {"status": "ignored"}

        # Each Telegram bot has its own command surface. Bot B is read-only/paper-only.
        CONTROL_COMMANDS = {"/start", "/help", "/status", "/stats", "/pips", "/logs", "/analyze", "/pause", "/resume", "/oneway_on", "/oneway_off", "/both", "/macro", "/live_a", "/live_b", "/live_c"}
        EXPERIMENTAL_COMMANDS = {"/start", "/help", "/status", "/stats", "/compare", "/last", "/oneway_on", "/oneway_off", "/both", "/macro", "/live_a", "/live_b", "/live_c"}
        BREAKOUT_COMMANDS = {"/start", "/help", "/status", "/stats", "/last", "/pips", "/analyze", "/c_both", "/c_buyonly", "/c_sellonly", "/macro", "/live_a", "/live_b", "/live_c", "/c_fib25_on", "/c_fib25_off", "/c_fib50_on", "/c_fib50_off", "/c_fib618_on", "/c_fib618_off", "/c_fib75_on", "/c_fib75_off", "/c_closebreak_on", "/c_closebreak_off", "/c_momentum_on", "/c_momentum_off"}
        allowed = BREAKOUT_COMMANDS if bot_role == "breakout" else (EXPERIMENTAL_COMMANDS if bot_role == "experimental" else CONTROL_COMMANDS)
        if raw_text not in allowed:
            return {"status": "ignored", "reason": "command_not_available_for_this_bot"}

        active_token = BREAKOUT_TELEGRAM_BOT_TOKEN if bot_role == "breakout" else (EXPERIMENTAL_TELEGRAM_BOT_TOKEN if bot_role == "experimental" else TELEGRAM_BOT_TOKEN)
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async def send_reply(text: str):
                await send_telegram_alert(client, text, target_chat_id=sender_chat_id, target_bot_token=active_token)

            if raw_text in ["/help", "/start"]:
                live_bot_label = (
                    "A (Mother Bar V2, M1)" if CONTROL_EXECUTION_MODE == "LIVE"
                    else "B (EMA 5/15)" if EXPERIMENTAL_EXECUTION_MODE == "LIVE"
                    else "C (Mother Bar Micro)" if BREAKOUT_EXECUTION_MODE == "LIVE"
                    else "none (all PAPER)"
                )
                if bot_role == "breakout":
                    trigger_status = ", ".join(
                        f"{name}={'ON' if _c_trigger_state[key] else 'OFF'}"
                        for key, name in (
                            ("fib25", "Fib25"), ("fib50", "Fib50"), ("fib618", "Fib62"), ("fib75", "Fib75"),
                            ("close_break", "CloseBreak"), ("momentum", "Momentum"),
                        )
                    )
                    reply = (
                        "📦 *MOTHER BAR MICRO-BREAKOUT (C) BOT COMMANDS:*\n\n"
                        "• `/status` - Mother Bar status\n"
                        "• `/stats` - Mother Bar performance\n"
                        "• `/pips` - Detailed pips/earnings report (TP1/TP2/SL breakdown)\n"
                        "• `/analyze` - Entry condition category breakdown (pips/$/R per trigger type)\n"
                        "• `/last` - Last Mother Bar signals\n"
                        "• `/c_both` - Allow both BUY and SELL (default)\n"
                        "• `/c_buyonly` - Restrict to BUY only\n"
                        "• `/c_sellonly` - Restrict to SELL only\n"
                        "• `/c_fib25_on` / `/c_fib25_off` - Toggle the Fib Retrace 25% entry\n"
                        "• `/c_fib50_on` / `/c_fib50_off` - Toggle the Fib Retrace 50% entry\n"
                        "• `/c_fib618_on` / `/c_fib618_off` - Toggle the Fib Retrace 62% (golden ratio) entry\n"
                        "• `/c_fib75_on` / `/c_fib75_off` - Toggle the Fib Retrace 75% entry\n"
                        "• `/c_closebreak_on` / `/c_closebreak_off` - Toggle the Close Break entry\n"
                        "• `/c_momentum_on` / `/c_momentum_off` - Toggle the Momentum Break entry\n"
                        "• `/live_a` / `/live_b` / `/live_c` - Switch which bot's signals reach live MT5\n"
                        "• `/macro` - Macro context for gold (real yields, USD, COT positioning)\n"
                        "• `/help` - Display this menu\n\n"
                        "🟣 Strategy: *Mother Bar + Inside Bars (Momentum Break / Close Break / Fib Retrace)*\n"
                        f"🎚️ Triggers: *{trigger_status}*\n"
                        f"{'🔴 LIVE — this bot is connected to live MT5 execution.' if BREAKOUT_EXECUTION_MODE == 'LIVE' else f'⚠️ PAPER ONLY — live MT5 execution is currently on *{live_bot_label}*.'}"
                    )
                elif bot_role == "experimental":
                    reply = (
                        "🔬 *EXPERIMENTAL A/B BOT COMMANDS:*\n\n"
                        "• `/stats` - Detailed B performance dashboard (TP1/TP2/SL breakdown, $, efficiency)\n"
                        "• `/compare` - A vs B vs C summary comparison\n"
                        "• `/status` - Read-only system status\n"
                        "• `/last` - Last 10 experimental trades\n"
                         "• `/oneway_on` - Dynamic 1H direction ON\n"
                         "• `/oneway_off` - Dynamic direction OFF → BUY ONLY\n"
                         "• `/both` - Allow BUY + SELL\n"
                        "• `/live_a` / `/live_b` / `/live_c` - Switch which bot's signals reach live MT5\n"
                        "• `/macro` - Macro context for gold (real yields, USD, COT positioning)\n"
                        "• `/help` - Display this command menu\n\n"
                        f"{'🔴 Strategy: *EMA 5/15 — LIVE (real MT5 trades)*' if EXPERIMENTAL_EXECUTION_MODE == 'LIVE' else '⚪ Strategy: *EMA 5/15 — PAPER only*'}\n"
                        f"{'🚨 This bot IS connected to live MT5 execution. Use /pause on the Control bot for the emergency kill switch.' if EXPERIMENTAL_EXECUTION_MODE == 'LIVE' else f'ℹ️ Live MT5 execution is currently on *{live_bot_label}*, not this one.'}\n"
                    )
                else:
                    reply = (
                        f"🤖 *CONTROL MOTHER BAR A (M1) BOT COMMANDS:*\n\n"
                        "• `/status` - Real-time MT5, server & API status\n"
                        "• `/stats` - Control performance dashboard (PAPER)\n"
                        "• `/pips` - Gross/net pips & USD breakdown\n"
                        "• `/logs` - Last 10 executed trades\n"
                        "• `/analyze` - Forward-test strategy analysis\n"
                        "• `/pause` - 🚨 Emergency kill switch (stops ALL strategies, including live execution)\n"
                        "• `/resume` - 🟢 Re-enable auto-trading\n"
                         "• `/oneway_on` - Dynamic 1H direction ON\n"
                         "• `/oneway_off` - Dynamic direction OFF → BUY ONLY\n"
                         "• `/both` - Allow BUY + SELL\n"
                        "• `/live_a` / `/live_b` / `/live_c` - Switch which bot's signals reach live MT5\n"
                        "• `/macro` - Macro context for gold (real yields, USD, COT positioning)\n"
                        "• `/help` - Display this command menu\n\n"
                        f"{'🔴 Strategy: *Mother Bar V2 (M1, all 6 triggers) — LIVE (MT5)*' if CONTROL_EXECUTION_MODE == 'LIVE' else '⚪ Strategy: *Mother Bar V2 (M1, all 6 triggers) — PAPER only*'}\n"
                        f"{'' if CONTROL_EXECUTION_MODE == 'LIVE' else f'ℹ️ Live MT5 execution is currently on *{live_bot_label}*, not this one.'}\n"
                    )
                await send_reply(reply)

            elif raw_text in ("/oneway_on", "/oneway_off", "/both") and bot_role in ("control", "experimental"):
                if raw_text == "/oneway_on":
                    new_mode = "DYNAMIC"
                    mode_text = "DYNAMIC — 1H EMA200 decides allowed direction"
                elif raw_text == "/oneway_off":
                    new_mode = "BUY_ONLY"
                    mode_text = "BUY_ONLY — dynamic 1H one-direction system OFF"
                else:
                    new_mode = "BOTH"
                    mode_text = "BOTH — BUY and SELL allowed; one-direction restriction OFF"

                if bot_role == "control":
                    CONTROL_DIRECTION_MODE = new_mode
                    set_setting("control_direction_mode", new_mode)
                    label = "Strategy A (Control) only"
                else:
                    EXPERIMENTAL_DIRECTION_MODE = new_mode
                    set_setting("experimental_direction_mode", new_mode)
                    label = "Strategy B (Experimental) only"

                await send_reply(
                    f"🎛️ *DIRECTION MODE UPDATED*\n\n"
                    f"Mode: *{new_mode}*\n"
                    f"{mode_text}\n\n"
                    f"Applies to *{label}* — A and B now have independent direction switches."
                )

            elif raw_text in ("/live_a", "/live_b", "/live_c"):
                # Exactly one of A/B/C is ever LIVE at a time -- the MT5 EA's
                # /get-latest-signal + /get-pending-signals bridge only ever
                # serves rows with execution_mode='LIVE', so making one bot
                # live means making the other two PAPER in the same call.
                CONTROL_EXECUTION_MODE = "LIVE" if raw_text == "/live_a" else "PAPER"
                EXPERIMENTAL_EXECUTION_MODE = "LIVE" if raw_text == "/live_b" else "PAPER"
                BREAKOUT_EXECUTION_MODE = "LIVE" if raw_text == "/live_c" else "PAPER"
                set_setting("control_execution_mode", CONTROL_EXECUTION_MODE)
                set_setting("experimental_execution_mode", EXPERIMENTAL_EXECUTION_MODE)
                set_setting("breakout_execution_mode", BREAKOUT_EXECUTION_MODE)
                live_label = {"/live_a": "A (Mother Bar V2, M1)", "/live_b": "B (EMA 5/15)", "/live_c": "C (Mother Bar Micro)"}[raw_text]
                logging.info(
                    f"[LIVE TOGGLE] Now live: {live_label} -- "
                    f"A={CONTROL_EXECUTION_MODE}, B={EXPERIMENTAL_EXECUTION_MODE}, C={BREAKOUT_EXECUTION_MODE}"
                )
                await send_reply(
                    f"🔴 *LIVE MT5 EXECUTION SWITCHED*\n\n"
                    f"Now live: *Strategy {live_label}*\n\n"
                    f"• A: *{CONTROL_EXECUTION_MODE}*\n"
                    f"• B: *{EXPERIMENTAL_EXECUTION_MODE}*\n"
                    f"• C: *{BREAKOUT_EXECUTION_MODE}*\n\n"
                    f"Only signals from the LIVE strategy reach the MT5 EA (/get-latest-signal, "
                    f"/get-pending-signals). The other two keep running and logging as PAPER."
                )

            elif raw_text in (
                "/c_fib25_on", "/c_fib25_off", "/c_fib50_on", "/c_fib50_off",
                "/c_fib618_on", "/c_fib618_off", "/c_fib75_on", "/c_fib75_off",
                "/c_closebreak_on", "/c_closebreak_off",
                "/c_momentum_on", "/c_momentum_off",
            ) and bot_role == "breakout":
                key, turn_on = {
                    "/c_fib25_on": ("fib25", True), "/c_fib25_off": ("fib25", False),
                    "/c_fib50_on": ("fib50", True), "/c_fib50_off": ("fib50", False),
                    "/c_fib618_on": ("fib618", True), "/c_fib618_off": ("fib618", False),
                    "/c_fib75_on": ("fib75", True), "/c_fib75_off": ("fib75", False),
                    "/c_closebreak_on": ("close_break", True), "/c_closebreak_off": ("close_break", False),
                    "/c_momentum_on": ("momentum", True), "/c_momentum_off": ("momentum", False),
                }[raw_text]
                _c_trigger_state[key] = turn_on
                set_setting(f"c_trigger_{key}", "on" if turn_on else "off")
                _apply_c_trigger_state()
                trigger_names = {
                    "fib25": "MB Fib Retrace 25%", "fib50": "MB Fib Retrace 50%",
                    "fib618": "MB Fib Retrace 62%", "fib75": "MB Fib Retrace 75%",
                    "close_break": "MB Close Break", "momentum": "MB Momentum Break",
                }
                active = [trigger_names[k] for k, v in _c_trigger_state.items() if v]
                await send_reply(
                    f"⚡ *{trigger_names[key]}* is now *{'ON' if turn_on else 'OFF'}* for Strategy C.\n\n"
                    f"Active triggers: {', '.join(active) if active else '_none -- C will never fire until at least one is on_'}"
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
                    active_direction_mode = CONTROL_DIRECTION_MODE if bot_role == "control" else EXPERIMENTAL_DIRECTION_MODE
                    if active_direction_mode == "BUY_ONLY":
                        bias_note = "Dynamic 1H system OFF — SELL blocked; BUY only"
                    elif active_direction_mode == "DYNAMIC":
                        bias_note = {
                            "BULLISH": "SELL blocked this cycle",
                            "BEARISH": "BUY blocked this cycle",
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
                        f"  \u2514\u2500 5M: {_twelve_data_calls_by_tf['5min']} | 15M: {_twelve_data_calls_by_tf['15min']} | 1H: {_twelve_data_calls_by_tf['1h']} | 1M(fallback): {_twelve_data_calls_by_tf['1min']}\n\n"
                         f"⚡ *C DATA SOURCE:* {'MT5 EA (fresh)' if _mt5_cache_fresh() else 'Twelve Data M5 fallback'}\n\n"
                        f"\U0001f4c8 *STRATEGY:*\n"
                        f"\u2022 Direction Mode ({'A' if bot_role == 'control' else 'B'}): *{active_direction_mode}*\n"
                        f"\u2022 A Execution (M1): *Mother Bar V2 -- Close/Fib TP1 1.25x/TP2 8x, Momentum TP1 2x/TP2 15x, reject risk > max($15, 2.25x ATR)*\n"
                        f"\u2022 B Execution (5M): *EMA {EXPERIMENTAL_EMA_FAST}/{EXPERIMENTAL_EMA_SLOW}*\n"
                        f"\u2022 Confluence (15M): *EMA {TREND_15M_EMA_FAST}/{TREND_15M_EMA_SLOW}* (derived locally from M5)\n"
                        f"\u2022 Strategy C: *Mother Bar Micro-Breakout (MBMB)* | Max {EXTREME_MAX_TRADES_PER_DAY_LABEL}/day\n"
                        f"\u2022 Live MT5 Execution: *{'A' if CONTROL_EXECUTION_MODE == 'LIVE' else ('B' if EXPERIMENTAL_EXECUTION_MODE == 'LIVE' else ('C' if BREAKOUT_EXECUTION_MODE == 'LIVE' else 'none'))}* (toggle with /live_a, /live_b, /live_c)\n\n"
                        f"{get_macro_status_line()}"
                    )
                else:
                    reply = "\U0001f534 *MT5 DISCONNECTED*\n\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\nThe server is running, but MT5 has not sent any pings since the last reboot."
                await send_reply(reply)

            elif raw_text == "/macro":
                now_wib_macro = datetime.now(timezone.utc) + timedelta(hours=7)
                async with httpx.AsyncClient() as macro_client:
                    await refresh_macro_context(macro_client, now_wib_macro)
                await send_reply(_macro_cache["text"])

            elif bot_role == "control" and raw_text == "/pause":
                SYSTEM_TRADING_ENABLED = False
                await send_reply("\U0001f6d1 *EMERGENCY KILL SWITCH ACTIVATED*\nMarket scanner paused. Send `/resume` to reactivate.")

            elif bot_role == "control" and raw_text == "/resume":
                SYSTEM_TRADING_ENABLED = True
                await send_reply("\U0001f7e2 *AUTO-TRADING SYSTEM RESUMED*\nScanner loop is now active.")

            elif bot_role == "breakout" and raw_text == "/status":
                now_wib_c = datetime.now(timezone.utc) + timedelta(hours=7)
                _extreme_roll_day(now_wib_c)
                mt5_src = "MT5 EA" if _mt5_cache_fresh() else "Twelve Data fallback"
                daily_r_today = 0.0
                if DATABASE_URL:
                    try:
                        conn = get_db_connection(); cur = conn.cursor()
                        cur.execute(
                            "SELECT COALESCE(SUM(result_r), 0) AS daily_r FROM signals WHERE strategy = %s AND outcome_timestamp LIKE %s",
                            (BREAKOUT_STRATEGY, now_wib_c.strftime("%Y-%m-%d") + "%")
                        )
                        daily_r_today = float(cur.fetchone()["daily_r"] or 0.0)
                        cur.close(); conn.close()
                    except Exception:
                        pass
                trigger_status = ", ".join(
                    f"{name}={'ON' if _c_trigger_state[key] else 'OFF'}"
                    for key, name in (
                        ("fib25", "Fib25"), ("fib50", "Fib50"), ("fib618", "Fib62"), ("fib75", "Fib75"),
                        ("close_break", "CloseBreak"), ("momentum", "Momentum"),
                    )
                )
                reply = (
                    "⚡ *MOTHER BAR MICRO-BREAKOUT (STRATEGY C) STATUS*\n\n"
                    f"Engine: *MB structure + inside-bar breakout, target sized off MB range*\n"
                    f"Data source: *{mt5_src}*\n"
                    f"Direction mode: *{EXTREME_DIRECTION_MODE}*\n"
                    f"Triggers: *{trigger_status}*\n"
                    f"Signals today: *{_extreme_state['trades_today']}/{EXTREME_MAX_TRADES_PER_DAY_LABEL}*\n"
                    f"Today's R: *{daily_r_today:+.2f}R*\n"
                    f"Daily loss breaker: *\U0001f7e2 OFF (removed)*\n"
                    f"Execution: *{BREAKOUT_EXECUTION_MODE}*\n"
                    f"MT5 feed cache: *{'FRESH' if _mt5_cache_fresh() else 'NOT FRESH'}*\n\n"
                    f"{get_macro_status_line()}"
                )
                await send_reply(reply)

            elif bot_role == "breakout" and raw_text == "/stats":
                try:
                    conn=get_db_connection(); cur=conn.cursor()
                    cur.execute("""SELECT COUNT(*) FILTER (WHERE status='EXECUTED') AS executed, COUNT(*) FILTER (WHERE status='CANCELLED') AS cancelled, COUNT(*) FILTER (WHERE status='EXECUTED' AND (outcome LIKE 'WIN%%' OR outcome LIKE 'CLOSED%%')) AS wins, COUNT(*) FILTER (WHERE status='EXECUTED' AND outcome LIKE 'LOSS%%') AS losses, COALESCE(SUM(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS total_r, COALESCE(AVG(result_r) FILTER (WHERE status='EXECUTED' AND result_r IS NOT NULL),0) AS avg_r FROM signals WHERE strategy=%s""",(BREAKOUT_STRATEGY,))
                    s=cur.fetchone(); cur.close(); conn.close(); ex=int(s['executed'] or 0); wins=int(s['wins'] or 0); losses=int(s['losses'] or 0)
                    await send_reply(f"⚡ *MOTHER BAR MICRO-BREAKOUT PERFORMANCE*\n━━━━━━━━━━━━━━━━━━━━\nExecuted: *{ex}*\nWins/Losses: *{wins}/{losses}*\nWin Rate: *{(wins/ex*100 if ex else 0):.1f}%*\nTotal R: *{float(s['total_r'] or 0):+.2f}R* | Avg R: *{float(s['avg_r'] or 0):+.3f}R*\nMode: *{BREAKOUT_EXECUTION_MODE}*\nEngine: *Mother Bar + inside bars, target = MB range*")
                except Exception as e: await send_reply(f"⚠️ Error querying breakout stats: {e}")

            elif bot_role == "breakout" and raw_text == "/last":
                try:
                    conn=get_db_connection(); cur=conn.cursor(); cur.execute("SELECT id,action,trigger_type,entry_price,outcome,created_at FROM signals WHERE strategy=%s ORDER BY id DESC LIMIT 10",(BREAKOUT_STRATEGY,)); rows=cur.fetchall(); cur.close(); conn.close()
                    if not rows: reply="📋 *LAST MOTHER BAR SIGNALS*\n\n_No Mother Bar signals yet._"
                    else: reply="📋 *LAST MOTHER BAR SIGNALS*\n\n"+"\n".join(f"#{r['id']} | {r['action']} | {r['trigger_type']} | ${float(r['entry_price'] or 0):.2f} | {r['outcome'] or 'N/A'}" for r in rows)
                    await send_reply(reply)
                except Exception as e: await send_reply(f"⚠️ Error querying breakout logs: {e}")

            elif bot_role == "breakout" and raw_text == "/pips":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p,
                               COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p,
                               exit_price, COALESCE(outcome, 'PENDING') AS outcome_val
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s
                    """, (BREAKOUT_STRATEGY,))
                    trades = cur.fetchall()
                    cur.close(); conn.close()

                    total_pips = gross_win_pips = gross_loss_pips = 0.0
                    winning_trades_count = losing_trades_count = 0
                    tp1_be_count = tp2_count = sl_count = 0

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

                        ov = t["outcome_val"]
                        if ov == "CLOSED (TP1 HIT / SL BE)":
                            tp1_be_count += 1
                        elif ov in ("WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"):
                            tp2_count += 1
                        elif "LOSS" in ov:
                            sl_count += 1

                    avg_win_pips = (gross_win_pips / winning_trades_count) if winning_trades_count > 0 else 0.0
                    avg_loss_pips = (gross_loss_pips / losing_trades_count) if losing_trades_count > 0 else 0.0
                    est_profit_usd = total_pips * 0.10
                    pip_efficiency = gross_win_pips / (gross_loss_pips + 1e-5)

                    reply = (
                        f"⚡ *MOTHER BAR C DETAILED PIPS & EARNINGS REPORT*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\U0001f4ca *SUMMARY:*\n"
                        f"\u2022 Total Net Pips: *{total_pips:+.1f} pips*\n"
                        f"\u2022 Net Profit (0.01 Lot): *${est_profit_usd:+.2f}*\n\n"
                        f"\U0001f3af *TP/SL BREAKDOWN:*\n"
                        f"\u2022 TP1 Hit + Runner @ BE: *{tp1_be_count}*\n"
                        f"\u2022 TP2 Hit (full runner): *{tp2_count}*\n"
                        f"\u2022 SL Hit (full loss): *{sl_count}*\n\n"
                        f"\U0001f4c8 *PIPS BREAKDOWN:*\n"
                        f"\u2022 Gross Gain: *+{gross_win_pips:.1f} pips*\n"
                        f"\u2022 Gross Loss: *-{gross_loss_pips:.1f} pips*\n\n"
                        f"\U0001f3af *AVERAGE METRICS:*\n"
                        f"\u2022 Avg Win Trade: *+{avg_win_pips:.1f} pips*\n"
                        f"\u2022 Avg Loss Trade: *-{avg_loss_pips:.1f} pips*\n"
                        f"\u2022 Pip Efficiency Ratio: *{pip_efficiency:.2f}*\n"
                        f"\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
                        f"\U0001f4a1 *Note:* PAPER simulation, same 2x0.01 lot model as A/B -- "
                        f"SL (before TP1) = both lots @ SL, TP1/BE = lot1 @ TP1 + lot2 @ BE, "
                        f"TP2 = lot1 @ TP1 + lot2 @ TP2."
                    )
                    await send_reply(reply)
                except Exception as e: await send_reply(f"⚠️ Error querying breakout pips: {e}")

            elif bot_role == "breakout" and raw_text == "/analyze":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT action, trigger_type, COALESCE(entry_price, price, 0) AS entry_p,
                               COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p,
                               COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price,
                               COALESCE(outcome, 'PENDING') AS outcome_val, regime, reasoning
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s
                    """, (BREAKOUT_STRATEGY,))
                    rows = cur.fetchall()
                    cur.close(); conn.close()

                    if not rows:
                        reply = (
                            "\U0001f4d0 *MOTHER BAR C -- ENTRY CONDITION ANALYSIS*\n\n"
                            "_Not enough closed trades yet to analyze. Check back after more signals complete._"
                        )
                    else:
                        # "Entry condition category" = trigger_type as returned by
                        # find_mother_bar_signal: "MB Momentum Break", "MB Close
                        # Break", or "MB Fib Retrace NN%" (one bucket per level,
                        # since a shallow vs deep retracement is a different bet).
                        categories = {}
                        data_sources = {}
                        overall_r, overall_pips, overall_usd = [], [], []
                        tp1_be_count = tp2_count = sl_count = 0

                        for r in rows:
                            r_mult = compute_r_multiple(
                                r["action"], float(r["entry_p"]), float(r["exit_price"]), float(r["sl_p"]),
                                float(r["tp1_p"]), float(r["tp2_p"]), r["outcome_val"]
                            )
                            pips, usd = compute_trade_pips({
                                "action": r["action"], "entry_price": r["entry_p"], "sl_price": r["sl_p"],
                                "tp1_price": r["tp1_p"], "tp2_price": r["tp2_p"], "exit_price": r["exit_price"],
                                "outcome": r["outcome_val"]
                            })
                            overall_r.append(r_mult); overall_pips.append(pips); overall_usd.append(usd)
                            cat = r["trigger_type"] or "Unknown"
                            categories.setdefault(cat, []).append((r_mult, pips, usd))
                            src = extract_mb_data_source(r["regime"], r["reasoning"])
                            data_sources.setdefault(src, []).append((r_mult, pips, usd))

                            ov = r["outcome_val"]
                            if ov == "CLOSED (TP1 HIT / SL BE)": tp1_be_count += 1
                            elif ov in ("WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"): tp2_count += 1
                            elif "LOSS" in ov: sl_count += 1

                        n_total = len(overall_r)
                        overall_wr = (sum(1 for x in overall_r if x > 0) / n_total * 100) if n_total else 0.0
                        overall_avg_r = (sum(overall_r) / n_total) if n_total else 0.0
                        total_pips = sum(overall_pips)
                        total_usd = sum(overall_usd)

                        reply_parts = [
                            "\U0001f4d0 *MOTHER BAR C -- ENTRY CONDITION ANALYSIS*",
                            "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | Avg R: *{overall_avg_r:+.2f}*",
                            f"Total Pips: *{total_pips:+.1f}* | Total $ (2x0.01 lot): *${total_usd:+.2f}*",
                            f"TP1\u2192BE: *{tp1_be_count}* | TP2 (full runner): *{tp2_count}* | SL: *{sl_count}*",
                            "",
                            "\U0001f9ee *Entry Condition Category Counter:*",
                            format_entry_condition_segment(categories),
                            "",
                            "\U0001f4e1 *By Data Source (MT5 live feed vs Twelve Data fallback):*",
                            format_entry_condition_segment(data_sources),
                            "",
                            "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015",
                            "\U0001f4a1 Categories need n\u22655 to be flagged \u26a0\ufe0f/\u2705 (smaller samples shown but noisy). "
                            "\"MB Fib Retrace NN%\" buckets separately per level -- a shallow retrace and a deep one "
                            "are different entries even off the same Mother Bar. Trades logged before this "
                            "breakdown existed show as \"Unknown (pre-tracking)\" under Data Source.\n"
                            "Pips/$ use the same 2x0.01-lot PAPER model as /pips."
                        ]
                        reply = "\n".join(reply_parts)
                    await send_reply(reply)
                except Exception as e:
                    await send_reply(f"⚠️ Error: {e}")

            elif bot_role == "breakout" and raw_text == "/c_both":
                EXTREME_DIRECTION_MODE = "BOTH"
                set_setting("extreme_direction_mode", "BOTH")
                await send_reply("⚡ Strategy C direction mode: *BOTH* -- BUY and SELL signals both active.")

            elif bot_role == "breakout" and raw_text == "/c_buyonly":
                EXTREME_DIRECTION_MODE = "BUY_ONLY"
                set_setting("extreme_direction_mode", "BUY_ONLY")
                await send_reply("⚡ Strategy C direction mode: *BUY_ONLY* -- SELL signals are now blocked.")

            elif bot_role == "breakout" and raw_text == "/c_sellonly":
                EXTREME_DIRECTION_MODE = "SELL_ONLY"
                set_setting("extreme_direction_mode", "SELL_ONLY")
                await send_reply("⚡ Strategy C direction mode: *SELL_ONLY* -- BUY signals are now blocked.")

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

            elif bot_role == "experimental" and raw_text == "/stats":
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) AS total FROM signals WHERE status = 'EXECUTED' AND strategy = %s", (EXPERIMENTAL_STRATEGY,))
                    total_executed = cur.fetchone()["total"] or 0
                    cur.execute("SELECT COUNT(*) AS vetoes FROM signals WHERE status = 'VETOED' AND strategy = %s", (EXPERIMENTAL_STRATEGY,))
                    total_vetoes = cur.fetchone()["vetoes"] or 0
                    cur.execute("SELECT COUNT(*) AS pending FROM signals WHERE status = 'EXECUTED' AND outcome = 'PENDING' AND strategy = %s", (EXPERIMENTAL_STRATEGY,))
                    total_pending = cur.fetchone()["pending"] or 0
                    cur.execute("SELECT COUNT(*) AS tp1_wins FROM signals WHERE strategy = %s AND (outcome LIKE 'WIN (TP1%%' OR outcome LIKE 'CLOSED%%')", (EXPERIMENTAL_STRATEGY,))
                    tp1_wins = cur.fetchone()["tp1_wins"] or 0
                    cur.execute("SELECT COUNT(*) AS tp2_wins FROM signals WHERE strategy = %s AND outcome LIKE 'WIN (TP2%%'", (EXPERIMENTAL_STRATEGY,))
                    tp2_wins = cur.fetchone()["tp2_wins"] or 0
                    cur.execute("SELECT COUNT(*) AS losses FROM signals WHERE strategy = %s AND outcome LIKE 'LOSS%%'", (EXPERIMENTAL_STRATEGY,))
                    losses = cur.fetchone()["losses"] or 0

                    cur.execute("SELECT action, COALESCE(entry_price, price, 0) AS entry_p, COALESCE(sl_price, sl, 0) AS sl_p, COALESCE(tp1_price, tp1, 0) AS tp1_p, COALESCE(tp2_price, tp2, 0) AS tp2_p, exit_price, COALESCE(outcome, 'PENDING') AS outcome_val FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s", (EXPERIMENTAL_STRATEGY,))
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
                        f"\U0001f4ca *PERFORMANCE ANALYTICS DASHBOARD (LIVE)*\n"
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

            elif bot_role == "experimental" and raw_text == "/compare":
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
                            ("🔴 Mother Bar A (M1, LIVE)", a.get("total_r", 0)),
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
                        f"🔴 *A — CONTROL (Mother Bar V2, M1, LIVE)*\n{block('', a)}\n\n"
                        f"🔵 *B — EXPERIMENT (EMA 5/15)*\n{block('', b)}\n\n"
                        f"⚡ *C — MOTHER BAR MICRO-BREAKOUT*\n{block('', c)}\n\n"
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
                               COALESCE(outcome, 'PENDING') AS outcome_val,
                               COALESCE(timestamp, created_at::text, '') AS log_time,
                               regime, reasoning, divergence_type
                        FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL AND strategy = %s
                    """, (CONTROL_STRATEGY,))
                    rows = cur.fetchall()
                    cur.close(); conn.close()

                    if not rows:
                        reply = (
                            "\U0001f4d0 *MOTHER BAR A-V2 -- ENTRY CONDITION ANALYSIS*\n\n"
                            "_Not enough closed trades yet to analyze. Check back after more signals complete._"
                        )
                    else:
                        # Entry condition category = trigger_type as returned by
                        # find_mother_bar_signal: "MB Momentum Break", "MB Close
                        # Break", or "MB Fib Retrace NN%" (one bucket per level).
                        # Same shape as bot C's /analyze -- A runs the same MB
                        # engine on M1 with a V2 tune (wider TPs, hard risk-reject
                        # instead of cap-and-shrink, no daily loss breaker).
                        categories, data_sources, gate_shadow, sessions = {}, {}, {}, {}
                        overall_r, overall_pips, overall_usd = [], [], []
                        tp1_be_count = tp2_count = sl_count = 0

                        for r in rows:
                            r_mult = compute_r_multiple(
                                r["action"], float(r["entry_p"]), float(r["exit_price"]), float(r["sl_p"]),
                                float(r["tp1_p"]), float(r["tp2_p"]), r["outcome_val"]
                            )
                            pips, usd = compute_trade_pips({
                                "action": r["action"], "entry_price": r["entry_p"], "sl_price": r["sl_p"],
                                "tp1_price": r["tp1_p"], "tp2_price": r["tp2_p"], "exit_price": r["exit_price"],
                                "outcome": r["outcome_val"]
                            })
                            overall_r.append(r_mult); overall_pips.append(pips); overall_usd.append(usd)
                            cat = r["trigger_type"] or "Unknown"
                            categories.setdefault(cat, []).append((r_mult, pips, usd))
                            src = extract_mb_data_source(r["regime"], r["reasoning"])
                            data_sources.setdefault(src, []).append((r_mult, pips, usd))
                            gate_shadow.setdefault(bucket_adx_gate_shadow(r["divergence_type"]), []).append((r_mult, pips, usd))
                            sessions.setdefault(bucket_session(r["log_time"]), []).append((r_mult, pips, usd))

                            ov = r["outcome_val"]
                            if ov == "CLOSED (TP1 HIT / SL BE)": tp1_be_count += 1
                            elif ov in ("WIN (TP2 HIT)", "WIN (TP2 HIT FULL)"): tp2_count += 1
                            elif "LOSS" in ov: sl_count += 1

                        n_total = len(overall_r)
                        overall_wr = (sum(1 for x in overall_r if x > 0) / n_total * 100) if n_total else 0.0
                        overall_avg_r = (sum(overall_r) / n_total) if n_total else 0.0
                        total_pips = sum(overall_pips)
                        total_usd = sum(overall_usd)

                        reply_parts = [
                            "\U0001f4d0 *MOTHER BAR A-V2 -- ENTRY CONDITION ANALYSIS*",
                            "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015",
                            f"Sample: *{n_total} closed trades*",
                            f"Overall Win Rate: *{overall_wr:.1f}%* | Avg R: *{overall_avg_r:+.2f}*",
                            f"Total Pips: *{total_pips:+.1f}* | Total $ (0.01 lot): *${total_usd:+.2f}*",
                            f"TP1\u2192BE: *{tp1_be_count}* | TP2 (full runner): *{tp2_count}* | SL/Reject-through: *{sl_count}*",
                            "",
                            "\U0001f9ee *Entry Condition Category Counter:*",
                            format_entry_condition_segment(categories),
                            "",
                            "\u2696\ufe0f *ADX Gate (shadow mode -- not yet blocking any trade):*",
                            format_entry_condition_segment(gate_shadow),
                            "",
                            "\U0001f4e1 *By Data Source (MT5 M1 push vs Twelve Data 1min fallback):*",
                            format_entry_condition_segment(data_sources),
                            "",
                            "\U0001f550 *By Session:*",
                            format_entry_condition_segment(sessions),
                            "",
                            "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015",
                            "\U0001f4a1 Categories need n\u22655 to be flagged \u26a0\ufe0f/\u2705 (smaller samples shown but noisy). "
                            "\"MB Fib Retrace NN%\" buckets separately per level. The ADX Gate breakdown is pure "
                            "instrumentation right now (MB_ADX_GATE_SHADOW_MODE=True) -- once a zone shows a clear "
                            "edge either way with enough trades, that's the signal to flip the gate live.\n"
                            "No daily loss breaker on A (removed 2026-09-24) and no daily trade cap -- every valid "
                            "M1 setup fires regardless of how the day's gone."
                        ]
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
