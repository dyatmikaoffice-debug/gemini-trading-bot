import os
import asyncio
import logging
import gc
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import httpx
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Request
import uvicorn

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()

# Sanitize Telegram Bot Token
BOT_TOKEN = "".join(TELEGRAM_BOT_TOKEN.split())

# Database Connection Helper
def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# ---------------------------------------------------------
# TECHNICAL INDICATORS & TREND LOGIC
# ---------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Exponential Moving Averages
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Average True Range (14)
    df['high_low'] = df['high'] - df['low']
    df['high_cp'] = (df['high'] - df['close'].shift(1)).abs()
    df['low_cp'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['high_low', 'high_cp', 'low_cp']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # Stochastic RSI (14, 3, 3)
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['rsi'].rolling(rsi_period).min()
    max_rsi = df['rsi'].rolling(rsi_period).max()
    stoch_rsi = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-10)
    df['stoch_k'] = stoch_rsi.rolling(3).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ADX (14)
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr_smooth = df['tr'].rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    return df

# ---------------------------------------------------------
# AI ANALYST RISK FILTER
# ---------------------------------------------------------
async def analyze_signal_with_ai(price: float, action: str, adx: float, stoch_k: float, ema50: float, ema200: float) -> str:
    if not GROQ_API_KEY:
        return "APPROVE"
        
    prompt = f"""
    You are an institutional Gold Risk Analyst. Evaluate this trade setup:
    - Pair: Spot Gold (XAU/USD)
    - Action Proposed: {action}
    - Current Price: ${price:.2f}
    - 5M EMA 50: ${ema50:.2f} \vert{} 5M EMA 200:${ema200:.2
