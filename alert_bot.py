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

# Logging Configuration
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
# TECHNICAL INDICATORS: EMA 9, 21, 100 + ATR + ADX
# ---------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 3 Exponential Moving Averages (9, 21, 100)
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Average True Range (14)
    df['high_low'] = df['high'] - df['low']
    df['high_cp'] = (df['high'] - df['close'].shift(1)).abs()
    df['low_cp'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['high_low', 'high_cp', 'low_cp']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # Average Directional Index - ADX (14)
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

async def analyze_signal_with_ai(price: float, action: str, adx: float, ema9: float, ema21: float, ema100: float, atr: float, trigger_reason: str) -> str:
    if not GROQ_API_KEY:
        return "APPROVE"
        
    prompt = f"""
    You are an institutional Gold Risk Analyst. Evaluate this trade setup:
    - Pair: Spot Gold (XAU/USD)
    - Action Proposed: {action}
    - Trigger Source: {trigger_reason}
    - Current Price: ${price:.2f}
    - 5M EMAs -> 9: ${ema9:.2f} \vert{} 21:${ema21:.2f} | 100: ${ema100:.2f}     - 15M ADX Trend Strength: {adx:.1f}     - Current ATR:${atr:.2f}

    STRICT RULES:
    1. VETO if proposed BUY occurs when 9 EMA is below 21 EMA or 100 EMA.
    2. VETO if proposed SELL occurs when 9 EMA is above 21 EMA or 100 EMA.
    3. VETO if ADX < 10 (extreme low-volatility chop zone).

    Respond strictly with APPROVE or VETO followed by a 1-sentence explanation.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }
            )
            data = res.json()
            content = data['choices'][0]['message']['content'].strip()
            return "VETO" if "VETO" in content.upper() else "APPROVE"
    except Exception as e:
        logging.error(f"AI Analyst Exception: {e}")
        return "APPROVE"

# ---------------------------------------------------------
# TELEGRAM MESSAGING & AUTOMATED WEBHOOK SETUP
# ---------------------------------------------------------

async def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            res = await client.post(url, json=payload)
            if res.status_code == 200:
                logging.info(f"[TELEGRAM SENT] Delivered to Chat ID {TELEGRAM_CHAT_ID}")
            else:
                logging.error
