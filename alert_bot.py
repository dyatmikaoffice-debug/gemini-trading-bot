import os
import asyncio
import httpx
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load API credentials from environment or defaults
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
TICKER_SYMBOL = "GC=F"

app = FastAPI()

class SignalOutput(BaseModel):
    action: str = Field(description="BUY, SELL, or HOLD")
    confidence: float
    summary: str

def calculate_indicators(df: pd.DataFrame, stoch_k=14, stoch_d=3, smooth_k=3, ema_period=50):
    if len(df) < max(stoch_k + stoch_d + smooth_k, ema_period):
        return None

    df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    low_min = df['Low'].rolling(window=stoch_k).min()
    high_max = df['High'].rolling(window=stoch_k).max()
    fast_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%K'] = fast_k.rolling(window=smooth_k).mean()
    df['%D'] = df['%K'].rolling(window=stoch_d).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    stoch_cross_up = (prev['%K'] < prev['%D']) and (latest['%K'] > latest['%D'])
    stoch_cross_down = (prev['%K'] > prev['%D']) and (latest['%K'] < latest['%D'])
    is_above_ema = latest['Close'] > latest['EMA_50']

    return {
        "close": float(latest['Close']),
        "ema_50": float(latest['EMA_50']),
        "is_above_ema": bool(is_above_ema),
        "stoch_k": float(latest['%K']),
        "stoch_d": float(latest['%D']),
        "stoch_cross_up": bool(stoch_cross_up),
        "stoch_cross_down": bool(stoch_cross_down)
    }

def get_mtf_data():
    ticker = yf.Ticker(TICKER_SYMBOL)
    df_5m = ticker.history(period="5d", interval="5m")
    df_15m = ticker.history(period="5d", interval="15m")
    df_1h = ticker.history(period="1mo", interval="1h")
    
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    return {
        "5m": calculate_indicators(df_5m),
        "15m": calculate_indicators(df_15m),
        "1h": calculate_indicators(df_1h),
        "4h": calculate_indicators(df_4h)
    }

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    httpx.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def analyze_and_alert():
    print("Checking Multi-Timeframe Gold Data...")
    mtf = get_mtf_data()

    if not all([mtf["5m"], mtf["15m"], mtf["1h"], mtf["4h"]]):
        return

    price = mtf["5m"]["close"]

    prompt = f"""
    Analyze this Full Multi-Timeframe (MTF) Alignment setup for Gold ({TICKER_SYMBOL}):
    Current Price: ${price:.2f}

    - 4H Timeframe: Above 50 EMA = {mtf['4h']['is_above_ema']}
    - 1H Timeframe: Above 50 EMA = {mtf['1h']['is_above_ema']}
    - 15M Timeframe: Above 50 EMA = {mtf['15m']['is_above_ema']}
    - 5M Timeframe: Above 50 EMA = {mtf['5m']['is_above_ema']}, Stoch %K = {mtf['5m']['stoch_k']:.1f}, Stoch Cross Up = {mtf['5m']['stoch_cross_up']}, Stoch Cross Down = {mtf['5m']['stoch_cross_down']}

    Strategy Rules:
    - BUY: 4H & 1H strictly Bullish (Price > 50 EMA), 5M pulls back and crosses up on Stochastic (14,3,3) while reclaiming 5M 50 EMA.
    - SELL: 4H & 1H strictly Bearish (Price < 50 EMA), 5M pulls back and crosses down on Stochastic (14,3,3) while falling below 5M 50 EMA.
    - HOLD: If timeframes conflict.

    Evaluate if 100% MTF alignment is confirmed.
    """

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=SignalOutput,
        system_instruction="You are a professional Gold (XAUUSD) analyst enforcing strict MTF Confluence."
    )

    response = genai_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=config
    )

    signal = SignalOutput.model_validate_json(response.text)
    print(f"Decision: {signal.action} ({signal.confidence * 100:.0f}%)")

    if signal.action in ["BUY", "SELL"]:
        msg = (
            f"🎯 **FULL MTF ALIGNMENT SIGNAL ({signal.action})**\n\n"
            f"• **Asset:** Gold (`GC=F`)\n"
            f"• **Price:** `${price:.2f}`\n"
            f"• **Macro (4H/1H):** `{'BULLISH' if mtf['1h']['is_above_ema'] else 'BEARISH'}`\n"
            f"• **Trigger (5M Stoch 14,3,3):** `%K={mtf['5m']['stoch_k']:.1f}`\n"
            f"• **Confidence:** `{int(signal.confidence * 100)}%`\n\n"
            f"📝 **Reasoning:** _{signal.summary}_"
        )
        send_telegram_message(msg)

async def background_loop():
    while True:
        try:
            analyze_and_alert()
        except Exception as e:
            print(f"Error in bot loop: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_loop())

@app.get("/")
def health_check():
    return {"status": "running", "bot": "Gemini MTF Gold Alert Bot Active"}
