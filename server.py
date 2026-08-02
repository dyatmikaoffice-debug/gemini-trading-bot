import os
import json
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

# Import official Google GenAI SDK
from google import genai
from google.genai import types

app = FastAPI(title="Realtime Gemini Trading Signal Feed")

# Configuration via Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize Gemini Client
genai_client = genai.Client(api_key= os.getenv("GEMINI_API_KEY")


# Define Strict JSON Output for Telegram
class SignalOutput(BaseModel):
    ticker: str = Field(description="Asset symbol")
    action: str = Field(description="BUY, SELL, or IGNORE")
    confidence: float = Field(description="Value between 0.0 and 1.0")
    entry_price: float = Field(description="Recommended entry")
    stop_loss: float = Field(description="Stop loss target")
    take_profit: float = Field(description="Take profit target")
    summary: str = Field(description="1-2 sentence quick rationale")


async def push_to_telegram(signal: SignalOutput):
    """Sends the formatted signal directly to your Telegram chat or channel."""
    emoji = "🟢" if signal.action == "BUY" else ("🔴" if signal.action == "SELL" else "🟡")
    
    text_message = (
        f"{emoji} **REAL-TIME SIGNAL: {signal.ticker}** {emoji}\n\n"
        f"• **Decision:** `{signal.action}`\n"
        f"• **Confidence:** `{int(signal.confidence * 100)}%`\n"
        f"• **Entry:** `${signal.entry_price:,.2f}`\n"
        f"• **Take Profit:** `${signal.take_profit:,.2f}`\n"
        f"• **Stop Loss:** `${signal.stop_loss:,.2f}`\n\n"
        f"💡 **AI Rationale:**\n_{signal.summary}_"
    )

    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    async with httpx.AsyncClient() as client:
        await client.post(telegram_url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text_message,
            "parse_mode": "Markdown"
        })


@app.post("/webhook")
async def receive_tradingview_webhook(request: Request):
    """Webhook endpoint receiving real-time JSON triggers."""
    try:
        # Read alert payload from TradingView
        payload = await request.json()
        
        # Build prompt from incoming alert metrics
        prompt = f"""
        A real-time trading trigger just fired:
        {json.dumps(payload, indent=2)}

        Analyze if this trigger warrants an immediate entry based on risk/reward ratio.
        """

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SignalOutput,
            system_instruction=(
                "You are an automated real-time trading assistant. Validate incoming alert data "
                "and decide whether to approve or reject the trade signal."
            )
        )

        # Generate Gemini analysis
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=config
        )

        # Parse output
        signal = SignalOutput.model_validate_json(response.text)

        # Push to Telegram if action is valid
        if signal.action in ["BUY", "SELL"]:
            await push_to_telegram(signal)

        return {"status": "processed", "action": signal.action}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


