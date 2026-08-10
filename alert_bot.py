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
# TECHNICAL INDICATORS: EMA 9, 21, 100 + ATR
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

    return df

# ---------------------------------------------------------
# AI ANALYST RISK FILTER
# ---------------------------------------------------------

async def analyze_signal_with_ai(price: float, action: str, ema9: float, ema21: float, ema100: float, atr: float, trigger_reason: str) -> str:
    if not GROQ_API_KEY:
        return "APPROVE"
        
    prompt = f"""
    You are an institutional Gold Risk Analyst. Evaluate this trade setup:
    - Pair: Spot Gold (XAU/USD)
    - Action Proposed: {action}
    - Trigger Source: {trigger_reason}
    - Current Price: ${price:.2f}
    - 5M EMAs -> 9: ${ema9:.2f} | 21: ${ema21:.2f} | 100: ${ema100:.2f}
    - Current ATR: ${atr:.2f}

    STRICT RULES:
    1. VETO if proposed BUY occurs when 9 EMA is below 21 EMA or 100 EMA.
    2. VETO if proposed SELL occurs when 9 EMA is above 21 EMA or 100 EMA.

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
                logging.error(f"[TELEGRAM ERROR] Status: {res.status_code} | Body: {res.text}")
    except Exception as e:
        logging.error(f"[TELEGRAM EXCEPTION] {e}")

async def setup_auto_webhook():
    if not APP_URL or not BOT_TOKEN:
        return
    webhook_url = f"{APP_URL.rstrip('/')}/telegram-webhook"
    target_api = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={webhook_url}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            res = await client.get(target_api)
            data = res.json()
            if data.get("ok"):
                logging.info(f"[WEBHOOK SETUP SUCCESS] Telegram Webhook registered to: {webhook_url}")
            else:
                logging.error(f"[WEBHOOK SETUP FAILED] Response: {data}")
    except Exception as e:
        logging.error(f"[WEBHOOK SETUP EXCEPTION] {e}")

# ---------------------------------------------------------
# TRADE TRACKER & DATABASE UPDATES
# ---------------------------------------------------------

def update_open_trades(current_price: float, high_3c: float, low_3c: float):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM signals WHERE status = 'EXECUTED' AND outcome NOT LIKE 'WIN (TP2%' AND outcome NOT LIKE 'LOSS%' AND outcome NOT LIKE 'CLOSED%'")
    open_trades = cur.fetchall()
    
    for trade in open_trades:
        trade_id = trade['id']
        action = trade['action']
        sl = float(trade['sl_price'])
        tp1 = float(trade['tp1_price'])
        tp2 = float(trade['tp2_price'])
        outcome = trade['outcome']
        
        new_outcome = None
        exit_price = None
        
        if action == "BUY":
            if low_3c <= sl:
                if "TP1 HIT" in outcome:
                    new_outcome = "CLOSED (TP1 HIT / SL BE)"
                    exit_price = tp1
                else:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
            elif high_3c >= tp2:
                new_outcome = "WIN (TP2 HIT)"
                exit_price = tp2
            elif high_3c >= tp1 and "TP1 HIT" not in outcome:
                new_outcome = "WIN (TP1 HIT)"
                exit_price = tp1
                
        elif action == "SELL":
            if high_3c >= sl:
                if "TP1 HIT" in outcome:
                    new_outcome = "CLOSED (TP1 HIT / SL BE)"
                    exit_price = tp1
                else:
                    new_outcome = "LOSS (SL HIT)"
                    exit_price = sl
            elif low_3c <= tp2:
                new_outcome = "WIN (TP2 HIT)"
                exit_price = tp2
            elif low_3c <= tp1 and "TP1 HIT" not in outcome:
                new_outcome = "WIN (TP1 HIT)"
                exit_price = tp1

        if new_outcome:
            cur.execute("""
                UPDATE signals 
                SET outcome = %s, exit_price = %s, closed_at = NOW() 
                WHERE id = %s
            """, (new_outcome, exit_price, trade_id))
            conn.commit()
            logging.info(f"[TRADE TRACKER] Signal ID {trade_id} updated -> {new_outcome}")
            
    cur.close()
    conn.close()

# ---------------------------------------------------------
# BACKGROUND MARKET SCANNER (ADX FILTER OFF)
# ---------------------------------------------------------

async def background_scanning_loop():
    while True:
        try:
            url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=5min&outputsize=100&apikey={TWELVE_DATA_API_KEY}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url)
                data = res.json()
                
            if "values" in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col].astype(float)
                df = df.sort_values('datetime').reset_index(drop=True)
                
                df = calculate_indicators(df)
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                price = float(latest['close'])
                ema9 = float(latest['ema9'])
                ema21 = float(latest['ema21'])
                ema100 = float(latest['ema100'])
                atr = float(latest['atr'])
                
                high_3c = float(df['high'].tail(3).max())
                low_3c = float(df['low'].tail(3).min())
                
                update_open_trades(price, high_3c, low_3c)

                # Individual Crossover Detections
                ema9_cross_above_21 = (prev['ema9'] <= prev['ema21']) and (latest['ema9'] > latest['ema21'])
                ema9_cross_above_100 = (prev['ema9'] <= prev['ema100']) and (latest['ema9'] > latest['ema100'])
                ema21_cross_above_100 = (prev['ema21'] <= prev['ema100']) and (latest['ema21'] > latest['ema100'])

                ema9_cross_below_21 = (prev['ema9'] >= prev['ema21']) and (latest['ema9'] < latest['ema21'])
                ema9_cross_below_100 = (prev['ema9'] >= prev['ema100']) and (latest['ema9'] < latest['ema100'])
                ema21_cross_below_100 = (prev['ema21'] >= prev['ema100']) and (latest['ema21'] < latest['ema100'])

                # Triple Crossover Detection
                is_triple_bullish_cross = ema9_cross_above_21 and ema9_cross_above_100 and ema21_cross_above_100
                is_triple_bearish_cross = ema9_cross_below_21 and ema9_cross_below_100 and ema21_cross_below_100

                proposed_action = "HOLD"
                trigger_reason = ""

                # --- 1. TRIPLE CROSSOVER (Highest Priority) ---
                if is_triple_bullish_cross:
                    proposed_action = "BUY"
                    trigger_reason = "🔥 TRIPLE EMA BULLISH EXPLOSION (Big Move Imminent)"
                elif is_triple_bearish_cross:
                    proposed_action = "SELL"
                    trigger_reason = "🔥 TRIPLE EMA BEARISH EXPLOSION (Big Move Imminent)"

                # --- 2. INDIVIDUAL CROSSOVERS ---
                elif ema9_cross_above_21 and latest['ema21'] > latest['ema100']:
                    proposed_action = "BUY"
                    trigger_reason = "EMA 9 Potong Ke Atas EMA 21 (Valid Uptrend)"
                elif ema9_cross_above_100:
                    proposed_action = "BUY"
                    trigger_reason = "EMA 9 Potong Ke Atas EMA 100 (Awal Reversal Bullish)"
                elif ema21_cross_above_100:
                    proposed_action = "BUY"
                    trigger_reason = "EMA 21 Potong Ke Atas EMA 100 (Konfirmasi Reversal Bullish)"

                elif ema9_cross_below_21 and latest['ema21'] < latest['ema100']:
                    proposed_action = "SELL"
                    trigger_reason = "EMA 9 Potong Ke Bawah EMA 21 (Valid Downtrend)"
                elif ema9_cross_below_100:
                    proposed_action = "SELL"
                    trigger_reason = "EMA 9 Potong Ke Bawah EMA 100 (Awal Reversal Bearish)"
                elif ema21_cross_below_100:
                    proposed_action = "SELL"
                    trigger_reason = "EMA 21 Potong Ke Bawah EMA 100 (Konfirmasi Reversal Bearish)"

                # Proximity Cap Guard Filter (2.5 x ATR)
                max_distance_cap = 2.5 * atr
                if proposed_action == "BUY" and (price - ema21) > max_distance_cap:
                    logging.info(f"[PROXIMITY VETO] BUY blocked: Price ${price:.2f} is ${price - ema21:.2f} above 21 EMA")
                    proposed_action = "HOLD"
                elif proposed_action == "SELL" and (ema21 - price) > max_distance_cap:
                    logging.info(f"[PROXIMITY VETO] SELL blocked: Price ${price:.2f} is ${ema21 - price:.2f} below 21 EMA")
                    proposed_action = "HOLD"

                # Cooldown Guard Filter
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT entry_price, outcome FROM signals WHERE status = 'EXECUTED' ORDER BY id DESC LIMIT 1")
                last_trade = cur.fetchone()
                
                min_distance = 2.5
                if last_trade and abs(price - float(last_trade['entry_price'])) < min_distance:
                    proposed_action = "HOLD"
                    
                logging.info(f"[MARKET SCAN] Price: ${price:.2f} | 9: ${ema9:.2f} | 21: ${ema21:.2f} | 100: ${ema100:.2f} | Action: {proposed_action}")

                # AI Evaluation & Signal Dispatch
                if proposed_action != "HOLD":
                    ai_decision = await analyze_signal_with_ai(price, proposed_action, ema9, ema21, ema100, atr, trigger_reason)
                    
                    sl_dist = max(4.0, min(7.0, atr * 1.5))
                    tp1_dist = sl_dist * 1.5
                    tp2_dist = sl_dist * 3.0
                    
                    if proposed_action == "BUY":
                        sl = price - sl_dist
                        tp1 = price + tp1_dist
                        tp2 = price + tp2_dist
                    else:
                        sl = price + sl_dist
                        tp1 = price - tp1_dist
                        tp2 = price - tp2_dist

                    status_str = "EXECUTED" if ai_decision == "APPROVE" else "VETOED"
                    
                    cur.execute("""
                        INSERT INTO signals (action, entry_price, sl_price, tp1_price, tp2_price, status, outcome, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, 'PENDING', NOW())
                        RETURNING id;
                    """, (proposed_action, float(price), float(sl), float(tp1), float(tp2), status_str))
                    conn.commit()
                    new_id = cur.fetchone()['id']

                    if ai_decision == "APPROVE":
                        msg = (
                            f"⚡ EMA CROSSOVER SIGNAL #{new_id}\n\n"
                            f"Action: *{proposed_action} XAU/USD*\n"
                            f"Entry Price: *${price:.2f}*\n"
                            f"Stop Loss: *${sl:.2f}*\n"
                            f"Take Profit 1: *${tp1:.2f}*\n"
                            f"Take Profit 2: *${tp2:.2f}*\n"
                            f"Pemicu: *{trigger_reason}*"
                        )
                        await send_telegram_alert(msg)

                cur.close()
                conn.close()
                del df
                gc.collect()

        except Exception as e:
            logging.error(f"[SCANNER EXCEPTION] {e}")
            
        await asyncio.sleep(360)

# ---------------------------------------------------------
# FASTAPI APP, MT5 BRIDGE & TELEGRAM WEBHOOK HANDLERS
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(background_scanning_loop())
    await setup_auto_webhook()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "Live", "bot": "Gold All EMA Crossover Signal Bot"}

# MT5 Bridge Endpoint
@app.get("/get-latest-signal")
def get_latest_signal():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM signals WHERE status = 'EXECUTED' ORDER BY id DESC LIMIT 1")
    signal = cur.fetchone()
    cur.close()
    conn.close()
    if signal:
        return {
            "id": signal['id'],
            "action": signal['action'],
            "price": float(signal['entry_price']),
            "sl": float(signal['sl_price']),
            "tp1": float(signal['tp1_price']),
            "tp2": float(signal['tp2_price']),
            "created_at": str(signal['created_at'])
        }
    return {"status": "NO_SIGNAL"}

# Telegram Webhook Handler
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    if "message" in data:
        message = data["message"]
        chat_id = str(message.get("chat", {}).get("id"))
        text = message.get("text", "").strip()

        if text in ["/stats", "/logs", "/pips", "/help"]:
            conn = get_db_connection()
            cur = conn.cursor()

            if text == "/stats":
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

                cur.execute("SELECT action, entry_price, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                closed_trades = cur.fetchall()

                total_pips = 0.0
                win_pips = 0.0
                loss_pips = 0.0
                total_wins_count = tp1_wins + tp2_wins

                for t in closed_trades:
                    entry = float(t['entry_price'])
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

                reply = (
                    f"📊 *EMA ALL-CROSS PERFORMANCE*\n"
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

            elif text == "/pips":
                cur.execute("SELECT action, entry_price, exit_price FROM signals WHERE status = 'EXECUTED' AND exit_price IS NOT NULL")
                trades = cur.fetchall()

                total_pips = 0.0
                gross_win_pips = 0.0
                gross_loss_pips = 0.0
                winning_trades_count = 0
                losing_trades_count = 0

                for t in trades:
                    entry = float(t['entry_price'])
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

            elif text == "/logs":
                cur.execute("""
                    SELECT id, action, entry_price, exit_price, outcome, created_at
                    FROM signals
                    WHERE status = 'EXECUTED'
                    ORDER BY id DESC
                    LIMIT 10
                """)
                logs = cur.fetchall()

                if not logs:
                    reply = "📜 *LAST 10 TRADE LOGS:*\n\n_Belum ada transaksi yang tereksekusi di database._"
                else:
                    reply = "📜 *LAST 10 DETAILED TRADE LOGS:*\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    for l in logs:
                        trade_id = l['id']
                        action = l['action']
                        entry = float(l['entry_price'])
                        exit_p = float(l['exit_price']) if l['exit_price'] else None
                        outcome = l['outcome']
                        date_str = l['created_at'].strftime("%m-%d %H:%M") if l['created_at'] else "N/A"

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

            elif text == "/help":
                reply = (
                    f"🤖 *EMA ALL-CROSS SIGNAL BOT COMMANDS:*\n\n"
                    f"`/stats` - Laporan Performa Win-Rate & Risiko Lengkap\n"
                    f"`/pips` - Rincian Pips Kotor/Bersih & Estimasi Profit USD\n"
                    f"`/logs` - Tampilan Detail 10 Sinyal Terakhir & Hasilnya\n"
                    f"`/help` - Menampilkan Panduan Perintah Interaktif"
                )

            cur.close()
            conn.close()

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json={"chat_id": chat_id, "text": reply, "parse_mode": "Markdown"})

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("alert_bot:app", host="0.0.0.0", port=8000)
