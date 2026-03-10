from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Set, Dict
from datetime import datetime, timedelta
import json
import uuid
import asyncio
import logging

from app.rag.processor import extract_text_from_pdf, chunk_text
from app.rag.embeddings import store_chunks, search_similar_chunks
from app.chat.engine import generate_answer, reset_session
from app.line.webhook import verify_line_signature, reply_line_message, push_line_message
from app.auth.utils import create_token, decode_token, generate_bot_id
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

app = FastAPI(title="AI Chatbot SaaS API")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ──────────────────────────────────────
# LINE Bot 狀態管理（in-memory）
# ──────────────────────────────────────

# 靜音名單：key = "{bot_id}:{line_user_id}"，資料蒐集完成後自動靜音
_muted_line_users: Set[str] = set()

# 防抖緩衝區：key = "{bot_id}:{line_user_id}"
# 值 = {"msgs": [], "reply_token": str, "task": asyncio.Task}
_line_buffers: Dict[str, dict] = {}

# 垃圾訊息關鍵字（通用，各 bot 可擴充）
_SPAM_KEYWORDS = ["資金週轉", "債務整合", "房屋二胎", "汽機車二貸", "若需要以上方案", "娛樂城", "博弈"]

DEBOUNCE_SECONDS = 15  # 防抖等待時間（LINE replyToken 60秒過期，15s 緩衝足夠安全）

# 允許跨域（前端呼叫用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────
# Auth
# ──────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
async def register(body: RegisterRequest):
    result = supabase.auth.admin.create_user({
        "email": body.email,
        "password": body.password,
        "email_confirm": True
    })
    if result.user:
        token = create_token(result.user.id)
        return {"token": token, "user_id": result.user.id}
    raise HTTPException(400, "註冊失敗")

@app.post("/auth/login")
async def login(body: LoginRequest):
    try:
        result = supabase.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password
        })
        if result.user:
            token = create_token(result.user.id)
            return {"token": token, "user_id": result.user.id}
    except Exception as e:
        raise HTTPException(401, f"帳號或密碼錯誤: {str(e)}")
    raise HTTPException(401, "帳號或密碼錯誤")


# ──────────────────────────────────────
# Bot 管理
# ──────────────────────────────────────

def get_user_id(authorization: str = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未授權")
    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    return payload["user_id"]

@app.post("/bots")
async def create_bot(
    name: str,
    authorization: Optional[str] = Header(None)
):
    user_id = get_user_id(authorization)
    bot_id = generate_bot_id()
    supabase.table("bots").insert({
        "id": bot_id,
        "user_id": user_id,
        "name": name
    }).execute()
    return {"bot_id": bot_id, "name": name}

@app.get("/bots")
async def list_bots(authorization: Optional[str] = Header(None)):
    user_id = get_user_id(authorization)
    result = supabase.table("bots").select("*").eq("user_id", user_id).execute()
    return result.data

@app.get("/bots/{bot_id}")
async def get_bot(bot_id: str, authorization: Optional[str] = Header(None)):
    """取得單一 Bot 完整設定（API Key 只回傳是否已設定）"""
    get_user_id(authorization)
    result = supabase.table("bots").select("*").eq("id", bot_id).execute()
    if not result.data:
        raise HTTPException(404, "Bot 不存在")
    bot = result.data[0]
    # 不回傳明文 API Key，只告訴前端有沒有設定
    bot["has_api_key"] = bool(bot.get("anthropic_api_key"))
    bot.pop("anthropic_api_key", None)
    return bot

@app.get("/bots/{bot_id}/welcome")
async def get_bot_welcome(bot_id: str):
    """無需驗證，widget 用 — 回傳歡迎訊息和快速選項"""
    result = supabase.table("bots").select("welcome_message, quick_replies, name").eq("id", bot_id).execute()
    if not result.data:
        raise HTTPException(404, "Bot 不存在")
    bot = result.data[0]
    return {
        "welcome_message": bot.get("welcome_message") or "",
        "quick_replies": bot.get("quick_replies") or [],
        "bot_name": bot.get("name", "AI 助理")
    }

class UpdateBotRequest(BaseModel):
    name: Optional[str] = None
    anthropic_api_key: Optional[str] = None   # 欄位名稱沿用，存的是 Gemini Key
    sheet_id: Optional[str] = None
    collect_fields: Optional[list] = None
    system_prompt: Optional[str] = None
    welcome_message: Optional[str] = None
    quick_replies: Optional[list] = None
    line_channel_secret: Optional[str] = None
    line_channel_access_token: Optional[str] = None
    # 預約系統
    calendar_id: Optional[str] = None
    slot_duration_minutes: Optional[int] = None
    business_hours: Optional[dict] = None
    # 關鍵字觸發
    keyword_triggers: Optional[list] = None
    # Instagram
    instagram_page_token: Optional[str] = None
    # 防抖
    debounce_seconds: Optional[int] = None

@app.patch("/bots/{bot_id}")
async def update_bot(
    bot_id: str,
    body: UpdateBotRequest,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    update_data = {}
    for k, v in body.model_dump().items():
        if v is not None:
            update_data[k] = v
        elif k in ("collect_fields", "quick_replies", "keyword_triggers"):
            # 允許儲存空 list
            update_data[k] = []
        elif k in ("system_prompt", "welcome_message") and v == "":
            update_data[k] = ""
    if update_data:
        supabase.table("bots").update(update_data).eq("id", bot_id).execute()
    return {"message": "更新成功"}


# ──────────────────────────────────────
# 知識庫上傳
# ──────────────────────────────────────

@app.post("/bots/{bot_id}/upload")
async def upload_document(
    bot_id: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")

    chunks = chunk_text(text)
    store_chunks(bot_id, chunks)

    return {"message": f"成功上傳，共 {len(chunks)} 個知識塊"}

class FAQRequest(BaseModel):
    content: str

@app.post("/bots/{bot_id}/faq")
async def add_faq(
    bot_id: str,
    body: FAQRequest,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    chunks = chunk_text(body.content)
    store_chunks(bot_id, chunks)
    return {"message": "FAQ 已加入知識庫"}

@app.get("/bots/{bot_id}/knowledge")
async def list_knowledge(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    result = supabase.table("knowledge_chunks")\
        .select("id, content, created_at")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)\
        .execute()
    return result.data

@app.delete("/bots/{bot_id}/knowledge")
async def clear_knowledge(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    supabase.table("knowledge_chunks").delete().eq("bot_id", bot_id).execute()
    return {"message": "知識庫已清除"}

@app.delete("/bots/{bot_id}/knowledge/{chunk_id}")
async def delete_chunk(
    bot_id: str,
    chunk_id: str,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    supabase.table("knowledge_chunks").delete().eq("id", chunk_id).eq("bot_id", bot_id).execute()
    return {"message": "已刪除"}

class UpdateChunkRequest(BaseModel):
    content: str

@app.patch("/bots/{bot_id}/knowledge/{chunk_id}")
async def update_chunk(
    bot_id: str,
    chunk_id: str,
    body: UpdateChunkRequest,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    supabase.table("knowledge_chunks")\
        .update({"content": body.content})\
        .eq("id", chunk_id)\
        .eq("bot_id", bot_id)\
        .execute()
    return {"message": "已更新"}


# ──────────────────────────────────────
# Session 管理
# ──────────────────────────────────────

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """清除指定 session 的對話記憶（前端重置用）"""
    from app.chat.engine import reset_session
    reset_session(session_id)
    return {"ok": True}


# ──────────────────────────────────────
# 對話（網站 Widget 用）
# ──────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.post("/bots/{bot_id}/chat")
async def chat(bot_id: str, body: ChatRequest):
    result = supabase.table("bots").select(
        "name, anthropic_api_key, sheet_id, collect_fields, system_prompt, "
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers"
    ).eq("id", bot_id).execute()
    bot_data = result.data[0] if result.data else {}
    bot_name = bot_data.get("name", "AI 助理")
    api_key = bot_data.get("anthropic_api_key")
    sheet_id = bot_data.get("sheet_id")
    collect_fields = bot_data.get("collect_fields") or []
    system_prompt = bot_data.get("system_prompt") or None
    calendar_id = bot_data.get("calendar_id") or None
    slot_duration = bot_data.get("slot_duration_minutes") or 60
    business_hours = bot_data.get("business_hours") or None
    keyword_triggers = bot_data.get("keyword_triggers") or None

    try:
        answer = generate_answer(
            bot_id, body.question, bot_name,
            api_key=api_key,
            collect_fields=collect_fields if collect_fields else None,
            sheet_id=sheet_id,
            session_id=body.session_id,
            custom_system_prompt=system_prompt,
            calendar_id=calendar_id,
            slot_duration_minutes=slot_duration,
            business_hours=business_hours,
            keyword_triggers=keyword_triggers,
        )
    except Exception as e:
        if "NO_API_KEY" in str(e):
            return {"answer": "⚠️ 尚未設定 Gemini API Key，請前往「⚙️ 設定」頁面填入後再試。"}
        raise

    supabase.table("conversations").insert({
        "bot_id": bot_id,
        "question": body.question,
        "answer": answer
    }).execute()

    return {"answer": answer}


# ──────────────────────────────────────
# LINE Webhook（升級版：防抖 + 靜音 + follow）
# ──────────────────────────────────────

def _get_bot_config(bot_id: str) -> dict:
    """從 Supabase 取得完整 bot 設定"""
    result = supabase.table("bots").select(
        "name, anthropic_api_key, sheet_id, collect_fields, system_prompt, welcome_message, "
        "line_channel_secret, line_channel_access_token, "
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers, debounce_seconds"
    ).eq("id", bot_id).execute()
    return result.data[0] if result.data else {}


async def _process_line_buffer(bot_id: str, user_id: str, buf_key: str, debounce_seconds: int = 15):
    """防抖計時到期後，合併訊息並呼叫 AI 回覆"""
    await asyncio.sleep(debounce_seconds)

    buf = _line_buffers.pop(buf_key, None)
    if not buf:
        return

    combined_msg = " ".join(buf["msgs"])
    session_id = f"line_{bot_id}_{user_id}"

    logging.info(f"[LINE] Processing buffered msgs for {user_id}: {combined_msg[:50]}")

    try:
        bot = _get_bot_config(bot_id)
        bot_name  = bot.get("name", "AI 助理")
        api_key   = bot.get("anthropic_api_key")
        sheet_id  = bot.get("sheet_id")
        collect_fields = bot.get("collect_fields") or []
        system_prompt  = bot.get("system_prompt") or None
        line_token     = bot.get("line_channel_access_token")
        calendar_id    = bot.get("calendar_id") or None
        slot_duration  = bot.get("slot_duration_minutes") or 60
        business_hours = bot.get("business_hours") or None
        keyword_triggers = bot.get("keyword_triggers") or None

        try:
            answer = generate_answer(
                bot_id, combined_msg, bot_name,
                api_key=api_key,
                collect_fields=collect_fields if collect_fields else None,
                sheet_id=sheet_id,
                session_id=session_id,
                custom_system_prompt=system_prompt,
                calendar_id=calendar_id,
                slot_duration_minutes=slot_duration,
                business_hours=business_hours,
                keyword_triggers=keyword_triggers,
            )
        except Exception as e:
            if "NO_API_KEY" in str(e):
                answer = "⚠️ 此 Bot 尚未設定 Gemini API Key，暫時無法回應。"
            else:
                raise

        # DATA_SAVE 已觸發 → engine 標記 handed_off → 同步靜音 in-memory set
        from app.chat.engine import get_session_status
        if get_session_status(session_id) == "handed_off":
            _muted_line_users.add(f"{bot_id}:{user_id}")
            logging.info(f"[LINE] Auto-muted {user_id} (handed_off)")

        # 優先用 push（不依賴 replyToken 過期），失敗才 fallback
        push_ok = await push_line_message(user_id, answer, access_token=line_token)
        if push_ok != 200:
            reply_token = buf.get("reply_token", "")
            if reply_token:
                await reply_line_message(reply_token, answer, access_token=line_token)
                logging.info(f"[LINE] Fallback to replyToken for {user_id}")

    except Exception as e:
        logging.error(f"[LINE] process_line_buffer error: {e}")


@app.post("/line/webhook/{bot_id}")
async def line_webhook(bot_id: str, request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    # 先取 bot 設定（需要用 bot 專屬的 Channel Secret 驗簽名）
    bot = _get_bot_config(bot_id)
    line_secret = bot.get("line_channel_secret")
    line_token  = bot.get("line_channel_access_token")

    if not verify_line_signature(body, signature, channel_secret=line_secret):
        raise HTTPException(400, "簽名驗證失敗")

    data = json.loads(body)
    events = data.get("events", [])

    for event in events:
        user_id = event.get("source", {}).get("userId", "unknown")
        buf_key = f"{bot_id}:{user_id}"

        # ── Follow 事件（加好友）→ 發歡迎語 ──
        if event["type"] == "follow":
            reply_token = event.get("replyToken")
            if reply_token:
                welcome = bot.get("welcome_message") or f"你好！我是{bot.get('name', 'AI 助理')}，有什麼可以幫您的嗎？😊"
                session_id = f"line_{bot_id}_{user_id}"
                from app.chat.session_store import get_or_create
                get_or_create(session_id)
                await reply_line_message(reply_token, welcome, access_token=line_token)
            continue

        # ── 只處理文字訊息 ──
        if event["type"] != "message" or event["message"]["type"] != "text":
            continue

        user_msg = event["message"]["text"]
        reply_token = event["replyToken"]

        # ── 用戶自助重置 ──
        if user_msg.strip() in ["/reset", "重來", "重置", "重新開始"]:
            session_id = f"line_{bot_id}_{user_id}"
            reset_session(session_id)
            _muted_line_users.discard(buf_key)
            welcome = bot.get("welcome_message") or f"（記憶已重置）你好！我是{bot.get('name', 'AI 助理')}，有什麼可以幫您的嗎？😊"
            await reply_line_message(reply_token, welcome, access_token=line_token)
            continue

        # ── 靜音檢查（已交接，不再 AI 回應）──
        if buf_key in _muted_line_users:
            logging.info(f"[LINE] Muted user {user_id}, ignoring message")
            continue

        # ── 垃圾訊息過濾 ──
        if any(kw in user_msg for kw in _SPAM_KEYWORDS) and len(user_msg) > 30:
            _muted_line_users.add(buf_key)
            logging.info(f"[LINE] Spam-muted {user_id}: {user_msg[:30]}...")
            continue

        # ── 防抖緩衝（15 秒）──
        if buf_key in _line_buffers:
            # 取消舊計時器，累積訊息
            old_task = _line_buffers[buf_key].get("task")
            if old_task and not old_task.done():
                old_task.cancel()
            _line_buffers[buf_key]["msgs"].append(user_msg)
            _line_buffers[buf_key]["reply_token"] = reply_token  # 永遠用最新的 replyToken
        else:
            _line_buffers[buf_key] = {
                "msgs": [user_msg],
                "reply_token": reply_token,
                "task": None
            }

        # 啟動新計時器
        bot_debounce = bot.get("debounce_seconds") or 15
        task = asyncio.create_task(_process_line_buffer(bot_id, user_id, buf_key, bot_debounce))
        _line_buffers[buf_key]["task"] = task
        logging.info(f"[LINE] Buffered msg from {user_id}: '{user_msg}' ({bot_debounce}s timer)")

    # 立即回 200 給 LINE Server，避免 timeout
    return {"status": "ok"}


# ──────────────────────────────────────
# Layer 2 AI 設定助手
# ──────────────────────────────────────

class AssistantRequest(BaseModel):
    bot_id: str
    message: str
    session_id: Optional[str] = None

@app.post("/assistant/chat")
async def assistant_chat(
    body: AssistantRequest,
    authorization: Optional[str] = Header(None)
):
    """AI 設定助手：用 Gemini Function Calling 幫用戶直接操作 Bot 設定"""
    user_id = get_user_id(authorization)
    # 驗證 bot 屬於該用戶，並取 API Key
    r = supabase.table("bots").select("anthropic_api_key").eq("id", body.bot_id).eq("user_id", user_id).execute()
    if not r.data:
        raise HTTPException(404, "Bot 不存在")
    api_key = r.data[0].get("anthropic_api_key")
    session_id = body.session_id or f"assistant_{user_id}_{body.bot_id}"

    from app.assistant.engine import run_assistant
    reply = run_assistant(body.bot_id, body.message, session_id, api_key)
    return {"reply": reply}


# ──────────────────────────────────────
# 對話記錄查詢
# ──────────────────────────────────────

@app.get("/bots/{bot_id}/analytics")
async def get_bot_analytics(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    """Bot 數據分析：總對話數、今日、本週、7天趨勢、熱門問題、峰值時段、週成長率"""
    user_id = get_user_id(authorization)
    bot = supabase.table("bots").select("id").eq("id", bot_id).eq("user_id", user_id).execute()
    if not bot.data:
        raise HTTPException(404, "Bot 不存在")

    now = datetime.utcnow()
    # 時區偏移（台灣 UTC+8）
    tw_offset = timedelta(hours=8)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_start      = (now - timedelta(days=7)).isoformat()
    prev_week_start = (now - timedelta(days=14)).isoformat()

    total_r = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).execute()
    today_r = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).gte("created_at", today_start).execute()
    week_r  = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).gte("created_at", week_start).execute()
    prev_week_r = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).gte("created_at", prev_week_start).lt("created_at", week_start).execute()

    # 週成長率
    this_week_count = week_r.count or 0
    prev_week_count = prev_week_r.count or 0
    if prev_week_count > 0:
        week_growth = round((this_week_count - prev_week_count) / prev_week_count * 100, 1)
    elif this_week_count > 0:
        week_growth = 100.0
    else:
        week_growth = 0.0

    # 7 天每日分佈（台灣時間）
    rows = supabase.table("conversations").select("created_at").eq("bot_id", bot_id).gte("created_at", week_start).execute()
    daily: dict = {}
    hourly: dict = {}
    for row in (rows.data or []):
        ts = row["created_at"]
        # 轉台灣時間
        from datetime import timezone
        try:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            dt_tw = dt_utc + tw_offset
            day = dt_tw.strftime("%Y-%m-%d")
            hour = dt_tw.hour
        except Exception:
            day = ts[:10]
            hour = int(ts[11:13]) if len(ts) > 12 else 0
        daily[day] = daily.get(day, 0) + 1
        hourly[hour] = hourly.get(hour, 0) + 1

    daily_counts = []
    for i in range(6, -1, -1):
        d = ((now + tw_offset) - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_counts.append({"date": d, "count": daily.get(d, 0)})

    # 24 小時分佈
    hourly_distribution = [{"hour": h, "count": hourly.get(h, 0)} for h in range(24)]

    # 熱門問題（最近 200 筆，計算重複次數，取 top 10）
    all_q = supabase.table("conversations").select("question").eq("bot_id", bot_id).order("created_at", desc=True).limit(200).execute()
    q_counter: dict = {}
    for row in (all_q.data or []):
        q = row["question"].strip()
        q_counter[q] = q_counter.get(q, 0) + 1
    top_questions = sorted(q_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_questions = [{"question": q, "count": c} for q, c in top_questions]

    # 最近 10 筆問題
    recent = supabase.table("conversations").select("question, created_at").eq("bot_id", bot_id).order("created_at", desc=True).limit(10).execute()

    return {
        "total":                total_r.count or 0,
        "today":                today_r.count or 0,
        "this_week":            this_week_count,
        "prev_week":            prev_week_count,
        "week_growth":          week_growth,
        "daily_counts":         daily_counts,
        "hourly_distribution":  hourly_distribution,
        "top_questions":        top_questions,
        "recent_questions":     [r["question"] for r in (recent.data or [])]
    }


@app.get("/bots/{bot_id}/conversations")
async def get_conversations(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    result = supabase.table("conversations")\
        .select("*")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)\
        .limit(100)\
        .execute()
    return result.data


# ──────────────────────────────────────
# Instagram Webhook
# ──────────────────────────────────────

@app.get("/instagram/webhook/{bot_id}")
async def instagram_webhook_verify(
    bot_id: str,
    hub_mode: Optional[str]         = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str]    = Query(None, alias="hub.challenge"),
):
    """Meta Webhook 驗證：Verify Token = bot_id"""
    if hub_mode == "subscribe" and hub_verify_token == bot_id:
        logging.info(f"[Instagram] Webhook verified for bot {bot_id[:8]}")
        return PlainTextResponse(hub_challenge)
    raise HTTPException(403, "驗證失敗：Verify Token 不符")


@app.post("/instagram/webhook/{bot_id}")
async def instagram_webhook(bot_id: str, request: Request):
    """接收 Instagram DM，呼叫 AI 回覆"""
    data = await request.json()

    # Meta 驗證 ping
    if data.get("object") not in ("instagram", "page"):
        return {"status": "ignored"}

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            msg       = event.get("message", {})
            text      = msg.get("text", "").strip()

            # 忽略：echo（自己發的訊息）、無文字
            if not text or msg.get("is_echo"):
                continue

            logging.info(f"[Instagram] bot={bot_id[:8]} sender={sender_id} msg={text[:50]}")
            asyncio.create_task(_process_instagram_message(bot_id, sender_id, text))

    return {"status": "ok"}


async def _process_instagram_message(bot_id: str, sender_id: str, text: str):
    """非同步處理 Instagram 訊息"""
    try:
        bot = _get_bot_config(bot_id)
        page_token = bot.get("instagram_page_token")
        if not page_token:
            logging.warning(f"[Instagram] bot {bot_id[:8]} has no page_token, skipping")
            return

        bot_name       = bot.get("name", "AI 助理")
        api_key        = bot.get("anthropic_api_key")
        sheet_id       = bot.get("sheet_id")
        collect_fields = bot.get("collect_fields") or []
        system_prompt  = bot.get("system_prompt") or None
        calendar_id    = bot.get("calendar_id") or None
        slot_duration  = bot.get("slot_duration_minutes") or 60
        business_hours = bot.get("business_hours") or None
        keyword_triggers = bot.get("keyword_triggers") or None
        session_id     = f"ig_{bot_id}_{sender_id}"

        try:
            answer = generate_answer(
                bot_id, text, bot_name,
                api_key=api_key,
                collect_fields=collect_fields if collect_fields else None,
                sheet_id=sheet_id,
                session_id=session_id,
                custom_system_prompt=system_prompt,
                calendar_id=calendar_id,
                slot_duration_minutes=slot_duration,
                business_hours=business_hours,
                keyword_triggers=keyword_triggers,
            )
        except Exception as e:
            if "NO_API_KEY" in str(e):
                answer = "⚠️ 此 Bot 尚未設定 Gemini API Key，暫時無法回應。"
            else:
                raise

        from app.instagram.webhook import send_instagram_message
        status = await send_instagram_message(sender_id, answer, page_token)
        logging.info(f"[Instagram] Sent reply to {sender_id}, status={status}")

        supabase.table("conversations").insert({
            "bot_id": bot_id, "question": text, "answer": answer
        }).execute()

    except Exception as e:
        logging.error(f"[Instagram] process error: {e}")


@app.get("/")
def root():
    return {"status": "AI Chatbot SaaS running 🔥"}
