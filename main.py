from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Set, Dict
from datetime import datetime, timedelta
import json
import time
import uuid
import asyncio
import os
import logging
import httpx

from app.rag.processor import extract_text_from_pdf, chunk_text
from app.rag.embeddings import store_chunks, search_similar_chunks
from app.chat.engine import generate_answer, reset_session
from app.line.webhook import verify_line_signature, reply_line_message, push_line_message
from app.auth.utils import create_token, decode_token, generate_bot_id
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

app = FastAPI(title="AI Chatbot SaaS API")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2026-03-25-v2", "delete_endpoint_loaded": True}

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
        created_at_str = result.user.created_at.isoformat() if result.user.created_at else ""
        token = create_token(result.user.id, email=result.user.email or "", created_at=created_at_str)
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
            created_at_str = result.user.created_at.isoformat() if result.user.created_at else ""
            token = create_token(result.user.id, email=result.user.email or "", created_at=created_at_str)
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

def get_bot_slots(user_id: str) -> int:
    """回傳該用戶目前有效的付費 Bot 名額總數（商業版=10, 單Bot=1）"""
    try:
        rows = supabase.table("bot_subscriptions") \
            .select("slots") \
            .eq("user_id", user_id) \
            .eq("status", "active") \
            .execute()
        return sum(r.get("slots", 1) for r in (rows.data or []))
    except Exception:
        return 0


def is_bot_paid(bot_id: str) -> bool:
    """檢查此 bot 是否為付費 bot（有付費名額）"""
    bot_row = supabase.table("bots").select("user_id").eq("id", bot_id).execute()
    if not bot_row.data:
        return False
    user_id = bot_row.data[0]["user_id"]
    slots   = get_bot_slots(user_id)
    all_bots = supabase.table("bots").select("id").eq("user_id", user_id).order("created_at").execute()
    bot_ids  = [b["id"] for b in (all_bots.data or [])]
    idx      = bot_ids.index(bot_id) if bot_id in bot_ids else len(bot_ids)
    return idx < slots


def check_message_allowed(bot_id: str) -> tuple[bool, str]:
    """
    檢查此 bot 是否允許再收一則訊息。
    回傳 (allowed: bool, reason: str)
    """
    bot_row = supabase.table("bots").select("user_id").eq("id", bot_id).execute()
    if not bot_row.data:
        return False, "Bot 不存在"

    user_id = bot_row.data[0]["user_id"]
    slots   = get_bot_slots(user_id)

    # 找出此 bot 在該用戶所有 bot 中的排序（最舊優先）
    all_bots = supabase.table("bots").select("id").eq("user_id", user_id).order("created_at").execute()
    bot_ids  = [b["id"] for b in (all_bots.data or [])]
    idx      = bot_ids.index(bot_id) if bot_id in bot_ids else len(bot_ids)
    is_paid  = idx < slots  # 前 N 個 bot 為付費

    if is_paid:
        return True, ""

    # 未付費 → 鎖定，不允許任何訊息
    return False, "此服務目前已暫停，如需繼續使用請聯絡我們。"


class CreateBotRequest(BaseModel):
    system_prompt: Optional[str] = None
    collect_fields: Optional[list] = None
    welcome_message: Optional[str] = None

@app.post("/bots")
async def create_bot(
    name: str,
    body: CreateBotRequest = CreateBotRequest(),
    authorization: Optional[str] = Header(None)
):
    user_id = get_user_id(authorization)

    # 限制：免費 1 個，每個付費訂閱 +1
    existing = supabase.table("bots").select("id", count="exact").eq("user_id", user_id).execute()
    current_count = existing.count or 0
    slots = get_bot_slots(user_id)
    max_bots = 1 + slots  # 1 免費 + N 付費

    if current_count >= max_bots:
        raise HTTPException(403, f"已達上限（{max_bots} 個 Bot）。請至定價頁購買更多名額。")

    bot_id = generate_bot_id()
    insert_data: dict = {"id": bot_id, "user_id": user_id, "name": name}
    if body.system_prompt is not None:
        insert_data["system_prompt"] = body.system_prompt
    if body.collect_fields is not None:
        insert_data["collect_fields"] = body.collect_fields
    if body.welcome_message is not None:
        insert_data["welcome_message"] = body.welcome_message
    supabase.table("bots").insert(insert_data).execute()
    return {"bot_id": bot_id, "name": name}

@app.get("/bots")
async def list_bots(authorization: Optional[str] = Header(None)):
    user_id   = get_user_id(authorization)
    slots     = get_bot_slots(user_id)
    result    = supabase.table("bots").select("*").eq("user_id", user_id).order("created_at").execute()
    bots      = result.data or []
    # 最早建立的 bot 依序分配付費名額
    for i, bot in enumerate(bots):
        bot["plan"] = "paid" if i < slots else "free"
    return bots

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
    # 下班時間
    off_hours_message: Optional[str] = None

@app.patch("/bots/{bot_id}")
async def update_bot(
    bot_id: str,
    body: UpdateBotRequest,
    authorization: Optional[str] = Header(None)
):
    get_user_id(authorization)
    update_data = {}
    # exclude_unset=True：只處理請求中明確傳入的欄位，避免未傳的欄位被誤清空
    for k, v in body.model_dump(exclude_unset=True).items():
        if v is not None:
            update_data[k] = v
        elif k in ("collect_fields", "quick_replies", "keyword_triggers"):
            # 明確傳入空 list → 允許清空
            update_data[k] = []
        elif k in ("system_prompt", "welcome_message"):
            # 明確傳入空字串 → 允許清空
            update_data[k] = ""

    # 儲存 Instagram token 時，自動抓 IG Business Account ID 存入 DB（用於 webhook 路由）
    if "instagram_page_token" in update_data and update_data["instagram_page_token"]:
        try:
            token_val = update_data["instagram_page_token"]
            async with httpx.AsyncClient() as _hc:
                # Step 1: 取得 Facebook Page ID
                _r = await _hc.get(
                    "https://graph.facebook.com/me",
                    params={"access_token": token_val, "fields": "id,name"},
                    timeout=5,
                )
                if _r.status_code == 200:
                    page_id = _r.json().get("id", "")
                    ig_account_id = ""

                    # Step 2: 從 Page 取得 Instagram Business Account ID
                    if page_id:
                        _r2 = await _hc.get(
                            f"https://graph.facebook.com/{page_id}",
                            params={"access_token": token_val, "fields": "instagram_business_account"},
                            timeout=5,
                        )
                        if _r2.status_code == 200:
                            ig_account_id = _r2.json().get("instagram_business_account", {}).get("id", "")

                    # 儲存 IG Business Account ID 用於發送訊息
                    # 同時儲存 Page ID 用於 webhook 路由（page 物件情況）
                    final_id = ig_account_id or page_id
                    if final_id:
                        update_data["instagram_account_id"] = final_id
                    if page_id:
                        update_data["facebook_page_id"] = page_id
                    logging.info(f"[Instagram] ig_account_id={ig_account_id}, page_id={page_id}, stored={final_id} for bot {bot_id[:8]}")
        except Exception as e:
            logging.warning(f"[Instagram] Failed to fetch account ID: {e}")

    if update_data:
        supabase.table("bots").update(update_data).eq("id", bot_id).execute()
    return {"message": "更新成功"}


@app.delete("/bots/{bot_id}")
async def delete_bot(bot_id: str, authorization: Optional[str] = Header(None)):
    """刪除 bot（包含相關的 knowledge、sessions、conversations）"""
    user_id = get_user_id(authorization)
    # 驗證 bot 屬於該 user
    bot = supabase.table("bots").select("user_id").eq("id", bot_id).execute()
    if not bot.data or bot.data[0]["user_id"] != user_id:
        raise HTTPException(403, "無權刪除此 Bot")

    # 刪除關聯資料
    supabase.table("knowledge_chunks").delete().eq("bot_id", bot_id).execute()
    supabase.table("sessions").delete().eq("bot_id", bot_id).execute()
    supabase.table("conversations").delete().eq("bot_id", bot_id).execute()
    # 刪除 bot 本身
    supabase.table("bots").delete().eq("id", bot_id).execute()
    return {"status": "deleted"}


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
    # 免費 Bot 不支援網站 Widget
    if not is_bot_paid(bot_id):
        return {"answer": "此 Bot 為免費方案，不支援網站嵌入功能。請升級方案後使用，或透過 LINE 試用。"}

    allowed, _ = check_message_allowed(bot_id)
    if not allowed:
        return {"answer": ""}

    result = supabase.table("bots").select(
        "name, anthropic_api_key, sheet_id, collect_fields, system_prompt, "
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers, off_hours_message"
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
    off_hours_message = bot_data.get("off_hours_message") or None

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
            off_hours_message=off_hours_message,
        )
    except Exception as e:
        if "NO_API_KEY" in str(e):
            return {"answer": "⚠️ 尚未設定 Gemini API Key，請前往「⚙️ 設定」頁面填入後再試。"}
        raise

    # 只記錄真實流量（widget_ 或 line_ 開頭），排除測試對話
    sid = body.session_id or ""
    if sid.startswith("widget_") or sid.startswith("line_"):
        supabase.table("conversations").insert({
            "bot_id": bot_id,
            "question": body.question,
            "answer": answer,
            "session_id": sid,
        }).execute()

    return {"answer": answer}


# ──────────────────────────────────────
# LINE Webhook（升級版：防抖 + 靜音 + follow）
# ──────────────────────────────────────

def _get_bot_config(bot_id: str) -> dict:
    """從 Supabase 取得完整 bot 設定"""
    result = supabase.table("bots").select(
        "name, anthropic_api_key, sheet_id, collect_fields, system_prompt, welcome_message, quick_replies, "
        "line_channel_secret, line_channel_access_token, "
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers, debounce_seconds, "
        "instagram_page_token, instagram_account_id, facebook_page_id, off_hours_message"
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

    # 訂閱檢查：未付費直接靜默，不回覆任何訊息
    allowed, _ = check_message_allowed(bot_id)
    if not allowed:
        return

    try:
        bot = _get_bot_config(bot_id)
        bot_name  = bot.get("name", "AI 助理")
        api_key   = bot.get("anthropic_api_key")
        sheet_id  = bot.get("sheet_id")
        collect_fields = bot.get("collect_fields") or []
        system_prompt  = bot.get("system_prompt") or None
        line_token     = bot.get("line_channel_access_token")
        quick_replies  = bot.get("quick_replies") or None
        calendar_id    = bot.get("calendar_id") or None
        slot_duration  = bot.get("slot_duration_minutes") or 60
        business_hours = bot.get("business_hours") or None
        keyword_triggers  = bot.get("keyword_triggers") or None
        off_hours_message = bot.get("off_hours_message") or None

        # 抓 LINE 暱稱，存入試算表方便對應聊天室
        line_display_name = ""
        if sheet_id and line_token:
            try:
                async with httpx.AsyncClient() as _hc:
                    _r = await _hc.get(
                        f"https://api.line.me/v2/bot/profile/{user_id}",
                        headers={"Authorization": f"Bearer {line_token}"},
                        timeout=5,
                    )
                    if _r.status_code == 200:
                        line_display_name = _r.json().get("displayName", "")
            except Exception:
                pass
        extra_sheet = {"LINE暱稱": line_display_name} if line_display_name else None

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
                extra_sheet_fields=extra_sheet,
                off_hours_message=off_hours_message,
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
            # 資料收集完成後不再顯示快速選項
            quick_replies = None

        # 任務完成後靜默（handed_off → engine 回空字串）
        if not answer:
            logging.info(f"[LINE] Silent (handed_off) for {user_id}")
            return

        # 優先用 push（不依賴 replyToken 過期），失敗才 fallback
        push_ok = await push_line_message(user_id, answer, access_token=line_token, quick_replies=quick_replies)
        if push_ok != 200:
            reply_token = buf.get("reply_token", "")
            if reply_token:
                await reply_line_message(reply_token, answer, access_token=line_token, quick_replies=quick_replies)
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
                await reply_line_message(reply_token, welcome, access_token=line_token, quick_replies=bot.get("quick_replies") or None)
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
            await reply_line_message(reply_token, welcome, access_token=line_token, quick_replies=bot.get("quick_replies") or None)
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
# 設定歷史紀錄
# ──────────────────────────────────────

@app.get("/bots/{bot_id}/settings-history")
async def get_settings_history(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    """取得 Bot 設定歷史快照（最近 30 筆）"""
    user_id = get_user_id(authorization)
    bot = supabase.table("bots").select("id").eq("id", bot_id).eq("user_id", user_id).execute()
    if not bot.data:
        raise HTTPException(404, "Bot 不存在")
    rows = supabase.table("bot_settings_history") \
        .select("id, source, system_prompt, collect_fields, welcome_message, quick_replies, created_at") \
        .eq("bot_id", bot_id) \
        .order("created_at", desc=True) \
        .limit(30) \
        .execute()
    return rows.data or []


@app.post("/bots/{bot_id}/settings-history/{snapshot_id}/restore")
async def restore_settings_snapshot(
    bot_id: str,
    snapshot_id: str,
    authorization: Optional[str] = Header(None)
):
    """還原 Bot 設定到指定快照"""
    user_id = get_user_id(authorization)
    bot = supabase.table("bots").select("id").eq("id", bot_id).eq("user_id", user_id).execute()
    if not bot.data:
        raise HTTPException(404, "Bot 不存在")

    snap = supabase.table("bot_settings_history") \
        .select("*").eq("id", snapshot_id).eq("bot_id", bot_id).execute()
    if not snap.data:
        raise HTTPException(404, "快照不存在")

    # 還原前先存一份當前狀態
    from app.assistant.engine import _save_snapshot
    _save_snapshot(bot_id, source="restore")

    s = snap.data[0]
    supabase.table("bots").update({
        "system_prompt":   s.get("system_prompt") or "",
        "collect_fields":  s.get("collect_fields") or [],
        "welcome_message": s.get("welcome_message") or "",
        "quick_replies":   s.get("quick_replies") or [],
    }).eq("id", bot_id).execute()
    return {"ok": True, "restored_at": s["created_at"]}


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

# ── 通用 Instagram Webhook（Meta App 層級，不帶 bot_id）──
IG_VERIFY_TOKEN = os.getenv("IG_VERIFY_TOKEN", "ldh_verify_token")

@app.get("/instagram/webhook")
async def instagram_webhook_verify_global(
    hub_mode: Optional[str]         = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str]    = Query(None, alias="hub.challenge"),
):
    """Meta App 層級 Webhook 驗證"""
    if hub_mode == "subscribe" and hub_verify_token == IG_VERIFY_TOKEN:
        logging.info("[Instagram] Global webhook verified")
        return PlainTextResponse(hub_challenge)
    raise HTTPException(403, "驗證失敗：Verify Token 不符")


@app.post("/instagram/webhook")
async def instagram_webhook_global(request: Request):
    """接收 Meta App 層級 Instagram Webhook，依 instagram_account_id 路由到對應 bot"""
    data = await request.json()

    # ── DEBUG：完整記錄收到的 payload ──
    import json as _json
    logging.info(f"[Instagram] Webhook received: object={data.get('object')}, raw={_json.dumps(data)[:500]}")

    if data.get("object") not in ("instagram", "page"):
        logging.warning(f"[Instagram] Unknown object type: {data.get('object')}")
        return {"status": "ignored"}

    for entry in data.get("entry", []):
        account_id = str(entry.get("id", ""))
        logging.info(f"[Instagram] Entry id={account_id}, keys={list(entry.keys())}")
        if not account_id:
            continue

        # 依 instagram_account_id 找 bot（精確匹配）
        rows = supabase.table("bots").select("id").eq("instagram_account_id", account_id).execute()
        if not rows.data:
            # Fallback: 嘗試用 facebook_page_id 欄位比對（page 物件情況）
            rows = supabase.table("bots").select("id").eq("facebook_page_id", account_id).execute()
        if not rows.data:
            logging.warning(f"[Instagram] No bot found for account_id={account_id}, payload_object={data.get('object')}")
            continue
        bot_id = rows.data[0]["id"]
        logging.info(f"[Instagram] Routed to bot={bot_id[:8]} for account_id={account_id}")

        # ── DM 事件 ──
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            msg       = event.get("message", {})
            text      = msg.get("text", "").strip()
            if not text or msg.get("is_echo"):
                continue
            logging.info(f"[Instagram] DM bot={bot_id[:8]} sender={sender_id}")
            asyncio.create_task(_process_instagram_message(bot_id, sender_id, text))

        # ── 留言事件 ──
        for change in entry.get("changes", []):
            if change.get("field") != "feed":
                continue
            value = change.get("value", {})
            if value.get("item") != "comment" or value.get("verb") not in ("add",):
                continue
            comment_id   = value.get("comment_id") or value.get("id")
            text         = value.get("message", "").strip()
            commenter_id = value.get("from", {}).get("id", "")
            if not comment_id or not text:
                continue
            logging.info(f"[Instagram] Comment bot={bot_id[:8]} comment={comment_id}")
            asyncio.create_task(_process_instagram_comment(bot_id, comment_id, commenter_id, text))

    return {"status": "ok"}


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
    """接收 Instagram DM 與貼文留言，呼叫 AI 回覆"""
    data = await request.json()

    # Meta 驗證 ping
    if data.get("object") not in ("instagram", "page"):
        return {"status": "ignored"}

    for entry in data.get("entry", []):
        # ── DM 事件（messaging）──
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            msg       = event.get("message", {})
            text      = msg.get("text", "").strip()

            # 忽略：echo（自己發的訊息）、無文字
            if not text or msg.get("is_echo"):
                continue

            logging.info(f"[Instagram] DM bot={bot_id[:8]} sender={sender_id} msg={text[:50]}")
            asyncio.create_task(_process_instagram_message(bot_id, sender_id, text))

        # ── 留言事件（changes/feed）──
        for change in entry.get("changes", []):
            if change.get("field") != "feed":
                continue
            value = change.get("value", {})
            # 只處理新增留言（verb=add）、item=comment
            if value.get("item") != "comment" or value.get("verb") not in ("add",):
                continue
            comment_id   = value.get("comment_id") or value.get("id")
            text         = value.get("message", "").strip()
            commenter_id = value.get("from", {}).get("id", "")
            if not comment_id or not text:
                continue
            logging.info(f"[Instagram] Comment bot={bot_id[:8]} comment={comment_id} msg={text[:50]}")
            asyncio.create_task(_process_instagram_comment(bot_id, comment_id, commenter_id, text))

    return {"status": "ok"}


async def _process_instagram_message(bot_id: str, sender_id: str, text: str):
    """非同步處理 Instagram 訊息"""
    try:
        # 訂閱檢查
        allowed, _ = check_message_allowed(bot_id)
        if not allowed:
            return

        bot = _get_bot_config(bot_id)
        page_token  = bot.get("instagram_page_token")
        ig_acct_id  = bot.get("instagram_account_id") or None
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
        status = await send_instagram_message(sender_id, answer, page_token, ig_account_id=ig_acct_id)
        logging.info(f"[Instagram] Sent reply to {sender_id}, status={status}")

        supabase.table("conversations").insert({
            "bot_id": bot_id, "question": text, "answer": answer
        }).execute()

    except Exception as e:
        logging.error(f"[Instagram] process error: {e}")


async def _process_instagram_comment(bot_id: str, comment_id: str, commenter_id: str, text: str):
    """非同步處理 Instagram 貼文留言，回覆 AI 答案"""
    try:
        bot = _get_bot_config(bot_id)
        page_token = bot.get("instagram_page_token")
        if not page_token:
            logging.warning(f"[Instagram] bot {bot_id[:8]} has no page_token, skipping comment")
            return

        bot_name         = bot.get("name", "AI 助理")
        api_key          = bot.get("anthropic_api_key")
        sheet_id         = bot.get("sheet_id")
        collect_fields   = bot.get("collect_fields") or []
        system_prompt    = bot.get("system_prompt") or None
        calendar_id      = bot.get("calendar_id") or None
        slot_duration    = bot.get("slot_duration_minutes") or 60
        business_hours   = bot.get("business_hours") or None
        keyword_triggers = bot.get("keyword_triggers") or None
        # 用 commenter_id 保持對話記憶（每位留言者獨立 session）
        session_id       = f"ig_cmt_{bot_id}_{commenter_id}"

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

        from app.instagram.webhook import reply_instagram_comment
        status = await reply_instagram_comment(comment_id, answer, page_token)
        logging.info(f"[Instagram] Replied to comment {comment_id}, status={status}")

        supabase.table("conversations").insert({
            "bot_id": bot_id, "question": text, "answer": answer
        }).execute()

    except Exception as e:
        logging.error(f"[Instagram] comment process error: {e}")


# ──────────────────────────────────────
# 藍新金流（NewebPay）付費整合
# ──────────────────────────────────────

import random
import string

from app.config import (
    NEWEBPAY_MERCHANT_ID, NEWEBPAY_HASH_KEY, NEWEBPAY_HASH_IV, NEWEBPAY_SANDBOX,
    PRICE_BOT_MONTHLY, PRICE_BOT_ANNUAL,
    PRICE_BUSINESS_MONTHLY, PRICE_BUSINESS_ANNUAL,
)
from app.newebpay.payment import build_checkout_params, parse_notify

def _plan_to_slots(plan: str) -> int:
    return 10 if plan == "business" else 1

def _plan_to_amount(plan: str, annual: bool) -> int:
    if plan == "business":
        return PRICE_BUSINESS_ANNUAL if annual else PRICE_BUSINESS_MONTHLY
    return PRICE_BOT_ANNUAL if annual else PRICE_BOT_MONTHLY

def _plan_to_desc(plan: str, annual: bool) -> str:
    cycle = "年付" if annual else "月付"
    name  = "商業版" if plan == "business" else "Bot 訂閱"
    return f"攬得回 {name}（{cycle}）"

def _make_order_no(user_id: str) -> str:
    """產生藍新訂單號（max 20 chars）：NP + 10位時間戳 + 8位user前綴"""
    ts     = str(int(time.time()))
    prefix = user_id.replace("-", "")[:8]
    return f"NP{ts}{prefix}"[:20]


class CheckoutRequest(BaseModel):
    plan: str = "bot"               # "bot" | "business"
    billing_cycle: str = "monthly"  # "monthly" | "annual"


@app.post("/stripe/checkout")
async def create_checkout(
    body: CheckoutRequest,
    authorization: Optional[str] = Header(None),
):
    """建立藍新金流付款參數（前端表單 POST）"""
    user_id = get_user_id(authorization)
    annual  = body.billing_cycle == "annual"
    plan    = body.plan if body.plan in ("bot", "business") else "bot"

    # 取用戶 email
    try:
        user_info = supabase.auth.admin.get_user_by_id(user_id)
        email = user_info.user.email if user_info.user else ""
    except Exception:
        email = ""

    amount   = _plan_to_amount(plan, annual)
    order_no = _make_order_no(user_id)
    item_desc = _plan_to_desc(plan, annual)
    slots    = _plan_to_slots(plan)

    # 暫存訂單（等 webhook 回來時找回 user_id + plan）
    renews_at = (
        (datetime.utcnow() + timedelta(days=365)).isoformat()
        if annual else
        (datetime.utcnow() + timedelta(days=31)).isoformat()
    )
    supabase.table("orders").upsert({
        "id":           order_no,
        "user_id":      user_id,
        "plan":         plan,
        "billing_cycle": body.billing_cycle,
        "amount":       amount,
        "slots":        slots,
        "renews_at":    renews_at,
        "status":       "pending",
        "created_at":   datetime.utcnow().isoformat(),
    }, on_conflict="id").execute()

    params = build_checkout_params(
        merchant_id = NEWEBPAY_MERCHANT_ID,
        hash_key    = NEWEBPAY_HASH_KEY,
        hash_iv     = NEWEBPAY_HASH_IV,
        order_no    = order_no,
        amount      = amount,
        item_desc   = item_desc,
        email       = email,
        return_url  = "https://landehui.online/dashboard?payment=success",
        notify_url  = "https://api.landehui.online/newebpay/webhook",
        sandbox     = NEWEBPAY_SANDBOX,
    )
    return params


@app.post("/newebpay/webhook")
async def newebpay_webhook(request: Request):
    """藍新金流付款通知 Webhook"""
    form   = await request.form()
    result = parse_notify(dict(form), NEWEBPAY_HASH_KEY, NEWEBPAY_HASH_IV)

    if not result:
        logging.warning("[NewebPay] Webhook ignored (invalid / non-success)")
        return {"status": "ignored"}

    order_no = result.get("MerOrderNo", "")
    logging.info(f"[NewebPay] Payment success: order={order_no}")

    # 查訂單 → 找回 user_id / plan / slots
    row = supabase.table("orders").select("*").eq("id", order_no).execute()
    if not row.data:
        logging.error(f"[NewebPay] Order not found: {order_no}")
        return {"status": "order_not_found"}

    order    = row.data[0]
    user_id  = order["user_id"]
    slots    = order.get("slots", 1)
    renews_at = order.get("renews_at")

    # 更新訂單狀態
    supabase.table("orders").update({"status": "paid"}).eq("id", order_no).execute()

    # 寫入 bot_subscriptions（訂閱生效）
    supabase.table("bot_subscriptions").upsert({
        "id":         order_no,
        "user_id":    user_id,
        "status":     "active",
        "slots":      slots,
        "renews_at":  renews_at,
        "created_at": datetime.utcnow().isoformat(),
    }, on_conflict="id").execute()

    logging.info(f"[NewebPay] Sub activated: user={user_id[:8]} slots={slots}")
    return {"status": "ok"}


@app.get("/me/subscription")
async def get_subscription(authorization: Optional[str] = Header(None)):
    """取得目前用戶的訂閱狀態（per-bot 模型）"""
    user_id   = get_user_id(authorization)
    slots     = get_bot_slots(user_id)
    bots_used = (supabase.table("bots").select("id", count="exact").eq("user_id", user_id).execute().count or 0)
    return {
        "plan":      "paid" if slots > 0 else "free",
        "bot_slots": slots,
        "max_bots":  1 + slots,
        "bots_used": bots_used,
        "status":    "active",
    }


@app.get("/me/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    """會員中心：帳號資訊 + 訂閱狀態"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未授權")
    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    user_id    = payload["user_id"]
    email      = payload.get("email", "")
    created_at = payload.get("created_at") or None

    # 訂閱狀態
    slots     = get_bot_slots(user_id)
    bots_used = (supabase.table("bots").select("id", count="exact").eq("user_id", user_id).execute().count or 0)

    # 最近一筆有效訂閱的到期日
    renews_at = None
    try:
        sub_row = supabase.table("bot_subscriptions").select("renews_at").eq("user_id", user_id).eq("status", "active").order("renews_at", desc=True).limit(1).execute()
        if sub_row.data:
            renews_at = sub_row.data[0].get("renews_at")
    except Exception:
        pass

    return {
        "email":      email,
        "created_at": created_at,
        "plan":       "paid" if slots > 0 else "free",
        "bot_slots":  slots,
        "bots_used":  bots_used,
        "renews_at":  renews_at,
    }


@app.get("/me/orders")
async def get_orders(authorization: Optional[str] = Header(None)):
    """會員中心：付款紀錄"""
    user_id = get_user_id(authorization)
    try:
        rows = supabase.table("orders").select("id, plan, billing_cycle, amount, status, created_at").eq("user_id", user_id).eq("status", "paid").order("created_at", desc=True).limit(20).execute()
        return rows.data or []
    except Exception:
        return []


class ChangePasswordRequest(BaseModel):
    new_password: str

@app.post("/me/change-password")
async def change_password(
    body: ChangePasswordRequest,
    authorization: Optional[str] = Header(None),
):
    """會員中心：修改密碼"""
    user_id = get_user_id(authorization)
    if len(body.new_password) < 8:
        raise HTTPException(400, "密碼至少需要 8 個字元")
    try:
        supabase.auth.admin.update_user_by_id(user_id, {"password": body.new_password})
        return {"message": "密碼已更新"}
    except Exception as e:
        raise HTTPException(500, f"更新密碼失敗：{str(e)}")


@app.get("/")
def root():
    return {"status": "AI Chatbot SaaS running 🔥"}


# ──────────────────────────────────────
# Admin
# ──────────────────────────────────────

ADMIN_EMAIL = "youfanliao444@gmail.com"

def require_admin(authorization: str = None) -> str:
    user_id = get_user_id(authorization)
    user_info = supabase.auth.admin.get_user_by_id(user_id)
    if not user_info.user or user_info.user.email != ADMIN_EMAIL:
        raise HTTPException(403, "無權限")
    return user_id


@app.get("/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    users = supabase.auth.admin.list_users()
    total_users = len(users) if isinstance(users, list) else 0
    bots = supabase.table("bots").select("id", count="exact").execute()

    # 付費用戶 = 有至少一筆 active bot_subscriptions 的用戶
    try:
        bot_subs = supabase.table("bot_subscriptions").select("user_id").eq("status", "active").execute()
        paid_user_ids = set(r["user_id"] for r in (bot_subs.data or []))
        paid_users = len(paid_user_ids)
        all_subs = supabase.table("bot_subscriptions").select("slots").eq("status", "active").execute()
        total_slots = sum(r.get("slots", 1) for r in (all_subs.data or []))
    except Exception:
        paid_users = 0
        total_slots = 0

    return {
        "total_users": total_users,
        "total_bots":  bots.count or 0,
        "paid_users":  paid_users,
        "total_slots": total_slots,
    }


@app.get("/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    users = supabase.auth.admin.list_users()
    user_list = users if isinstance(users, list) else []

    # bot_subscriptions: 每個用戶的 active slots 總和
    slots_map: dict = {}
    renews_map: dict = {}
    try:
        subs_rows = supabase.table("bot_subscriptions").select("user_id, slots, status, renews_at").eq("status", "active").execute()
        for r in (subs_rows.data or []):
            uid = r["user_id"]
            slots_map[uid] = slots_map.get(uid, 0) + r.get("slots", 1)
            if not renews_map.get(uid):
                renews_map[uid] = r.get("renews_at")
    except Exception:
        pass

    bots_rows = supabase.table("bots").select("user_id").execute()
    bot_count: dict = {}
    for b in (bots_rows.data or []):
        uid = b["user_id"]
        bot_count[uid] = bot_count.get(uid, 0) + 1

    result = []
    for u in user_list:
        uid = u.id
        slots = slots_map.get(uid, 0)
        result.append({
            "user_id":    uid,
            "email":      u.email,
            "created_at": str(u.created_at),
            "bot_slots":  slots,
            "max_bots":   1 + slots,
            "bots_used":  bot_count.get(uid, 0),
            "renews_at":  renews_map.get(uid),
        })

    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


class AdminSlotsUpdate(BaseModel):
    slots: int  # 0 = free only, 1 = 1 paid bot, 10 = business


@app.put("/admin/users/{target_user_id}/slots")
async def admin_set_slots(
    target_user_id: str,
    body: AdminSlotsUpdate,
    authorization: Optional[str] = Header(None),
):
    require_admin(authorization)
    try:
        supabase.table("bot_subscriptions").delete().eq("id", f"admin_{target_user_id}").execute()
        if body.slots > 0:
            supabase.table("bot_subscriptions").upsert({
                "id":       f"admin_{target_user_id}",
                "user_id":  target_user_id,
                "status":   "active",
                "slots":    body.slots,
            }).execute()
    except Exception as e:
        raise HTTPException(500, f"DB 寫入失敗：{str(e)}")
    return {"ok": True, "user_id": target_user_id, "slots": body.slots}
