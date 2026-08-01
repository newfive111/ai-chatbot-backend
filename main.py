from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
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
from app.line.webhook import verify_line_signature, reply_line_message, push_line_message, download_line_content
from app.auth.utils import create_token, decode_token, generate_bot_id
from app.config import (
    SUPABASE_URL, SUPABASE_KEY,
    LINE_LOGIN_CHANNEL_ID, LINE_LOGIN_CHANNEL_SECRET,
    FRONTEND_BASE_URL, BACKEND_BASE_URL,
    ADMIN_LINE_CHANNEL_SECRET, ADMIN_LINE_CHANNEL_ACCESS_TOKEN,
)
from supabase import create_client

app = FastAPI(title="AI Chatbot SaaS API")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# 專用 admin client，永遠不呼叫 sign_in，避免 user session 污染 auth.admin.* 呼叫
_admin = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.on_event("startup")
async def _startup():
    load_muted_users()
    asyncio.create_task(_expiry_scheduler_loop())


@app.get("/health")
async def health():
    return {"status": "ok"}

# ──────────────────────────────────────
# LINE Bot 狀態管理（in-memory）
# ──────────────────────────────────────

# 靜音名單：key = "{bot_id}:{line_user_id}"，記憶體快取（webhook 高頻查詢用）
# 持久化存於 Supabase chat_mutes 表，重新部署時由 load_muted_users() 重新載入
_muted_line_users: Set[str] = set()


def _mute_key(bot_id: str, line_user_id: str) -> str:
    return f"{bot_id}:{line_user_id}"


def load_muted_users():
    """啟動時把 DB 的靜音名單載入記憶體快取。"""
    try:
        rows = supabase.table("chat_mutes").select("bot_id, line_user_id").execute()
        for r in rows.data or []:
            _muted_line_users.add(_mute_key(r["bot_id"], r["line_user_id"]))
        logging.info(f"[MUTE] Loaded {len(_muted_line_users)} muted chats from DB")
    except Exception as e:
        logging.warning(f"[MUTE] load_muted_users failed: {e}")


def add_mute(bot_id: str, line_user_id: str, muted_by: Optional[str] = None):
    """靜音某聊天室（AI 停止回覆），同步寫入記憶體快取 + DB。"""
    _muted_line_users.add(_mute_key(bot_id, line_user_id))
    try:
        supabase.table("chat_mutes").upsert(
            {"bot_id": bot_id, "line_user_id": line_user_id, "muted_by": muted_by},
            on_conflict="bot_id,line_user_id",
        ).execute()
    except Exception as e:
        logging.warning(f"[MUTE] add_mute DB write failed: {e}")


def remove_mute(bot_id: str, line_user_id: str):
    """取消靜音（AI 恢復回覆），同步清除記憶體快取 + DB。"""
    _muted_line_users.discard(_mute_key(bot_id, line_user_id))
    try:
        supabase.table("chat_mutes").delete() \
            .eq("bot_id", bot_id).eq("line_user_id", line_user_id).execute()
    except Exception as e:
        logging.warning(f"[MUTE] remove_mute DB write failed: {e}")


# LINE 暱稱快取：key = "{bot_id}:{line_user_id}" -> displayName（記憶體，收訊息時填入）
_line_profile_cache: Dict[str, str] = {}


async def fetch_line_display_name(bot_id: str, user_id: str, line_token: str) -> str:
    """抓 LINE 暱稱（優先讀快取），失敗回空字串。"""
    key = _mute_key(bot_id, user_id)
    if key in _line_profile_cache:
        return _line_profile_cache[key]
    if not line_token:
        return ""
    try:
        async with httpx.AsyncClient() as _hc:
            _r = await _hc.get(
                f"https://api.line.me/v2/bot/profile/{user_id}",
                headers={"Authorization": f"Bearer {line_token}"},
                timeout=5,
            )
            if _r.status_code == 200:
                name = _r.json().get("displayName", "") or ""
                if name:
                    _line_profile_cache[key] = name
                return name
    except Exception:
        pass
    return ""

# 防抖緩衝區：key = "{bot_id}:{line_user_id}"
# 值 = {"msgs": [], "reply_token": str, "task": asyncio.Task}
_line_buffers: Dict[str, dict] = {}

# 客戶只回貼圖時，餵給 AI 的系統事件訊息（讓 AI 依上下文決定追問或略過）
_STICKER_SKIP_TOKEN = "SKIP_NO_REPLY"
_STICKER_EVENT_MSG = (
    "（系統事件：客戶只傳了一個貼圖，沒有任何文字。）"
    "請依照對話上下文判斷：若你上一則訊息問了問題、而客戶到目前為止還沒回答，"
    "請用親切自然的語氣把那個問題再問一次；"
    f"若客戶先前已經回答過你的問題、這個貼圖只是表達情緒或打招呼，"
    f"請「只」輸出 {_STICKER_SKIP_TOKEN} 這幾個字、不要有其他任何內容。"
)

# 垃圾訊息關鍵字（通用，各 bot 可擴充）
_SPAM_KEYWORDS = ["資金週轉", "債務整合", "房屋二胎", "汽機車二貸", "若需要以上方案", "娛樂城", "博弈"]

DEBOUNCE_SECONDS = 15  # 防抖等待時間（LINE replyToken 60秒過期，15s 緩衝足夠安全）

# 允許跨域（前端呼叫用）
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://www.landehui.online,https://landehui.online").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
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
    try:
        result = _admin.auth.admin.create_user({
            "email": body.email,
            "password": body.password,
            "email_confirm": True
        })
    except Exception as e:
        err_str = str(e).lower()
        if "already" in err_str or "exist" in err_str or "duplicate" in err_str:
            raise HTTPException(400, "此 Email 已被註冊，請直接登入或換一個 Email")
        if "password" in err_str:
            raise HTTPException(400, "密碼強度不足，請使用至少 8 位包含英數字的密碼")
        raise HTTPException(400, f"註冊失敗：{str(e)}")
    if result.user:
        created_at_str = result.user.created_at.isoformat() if result.user.created_at else ""
        try:
            ensure_app_user(supabase_uid=result.user.id, email=result.user.email or "")
        except Exception as e:
            logging.warning(f"[register] ensure_app_user failed: {e}")
        token = create_token(result.user.id, email=result.user.email or "", created_at=created_at_str)
        return {"token": token, "user_id": result.user.id}
    raise HTTPException(400, "註冊失敗，請稍後再試")

@app.post("/auth/login")
async def login(body: LoginRequest):
    # 每次登入用獨立 client，避免 user session 寫入共用 supabase client
    _login_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    try:
        result = _login_client.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password
        })
        if result.user:
            created_at_str = result.user.created_at.isoformat() if result.user.created_at else ""
            try:
                ensure_app_user(supabase_uid=result.user.id, email=result.user.email or "")
            except Exception as e:
                logging.warning(f"[login] ensure_app_user failed: {e}")
            token = create_token(result.user.id, email=result.user.email or "", created_at=created_at_str)
            return {"token": token, "user_id": result.user.id}
    except Exception as e:
        raise HTTPException(401, f"帳號或密碼錯誤: {str(e)}")
    raise HTTPException(401, "帳號或密碼錯誤")


# ──────────────────────────────────────
# LINE 快速登入 (LINE Login OAuth 2.1)
# ──────────────────────────────────────

# state 暫存（CSRF 防護），value = 過期 timestamp，10 分鐘 TTL
_line_login_states: Dict[str, float] = {}


def _find_or_create_supabase_user(email: str, display_name: str):
    """依 email 找出既有 Supabase auth 使用者；找不到就建立一個（LINE 使用者用隨機密碼）。"""
    email_l = email.lower()
    try:
        users = _admin.auth.admin.list_users()
        for u in (users or []):
            if (getattr(u, "email", "") or "").lower() == email_l:
                return u
    except Exception as e:
        logging.warning(f"[LINE login] list_users failed: {e}")

    import secrets
    res = _admin.auth.admin.create_user({
        "email": email,
        "password": secrets.token_urlsafe(24),
        "email_confirm": True,
        "user_metadata": {"display_name": display_name, "provider": "line"},
    })
    return res.user


@app.get("/auth/line/login")
async def line_login_start():
    if not LINE_LOGIN_CHANNEL_ID or not LINE_LOGIN_CHANNEL_SECRET:
        raise HTTPException(503, "LINE 登入尚未設定")

    now = time.time()
    for s, exp in list(_line_login_states.items()):
        if exp < now:
            _line_login_states.pop(s, None)

    state = uuid.uuid4().hex
    _line_login_states[state] = now + 600

    from urllib.parse import urlencode
    params = {
        "response_type": "code",
        "client_id": LINE_LOGIN_CHANNEL_ID,
        "redirect_uri": f"{BACKEND_BASE_URL}/auth/line/callback",
        "state": state,
        # 只要 profile + openid；不要 email（LINE 需另外申請 Email 權限，後端會自動合成 email）
        "scope": "profile openid",
    }
    auth_url = "https://access.line.me/oauth2/v2.1/authorize?" + urlencode(params)
    return {"auth_url": auth_url}


@app.get("/auth/line/callback")
async def line_login_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    front = FRONTEND_BASE_URL.rstrip("/")
    if error or not code or not state:
        return RedirectResponse(f"{front}/login?line_error=1")

    exp = _line_login_states.pop(state, None)
    if not exp or exp < time.time():
        return RedirectResponse(f"{front}/login?line_error=state")

    redirect_uri = f"{BACKEND_BASE_URL}/auth/line/callback"

    # 1) 用 code 換 token
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            token_res = await client.post(
                "https://api.line.me/oauth2/v2.1/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": LINE_LOGIN_CHANNEL_ID,
                    "client_secret": LINE_LOGIN_CHANNEL_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        token_res.raise_for_status()
        tok = token_res.json()
    except Exception as e:
        logging.error(f"[LINE login] token exchange failed: {e}")
        return RedirectResponse(f"{front}/login?line_error=token")

    id_token = tok.get("id_token")
    access_token = tok.get("access_token")
    line_sub = None
    email = None
    display_name = "LINE 使用者"

    # 2) 解 id_token（channel secret 簽的 HS256）
    if id_token:
        try:
            import jwt as _jwt
            claims = _jwt.decode(
                id_token, LINE_LOGIN_CHANNEL_SECRET,
                algorithms=["HS256"], audience=LINE_LOGIN_CHANNEL_ID,
            )
            line_sub = claims.get("sub")
            email = claims.get("email")
            display_name = claims.get("name") or display_name
        except Exception as e:
            logging.warning(f"[LINE login] id_token decode failed: {e}")

    # 3) fallback：用 access_token 取 profile
    if not line_sub and access_token:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                p = await client.get(
                    "https://api.line.me/v2/profile",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            p.raise_for_status()
            pj = p.json()
            line_sub = pj.get("userId")
            display_name = pj.get("displayName") or display_name
        except Exception as e:
            logging.error(f"[LINE login] profile fetch failed: {e}")

    if not line_sub:
        return RedirectResponse(f"{front}/login?line_error=profile")

    # LINE 未提供 email（未開通 email 權限）→ 用穩定的合成 email
    if not email:
        email = f"line_{line_sub}@line.landehui.online"

    # 4) 找出/建立 Supabase 使用者，發自家 token
    try:
        user = _find_or_create_supabase_user(email, display_name)
    except Exception as e:
        logging.error(f"[LINE login] user provisioning failed: {e}")
        return RedirectResponse(f"{front}/login?line_error=user")

    # 建立/連結 app_user（帶上 line_user_id、顯示名稱），並確保有個人團隊
    try:
        ensure_app_user(
            supabase_uid=user.id, email=user.email or email,
            line_user_id=line_sub, display_name=display_name,
        )
    except Exception as e:
        logging.warning(f"[LINE login] ensure_app_user failed: {e}")

    created_at_str = user.created_at.isoformat() if getattr(user, "created_at", None) else ""
    token = create_token(user.id, email=user.email or email, created_at=created_at_str)
    return RedirectResponse(f"{front}/auth/line/callback?token={token}")


# ──────────────────────────────────────
# 團隊 / 成員 / 邀請
# 權限：看成員 viewer；改角色/移除/邀請 admin；owner 不可被改動
# ──────────────────────────────────────

_VALID_ASSIGN_ROLES = {"admin", "editor", "viewer"}   # 不能透過 API 直接指派 owner
_INVITE_TTL_DAYS = 7


@app.get("/orgs")
async def list_my_orgs(authorization: Optional[str] = Header(None)):
    app_user = get_app_user(authorization)
    # 一次撈出所有 membership（含 role），省掉每個團隊各查一次角色的 N+1
    memberships = supabase.table("memberships").select("org_id, role") \
        .eq("user_id", app_user["id"]).execute().data or []
    if not memberships:
        return []
    role_by_org = {m["org_id"]: m["role"] for m in memberships}
    org_ids = list(role_by_org.keys())
    rows = supabase.table("organizations").select("id, name, owner_id").in_("id", org_ids).execute()
    result = []
    for o in (rows.data or []):
        result.append({
            "id": o["id"],
            "name": o["name"],
            "is_owner": o["owner_id"] == app_user["id"],
            "role": role_by_org.get(o["id"]),
        })
    return result


@app.get("/orgs/{org_id}/members")
async def list_members(org_id: str, authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="viewer")
    rows = supabase.table("memberships").select("id, user_id, role, created_at, note") \
        .eq("org_id", org_id).order("created_at").execute()
    members = rows.data or []
    user_ids = [m["user_id"] for m in members]
    users_map = {}
    if user_ids:
        urows = supabase.table("app_users").select("id, display_name, email, picture_url") \
            .in_("id", user_ids).execute()
        users_map = {u["id"]: u for u in (urows.data or [])}
    for m in members:
        u = users_map.get(m["user_id"], {})
        m["display_name"] = u.get("display_name") or u.get("email") or "使用者"
        m["email"] = u.get("email")
        m["picture_url"] = u.get("picture_url")
        m["note"] = m.get("note") or ""
    return members


class UpdateMemberRequest(BaseModel):
    role: str


@app.patch("/orgs/{org_id}/members/{member_user_id}")
async def update_member(org_id: str, member_user_id: str, body: UpdateMemberRequest,
                        authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    if body.role not in _VALID_ASSIGN_ROLES:
        raise HTTPException(400, "無效的角色")
    target = supabase.table("memberships").select("role") \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    if not target.data:
        raise HTTPException(404, "成員不存在")
    if target.data[0]["role"] == "owner":
        raise HTTPException(403, "無法變更擁有者的角色")
    supabase.table("memberships").update({"role": body.role}) \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    return {"ok": True}


class UpdateMemberNoteRequest(BaseModel):
    note: str = ""


@app.patch("/orgs/{org_id}/members/{member_user_id}/note")
async def update_member_note(org_id: str, member_user_id: str, body: UpdateMemberNoteRequest,
                             authorization: Optional[str] = Header(None)):
    """幫團隊成員加備注（例：這支 LINE 是誰的），存在 membership.note，只在此團隊內顯示。"""
    require_org_access(org_id, authorization, min_role="admin")
    target = supabase.table("memberships").select("id") \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    if not target.data:
        raise HTTPException(404, "成員不存在")
    note = (body.note or "")[:200]
    supabase.table("memberships").update({"note": note}) \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    return {"ok": True, "note": note}


@app.delete("/orgs/{org_id}/members/{member_user_id}")
async def remove_member(org_id: str, member_user_id: str,
                        authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    target = supabase.table("memberships").select("role") \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    if not target.data:
        raise HTTPException(404, "成員不存在")
    if target.data[0]["role"] == "owner":
        raise HTTPException(403, "無法移除團隊擁有者")
    supabase.table("memberships").delete() \
        .eq("org_id", org_id).eq("user_id", member_user_id).execute()
    return {"ok": True}


class CreateInviteRequest(BaseModel):
    role: str = "editor"


@app.post("/orgs/{org_id}/invites")
async def create_invite(org_id: str, body: CreateInviteRequest,
                        authorization: Optional[str] = Header(None)):
    ctx = require_org_access(org_id, authorization, min_role="admin")
    if body.role not in _VALID_ASSIGN_ROLES:
        raise HTTPException(400, "無效的角色")
    import secrets as _secrets
    token = _secrets.token_urlsafe(24)
    expires_at = (datetime.utcnow() + timedelta(days=_INVITE_TTL_DAYS)).isoformat() + "+00:00"
    supabase.table("invites").insert({
        "token": token,
        "org_id": org_id,
        "role": body.role,
        "created_by": ctx["app_user"]["id"],
        "expires_at": expires_at,
    }).execute()
    invite_url = f"{FRONTEND_BASE_URL.rstrip('/')}/invite/{token}"
    return {"token": token, "invite_url": invite_url, "role": body.role, "expires_at": expires_at}


@app.get("/orgs/{org_id}/invites")
async def list_invites(org_id: str, authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    rows = supabase.table("invites").select("token, role, created_at, expires_at, used_at") \
        .eq("org_id", org_id).is_("used_at", "null").order("created_at", desc=True).execute()
    return rows.data or []


@app.delete("/orgs/{org_id}/invites/{token}")
async def revoke_invite(org_id: str, token: str, authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    supabase.table("invites").delete().eq("token", token).eq("org_id", org_id).execute()
    return {"ok": True}


@app.get("/invites/{token}")
async def get_invite(token: str):
    """公開：顯示「XX 邀請你加入」用，不需登入。"""
    r = supabase.table("invites").select("token, org_id, role, expires_at, used_at").eq("token", token).execute()
    if not r.data:
        raise HTTPException(404, "邀請連結無效")
    inv = r.data[0]
    if inv.get("used_at"):
        raise HTTPException(410, "此邀請連結已被使用")
    if inv.get("expires_at") and inv["expires_at"] < datetime.utcnow().isoformat():
        raise HTTPException(410, "此邀請連結已過期")
    org = supabase.table("organizations").select("name").eq("id", inv["org_id"]).execute()
    org_name = org.data[0]["name"] if org.data else "團隊"
    return {"org_name": org_name, "role": inv["role"]}


@app.post("/invites/{token}/accept")
async def accept_invite(token: str, authorization: Optional[str] = Header(None)):
    """登入後呼叫：把自己加入該團隊。"""
    app_user = get_app_user(authorization)
    r = supabase.table("invites").select("*").eq("token", token).execute()
    if not r.data:
        raise HTTPException(404, "邀請連結無效")
    inv = r.data[0]
    if inv.get("used_at"):
        raise HTTPException(410, "此邀請連結已被使用")
    if inv.get("expires_at") and inv["expires_at"] < datetime.utcnow().isoformat():
        raise HTTPException(410, "此邀請連結已過期")
    org_id = inv["org_id"]
    existing = get_membership_role(org_id, app_user["id"])
    if existing is None:
        supabase.table("memberships").insert({
            "org_id": org_id,
            "user_id": app_user["id"],
            "role": inv["role"],
        }).execute()
    supabase.table("invites").update({"used_at": datetime.utcnow().isoformat() + "+00:00"}) \
        .eq("token", token).execute()
    return {"ok": True, "org_id": org_id}


# ── 團隊邀請碼（一組 6 碼、可重複用）：員工用 LINE 加管理助手 bot → 傳碼即加入團隊 ──
# 與個人綁定碼（line_binding_codes，一次性、綁到特定 app_user）不同：
# 這組碼綁在「團隊」上，讓多位員工各自成為獨立成員。

def _gen_team_join_code(org_id: str, role: str, created_by: str) -> str:
    import random
    supabase.table("team_join_codes").delete().eq("org_id", org_id).execute()
    for _ in range(8):
        cand = f"{random.randint(0, 999999):06d}"
        try:
            supabase.table("team_join_codes").insert({
                "code": cand, "org_id": org_id, "role": role, "created_by": created_by,
            }).execute()
            return cand
        except Exception:
            continue
    raise HTTPException(500, "產生團隊邀請碼失敗，請重試")


@app.get("/orgs/{org_id}/join-code")
async def get_team_join_code(org_id: str, authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    r = supabase.table("team_join_codes").select("code, role") \
        .eq("org_id", org_id).limit(1).execute()
    return r.data[0] if r.data else {"code": None}


class CreateJoinCodeRequest(BaseModel):
    role: str = "editor"


@app.post("/orgs/{org_id}/join-code")
async def create_team_join_code(org_id: str, body: CreateJoinCodeRequest,
                                authorization: Optional[str] = Header(None)):
    ctx = require_org_access(org_id, authorization, min_role="admin")
    if body.role not in _VALID_ASSIGN_ROLES:
        raise HTTPException(400, "無效的角色")
    code = _gen_team_join_code(org_id, body.role, ctx["app_user"]["id"])
    return {"code": code, "role": body.role}


@app.delete("/orgs/{org_id}/join-code")
async def revoke_team_join_code(org_id: str, authorization: Optional[str] = Header(None)):
    require_org_access(org_id, authorization, min_role="admin")
    supabase.table("team_join_codes").delete().eq("org_id", org_id).execute()
    return {"ok": True}


# ──────────────────────────────────────
# Bot 管理
# ──────────────────────────────────────

def get_user_id(authorization: str = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未授權")
    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    return payload["user_id"]

def _sub_is_valid(sub: dict) -> bool:
    """訂閱是否有效：status=active 且（沒設到期日 或 到期日還沒過）。
    renews_at 為空 → 視為永久有效（保護沒設到期日的舊訂閱，不誤殺）。"""
    if sub.get("status") != "active":
        return False
    renews_at = sub.get("renews_at")
    if not renews_at:
        return True
    try:
        exp = datetime.fromisoformat(str(renews_at).replace("Z", "+00:00"))
        if exp.tzinfo:
            exp = exp.replace(tzinfo=None)
        return datetime.utcnow() <= exp
    except Exception:
        # 日期格式怪 → 保守當作有效，避免誤停用
        return True


def get_bot_slots(user_id: str) -> int:
    """回傳該用戶目前有效的付費 Bot 名額總數（商業版=10, 單Bot=1）。
    已過期（renews_at 已過）的訂閱不計入 → 到期自動停用。"""
    try:
        rows = supabase.table("bot_subscriptions") \
            .select("slots, status, renews_at") \
            .eq("user_id", user_id) \
            .eq("status", "active") \
            .execute()
        return sum(r.get("slots", 1) for r in (rows.data or []) if _sub_is_valid(r))
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


# ──────────────────────────────────────
# 多租戶授權（團隊 / 成員 / 角色）
# ──────────────────────────────────────

_ROLE_RANK = {"viewer": 1, "editor": 2, "admin": 3, "owner": 4}


def ensure_app_user(supabase_uid: str = None, email: str = "",
                    line_user_id: str = None, display_name: str = "",
                    picture_url: str = "") -> dict:
    """確保 app_users 有這位使用者，沒有就建立（冪等）。
    注意：不再自動建立個人團隊。個人團隊改為「真的需要時」才建（見 ensure_personal_org），
    避免只是透過邀請加入別人團隊的成員（尤其 LINE 登入員工）也被開一堆空團隊。"""
    row = None
    if line_user_id:
        r = supabase.table("app_users").select("*").eq("line_user_id", line_user_id).execute()
        row = r.data[0] if r.data else None
    if row is None and supabase_uid:
        r = supabase.table("app_users").select("*").eq("supabase_uid", supabase_uid).execute()
        row = r.data[0] if r.data else None
    if row is None and email:
        r = supabase.table("app_users").select("*").eq("email", email).execute()
        row = r.data[0] if r.data else None

    if row is None:
        insert_data = {
            "email": email or None,
            "supabase_uid": supabase_uid or None,
            "line_user_id": line_user_id or None,
            "display_name": display_name or (email.split("@")[0] if email else "使用者"),
            "picture_url": picture_url or None,
        }
        row = supabase.table("app_users").insert(insert_data).execute().data[0]
    else:
        patch = {}
        if line_user_id and not row.get("line_user_id"):
            patch["line_user_id"] = line_user_id
        if supabase_uid and not row.get("supabase_uid"):
            patch["supabase_uid"] = supabase_uid
        if email and not row.get("email"):
            patch["email"] = email
        if patch:
            supabase.table("app_users").update(patch).eq("id", row["id"]).execute()
            row.update(patch)

    return row


def ensure_personal_org(app_user: dict) -> str:
    """確保這位使用者有自己的個人團隊（owner），沒有就建立。回傳 org_id。
    只在使用者真的需要當 owner 時呼叫（例如建立第一個 bot），
    純粹被邀請加入別人團隊的成員不會被開個人團隊。"""
    owned = supabase.table("organizations").select("id") \
        .eq("owner_id", app_user["id"]).limit(1).execute()
    if owned.data:
        return owned.data[0]["id"]
    org_name = app_user.get("display_name") or "我的團隊"
    new_org = supabase.table("organizations").insert(
        {"name": org_name, "owner_id": app_user["id"]}
    ).execute().data[0]
    supabase.table("memberships").insert(
        {"org_id": new_org["id"], "user_id": app_user["id"], "role": "owner"}
    ).execute()
    return new_org["id"]


def get_app_user(authorization: str = None) -> dict:
    """從 JWT 解析出 app_user row（root token 帶 user_id=supabase uid，自動補建）。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未授權")
    payload = decode_token(authorization.replace("Bearer ", ""))
    app_user_id = payload.get("app_user_id")
    if app_user_id:
        r = supabase.table("app_users").select("*").eq("id", app_user_id).execute()
        if r.data:
            return r.data[0]
    legacy_uid = payload.get("user_id")
    if legacy_uid:
        return ensure_app_user(supabase_uid=legacy_uid, email=payload.get("email", ""))
    raise HTTPException(401, "使用者不存在")


def get_org(org_id: str) -> Optional[dict]:
    r = supabase.table("organizations").select("*").eq("id", org_id).execute()
    return r.data[0] if r.data else None


def get_user_org_ids(app_user_id: str) -> list:
    r = supabase.table("memberships").select("org_id").eq("user_id", app_user_id).execute()
    return [m["org_id"] for m in (r.data or [])]


def get_membership_role(org_id: str, app_user_id: str) -> Optional[str]:
    r = supabase.table("memberships").select("role") \
        .eq("org_id", org_id).eq("user_id", app_user_id).execute()
    return r.data[0]["role"] if r.data else None


def _org_owner_uid(org: dict) -> Optional[str]:
    if not org or not org.get("owner_id"):
        return None
    owner = supabase.table("app_users").select("supabase_uid").eq("id", org["owner_id"]).execute()
    return owner.data[0].get("supabase_uid") if owner.data else None


def get_org_slots(org_id: str) -> int:
    """該團隊可用的付費 bot 名額（依團隊 owner 的訂閱計算）。"""
    org = get_org(org_id)
    uid = _org_owner_uid(org)
    return get_bot_slots(uid) if uid else 0


def require_org_access(org_id: str, authorization: str = None, min_role: str = "viewer") -> dict:
    """驗證使用者對該 org 有足夠權限，回傳 {app_user, role}。非成員丟 403。"""
    app_user = get_app_user(authorization)
    role = get_membership_role(org_id, app_user["id"])
    if role is None:
        raise HTTPException(403, "無權存取此團隊")
    if _ROLE_RANK.get(role, 0) < _ROLE_RANK.get(min_role, 99):
        raise HTTPException(403, "權限不足")
    return {"app_user": app_user, "role": role}


def require_bot_access(bot_id: str, authorization: str = None, min_role: str = "viewer") -> dict:
    """驗證使用者對該 bot 有足夠權限，回傳 {app_user, bot, role, org_id}。"""
    app_user = get_app_user(authorization)
    bot = supabase.table("bots").select("id, user_id, org_id").eq("id", bot_id).execute()
    if not bot.data:
        raise HTTPException(404, "Bot 不存在")
    bot_row = bot.data[0]
    org_id = bot_row.get("org_id")
    role = get_membership_role(org_id, app_user["id"]) if org_id else None
    # 相容尚未遷移 org_id 的舊 bot：建立者本人視為 owner
    if role is None and bot_row.get("user_id") and bot_row["user_id"] == app_user.get("supabase_uid"):
        role = "owner"
    if role is None:
        raise HTTPException(403, "無權存取此 Bot")
    if _ROLE_RANK.get(role, 0) < _ROLE_RANK.get(min_role, 99):
        raise HTTPException(403, "權限不足")
    return {"app_user": app_user, "bot": bot_row, "role": role, "org_id": org_id}


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
    app_user = get_app_user(authorization)
    # 建 bot 的人 = 自己團隊的 owner；沒有個人團隊才 lazy 建立（不會動到被邀請加入的別人團隊）
    org_id = ensure_personal_org(app_user)
    user_id = app_user.get("supabase_uid") or get_user_id(authorization)

    # 限制：免費 1 個，每個付費訂閱 +1（依團隊計算）
    existing = supabase.table("bots").select("id", count="exact").eq("org_id", org_id).execute()
    current_count = existing.count or 0
    slots = get_org_slots(org_id)
    max_bots = 1 + slots  # 1 免費 + N 付費

    if current_count >= max_bots:
        raise HTTPException(403, f"已達上限（{max_bots} 個 Bot）。請至定價頁購買更多名額。")

    bot_id = generate_bot_id()
    insert_data: dict = {"id": bot_id, "user_id": user_id, "org_id": org_id, "name": name}
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
    app_user = get_app_user(authorization)
    org_ids = get_user_org_ids(app_user["id"])
    seen: dict = {}
    if org_ids:
        result = supabase.table("bots").select("*").in_("org_id", org_ids).order("created_at").execute()
        for b in (result.data or []):
            seen[b["id"]] = b
    # 相容尚未遷移 org_id 的舊 bot（建立者本人）
    my_uid = app_user.get("supabase_uid")
    if my_uid:
        legacy = supabase.table("bots").select("*").eq("user_id", my_uid).order("created_at").execute()
        for b in (legacy.data or []):
            seen.setdefault(b["id"], b)
    bots = sorted(seen.values(), key=lambda b: b.get("created_at") or "")
    # 依「該 bot 的擁有者名額」標記付費/免費
    for bot in bots:
        bot["plan"] = "paid" if is_bot_paid(bot["id"]) else "free"
    return bots

@app.get("/bots/{bot_id}")
async def get_bot(bot_id: str, authorization: Optional[str] = Header(None)):
    """取得單一 Bot 完整設定（API Key 只回傳是否已設定）"""
    access = require_bot_access(bot_id, authorization, min_role="viewer")
    result = supabase.table("bots").select("*").eq("id", bot_id).execute()
    if not result.data:
        raise HTTPException(404, "Bot 不存在")
    bot = result.data[0]
    # 不回傳明文 API Key，只告訴前端有沒有設定
    bot["has_api_key"] = bool(bot.get("anthropic_api_key"))
    bot.pop("anthropic_api_key", None)
    bot["my_role"] = access["role"]
    return bot

class GeneratePersonaRequest(BaseModel):
    description: str


_PERSONA_JSON_SPEC = """請只輸出一個 JSON 物件，不要有其他文字，格式如下：
{
  "business": "用一到兩句描述這個生意在做什麼（繁體中文）",
  "role": "從這五個擇一：customer_service（客服）/ sales（業務銷售）/ booking（預約）/ consultant（諮詢顧問）/ general（一般助理）",
  "tones": ["從這些語氣挑 1-3 個：親切、專業、簡潔、熱情活潑、正式禮貌、幽默輕鬆"],
  "highlights": "客服最該讓客戶知道的重點（營業項目、特色、常見問答方向），2-4 行，繁體中文；不確定就留空字串",
  "taboos": "這個角色絕對不該做或說的事，1-2 行，繁體中文；不確定就留空字串"
}"""


def _ai_persona_form(api_key: str, instruction: str) -> dict:
    """呼叫 Gemini 產出結構化角色填空 dict，並正規化 role / tones。"""
    from google import genai
    from google.genai import types
    import time as _time

    client = genai.Client(api_key=api_key)
    last_err = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=instruction)])],
                config=types.GenerateContentConfig(
                    max_output_tokens=1200,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                ),
            )
            parts = response.candidates[0].content.parts if response.candidates else []
            raw = "".join(p.text for p in parts if hasattr(p, "text") and not getattr(p, "thought", False))
            data = json.loads(raw)
            valid_roles = {"customer_service", "sales", "booking", "consultant", "general"}
            valid_tones = {"親切", "專業", "簡潔", "熱情活潑", "正式禮貌", "幽默輕鬆"}
            role = data.get("role") if data.get("role") in valid_roles else "customer_service"
            tones = [t for t in (data.get("tones") or []) if t in valid_tones][:3] or ["親切"]
            return {
                "business":   str(data.get("business") or "").strip(),
                "role":       role,
                "tones":      tones,
                "highlights": str(data.get("highlights") or "").strip(),
                "taboos":     str(data.get("taboos") or "").strip(),
            }
        except Exception as e:
            last_err = e
            if "503" in str(e) or "overloaded" in str(e).lower():
                _time.sleep(3 * (attempt + 1))
            else:
                raise
    raise last_err


@app.post("/bots/{bot_id}/generate-persona")
async def generate_persona(
    bot_id: str,
    body: GeneratePersonaRequest,
    authorization: Optional[str] = Header(None)
):
    """方案 2：用戶用一句話描述生意 → AI 產出結構化角色填空（給簡單模式表單）。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    desc = (body.description or "").strip()
    if not desc:
        raise HTTPException(400, "請先描述你的生意")

    bot_row = supabase.table("bots").select("anthropic_api_key, name").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key  = bot_row.data[0].get("anthropic_api_key")
    bot_name = bot_row.data[0].get("name", "Bot")
    if not api_key:
        raise HTTPException(400, "請先設定 Gemini API Key")

    instruction = f"""用戶用一句話描述他的生意，請幫他規劃這個 AI 客服 Bot（名稱：「{bot_name}」）的角色設定。

用戶描述：{desc}

{_PERSONA_JSON_SPEC}"""
    try:
        return _ai_persona_form(api_key, instruction)
    except Exception as e:
        raise HTTPException(500, f"AI 生成失敗：{str(e)[:200]}")


@app.post("/bots/{bot_id}/extract-persona")
async def extract_persona(bot_id: str, authorization: Optional[str] = Header(None)):
    """把 bot 現有的 system_prompt 反向拆解成結構化角色填空（給老 bot 切換簡單模式）。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    bot_row = supabase.table("bots").select("anthropic_api_key, name, system_prompt").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key  = bot_row.data[0].get("anthropic_api_key")
    bot_name = bot_row.data[0].get("name", "Bot")
    sp = (bot_row.data[0].get("system_prompt") or "").strip()
    if not api_key:
        raise HTTPException(400, "請先設定 Gemini API Key")
    if not sp:
        raise HTTPException(400, "這個 Bot 還沒有角色設定可以拆解")

    instruction = f"""以下是一個 AI 客服 Bot（名稱：「{bot_name}」）目前的角色設定文字。請把它拆解、歸納成結構化欄位，盡量保留原意，不要自行新增原文沒有的內容。

=== 目前的角色設定 ===
{sp[:2000]}

{_PERSONA_JSON_SPEC}"""
    try:
        return _ai_persona_form(api_key, instruction)
    except Exception as e:
        raise HTTPException(500, f"AI 拆解失敗：{str(e)[:200]}")


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
    persona_form: Optional[dict] = None       # 結構化角色填空（簡單模式來源）
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
    # 觀察模式：開啟時 AI 不回覆，只記錄客戶訊息並通知員工代回
    observe_mode: Optional[bool] = None
    # 客戶名單資料卡排版範本（{欄位} 佔位符）
    card_template: Optional[str] = None

@app.patch("/bots/{bot_id}")
async def update_bot(
    bot_id: str,
    body: UpdateBotRequest,
    authorization: Optional[str] = Header(None)
):
    require_bot_access(bot_id, authorization, min_role="editor")
    update_data = {}
    # exclude_unset=True：只處理請求中明確傳入的欄位，避免未傳的欄位被誤清空
    for k, v in body.model_dump(exclude_unset=True).items():
        if v is not None:
            update_data[k] = v
        elif k in ("collect_fields", "quick_replies", "keyword_triggers"):
            # 明確傳入空 list → 允許清空
            update_data[k] = []
        elif k in ("system_prompt", "welcome_message", "card_template"):
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
    require_bot_access(bot_id, authorization, min_role="admin")

    # 刪除關聯資料
    try:
        supabase.table("knowledge_chunks").delete().eq("bot_id", bot_id).execute()
        supabase.table("conversations").delete().eq("bot_id", bot_id).execute()
        # 刪除 bot 本身
        supabase.table("bots").delete().eq("id", bot_id).execute()
        return {"status": "deleted"}
    except Exception as e:
        logging.error(f"[Delete Bot] Failed to delete bot {bot_id}: {e}")
        raise HTTPException(500, f"刪除失敗: {str(e)}")


# ──────────────────────────────────────
# 知識庫上傳
# ──────────────────────────────────────

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

@app.post("/bots/{bot_id}/upload")
async def upload_document(
    bot_id: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    require_bot_access(bot_id, authorization, min_role="editor")
    bot_row = supabase.table("bots").select("anthropic_api_key").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key = bot_row.data[0].get("anthropic_api_key", "")
    if not api_key:
        raise HTTPException(400, "請先在 Bot 設定中填入 Gemini API Key")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"檔案過大，上限為 10 MB")

    try:
        if file.filename and file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode("utf-8")
        chunks = chunk_text(text)
        store_chunks(bot_id, chunks, api_key=api_key)
    except Exception as e:
        raise HTTPException(500, f"上傳失敗：{str(e)}")
    return {"message": f"成功上傳，共 {len(chunks)} 個知識塊"}

class FAQRequest(BaseModel):
    content: str

@app.post("/bots/{bot_id}/faq")
async def add_faq(
    bot_id: str,
    body: FAQRequest,
    authorization: Optional[str] = Header(None)
):
    require_bot_access(bot_id, authorization, min_role="editor")
    bot_row = supabase.table("bots").select("anthropic_api_key").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key = bot_row.data[0].get("anthropic_api_key", "")
    if not api_key:
        raise HTTPException(400, "請先在 Bot 設定中填入 Gemini API Key")

    chunks = chunk_text(body.content)
    try:
        store_chunks(bot_id, chunks, api_key=api_key)
    except Exception as e:
        raise HTTPException(500, f"Embedding 失敗：{str(e)}")
    return {"message": "FAQ 已加入知識庫"}

@app.get("/bots/{bot_id}/knowledge")
async def list_knowledge(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    require_bot_access(bot_id, authorization, min_role="viewer")
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
    require_bot_access(bot_id, authorization, min_role="editor")
    supabase.table("knowledge_chunks").delete().eq("bot_id", bot_id).execute()
    return {"message": "知識庫已清除"}

@app.delete("/bots/{bot_id}/knowledge/{chunk_id}")
async def delete_chunk(
    bot_id: str,
    chunk_id: str,
    authorization: Optional[str] = Header(None)
):
    require_bot_access(bot_id, authorization, min_role="editor")
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
    require_bot_access(bot_id, authorization, min_role="editor")
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
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers, off_hours_message, card_template"
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
    card_template = bot_data.get("card_template") or None

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
            card_template=card_template,
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
# LINE 連線測試 + 自動設定 Webhook（簡化綁定）
# ──────────────────────────────────────

@app.post("/bots/{bot_id}/line/verify")
async def line_verify_and_setup(bot_id: str, authorization: Optional[str] = Header(None)):
    """用已存的 token 驗證 LINE 連線 → 回傳 OA 名稱/頭像，並自動幫忙設定 Webhook。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    bot = _get_bot_config(bot_id)
    token = bot.get("line_channel_access_token")
    if not token:
        raise HTTPException(400, "請先儲存 Channel Access Token 再測試")

    headers = {"Authorization": f"Bearer {token}"}
    webhook_url = f"{BACKEND_BASE_URL}/line/webhook/{bot_id}"
    result: dict = {"ok": False, "oa_name": None, "oa_picture": None,
                    "webhook_set": False, "webhook_active": False, "warnings": []}

    async with httpx.AsyncClient(timeout=10) as client:
        # 1) 驗證 token + 取得 OA 資料
        try:
            info = await client.get("https://api.line.me/v2/bot/info", headers=headers)
        except Exception as e:
            raise HTTPException(502, f"連線 LINE 失敗：{str(e)[:120]}")
        if info.status_code == 401:
            raise HTTPException(400, "Channel Access Token 無效或已過期，請重新從 LINE 後台 Issue 一組再貼上")
        if info.status_code != 200:
            raise HTTPException(400, f"LINE 驗證失敗（{info.status_code}），請確認 token 是否正確")
        j = info.json()
        result["ok"] = True
        result["oa_name"] = j.get("displayName")
        result["oa_picture"] = j.get("pictureUrl")

        # 2) 自動設定 Webhook URL
        try:
            r = await client.put(
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
                headers={**headers, "Content-Type": "application/json"},
                json={"endpoint": webhook_url},
            )
            result["webhook_set"] = r.status_code == 200
            if r.status_code != 200:
                result["warnings"].append("無法自動設定 Webhook URL，請手動貼到 LINE 後台")
        except Exception:
            result["warnings"].append("無法自動設定 Webhook URL，請手動貼到 LINE 後台")

        # 3) 測試 Webhook 是否可連通
        try:
            t = await client.post(
                "https://api.line.me/v2/bot/channel/webhook/test",
                headers={**headers, "Content-Type": "application/json"},
                json={"endpoint": webhook_url},
            )
            if t.status_code == 200:
                result["webhook_active"] = bool(t.json().get("success"))
            if not result["webhook_active"]:
                result["warnings"].append("Webhook 測試未通過，請確認 LINE 後台「使用 webhook」已開啟")
        except Exception:
            pass

    return result


# ──────────────────────────────────────
# LINE Webhook（升級版：防抖 + 靜音 + follow）
# ──────────────────────────────────────

def _get_bot_config(bot_id: str) -> dict:
    """從 Supabase 取得完整 bot 設定"""
    result = supabase.table("bots").select(
        "name, anthropic_api_key, sheet_id, collect_fields, system_prompt, welcome_message, quick_replies, "
        "line_channel_secret, line_channel_access_token, "
        "calendar_id, slot_duration_minutes, business_hours, keyword_triggers, debounce_seconds, "
        "instagram_page_token, instagram_account_id, facebook_page_id, off_hours_message, observe_mode, card_template"
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
        card_template     = bot.get("card_template") or None

        # ── 觀察模式 / 已被真人接手：AI 不回覆，只記錄客戶訊息並轉達給員工 ──
        observe_mode = bool(bot.get("observe_mode"))
        muted = _mute_key(bot_id, user_id) in _muted_line_users
        if observe_mode or muted:
            try:
                supabase.table("conversations").insert({
                    "bot_id": bot_id,
                    "question": combined_msg,
                    "answer": "",
                    "session_id": session_id,
                }).execute()
            except Exception as _e:
                logging.warning(f"[LINE] passive log failed: {_e}")
            await _relay_customer_msg(bot_id, user_id, combined_msg, observe_mode=observe_mode)
            logging.info(f"[LINE] Passive (observe={observe_mode}, muted={muted}) for {user_id}")
            return

        # 抓 LINE 暱稱：填入記憶體快取（對話清單顯示用）＋存入試算表方便對應聊天室
        line_display_name = await fetch_line_display_name(bot_id, user_id, line_token)
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
                card_template=card_template,
            )
        except Exception as e:
            if "NO_API_KEY" in str(e):
                answer = "⚠️ 此 Bot 尚未設定 Gemini API Key，暫時無法回應。"
            else:
                # AI 生成失敗（Gemini 超載/逾時等）→ 給客戶一句 fallback，避免收到一片空白
                logging.error(f"[LINE] generate_answer failed for {user_id}: {e}")
                fallback = "不好意思，系統剛剛有點忙碌 😥 可以麻煩您稍等一下再傳一次嗎？"
                await push_line_message(user_id, fallback, access_token=line_token,
                                        quick_replies=quick_replies)
                return

        # DATA_SAVE 已觸發 → engine 標記 handed_off → 同步靜音 in-memory set
        from app.chat.engine import get_session_status
        if get_session_status(session_id) == "handed_off":
            add_mute(bot_id, user_id)
            logging.info(f"[LINE] Auto-muted {user_id} (handed_off)")
            # 資料收集完成後不再顯示快速選項
            quick_replies = None

        # 任務完成後靜默（handed_off → engine 回空字串）
        if not answer:
            logging.info(f"[LINE] Silent (handed_off) for {user_id}")
            return

        # 貼圖事件：AI 判斷客戶已回答過 → 輸出 sentinel，靜默略過不回、不記錄
        if _STICKER_SKIP_TOKEN in answer:
            logging.info(f"[LINE] Sticker skip for {user_id}")
            return

        # 優先用「回覆 token」（免費、不吃推播月額度）；失敗或過期才退回 push。
        # 客戶剛傳訊息、debounce 幾秒後就回，reply token 通常仍有效，可大幅節省推播額度。
        sent = False
        reply_token = buf.get("reply_token", "")
        if reply_token:
            reply_status = await reply_line_message(
                reply_token, answer, access_token=line_token, quick_replies=quick_replies)
            if reply_status == 200:
                sent = True
            else:
                logging.info(f"[LINE] reply token failed ({reply_status}) for {user_id}, fallback to push")
        if not sent:
            push_ok = await push_line_message(user_id, answer, access_token=line_token, quick_replies=quick_replies)
            if push_ok != 200:
                logging.warning(f"[LINE] push also failed ({push_ok}) for {user_id}")

        # 記錄對話到 DB（供 AI 分析使用）
        try:
            supabase.table("conversations").insert({
                "bot_id": bot_id,
                "question": combined_msg,
                "answer": answer,
                "session_id": session_id,
            }).execute()
        except Exception as _e:
            logging.warning(f"[LINE] conversations insert failed: {_e}")

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

        # ── 只處理 message 事件 ──
        if event["type"] != "message":
            continue

        msg_type = event["message"].get("type")
        reply_token = event["replyToken"]

        # 觀察模式 / 已被真人接手：一律不讓 AI 回覆（訊息仍會被記錄＋轉達員工）
        _passive = bool(bot.get("observe_mode")) or (buf_key in _muted_line_users)

        # ── 圖片訊息：非同步下載 + Gemini 讀圖，轉成文字後走一般流程 ──
        # （被動狀態下仍讀圖，讓員工能看到客戶傳的圖片內容）
        if msg_type == "image":
            asyncio.create_task(_ingest_line_image(
                bot, bot_id, user_id, buf_key, event["message"]["id"], reply_token, line_token))
            continue

        # ── 貼圖：交給 AI 依上下文判斷（沒答的題目再問一次，已答過就略過不回）──
        if msg_type == "sticker":
            if _passive:
                continue
            _enqueue_line_text(bot, bot_id, user_id, buf_key, _STICKER_EVENT_MSG, reply_token)
            continue

        # ── 其他非文字（語音/影片/位置/檔案）：回一句提示引導改打字 ──
        if msg_type != "text":
            if _passive:
                continue
            await reply_line_message(
                reply_token,
                "我這邊收到您的訊息了 😊 但這類訊息我看不懂內容，麻煩您用文字告訴我需要什麼，或直接傳圖片給我看喔！",
                access_token=line_token, quick_replies=bot.get("quick_replies") or None)
            continue

        user_msg = event["message"]["text"]

        # ── 用戶自助重置 ──
        if user_msg.strip() in ["/reset", "重來", "重置", "重新開始"]:
            session_id = f"line_{bot_id}_{user_id}"
            reset_session(session_id)
            remove_mute(bot_id, user_id)
            welcome = bot.get("welcome_message") or f"（記憶已重置）你好！我是{bot.get('name', 'AI 助理')}，有什麼可以幫您的嗎？😊"
            await reply_line_message(reply_token, welcome, access_token=line_token, quick_replies=bot.get("quick_replies") or None)
            continue

        # 注意：靜音／觀察模式的訊息不在此攔截，改由 _process_line_buffer
        # 記錄並轉達給員工（讓真人代回時仍能即時看到客戶新訊息）。

        # ── 垃圾訊息過濾 ──
        if any(kw in user_msg for kw in _SPAM_KEYWORDS) and len(user_msg) > 30:
            add_mute(bot_id, user_id)
            logging.info(f"[LINE] Spam-muted {user_id}: {user_msg[:30]}...")
            continue

        # ── 關鍵字警示（主動通知 owner/admin，非阻塞）──
        asyncio.create_task(_fire_keyword_alert(bot_id, user_id, user_msg))

        _enqueue_line_text(bot, bot_id, user_id, buf_key, user_msg, reply_token)

    # 立即回 200 給 LINE Server，避免 timeout
    return {"status": "ok"}


def _enqueue_line_text(bot: dict, bot_id: str, user_id: str, buf_key: str,
                       user_msg: str, reply_token: str):
    """把一則文字塞進防抖緩衝並（重新）啟動計時器。文字與圖片辨識結果共用。"""
    if buf_key in _line_buffers:
        old_task = _line_buffers[buf_key].get("task")
        if old_task and not old_task.done():
            old_task.cancel()
        _line_buffers[buf_key]["msgs"].append(user_msg)
        _line_buffers[buf_key]["reply_token"] = reply_token
    else:
        _line_buffers[buf_key] = {"msgs": [user_msg], "reply_token": reply_token, "task": None}

    bot_debounce = bot.get("debounce_seconds") or 15
    task = asyncio.create_task(_process_line_buffer(bot_id, user_id, buf_key, bot_debounce))
    _line_buffers[buf_key]["task"] = task
    logging.info(f"[LINE] Buffered msg from {user_id}: '{user_msg[:40]}' ({bot_debounce}s timer)")


async def _ingest_line_image(bot: dict, bot_id: str, user_id: str, buf_key: str,
                             message_id: str, reply_token: str, line_token: str):
    """下載 LINE 圖片 → Gemini 讀圖 → 把辨識內容當文字塞進一般流程。失敗則提示客戶。"""
    api_key = bot.get("anthropic_api_key")
    if not api_key:
        return
    try:
        img_bytes, mime = await download_line_content(message_id, access_token=line_token)
    except Exception as e:
        logging.warning(f"[Vision] download failed {message_id}: {e}")
        await push_line_message(user_id, "圖片我這邊沒收到，可以麻煩您再傳一次，或直接用打字的嗎？🙏",
                                access_token=line_token)
        return

    from app.chat.engine import describe_image
    desc = await asyncio.to_thread(describe_image, api_key, img_bytes, mime)
    if not desc:
        await push_line_message(user_id, "這張圖片我看不太清楚內容，可以麻煩您用文字說明，或換一張清楚一點的嗎？🙏",
                                access_token=line_token)
        return

    text = f"（客戶傳了一張圖片，圖片內容如下）\n{desc}"
    _enqueue_line_text(bot, bot_id, user_id, buf_key, text, reply_token)


# ──────────────────────────────────────
# 管理用 LINE bot（員工遠端接手 / 操作客戶 bot）
# ──────────────────────────────────────

import urllib.parse as _urlparse

# 員工正在「連續代回」某位客戶：之後輸入的每則文字都直接轉給該客戶，
# 直到輸入「結束」離開。key = 管理 bot 的 line userId -> {"bot_id":..., "uid":..., "name":...}
_admin_active_chat: Dict[str, dict] = {}

# 觀察模式：客戶訊息主動通知 owner/admin 的冷卻（避免洗版）
import time as _time_mod
_observe_notify_last: Dict[str, float] = {}   # key=bot|uid -> 上次通知 epoch 秒
_OBSERVE_NOTIFY_COOLDOWN = 90

# 剛綁定完、正在等對方回覆「怎麼稱呼」的暫存（key = 管理 bot 的 line userId）
_admin_pending_name: Dict[str, bool] = {}


def _staff_in_active_chat(bot_id: str, uid: str) -> list:
    """回傳目前正在與這位客戶連續對話的員工 line userId 清單。"""
    return [lid for lid, a in _admin_active_chat.items()
            if a.get("bot_id") == bot_id and a.get("uid") == uid]


async def _relay_customer_msg(bot_id: str, uid: str, msg: str, observe_mode: bool = False):
    """被動狀態下，把客戶新訊息轉達給員工。
    ① 有員工正在連續代回這位客戶 → 直接把訊息推給他，保持對話連續。
    ② 沒人在線上對話時，只有『觀察模式』才主動通知 owner/admin（附卡片可接手／代回），
       已接手但沒在對話的情況不打擾。"""
    try:
        name = await fetch_line_display_name(
            bot_id, uid, _get_bot_config(bot_id).get("line_channel_access_token")) or "LINE 客戶"
    except Exception:
        name = "LINE 客戶"

    active_staff = _staff_in_active_chat(bot_id, uid)
    if active_staff:
        for lid in active_staff:
            try:
                await _admin_push(lid, [_admin_text_msg(f"👤 {name}：{msg}")])
            except Exception as e:
                logging.warning(f"[relay] push to staff failed: {e}")
        return

    if not observe_mode:
        return

    key = f"{bot_id}|{uid}"
    now = _time_mod.time()
    if now - _observe_notify_last.get(key, 0) < _OBSERVE_NOTIFY_COOLDOWN:
        return
    _observe_notify_last[key] = now

    targets = _org_notify_line_ids(bot_id, roles=("owner", "admin"))
    if not targets:
        return
    head = _admin_text_msg(
        f"👀 觀察模式\n客戶「{name}」傳來訊息：\n「{msg[:60]}」\n可直接接手或代回 👇")
    try:
        flex = await _single_conv_flex(bot_id, uid)
    except Exception:
        flex = None
    msgs = [head] + ([flex] if flex else [])
    for lid in targets:
        try:
            await _admin_push(lid, msgs)
        except Exception as e:
            logging.warning(f"[observe] notify failed: {e}")


async def _admin_push(to: str, messages: list) -> int:
    """用管理 bot 主動推播（支援 Flex / 多則訊息）。"""
    if not ADMIN_LINE_CHANNEL_ACCESS_TOKEN:
        return 0
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {ADMIN_LINE_CHANNEL_ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"to": to, "messages": messages[:5]},
        )
        return resp.status_code


async def _admin_reply(reply_token: str, messages: list) -> int:
    """用管理 bot 回覆（reply token）。"""
    if not ADMIN_LINE_CHANNEL_ACCESS_TOKEN or not reply_token:
        return 0
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {ADMIN_LINE_CHANNEL_ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"replyToken": reply_token, "messages": messages[:5]},
        )
        return resp.status_code


def _line_id_for_app_user(app_user_id: str) -> Optional[str]:
    """某個後台帳號可用管理助手推播的 LINE userId：優先 staff_line，其次 app_users.line_user_id。"""
    try:
        sl = supabase.table("staff_line").select("line_user_id").eq("app_user_id", app_user_id).execute()
        if sl.data and sl.data[0].get("line_user_id"):
            return sl.data[0]["line_user_id"]
        au = supabase.table("app_users").select("line_user_id").eq("id", app_user_id).execute()
        return au.data[0].get("line_user_id") if au.data else None
    except Exception:
        return None


def _org_notify_line_ids(bot_id: str, roles=("owner", "admin")) -> list:
    """這個 bot 所屬團隊裡、指定角色、且有綁 LINE 的成員的 LINE userId（去重）。"""
    try:
        b = supabase.table("bots").select("org_id").eq("id", bot_id).execute()
        org_id = b.data[0].get("org_id") if b.data else None
        if not org_id:
            return []
        mems = supabase.table("memberships").select("user_id, role") \
            .eq("org_id", org_id).in_("role", list(roles)).execute().data or []
        ids, seen = [], set()
        for m in mems:
            lid = _line_id_for_app_user(m["user_id"])
            if lid and lid not in seen:
                seen.add(lid)
                ids.append(lid)
        return ids
    except Exception as e:
        logging.warning(f"[notify] resolve org line ids failed: {e}")
        return []


def _muted_by_name_map(bot_id: str, uids: list) -> dict:
    """回傳 {line_user_id: 接手者顯示名稱}，供防撞單顯示是誰接手的。"""
    if not uids:
        return {}
    try:
        rows = supabase.table("chat_mutes").select("line_user_id, muted_by") \
            .eq("bot_id", bot_id).in_("line_user_id", uids).execute().data or []
        by_ids = list({r["muted_by"] for r in rows if r.get("muted_by")})
        name_by_id = {}
        if by_ids:
            us = supabase.table("app_users").select("id, display_name, email") \
                .in_("id", by_ids).execute().data or []
            name_by_id = {u["id"]: (u.get("display_name") or u.get("email") or "員工") for u in us}
        out = {}
        for r in rows:
            if r.get("muted_by"):
                out[r["line_user_id"]] = name_by_id.get(r["muted_by"], "員工")
        return out
    except Exception:
        return {}


def _customer_background(bot_id: str, uid: str) -> str:
    """接手時給員工的客戶背景：最近幾則問答 + 該 bot 要收集的聯絡欄位提示。"""
    lines = []
    try:
        rows = supabase.table("conversations") \
            .select("question, answer, created_at") \
            .eq("bot_id", bot_id).eq("session_id", f"line_{bot_id}_{uid}") \
            .order("created_at", desc=True).limit(5).execute().data or []
        rows = list(reversed(rows))  # 由舊到新，讀起來順
        for c in rows:
            q = (c.get("question") or "").strip().replace("\n", " ")[:60]
            a = (c.get("answer") or "").strip().replace("\n", " ")[:60]
            if q:
                lines.append(f"👤 {q}")
            if a:
                lines.append(f"🤖 {a}")
    except Exception:
        pass
    header = "📋 客戶背景（最近對話）"
    body = "\n".join(lines) if lines else "（尚無對話紀錄）"
    try:
        bot = _get_bot_config(bot_id)
        fields = bot.get("collect_fields") or []
        labels = [f.get("label") or f.get("name") for f in fields if isinstance(f, dict)]
        labels = [x for x in labels if x]
        if labels:
            body += f"\n\n📝 需收集：{ '、'.join(labels) }"
    except Exception:
        pass
    return f"{header}\n{body}"


# 客戶訊息含這些字 → 管理助手主動警示 owner+admin
_ALERT_KEYWORDS = ["退費", "退款", "退貨", "客訴", "投訴", "不要了", "太貴",
                   "很爛", "爛透", "律師", "檢舉", "詐騙", "生氣", "失望"]
_alert_last: Dict[str, float] = {}   # key=bot|uid -> 上次警示 epoch 秒
_ALERT_COOLDOWN = 600                 # 同一位客戶 10 分鐘內只警示一次


async def _fire_keyword_alert(bot_id: str, uid: str, user_msg: str):
    """客戶訊息命中敏感詞時，主動推播警示給 owner+admin（附對話卡可直接接手）。"""
    hit = next((k for k in _ALERT_KEYWORDS if k in user_msg), None)
    if not hit:
        return
    import time as _time
    key = f"{bot_id}|{uid}"
    now = _time.time()
    if now - _alert_last.get(key, 0) < _ALERT_COOLDOWN:
        return
    _alert_last[key] = now
    targets = _org_notify_line_ids(bot_id, roles=("owner", "admin"))
    if not targets:
        return
    try:
        name = await fetch_line_display_name(
            bot_id, uid, _get_bot_config(bot_id).get("line_channel_access_token")) or "LINE 客戶"
    except Exception:
        name = "LINE 客戶"
    alert = _admin_text_msg(
        f"⚠️ 關鍵字警示\n客戶「{name}」提到「{hit}」：\n「{user_msg[:60]}」\n可直接接手或代回 👇")
    try:
        flex = await _single_conv_flex(bot_id, uid)
    except Exception:
        flex = None
    msgs = [alert] + ([flex] if flex else [])
    for lid in targets:
        try:
            await _admin_push(lid, msgs)
        except Exception as e:
            logging.warning(f"[alert] push failed: {e}")


def _menu_quick_items() -> list:
    return [{"type": "action", "action": {"type": "message", "label": "📋 清單", "text": "清單"}}]


def _admin_text_msg(text: str, quick_items: Optional[list] = None) -> dict:
    msg: dict = {"type": "text", "text": text}
    if quick_items:
        msg["quickReply"] = {"items": quick_items}
    return msg


def get_staff_by_line(line_user_id: str) -> Optional[dict]:
    """管理 bot userId → app_user（未綁定回 None）。"""
    r = supabase.table("staff_line").select("app_user_id").eq("line_user_id", line_user_id).execute()
    if not r.data:
        return None
    au = supabase.table("app_users").select("*").eq("id", r.data[0]["app_user_id"]).execute()
    return au.data[0] if au.data else None


def _staff_bots(app_user: dict) -> list:
    """員工可操作的所有客戶 bot（團隊內 + 自己建立的舊 bot）。"""
    org_ids = get_user_org_ids(app_user["id"])
    seen: dict = {}
    cols = "id, name, org_id, user_id, line_channel_access_token"
    if org_ids:
        r = supabase.table("bots").select(cols).in_("org_id", org_ids).execute()
        for b in (r.data or []):
            seen[b["id"]] = b
    my_uid = app_user.get("supabase_uid")
    if my_uid:
        r = supabase.table("bots").select(cols).eq("user_id", my_uid).execute()
        for b in (r.data or []):
            seen.setdefault(b["id"], b)
    return list(seen.values())


def _autobind_staff_by_line(line_user_id: str) -> Optional[dict]:
    """免 6 碼自動綁定：若這支 LINE userId 已對應到某個後台帳號
    （同 provider 的 LINE 登入時存進 app_users.line_user_id），
    直接建立 staff_line 完成綁定。找不到回 None。"""
    if not line_user_id:
        return None
    try:
        au = supabase.table("app_users").select("*").eq("line_user_id", line_user_id).execute()
        if not au.data:
            return None
        app_user = au.data[0]
        supabase.table("staff_line").upsert(
            {"line_user_id": line_user_id, "app_user_id": app_user["id"]},
            on_conflict="line_user_id",
        ).execute()
        return app_user
    except Exception:
        return None


def _try_bind_staff(line_user_id: str, text: str) -> Optional[dict]:
    """用綁定碼把員工 LINE 綁到 app_user，成功回 app_user，失敗回 None。"""
    import re
    m = re.search(r"\d{6}", text)
    if not m:
        return None
    code = m.group(0)
    r = supabase.table("line_binding_codes").select("*").eq("code", code).execute()
    if not r.data:
        return None
    rec = r.data[0]
    try:
        exp = datetime.fromisoformat(str(rec["expires_at"]).replace("Z", "+00:00"))
        if exp.tzinfo:
            exp = exp.replace(tzinfo=None)
        if datetime.utcnow() > exp:
            supabase.table("line_binding_codes").delete().eq("code", code).execute()
            return None
    except Exception:
        pass
    app_user_id = rec["app_user_id"]
    supabase.table("staff_line").upsert(
        {"line_user_id": line_user_id, "app_user_id": app_user_id},
        on_conflict="line_user_id",
    ).execute()
    supabase.table("line_binding_codes").delete().eq("code", code).execute()
    # 同步 LINE 身分到主帳號：讓之後「LINE 登入」能認出同一人、不再開分身帳號
    # （6 碼綁定與自動綁定兩條路結果一致）
    try:
        others = supabase.table("app_users").select("id") \
            .eq("line_user_id", line_user_id).neq("id", app_user_id).execute().data or []
        for o in others:
            supabase.table("app_users").update({"line_user_id": None}).eq("id", o["id"]).execute()
        supabase.table("app_users").update({"line_user_id": line_user_id}).eq("id", app_user_id).execute()
    except Exception as e:
        logging.warning(f"[bind] sync line_user_id to app_user failed: {e}")
    au = supabase.table("app_users").select("*").eq("id", app_user_id).execute()
    return au.data[0] if au.data else None


def _try_join_team_by_code(line_user_id: str, text: str) -> Optional[dict]:
    """用「團隊邀請碼」讓員工用 LINE 加入團隊並綁定：
    從他的 LINE 身分建立/找到 app_user → 加進團隊 → 綁 staff_line。成功回 app_user。
    這組碼可重複使用（不刪除），每位員工各自成為獨立成員。"""
    import re
    m = re.search(r"\d{6}", text)
    if not m:
        return None
    code = m.group(0)
    r = supabase.table("team_join_codes").select("*").eq("code", code).execute()
    if not r.data:
        return None
    rec = r.data[0]
    org_id = rec["org_id"]
    role = rec.get("role") or "editor"
    # 從 LINE 身分建立/找到員工的 app_user（沒有 supabase 帳號也可以，之後 LINE 登入會對上）
    app_user = ensure_app_user(line_user_id=line_user_id)
    app_user_id = app_user["id"]
    # 加進團隊（已是成員就不動）
    if get_membership_role(org_id, app_user_id) is None:
        supabase.table("memberships").insert({
            "org_id": org_id, "user_id": app_user_id, "role": role,
        }).execute()
    # 綁定管理助手
    supabase.table("staff_line").upsert(
        {"line_user_id": line_user_id, "app_user_id": app_user_id},
        on_conflict="line_user_id",
    ).execute()
    return app_user


def _pb(action: str, bot_id: str, uid: str) -> str:
    return _urlparse.urlencode({"a": action, "b": bot_id, "u": uid})


def _fmt_time(iso: str) -> str:
    """ISO 時間 → 台北時間 MM/DD HH:MM。"""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        off = dt.utcoffset() or timedelta()
        dt = dt.replace(tzinfo=None) - off + timedelta(hours=8)
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return ""


def _info_row(label: str, value: str, value_color: str = "#111111", value_weight: str = "regular") -> dict:
    return {
        "type": "box", "layout": "baseline", "spacing": "sm",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": "#AAAAAA", "flex": 2},
            {"type": "text", "text": value or "—", "size": "sm", "color": value_color,
             "weight": value_weight, "flex": 5, "wrap": True},
        ],
    }


def _conv_bubble(it: dict) -> dict:
    muted = it["muted"]
    accent = "#EA580C" if muted else "#16A34A"          # 接手中橘 / AI 綠
    if muted:
        who = it.get("muted_by_name")
        status = f"🙋 {who} 接手中" if who else "🙋 真人接手中"
    else:
        status = "🤖 AI 自動回覆中"
    if muted:
        toggle_btn = {"type": "button", "style": "primary", "color": "#2563EB", "height": "sm",
                      "action": {"type": "postback", "label": "🤖 恢復 AI 回覆",
                                 "data": _pb("unmute", it["bot_id"], it["uid"]),
                                 "displayText": f"恢復 AI 回覆 {it['name']}"}}
    else:
        toggle_btn = {"type": "button", "style": "primary", "color": "#EA580C", "height": "sm",
                      "action": {"type": "postback", "label": "🙋 真人接手",
                                 "data": _pb("mute", it["bot_id"], it["uid"]),
                                 "displayText": f"接手 {it['name']}"}}
    reply_btn = {"type": "button", "style": "secondary", "height": "sm",
                 "action": {"type": "postback", "label": "💬 代回訊息",
                            "data": _pb("reply", it["bot_id"], it["uid"]),
                            "displayText": f"代回 {it['name']}"}}
    return {
        "type": "bubble", "size": "mega",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": accent,
            "paddingAll": "16px", "spacing": "xs",
            "contents": [
                {"type": "text", "text": it["name"], "weight": "bold", "size": "lg",
                 "color": "#FFFFFF", "wrap": True},
                {"type": "text", "text": f"🤖 {it['bot_name']}", "size": "xs", "color": "#FFFFFFCC"},
            ],
        },
        "body": {
            "type": "box", "layout": "vertical", "spacing": "md", "paddingAll": "16px",
            "contents": [
                _info_row("狀態", status, accent, "bold"),
                _info_row("時間", it.get("time_label") or ""),
                {"type": "separator", "margin": "md", "color": "#EEEEEE"},
                {"type": "text", "text": "最後訊息", "size": "xs", "color": "#AAAAAA", "margin": "md"},
                {"type": "text",
                 "text": (it["last_q"] if it["last_q"] else "（無內容）"),
                 "size": "sm", "color": "#333333", "wrap": True},
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical", "spacing": "sm", "paddingAll": "12px",
            "contents": [toggle_btn, reply_btn],
        },
    }


async def _single_conv_flex(bot_id: str, uid: str) -> dict:
    """重建單一對話的 Flex 卡片（按鈕動作後即時回覆用）。"""
    bot = _get_bot_config(bot_id)
    token = bot.get("line_channel_access_token")
    name = await fetch_line_display_name(bot_id, uid, token) or "LINE 客戶"
    last = supabase.table("conversations").select("question, created_at") \
        .eq("bot_id", bot_id).eq("session_id", f"line_{bot_id}_{uid}") \
        .order("created_at", desc=True).limit(1).execute()
    last_q, last_at = "", ""
    if last.data:
        last_q = (last.data[0].get("question") or "")[:40]
        last_at = last.data[0].get("created_at") or ""
    muted = _mute_key(bot_id, uid) in _muted_line_users
    it = {
        "bot_id": bot_id, "bot_name": bot.get("name") or "Bot", "uid": uid,
        "last_q": last_q, "last_at": last_at, "time_label": _fmt_time(last_at),
        "muted": muted, "name": name,
        "muted_by_name": _muted_by_name_map(bot_id, [uid]).get(uid) if muted else None,
    }
    return {"type": "flex", "altText": name, "contents": _conv_bubble(it)}


async def _build_admin_conversation_list(app_user: dict) -> list:
    """建立「可接手對話」Flex 清單（回傳 messages）。"""
    bots = _staff_bots(app_user)
    if not bots:
        return [_admin_text_msg("你的團隊目前沒有任何 LINE bot。")]

    since = (datetime.utcnow() - timedelta(days=2)).isoformat()
    items = []
    for b in bots:
        rows = supabase.table("conversations") \
            .select("session_id, question, created_at") \
            .eq("bot_id", b["id"]).gte("created_at", since) \
            .order("created_at", desc=True).limit(60).execute()
        latest: dict = {}
        for c in (rows.data or []):
            uid = _line_user_id_from_session(b["id"], c.get("session_id") or "")
            if not uid or uid in latest:
                continue
            latest[uid] = c  # desc 排序，第一筆即最新
        for uid, c in latest.items():
            items.append({
                "bot_id": b["id"],
                "bot_name": b.get("name") or "Bot",
                "uid": uid,
                "last_q": (c.get("question") or "")[:40],
                "last_at": c.get("created_at") or "",
                "time_label": _fmt_time(c.get("created_at")),
                "muted": _mute_key(b["id"], uid) in _muted_line_users,
                "token": b.get("line_channel_access_token"),
            })
    if not items:
        return [_admin_text_msg("近 2 天內沒有 LINE 對話。", _menu_quick_items())]

    items.sort(key=lambda x: x["last_at"], reverse=True)
    items = items[:10]
    # 防撞單：查出每個已接手對話是誰接手的
    by_bot: dict = {}
    for it in items:
        if it.get("muted"):
            by_bot.setdefault(it["bot_id"], []).append(it["uid"])
    name_map: dict = {}
    for bid, uids in by_bot.items():
        for k, v in _muted_by_name_map(bid, uids).items():
            name_map[(bid, k)] = v
    for it in items:
        it["name"] = await fetch_line_display_name(it["bot_id"], it["uid"], it["token"]) or "LINE 客戶"
        if it.get("muted"):
            it["muted_by_name"] = name_map.get((it["bot_id"], it["uid"]))

    flex = {
        "type": "flex",
        "altText": f"可接手對話（{len(items)}）",
        "contents": {"type": "carousel", "contents": [_conv_bubble(it) for it in items]},
    }
    return [flex]


async def _admin_send_customer_reply(staff: dict, reply_token: str, pending: dict, text: str,
                                     light: bool = False):
    """員工代回：用客戶 bot 把訊息推給客戶，並自動接手（AI 靜音）。
    light=True 用於連續對話模式：只回一句簡短確認，不附大張卡片，避免洗版。"""
    bot_id, uid = pending["bot_id"], pending["uid"]
    name = pending.get("name") or "客戶"
    if bot_id not in {b["id"] for b in _staff_bots(staff)}:
        await _admin_reply(reply_token, [_admin_text_msg("你沒有這個對話的操作權限。")])
        return
    token = _get_bot_config(bot_id).get("line_channel_access_token")
    status = await push_line_message(uid, text, access_token=token)
    if status == 200:
        add_mute(bot_id, uid, muted_by=staff["id"])
        try:
            supabase.table("conversations").insert({
                "bot_id": bot_id,
                "question": f"（真人 {staff.get('display_name') or ''} 代回）",
                "answer": text,
                "session_id": f"line_{bot_id}_{uid}",
            }).execute()
        except Exception:
            pass
        if light:
            await _admin_reply(reply_token, [_admin_text_msg(
                f"✅ 已傳給「{name}」（輸入「結束」離開對話）")])
        else:
            flex = await _single_conv_flex(bot_id, uid)
            await _admin_reply(reply_token, [_admin_text_msg(
                f"✅ 已傳給「{name}」。AI 已暫停，需要時按「恢復 AI 回覆」。"), flex])
    else:
        await _admin_reply(reply_token, [_admin_text_msg(f"❌ 傳送失敗（{status}），請稍後再試。")])


async def _handle_admin_postback(staff: dict, line_uid: str, reply_token: str, params: dict):
    action, bot_id, uid = params.get("a"), params.get("b"), params.get("u")
    if not action or not bot_id or not uid:
        return
    if bot_id not in {b["id"] for b in _staff_bots(staff)}:
        await _admin_reply(reply_token, [_admin_text_msg("你沒有這個對話的操作權限。")])
        return
    name = await fetch_line_display_name(
        bot_id, uid, _get_bot_config(bot_id).get("line_channel_access_token")) or "客戶"
    if action == "mute":
        add_mute(bot_id, uid, muted_by=staff["id"])
        _admin_active_chat[line_uid] = {"bot_id": bot_id, "uid": uid, "name": name}
        flex = await _single_conv_flex(bot_id, uid)
        bg = _customer_background(bot_id, uid)
        await _admin_reply(reply_token, [_admin_text_msg(
            f"✅ 已接手「{name}」，AI 已暫停。\n💬 直接輸入文字即可回覆對方，輸入「結束」離開對話。"),
            _admin_text_msg(bg), flex])
    elif action == "unmute":
        remove_mute(bot_id, uid)
        if (_admin_active_chat.get(line_uid) or {}).get("uid") == uid:
            _admin_active_chat.pop(line_uid, None)
        flex = await _single_conv_flex(bot_id, uid)
        await _admin_reply(reply_token, [_admin_text_msg(
            f"✅ 已恢復「{name}」的 AI 自動回覆。"), flex])
    elif action == "reply":
        add_mute(bot_id, uid, muted_by=staff["id"])
        _admin_active_chat[line_uid] = {"bot_id": bot_id, "uid": uid, "name": name}
        bg = _customer_background(bot_id, uid)
        await _admin_reply(reply_token, [
            _admin_text_msg(bg),
            _admin_text_msg(f"💬 你正在與「{name}」對話，直接輸入文字即可傳給對方。\n輸入「結束」離開對話。")])


async def _handle_admin_event(event: dict):
    etype = event.get("type")
    line_uid = event.get("source", {}).get("userId", "")
    reply_token = event.get("replyToken", "")
    if not line_uid:
        return

    staff = get_staff_by_line(line_uid)

    if etype == "follow":
        if staff:
            await _admin_reply(reply_token, [_admin_text_msg(
                f"歡迎回來，{staff.get('display_name') or ''}！輸入「清單」查看可接手的對話。",
                _menu_quick_items())])
        else:
            # 加好友即嘗試自動綁定（同 provider 的 LINE 登入身分），免 6 碼
            auto = _autobind_staff_by_line(line_uid)
            if auto:
                _admin_pending_name[line_uid] = True
                await _admin_reply(reply_token, [_admin_text_msg(
                    "✅ 綁定成功！請問我該怎麼稱呼你？（直接回覆你的稱呼即可，例如：小明、王經理）")])
            else:
                await _admin_reply(reply_token, [_admin_text_msg(
                    "歡迎使用懶得回管理助手 🤖\n\n請先綁定身分：登入後台 →「團隊成員」頁 → 點「綁定我的 LINE」"
                    "取得 6 碼綁定碼，把數字傳給我即可完成綁定。")])
        return

    if etype == "postback":
        if not staff:
            await _admin_reply(reply_token, [_admin_text_msg("請先綁定身分再操作。")])
            return
        params = dict(_urlparse.parse_qsl(event.get("postback", {}).get("data", "")))
        await _handle_admin_postback(staff, line_uid, reply_token, params)
        return

    if etype == "message" and event.get("message", {}).get("type") == "text":
        text = event["message"]["text"].strip()

        if not staff:
            # 先試自動綁定（LINE 登入身分），再試個人綁定碼，最後試團隊邀請碼
            bound = (_autobind_staff_by_line(line_uid)
                     or _try_bind_staff(line_uid, text)
                     or _try_join_team_by_code(line_uid, text))
            if bound:
                _admin_pending_name[line_uid] = True
                await _admin_reply(reply_token, [_admin_text_msg(
                    "✅ 綁定成功！請問我該怎麼稱呼你？（直接回覆你的稱呼即可，例如：小明、王經理）")])
            else:
                await _admin_reply(reply_token, [_admin_text_msg(
                    "尚未綁定。請到後台「團隊成員」頁點「綁定我的 LINE」取得 6 碼綁定碼，傳給我完成綁定。")])
            return

        # 剛綁定完，這則訊息當作「怎麼稱呼」的回覆
        if _admin_pending_name.pop(line_uid, None):
            name = text.strip()[:40]
            if name and name not in ["清單", "選單", "menu", "list", "對話", "跳過", "skip"]:
                try:
                    supabase.table("app_users").update({"display_name": name}).eq("id", staff["id"]).execute()
                except Exception:
                    pass
                await _admin_reply(reply_token, [_admin_text_msg(
                    f"好的，之後就稱呼你「{name}」😊 輸入「清單」查看可接手的對話。", _menu_quick_items())])
            else:
                await _admin_reply(reply_token, [_admin_text_msg(
                    "好的，先跳過稱呼。輸入「清單」查看可接手的對話。", _menu_quick_items())])
            return

        # ── 連續代回：員工正在與某位客戶對話中 ──
        active = _admin_active_chat.get(line_uid)
        if active:
            aname = active.get("name") or "客戶"
            if text in ["結束", "離開", "退出", "結束對話", "end", "bye", "掰掰"]:
                _admin_active_chat.pop(line_uid, None)
                await _admin_reply(reply_token, [_admin_text_msg(
                    f"已離開與「{aname}」的對話。AI 仍為暫停狀態，需要時在卡片按「恢復 AI 回覆」。",
                    _menu_quick_items())])
                return
            # 切換到指令（清單／摘要等）→ 先離開對話，再往下走指令處理
            if text in ["清單", "選單", "menu", "list", "對話", "摘要", "統計", "今日", "報表"]:
                _admin_active_chat.pop(line_uid, None)
            else:
                # 一般文字 → 直接傳給客戶，維持連續對話
                await _admin_send_customer_reply(staff, reply_token, active, text, light=True)
                return

        if text in ["清單", "選單", "menu", "list", "對話"]:
            msgs = await _build_admin_conversation_list(staff)
            await _admin_reply(reply_token, msgs)
            return

        if text in ["摘要", "統計", "今日", "報表"]:
            summary = _build_daily_summary_for_user(staff)
            await _admin_reply(reply_token, [_admin_text_msg(
                summary or "今天目前還沒有任何對話。", _menu_quick_items())])
            return

        await _admin_reply(reply_token, [_admin_text_msg(
            "指令：\n・輸入「清單」查看可接手的對話\n・輸入「摘要」查看今日營運統計\n・在對話卡片按「接手／恢復 AI／代回」操作",
            _menu_quick_items())])
        return


@app.post("/line/admin/webhook")
async def admin_line_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not ADMIN_LINE_CHANNEL_SECRET or not ADMIN_LINE_CHANNEL_ACCESS_TOKEN:
        raise HTTPException(503, "管理 bot 尚未設定")
    if not verify_line_signature(body, signature, channel_secret=ADMIN_LINE_CHANNEL_SECRET):
        raise HTTPException(400, "簽名驗證失敗")
    data = json.loads(body)
    for event in data.get("events", []):
        try:
            await _handle_admin_event(event)
        except Exception as e:
            logging.error(f"[ADMIN LINE] event error: {e}")
    return {"status": "ok"}


@app.post("/me/line-bind-code")
async def create_line_bind_code(authorization: Optional[str] = Header(None)):
    """產生 6 碼綁定碼，員工傳給管理 bot 完成綁定。"""
    import random
    app_user = get_app_user(authorization)
    supabase.table("line_binding_codes").delete().eq("app_user_id", app_user["id"]).execute()
    expires = (datetime.utcnow() + timedelta(minutes=15)).isoformat()
    code = None
    for _ in range(6):
        cand = f"{random.randint(0, 999999):06d}"
        try:
            supabase.table("line_binding_codes").insert(
                {"code": cand, "app_user_id": app_user["id"], "expires_at": expires}
            ).execute()
            code = cand
            break
        except Exception:
            continue
    if not code:
        raise HTTPException(500, "產生綁定碼失敗，請重試")
    return {"code": code, "expires_in": 900}


@app.get("/me/line-bind-status")
async def line_bind_status(authorization: Optional[str] = Header(None)):
    """查詢目前登入者是否已綁定管理 bot。"""
    app_user = get_app_user(authorization)
    r = supabase.table("staff_line").select("line_user_id").eq("app_user_id", app_user["id"]).execute()
    return {"bound": bool(r.data)}


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
    access = require_bot_access(body.bot_id, authorization, min_role="editor")
    r = supabase.table("bots").select("anthropic_api_key").eq("id", body.bot_id).execute()
    if not r.data:
        raise HTTPException(404, "Bot 不存在")
    api_key = r.data[0].get("anthropic_api_key")
    session_id = body.session_id or f"assistant_{access['app_user']['id']}_{body.bot_id}"

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
    require_bot_access(bot_id, authorization, min_role="viewer")
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
    require_bot_access(bot_id, authorization, min_role="editor")

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
    require_bot_access(bot_id, authorization, min_role="viewer")

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
    require_bot_access(bot_id, authorization, min_role="viewer")
    result = supabase.table("conversations")\
        .select("*")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)\
        .limit(100)\
        .execute()
    return result.data


@app.delete("/bots/{bot_id}/conversations/all")
async def delete_all_conversations(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    """刪除此 bot 全部對話記錄"""
    require_bot_access(bot_id, authorization, min_role="editor")

    # 先算總數再刪
    cnt = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).execute()
    total = cnt.count or 0
    supabase.table("conversations").delete().eq("bot_id", bot_id).execute()
    return {"deleted": total}


class AnalysisRequest(BaseModel):
    days: int = 30  # 分析最近幾天，0 = 不限時間


@app.post("/bots/{bot_id}/ai-analysis")
async def ai_analysis(
    bot_id: str,
    body: AnalysisRequest = AnalysisRequest(),
    authorization: Optional[str] = Header(None)
):
    """用 Gemini 分析最近對話 session，回傳借貸業務洞察報告"""
    require_bot_access(bot_id, authorization, min_role="viewer")

    # 取 bot 設定（含角色描述）
    bot_row = supabase.table("bots").select("anthropic_api_key, name, system_prompt").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key      = bot_row.data[0].get("anthropic_api_key")
    bot_name     = bot_row.data[0].get("name", "Bot")
    system_prompt = bot_row.data[0].get("system_prompt") or ""
    if not api_key:
        raise HTTPException(400, "請先設定 Gemini API Key")

    # 取對話，依 days 篩選時間範圍
    query = supabase.table("conversations")\
        .select("question, answer, session_id, created_at")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=False)
    if body.days > 0:
        since = (datetime.utcnow() - timedelta(days=body.days)).isoformat()
        query = query.gte("created_at", since)
    rows = query.limit(200).execute()
    convs = rows.data or []
    if not convs:
        # 查 DB 全部總數，協助 debug
        total_all = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).execute().count or 0
        if total_all == 0:
            raise HTTPException(400, f"此 Bot 沒有任何對話記錄（資料庫共 0 筆）。如果是 LINE Bot，請確認最近版本已部署、且有實際對話發生。")
        else:
            range_label = f"{body.days} 天內" if body.days > 0 else "全部"
            raise HTTPException(400, f"{range_label}沒有對話記錄。資料庫實際共有 {total_all} 筆，請選擇更長的時間範圍（例如「全部」）。")

    # ── 依 session_id 分組，組成對話串 ──
    from collections import OrderedDict
    sessions: OrderedDict = OrderedDict()
    for c in convs:
        sid = c.get("session_id") or "unknown"
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(c)

    # ── 計算完成率（答覆中含 DATA_SAVE 視為完成資料收集）──
    total_sessions  = len(sessions)
    completed_sessions = sum(
        1 for msgs in sessions.values()
        if any("DATA_SAVE" in str(m.get("answer", "")) for m in msgs)
    )
    completion_rate = round(completed_sessions / total_sessions * 100) if total_sessions else 0

    # ── 組成給 AI 的對話文本（最多取 40 個 session）──
    def _channel_of(sid: str) -> str:
        if sid.startswith("line_"):       return "LINE"
        if sid.startswith("widget_") or sid.startswith("test_"): return "網頁/測試"
        if sid.startswith("ig_cmt_"):     return "IG留言"
        if sid.startswith("ig_"):         return "IG"
        if sid.startswith("assistant_"):  return "設定助手"
        return "其他"

    session_blocks = []
    session_labels = []  # 給 AI 在報告中引用用
    for idx, (sid, msgs) in enumerate(list(sessions.items())[-40:], 1):
        is_done = any("DATA_SAVE" in str(m.get("answer", "")) for m in msgs)
        status  = "✅ 完成資料收集" if is_done else "❌ 未完成"
        channel = _channel_of(sid)
        short_id = sid[-8:] if len(sid) > 8 else sid
        first_at = str(msgs[0].get("created_at", ""))[:16].replace("T", " ")
        label = f"客戶 #{idx}（{channel}・{short_id}・{first_at}）"
        session_labels.append(label)
        turns = []
        for m in msgs:
            q = str(m.get("question", "")).strip()[:200]
            a = str(m.get("answer", "")).strip()[:200]
            import re as _re
            a = _re.sub(r'DATA_(?:SAVE|PARTIAL):\s*\{.*?\}', '', a, flags=_re.DOTALL).strip()
            turns.append(f"  客戶：{q}\n  Bot：{a}")
        session_blocks.append(
            f"【{label}｜{status}】\n" + "\n".join(turns)
        )
    conversation_text = "\n\n---\n\n".join(session_blocks)

    # ── 角色設定摘要（給 AI 參考，只取前 300 字）──
    role_context = f"\n\n【Bot 角色設定（供參考）】\n{system_prompt[:300]}" if system_prompt.strip() else ""

    prompt = f"""你是一位專精借貸業務的客服優化顧問。以下是「{bot_name}」這個貸款諮詢 AI Bot 最近的對話記錄，已按 session 分組，共 {total_sessions} 組對話、{completed_sessions} 組完成資料收集（完成率 {completion_rate}%）。{role_context}

=== 對話記錄 ===

{conversation_text}

=== 分析任務 ===

請以繁體中文輸出一份結構化分析報告，**逐一檢視每組對話**，再做總結。格式如下：

## 各客戶對話分析

針對「上面每一組對話」都各做一段小分析（用 ### 當標題，標題格式：`### 客戶 #編號（來源・尾碼）`，要跟上方對話的 label 完全對得起來，方便我去後台對照）。每段請包含：
- **狀態**：完成 / 未完成（在哪個環節中斷）
- **客戶疑慮**：這位客戶最在意什麼、有沒有猶豫點
- **Bot 表現**：這次回答有沒有問題（重複詢問、失憶、答非所問、語氣等），引用一句實際對話佐證
- **建議**：針對這位客戶，下次應該怎麼處理會更好（一句話）

請務必每一組都寫，不要跳過。如果某組對話太短（例如只有 1-2 句），仍要簡短註記。

## 整體總結

- **完成率**：{completed_sessions}/{total_sessions}（{completion_rate}%）是否正常、最常見的斷點是什麼
- **客戶共通疑慮**：歸納上面所有客戶共同關心的 2-3 個點
- **Bot 反覆出現的問題**：哪些錯誤模式在多個對話中重複出現

## 改善建議（優先順序）

針對整體找出 3-5 個可執行的改善方向，每項標明【優先級：高/中/低】，並具體說明：
- 要改 Bot 的哪個部分（system_prompt / collect_fields / welcome_message / quick_replies / FAQ 知識庫）
- 改成什麼樣的具體內容
- 預期能解決上面哪幾位客戶遇到的問題（引用客戶編號）

## 整體評估

- 一段話總結這個 Bot 目前在貸款詢問上的表現
- 給出 1-10 的綜合評分並說明理由

只輸出報告本身，不要加前言或後記。"""

    try:
        from google import genai
        from google.genai import types
        import time as _time

        client = genai.Client(api_key=api_key)
        last_err = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(
                        max_output_tokens=6000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                parts = response.candidates[0].content.parts if response.candidates else []
                report = "".join(p.text for p in parts if hasattr(p, "text") and not getattr(p, "thought", False))
                return {
                    "report": report.strip(),
                    "stats": {
                        "total_sessions":     total_sessions,
                        "completed_sessions": completed_sessions,
                        "completion_rate":    completion_rate,
                        "total_messages":     len(convs),
                    }
                }
            except Exception as e:
                last_err = e
                if "503" in str(e) or "overloaded" in str(e).lower():
                    _time.sleep(3 * (attempt + 1))
                else:
                    raise
        raise last_err
    except Exception as e:
        raise HTTPException(500, f"AI 分析失敗：{str(e)[:200]}")


@app.post("/bots/{bot_id}/style-analysis")
async def style_analysis(
    bot_id: str,
    body: AnalysisRequest = AnalysisRequest(),
    authorization: Optional[str] = Header(None)
):
    """語氣風格分析：讀客戶訊息 + 員工真人代回 + Bot 回覆，產出話術風格指南。"""
    require_bot_access(bot_id, authorization, min_role="viewer")

    bot_row = supabase.table("bots").select("anthropic_api_key, name, system_prompt").eq("id", bot_id).execute()
    if not bot_row.data:
        raise HTTPException(404, "Bot 不存在")
    api_key       = bot_row.data[0].get("anthropic_api_key")
    bot_name      = bot_row.data[0].get("name", "Bot")
    system_prompt = bot_row.data[0].get("system_prompt") or ""
    if not api_key:
        raise HTTPException(400, "請先設定 Gemini API Key")

    query = supabase.table("conversations")\
        .select("question, answer, session_id, created_at")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=False)
    if body.days > 0:
        since = (datetime.utcnow() - timedelta(days=body.days)).isoformat()
        query = query.gte("created_at", since)
    rows = query.limit(200).execute()
    convs = rows.data or []
    if not convs:
        total_all = supabase.table("conversations").select("id", count="exact").eq("bot_id", bot_id).execute().count or 0
        if total_all == 0:
            raise HTTPException(400, "此 Bot 沒有任何對話記錄，無法分析語氣風格。")
        range_label = f"{body.days} 天內" if body.days > 0 else "全部"
        raise HTTPException(400, f"{range_label}沒有對話記錄。資料庫實際共有 {total_all} 筆，請選更長的時間範圍。")

    # ── 依 session 分組並區分：客戶 / Bot / 員工真人代回 ──
    from collections import OrderedDict
    import re as _re
    sessions: OrderedDict = OrderedDict()
    human_examples = []          # 員工真人回覆範例（代回）
    for c in convs:
        sid = c.get("session_id") or "unknown"
        sessions.setdefault(sid, []).append(c)
        q = str(c.get("question", "")).strip()
        if q.startswith("（真人"):
            a = str(c.get("answer", "")).strip()
            if a:
                human_examples.append(a[:200])

    total_sessions = len(sessions)
    human_reply_count = len(human_examples)

    # ── 組對話文本（最多 40 個 session）──
    session_blocks = []
    for idx, (sid, msgs) in enumerate(list(sessions.items())[-40:], 1):
        short_id = sid[-8:] if len(sid) > 8 else sid
        turns = []
        for m in msgs:
            q = str(m.get("question", "")).strip()[:200]
            a = str(m.get("answer", "")).strip()[:200]
            a = _re.sub(r'DATA_(?:SAVE|PARTIAL):\s*\{.*?\}', '', a, flags=_re.DOTALL).strip()
            if q.startswith("（真人"):
                if a:
                    turns.append(f"  員工真人回覆：{a}")
            else:
                if q:
                    turns.append(f"  客戶：{q}")
                if a:
                    turns.append(f"  Bot：{a}")
        if turns:
            session_blocks.append(f"【客戶 #{idx}（{short_id}）】\n" + "\n".join(turns))
    conversation_text = "\n\n---\n\n".join(session_blocks)

    human_block = ""
    if human_examples:
        picked = human_examples[-20:]
        human_block = "\n\n=== 員工真人回覆範例（這是你們客服實際說的話，請優先當作語氣範本）===\n" + \
            "\n".join(f"・{h}" for h in picked)
    role_context = f"\n\n【Bot 目前的角色設定（供參考）】\n{system_prompt[:300]}" if system_prompt.strip() else ""

    prompt = f"""你是一位資深客服話術與品牌語氣顧問。以下是「{bot_name}」這個 AI 客服 Bot 最近的對話記錄。請專注分析「語氣、口吻、遣詞用字」，不是分析流程或完成率。{role_context}{human_block}

=== 對話記錄 ===

{conversation_text}

=== 分析任務 ===

請用繁體中文輸出一份「語氣風格指南」，重點在回覆的方式與用字遣詞。格式如下：

## 客戶怎麼說話
- 歸納客戶的語氣特徵（正式/口語、有沒有情緒、常用詞、常見稱呼與問法），引用 2-3 句真實例子。

## 員工真人回覆的風格
- {'根據上面「員工真人回覆範例」，歸納你們客服的語氣特色、常用開頭/結尾、用字習慣、貼心之處，引用實際句子。' if human_examples else '（目前沒有員工真人代回的資料，這段請說明「尚無真人範例可學習」，並改用一般優質客服的語氣建議。）'}

## 目前 Bot 回覆的語氣問題
- 指出 Bot 現在回覆哪裡太生硬、太官方、太冗長、或不夠貼近客戶，引用實際句子佐證。

## 建議的回覆風格指南
給出可直接照做的準則，越具體越好：
- **語氣定位**：一句話定調（例如：親切但專業、像鄰家店員）
- **常用開頭 / 結尾**：列 2-3 個範例句
- **建議用詞**：哪些詞多用（列出來）
- **避免用詞**：哪些字眼少用或別用（列出來）
- **句子長度 / emoji / 標點**：具體建議
- **改寫示範**：挑 2-3 句現在 Bot 說得不好的話，示範「原本 →建議」的改寫

## 可貼進設定的風格段落
最後產出一段可直接貼進 Bot system_prompt 的「語氣與話術規範」文字（3-6 句、祈使句、明確可執行），讓 Bot 之後照這個口吻回覆。

只輸出報告本身，不要前言後記。"""

    try:
        from google import genai
        from google.genai import types
        import time as _time

        client = genai.Client(api_key=api_key)
        last_err = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(
                        max_output_tokens=6000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                parts = response.candidates[0].content.parts if response.candidates else []
                report = "".join(p.text for p in parts if hasattr(p, "text") and not getattr(p, "thought", False))
                return {
                    "report": report.strip(),
                    "stats": {
                        "total_sessions":    total_sessions,
                        "human_reply_count": human_reply_count,
                        "total_messages":    len(convs),
                    }
                }
            except Exception as e:
                last_err = e
                if "503" in str(e) or "overloaded" in str(e).lower():
                    _time.sleep(3 * (attempt + 1))
                else:
                    raise
        raise last_err
    except Exception as e:
        raise HTTPException(500, f"語氣分析失敗：{str(e)[:200]}")


class MuteRequest(BaseModel):
    session_id: str


def _line_user_id_from_session(bot_id: str, session_id: str) -> str:
    """從 LINE session_id 取出 line_user_id；非 LINE session 回傳空字串。"""
    prefix = f"line_{bot_id}_"
    if session_id.startswith(prefix):
        return session_id[len(prefix):]
    return ""


@app.post("/bots/{bot_id}/mute")
async def mute_chat(
    bot_id: str,
    body: MuteRequest,
    authorization: Optional[str] = Header(None)
):
    """真人接手：靜音某個 LINE 聊天室，AI 停止自動回覆（員工操作）。"""
    access = require_bot_access(bot_id, authorization, min_role="editor")
    line_user_id = _line_user_id_from_session(bot_id, body.session_id)
    if not line_user_id:
        raise HTTPException(400, "只能對 LINE 對話切換手動接手")
    add_mute(bot_id, line_user_id, muted_by=access["app_user"]["id"])
    return {"ok": True, "muted": True}


@app.delete("/bots/{bot_id}/mute")
async def unmute_chat(
    bot_id: str,
    session_id: str = Query(...),
    authorization: Optional[str] = Header(None)
):
    """恢復 AI：取消某個 LINE 聊天室的靜音，AI 重新自動回覆（員工操作）。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    line_user_id = _line_user_id_from_session(bot_id, session_id)
    if not line_user_id:
        raise HTTPException(400, "只能對 LINE 對話切換手動接手")
    remove_mute(bot_id, line_user_id)
    return {"ok": True, "muted": False}


class ReplyRequest(BaseModel):
    session_id: str
    text: str


@app.post("/bots/{bot_id}/reply")
async def reply_chat(
    bot_id: str,
    body: ReplyRequest,
    authorization: Optional[str] = Header(None)
):
    """真人代回：從後台對話窗直接把訊息推給 LINE 客戶，並自動接手（AI 靜音）＋記錄。"""
    access = require_bot_access(bot_id, authorization, min_role="editor")
    line_user_id = _line_user_id_from_session(bot_id, body.session_id)
    if not line_user_id:
        raise HTTPException(400, "只能對 LINE 對話代回")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "訊息不能為空")

    token = _get_bot_config(bot_id).get("line_channel_access_token")
    if not token:
        raise HTTPException(400, "此 Bot 尚未設定 LINE，無法代回")

    # 直接呼叫 LINE push，並把錯誤原因帶回前端（方便診斷：配額用盡、token 失效、非好友等）
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": line_user_id, "messages": [{"type": "text", "text": text}]},
            timeout=15,
        )
    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except Exception:
            detail = (resp.text or "")[:200]
        logging.warning(f"[reply] push failed {resp.status_code} to {line_user_id[:8]}…: {detail}")
        raise HTTPException(502, f"LINE 傳送失敗（{resp.status_code}）：{detail or '請稍後再試'}")
    logging.info(f"[reply] pushed ok to {line_user_id[:8]}… by {access['app_user'].get('email','')}")

    add_mute(bot_id, line_user_id, muted_by=access["app_user"]["id"])
    staff_name = access["app_user"].get("display_name") or access["app_user"].get("email") or ""
    try:
        supabase.table("conversations").insert({
            "bot_id": bot_id,
            "question": f"（真人 {staff_name} 代回）",
            "answer": text,
            "session_id": body.session_id,
        }).execute()
    except Exception as e:
        logging.warning(f"[reply] conversations insert failed: {e}")
    return {"ok": True, "muted": True}


@app.get("/bots/{bot_id}/submissions")
async def list_submissions(
    bot_id: str,
    days: int = 0,
    authorization: Optional[str] = Header(None)
):
    """列出 bot 收集完成的客戶名單（DATA_SAVE 存下的資料卡，可複製）。"""
    require_bot_access(bot_id, authorization, min_role="viewer")
    query = supabase.table("submissions")\
        .select("id, session_id, display_name, data, card_text, handled, created_at")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)
    if days > 0:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = query.gte("created_at", since)
    rows = query.limit(500).execute()
    return {"submissions": rows.data or []}


@app.get("/bots/{bot_id}/submission-fields")
async def list_submission_fields(
    bot_id: str,
    authorization: Optional[str] = Header(None)
):
    """回傳這個 bot 最近幾筆 submission 實際出現過的欄位 key，給範本編輯器當可點按鈕。"""
    require_bot_access(bot_id, authorization, min_role="viewer")
    rows = supabase.table("submissions")\
        .select("data")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)\
        .limit(20).execute()
    # 依「最近先出現」順序收集欄位，保留第一次見到的排序、去重
    fields: list[str] = []
    seen: set = set()
    for r in (rows.data or []):
        data = r.get("data") or {}
        if not isinstance(data, dict):
            continue
        for k in data.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    return {"fields": fields}


class SubmissionHandledRequest(BaseModel):
    handled: bool


@app.patch("/bots/{bot_id}/submissions/{submission_id}")
async def update_submission_handled(
    bot_id: str,
    submission_id: str,
    body: SubmissionHandledRequest,
    authorization: Optional[str] = Header(None)
):
    """標記客戶名單某筆為已處理／未處理。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    supabase.table("submissions").update({"handled": body.handled})\
        .eq("id", submission_id).eq("bot_id", bot_id).execute()
    return {"ok": True, "handled": body.handled}


@app.delete("/bots/{bot_id}/submissions/{submission_id}")
async def delete_submission(
    bot_id: str,
    submission_id: str,
    authorization: Optional[str] = Header(None)
):
    """刪除一筆客戶名單資料。"""
    require_bot_access(bot_id, authorization, min_role="editor")
    supabase.table("submissions").delete()\
        .eq("id", submission_id).eq("bot_id", bot_id).execute()
    return {"ok": True}


@app.get("/bots/{bot_id}/conversations/sessions")
async def list_conversation_sessions(
    bot_id: str,
    days: int = 0,
    authorization: Optional[str] = Header(None)
):
    """列出 bot 的對話 session（給「查看對話紀錄」用），依 session_id 分組回傳。"""
    require_bot_access(bot_id, authorization, min_role="viewer")
    # 取「最新」的訊息：先 desc 抓最近 1000 筆，之後每個 session 內再改回時間正序顯示。
    # （若用 asc + limit 會拿到最舊的資料、把最新的回覆截掉）
    query = supabase.table("conversations")\
        .select("question, answer, session_id, created_at")\
        .eq("bot_id", bot_id)\
        .order("created_at", desc=True)
    if days > 0:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = query.gte("created_at", since)
    rows = query.limit(1000).execute()
    convs = rows.data or []

    from collections import OrderedDict
    sessions: OrderedDict = OrderedDict()
    for c in convs:
        sid = c.get("session_id") or "unknown"
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(c)
    # 每個 session 內改回時間正序（由舊到新），讓對話讀起來順、first/last_at 正確
    for sid in sessions:
        sessions[sid].sort(key=lambda m: m.get("created_at") or "")

    out = []
    for sid, msgs in sessions.items():
        if sid.startswith("line_"):
            channel = "LINE"
        elif sid.startswith("widget_") or sid.startswith("test_"):
            channel = "網頁/測試"
        elif sid.startswith("ig_cmt_"):
            channel = "IG 留言"
        elif sid.startswith("ig_"):
            channel = "IG"
        elif sid.startswith("assistant_"):
            channel = "設定助手"
        else:
            channel = "其他"
        is_done = any("DATA_SAVE" in str(m.get("answer", "")) for m in msgs)
        line_uid = _line_user_id_from_session(bot_id, sid)
        muted = bool(line_uid) and _mute_key(bot_id, line_uid) in _muted_line_users
        display_name = _line_profile_cache.get(_mute_key(bot_id, line_uid), "") if line_uid else ""
        out.append({
            "session_id":   sid,
            "channel":      channel,
            "message_count": len(msgs),
            "first_at":     msgs[0].get("created_at"),
            "last_at":      msgs[-1].get("created_at"),
            "completed":    is_done,
            "can_mute":     bool(line_uid),
            "muted":        muted,
            "line_user_id": line_uid,
            "display_name": display_name,
            "messages": [
                {
                    "q": str(m.get("question", ""))[:500],
                    "a": str(m.get("answer", ""))[:500],
                    "at": m.get("created_at"),
                }
                for m in msgs
            ],
        })

    # 補抓缺少暱稱的 LINE 對話（並行，寫回快取）
    missing = [s for s in out if s["line_user_id"] and not s["display_name"]]
    if missing:
        line_token = _get_bot_config(bot_id).get("line_channel_access_token")
        if line_token:
            names = await asyncio.gather(*[
                fetch_line_display_name(bot_id, s["line_user_id"], line_token) for s in missing
            ])
            for s, name in zip(missing, names):
                s["display_name"] = name

    # 最新的在前
    out.sort(key=lambda s: s["last_at"] or "", reverse=True)
    return {"total_sessions": len(out), "total_messages": len(convs), "sessions": out}


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
        card_template  = bot.get("card_template") or None
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
                card_template=card_template,
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
        card_template    = bot.get("card_template") or None
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
                card_template=card_template,
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

    admin_email = os.getenv("ADMIN_EMAIL", "")
    is_admin = bool(admin_email) and email.lower() == admin_email.lower()

    return {
        "email":      email,
        "created_at": created_at,
        "plan":       "paid" if slots > 0 else "free",
        "bot_slots":  slots,
        "bots_used":  bots_used,
        "renews_at":  renews_at,
        "is_admin":   is_admin,
    }


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

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")

def require_admin(authorization: str = None) -> str:
    if not ADMIN_EMAIL:
        raise HTTPException(503, "Admin not configured")
    user_id = get_user_id(authorization)
    user_info = _admin.auth.admin.get_user_by_id(user_id)
    if not user_info.user or user_info.user.email != ADMIN_EMAIL:
        raise HTTPException(403, "無權限")
    return user_id


def _admin_collect_users() -> list:
    """組出後台用戶清單，資料來源改用我們自己的 app_users 表（快、穩），
    不再依賴慢又常逾時的 Supabase Auth list_users()。每一步都防禦性處理，絕不 500。"""
    # app_users：我們自己維護的使用者表（含 email / supabase_uid / 顯示名稱）
    try:
        au_rows = supabase.table("app_users").select(
            "id, email, supabase_uid, display_name, created_at, admin_note"
        ).execute().data or []
    except Exception as e:
        logging.warning(f"[Admin] app_users query failed: {e}")
        au_rows = []

    # bot_subscriptions：每個 supabase uid 的有效名額 + 最近到期日
    slots_map: dict = {}
    renews_map: dict = {}
    try:
        subs_rows = supabase.table("bot_subscriptions").select(
            "user_id, slots, status, renews_at"
        ).eq("status", "active").execute().data or []
        for r in subs_rows:
            if not _sub_is_valid(r):   # 過期的不算，跟付費牆一致
                continue
            uid = r["user_id"]
            slots_map[uid] = slots_map.get(uid, 0) + (r.get("slots") or 1)
            if not renews_map.get(uid):
                renews_map[uid] = r.get("renews_at")
    except Exception as e:
        logging.warning(f"[Admin] subscriptions query failed: {e}")

    # bots：每個 supabase uid 開了幾隻
    bot_count: dict = {}
    try:
        for b in (supabase.table("bots").select("user_id").execute().data or []):
            uid = b.get("user_id")
            if uid:
                bot_count[uid] = bot_count.get(uid, 0) + 1
    except Exception as e:
        logging.warning(f"[Admin] bots query failed: {e}")

    result = []
    for u in au_rows:
        suid = u.get("supabase_uid")
        # 名額/bot 數以 supabase uid 為 key（訂閱與 bot 都存 supabase uid）
        slots = slots_map.get(suid, 0) if suid else 0
        result.append({
            "user_id":    suid or u.get("id"),   # 給前端做延長/授權用（優先 supabase uid）
            "email":      u.get("email") or u.get("display_name") or "—",
            "created_at": str(u.get("created_at") or ""),
            "bot_slots":  slots,
            "max_bots":   1 + slots,
            "bots_used":  bot_count.get(suid, 0) if suid else 0,
            "renews_at":  renews_map.get(suid) if suid else None,
            "admin_note": u.get("admin_note") or "",
        })

    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


@app.get("/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    users = _admin_collect_users()
    total_bots = 0
    try:
        total_bots = supabase.table("bots").select("id", count="exact").execute().count or 0
    except Exception:
        pass
    paid_users  = sum(1 for u in users if u["bot_slots"] > 0)
    total_slots = sum(u["bot_slots"] for u in users)
    return {
        "total_users": len(users),
        "total_bots":  total_bots,
        "paid_users":  paid_users,
        "total_slots": total_slots,
    }


@app.get("/admin/assistant-engine")
async def admin_assistant_engine(authorization: Optional[str] = Header(None)):
    """診斷小懶目前實際走哪個引擎：回報後端有沒有讀到 PLATFORM_ANTHROPIC_KEY，
    並對 Claude 做一次最小 ping，把真正的錯誤吐出來（唯讀，不改任何資料）。"""
    require_admin(authorization)
    from app.config import PLATFORM_ANTHROPIC_KEY, ASSISTANT_CLAUDE_MODEL
    key = PLATFORM_ANTHROPIC_KEY or ""
    out = {
        "key_present": bool(key),
        "key_prefix":  (key[:7] + "…" + key[-4:]) if len(key) > 12 else ("(空)" if not key else "(太短)"),
        "model":       ASSISTANT_CLAUDE_MODEL,
        "engine":      "claude" if key else "gemini",
        "ping_ok":     None,
        "ping_error":  None,
    }
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            resp = client.messages.create(
                model=ASSISTANT_CLAUDE_MODEL,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
            out["ping_ok"] = True
            out["ping_reply"] = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )[:40]
        except Exception as e:
            out["ping_ok"] = False
            out["ping_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


@app.get("/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    return _admin_collect_users()


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


class AdminExtendRequest(BaseModel):
    days: int = 30
    slots: int = 1  # 若目前沒有任何有效名額，開通時給幾個名額


@app.post("/admin/users/{target_user_id}/extend")
async def admin_extend_subscription(
    target_user_id: str,
    body: AdminExtendRequest,
    authorization: Optional[str] = Header(None),
):
    """手動幫租客延長訂閱 N 天（收到轉帳後用）。從現有到期日往後加，
    到期日已過或沒有則從今天算起。同時清空 warned_at 讓下一輪能再提醒。"""
    require_admin(authorization)
    days = max(1, min(body.days, 3650))
    sub_id = f"admin_{target_user_id}"
    now = datetime.utcnow()

    # 找現有的手動訂閱，取其到期日當基準
    base = now
    try:
        existing = supabase.table("bot_subscriptions").select("renews_at, slots").eq("id", sub_id).execute()
        cur_slots = body.slots
        if existing.data:
            cur_slots = existing.data[0].get("slots") or body.slots
            cur_renews = existing.data[0].get("renews_at")
            if cur_renews:
                try:
                    exp = datetime.fromisoformat(str(cur_renews).replace("Z", "+00:00"))
                    if exp.tzinfo:
                        exp = exp.replace(tzinfo=None)
                    if exp > now:
                        base = exp  # 還沒過期 → 從現有到期日往後疊加
                except Exception:
                    pass
        new_renews = (base + timedelta(days=days)).isoformat()
        supabase.table("bot_subscriptions").upsert({
            "id":         sub_id,
            "user_id":    target_user_id,
            "status":     "active",
            "slots":      max(1, cur_slots),
            "renews_at":  new_renews,
            "warned_at":  None,
        }, on_conflict="id").execute()
    except Exception as e:
        raise HTTPException(500, f"DB 寫入失敗：{str(e)}")
    return {"ok": True, "user_id": target_user_id, "renews_at": new_renews}


@app.get("/admin/orgs-map")
async def admin_orgs_map(authorization: Optional[str] = Header(None)):
    """唯讀診斷：列出所有團隊、擁有者、成員，用來釐清「同一人多帳號 / 成員跑錯團隊」。"""
    require_admin(authorization)
    users = supabase.table("app_users").select(
        "id, email, display_name, line_user_id"
    ).execute().data or []
    umap = {u["id"]: u for u in users}
    orgs = supabase.table("organizations").select("id, name, owner_id").execute().data or []
    mships = supabase.table("memberships").select("org_id, user_id, role").execute().data or []
    by_org: dict = {}
    for m in mships:
        by_org.setdefault(m["org_id"], []).append(m)

    def _label(u: dict) -> str:
        return (u.get("email") or u.get("display_name") or "—")

    result = []
    for o in orgs:
        owner = umap.get(o["owner_id"], {})
        members = []
        for m in by_org.get(o["id"], []):
            mu = umap.get(m["user_id"], {})
            members.append({"role": m["role"], "label": _label(mu)})
        members.sort(key=lambda x: (x["role"] != "owner", x["label"]))
        result.append({
            "org_id":      o["id"],
            "org_name":    o["name"],
            "owner_label": _label(owner),
            "member_count": len(members),
            "members":     members,
        })
    result.sort(key=lambda x: (-x["member_count"], x["owner_label"]))
    return result


class AdminNoteUpdate(BaseModel):
    note: str = ""


def _find_app_user_by_target(target_user_id: str) -> Optional[dict]:
    """後台清單的 user_id 優先是 supabase uid，找不到再退回 app_users.id。"""
    try:
        r = supabase.table("app_users").select("*").eq("supabase_uid", target_user_id).execute()
        if r.data:
            return r.data[0]
    except Exception:
        pass
    try:
        r = supabase.table("app_users").select("*").eq("id", target_user_id).execute()
        if r.data:
            return r.data[0]
    except Exception:
        pass
    return None


@app.post("/admin/users/{target_user_id}/note")
async def admin_set_note(
    target_user_id: str,
    body: AdminNoteUpdate,
    authorization: Optional[str] = Header(None),
):
    """幫某位使用者加/改備注，只存在 app_users.admin_note，純內部用。"""
    require_admin(authorization)
    au = _find_app_user_by_target(target_user_id)
    if not au:
        raise HTTPException(404, "找不到這位使用者")
    note = (body.note or "")[:500]
    try:
        supabase.table("app_users").update({"admin_note": note}).eq("id", au["id"]).execute()
    except Exception as e:
        raise HTTPException(500, f"DB 寫入失敗：{str(e)}")
    return {"ok": True, "user_id": target_user_id, "note": note}


@app.delete("/admin/users/{target_user_id}")
async def admin_delete_user(
    target_user_id: str,
    authorization: Optional[str] = Header(None),
):
    """徹底刪除一位使用者（清測試帳號用）。連帶刪除：他開的 bots（含 knowledge_chunks、
    conversations）、bot_subscriptions、memberships、他擁有的 organizations（含底下 memberships、
    invites），最後刪 app_users 與 Supabase auth 帳號。防呆：不能刪自己 / 管理員帳號。"""
    admin_uid = require_admin(authorization)

    au = _find_app_user_by_target(target_user_id)
    if not au:
        raise HTTPException(404, "找不到這位使用者")

    suid = au.get("supabase_uid")
    app_user_id = au["id"]

    # 防呆：不能刪自己，也不能刪管理員帳號
    if suid and suid == admin_uid:
        raise HTTPException(400, "不能刪除自己")
    if (au.get("email") or "").lower() == (ADMIN_EMAIL or "").lower():
        raise HTTPException(400, "不能刪除管理員帳號")

    deleted = {"bots": 0, "orgs": 0}

    # 1) 刪他開的 bots（bots.user_id = supabase uid），連同知識庫/對話
    if suid:
        try:
            bots = supabase.table("bots").select("id").eq("user_id", suid).execute().data or []
            for b in bots:
                bid = b["id"]
                supabase.table("knowledge_chunks").delete().eq("bot_id", bid).execute()
                supabase.table("conversations").delete().eq("bot_id", bid).execute()
                supabase.table("bots").delete().eq("id", bid).execute()
                deleted["bots"] += 1
        except Exception as e:
            logging.warning(f"[Admin Delete] bots cleanup failed: {e}")

        # 2) 刪訂閱（含手動 admin_ 前綴那筆）
        try:
            supabase.table("bot_subscriptions").delete().eq("user_id", suid).execute()
        except Exception as e:
            logging.warning(f"[Admin Delete] subscriptions cleanup failed: {e}")

    # 3) 刪他擁有的團隊（organizations.owner_id = app_user.id），連同底下 memberships / invites
    try:
        orgs = supabase.table("organizations").select("id").eq("owner_id", app_user_id).execute().data or []
        for o in orgs:
            oid = o["id"]
            supabase.table("memberships").delete().eq("org_id", oid).execute()
            supabase.table("invites").delete().eq("org_id", oid).execute()
            supabase.table("organizations").delete().eq("id", oid).execute()
            deleted["orgs"] += 1
    except Exception as e:
        logging.warning(f"[Admin Delete] orgs cleanup failed: {e}")

    # 4) 刪他在別人團隊裡的 membership
    try:
        supabase.table("memberships").delete().eq("user_id", app_user_id).execute()
    except Exception as e:
        logging.warning(f"[Admin Delete] memberships cleanup failed: {e}")

    # 4.5) 刪 LINE 綁定資料（staff_line / line_binding_codes），避免留下孤兒綁定
    #      讓管理助手仍把這支 LINE 認成已綁定的員工
    try:
        supabase.table("staff_line").delete().eq("app_user_id", app_user_id).execute()
        line_uid = au.get("line_user_id")
        if line_uid:
            supabase.table("staff_line").delete().eq("line_user_id", line_uid).execute()
        supabase.table("line_binding_codes").delete().eq("app_user_id", app_user_id).execute()
    except Exception as e:
        logging.warning(f"[Admin Delete] line binding cleanup failed: {e}")

    # 5) 刪 app_users 本身
    try:
        supabase.table("app_users").delete().eq("id", app_user_id).execute()
    except Exception as e:
        raise HTTPException(500, f"刪除 app_user 失敗：{str(e)}")

    # 6) 刪 Supabase auth 帳號（有 supabase uid 才有）
    if suid:
        try:
            _admin.auth.admin.delete_user(suid)
        except Exception as e:
            logging.warning(f"[Admin Delete] auth delete failed for {suid[:8]}: {e}")

    return {"ok": True, "deleted_user": target_user_id, **deleted}


# ──────────────────────────────────────
# 訂閱到期提醒排程（到期前 7 天用管理助手推 LINE）
# ──────────────────────────────────────

def _resolve_tenant_line_id(supabase_uid: str) -> Optional[str]:
    """由訂閱的 user_id(supabase uid) 找出可用管理助手推播的 LINE userId。
    優先用 staff_line（確定綁過管理助手），其次 app_users.line_user_id。"""
    try:
        au = supabase.table("app_users").select("id, line_user_id").eq("supabase_uid", supabase_uid).execute()
        if not au.data:
            return None
        app_user_id = au.data[0]["id"]
        sl = supabase.table("staff_line").select("line_user_id").eq("app_user_id", app_user_id).execute()
        if sl.data and sl.data[0].get("line_user_id"):
            return sl.data[0]["line_user_id"]
        return au.data[0].get("line_user_id")
    except Exception:
        return None


async def _expiry_warning_tick():
    """掃描 7 天內到期、還沒提醒過的手動訂閱，用管理助手推 LINE 提醒。"""
    if not ADMIN_LINE_CHANNEL_ACCESS_TOKEN:
        return
    now = datetime.utcnow()
    try:
        rows = supabase.table("bot_subscriptions").select(
            "id, user_id, renews_at, warned_at, status"
        ).eq("status", "active").execute()
    except Exception as e:
        logging.warning(f"[Expiry] query failed: {e}")
        return

    for sub in (rows.data or []):
        renews_at = sub.get("renews_at")
        if not renews_at or sub.get("warned_at"):
            continue
        try:
            exp = datetime.fromisoformat(str(renews_at).replace("Z", "+00:00"))
            if exp.tzinfo:
                exp = exp.replace(tzinfo=None)
        except Exception:
            continue
        days_left = (exp - now).days
        # 只在「未過期」且「剩 7 天內」時提醒（過期就靠付費牆停用，不再提醒）
        if not (0 <= days_left <= 7):
            continue
        line_id = _resolve_tenant_line_id(sub["user_id"])
        if not line_id:
            continue
        msg = (
            f"⏰ 訂閱即將到期提醒\n\n"
            f"您的「懶得回」AI 客服服務將於 {exp.strftime('%Y/%m/%d')} 到期"
            f"（剩 {days_left} 天）。\n\n"
            f"到期後 Bot 將暫停自動回覆，請記得續約以免影響客戶服務 🙏"
        )
        try:
            code = await _admin_push(line_id, [_admin_text_msg(msg)])
            if code == 200:
                supabase.table("bot_subscriptions").update(
                    {"warned_at": now.isoformat()}
                ).eq("id", sub["id"]).execute()
                logging.info(f"[Expiry] warned user={sub['user_id'][:8]} days_left={days_left}")
        except Exception as e:
            logging.warning(f"[Expiry] push failed for {sub['id']}: {e}")


_daily_summary_date: Optional[str] = None  # 已發送摘要的台北日期 YYYY-MM-DD


def _build_daily_summary_for_user(app_user: dict) -> Optional[str]:
    """算某位使用者所屬團隊今日（台北）的營運摘要文字；沒對話回 None。"""
    tpe = datetime.utcnow() + timedelta(hours=8)
    today = tpe.strftime("%Y-%m-%d")
    start_utc = (datetime.strptime(today, "%Y-%m-%d") - timedelta(hours=8)).isoformat()
    org_ids = get_user_org_ids(app_user["id"])
    if not org_ids:
        return None
    try:
        bots = supabase.table("bots").select("id").in_("org_id", org_ids).execute().data or []
        bot_ids = [b["id"] for b in bots]
        if not bot_ids:
            return None
        rows = supabase.table("conversations").select("session_id, question") \
            .in_("bot_id", bot_ids).gte("created_at", start_utc).execute().data or []
    except Exception as e:
        logging.warning(f"[Summary] manual stats failed: {e}")
        return None
    total = len(rows)
    if total == 0:
        return None
    customers = len({r.get("session_id") for r in rows if r.get("session_id")})
    handoffs = sum(1 for r in rows if "代回" in (r.get("question") or ""))
    return (
        f"📊 今日營運摘要（{today}）\n\n"
        f"👥 客戶數：{customers} 位\n"
        f"💬 對話則數：{total} 則\n"
        f"🙋 真人代回：{handoffs} 則\n\n"
        f"辛苦了，明天繼續加油 💪"
    )


async def _daily_summary_tick():
    """台北時間 21 點，推當日營運摘要給各團隊 owner（一天一次）。"""
    global _daily_summary_date
    if not ADMIN_LINE_CHANNEL_ACCESS_TOKEN:
        return
    tpe = datetime.utcnow() + timedelta(hours=8)
    today = tpe.strftime("%Y-%m-%d")
    if tpe.hour != 21 or _daily_summary_date == today:
        return
    _daily_summary_date = today
    # 今日台北 00:00 換算成 UTC 字串，跟 conversations.created_at 比對
    start_utc = (datetime.strptime(today, "%Y-%m-%d") - timedelta(hours=8)).isoformat()
    try:
        orgs = supabase.table("organizations").select("id, name, owner_id").execute().data or []
    except Exception as e:
        logging.warning(f"[Summary] orgs query failed: {e}")
        return
    for org in orgs:
        owner_id = org.get("owner_id")
        if not owner_id:
            continue
        line_id = _line_id_for_app_user(owner_id)
        if not line_id:
            continue
        try:
            bots = supabase.table("bots").select("id").eq("org_id", org["id"]).execute().data or []
            bot_ids = [b["id"] for b in bots]
            if not bot_ids:
                continue
            rows = supabase.table("conversations").select("session_id, question") \
                .in_("bot_id", bot_ids).gte("created_at", start_utc).execute().data or []
        except Exception as e:
            logging.warning(f"[Summary] stats failed org={org['id']}: {e}")
            continue
        total = len(rows)
        if total == 0:
            continue
        customers = len({r.get("session_id") for r in rows if r.get("session_id")})
        handoffs = sum(1 for r in rows if "代回" in (r.get("question") or ""))
        msg = (
            f"📊 今日營運摘要（{today}）\n\n"
            f"👥 客戶數：{customers} 位\n"
            f"💬 對話則數：{total} 則\n"
            f"🙋 真人代回：{handoffs} 則\n\n"
            f"辛苦了，明天繼續加油 💪"
        )
        try:
            await _admin_push(line_id, [_admin_text_msg(msg)])
        except Exception as e:
            logging.warning(f"[Summary] push failed org={org['id']}: {e}")


async def _expiry_scheduler_loop():
    """每小時檢查一次訂閱到期 + 每日營運摘要。"""
    while True:
        try:
            await _expiry_warning_tick()
        except Exception as e:
            logging.warning(f"[Expiry] tick error: {e}")
        try:
            await _daily_summary_tick()
        except Exception as e:
            logging.warning(f"[Summary] tick error: {e}")
        await asyncio.sleep(3600)
