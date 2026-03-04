"""
Session 持久化層（write-through cache）

架構：
  讀取：memory cache 優先 → cache miss 才查 Supabase
  寫入：同時寫 memory + Supabase
  重啟後：memory 空了，第一次對話從 Supabase 補回

Supabase 需要先建這張表（在 SQL Editor 執行一次）：
  CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    status       TEXT DEFAULT 'active',
    history      JSONB DEFAULT '[]',
    metadata     JSONB DEFAULT '{}',
    last_interaction TIMESTAMPTZ DEFAULT NOW(),
    created_at   TIMESTAMPTZ DEFAULT NOW()
  );
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

_sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# write-through memory cache
_cache: dict = {}

MAX_HISTORY = 40          # 最多保留 40 則（20 輪對話），超過自動截斷
TTL_DAYS    = 3           # session 超過 3 天沒互動 → 自動過期


# ──────────────────────────────────────
# 內部工具
# ──────────────────────────────────────

def _is_expired(last_interaction_iso: str) -> bool:
    try:
        last = datetime.fromisoformat(last_interaction_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - last) > timedelta(days=TTL_DAYS)
    except Exception:
        return False


def _trim_history(session: dict) -> dict:
    h = session.get("history", [])
    if len(h) > MAX_HISTORY:
        session["history"] = h[-MAX_HISTORY:]
    return session


# ──────────────────────────────────────
# 公開介面
# ──────────────────────────────────────

def get_session(session_id: str) -> Optional[dict]:
    """從 cache 或 Supabase 讀取 session；過期或不存在回傳 None"""
    # 1. cache
    if session_id in _cache:
        s = _cache[session_id]
        age = time.time() - s.get("_ts", 0)
        if age < TTL_DAYS * 86400:
            return s
        del _cache[session_id]

    # 2. Supabase
    try:
        res = _sb.table("sessions").select("*").eq("session_id", session_id).execute()
        if res.data:
            row = res.data[0]
            if row.get("last_interaction") and _is_expired(row["last_interaction"]):
                # TTL 過期 → 刪掉
                _sb.table("sessions").delete().eq("session_id", session_id).execute()
                logging.info(f"[SessionStore] TTL expired, deleted {session_id[:8]}")
                return None

            session = {
                "history":  row.get("history") or [],
                "status":   row.get("status", "active"),
                "_ts":      time.time(),
                **(row.get("metadata") or {}),   # step, done, fields, collected 等還原
            }
            _cache[session_id] = session
            return session
    except Exception as e:
        logging.warning(f"[SessionStore] get failed {session_id[:8]}: {e}")

    return None


def get_or_create(session_id: str) -> dict:
    """取得或建立 session（新 session 預設 active）"""
    session = get_session(session_id)
    if session is None:
        session = {"history": [], "status": "active", "_ts": time.time()}
        _cache[session_id] = session
    return session


def save(session_id: str, session: dict):
    """寫入 cache + Supabase（異步安全，失敗只 warning 不 crash）"""
    _trim_history(session)
    session["_ts"] = time.time()
    _cache[session_id] = session

    # 整理 metadata（非 history/status/_ts 的欄位）
    meta_keys = [k for k in session if k not in ("history", "status", "_ts")]
    metadata = {k: session[k] for k in meta_keys}

    try:
        _sb.table("sessions").upsert({
            "session_id":        session_id,
            "status":            session.get("status", "active"),
            "history":           session.get("history", []),
            "metadata":          metadata,
            "last_interaction":  datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logging.warning(f"[SessionStore] save failed {session_id[:8]}: {e}")


def delete(session_id: str):
    """刪除 session（重置對話用）"""
    _cache.pop(session_id, None)
    try:
        _sb.table("sessions").delete().eq("session_id", session_id).execute()
    except Exception as e:
        logging.warning(f"[SessionStore] delete failed {session_id[:8]}: {e}")


def get_status(session_id: str) -> str:
    """快速查 session 狀態，不存在回傳 'new'"""
    s = get_session(session_id)
    return s.get("status", "active") if s else "new"
