"""
重送客戶單到合作方(Sharing)—— 用於對方「來源已開通」後補送先前被擋掉的單。

重送安全：對方用 ref(=session_id) 去重，重複送不會變成兩張單。
payload 組裝邏輯與後端 _build_lead_payload 一致（含 conversation 問答紀錄）。

用法（需正式環境 SUPABASE_URL / SUPABASE_KEY，例如 railway run）：
    # 1) 先乾跑（預設）：檢查網址/金鑰、列出會重送哪些單、看 payload，但「不」真的送
    python3 scripts/resend_to_sharing.py --name 697 --days 3
    # 2) 確認無誤、對方也開通後，加 --send 真的送出
    python3 scripts/resend_to_sharing.py --name 697 --days 3 --send

參數：
    --bot <id>    指定 bot_id
    --name <字>   用 bot 名稱關鍵字找（例如 697）；--bot 優先
    --days N      往前幾天的完成單（預設 3）
    --send        真的 POST 送出（不加＝只檢查與預覽，不送）
    --session <id> 只重送指定 session_id（可給多個，逗號分隔）
"""
import os
import re
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import SUPABASE_URL, SUPABASE_KEY  # noqa: E402
from supabase import create_client  # noqa: E402
import httpx  # noqa: E402

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
TW = timezone(timedelta(hours=8))

_PHONE_KEYS = ["電話", "手機", "聯絡電話", "電話號碼", "手機號碼", "phone", "mobile", "tel"]
_LINE_KEYS = ["lineid", "line id", "line", "line帳號", "賴", "line_id"]
_NAME_KEYS = ["姓名", "名字", "稱呼", "客戶姓名", "name", "fullname"]
_MSG_KEYS = ["訊息", "message", "諮詢", "諮詢內容", "需求", "用途", "備註", "note"]
_PRIORITY = ["需求金額", "地區", "工作", "用途"]


def _pick(data, keys):
    if not data:
        return ""
    norm = {str(k).strip().lower().replace(" ", "").replace("　", ""): v for k, v in data.items()}
    for key in keys:
        kk = key.lower().replace(" ", "")
        if kk in norm and str(norm[kk]).strip():
            return str(norm[kk]).strip()
    return ""


def _phone(data):
    raw = _pick(data, _PHONE_KEYS)
    d = re.sub(r"\D", "", raw)
    return d if re.fullmatch(r"09\d{8}", d) else ""


def _format_transcript(history, max_chars=8000):
    if not history:
        return ""
    lines = []
    for m in history or []:
        role = "客戶" if m.get("role") == "user" else "客服"
        c = str(m.get("content", "")).strip()
        if c:
            lines.append(f"{role}：{c}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = "…(前略)\n" + text[-max_chars:]
    return text


def _build_payload(data, display_name, session_id, card_text, transcript):
    """複製後端 _build_lead_payload 的輸出格式。"""
    name = _pick(data, _NAME_KEYS) or (display_name or "")
    phone = _pick(data, _PHONE_KEYS)
    line_id = _pick(data, _LINE_KEYS)
    message = (card_text or "").strip()
    if not message:
        message = _pick(data, _MSG_KEYS)
    now_iso = datetime.now(TW).isoformat(timespec="seconds")
    payload, used = {}, set()
    norm = lambda s: s.replace(" ", "").lower()
    items = [(str(k).strip(), v) for k, v in (data or {}).items()
             if str(k).strip() and v not in (None, "")]
    for pk in _PRIORITY:
        for k, v in items:
            if k not in used and norm(k) == norm(pk):
                payload[k] = v
                used.add(k)
                break
    for k, v in items:
        if k not in used:
            payload[k] = v
            used.add(k)
    if transcript and transcript.strip():
        payload["conversation"] = transcript.strip()
    payload.update({
        "name": name, "phone": phone, "lineId": line_id, "message": message,
        "campaign": "LINE加好友", "ref": session_id, "time": now_iso,
    })
    return payload


def _secret_sanity(secret):
    if secret is None:
        return "（空）"
    s = secret
    notes = [f"長度 {len(s)}"]
    if s != s.strip():
        notes.append("⚠️ 前後有空白/換行（建議重存去除）")
    if "\n" in s or "\r" in s:
        notes.append("⚠️ 內含換行")
    return "，".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--session", type=str, default=None)
    args = ap.parse_args()

    # 找 bot
    rows = sb.table("bots").select("id, name, webhook_url, webhook_secret, webhook_enabled").execute().data or []
    bot = None
    if args.bot:
        bot = next((b for b in rows if b["id"] == args.bot), None)
    elif args.name:
        cands = [b for b in rows if args.name.lower() in str(b.get("name") or "").lower()]
        if len(cands) > 1:
            print("符合的 bot 不只一個，請改用 --bot 指定：")
            for b in cands:
                print(f"  {b['id']}  {b.get('name')}")
            return
        bot = cands[0] if cands else None
    if not bot:
        print("找不到 bot，請用 --bot <id> 或 --name <名稱關鍵字>")
        return

    url = bot.get("webhook_url") or ""
    secret = bot.get("webhook_secret") or ""
    print("=" * 64)
    print(f"Bot：{bot.get('name')}（{bot['id']}）")
    print(f"webhook 啟用：{'是' if bot.get('webhook_enabled') else '否（未啟用，正式送單不會發）'}")
    print(f"網址：{url}")
    print(f"金鑰：{secret[:4]}…{secret[-2:] if len(secret) > 6 else ''}　{_secret_sanity(secret)}")
    print("=" * 64)
    if not url:
        print("此 bot 沒有設定 webhook 網址，無法送。")
        return

    # 找要重送的單
    q = sb.table("submissions").select("session_id, display_name, data, card_text, status, created_at") \
        .eq("bot_id", bot["id"]).eq("status", "complete")
    if args.session:
        q = q.in_("session_id", [s.strip() for s in args.session.split(",") if s.strip()])
    else:
        since = (datetime.now(TW) - timedelta(days=args.days)).astimezone(timezone.utc).isoformat()
        q = q.gte("created_at", since)
    subs = q.order("created_at", desc=False).execute().data or []

    todo = [s for s in subs if _phone(s.get("data") or {})]
    skipped = len(subs) - len(todo)
    mode = "真的送出" if args.send else "乾跑預覽（未送出，加 --send 才會送）"
    print(f"\n符合重送條件（完成＋有電話）：{len(todo)} 筆；略過無電話：{skipped} 筆")
    print(f"模式：{mode}\n" + "-" * 64)

    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Intake-Key"] = secret

    for s in todo:
        data = s.get("data") or {}
        sid = s.get("session_id") or ""
        name = _pick(data, _NAME_KEYS) or s.get("display_name") or "（無名）"
        t = (s.get("created_at") or "")[:19].replace("T", " ")
        transcript = ""
        try:
            sess = sb.table("sessions").select("history").eq("session_id", sid).limit(1).execute()
            if sess.data:
                transcript = _format_transcript(sess.data[0].get("history") or [])
        except Exception:
            pass
        payload = _build_payload(data, s.get("display_name"), sid, s.get("card_text"), transcript)

        print(f"\n[{t}] {name}　電話:{payload['phone']}　ref:{sid[:12]}…")
        print(f"     欄位數:{len(payload)}　問答紀錄:{'有' if payload.get('conversation') else '無'}")
        if not args.send:
            continue
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=10)
            body = (resp.text or "")[:200]
            tag = "✅" if resp.status_code == 200 else "❌"
            print(f"     {tag} 回應 {resp.status_code}：{body}")
        except Exception as e:
            print(f"     ❌ 連線錯誤：{e}")

    print("\n" + "=" * 64)
    if not args.send:
        print("這是乾跑，沒有真的送出。確認網址/金鑰正確、且對方已開通來源後，加 --send 重送。")
    else:
        print("重送完成。403=對方來源尚未開通；200=收下（重複送不會變兩張單）。")


if __name__ == "__main__":
    main()
