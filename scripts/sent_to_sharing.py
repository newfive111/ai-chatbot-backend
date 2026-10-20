"""
還原「發去合作方(Sharing)的客戶資料」。

webhook 送出是背景動作、沒有獨立送出紀錄表，所以用「實際會送出的條件」回推：
  bot 有開 webhook(webhook_enabled) + 客戶單已完成(status=complete) + 有電話。
這三個條件同時成立的單，就是當天會打到對方的客戶資料。

用法（需正式環境 SUPABASE_URL / SUPABASE_KEY，例如 railway run）：
    python3 scripts/sent_to_sharing.py                 # 今天（台灣時間）
    python3 scripts/sent_to_sharing.py --days 3        # 最近 3 天
    python3 scripts/sent_to_sharing.py --bot <bot_id>  # 只看單一 bot
    python3 scripts/sent_to_sharing.py --full          # 連同問答紀錄一起印

只讀取資料、不送出、不修改。注意：以「目前」的 webhook_enabled 判斷，
若某 bot 今天才關掉/打開，結果會以現在的設定為準。
"""
import os
import re
import sys
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import SUPABASE_URL, SUPABASE_KEY  # noqa: E402
from supabase import create_client  # noqa: E402

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
TW = timezone(timedelta(hours=8))

_PHONE_KEYS = ["電話", "手機", "聯絡電話", "電話號碼", "手機號碼", "phone", "mobile", "tel"]
_LINE_KEYS = ["lineid", "line id", "line", "line帳號", "賴", "line_id"]
_NAME_KEYS = ["姓名", "名字", "稱呼", "客戶姓名", "name", "fullname"]


def _pick(data: dict, keys: list) -> str:
    if not data:
        return ""
    norm = {str(k).strip().lower().replace(" ", ""): v for k, v in data.items()}
    for key in keys:
        kk = key.lower().replace(" ", "")
        if kk in norm and str(norm[kk]).strip():
            return str(norm[kk]).strip()
    return ""


def _phone(data: dict) -> str:
    raw = _pick(data, _PHONE_KEYS)
    d = re.sub(r"\D", "", raw)
    return d if re.fullmatch(r"09\d{8}", d) else (raw or "")


def _transcript_len(session_id: str) -> int:
    try:
        r = sb.table("sessions").select("history").eq("session_id", session_id).limit(1).execute()
        if not r.data:
            return 0
        hist = r.data[0].get("history") or []
        return sum(len(str(m.get("content", ""))) for m in hist if str(m.get("content", "")).strip())
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="往前查幾天（0=只看今天，台灣時間）")
    ap.add_argument("--bot", type=str, default=None, help="只看指定 bot_id")
    ap.add_argument("--full", action="store_true", help="連同問答紀錄一起印")
    args = ap.parse_args()

    now_tw = datetime.now(TW)
    if args.days > 0:
        start = (now_tw - timedelta(days=args.days)).replace(hour=0, minute=0, second=0, microsecond=0)
        label = f"最近 {args.days} 天"
    else:
        start = now_tw.replace(hour=0, minute=0, second=0, microsecond=0)
        label = f"今天（{now_tw:%Y-%m-%d}）"
    since_utc = start.astimezone(timezone.utc).isoformat()

    # 1) 有開 webhook 的 bot
    bq = sb.table("bots").select("id, name, webhook_enabled, webhook_send_conversation")
    if args.bot:
        bq = bq.eq("id", args.bot)
    bots = {b["id"]: b for b in (bq.execute().data or []) if b.get("webhook_enabled")}

    print(f"發去 Sharing 的客戶資料 · {label}")
    print(f"有開 webhook 的 bot：{len(bots)} 台")
    if not bots:
        print("（沒有任何 bot 開啟 webhook，今天不會有資料送出）")
        return
    print("=" * 64)

    total = 0
    for bid, b in bots.items():
        subs = sb.table("submissions").select(
            "session_id, display_name, data, status, created_at"
        ).eq("bot_id", bid).eq("status", "complete").gte("created_at", since_utc) \
            .order("created_at", desc=False).execute().data or []

        sent = []
        for s in subs:
            data = s.get("data") or {}
            ph = _phone(data)
            if not ph:        # 沒電話 → 不會送
                continue
            sent.append((s, data, ph))

        if not sent:
            continue

        conv_on = b.get("webhook_send_conversation") is not False
        print(f"\n🤖 {b.get('name') or bid}　（問答紀錄：{'送' if conv_on else '不送'}）　{len(sent)} 筆")
        print("-" * 64)
        for s, data, ph in sent:
            total += 1
            t = (s.get("created_at") or "")[:19].replace("T", " ")
            name = _pick(data, _NAME_KEYS) or s.get("display_name") or "（無名）"
            line_id = _pick(data, _LINE_KEYS)
            amt = _pick(data, ["需求金額", "金額", "貸款金額"])
            area = _pick(data, ["地區", "居住地區"])
            tlen = _transcript_len(s.get("session_id") or "") if conv_on else 0
            print(f"[{t}] {name}")
            print(f"    電話:{ph}  LINE:{line_id or '—'}  金額:{amt or '—'}  地區:{area or '—'}")
            if conv_on:
                print(f"    問答紀錄:{'有（約 ' + str(tlen) + ' 字）' if tlen else '查無（session 可能已過期清除）'}")
            if args.full:
                print("    ── 完整欄位 ──")
                for k, v in data.items():
                    if str(v).strip():
                        print(f"      {k}：{v}")

    print("\n" + "=" * 64)
    print(f"合計：{total} 筆客戶資料（符合送出條件）")


if __name__ == "__main__":
    main()
