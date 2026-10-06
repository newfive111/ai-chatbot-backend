"""
一次性稽核腳本：比對最近的客戶單電話 vs 客戶在對話裡「真的打過」的電話。

背景：長對話超過 history 上限時，模型曾在 DATA_SAVE 憑空編造早期欄位（含電話）。
此腳本把每張 submission 的電話，跟該 session 對話歷史中客戶實際輸入的手機號比對，
列出「單上的電話，客戶從頭到尾沒打過」的可疑單，方便人工覆核/通知合作方。

用法（需有正式環境的 SUPABASE_URL / SUPABASE_KEY，例如在 Railway 或本機 .env）：
    python3 scripts/check_phones.py            # 預設查最近 7 天
    python3 scripts/check_phones.py --days 30
    python3 scripts/check_phones.py --bot <bot_id>   # 只查單一 bot

只讀取資料、不做任何修改。
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

# 台灣手機：09 + 8 碼（允許中間有空白、- 、. 分隔）
_PHONE_RE = re.compile(r"09[\d\s\-\.]{8,12}")
_PHONE_KEYS = ["電話", "手機", "聯絡電話", "電話號碼", "手機號碼", "phone", "mobile", "tel"]


def _digits(s: str) -> str:
    """取出字串中的數字；若是 09 開頭的 10 碼手機就回傳正規化後的號碼，否則回空字串。"""
    d = re.sub(r"\D", "", s or "")
    return d if re.fullmatch(r"09\d{8}", d) else ""


def _submission_phone(data: dict) -> str:
    """從 submission.data 依同義字找電話欄位，正規化成 09xxxxxxxx。"""
    if not data:
        return ""
    norm = {str(k).strip().lower().replace(" ", ""): v for k, v in data.items()}
    for key in _PHONE_KEYS:
        kk = key.lower().replace(" ", "")
        if kk in norm and norm[kk]:
            p = _digits(str(norm[kk]))
            if p:
                return p
    return ""


def _phones_in_history(history: list) -> set:
    """抓出客戶訊息中出現過的所有手機號（正規化）。只看 user 角色，避免抓到 bot 複誦的號碼。"""
    found = set()
    for m in history or []:
        if m.get("role") != "user":
            continue
        for raw in _PHONE_RE.findall(str(m.get("content", ""))):
            p = _digits(raw)
            if p:
                found.add(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="往前查幾天（預設 7）")
    ap.add_argument("--bot", type=str, default=None, help="只查指定 bot_id")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    q = sb.table("submissions").select("id, bot_id, session_id, display_name, data, created_at") \
        .gte("created_at", since).order("created_at", desc=True)
    if args.bot:
        q = q.eq("bot_id", args.bot)
    subs = q.execute().data or []

    print(f"稽核範圍：最近 {args.days} 天，共 {len(subs)} 張單\n" + "=" * 60)

    suspicious = []
    no_phone = []
    no_session = []
    ok = 0

    for s in subs:
        sub_phone = _submission_phone(s.get("data") or {})
        name = s.get("display_name") or "（無名）"
        created = (s.get("created_at") or "")[:19].replace("T", " ")
        sid = s.get("session_id") or ""

        if not sub_phone:
            no_phone.append((created, name))
            continue

        sess = sb.table("sessions").select("history").eq("session_id", sid).limit(1).execute()
        if not sess.data:
            no_session.append((created, name, sub_phone))
            continue

        said = _phones_in_history(sess.data[0].get("history") or [])
        if not said:
            # 客戶可能用非 09 格式或拆成多則輸入，標為需人工看
            no_session.append((created, name, sub_phone))
        elif sub_phone in said:
            ok += 1
        else:
            suspicious.append((created, name, sub_phone, sorted(said), sid))

    print(f"\n✅ 電話相符：{ok} 張")
    print(f"⚠️  可疑（單上電話客戶從沒打過）：{len(suspicious)} 張")
    print(f"❓ 無法比對（對話查無 09 手機號／session 不存在）：{len(no_session)} 張")
    print(f"➖ 單上沒有電話欄位：{len(no_phone)} 張")

    if suspicious:
        print("\n" + "=" * 60 + "\n⚠️  可疑單明細（請人工覆核、必要時通知合作方）\n" + "=" * 60)
        for created, name, sub_phone, said, sid in suspicious:
            print(f"\n[{created}] {name}")
            print(f"  單上電話　：{sub_phone}")
            print(f"  客戶實際打：{', '.join(said)}")
            print(f"  session　 ：{sid}")

    if no_session:
        print("\n" + "-" * 60 + "\n❓ 無法自動比對（建議一併人工看對話紀錄）\n" + "-" * 60)
        for created, name, sub_phone in no_session:
            print(f"  [{created}] {name}　單上電話：{sub_phone}")


if __name__ == "__main__":
    main()
