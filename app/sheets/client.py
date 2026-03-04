import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, List
import os
import json
import base64
import logging
from datetime import datetime

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# 本機開發：讀 JSON 檔案
# Railway 部署：讀 GOOGLE_CREDENTIALS_JSON_B64 環境變數（base64 編碼的 JSON 內容）
_CREDS_FILE = os.path.join(os.path.dirname(__file__), "../../bothelper-489007-d6c2749a7ac0.json")


def _get_credentials() -> Credentials:
    b64 = os.getenv("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        # Railway 模式：從環境變數解碼
        creds_json = json.loads(base64.b64decode(b64).decode("utf-8"))
        return Credentials.from_service_account_info(creds_json, scopes=SCOPES)
    elif os.path.exists(_CREDS_FILE):
        # 本機開發模式：直接讀檔
        return Credentials.from_service_account_file(_CREDS_FILE, scopes=SCOPES)
    else:
        raise RuntimeError(
            "Google 服務帳號憑證未設定。"
            "本機請放 bothelper-xxx.json，Railway 請設定 GOOGLE_CREDENTIALS_JSON_B64 環境變數。"
        )


def get_sheet(sheet_id: str):
    creds = _get_credentials()
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id).sheet1


def ensure_headers(sheet, fields: List[str], with_session_id: bool = False):
    """確保第一行有欄位標題"""
    existing = sheet.row_values(1)
    if with_session_id:
        all_fields = ["session_id"] + fields + ["更新時間"]
    else:
        all_fields = fields + ["時間"]
    if existing != all_fields:
        sheet.update("A1", [all_fields])


def upsert_row(sheet_id: str, session_id: str, fields: List[str], data: Dict[str, str]):
    """
    Incremental save：每收到一個欄位就即時更新。
    - 用 session_id 找到已有的行 → 更新它
    - 找不到 → 新增一行
    這樣不會因用戶中途離開或 server 重啟而漏資料。
    """
    sheet = get_sheet(sheet_id)
    ensure_headers(sheet, fields, with_session_id=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row_data = [session_id] + [data.get(f, "") for f in fields] + [now]

    # 搜尋第一欄（session_id）找是否已有這個 session 的行
    col_a = sheet.col_values(1)  # [header, sid1, sid2, ...]
    try:
        row_index = col_a.index(session_id) + 1  # 1-based
        sheet.update(f"A{row_index}", [row_data])
        logging.info(f"[Sheet] updated row {row_index} for session {session_id[:8]}")
    except ValueError:
        # 找不到 → 新增
        sheet.append_row(row_data)
        logging.info(f"[Sheet] new row for session {session_id[:8]}")


def append_row(sheet_id: str, fields: List[str], data: Dict[str, str]):
    """舊版相容：直接新增一行（不含 session_id）"""
    sheet = get_sheet(sheet_id)
    ensure_headers(sheet, fields)
    row = [data.get(f, "") for f in fields] + [datetime.now().strftime("%Y-%m-%d %H:%M")]
    sheet.append_row(row)
