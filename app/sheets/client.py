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


def ensure_headers(sheet, fields: List[str], extra_fields: List[str] = None) -> List[str]:
    """
    確保第一行有欄位標題。
    - 固定格式：session_id | fields... | extra_fields... | 更新時間
    - 若標題列缺少欄位則補上（不刪除已有欄位）
    """
    existing = [h for h in sheet.row_values(1) if h]  # 去掉空格
    base = ["session_id"] + fields
    extras = extra_fields or []

    # 保留既有非標準欄位（不強制覆蓋），只補上缺少的
    without_time = [h for h in existing if h != "更新時間"]
    new_extras = [e for e in extras if e not in without_time]
    final = without_time + new_extras + ["更新時間"]

    if existing + ["更新時間"] != final:
        sheet.update("A1", [final])

    return final


def upsert_row(
    sheet_id: str,
    session_id: str,
    fields: List[str],
    data: Dict[str, str],
    display_name: str = None,
    extra_fields: Dict[str, str] = None,
):
    """
    Incremental save：每收到一個欄位就即時更新。
    - display_name: 有姓名欄位時顯示名稱（如「陳大明」），否則顯示 session_id
    - extra_fields: 額外欄位 dict，如 {"LINE暱稱": "王小明"}
    - 用 session_id 或 display_name 找到已有的行 → 更新它
    - 找不到 → 新增一行
    """
    sheet = get_sheet(sheet_id)
    extra_keys = list((extra_fields or {}).keys())
    headers = ensure_headers(sheet, fields, extra_fields=extra_keys)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    display_key = display_name if display_name else session_id

    # 依照實際 headers 順序組 row
    row_data = []
    for h in headers:
        if h == "session_id":
            row_data.append(display_key)
        elif h == "更新時間":
            row_data.append(now)
        elif h in data:
            row_data.append(data[h])
        elif extra_fields and h in extra_fields:
            row_data.append(extra_fields[h])
        else:
            row_data.append("")

    # 搜尋第一欄找已有的行
    col_a = sheet.col_values(1)
    row_index = None
    if display_key in col_a:
        row_index = col_a.index(display_key) + 1
    elif display_name and session_id in col_a:
        row_index = col_a.index(session_id) + 1

    if row_index:
        sheet.update(f"A{row_index}", [row_data])
        logging.info(f"[Sheet] updated row {row_index} → {display_key}")
    else:
        sheet.append_row(row_data)
        logging.info(f"[Sheet] new row → {display_key}")


def append_row(sheet_id: str, fields: List[str], data: Dict[str, str]):
    """舊版相容：直接新增一行（不含 session_id）"""
    sheet = get_sheet(sheet_id)
    ensure_headers(sheet, fields)
    row = [data.get(f, "") for f in fields] + [datetime.now().strftime("%Y-%m-%d %H:%M")]
    sheet.append_row(row)
