import json
import logging
import re
from typing import List, Optional, Tuple

from app.rag.embeddings import search_similar_chunks
import app.chat.session_store as session_store

# 資料交接完成後回給客人的等待語
HANDOFF_REPLY = "您好！您的資料我們已收到，專員將盡快與您聯繫，請稍候 🙏"


# ──────────────────────────────────────
# Session 操作（對外介面，engine 以外的模組用這個）
# ──────────────────────────────────────

def reset_session(session_id: str):
    session_store.delete(session_id)


def get_session_status(session_id: str) -> str:
    return session_store.get_status(session_id)


# ──────────────────────────────────────
# System Prompt
# ──────────────────────────────────────

# 客戶沒有自訂 prompt 時的預設角色（純角色描述，智能規則由 PLATFORM_RULES 統一注入）
DEFAULT_ROLE_PROMPT = "你是「{bot_name}」的客服人員，負責解答客戶問題、處理服務需求，保持親切耐心的態度。"

# 平台底層規則：不管客戶寫了什麼 prompt，每個 bot 都自動具備這些能力
PLATFORM_RULES = """
---【平台行為規則 - 以下規則優先於所有指示】---

【防失憶原則 - 最高優先】
回覆前必須逐條掃描整段對話歷史，列出所有已收集的資訊，然後只問還沒收集到的。
絕對禁止詢問客戶已經在任何一則訊息中提供過的資訊，就算那則訊息同時包含多個資訊也算已提供。
客戶在一則訊息中同時提供多個欄位（如「我叫林東東 LINE是cc123」），所有欄位都必須一次記錄完畢。

【格式驗證】
收到客戶提供的資料時，主動驗證格式：
- 電話號碼：台灣手機必須 09 開頭共 10 碼；市話含區碼共 8-9 碼。若格式不符，立刻說「電話好像少一碼，再確認一下？」，不繼續往下問。
- 金額、數字類欄位若明顯不合理，婉轉確認。

【跟追原則】
若你問了一個問題，但客戶沒有正面回答（只說「喔」「好」「不知道」「沒關係」，或完全沒提到你問的內容）：
1. 用一句話回應客戶說的話
2. 立刻把那個問題重新問一次，語氣自然，例如「不過，您剛才的 [欄位] 還沒告訴我呢？」
絕對不能因為客戶沒答就跳過那個欄位。

【身份保護】
絕對不能說「我是 AI」「作為語言模型」「作為機器人」「我沒有相關資料」等詞句。
永遠以真人客服身份回覆，用繁體中文，語氣親切自然。

【資料儲存規則 - 最高優先】
收集資料分兩個階段：

階段一：收集中（每收到一個欄位立即觸發）
每當客人提供任何欄位資料時，立即在回覆末尾另起一行輸出：
DATA_PARTIAL: {"已收集欄位1": "值1", "已收集欄位2": "值2"}
注意：
- 只輸出已收集到的欄位，尚未收集的不要放進去
- JSON 必須非空（至少有一個欄位）；若尚未收集到任何資料，絕對不要輸出 DATA_PARTIAL
- 每次有新資料都要重新輸出完整已收集內容（系統自動合併同一筆）
- 輸出後繼續詢問下一個尚未收集的欄位

階段二：全部完成（所有欄位都收集到後）
列出所有資料摘要，詢問客人「請問以上資料是否正確？確認後我幫您完成登記。」
等客人明確確認後，改輸出：
DATA_SAVE: {"所有欄位1": "值1", "所有欄位2": "值2"}
注意：
- DATA_SAVE 代表收集完成，只在客人確認後才輸出
- 客人說要修改 → 更新後重新列摘要確認，輸出 DATA_PARTIAL 紀錄修改
- 格式規定：合法 JSON，字串用雙引號，鍵名用原始欄位名稱

【DATA_SAVE 完成交接語】
輸出 DATA_SAVE 後，必須在同一則訊息加上一句簡短的完成語，告知客戶資料已登記完成、稍候聯繫。語氣符合你的角色設定，不要提及「主管」「專員」等詞語，除非角色設定中有特別說明。

【欄位限制 - 最高優先】
絕對不能自行詢問角色設定或收集欄位以外的問題。不能自行加入 LINE ID、Email、備注等未指定欄位。只收集明確被要求的欄位。"""


CALENDAR_BOOKING_RULES = """

---【預約系統規則】---
你具備查詢空檔和建立預約的能力。

【預約流程】
1. 客人說想預約時，先問「請問您希望哪天預約？」
2. 客人說出日期後，立刻呼叫 check_availability 工具查詢
3. 告知客人當天可選的時段，例如「今天還有 10:00、14:00、16:00 可以預約」
4. 客人選好時段後，詢問：姓名、電話、服務項目（若還沒有）
5. 全部確認後，呼叫 book_appointment 完成預約
6. 回覆「✅ 預約成功！[日期] [時間] 已為您登記，期待您的光臨！」

【注意事項】
- 不要自己猜測有沒有空，一定要呼叫工具確認
- 若當天無空位，主動說「那天已滿，要不要看看其他日期？」
- 客人說「明天」「下週一」等，請換算成正確的 YYYY-MM-DD 格式"""


def _get_system_prompt(
    bot_name: str,
    context: str,
    custom_system_prompt: Optional[str] = None,
    has_sheet: bool = False,
    has_calendar: bool = False
) -> str:
    """
    組合最終 system prompt：[角色設定] + [知識庫] + [預約規則?] + [平台底層規則]
    """
    from datetime import datetime, timezone, timedelta
    TW = timezone(timedelta(hours=8))
    now_tw = datetime.now(TW)
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    date_info = f"\n\n【目前時間】{now_tw.strftime('%Y-%m-%d')} 星期{weekday_names[now_tw.weekday()]} {now_tw.strftime('%H:%M')}\n"

    role_section = custom_system_prompt.strip() if (custom_system_prompt and custom_system_prompt.strip()) \
                   else DEFAULT_ROLE_PROMPT.format(bot_name=bot_name)

    kb_section = f"\n\n【知識庫參考資料】\n{context}" if context else ""

    calendar_section = CALENDAR_BOOKING_RULES if has_calendar else ""

    rules = PLATFORM_RULES
    if not has_sheet:
        rules = rules.split("【DATA_SAVE 完成交接語】")[0].rstrip()

    return f"{role_section}{date_info}{kb_section}{calendar_section}{rules}"


# ──────────────────────────────────────
# 模型呼叫（Gemini 2.5 Flash）
# ──────────────────────────────────────

def _call_ai(api_key: str, system_prompt: str, history: list, question: str) -> str:
    """呼叫 Gemini 2.5 Flash（無工具版）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1024,
        ),
    )
    return response.text


def _call_ai_with_ziwei(
    api_key: str,
    system_prompt: str,
    history: list,
    question: str
) -> str:
    """呼叫 Gemini 2.5 Flash（含紫微排盤 Function Calling）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    tools = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="generate_ziwei_chart",
            description="用客人提供的出生年月日、時辰、性別排出紫微斗數命盤。收集完生辰資料後立刻呼叫此工具。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "solar_date": types.Schema(
                        type=types.Type.STRING,
                        description="國曆出生日期，格式 YYYY-M-D，例如 1990-1-15"
                    ),
                    "birth_hour": types.Schema(
                        type=types.Type.INTEGER,
                        description="出生時辰索引：0=子時(23-01), 1=丑時(01-03), 2=寅時(03-05), 3=卯時(05-07), 4=辰時(07-09), 5=巳時(09-11), 6=午時(11-13), 7=未時(13-15), 8=申時(15-17), 9=酉時(17-19), 10=戌時(19-21), 11=亥時(21-23)。若客人說幾點出生，換算成對應時辰索引。若客人不知道時辰，預設用 0（子時）並說明。"
                    ),
                    "gender": types.Schema(
                        type=types.Type.STRING,
                        description="性別：男 或 女"
                    ),
                },
                required=["solar_date", "birth_hour", "gender"]
            )
        ),
    ])

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    for _ in range(4):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[tools],
                max_output_tokens=2048,
            )
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)

        fc_parts = [p for p in candidate.content.parts if p.function_call]
        if not fc_parts:
            return "".join(
                p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
            ).strip() or "請提供您的出生資訊，我來為您排盤。"

        tool_response_parts = []
        for p in fc_parts:
            fc = p.function_call
            args = dict(fc.args) if fc.args else {}

            try:
                if fc.name == "generate_ziwei_chart":
                    from app.fortune.ziwei import generate_chart
                    chart = generate_chart(
                        solar_date=args["solar_date"],
                        birth_hour=int(args["birth_hour"]),
                        gender=args["gender"],
                    )
                    result = chart if chart else "排盤失敗，請確認出生資訊是否正確。"
                else:
                    result = "未知工具"
            except Exception as e:
                result = f"排盤失敗：{str(e)[:80]}"

            logging.info(f"[ZiWei] generate_chart → {len(result)} chars")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name, response={"result": result}
                    )
                )
            )

        contents.append(types.Content(role="tool", parts=tool_response_parts))

    return "排盤處理超時，請再試一次。"


def _call_ai_with_calendar(
    api_key: str,
    system_prompt: str,
    history: list,
    question: str,
    calendar_id: str,
    slot_duration: int,
    business_hours: dict
) -> str:
    """呼叫 Gemini 2.5 Flash（含預約 Function Calling）"""
    from google import genai
    from google.genai import types
    from datetime import datetime, timezone, timedelta

    TW = timezone(timedelta(hours=8))
    today_tw = datetime.now(TW).strftime("%Y-%m-%d")

    client = genai.Client(api_key=api_key)

    tools = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="check_availability",
            description="查詢指定日期的可預約時段，詢問客人想預約哪天後立刻呼叫此工具",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "date": types.Schema(
                        type=types.Type.STRING,
                        description=f"日期，格式 YYYY-MM-DD，今天是 {today_tw}"
                    )
                },
                required=["date"]
            )
        ),
        types.FunctionDeclaration(
            name="book_appointment",
            description="確認預約，在行事曆建立事件。客人確認日期、時間、姓名、電話後呼叫",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "date":            types.Schema(type=types.Type.STRING, description="日期 YYYY-MM-DD"),
                    "time":            types.Schema(type=types.Type.STRING, description="時間 HH:MM"),
                    "customer_name":   types.Schema(type=types.Type.STRING, description="客人姓名"),
                    "customer_phone":  types.Schema(type=types.Type.STRING, description="客人電話"),
                    "service":         types.Schema(type=types.Type.STRING, description="服務項目，例如：剪髮、染髮"),
                },
                required=["date", "time", "customer_name", "customer_phone"]
            )
        )
    ])

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    for _ in range(6):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[tools],
                max_output_tokens=1024,
            )
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)

        fc_parts = [p for p in candidate.content.parts if p.function_call]
        if not fc_parts:
            # 最終文字回覆
            return "".join(
                p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
            ).strip() or "處理完成"

        # 執行工具
        tool_response_parts = []
        for p in fc_parts:
            fc   = p.function_call
            args = dict(fc.args) if fc.args else {}

            try:
                if fc.name == "check_availability":
                    from app.calendar.client import get_available_slots
                    slots = get_available_slots(
                        calendar_id, args["date"], slot_duration, business_hours
                    )
                    if slots:
                        result = f"{args['date']} 可預約時段：{', '.join(slots)}"
                    else:
                        result = f"{args['date']} 當天沒有可預約的時段（休假或已全滿）"

                elif fc.name == "book_appointment":
                    from app.calendar.client import create_booking
                    svc   = args.get("service", "預約")
                    name  = args.get("customer_name", "客人")
                    phone = args.get("customer_phone", "")
                    title = f"{name} - {svc}"
                    desc  = f"姓名：{name}\n電話：{phone}\n服務：{svc}"
                    data  = create_booking(
                        calendar_id, title, args["date"], args["time"], slot_duration, desc
                    )
                    result = f"預約成功！{args['date']} {args['time']}，{name} 的 {svc} 已登記。"
                else:
                    result = "未知工具"

            except Exception as e:
                result = f"操作失敗：{str(e)[:80]}"

            logging.info(f"[Calendar] {fc.name} → {result[:80]}")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name, response={"result": result}
                    )
                )
            )

        contents.append(types.Content(role="tool", parts=tool_response_parts))

    return "處理超時，請再試一次。"


# ──────────────────────────────────────
# DATA_SAVE 偵測
# ──────────────────────────────────────

def _get_display_name(data: dict) -> str:
    """從收集資料中找姓名欄位，作為 Sheet 第一欄的顯示名稱"""
    name_keywords = ["姓名", "名字", "稱呼", "姓", "name"]
    for key in data:
        if any(kw in key for kw in name_keywords):
            val = str(data[key]).strip()
            if val:
                return val
    return None


def _extract_json_object(text: str, marker: str = "DATA_SAVE") -> Optional[tuple]:
    """
    從指定 marker（DATA_SAVE 或 DATA_PARTIAL）之後提取完整 JSON 物件。
    使用括號平衡法，正確處理巢狀結構與字串內的括號。
    支援英文冒號 ':' 和中文全形冒號 '：'。
    回傳 (json_str, marker_start, json_end) 或 None。
    """
    m = re.search(rf'{re.escape(marker)}\s*[:\uff1a]\s*(\{{)', text, re.IGNORECASE)
    if not m:
        return None
    marker_start = m.start()   # marker 開始位置（含 DATA_PARTIAL/DATA_SAVE）
    start = m.start(1)         # { 的位置
    depth = 0
    in_str = False
    escape = False
    for i, c in enumerate(text[start:]):
        if escape:
            escape = False
            continue
        if c == '\\' and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                json_end = start + i + 1
                return text[start:json_end], marker_start, json_end
    return None


def _write_data_to_sheet(json_str: str, sheet_id: str, session_id: str, marker: str, extra_sheet_fields: Optional[dict]) -> Optional[str]:
    """解析 JSON 並寫入 Sheet，失敗時只 log warning。回傳 display_name（或 None）。"""
    try:
        data: dict = json.loads(json_str)
        fields = list(data.keys())
        from app.sheets.client import upsert_row
        display_name = _get_display_name(data)
        upsert_row(sheet_id, session_id, fields, data, display_name=display_name, extra_fields=extra_sheet_fields)
        logging.info(f"[Sheet] {marker} written session={session_id[:8]} fields={fields}")
        return display_name
    except json.JSONDecodeError as e:
        logging.warning(f"[Engine] {marker} JSON parse error: {e} | raw={json_str[:200]}")
    except Exception as e:
        logging.warning(f"[Sheet] {marker} write failed: {e}")
    return None


def _extract_and_save_data(
    raw_reply: str,
    sheet_id: str,
    session_id: str,
    extra_sheet_fields: Optional[dict] = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    偵測 AI 回覆中的資料標記，寫入 Google Sheet。
    - DATA_PARTIAL: {...} → 部分存檔，不觸發 handed_off
    - DATA_SAVE: {...}    → 完整存檔，觸發 handed_off（靜默）
    支援英文冒號與中文全形冒號，使用括號平衡法提取 JSON。
    回傳 (清理後文字, 是否找到DATA_SAVE, display_name)
    """
    cleaned = raw_reply
    found_final = False
    saved_display_name: Optional[str] = None

    def _strip_marker(text: str, result: tuple) -> str:
        """用精確位置移除 marker 段落，避免 regex 截斷問題。"""
        json_str, marker_start, json_end = result
        # 移除前後的換行
        start = marker_start
        end = json_end
        if start > 0 and text[start - 1] == '\n':
            start -= 1
        if end < len(text) and text[end] == '\n':
            end += 1
        return (text[:start] + text[end:]).strip()

    # ── 處理 DATA_PARTIAL（部分資料，不靜默）──
    partial_result = _extract_json_object(raw_reply, marker="DATA_PARTIAL")
    if partial_result:
        _write_data_to_sheet(partial_result[0], sheet_id, session_id, "DATA_PARTIAL", extra_sheet_fields)
        cleaned = _strip_marker(cleaned, partial_result)

    # ── 處理 DATA_SAVE（完整資料，觸發靜默）──
    final_result = _extract_json_object(cleaned, marker="DATA_SAVE")
    if final_result:
        saved_display_name = _write_data_to_sheet(final_result[0], sheet_id, session_id, "DATA_SAVE", extra_sheet_fields)
        cleaned = _strip_marker(cleaned, final_result)
        found_final = True

    if not partial_result and not final_result:
        logging.debug(f"[Engine] No data marker in reply (len={len(raw_reply)})")

    return cleaned, found_final, saved_display_name


def _generate_conversation_summary(api_key: str, history: list, last_question: str) -> str:
    """用 AI 生成一句話對話摘要（30字以內）。失敗時回傳空字串。"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        lines = []
        for m in (history or []):
            role = "客戶" if m.get("role") == "user" else "AI"
            lines.append(f"{role}：{str(m.get('content', ''))[:120]}")
        lines.append(f"客戶：{last_question[:120]}")
        conversation_text = "\n".join(lines[-20:])  # 最多 20 輪

        prompt = (
            "以下是一段客服對話，請用一句話（30字以內）摘要客戶的主要需求或情況。"
            "只輸出摘要本身，不要加任何前綴或標點說明：\n\n"
            f"{conversation_text}"
        )
        response = model.generate_content(prompt)
        return (response.text or "").strip()[:60]
    except Exception as e:
        logging.warning(f"[Engine] Summary generation failed: {e}")
        return ""


# ──────────────────────────────────────
# 主入口
# ──────────────────────────────────────

def generate_answer(
    bot_id: str,
    question: str,
    bot_name: str = "AI 助理",
    api_key: Optional[str] = None,
    collect_fields: Optional[List[str]] = None,
    sheet_id: Optional[str] = None,
    session_id: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    handoff_reply: Optional[str] = None,
    # 預約系統
    calendar_id: Optional[str] = None,
    slot_duration_minutes: int = 60,
    business_hours: Optional[dict] = None,
    # 關鍵字觸發
    keyword_triggers: Optional[list] = None,
    # 額外寫入試算表的欄位（如 LINE暱稱）
    extra_sheet_fields: Optional[dict] = None,
    # 下班時間自動回應
    off_hours_message: Optional[str] = None,
    # 紫微斗數排盤
    enable_ziwei: bool = False,
) -> str:
    if not api_key:
        raise Exception("NO_API_KEY")

    # ── 下班時間判斷（供後段使用）──
    _is_off_hours = False
    if off_hours_message:
        from datetime import datetime, timezone, timedelta
        TZ = timezone(timedelta(hours=8))
        now = datetime.now(TZ)
        bh = business_hours or {}
        weekdays = bh.get("weekdays", [1, 2, 3, 4, 5])
        _start   = bh.get("start", "09:00")
        _end     = bh.get("end", "18:00")
        cur_time = now.strftime("%H:%M")
        _is_off_hours = now.isoweekday() not in weekdays or not (_start <= cur_time <= _end)

    # ── 關鍵字觸發（最高優先，不耗 token）──
    if keyword_triggers:
        q_lower = question.lower()
        for kt in keyword_triggers:
            kw = kt.get("keyword", "").lower()
            if kw and kw in q_lower:
                logging.info(f"[Engine] Keyword match: '{kw}'")
                return kt.get("reply", "")

    relevant_chunks = search_similar_chunks(bot_id, question, top_k=5)
    context = "\n\n".join(relevant_chunks)
    has_calendar = bool(calendar_id)
    system_prompt = _get_system_prompt(
        bot_name, context, custom_system_prompt,
        has_sheet=bool(sheet_id),
        has_calendar=has_calendar
    )
    _bh = business_hours or {"start": "09:00", "end": "18:00", "weekdays": [1, 2, 3, 4, 5]}

    # ── 路徑 A：有自訂 prompt → LLM 全程主導（含 calendar 支援）──
    if custom_system_prompt and custom_system_prompt.strip():
        if session_id:
            session = session_store.get_or_create(session_id)
            if session.get("status") == "handed_off":
                logging.info(f"[Engine] {session_id[:8]} handed_off → silent")
                return ""
            history = session.get("history", [])
        else:
            session = None
            history = []

        # 路徑選擇：calendar / ziwei / 純文字
        if has_calendar:
            raw_reply = _call_ai_with_calendar(
                api_key, system_prompt, history, question,
                calendar_id, slot_duration_minutes, _bh
            )
        elif enable_ziwei:
            raw_reply = _call_ai_with_ziwei(
                api_key, system_prompt, history, question
            )
        else:
            raw_reply = _call_ai(api_key, system_prompt, history, question)

        if sheet_id and session_id:
            clean_reply, data_saved, saved_display_name = _extract_and_save_data(raw_reply, sheet_id, session_id, extra_sheet_fields=extra_sheet_fields)
            if data_saved:
                session["status"] = "handed_off"
                logging.info(f"[Engine] {session_id[:8]} → handed_off (DATA_SAVE)")
                # 對話摘要：非同步補寫到試算表
                try:
                    summary = _generate_conversation_summary(api_key, history, question)
                    if summary:
                        from app.sheets.client import update_extra_fields
                        update_extra_fields(sheet_id, session_id, {"對話摘要": summary}, display_name=saved_display_name)
                        logging.info(f"[Engine] Summary written for {session_id[:8]}: {summary[:30]}")
                except Exception as _e:
                    logging.warning(f"[Engine] Summary write failed: {_e}")
                # 下班時間：資料收完後附上通知
                if _is_off_hours and off_hours_message:
                    clean_reply = clean_reply + "\n\n" + off_hours_message
                    logging.info(f"[Engine] Off-hours message appended after DATA_SAVE")
        else:
            clean_reply = re.sub(r'\n?DATA_(?:SAVE|PARTIAL):\s*\{.*?\}\n?', '', raw_reply, flags=re.DOTALL).strip()

        if session_id and session is not None:
            session["history"] = history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": clean_reply},
            ]
            session_store.save(session_id, session)

        return clean_reply

    # ── 路徑 B：無自訂 prompt → 逐題收集 collect_fields ──
    if collect_fields and sheet_id and session_id:
        session = session_store.get_session(session_id)

        # ① 已收集完 → 一般問答
        if session and session.get("done"):
            history = session.get("history", [])
            ai_reply = _call_ai(api_key, system_prompt, history, question)
            session["history"] = history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ai_reply},
            ]
            session_store.save(session_id, session)
            return ai_reply

        # ② 全新 session
        if session is None:
            session = {
                "fields":    collect_fields,
                "collected": {},
                "step":      0,
                "done":      False,
                "history":   [],
                "status":    "active",
            }
            ai_reply = _call_ai(api_key, system_prompt, [], question)
            first_field = collect_fields[0]
            full_reply = f"{ai_reply}\n\n---\n請問您的{first_field}是？"
            session["history"] = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_reply},
            ]
            session_store.save(session_id, session)
            return full_reply

        # ③ 收集進行中
        if not session.get("done"):
            step = session["step"]
            fields = session["fields"]
            collected = session["collected"]

            if step < len(fields):
                collected[fields[step]] = question.strip()
                session["step"] = step + 1

                try:
                    from app.sheets.client import upsert_row
                    display_name = _get_display_name(collected)
                    upsert_row(sheet_id, session_id, fields, collected, display_name=display_name)
                except Exception as e:
                    logging.warning(f"[Sheet] incremental save failed: {e}")

                if session["step"] < len(fields):
                    reply = f"收到！請問您的{fields[session['step']]}是？"
                else:
                    session["done"] = True
                    reply = "感謝您提供資料！我們已收到您的資訊，稍後會與您聯繫 😊\n\n如果還有其他問題，歡迎繼續詢問！"

                session["history"].append({"role": "user", "content": question})
                session["history"].append({"role": "assistant", "content": reply})
                session_store.save(session_id, session)
                return reply

    # ── 路徑 C：一般問答（無收集）──
    if session_id:
        session = session_store.get_or_create(session_id)
        history = session.get("history", [])
        ai_reply = _call_ai(api_key, system_prompt, history, question)
        session["history"] = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ai_reply},
        ]
        session_store.save(session_id, session)
        return ai_reply

    return _call_ai(api_key, system_prompt, [], question)
