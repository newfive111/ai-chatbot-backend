"""
Layer 2 AI 設定助手引擎

使用 Gemini 2.5 Pro + Function Calling 直接操作 Bot 設定。
"""

import json
import logging
from google import genai
from google.genai import types

from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

_sb = create_client(SUPABASE_URL, SUPABASE_KEY)

ASSISTANT_MODEL = "gemini-2.5-flash"

ASSISTANT_SYSTEM_PROMPT = """
你是 LazyReply 平台的 AI 設定助手「小懶」。
你同時扮演兩個角色：
1. 直接幫用戶修改 Bot 設定（用工具操作）
2. 回答用戶關於平台功能的問題（給說明和步驟）

【能直接操作的設定】
- 角色設定（系統提示詞）
- 收集欄位（姓名、電話、需求等）
- 歡迎訊息 + 快速回覆按鈕
- 查看目前設定

【能回答但需用戶自己操作的事】
- 嵌入代碼：在後台「🔗 嵌入代碼」tab 有完整的 HTML 代碼，複製貼到網站的 </body> 前即可
- Google Sheet 串接：在「⚙️ 設定」→「Google Sheet 資料收集」，按照頁面步驟操作
- LINE Bot 串接：在「🔗 嵌入代碼」→「LINE Bot 串接」，按照頁面步驟填入 Channel Secret 和 Access Token
- Gemini API Key：在「⚙️ 設定」→「Gemini API Key」，從 Google AI Studio 取得後填入
- 知識庫：在「📚 知識庫」tab，上傳 PDF/TXT 或手動輸入 FAQ
- 預約系統（Google Calendar 串接）：在「⚙️ 設定」→「📅 預約系統」，步驟如下：
  1. 到 Google Calendar 建立一個專用行事曆（例如「預約系統」）
  2. 把行事曆共用給服務帳號 bothelper-sheets@bothelper-489007.iam.gserviceaccount.com（設為「更改事件」）
  3. 在行事曆設定的「整合行事曆」找到 Calendar ID（格式：xxx@group.calendar.google.com）
  4. 在後台「預約系統」填入 Calendar ID、設定每次時長和上班時間
  5. 儲存後，Bot 就能自動查詢空檔、讓客人選時間並直接建立行事曆事件
  注意：預約功能啟用後，Bot 的角色設定必須存在（在「🤖 角色」tab 設定），否則預約邏輯不會生效

【操作原則 — 嚴格遵守】

角色設定（system_prompt）修改流程：
1. 先呼叫 get_bot_config 取得現有 system_prompt 原文
2. 從原文中找出要修改的「精確舊文字」（逐字複製，不要改動）
3. 決定「新文字」（只包含要替換的部分，不是整段 prompt）
4. 呼叫 replace_in_system_prompt(find=精確舊文字, replace=新文字) — 後端會做精準字串替換，其他內容一字不動
5. 若原本完全沒有 system_prompt（空白），才使用 set_system_prompt 從頭建立
6. 改完後告知用戶改了什麼，並提醒「如有問題可點 🕐 歷史 還原」
7. 不需要確認步驟，直接套用，歷史紀錄會自動備份

其他設定（collect_fields、welcome）：
- 先呼叫 get_bot_config，直接呼叫對應工具修改，不需要確認

- 回答問題時：直接給清楚的步驟說明，不要說「超出範圍」或「請問工程師」
- 修改完後：告知「✅ 已套用」，建議去「測試對話」確認
- 語氣：專業友善，用繁體中文
- 每次只做一件事

【格式參考】
collect_fields 例：["姓名", "電話", "需求"]
quick_replies 例：[{"label": "了解方案"}, {"label": "預約諮詢"}]
system_prompt 包含：角色定位、語氣風格、專業領域、禁忌事項
"""

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_bot_config",
        description="取得目前 Bot 的所有設定，包含角色描述、收集欄位、歡迎訊息等。修改任何設定前必須先呼叫此工具，才能在現有內容基礎上做局部修改，而非從頭覆蓋。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={}
        )
    ),
    types.FunctionDeclaration(
        name="replace_in_system_prompt",
        description="在現有 system_prompt 中做精準字串替換。必須先呼叫 get_bot_config 取得原文，從中逐字複製要替換的舊文字，再提供新文字。後端直接做 string replace，其他內容完全不動。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "find": types.Schema(
                    type=types.Type.STRING,
                    description="要被替換的精確舊文字，必須與 get_bot_config 回傳的原文完全一致（逐字複製）"
                ),
                "replace": types.Schema(
                    type=types.Type.STRING,
                    description="替換後的新文字"
                )
            },
            required=["find", "replace"]
        )
    ),
    types.FunctionDeclaration(
        name="set_system_prompt",
        description="從頭建立全新的 system_prompt。只有在 get_bot_config 確認原本 system_prompt 完全是空白時才使用，其他情況請用 replace_in_system_prompt。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "system_prompt": types.Schema(
                    type=types.Type.STRING,
                    description="完整的新系統提示詞"
                )
            },
            required=["system_prompt"]
        )
    ),
    types.FunctionDeclaration(
        name="update_collect_fields",
        description="更新 Bot 需要收集的客戶資料欄位清單。用戶確認後才呼叫。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "fields": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="欄位名稱清單，例如 [\"姓名\", \"電話\", \"需求\"]"
                ),
                "sheet_id": types.Schema(
                    type=types.Type.STRING,
                    description="Google Sheet ID（選填，不改就不傳）"
                )
            },
            required=["fields"]
        )
    ),
    types.FunctionDeclaration(
        name="update_welcome",
        description="更新 Bot 的歡迎訊息和快速回覆按鈕。用戶確認後才呼叫。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "welcome_message": types.Schema(
                    type=types.Type.STRING,
                    description="開場歡迎語，用戶進入聊天時自動顯示"
                ),
                "quick_replies": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "label": types.Schema(type=types.Type.STRING)
                        }
                    ),
                    description="快速回覆按鈕清單，每個物件含 label 欄位"
                )
            },
            required=["welcome_message", "quick_replies"]
        )
    )
]


def _save_snapshot(bot_id: str, source: str = "assistant") -> None:
    """更新前把目前設定存一份快照到 bot_settings_history"""
    try:
        r = _sb.table("bots").select(
            "system_prompt, collect_fields, welcome_message, quick_replies"
        ).eq("id", bot_id).execute()
        if r.data:
            row = r.data[0]
            _sb.table("bot_settings_history").insert({
                "bot_id":          bot_id,
                "source":          source,
                "system_prompt":   row.get("system_prompt") or "",
                "collect_fields":  row.get("collect_fields") or [],
                "welcome_message": row.get("welcome_message") or "",
                "quick_replies":   row.get("quick_replies") or [],
            }).execute()
    except Exception as e:
        logging.warning(f"[Assistant] snapshot failed: {e}")


def _execute_tool(tool_name: str, tool_args: dict, bot_id: str, session: dict) -> str:
    """執行工具，直接操作 Supabase"""
    try:
        if tool_name == "get_bot_config":
            r = _sb.table("bots").select(
                "name, system_prompt, collect_fields, sheet_id, welcome_message, quick_replies"
            ).eq("id", bot_id).execute()
            if r.data:
                return json.dumps(r.data[0], ensure_ascii=False)
            return json.dumps({"error": "Bot 不存在"})

        elif tool_name == "replace_in_system_prompt":
            find_str    = tool_args.get("find", "")
            replace_str = tool_args.get("replace", "")
            if not find_str:
                return "❌ find 不能為空"
            r = _sb.table("bots").select("system_prompt").eq("id", bot_id).execute()
            current = (r.data[0].get("system_prompt") or "") if r.data else ""
            if find_str not in current:
                return f"❌ 找不到指定文字，請重新呼叫 get_bot_config 確認原文後再試。找尋內容：「{find_str[:40]}」"
            new_prompt = current.replace(find_str, replace_str, 1)
            _save_snapshot(bot_id)
            _sb.table("bots").update({"system_prompt": new_prompt}).eq("id", bot_id).execute()
            return "✅ 已精準替換指定段落，其他內容完全保留"

        elif tool_name == "set_system_prompt":
            prompt = tool_args.get("system_prompt", "")
            _save_snapshot(bot_id)
            _sb.table("bots").update({"system_prompt": prompt}).eq("id", bot_id).execute()
            return "✅ 角色設定已建立"

        elif tool_name == "update_collect_fields":
            _save_snapshot(bot_id)
            fields = tool_args.get("fields", [])
            upd: dict = {"collect_fields": fields}
            if tool_args.get("sheet_id"):
                upd["sheet_id"] = tool_args["sheet_id"]
            _sb.table("bots").update(upd).eq("id", bot_id).execute()
            return f"收集欄位已成功更新：{fields}"

        elif tool_name == "update_welcome":
            _save_snapshot(bot_id)
            _sb.table("bots").update({
                "welcome_message": tool_args.get("welcome_message", ""),
                "quick_replies": tool_args.get("quick_replies", [])
            }).eq("id", bot_id).execute()
            return "歡迎訊息和快速回覆按鈕已成功更新"

        else:
            return f"未知工具：{tool_name}"

    except Exception as e:
        logging.error(f"[Assistant] Tool {tool_name} error: {e}")
        return f"執行失敗：{str(e)}"


def _load_system_prompt() -> str:
    """從 Supabase assistant_config 讀取 system prompt，失敗時 fallback 到程式碼內建版本"""
    try:
        r = _sb.table("assistant_config").select("value").eq("key", "system_prompt").execute()
        if r.data and r.data[0].get("value"):
            return r.data[0]["value"]
    except Exception as e:
        logging.warning(f"[Assistant] Failed to load system prompt from DB: {e}")
    return ASSISTANT_SYSTEM_PROMPT


def run_assistant(bot_id: str, user_message: str, session_id: str, gemini_api_key: str) -> str:
    """
    執行 AI 助手對話（含 Function Calling 迴圈）
    回傳助手的最終文字回覆
    """
    if not gemini_api_key:
        return "⚠️ 請先在「設定」頁面填入 Gemini API Key，才能使用 AI 助手。"

    client = genai.Client(api_key=gemini_api_key)
    tools = types.Tool(function_declarations=_FUNCTION_DECLARATIONS)
    system_prompt = _load_system_prompt()

    # 讀取對話歷史
    from app.chat import session_store
    session = session_store.get_or_create(session_id)
    history: list = session.get("history", [])

    # 組 contents（多輪對話）
    contents: list[types.Content] = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    # Function Calling 迴圈（最多 6 次工具呼叫）
    for iteration in range(6):
        try:
            response = client.models.generate_content(
                model=ASSISTANT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[tools],
                    max_output_tokens=2048,
                    temperature=0.7
                )
            )
        except Exception as e:
            logging.error(f"[Assistant] Gemini API error: {e}")
            return f"⚠️ AI 服務暫時無法連線，請稍後再試。（{str(e)[:60]}）"

        if not response.candidates:
            logging.error(f"[Assistant] Empty candidates from Gemini, prompt_feedback={getattr(response, 'prompt_feedback', None)}")
            return "⚠️ AI 回應被過濾或配額不足，請稍後再試。"

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            finish = getattr(candidate, "finish_reason", "unknown")
            logging.error(f"[Assistant] No content in candidate, finish_reason={finish}")
            return "⚠️ AI 回應為空，請稍後再試。"

        contents.append(candidate.content)

        # 找出 function_call parts
        fc_parts = [p for p in candidate.content.parts if p.function_call]

        if fc_parts:
            # 執行所有工具呼叫，回傳結果
            tool_response_parts = []
            for p in fc_parts:
                fc = p.function_call
                args = dict(fc.args) if fc.args else {}
                result = _execute_tool(fc.name, args, bot_id, session)
                logging.info(f"[Assistant] Tool call: {fc.name}({list(args.keys())}) → {result[:80]}")
                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result}
                        )
                    )
                )
            contents.append(types.Content(role="tool", parts=tool_response_parts))

        else:
            # 最終文字回覆
            final_text = "".join(
                p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
            ).strip()

            if not final_text:
                final_text = "處理完成，請確認設定是否正確。"

            # 儲存對話歷史
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": final_text})
            session["history"] = history
            session_store.save(session_id, session)

            logging.info(f"[Assistant] Done in {iteration + 1} iterations for bot {bot_id[:8]}")
            return final_text

    # 超過最大迭代次數
    return "⚠️ 處理超時，請重新描述你的需求。"
