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

ASSISTANT_MODEL = "gemini-2.5-pro-preview-03-25"

ASSISTANT_SYSTEM_PROMPT = """
你是 LazyReply 平台的 AI 設定助手「小懶」。
你的工作是幫助平台用戶設定他們的聊天機器人。

【你能做的事】
1. 撰寫專業的角色設定（系統提示詞）
2. 設定需要收集的客戶資料欄位（如：姓名、電話、需求）
3. 設定歡迎訊息和快速回覆按鈕
4. 查看目前的 Bot 設定

【工作原則】
- 在呼叫任何修改工具前，先用清楚的條列告知用戶你打算做什麼
- 只有當用戶明確確認（說「好」「確認」「沒問題」「ok」等）才執行工具
- 修改完後告知「✅ 已套用」，並建議用戶去「測試對話」確認效果
- 用繁體中文回覆，語氣專業但親切
- 每次只做一件事，不要一次改太多
- 如果用戶描述不清楚，可以主動詢問釐清

【collect_fields 格式】例：["姓名", "電話", "需求"]
【quick_replies 格式】例：[{"label": "了解方案"}, {"label": "預約諮詢"}, {"label": "聯絡我們"}]
【system_prompt 撰寫提示】包含：角色定位、語氣風格、專業領域、禁忌事項
"""

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_bot_config",
        description="取得目前 Bot 的所有設定，包含角色描述、收集欄位、歡迎訊息等。在建議改動前先呼叫這個工具了解現狀。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={}
        )
    ),
    types.FunctionDeclaration(
        name="update_system_prompt",
        description="更新 Bot 的角色設定（系統提示詞）。用戶確認後才呼叫。",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "system_prompt": types.Schema(
                    type=types.Type.STRING,
                    description="完整的系統提示詞內容"
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


def _execute_tool(tool_name: str, tool_args: dict, bot_id: str) -> str:
    """執行工具，直接操作 Supabase"""
    try:
        if tool_name == "get_bot_config":
            r = _sb.table("bots").select(
                "name, system_prompt, collect_fields, sheet_id, welcome_message, quick_replies"
            ).eq("id", bot_id).execute()
            if r.data:
                return json.dumps(r.data[0], ensure_ascii=False)
            return json.dumps({"error": "Bot 不存在"})

        elif tool_name == "update_system_prompt":
            prompt = tool_args.get("system_prompt", "")
            _sb.table("bots").update({"system_prompt": prompt}).eq("id", bot_id).execute()
            return "角色設定已成功更新"

        elif tool_name == "update_collect_fields":
            fields = tool_args.get("fields", [])
            upd: dict = {"collect_fields": fields}
            if tool_args.get("sheet_id"):
                upd["sheet_id"] = tool_args["sheet_id"]
            _sb.table("bots").update(upd).eq("id", bot_id).execute()
            return f"收集欄位已成功更新：{fields}"

        elif tool_name == "update_welcome":
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


def run_assistant(bot_id: str, user_message: str, session_id: str, gemini_api_key: str) -> str:
    """
    執行 AI 助手對話（含 Function Calling 迴圈）
    回傳助手的最終文字回覆
    """
    if not gemini_api_key:
        return "⚠️ 請先在「設定」頁面填入 Gemini API Key，才能使用 AI 助手。"

    client = genai.Client(api_key=gemini_api_key)
    tools = types.Tool(function_declarations=_FUNCTION_DECLARATIONS)

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
                    system_instruction=ASSISTANT_SYSTEM_PROMPT,
                    tools=[tools],
                    max_output_tokens=2048,
                    temperature=0.7
                )
            )
        except Exception as e:
            logging.error(f"[Assistant] Gemini API error: {e}")
            return f"⚠️ AI 服務暫時無法連線，請稍後再試。（{str(e)[:60]}）"

        candidate = response.candidates[0]
        contents.append(candidate.content)

        # 找出 function_call parts
        fc_parts = [p for p in candidate.content.parts if p.function_call]

        if fc_parts:
            # 執行所有工具呼叫，回傳結果
            tool_response_parts = []
            for p in fc_parts:
                fc = p.function_call
                args = dict(fc.args) if fc.args else {}
                result = _execute_tool(fc.name, args, bot_id)
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
