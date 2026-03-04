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

【防失憶原則】
每次回覆前，先檢查完整的對話歷史。客戶已提供過的資訊直接使用，嚴禁重複詢問同一件事。

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

【DATA_SAVE 完成交接語】
若你輸出了 DATA_SAVE，必須在同一則訊息加上：
「好的，資料都登記完成了！請您稍等一下，我現在把資料轉給主管確認，待會由專員直接回覆您。」"""


def _get_system_prompt(
    bot_name: str,
    context: str,
    custom_system_prompt: Optional[str] = None,
    has_sheet: bool = False
) -> str:
    """
    組合最終 system prompt：[角色設定] + [知識庫] + [平台底層規則]
    平台規則永遠注入，不受客戶自訂 prompt 影響。
    """
    role_section = custom_system_prompt.strip() if (custom_system_prompt and custom_system_prompt.strip()) \
                   else DEFAULT_ROLE_PROMPT.format(bot_name=bot_name)

    kb_section = f"\n\n【知識庫參考資料】\n{context}" if context else ""

    rules = PLATFORM_RULES
    if not has_sheet:
        rules = rules.split("【DATA_SAVE 完成交接語】")[0].rstrip()

    return f"{role_section}{kb_section}{rules}"


# ──────────────────────────────────────
# 模型呼叫（Gemini 2.5 Flash）
# ──────────────────────────────────────

def _call_ai(api_key: str, system_prompt: str, history: list, question: str) -> str:
    """呼叫 Gemini 2.5 Flash"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # 把 {role: user/assistant} 格式轉成 Gemini 的 {role: user/model}
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


# ──────────────────────────────────────
# DATA_SAVE 偵測
# ──────────────────────────────────────

def _extract_and_save_data(raw_reply: str, sheet_id: str, session_id: str) -> Tuple[str, bool]:
    """
    偵測 Claude 回覆中的 DATA_SAVE: {...}，寫入 Google Sheet。
    回傳 (清理後文字, 是否找到DATA_SAVE)
    """
    pattern = r'DATA_SAVE:\s*(\{.*?\})'
    match = re.search(pattern, raw_reply, re.DOTALL)
    if not match:
        return raw_reply, False

    json_str = match.group(1)
    try:
        data: dict = json.loads(json_str)
        fields = list(data.keys())
        try:
            from app.sheets.client import upsert_row
            upsert_row(sheet_id, session_id, fields, data)
            logging.info(f"[Sheet] DATA_SAVE written session={session_id[:8]} fields={fields}")
        except Exception as e:
            logging.warning(f"[Sheet] DATA_SAVE write failed: {e}")
    except json.JSONDecodeError as e:
        logging.warning(f"[Engine] DATA_SAVE JSON parse error: {e} | raw={json_str[:100]}")

    cleaned = re.sub(r'\n?DATA_SAVE:\s*\{.*?\}\n?', '', raw_reply, flags=re.DOTALL).strip()
    return cleaned, True


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
) -> str:
    # 強制使用客戶自己的 Key，不 fallback 到平台 Key
    # 平台費用不代墊，客戶必須 BYOK
    if not api_key:
        raise Exception("NO_API_KEY")

    relevant_chunks = search_similar_chunks(bot_id, question, top_k=5)
    context = "\n\n".join(relevant_chunks)
    system_prompt = _get_system_prompt(
        bot_name, context, custom_system_prompt,
        has_sheet=bool(sheet_id)
    )

    # ── 路徑 A：有自訂 prompt → Claude 全程主導 ──
    if custom_system_prompt and custom_system_prompt.strip():
        if session_id:
            session = session_store.get_or_create(session_id)

            # 🔇 已交接 → 回等待語，不打 Claude
            if session.get("status") == "handed_off":
                logging.info(f"[Engine] {session_id[:8]} handed_off → holding reply")
                return handoff_reply or HANDOFF_REPLY

            history = session.get("history", [])
        else:
            session = None
            history = []

        raw_reply = _call_ai(api_key, system_prompt, history, question)

        if sheet_id and session_id:
            clean_reply, data_saved = _extract_and_save_data(raw_reply, sheet_id, session_id)
            if data_saved:
                session["status"] = "handed_off"
                logging.info(f"[Engine] {session_id[:8]} → handed_off (DATA_SAVE)")
        else:
            clean_reply = re.sub(r'\n?DATA_SAVE:\s*\{.*?\}\n?', '', raw_reply, flags=re.DOTALL).strip()
            data_saved = False

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
                    upsert_row(sheet_id, session_id, fields, collected)
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
