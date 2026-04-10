import hashlib
import hmac
import base64
import httpx
from app.config import LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET


def verify_line_signature(body: bytes, signature: str, channel_secret: str = None) -> bool:
    """驗證 LINE webhook 簽名，支援 per-bot Channel Secret"""
    secret = channel_secret or LINE_CHANNEL_SECRET
    if not secret:
        return False
    digest = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _build_quick_reply(quick_replies: list) -> dict | None:
    """將 quick_replies ([{label}]) 轉為 LINE quickReply 格式"""
    if not quick_replies:
        return None
    items = []
    for q in quick_replies[:13]:  # LINE 最多 13 個
        label = (q.get("label") if isinstance(q, dict) else str(q))[:20]  # label 最多 20 字
        if label:
            items.append({
                "type": "action",
                "action": {
                    "type": "message",
                    "label": label,
                    "text": label,
                }
            })
    return {"items": items} if items else None


async def reply_line_message(reply_token: str, text: str, access_token: str = None, quick_replies: list = None):
    """回覆訊息（reply token，只能用一次）"""
    token = access_token or LINE_CHANNEL_ACCESS_TOKEN
    msg: dict = {"type": "text", "text": text}
    qr = _build_quick_reply(quick_replies)
    if qr:
        msg["quickReply"] = qr
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"replyToken": reply_token, "messages": [msg]}
        )


async def push_line_message(user_id: str, text: str, access_token: str = None, quick_replies: list = None) -> int:
    """主動推播訊息（不需要 reply token）"""
    token = access_token or LINE_CHANNEL_ACCESS_TOKEN
    msg: dict = {"type": "text", "text": text}
    qr = _build_quick_reply(quick_replies)
    if qr:
        msg["quickReply"] = qr
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": [msg]}
        )
        return resp.status_code
