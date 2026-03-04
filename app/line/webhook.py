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


async def reply_line_message(reply_token: str, text: str, access_token: str = None):
    """回覆訊息（reply token，只能用一次）"""
    token = access_token or LINE_CHANNEL_ACCESS_TOKEN
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
        )


async def push_line_message(user_id: str, text: str, access_token: str = None) -> int:
    """主動推播訊息（不需要 reply token）"""
    token = access_token or LINE_CHANNEL_ACCESS_TOKEN
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": [{"type": "text", "text": text}]}
        )
        return resp.status_code
