import httpx
import logging


async def send_instagram_message(recipient_id: str, text: str, page_access_token: str) -> int:
    """透過 Meta Graph API 發送 Instagram DM"""
    # Instagram 訊息長度限制 1000 字元
    if len(text) > 1000:
        text = text[:997] + "..."

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://graph.facebook.com/v19.0/me/messages",
            params={"access_token": page_access_token},
            json={
                "recipient": {"id": recipient_id},
                "message": {"text": text}
            },
            timeout=10.0
        )
        if resp.status_code != 200:
            logging.warning(f"[Instagram] Send failed {resp.status_code}: {resp.text[:200]}")
        return resp.status_code


async def reply_instagram_comment(comment_id: str, text: str, page_access_token: str) -> int:
    """透過 Meta Graph API 回覆 Instagram 貼文留言"""
    # 留言回覆長度限制 2000 字元
    if len(text) > 2000:
        text = text[:1997] + "..."

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v19.0/{comment_id}/replies",
            params={"access_token": page_access_token},
            json={"message": text},
            timeout=10.0
        )
        if resp.status_code != 200:
            logging.warning(f"[Instagram] Comment reply failed {resp.status_code}: {resp.text[:200]}")
        return resp.status_code
