import jwt
import uuid
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.config import JWT_SECRET


def create_token(user_id: str, email: str = "", created_at: str = "") -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "created_at": created_at,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token 過期，請重新登入")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Token 無效，請重新登入")


def generate_bot_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:16]
