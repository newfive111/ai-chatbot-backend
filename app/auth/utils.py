import jwt
import uuid
from datetime import datetime, timedelta
from app.config import JWT_SECRET


def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise Exception("Token 過期")
    except jwt.InvalidTokenError:
        raise Exception("Token 無效")


def generate_bot_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:16]
