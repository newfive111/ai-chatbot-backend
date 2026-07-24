from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# LINE Login（快速登入用，與上面的 Messaging API channel 不同）
LINE_LOGIN_CHANNEL_ID = os.getenv("LINE_LOGIN_CHANNEL_ID", "")
LINE_LOGIN_CHANNEL_SECRET = os.getenv("LINE_LOGIN_CHANNEL_SECRET", "")

# 管理用 LINE bot（員工遠端接手/操作客戶 bot 用）
ADMIN_LINE_CHANNEL_SECRET = os.getenv("ADMIN_LINE_CHANNEL_SECRET", "")
ADMIN_LINE_CHANNEL_ACCESS_TOKEN = os.getenv("ADMIN_LINE_CHANNEL_ACCESS_TOKEN", "")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 環境變數未設定，請在 Railway 加入此變數")

# 平台管理的 Claude API Key（給後台設定助手「小懶」用；沒設就自動退回租客的 Gemini Key）
PLATFORM_ANTHROPIC_KEY = os.getenv("PLATFORM_ANTHROPIC_KEY", "")
ASSISTANT_CLAUDE_MODEL = os.getenv("ASSISTANT_CLAUDE_MODEL", "claude-sonnet-4-5")
