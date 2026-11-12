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

# 平台管理的 Gemini API Key，只用於知識庫 embedding（Claude 無 embedding API）。
# 平台統一付費，租客不必再自備 Gemini Key；沒設就退回租客 Key、再退本地 hash 向量。
PLATFORM_GEMINI_KEY = os.getenv("PLATFORM_GEMINI_KEY", "")

# LIFF 快速填表（目前僅 697 使用，靠 bots.liff_intake_enabled 開關）。
# PLATFORM_LIFF_ID：平台級 LIFF app 的 ID（所有要用的 bot 共用，靠網址帶 bot/sid 區分）。
# PLATFORM_LIFF_CHANNEL_ID：該 LIFF 所屬 LINE Login channel 的 channel ID，用來驗證表單送回的 idToken；沒設則略過驗證。
PLATFORM_LIFF_ID = os.getenv("PLATFORM_LIFF_ID", "")
PLATFORM_LIFF_CHANNEL_ID = os.getenv("PLATFORM_LIFF_CHANNEL_ID", "")
