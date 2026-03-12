from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")

# Lemon Squeezy
LS_API_KEY              = os.getenv("LS_API_KEY", "")
LS_WEBHOOK_SECRET       = os.getenv("LS_WEBHOOK_SECRET", "")
LS_STORE_ID             = os.getenv("LS_STORE_ID", "")

# Lemon Squeezy Variant IDs
LS_VARIANT_BOT_MONTHLY      = os.getenv("LS_VARIANT_PRO_MONTHLY", "")       # Bot 訂閱 1290/月
LS_VARIANT_BOT_ANNUAL       = os.getenv("LS_VARIANT_BOT_ANNUAL", "")        # Bot 訂閱 12900/年
LS_VARIANT_BUSINESS_MONTHLY = os.getenv("LS_VARIANT_BUSINESS_MONTHLY", "")  # 商業版 4680/月
LS_VARIANT_BUSINESS_ANNUAL  = os.getenv("LS_VARIANT_BUSINESS_ANNUAL", "")   # 商業版 46800/年
