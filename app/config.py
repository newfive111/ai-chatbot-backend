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

# Lemon Squeezy Variant IDs（在後台 Products → Variants 取得）
LS_VARIANT_PRO_MONTHLY  = os.getenv("LS_VARIANT_PRO_MONTHLY", "")
LS_VARIANT_PRO_ANNUAL   = os.getenv("LS_VARIANT_PRO_ANNUAL", "")
LS_VARIANT_BIZ_MONTHLY  = os.getenv("LS_VARIANT_BIZ_MONTHLY", "")
LS_VARIANT_BIZ_ANNUAL   = os.getenv("LS_VARIANT_BIZ_ANNUAL", "")
