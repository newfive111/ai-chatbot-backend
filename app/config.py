from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")

# 藍新金流 (NewebPay)
NEWEBPAY_MERCHANT_ID = os.getenv("NEWEBPAY_MERCHANT_ID", "MS323238228")   # 換成你的商店代號
NEWEBPAY_HASH_KEY    = os.getenv("NEWEBPAY_HASH_KEY",    "n_sdk_test_hash_key_1234")  # 換成你的 HashKey
NEWEBPAY_HASH_IV     = os.getenv("NEWEBPAY_HASH_IV",     "n_sdk_test_iv12")            # 換成你的 HashIV
NEWEBPAY_SANDBOX     = os.getenv("NEWEBPAY_SANDBOX", "true").lower() == "true"

# 方案金額
PRICE_BOT_MONTHLY      = int(os.getenv("PRICE_BOT_MONTHLY",      "1290"))
PRICE_BOT_ANNUAL       = int(os.getenv("PRICE_BOT_ANNUAL",       "12900"))
PRICE_BUSINESS_MONTHLY = int(os.getenv("PRICE_BUSINESS_MONTHLY", "4680"))
PRICE_BUSINESS_ANNUAL  = int(os.getenv("PRICE_BUSINESS_ANNUAL",  "46800"))

# Lemon Squeezy（保留，暫時停用）
LS_API_KEY              = os.getenv("LS_API_KEY", "")
LS_WEBHOOK_SECRET       = os.getenv("LS_WEBHOOK_SECRET", "")
LS_STORE_ID             = os.getenv("LS_STORE_ID", "")
