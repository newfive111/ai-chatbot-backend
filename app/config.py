from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")

# Stripe
STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Stripe Price IDs（填入你在 Stripe 後台建立的 price_xxx）
STRIPE_PRICE_PRO_MONTHLY      = os.getenv("STRIPE_PRICE_PRO_MONTHLY", "")
STRIPE_PRICE_PRO_ANNUAL       = os.getenv("STRIPE_PRICE_PRO_ANNUAL", "")
STRIPE_PRICE_BIZ_MONTHLY      = os.getenv("STRIPE_PRICE_BIZ_MONTHLY", "")
STRIPE_PRICE_BIZ_ANNUAL       = os.getenv("STRIPE_PRICE_BIZ_ANNUAL", "")
