-- 執行一次即可
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id               TEXT PRIMARY KEY,
    stripe_customer_id    TEXT,
    stripe_subscription_id TEXT,
    plan                  TEXT    DEFAULT 'free',   -- 'free' | 'pro' | 'business'
    billing_cycle         TEXT,                      -- 'monthly' | 'annual'
    status                TEXT    DEFAULT 'active',  -- 'active' | 'canceled' | 'past_due'
    current_period_end    TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- 讓後端可以用 service role key 操作
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_full_access" ON subscriptions USING (true) WITH CHECK (true);
