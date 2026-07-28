-- 客戶名單「已處理」標記
-- 在 Supabase SQL Editor 執行一次即可
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS handled BOOLEAN DEFAULT false;
