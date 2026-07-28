-- 客戶名單：收集完成（DATA_SAVE）的資料
-- 在 Supabase SQL Editor 執行一次即可
CREATE TABLE IF NOT EXISTS submissions (
  id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  bot_id       UUID NOT NULL,
  session_id   TEXT,
  display_name TEXT,
  data         JSONB DEFAULT '{}',   -- 結構化欄位 {"姓名":"...","電話":"..."}
  card_text    TEXT,                 -- 依範本排版好、可直接複製的資料卡
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_submissions_bot_created
  ON submissions (bot_id, created_at DESC);

-- 啟用 RLS（後端用 service key 存取會 bypass；不開放匿名/前端直接讀寫）
ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;

-- bot 設定新增「資料卡格式範本」欄位（可為空 → 用預設每欄一行）
ALTER TABLE bots ADD COLUMN IF NOT EXISTS card_template TEXT;
