-- 啟用 pgvector 擴充功能
create extension if not exists vector;

-- 用戶的 Bot 列表
create table bots (
  id text primary key,
  user_id uuid references auth.users(id),
  name text not null,
  created_at timestamp default now()
);

-- 知識庫內容（向量化後存在這裡）
-- 使用 Gemini text-embedding-004（768 維）
create table knowledge_chunks (
  id uuid default gen_random_uuid() primary key,
  bot_id text references bots(id) on delete cascade,
  content text not null,
  embedding vector(768),
  created_at timestamp default now()
);

-- 對話記錄
create table conversations (
  id uuid default gen_random_uuid() primary key,
  bot_id text references bots(id) on delete cascade,
  question text not null,
  answer text not null,
  created_at timestamp default now()
);

-- 建立向量搜尋索引
create index on knowledge_chunks using ivfflat (embedding vector_cosine_ops);

-- 向量搜尋 function（給 RAG 用）
create or replace function match_chunks(
  query_embedding vector(768),
  bot_id text,
  match_count int
)
returns table(content text, similarity float)
language sql stable
as $$
  select
    content,
    1 - (embedding <=> query_embedding) as similarity
  from knowledge_chunks
  where knowledge_chunks.bot_id = match_chunks.bot_id
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- RLS 安全性設定
alter table bots enable row level security;
alter table knowledge_chunks enable row level security;
alter table conversations enable row level security;
alter table orders enable row level security;
alter table bot_subscriptions enable row level security;
alter table sessions enable row level security;
alter table fortune_usage enable row level security;
alter table bot_settings_history enable row level security;
alter table assistant_config enable row level security;

create policy "用戶只能看自己的 bot"
  on bots for all using (auth.uid() = user_id);

create policy "知識庫跟著 bot 走"
  on knowledge_chunks for all using (
    exists (select 1 from bots where bots.id = knowledge_chunks.bot_id and bots.user_id = auth.uid())
  );

create policy "對話記錄跟著 bot 走"
  on conversations for all using (
    exists (select 1 from bots where bots.id = conversations.bot_id and bots.user_id = auth.uid())
  );

create policy "用戶只能看自己的訂單"
  on orders for all using (user_id = auth.uid()::text);

create policy "用戶只能看自己的訂閱"
  on bot_subscriptions for all using (user_id::text = auth.uid()::text);

create policy "設定歷史跟著 bot 走"
  on bot_settings_history for all using (
    exists (select 1 from bots where bots.id = bot_settings_history.bot_id and bots.user_id = auth.uid())
  );

-- sessions / fortune_usage / assistant_config 只由 backend service role 存取
create policy "禁止匿名存取 sessions"
  on sessions for all using (false);

create policy "禁止匿名存取 fortune_usage"
  on fortune_usage for all using (false);

create policy "禁止匿名存取 assistant_config"
  on assistant_config for all using (false);
