-- Migration: 將 knowledge_chunks embedding 從 vector(1536) 改為 vector(768)
-- 原因：改用 Gemini text-embedding-004（真實語意向量，768 維）
-- 注意：此操作會刪除所有現有知識庫資料（原本是假 embeddings，無效）
--       執行後請重新上傳所有知識庫文件

-- 清空舊的假 embeddings 資料
TRUNCATE TABLE knowledge_chunks;

-- 修改欄位維度
ALTER TABLE knowledge_chunks
  ALTER COLUMN embedding TYPE vector(768);

-- 重建向量索引
DROP INDEX IF EXISTS knowledge_chunks_embedding_idx;
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops);

-- 更新 match_chunks function
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(768),
  bot_id text,
  match_count int
)
RETURNS TABLE(content text, similarity float)
LANGUAGE sql STABLE
AS $$
  SELECT
    content,
    1 - (embedding <=> query_embedding) AS similarity
  FROM knowledge_chunks
  WHERE knowledge_chunks.bot_id = match_chunks.bot_id
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
