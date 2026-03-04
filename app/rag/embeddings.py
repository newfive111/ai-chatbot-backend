import anthropic
import json
from typing import List
from app.config import ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embedding(text: str) -> List[float]:
    """用 Voyage AI（Anthropic 的 embedding）產生向量"""
    response = client.beta.messages.batches  # placeholder
    # 暫時用簡單的 hash-based mock，之後換 voyage-ai
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    # 產生 1536 維的假向量（之後換真實 embedding）
    seed = int(h, 16)
    import random
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(1536)]


def store_chunks(bot_id: str, chunks: List[str]) -> bool:
    """把文字塊和向量存到 Supabase"""
    embeddings = [get_embedding(chunk) for chunk in chunks]
    rows = [
        {
            "bot_id": bot_id,
            "content": chunk,
            "embedding": embedding
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    result = supabase.table("knowledge_chunks").insert(rows).execute()
    return True


def search_similar_chunks(bot_id: str, query: str, top_k: int = 5) -> List[str]:
    """用問題去找最相關的知識庫內容"""
    query_embedding = get_embedding(query)
    result = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "bot_id": bot_id,
            "match_count": top_k
        }
    ).execute()
    return [row["content"] for row in result.data]
