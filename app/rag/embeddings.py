from typing import List
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embedding(text: str, api_key: str) -> List[float]:
    """使用 Gemini embedding-001 產生語意向量（768 維）"""
    import httpx
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    resp = httpx.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def store_chunks(bot_id: str, chunks: List[str], api_key: str = "") -> bool:
    """把文字塊和向量存到 Supabase"""
    embeddings = [get_embedding(chunk, api_key) for chunk in chunks]
    rows = [
        {
            "bot_id": bot_id,
            "content": chunk,
            "embedding": embedding
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    supabase.table("knowledge_chunks").insert(rows).execute()
    return True


def search_similar_chunks(bot_id: str, query: str, top_k: int = 5, api_key: str = "") -> List[str]:
    """用問題去找最相關的知識庫內容"""
    query_embedding = get_embedding(query, api_key)
    result = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "bot_id": bot_id,
            "match_count": top_k
        }
    ).execute()
    return [row["content"] for row in result.data]
