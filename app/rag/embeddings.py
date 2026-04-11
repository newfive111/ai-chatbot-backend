from typing import List
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embedding(text: str, api_key: str) -> List[float]:
    """使用 Gemini text-embedding-004 產生真實語意向量（768 維）"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


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
