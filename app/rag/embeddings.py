from typing import List
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embedding(text: str, api_key: str) -> List[float]:
    """依序嘗試各 Gemini embedding 模型，回傳第一個成功的向量"""
    import httpx
    candidates = [
        ("v1beta", "text-embedding-005"),
        ("v1beta", "text-embedding-004"),
        ("v1",     "text-embedding-004"),
    ]
    payload = {"content": {"parts": [{"text": text}]}}
    last_err = None
    for version, model in candidates:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:embedContent?key={api_key}"
        resp = httpx.post(url, json=payload, timeout=30)
        print(f"[Embedding] {model}/{version}: {resp.status_code}", flush=True)
        if resp.is_success and "embedding" in resp.json():
            return resp.json()["embedding"]["values"]
        last_err = f"{model}/{version} {resp.status_code}: {resp.text[:100]}"
    raise ValueError(f"所有 Embedding 模型都失敗：{last_err}")


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
