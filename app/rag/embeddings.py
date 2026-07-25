from typing import List
from app.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _local_embedding(text: str) -> List[float]:
    """
    本地 feature-hashing 向量（768 維）。
    用於 Gemini embedding API 無法存取時的備援。
    相似文字因共享 token hash 而有相近的 cosine 相似度。
    """
    import hashlib, math
    text = text.lower()
    vec = [0.0] * 768
    words = text.split()
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 768
        vec[idx] += 1.0
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i + 1]
        idx = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % 768
        vec[idx] += 0.5
    for i in range(len(text) - 2):
        idx = int(hashlib.md5(text[i:i+3].encode()).hexdigest(), 16) % 768
        vec[idx] += 0.3
    magnitude = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / magnitude for x in vec]


def get_embedding(text: str, api_key: str) -> List[float]:
    """先嘗試 Gemini embedding API，失敗則用本地 hash 向量"""
    import httpx
    if api_key:
        for version, model in [("v1beta", "text-embedding-005"), ("v1beta", "text-embedding-004")]:
            try:
                url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:embedContent?key={api_key}"
                resp = httpx.post(url, json={"content": {"parts": [{"text": text}]}}, timeout=15)
                if resp.is_success and "embedding" in resp.json():
                    return resp.json()["embedding"]["values"]
            except Exception:
                pass
    print("[Embedding] Gemini API 不可用，使用本地 hash 向量", flush=True)
    return _local_embedding(text)


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


def bot_has_knowledge(bot_id: str) -> bool:
    """這個 bot 有沒有任何知識庫 chunk（便宜的 count，避免無知識庫還白跑 embedding API）。"""
    try:
        r = supabase.table("knowledge_chunks").select("id", count="exact") \
            .eq("bot_id", bot_id).limit(1).execute()
        return bool(r.count and r.count > 0)
    except Exception:
        return True  # 查詢失敗時保守走原路，不誤殺檢索


# 相似度門檻：低於此值視為不相關，不塞進 prompt（1 - cosine 距離，範圍約 0~1）
SIMILARITY_THRESHOLD = 0.55


def search_similar_chunks(bot_id: str, query: str, top_k: int = 5, api_key: str = "") -> List[str]:
    """用問題去找最相關的知識庫內容；只回傳相似度過門檻的段落。"""
    # ① 沒知識庫就直接跳過，省掉 embedding API + RPC
    if not bot_has_knowledge(bot_id):
        return []

    query_embedding = get_embedding(query, api_key)
    result = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "bot_id": bot_id,
            "match_count": top_k
        }
    ).execute()
    rows = result.data or []

    # ② 相似度門檻過濾：不夠相關的段落不塞進 prompt，避免雜訊干擾回覆
    filtered = [r["content"] for r in rows
                if r.get("similarity") is None or r["similarity"] >= SIMILARITY_THRESHOLD]
    # 全部都被濾掉時，至少保留最相關的一段（避免門檻誤殺導致完全沒參考）
    if not filtered and rows:
        top = max(rows, key=lambda r: r.get("similarity") or 0)
        if (top.get("similarity") or 0) >= SIMILARITY_THRESHOLD * 0.8:
            filtered = [top["content"]]
    return filtered
