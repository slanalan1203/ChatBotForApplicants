import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION = "hse_admission"
METHOD = os.environ.get("RETRIEVE_METHOD", "dense")
RRF_K = 60

_model = None
_bm25 = None
_bm25_ids = None
_bm25_docs = None
_bm25_metas = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model

def embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if "e5" in MODEL.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    return get_model().encode(texts, normalize_embeddings=True).tolist()

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def _build_bm25():
    global _bm25, _bm25_ids, _bm25_docs, _bm25_metas
    from rank_bm25 import BM25Okapi
    coll = get_collection()
    data = coll.get(include=["documents", "metadatas"])
    _bm25_ids = data["ids"]
    _bm25_docs = data["documents"]
    _bm25_metas = data["metadatas"]
    tokenized = [_tokenize(d) for d in _bm25_docs]
    _bm25 = BM25Okapi(tokenized)

def dense_retrieve(query: str, k: int = 5) -> list[dict]:
    coll = get_collection()
    res = coll.query(query_embeddings=embed([query], is_query=True), n_results=k)
    return [
        {"id": id_, "text": doc, "metadata": meta, "distance": dist}
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

def bm25_retrieve(query: str, k: int = 5) -> list[dict]:
    if _bm25 is None:
        _build_bm25()
    scores = _bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [{"id": _bm25_ids[i], "text": _bm25_docs[i], "metadata": _bm25_metas[i]} for i in top_idx]

def hybrid_retrieve(query: str, k: int = 5) -> list[dict]:
    dense = dense_retrieve(query, k=k * 4)
    bm25 = bm25_retrieve(query, k=k * 4)
    rrf = {}
    for rank, item in enumerate(dense, start=1):
        rrf[item["id"]] = rrf.get(item["id"], 0) + 1 / (RRF_K + rank)
    for rank, item in enumerate(bm25, start=1):
        rrf[item["id"]] = rrf.get(item["id"], 0) + 1 / (RRF_K + rank)
    by_id = {it["id"]: it for it in dense + bm25}
    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [by_id[i] for i in top_ids]

def rewrite_query(query: str) -> str:
    import ollama
    model = os.environ.get("OLLAMA_MODEL", "mistral:latest")
    prompt = (
        f"Перепиши краткий вопрос абитуриента в более развёрнутую формулировку, "
        f"добавив 1-2 ключевых синонима. Только переписанный вопрос, без комментариев.\n\n"
        f"Вопрос: {query}\n\nРазвёрнутая формулировка:"
    )
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 80},
    )
    return resp["message"]["content"].strip().split("\n")[0]

def hybrid_rewrite_retrieve(query: str, k: int = 5) -> list[dict]:
    expanded = rewrite_query(query)
    return hybrid_retrieve(f"{query} {expanded}", k=k)

RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "")
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None and RERANKER_MODEL:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    rer = get_reranker()
    if rer is None or not candidates:
        return candidates[:k]
    pairs = [(query, c["text"]) for c in candidates]
    scores = rer.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:k]]

def retrieve(query: str, k: int = 5) -> list[dict]:
    if METHOD == "bm25":
        out = bm25_retrieve(query, k * 4 if RERANKER_MODEL else k)
    elif METHOD == "hybrid":
        out = hybrid_retrieve(query, k * 4 if RERANKER_MODEL else k)
    elif METHOD == "hybrid_rewrite":
        out = hybrid_rewrite_retrieve(query, k * 4 if RERANKER_MODEL else k)
    else:
        out = dense_retrieve(query, k * 4 if RERANKER_MODEL else k)
    if RERANKER_MODEL:
        out = rerank(query, out, k)
    return out
