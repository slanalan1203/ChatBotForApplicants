from src.rag.retriever import embed, get_collection

def upsert_chunks(chunks: list[dict]):
    coll = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{k: v for k, v in c.items() if k not in ("text", "chunk_id")} for c in chunks]
    embeddings = embed(docs, is_query=False)
    coll.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
