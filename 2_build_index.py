# -*- coding: utf-8 -*-
"""2_build_index.py - 建 BM25 + 向量索引"""
import os, json, pickle, re
import chromadb
from rank_bm25 import BM25Okapi
from tqdm import tqdm

CHUNKS_FILE = "chunks.json"
BM25_PKL = "bm25_index.pkl"
CHROMA_DIR = "./chroma_db"
COLLECTION = "financial_reports"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def tokenize(text):
    return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower())


def build_bm25(chunks):
    print("\n[1/2] 构建 BM25 索引...")
    corpus = [c["content"] for c in chunks]
    tokenized = [tokenize(t) for t in tqdm(corpus, desc="分词")]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"    BM25 完成，语料数={len(corpus)}")


def build_vector(chunks):
    print("\n[2/2] 构建 ChromaDB 向量索引...")
    print(f"    加载 Embedding 模型: {EMBED_MODEL}（首次运行会下载约 100MB）")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    texts = [c["content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    BATCH = 64
    for i in tqdm(range(0, len(texts), BATCH), desc="写入向量"):
        bt = texts[i:i+BATCH]
        bm = metas[i:i+BATCH]
        bi = ids[i:i+BATCH]
        emb = model.encode(bt, normalize_embeddings=True).tolist()
        col.add(documents=bt, embeddings=emb, metadatas=bm, ids=bi)

    print(f"    向量库完成，条数={col.count()}")


def main():
    if not os.path.exists(CHUNKS_FILE):
        print(f"[错误] 找不到 {CHUNKS_FILE}，请先跑 1_process_jsons.py")
        return
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载 chunks: {len(chunks)} 条")
    build_bm25(chunks)
    build_vector(chunks)
    print("\n============ 索引构建完成 ============")


if __name__ == "__main__":
    main()
