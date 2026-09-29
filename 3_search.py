# -*- coding: utf-8 -*-
"""3_search.py - 命令行验证混合检索"""
import json, pickle, re
import chromadb
from sentence_transformers import SentenceTransformer

BM25_PKL = "bm25_index.pkl"
CHROMA_DIR = "./chroma_db"
COLLECTION = "financial_reports"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 5

print("加载 BM25...")
with open(BM25_PKL, "rb") as f:
    d = pickle.load(f)
bm25 = d["bm25"]
chunks = d["chunks"]

print("加载向量库...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(COLLECTION)
model = SentenceTransformer(EMBED_MODEL)


def tokenize(t):
    return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", t.lower())


def bm25_search(q, k=TOP_K):
    scores = bm25.get_scores(tokenize(q))
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in idx]


def vector_search(q, k=TOP_K):
    emb = model.encode([q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "content": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
        })
    return out


def rrf(a, b, k=60):
    score, info = {}, {}
    for lst in (a, b):
        for rank, item in enumerate(lst):
            key = item["content"][:80]
            score[key] = score.get(key, 0) + 1.0 / (k + rank + 1)
            info[key] = item
    keys = sorted(score.keys(), key=lambda x: score[x], reverse=True)
    return [info[x] for x in keys][:TOP_K]


def hybrid_search(q):
    return rrf(bm25_search(q), vector_search(q))


if __name__ == "__main__":
    while True:
        q = input("\n查询（q 退出）: ").strip()
        if q.lower() == "q":
            break
        for i, r in enumerate(hybrid_search(q), 1):
            md = r["metadata"]
            print(f"\n[{i}] {md['company']} {md['year']} 第{md['page']}页")
            print(f"    {r['content'][:150]}...")
