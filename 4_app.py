# -*- coding: utf-8 -*-
"""4_app.py - Streamlit 问答页面 + DeepSeek 生成"""
import json, pickle, re, os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============ 配置 ============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-你的key填这里")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

BM25_PKL = "bm25_index.pkl"
CHROMA_DIR = "./chroma_db"
COLLECTION = "financial_reports"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 5
# ==============================


@st.cache_resource
def load_resources():
    with open(BM25_PKL, "rb") as f:
        d = pickle.load(f)
    bm25, chunks = d["bm25"], d["chunks"]
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION)
    model = SentenceTransformer(EMBED_MODEL)
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return bm25, chunks, col, model, llm


def tokenize(t):
    return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", t.lower())


def bm25_search(q, bm25, chunks, k=TOP_K):
    scores = bm25.get_scores(tokenize(q))
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in idx]


def vector_search(q, col, model, k=TOP_K):
    emb = model.encode([q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    return [{"content": res["documents"][0][i],
             "metadata": res["metadatas"][0][i]}
            for i in range(len(res["ids"][0]))]


def rrf(a, b, k=60, top=TOP_K):
    score, info = {}, {}
    for lst in (a, b):
        for rank, item in enumerate(lst):
            key = item["content"][:80]
            score[key] = score.get(key, 0) + 1.0 / (k + rank + 1)
            info[key] = item
    keys = sorted(score.keys(), key=lambda x: score[x], reverse=True)
    return [info[x] for x in keys][:top]


def build_prompt(query, contexts):
    ctx_text = ""
    for i, c in enumerate(contexts, 1):
        md = c["metadata"]
        ctx_text += f"\n[资料{i}] 来源：{md['company']} {md['year']}年半年报 第{md['page']}页\n{c['content']}\n"
    return f"""你是一位财务分析师。请**只根据**下面提供的资料回答问题，不要编造。
如果资料中找不到答案，请明确说"资料中未提及"。
回答后必须列出用到的资料来源（公司、页码）。

资料：
{ctx_text}

问题：{query}

请用中文回答，结构化清晰。"""


def main():
    st.set_page_config(page_title="白酒半年报问答", layout="wide")
    st.title("🍶 白酒行业 2026 半年报问答")

    bm25, chunks, col, model, llm = load_resources()

    query = st.text_input("请输入你的问题（例如：贵州茅台2026上半年的营业收入是多少？）")

    if st.button("提问") and query:
        with st.spinner("检索中..."):
            bm = bm25_search(query, bm25, chunks)
            vc = vector_search(query, col, model)
            ctx = rrf(bm, vc)

        with st.spinner("生成回答中..."):
            prompt = build_prompt(query, ctx)
            resp = llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content

        st.markdown("### 💡 回答")
        st.write(answer)

        st.markdown("### 📚 参考来源")
        for i, c in enumerate(ctx, 1):
            md = c["metadata"]
            with st.expander(f"[{i}] {md['company']} 第{md['page']}页"):
                st.write(c["content"][:800])


if __name__ == "__main__":
    main()