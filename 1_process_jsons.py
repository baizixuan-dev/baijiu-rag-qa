# -*- coding: utf-8 -*-
"""1_process_jsons.py - 把 data/extracted/*.json 按页切块，输出 chunks.json"""
import os, json, glob, re
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTRACTED_DIR = "./data/extracted"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def parse_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    if len(parts) >= 2 and re.match(r"20\d{2}", parts[1]):
        return parts[0], parts[1]
    m = re.search(r"(20\d{2})", name)
    year = m.group(1) if m else ""
    company = name.replace("半年报", "").replace("年报", "").replace(year, "").strip("_-— ")
    return company, year


def process_single_json(json_path):
    filename = os.path.basename(json_path)
    company, year = parse_filename(filename)
    print(f"\n>>> {filename}  (公司={company}, 年份={year})")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("pages", [])
    print(f"    页数={len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
    )

    chunks = []
    for p in pages:
        page_num = p.get("page", None)
        text = p.get("text", "")
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            chunks.append({
                "content": piece,
                "metadata": {
                    "company": company,
                    "year": year,
                    "page": page_num,
                    "source": filename,
                }
            })
    print(f"    切块数={len(chunks)}")
    return chunks


def main():
    json_files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, "*.json")))
    print(f"共 {len(json_files)} 个 JSON 文件")
    all_chunks = []
    for jf in json_files:
        all_chunks.extend(process_single_json(jf))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n============ 完成 ============")
    print(f"总切块数: {len(all_chunks)}")
    print(f"输出: {OUTPUT_FILE}")
    for i, c in enumerate(all_chunks[:2]):
        print(f"\n[样例{i+1}] {c['metadata']}")
        print(c['content'][:120], "...")


if __name__ == "__main__":
    main()
