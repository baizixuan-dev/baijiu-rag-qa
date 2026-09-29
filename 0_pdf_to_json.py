# -*- coding: utf-8 -*-
"""0_pdf_to_json.py - 把 data/*.pdf 逐页提取为 data/extracted/*.json"""
import os, json, glob
import pdfplumber
from tqdm import tqdm

PDF_DIR = "./data"
OUT_DIR = "./data/extracted"


def table_to_markdown(table):
    if not table or not table[0]:
        return ""
    cleaned = []
    for row in table:
        cleaned.append([("" if c is None else str(c).replace("\n", " ").strip()) for c in row])
    if len(cleaned) < 2:
        return ""
    max_cols = max(len(r) for r in cleaned)
    for r in cleaned:
        while len(r) < max_cols:
            r.append("")
    lines = ["| " + " | ".join(cleaned[0]) + " |",
             "| " + " | ".join(["---"] * max_cols) + " |"]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            parts = []
            try:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            except Exception:
                pass
            try:
                for i, tb in enumerate(page.extract_tables() or []):
                    md = table_to_markdown(tb)
                    if md:
                        parts.append(f"\n[表格 {i+1}]\n{md}\n")
            except Exception:
                pass
            pages.append({"page": idx, "text": "\n".join(parts)})
    return pages


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_files:
        print(f"[错误] {PDF_DIR} 下没有 PDF")
        return
    print(f"共 {len(pdf_files)} 个 PDF\n")
    for pdf_path in pdf_files:
        name = os.path.basename(pdf_path)
        out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".json")
        if os.path.exists(out_path):
            print(f"  跳过（已存在）: {name}")
            continue
        print(f">>> 解析: {name}")
        try:
            pages = extract_pdf(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"page_count": len(pages), "pages": pages},
                          f, ensure_ascii=False, indent=2)
            print(f"    页数={len(pages)}  ->  {out_path}")
        except Exception as e:
            print(f"    失败: {e}")
    print("\n完成。检查 ./data/extracted/")


if __name__ == "__main__":
    main()
