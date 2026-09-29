# 白酒行业 2026 半年报 RAG 问答系统

基于检索增强生成（RAG）的白酒行业财报问答系统，覆盖 10 家 A 股白酒上市公司 2026 年半年报。
## 界面截图

### 成功案例

![案例1](screenshot_1.png)
### 翻车案例

![案例2](screenshot_2.png)
## 功能

- PDF 财报全文解析（文本 + 表格 Markdown 化）
- 双路检索：BM25 关键词 + 向量语义（RRF 融合）
- DeepSeek API 生成答案，带页码溯源
- Streamlit 网页界面

## 技术栈

- Python 3.13
- pdfplumber（PDF 解析）
- LangChain TextSplitter（切块）
- ChromaDB（向量库）
- rank_bm25（关键词检索）
- sentence-transformers（BAAI/bge-small-zh-v1.5 嵌入）
- DeepSeek API（对话生成）
- Streamlit（Web 界面）

## 使用

1. 安装依赖：`pip install -r requirements.txt`
2. 设置 API Key：`$env:DEEPSEEK_API_KEY = "sk-xxx"`
3. 依次运行：

python 0_pdf_to_json.py
python 1_process_jsons.py
python 2_build_index.py
python 3_search.py
streamlit run 4_app.py

## 数据来源

10 家白酒上市公司 2026 年半年度报告（巨潮资讯网）：

- 贵州茅台、五粮液、泸州老窖、洋河股份、山西汾酒
- 古井贡酒、今世缘、口子窖、水井坊、酒鬼酒

## 项目结构

