

"""
增量入库器：新增文件 → 解析 → 切分 → 追加入库（不重写全量）
写入顺序：MySQL（正本，拿 id）→ Milvus（向量）→ ES（关键词）→ documents 登记表

用法：
  python ingest.py                    # 扫描 documents/ 目录，处理所有未入库/有变更的文件
  python ingest.py <文件路径> [...]    # 处理指定文件（不在 documents/ 的会先复制进去）

文件级幂等（sha256）：
  新文件      → 追加
  哈希未变    → 跳过
  哈希变了    → 按来源清理旧数据后重插（替换）
"""
import os
import re
import shutil
import sys

sys.stdout.reconfigure(errors="replace")

import jieba.analyse
from elasticsearch.helpers import bulk
from pymilvus import connections

import store
import milvus_config as config

MAX_TEXT_CHARS = 7000  # 单段上限（embedding 输入 / MySQL TEXT 字段保护）


# ==================== 解析器（可插拔后端） ====================

def parse_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _ocr_pdf(path):
    """图片版/扫描版 PDF 的 OCR 后端：渲染每页为图 → qwen-vl-ocr 识别"""
    import base64

    import pymupdf
    from openai import OpenAI

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.EMBEDDING_BASE_URL)
    doc = pymupdf.open(path)
    texts = []
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=120)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model="qwen-vl-ocr", max_tokens=4000,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "请完整识别并输出图中的所有文字，保持原有顺序。"
                                                  "表格请用结构化语法输出（优先 Markdown 表格；含合并单元格/多级表头时可用 LaTeX tabular）。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]}],
                )
                texts.append(resp.choices[0].message.content or "")
                break
            except Exception as e:
                if attempt == 1:
                    print(f"    第 {i} 页 OCR 失败: {e}")
        print(f"    OCR {i}/{len(doc)} 页")
    return "\n\n".join(texts)


def parse_pdf(path):
    """PDF 智能解析：先用 pdfplumber 抽文字；文字量过少（图片版/扫描版）自动切换 OCR"""
    import pdfplumber

    pages = []
    n_pages = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    content = "\n\n".join(pages)

    # 判定：平均每页文字太少 → 视为图片版，走 OCR
    if len(content) < max(100, n_pages * 50):
        print(f"  文字量过少（{len(content)} 字符 / {n_pages} 页），切换 qwen-vl-ocr 识别...")
        content = _ocr_pdf(path)
    return content


def parse_docx(path):
    """Word 文档：按文档流顺序输出段落与表格（表格按行拼成文本，合并单元格去重）"""
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(path)
    parts = []
    # 按 body 实际顺序遍历：段落与表格交错出现时保持原文顺序
    for child in d.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, d).text.strip()
            if text:
                parts.append(text)
        elif isinstance(child, CT_Tbl):
            for row in Table(child, d).rows:
                seen, cells = set(), []
                for c in row.cells:  # 水平合并的单元格会被重复列出，去重
                    if c._tc not in seen:
                        seen.add(c._tc)
                        cells.append(c.text.strip())
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n\n".join(parts)


PARSERS = {".md": parse_markdown, ".txt": parse_txt, ".pdf": parse_pdf, ".docx": parse_docx}


# ==================== 切分器（两种策略） ====================

def split_by_heading(content, source):
    """Markdown 标题切分：按 ## / ###（沿用原项目逻辑）"""
    chunks = []
    h2_sections = re.split(r"\n## ", content)

    for i, h2_section in enumerate(h2_sections):
        if i == 0 and not h2_section.startswith("## "):
            if h2_section.strip():
                chunks.append({
                    "text": h2_section.strip(),
                    "source": source,
                    "section": "前言",
                })
            continue

        lines = h2_section.split("\n")
        h2_title = lines[0].strip()
        h2_content = "\n".join(lines[1:]) if len(lines) > 1 else ""

        if "### " in h2_content:
            h3_sections = re.split(r"\n### ", h2_content)
            for j, h3_section in enumerate(h3_sections):
                if j == 0 and h3_section.strip():
                    chunks.append({
                        "text": f"## {h2_title}\n{h3_section.strip()}",
                        "source": source,
                        "section": h2_title,
                    })
                elif j > 0:
                    h3_lines = h3_section.split("\n")
                    h3_title = h3_lines[0].strip()
                    h3_content = "\n".join(h3_lines[1:]) if len(h3_lines) > 1 else ""
                    chunks.append({
                        "text": f"## {h2_title}\n### {h3_title}\n{h3_content.strip()}",
                        "source": source,
                        "section": f"{h2_title} - {h3_title}",
                    })
        else:
            chunks.append({
                "text": f"## {h2_title}\n{h2_content.strip()}",
                "source": source,
                "section": h2_title,
            })
    return chunks


def _split_long_text(text, target):
    """超长文本（无空行）：按句子边界切成 target 长度附近的小段"""
    sents = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > target:
            out.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        out.append(cur)
    return out


def split_by_window(content, source, target=None, overlap_ratio=None):
    """通用窗口切分：自然段聚合到目标长度，相邻段带重叠（供无标题的 pdf/txt 用）"""
    target = target or config.WINDOW_TARGET_CHARS
    overlap_ratio = overlap_ratio or config.WINDOW_OVERLAP_RATIO
    overlap_chars = int(target * overlap_ratio)

    # 先按空行拆自然段；超长段再按句子细拆
    paragraphs = []
    for p in re.split(r"\n\s*\n", content):
        p = p.strip()
        if not p:
            continue
        if len(p) > target * 2:
            paragraphs.extend(_split_long_text(p, target))
        else:
            paragraphs.append(p)

    chunks, cur = [], ""
    for p in paragraphs:
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur)
            tail = cur[-overlap_chars:] if overlap_chars else ""
            cur = (tail + "\n\n" + p).strip()
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur.strip():
        chunks.append(cur)

    return [
        {"text": c, "source": source, "section": f"窗口 {i + 1}"}
        for i, c in enumerate(chunks)
    ]


# ==================== 单文件入库流程 ====================

def ingest_file(path):
    path = os.path.abspath(path)
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in PARSERS:
        print(f"⚠ 不支持的格式 {ext}（支持: {', '.join(PARSERS)}）: {filename}")
        return False

    # 1. 保证文件在 documents/ 里（知识库自包含）
    target_path = os.path.abspath(os.path.join(config.DOC_DIR, filename))
    if path != target_path:
        if os.path.exists(target_path):
            print(f"⚠ documents/ 已有同名文件，将用新文件覆盖: {filename}")
        shutil.copy2(path, target_path)
        print(f"  已复制到 documents/: {filename}")

    # 2. 文件级查重（sha256）
    fhash = store.sha256_file(target_path)
    conn = store.mysql_conn()
    with conn.cursor() as cur:
        row = store.get_document(cur, filename)
    conn.close()

    if row:
        old_hash, status, _ = row
        if old_hash == fhash and status == "done":
            print(f"⏭ 跳过（内容未变）: {filename}")
            return False
        print(f"♻ 检测到内容变更/上次未完成，替换旧版本: {filename}")
        removed = store.purge_source(filename)
        print(f"  已清理旧数据 {removed} 段")

    # 3. 解析 + 切分
    content = PARSERS[ext](target_path)
    if not content.strip():
        print(f"⚠ 抽不出文字（可能是图片版 PDF），跳过: {filename}")
        return False
    chunks = split_by_heading(content, filename) if ext == ".md" else split_by_window(content, filename)

    for c in chunks:
        if len(c["text"]) > MAX_TEXT_CHARS:
            c["text"] = c["text"][:MAX_TEXT_CHARS]
    print(f"  切分完成: {len(chunks)} 段")

    # 4. 登记 processing 状态（失败可重跑）
    conn = store.mysql_conn()
    with conn.cursor() as cur:
        store.upsert_document(cur, filename, fhash, ext.lstrip("."), 0, status="processing")
    conn.commit()

    # 5. 向量化
    print("  向量化中...")
    texts = [c["text"] for c in chunks]
    vectors = store.embed_texts(texts)

    # 6. 写 MySQL（正本，拿自增 id）
    ids = []
    with conn.cursor() as cur:
        for c in chunks:
            keywords = " ".join(jieba.analyse.extract_tags(c["text"], topK=10))
            cur.execute(
                f"INSERT INTO {config.MYSQL_TABLE} (text, source, section, keywords) "
                f"VALUES (%s, %s, %s, %s)",
                (c["text"], c["source"], c["section"], keywords),
            )
            ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    print(f"  MySQL 写入 {len(ids)} 行（id {ids[0]} ~ {ids[-1]}）")

    # 7. 用同一批 id 写 Milvus
    col = store.milvus_collection()
    col.insert([ids, vectors])
    col.flush()
    connections.disconnect("default")
    print("  Milvus 写入完成")

    # 8. 写 ES（jieba 预分词 + whitespace 索引）
    es = store.es_client()
    store.ensure_es_index(es)
    actions = [
        {
            "_index": config.ES_INDEX,
            "_id": i,
            "id": i,
            "text_seg": store.segment(t),
            "source": filename,
        }
        for i, t in zip(ids, texts)
    ]
    bulk(es, actions, chunk_size=100, refresh=True)
    print("  ES 写入完成")

    # 9. 标记完成
    conn = store.mysql_conn()
    with conn.cursor() as cur:
        store.upsert_document(cur, filename, fhash, ext.lstrip("."), len(ids), status="done")
    conn.commit()
    conn.close()

    print(f"✓ {filename} 入库完成: {len(ids)} 段")
    return True


# ==================== 入口 ====================

if __name__ == "__main__":
    store.ensure_schema()

    args = sys.argv[1:]
    if args:
        files = args
    else:
        files = [
            os.path.join(config.DOC_DIR, f)
            for f in sorted(os.listdir(config.DOC_DIR))
            if os.path.splitext(f)[1].lower() in PARSERS
        ]
        print(f"扫描 documents/ 目录: {len(files)} 个候选文件")

    done = 0
    for f in files:
        print(f"\n--- {os.path.basename(f)} ---")
        try:
            if ingest_file(f):
                done += 1
        except Exception as e:
            print(f"✗ {os.path.basename(f)} 入库失败: {e}")

    print(f"\n处理结束: {done} 个文件新入库，其余为跳过/失败")
