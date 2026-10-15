

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
import json
import os
import re
import shutil
import sys

sys.stdout.reconfigure(errors="replace")

import jieba.analyse
from elasticsearch.helpers import bulk
from openai import OpenAI
from pymilvus import connections

import store
import milvus_config as config

MAX_TEXT_CHARS = 7000  # 单段上限（embedding 输入 / MySQL TEXT 字段保护）

_llm_client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.EMBEDDING_BASE_URL)


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


def _row_span_and_gap(row):
    """一行的横向跨度与"段间最大空隙"（词按 x 排序后相邻间隔的最大值）"""
    xs = sorted((w["x0"], w["x1"]) for w in row)
    span = xs[-1][1] - xs[0][0]
    gap = max([0.0] + [b0 - a1 for (_, a1), (b0, _) in zip(xs, xs[1:])])
    return span, gap


def _row_is_full_width(row, page):
    """整行 = 跨度≥70%页宽 且 内部无栏级空隙（单个连续段）

    双栏页里"左右两栏各一个词恰好同行"的组合会因中间存在栏间距空隙
    （一般 ≥2% 页宽）而不满足，不会把正文行误判成整行。
    """
    span, gap = _row_span_and_gap(row)
    return span >= page.width * 0.7 and gap < page.width * 0.015


def _detect_columns(page):
    """双栏版面检测：返回 (分栏线x, 正文起始y)；非双栏返回 (None, None)

    原理：双栏正文之间存在一条贯穿上下的"零穿越"纵向走廊——没有任何正文词横跨它。
    标题/作者这类横跨版面的整行先识别剔除，不参与走廊判定，避免误伤检测。
    """
    words = page.extract_words()
    if len(words) < 15:
        return None, None

    # 聚行：按 top 把词归并成行（容差 3pt）
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows, cur = [], [words[0]]
    for w in words[1:]:
        if abs(w["top"] - cur[0]["top"]) <= 3:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)

    body_words = [w for row in rows if not _row_is_full_width(row, page) for w in row]
    # 页眉/页脚带（上下 6%）的居中文字（页码等）可能横跨走廊，排除出判定集
    y0, y1 = page.height * 0.06, page.height * 0.94
    body_words = [w for w in body_words if w["top"] >= y0 and w["bottom"] <= y1]
    if len(body_words) < 12:
        return None, None

    # 在页宽 25%~75% 扫走廊；允许 ≤2 个词穿越（容忍个别居中短元素，如封面式小标题）
    lo, hi = page.width * 0.25, page.width * 0.75
    runs, start = [], None
    for x in range(int(lo), int(hi) + 1, 2):
        crossing = sum(1 for w in body_words if w["x0"] < x < w["x1"])
        if crossing <= 2:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, int(hi)))
    if not runs:
        return None, None

    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < page.width * 0.015:      # 走廊过窄 → 只是词间空白，不是栏间距
        return None, None
    split = (a + b) / 2

    # 两侧词量都要够（各 ≥25%），排除"单栏正文 + 大片页边空白"的误判
    n_left = sum(1 for w in body_words if w["x1"] <= split)
    n_right = sum(1 for w in body_words if w["x0"] >= split)
    if n_left < len(body_words) * 0.25 or n_right < len(body_words) * 0.25:
        return None, None

    # 横跨走廊的词 = 整行元素（标题/作者等）——走廊能成立，正是因为正文词从不跨它，
    # 少数被容许穿越的（每扫描点 ≤2 个）就是这类元素
    wide_ids = {
        id(w)
        for row in rows for w in row
        if any(w["x0"] < x < w["x1"] for x in range(a, b, 2))
    }
    # 正文起始 y：从页顶连续、含整行元素的行（标题区）之下即为正文
    top_y = 0.0
    for row in rows:
        if not any(id(w) in wide_ids for w in row):
            break
        top_y = max(w["bottom"] for w in row)
    return split, top_y


def _extract_page_text(page, page_no=None):
    """单页文字提取：双栏页面按"整行标题 → 左栏 → 右栏"的阅读顺序输出"""
    split, top_y = _detect_columns(page)
    if split is None:
        return page.extract_text() or ""

    if page_no:
        print(f"    第 {page_no} 页：检测到双栏版面，按左栏→右栏还原阅读顺序")
    parts = []
    # 正文裁剪线下移 2pt：标题字符底边恰好等于裁剪线时会因浮点相等被重复裁进两侧
    cut = top_y + 2 if top_y > 0 else 0.0
    if cut > 0:
        head = page.crop((0, 0, page.width, cut)).extract_text() or ""
        if head.strip():
            parts.append(head)
    for x0, x1 in ((0, split), (split, page.width)):
        col = page.crop((x0, cut, x1, page.height)).extract_text() or ""
        if col.strip():
            parts.append(col)
    return "\n\n".join(parts)


def parse_pdf(path):
    """PDF 智能解析：先用 pdfplumber 抽文字（双栏页面自动按栏还原阅读顺序）；
    文字量过少（图片版/扫描版）自动切换 OCR"""
    import pdfplumber

    pages = []
    n_pages = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = _extract_page_text(page, page_no=i)
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


# ==================== 领域归类（数据驱动） ====================

DOMAIN_PROMPT = (
    "你在为一个知识库做领域归类。现有领域如下（name：说明）：\n"
    "{existing}\n\n"
    "请阅读文档节选，判断它属于哪个现有领域；若都不匹配，请创建一个新的简短领域名"
    "（英文小写单词，如 medical），并给出一句中文说明。注意：能复用现有领域时必须复用，不要造近义词。\n"
    '只输出 JSON：{"domain": "领域名", "is_new": true 或 false, "description": "一句中文说明"}'
)


def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def determine_domain(content):
    """让模型通读文档节选 → 复用现有领域或新建（新领域写入领域表）；失败按 general 兜底"""
    existing = store.list_domains()
    existing_names = {d["name"] for d in existing}
    existing_txt = "\n".join(
        f"- {d['name']}：{d['description'] or '（无说明）'}" for d in existing
    ) or "- （暂无领域）"
    try:
        resp = _llm_client.chat.completions.create(
            model="qwen-plus", temperature=0.1, max_tokens=200,
            messages=[
                # 用 replace 而非 format：prompt 里的 JSON 示例含花括号，会被 format 误当占位符
                {"role": "system", "content": DOMAIN_PROMPT.replace("{existing}", existing_txt)},
                {"role": "user", "content": f"文档节选：\n{content[:2000]}"},
            ],
        )
        data = _extract_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"  领域归类失败（按 general 处理）: {e}")
        return "general"
    if not data or not data.get("domain"):
        return "general"
    name = re.sub(r"[^a-z0-9_\-]", "", str(data["domain"]).strip().lower())[:50]
    if not name:
        return "general"
    desc = str(data.get("description", ""))[:500]
    if name in existing_names:
        print(f"  领域归类: 复用现有领域「{name}」")
    else:
        store.upsert_domain(name, desc)
        print(f"  领域归类: 新建领域「{name}」（{desc}）")
    return name


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

    # 3.5 领域归类（数据驱动：优先复用现有领域，否则新建并入领域表）
    domain = determine_domain(content)

    # 4. 登记 processing 状态（失败可重跑）
    conn = store.mysql_conn()
    with conn.cursor() as cur:
        store.upsert_document(cur, filename, fhash, ext.lstrip("."), 0, status="processing", domain=domain)
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
                f"INSERT INTO {config.MYSQL_TABLE} (text, source, section, keywords, domain) "
                f"VALUES (%s, %s, %s, %s, %s)",
                (c["text"], c["source"], c["section"], keywords, domain),
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
        store.upsert_document(cur, filename, fhash, ext.lstrip("."), len(ids), status="done", domain=domain)
    conn.commit()
    conn.close()

    # 10. 刷新领域文件数（覆盖替换场景；归零领域自动清除）
    store.refresh_domain_counts()

    print(f"✓ {filename} 入库完成: {len(ids)} 段（领域: {domain}）")
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
