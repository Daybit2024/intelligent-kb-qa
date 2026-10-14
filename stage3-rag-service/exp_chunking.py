"""
Chunking 实验：不同切分策略的检索质量对比（LLM 判定口径）

矩阵：
  baseline  生产配置（md 标题切分 + pdf 窗口 500/15%），直接调生产 8000 服务
  w256_o15  全语料统一窗口 256 字符 / 重叠 15%
  w500_o15  全语料统一窗口 500 / 15%
  w1024_o15 全语料统一窗口 1024 / 15%
  w500_o0   全语料统一窗口 500 / 无重叠
  w500_o30  全语料统一窗口 500 / 重叠 30%

设计：
  - 不碰生产库：每配置独立"临时 Milvus 集合 + 临时 ES 索引"，跑完即删
  - 单一变量：embedding 模型 / 检索管线（向量+BM25 → RRF k=60 → gte-rerank）/ 评审员 prompt / 题目集 全部锁定
  - 复用 eval_rag 的评审员与磁盘缓存（重复判定不再花钱）

运行:  python -u exp_chunking.py    （建议后台，约 30~40 分钟）
输出:  控制台进度 + exp_chunking_results.md
"""
import os
import sys
import time

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2-milvus"))  # milvus_config / store / ingest

from dotenv import load_dotenv

load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

import milvus_config as config
import store
import ingest
import eval_rag  # 复用：评测集 / 评审员(带缓存) / 生产检索

EVALSET = eval_rag.load_eval_set()
QUESTIONS = [item["question"] for item in EVALSET["answerable"]]
DOCS_DIR = config.DOC_DIR

RRF_K = 60
POOL = 20          # 每路召回池（与生产管线一致）
TOP_K = 10         # 评测取前 10

CONFIGS = [
    ("w256_o15", 256, 0.15),
    ("w500_o15", 500, 0.15),
    ("w1024_o15", 1024, 0.15),
    ("w500_o0", 500, 0.0),
    ("w500_o30", 500, 0.30),
]


def parse_all_once():
    """预解析全部文档（各配置复用，避免重复解析/OCR）"""
    docs = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ingest.PARSERS or not os.path.isfile(path):
            continue
        try:
            content = ingest.PARSERS[ext](path)
        except Exception as e:
            print(f"  解析失败 {fname}: {e}", flush=True)
            continue
        if content.strip():
            docs.append((fname, content))
    return docs


def chunk_corpus(docs, target, overlap):
    """统一窗口切分全语料，返回 (ids, texts, sources)"""
    ids, texts, sources = [], [], []
    next_id = 1
    for fname, content in docs:
        for c in ingest.split_by_window(content, fname, target=target, overlap_ratio=overlap):
            t = c["text"]
            if len(t) > ingest.MAX_TEXT_CHARS:
                t = t[: ingest.MAX_TEXT_CHARS]
            ids.append(next_id)
            texts.append(t)
            sources.append(fname)
            next_id += 1
    return ids, texts, sources


def build_temp_collection(name, ids, vectors):
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    if utility.has_collection(name):
        utility.drop_collection(name)
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
    ], description="chunking 实验临时集合")
    col = Collection(name, schema)
    B = 500
    for i in range(0, len(ids), B):
        col.insert([ids[i:i + B], vectors[i:i + B]])
    col.flush()
    col.create_index(
        field_name="embedding",
        index_params={"index_type": config.INDEX_TYPE, "metric_type": config.METRIC_TYPE,
                      "params": {"nlist": config.NLIST}},
    )
    col.load()
    return col


def build_temp_es(es, index_name, ids, texts, sources):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, mappings={"properties": {
        "id": {"type": "integer"},
        "text_seg": {"type": "text", "analyzer": "whitespace"},
        "source": {"type": "keyword"},
    }})
    actions = ({"_index": index_name, "_id": i, "id": i, "text_seg": store.segment(t), "source": s}
               for i, t, s in zip(ids, texts, sources))
    bulk(es, actions, chunk_size=200, refresh=True)


def retrieve(col, es, index_name, texts_map, question):
    """复刻生产管线：向量 + BM25 各取 POOL → RRF → 候选 20 → gte-rerank → top10 [(id, text)]"""
    qvec = store.embed_texts([question])[0]
    res = col.search(
        data=[qvec], anns_field="embedding",
        param={"metric_type": config.METRIC_TYPE, "params": {"nprobe": config.NPROBE}},
        limit=POOL,
    )
    vec_ids = [h.id for h in res[0]]

    r = es.search(index=index_name, query={"match": {"text_seg": store.segment(question)}}, size=POOL)
    kw_ids = [int(h["_id"]) for h in r["hits"]["hits"]]

    scores = {}
    for rank, i in enumerate(vec_ids, 1):
        scores[i] = scores.get(i, 0) + 1.0 / (RRF_K + rank)
    for rank, i in enumerate(kw_ids, 1):
        scores[i] = scores.get(i, 0) + 1.0 / (RRF_K + rank)
    fused = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])][: TOP_K * 2]
    if not fused:
        return []

    docs = [texts_map[i] for i in fused]
    resp = requests.post(
        config.RERANK_URL,
        headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
        json={"model": config.RERANK_MODEL, "input": {"query": question, "documents": docs},
              "parameters": {"top_n": TOP_K, "return_documents": False}},
        timeout=60,
    )
    resp.raise_for_status()
    return [(fused[r["index"]], texts_map[fused[r["index"]]]) for r in resp.json()["output"]["results"]]


def evaluate(retrieval_fn, tag):
    """33 题：串行检索 → 并行评审（缓存复用）→ 指标"""
    t0 = time.perf_counter()

    collected = []
    for q in QUESTIONS:
        try:
            collected.append(retrieval_fn(q))
        except Exception as e:
            print(f"    [{tag}] 检索失败: {e}", flush=True)
            collected.append([])

    def _task(k):
        return k, eval_rag.judge_relevance(QUESTIONS[k], collected[k])

    rel_map = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(_task, k) for k in range(len(QUESTIONS))]):
            k, rel = fut.result()
            rel_map[k] = rel
    eval_rag._save_cache()

    n = len(QUESTIONS)
    h5 = h10 = 0
    rrs = []
    for k in range(n):
        flags = [(j + 1) in rel_map[k] for j in range(len(collected[k]))]
        h5 += any(flags[:5])
        h10 += any(flags[:10])
        rrs.append(next((1.0 / (j + 1) for j, f in enumerate(flags) if f), 0.0))
    return {"hit@5": h5 / n * 100, "hit@10": h10 / n * 100, "MRR": sum(rrs) / n,
            "time_s": round(time.perf_counter() - t0)}


def write_report(results):
    lines = [
        "# Chunking 实验结果（LLM 判定口径，33 道可答题）\n",
        "| 配置 | 段数 | 平均长度(字符) | hit@5 | hit@10 | MRR | 用时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.get('chunks', '-')} | {r.get('avg_chars', '-')} | "
            f"{r['hit@5']:.1f}% | {r['hit@10']:.1f}% | {r['MRR']:.3f} | {r['time_s']} |"
        )
    path = os.path.join(HERE, "exp_chunking_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 报告已写入 {path}", flush=True)


def main():
    results = {}

    # 0. baseline：直接评测生产检索（零重建；评审结果大概率走缓存）
    print("=== [baseline] 生产配置（md 标题切分 + pdf 窗口 500/15%）===", flush=True)
    baseline_fn = lambda q: [(x["id"], x["text"]) for x in eval_rag.search(q, "rerank", TOP_K)]
    results["baseline(生产)"] = evaluate(baseline_fn, "baseline")
    print(f"  baseline: {results['baseline(生产)']}", flush=True)

    # 预解析（各配置复用）
    print("\n=== 预解析全部文档（一次）===", flush=True)
    docs = parse_all_once()
    print(f"  已解析 {len(docs)} 个文档", flush=True)

    es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")

    for idx, (tag, target, overlap) in enumerate(CONFIGS, 1):
        print(f"\n=== [{idx}/{len(CONFIGS)}] 配置 {tag} ===", flush=True)

        ids, texts, sources = chunk_corpus(docs, target, overlap)
        avg = sum(len(t) for t in texts) // len(ids)
        print(f"  切分完成: {len(ids)} 段（均值 {avg} 字符）", flush=True)

        print("  向量化中...", flush=True)
        vectors = store.embed_texts(texts)
        texts_map = {i: t for i, t in zip(ids, texts)}

        col_name = f"exp_chunks_{tag}"
        col = build_temp_collection(col_name, ids, vectors)
        build_temp_es(es, col_name, ids, texts, sources)
        print("  临时集合与索引就绪，开始评测", flush=True)

        row = evaluate(lambda q: retrieve(col, es, col_name, texts_map, q), tag)
        row["chunks"] = len(ids)
        row["avg_chars"] = avg
        results[tag] = row
        print(f"  {tag}: {row}", flush=True)

        utility.drop_collection(col_name)
        es.indices.delete(index=col_name)
        connections.disconnect("default")
        print("  临时资源已清理", flush=True)

    print("\n========== 汇总 ==========", flush=True)
    for name, r in results.items():
        print(f"  {name:<16} hit@5={r['hit@5']:.1f}%  hit@10={r['hit@10']:.1f}%  MRR={r['MRR']:.3f}", flush=True)
    write_report(results)


if __name__ == "__main__":
    main()
