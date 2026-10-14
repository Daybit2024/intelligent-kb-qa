"""
stage2 第四课：四种检索（函数可被 2-5 的 API 直接 import 复用）
  1. 向量检索   —— Milvus ANN（IVF_SQ8, nprobe=16）
  2. 关键词检索 —— Elasticsearch BM25（jieba 预分词 + whitespace 索引）
  3. 混合检索   —— 两路候选 → RRF 融合（k=60，工业标准）
  4. 混合+重排  —— RRF 候选池 → 百炼重排模型（gte-rerank-v2）精排

运行（交互式菜单）:  python query_engine.py
"""
import sys

sys.stdout.reconfigure(errors="replace")

import requests
from pymilvus import connections

import store
import milvus_config as config

RRF_K = 60        # RRF 平滑常数（工业默认值）
POOL_FACTOR = 2   # 每路候选池 = top_k × 2


# ==================== ① 向量检索（Milvus） ====================

def vector_search(query, top_k=10):
    """返回 [{"id", "distance"}]，L2 距离越小越相近"""
    qvec = store.embed_texts([query])[0]
    col = store.milvus_collection()
    res = col.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": config.METRIC_TYPE, "params": {"nprobe": config.NPROBE}},
        limit=top_k,
    )
    connections.disconnect("default")
    return [{"id": hit.id, "distance": hit.distance} for hit in res[0]]


# ==================== ② 关键词检索（ES BM25） ====================

def keyword_search(query, top_k=10):
    """返回 [{"id", "score"}]，score 为 BM25 分数"""
    es = store.es_client()
    res = es.search(
        index=config.ES_INDEX,
        query={"match": {"text_seg": store.segment(query)}},
        size=top_k,
    )
    return [{"id": int(h["_id"]), "score": h["_score"]} for h in res["hits"]["hits"]]


# ==================== ③ 混合检索（RRF 融合） ====================

def hybrid_search(query, top_k=10):
    """两路各取 top_k×2 候选，用 RRF 按排名融合（不比较分数，只看名次）"""
    pool = top_k * POOL_FACTOR
    vec_results = vector_search(query, pool)
    kw_results = keyword_search(query, pool)

    scores = {}   # id -> RRF 累加分
    detail = {}   # id -> 各路排名（展示用）
    for rank, r in enumerate(vec_results, 1):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (RRF_K + rank)
        detail.setdefault(r["id"], {})["vec_rank"] = rank
    for rank, r in enumerate(kw_results, 1):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (RRF_K + rank)
        detail.setdefault(r["id"], {})["kw_rank"] = rank

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [{"id": doc_id, "rrf": s, **detail[doc_id]} for doc_id, s in ranked]


# ==================== ④ 混合 + 重排（百炼） ====================

def rerank_search(query, top_k=10):
    """RRF 取更宽的候选池 → gte-rerank-v2 精排 → top_k"""
    candidates = hybrid_search(query, top_k * POOL_FACTOR)
    if not candidates:
        return []

    texts_map = fetch_texts([c["id"] for c in candidates])
    documents = [texts_map[c["id"]]["text"] for c in candidates]

    resp = requests.post(
        config.RERANK_URL,
        headers={
            "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.RERANK_MODEL,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_k, "return_documents": False},
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json()["output"]["results"]

    out = []
    for r in results:
        c = candidates[r["index"]]
        item = {"id": c["id"], "rerank": r["relevance_score"]}
        item.update({k: v for k, v in c.items() if k != "id"})
        out.append(item)
    return out


# ==================== 候选精排（两阶段检索的第二阶段） ====================

def rerank_documents(query: str, ids: list, top_k: int = 10):
    """对给定的候选 id 列表做一次精排：id → 回表取正文 → gte-rerank → 按分排序"""
    if not ids:
        return []
    texts_map = fetch_texts(ids)
    valid_ids = [i for i in ids if i in texts_map]
    if not valid_ids:
        return []
    documents = [texts_map[i]["text"] for i in valid_ids]

    resp = requests.post(
        config.RERANK_URL,
        headers={
            "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.RERANK_MODEL,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(top_k, len(documents)), "return_documents": False},
        },
        timeout=60,
    )
    resp.raise_for_status()

    out = []
    for r in resp.json()["output"]["results"]:
        doc_id = valid_ids[r["index"]]
        m = texts_map[doc_id]
        out.append({
            "id": doc_id,
            "text": m["text"],
            "source": m["source"],
            "section": m["section"],
            "scores": {"rerank": r["relevance_score"]},
        })
    return out


# ==================== 回表（正文永远从 MySQL 取） ====================

def fetch_texts(ids):
    """按 id 列表回 MySQL 取正文/来源/章节，返回 {id: {...}}"""
    if not ids:
        return {}
    conn = store.mysql_conn()
    placeholders = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, text, source, section FROM {config.MYSQL_TABLE} "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        rows = cur.fetchall()
    conn.close()
    return {r[0]: {"text": r[1], "source": r[2], "section": r[3]} for r in rows}


# ==================== 展示 ====================

def display(results, query, mode_name, preview_len=300):
    print()
    print("=" * 70)
    print(f"【{mode_name}】查询: {query} —— {len(results)} 条结果")
    print("=" * 70)
    if not results:
        print("没有找到结果")
        return
    texts = fetch_texts([r["id"] for r in results])
    for i, r in enumerate(results, 1):
        meta = texts.get(r["id"], {})
        print(f"\n--- 结果 {i} --- id={r['id']} | {meta.get('source', '?')} · {meta.get('section', '?')}")
        parts = []
        if "distance" in r:
            parts.append(f"L2距离={r['distance']:.4f}")
        if "score" in r:
            parts.append(f"BM25={r['score']:.3f}")
        if "rrf" in r:
            parts.append(f"RRF={r['rrf']:.5f}")
        if "rerank" in r:
            parts.append(f"重排分={r['rerank']:.4f}")
        if "vec_rank" in r:
            parts.append(f"向量路#{r['vec_rank']}")
        if "kw_rank" in r:
            parts.append(f"关键词路#{r['kw_rank']}")
        if parts:
            print("   " + " | ".join(parts))
        text = meta.get("text", "")
        if len(text) > preview_len:
            text = text[:preview_len] + "..."
        print(f"   {text}")


# ==================== 交互式菜单 ====================

def main():
    print("=" * 70)
    print("RAG 检索引擎 — 四种模式")
    print("  1. 向量检索（Milvus ANN）")
    print("  2. 关键词检索（ES BM25）")
    print("  3. 混合检索（RRF 融合）")
    print("  4. 混合 + 重排（百炼 gte-rerank-v2）")
    print("  0. 退出")
    print("=" * 70)

    while True:
        mode = input("\n选择模式 (0-4): ").strip()
        if mode == "0":
            break
        if mode not in ("1", "2", "3", "4"):
            print("无效选择")
            continue
        query = input("输入查询: ").strip()
        if not query:
            continue
        top_k = input("返回条数 (默认10): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 10

        try:
            if mode == "1":
                display(vector_search(query, top_k), query, "向量检索")
            elif mode == "2":
                display(keyword_search(query, top_k), query, "关键词检索")
            elif mode == "3":
                display(hybrid_search(query, top_k), query, "混合检索（RRF）")
            else:
                display(rerank_search(query, top_k), query, "混合 + 重排")
        except Exception as e:
            print(f"检索出错: {e}")


if __name__ == "__main__":
    main()
