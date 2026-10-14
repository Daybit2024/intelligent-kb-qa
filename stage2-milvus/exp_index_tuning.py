"""
stage2 进阶实验：向量索引选型
要回答的问题：索引类型（FLAT/IVF_FLAT/IVF_SQ8/IVF_PQ/HNSW）、距离度量（L2/IP/COSINE）、
             nlist/nprobe、HNSW 参数，在这个项目里选什么、为什么？

实验方法：
1. 从 Milvus 拉出全量向量，用 numpy 暴力精确检索当"标准答案"（真值，不依赖索引）
2. 真实集合（135 条）：轮换索引配置（release→drop→create→load），测召回@10 + 延迟
3. 合成压力集（5000 条 = 真实向量加噪声扩增）：放大规模，让召回差异显现
4. 结论写入 exp_index_results.md
"""
import os
import sys
import time
import warnings

sys.stdout.reconfigure(errors="replace")
warnings.filterwarnings("ignore", message=r".*ORM-style.*")

import numpy as np
from openai import OpenAI
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType,
)
import milvus_config as config

TOP_K = 10
REPEAT = 3          # 每个查询重复检索次数（用于延迟统计）
N_SYNTH = 5000      # 合成压力集规模
STRESS_COLLECTION = "exp_stress_5000"

QUERIES = [
    "什么是大模型微调？",
    "Transformer 的注意力机制是怎么工作的？",
    "如何调用大模型的 API？",
    "向量数据库在 RAG 中起什么作用？",
    "什么是提示词工程？",
    "大模型训练需要哪些数据？",
    "什么是 RAG 检索增强生成？",
    "大模型的上下文窗口是什么？",
    "劳动合同的试用期最长可以约定多久？",
    "什么是正当防卫？",
    "合同违约需要承担什么责任？",
    "房屋租赁合同要注意哪些条款？",
    "劳动争议怎么申请仲裁？",
    "什么是市盈率？",
    "基金和股票有什么区别？",
    "如何进行资产配置？",
    "什么是股票的除权除息？",
    "什么是财务杠杆？",
    "债券的收益率怎么计算？",
    "什么是货币基金？",
]

client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.EMBEDDING_BASE_URL)


def embed(texts, batch=10):
    vectors = []
    for start in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts[start:start + batch],
            dimensions=config.EMBEDDING_DIM,
            encoding_format="float",
        )
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)


def fetch_corpus(col):
    """拉出集合里的全部 id + 向量"""
    res = col.query(expr="id >= 0", output_fields=["id", "embedding"], limit=16384)
    ids = np.array([r["id"] for r in res], dtype=np.int64)
    vecs = np.array([r["embedding"] for r in res], dtype=np.float32)
    return ids, vecs


def exact_topk(qvecs, cvecs, metric, k=TOP_K):
    """numpy 暴力精确检索（真值）"""
    if metric == "L2":
        # ||q||^2 + ||c||^2 - 2 q·c
        d = (qvecs ** 2).sum(1)[:, None] + (cvecs ** 2).sum(1)[None, :] - 2 * qvecs @ cvecs.T
        return np.argsort(d, axis=1)[:, :k]
    if metric == "IP":
        return np.argsort(-(qvecs @ cvecs.T), axis=1)[:, :k]
    # COSINE
    qn = qvecs / np.linalg.norm(qvecs, axis=1, keepdims=True)
    cn = cvecs / np.linalg.norm(cvecs, axis=1, keepdims=True)
    return np.argsort(-(qn @ cn.T), axis=1)[:, :k]


def swap_index(col, index_type, metric, params):
    """在同一个集合上轮换索引配置，返回建索引耗时（毫秒）"""
    col.release()
    try:
        col.drop_index()
    except Exception:
        pass
    t0 = time.perf_counter()
    col.create_index(
        field_name="embedding",
        index_params={"index_type": index_type, "metric_type": metric, "params": params},
    )
    build_ms = (time.perf_counter() - t0) * 1000
    col.load()
    return build_ms


def evaluate(col, qvecs, gt_ids, metric, search_params, label, build_ms=None):
    """返回 (label, recall%, mean_ms, p95_ms)"""
    lat = []
    hit_ids = []
    for qi, q in enumerate(qvecs):
        for r in range(REPEAT):
            t0 = time.perf_counter()
            res = col.search(
                data=[q], anns_field="embedding",
                param={"metric_type": metric, "params": search_params},
                limit=TOP_K,
            )
            lat.append((time.perf_counter() - t0) * 1000)
            if r == 0:
                hit_ids.append([h.id for h in res[0]])
    recall = float(np.mean([
        len(set(h) & set(g)) / TOP_K for h, g in zip(hit_ids, gt_ids)
    ]))
    row = {
        "label": label,
        "recall": recall * 100,
        "mean_ms": float(np.mean(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "build_ms": build_ms,
    }
    build_str = f"  build={build_ms:6.1f}ms" if build_ms is not None else ""
    print(f"  {label:<42} recall@10={row['recall']:6.1f}%  mean={row['mean_ms']:6.2f}ms  p95={row['p95_ms']:6.2f}ms{build_str}")
    return row


def run_ivf_matrix(col, qvecs, ids, cvecs, results, nlist_list, nprobe_list):
    for nlist in nlist_list:
        gt = exact_topk(qvecs, cvecs, "L2")
        gt_ids = ids[gt]
        for nprobe in nprobe_list:
            if nprobe > nlist:
                continue
            b = swap_index(col, "IVF_FLAT", "L2", {"nlist": nlist})
            results.append(evaluate(
                col, qvecs, gt_ids, "L2", {"nprobe": nprobe},
                f"IVF_FLAT L2 nlist={nlist} nprobe={nprobe}", b,
            ))


def main():
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    col = Collection(config.COLLECTION_NAME)
    col.load()

    ids, cvecs = fetch_corpus(col)
    print(f"✓ 真实集合：{len(ids)} 条向量（{cvecs.shape[1]} 维）")

    print("向量化 20 个查询...")
    qvecs = embed(QUERIES)

    # ---- 先导：向量模长（判断是否归一化）----
    cn = np.linalg.norm(cvecs, axis=1)
    qn = np.linalg.norm(qvecs, axis=1)
    print()
    print("=== 先导实验：向量模长 ===")
    print(f"  语料向量模长：min={cn.min():.4f} max={cn.max():.4f} mean={cn.mean():.4f} std={cn.std():.4f}")
    print(f"  查询向量模长：min={qn.min():.4f} max={qn.max():.4f} mean={qn.mean():.4f}")
    normalized = cn.std() < 1e-4 and abs(cn.mean() - 1.0) < 1e-3
    print(f"  是否已归一化：{'是 → L2/IP/COSINE 排序理论上等价' if normalized else '否 → 三种度量可能有差异'}")

    results_real = []

    # ---- Round 1：FLAT 基准 ----
    print()
    print("=== Round 1：真实数据（135 条）· FLAT 基准 ===")
    gt_l2 = ids[exact_topk(qvecs, cvecs, "L2")]
    b = swap_index(col, "FLAT", "L2", {})
    results_real.append(evaluate(col, qvecs, gt_l2, "L2", {}, "FLAT L2（精确基准）", b))

    # ---- Round 2：真实数据 · IVF / 度量 / HNSW ----
    print()
    print("=== Round 2：真实数据 · IVF_FLAT nlist × nprobe ===")
    run_ivf_matrix(col, qvecs, ids, cvecs, results_real, [8, 32, 128], [1, 4, 8, 16])

    print()
    print("=== Round 2b：真实数据 · 度量对比（IVF nlist=32, nprobe=8）===")
    for metric in ["L2", "IP", "COSINE"]:
        gt = ids[exact_topk(qvecs, cvecs, metric)]
        b = swap_index(col, "IVF_FLAT", metric, {"nlist": 32})
        results_real.append(evaluate(
            col, qvecs, gt, metric, {"nprobe": 8}, f"IVF_FLAT {metric} nlist=32 nprobe=8", b,
        ))

    print()
    print("=== Round 2c：真实数据 · HNSW（M=16, efConstruction=200）===")
    gt = ids[exact_topk(qvecs, cvecs, "L2")]
    for ef in [16, 64, 128]:
        b = swap_index(col, "HNSW", "L2", {"M": 16, "efConstruction": 200})
        results_real.append(evaluate(
            col, qvecs, gt, "L2", {"ef": ef}, f"HNSW M=16 efC=200 ef={ef}", b,
        ))

    # ---- Round 3：合成压力集（5000 条）----
    print()
    print(f"=== Round 3：合成压力集（{N_SYNTH} 条 = 真实向量 + 1%~8% 噪声扩增）===")
    rng = np.random.default_rng(42)
    pick = rng.integers(0, len(cvecs), N_SYNTH)
    base = cvecs[pick]
    norm = np.linalg.norm(base, axis=1, keepdims=True)
    ratio = rng.uniform(0.01, 0.08, N_SYNTH)[:, None]
    synth = (base + rng.normal(0, 1, base.shape).astype(np.float32) * ratio * norm).astype(np.float32)
    synth_ids = np.arange(1_000_000, 1_000_000 + N_SYNTH, dtype=np.int64)

    if utility.has_collection(STRESS_COLLECTION):
        utility.drop_collection(STRESS_COLLECTION)
    scol = Collection(
        STRESS_COLLECTION,
        CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
        ], description="压力测试用临时集合"),
    )
    scol.insert([synth_ids.tolist(), synth.tolist()])
    scol.flush()
    print(f"✓ 压力集合已建：{N_SYNTH} 条")

    results_stress = []
    gt_stress = synth_ids[exact_topk(qvecs, synth, "L2")]

    print()
    print("--- 3a. FLAT 基准 ---")
    b = swap_index(scol, "FLAT", "L2", {})
    results_stress.append(evaluate(scol, qvecs, gt_stress, "L2", {}, "FLAT L2（精确基准）", b))

    print()
    print("--- 3b. IVF_FLAT nlist × nprobe ---")
    run_ivf_matrix(scol, qvecs, synth_ids, synth, results_stress, [32, 128], [1, 4, 8, 16, 32])

    print()
    print("--- 3c. 压缩索引（nlist=128, nprobe=16）---")
    b = swap_index(scol, "IVF_SQ8", "L2", {"nlist": 128})
    results_stress.append(evaluate(scol, qvecs, gt_stress, "L2", {"nprobe": 16}, "IVF_SQ8 nlist=128 nprobe=16", b))
    b = swap_index(scol, "IVF_PQ", "L2", {"nlist": 128, "m": 16, "nbits": 8})
    results_stress.append(evaluate(scol, qvecs, gt_stress, "L2", {"nprobe": 16}, "IVF_PQ nlist=128 m=16 nbits=8", b))

    print()
    print("--- 3d. 度量对比（IVF nlist=128, nprobe=16）---")
    for metric in ["L2", "IP", "COSINE"]:
        gt = synth_ids[exact_topk(qvecs, synth, metric)]
        b = swap_index(scol, "IVF_FLAT", metric, {"nlist": 128})
        results_stress.append(evaluate(
            scol, qvecs, gt, metric, {"nprobe": 16}, f"IVF_FLAT {metric} nlist=128 nprobe=16", b,
        ))

    print()
    print("--- 3e. HNSW（M=16, efConstruction=200）---")
    for ef in [16, 64, 128, 256]:
        b = swap_index(scol, "HNSW", "L2", {"M": 16, "efConstruction": 200})
        results_stress.append(evaluate(
            scol, qvecs, gt_stress, "L2", {"ef": ef}, f"HNSW M=16 efC=200 ef={ef}", b,
        ))

    # 清理压力集合（实验专用，不影响真实数据）
    utility.drop_collection(STRESS_COLLECTION)
    print()
    print("✓ 压力集合已删除")

    # ---- 结果写 markdown ----
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_index_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 索引选型实验结果\n\n")
        f.write("## 先导：向量模长\n\n")
        f.write(f"- 语料向量模长：min={cn.min():.4f} max={cn.max():.4f} mean={cn.mean():.4f} std={cn.std():.4f}\n")
        f.write(f"- 是否归一化：{'是（L2/IP/COSINE 排序理论等价）' if normalized else '否'}\n\n")
        for title, rows in [("真实数据（135 条）", results_real), ("合成压力集（5000 条）", results_stress)]:
            f.write(f"## {title}\n\n")
            f.write("| 配置 | recall@10 | 平均延迟(ms) | P95(ms) | 建索引(ms) |\n|---|---|---|---|---|\n")
            for r in rows:
                b = f"{r['build_ms']:.1f}" if r["build_ms"] is not None else "-"
                f.write(f"| {r['label']} | {r['recall']:.1f}% | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | {b} |\n")
            f.write("\n")
    print(f"✓ 结果已写入 {out_path}")

    connections.disconnect("default")
    print("✓ 实验完成")


if __name__ == "__main__":
    main()
