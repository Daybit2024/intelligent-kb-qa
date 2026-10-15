"""
意图识别 + 领域过滤 消融实验（A/B）

A 组: enable_intent=True  （意图识别 + 领域过滤生效，生产默认）
B 组: enable_intent=False （关闭意图识别，过滤自动失效）

度量（33 可答 + 10 陷阱，top_k=5，判定的对象是实际喂给 LLM 的 sources）：
  检索级: hit@5 / MRR@5（LLM 判定相关性口径）
  拒答:   陷阱题拒答率 + low_confidence 触发率
  性能:   各阶段耗时对比（意图识别的代价）

运行:  python -u exp_intent_ablation.py   （约 25 分钟）
输出:  控制台 + exp_intent_results.md
"""
import os
import sys
import time

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage2-milvus"))

from dotenv import load_dotenv

load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import eval_rag
import query_engine  # fetch_texts：按 id 取正文（判定用）

RAG_API = "http://127.0.0.1:8001"
TOP_K = 5


def chat(question, enable_intent):
    return requests.post(
        f"{RAG_API}/chat",
        json={"question": question, "top_k": TOP_K, "enable_intent": enable_intent},
        timeout=300,
    ).json()


def run_config(tag, enable_intent):
    evalset = eval_rag.load_eval_set()

    print(f"=== [{tag}] 检索 {len(evalset['answerable'])} 道可答题 ===", flush=True)
    ans_rows = []
    for i, item in enumerate(evalset["answerable"], 1):
        t0 = time.perf_counter()
        r = chat(item["question"], enable_intent)
        ans_rows.append({
            "question": item["question"],
            "ids": [s["id"] for s in r["sources"]],
            "low_conf": r["low_confidence"],
            "timings": r["timings"],
            "wall_s": round(time.perf_counter() - t0, 1),
        })
        if i % 10 == 0:
            print(f"    {i}/{len(evalset['answerable'])}", flush=True)

    print(f"=== [{tag}] 陷阱题 {len(evalset['unanswerable'])} 道 ===", flush=True)
    un_rows = []
    for item in evalset["unanswerable"]:
        r = chat(item["question"], enable_intent)
        un_rows.append({"question": item["question"], "answer": r.get("answer", ""),
                        "low_conf": r["low_confidence"]})
    return ans_rows, un_rows


def metrics(ans_rows, un_rows):
    # 收集全部候选 id 的正文（一次批量回表）
    all_ids = list({i for row in ans_rows for i in row["ids"]})
    tmap = query_engine.fetch_texts(all_ids)

    # 并行 LLM 判定（走 eval_rag 的磁盘缓存）
    def task(i):
        row = ans_rows[i]
        chunks = [(x, tmap[x]["text"]) for x in row["ids"] if x in tmap]
        return i, eval_rag.judge_relevance(row["question"], chunks)

    rel_map = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(task, i) for i in range(len(ans_rows))]):
            k, rel = fut.result()
            rel_map[k] = rel
    eval_rag._save_cache()

    n = len(ans_rows)
    hit5 = 0
    rrs = []
    for i, row in enumerate(ans_rows):
        flags = [(j + 1) in rel_map[i] for j in range(len(row["ids"]))]
        hit5 += any(flags[:5])
        rrs.append(next((1.0 / (j + 1) for j, f in enumerate(flags) if f), 0.0))

    # 陷阱题拒答判定
    refused = 0
    for row in un_rows:
        if eval_rag.judge_refusal(row["answer"]):
            refused += 1

    def avg(key):
        vals = [row["timings"].get(key, 0) for row in ans_rows]
        return sum(vals) / len(vals) if vals else 0

    return {
        "hit@5": hit5 / n * 100,
        "MRR@5": sum(rrs) / n,
        "low_conf_rate": sum(r["low_conf"] for r in ans_rows) / n * 100,
        "un_low_conf_rate": sum(r["low_conf"] for r in un_rows) / len(un_rows) * 100 if un_rows else 0,
        "refusal_rate": refused / len(un_rows) * 100 if un_rows else 0,
        "avg_intent_ms": avg("intent_ms"),
        "avg_retrieval_ms": avg("retrieval_ms"),
        "avg_wall_s": sum(r["wall_s"] for r in ans_rows) / n,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = {}

    ans_a, un_a = run_config("A: 意图识别 ON", True)
    ans_b, un_b = run_config("B: 意图识别 OFF", False)

    print("\n=== 判定与统计 ===", flush=True)
    results["A: 意图识别+域过滤 ON"] = metrics(ans_a, un_a)
    results["B: 全部关闭 OFF"] = metrics(ans_b, un_b)

    print("\n========== 对比汇总 ==========", flush=True)
    keys = ["hit@5", "MRR@5", "refusal_rate", "un_low_conf_rate", "low_conf_rate", "avg_intent_ms", "avg_retrieval_ms", "avg_wall_s"]
    header = f"{'指标':<18}{'A(ON)':>12}{'B(OFF)':>12}"
    print(header)
    for k in keys:
        print(f"{k:<18}{results['A: 意图识别+域过滤 ON'][k]:>12.2f}{results['B: 全部关闭 OFF'][k]:>12.2f}")

    lines = [
        "# 意图识别 + 领域过滤 消融实验\n",
        f"（33 可答 + 10 陷阱，top_k={TOP_K}，LLM 判定口径）\n",
        "| 指标 | A: 意图识别+域过滤 ON | B: 全部关闭 OFF | 差值 |",
        "|---|---|---|---|",
    ]
    for k in keys:
        a, b = results["A: 意图识别+域过滤 ON"][k], results["B: 全部关闭 OFF"][k]
        lines.append(f"| {k} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
    path = os.path.join(HERE, "exp_intent_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 报告已写入 {path}（总耗时 {time.perf_counter()-t0:.0f}s）", flush=True)
