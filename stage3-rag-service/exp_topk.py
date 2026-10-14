"""
Top-k 实验：上下文条数对答案质量的影响（端到端 LLM 判分）

问题：最终喂给 LLM 的上下文条数 k，答案质量如何随 k 变化？
变量：k ∈ {3, 4, 5, 7}
度量：33 道可答题的端到端判分均值（主）+ 10 道陷阱题的拒答率（护栏）
判读：找"质量平台期起点"，在平台期内取最小的 k（成本侧最优）

运行: python -u exp_topk.py   （后台，约 40 分钟）
输出: 控制台进度 + exp_topk_results.md
"""
import os
import sys
import time

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dotenv import load_dotenv

load_dotenv()

import eval_rag

CONFIGS = [3, 4, 5, 7]


def run_config(k):
    evalset = eval_rag.load_eval_set()

    # 可答题：判分
    scores, verdicts = [], {}
    for i, item in enumerate(evalset["answerable"], 1):
        try:
            resp = eval_rag.chat(item["question"], top_k=k)
            judge = eval_rag.judge_answer(item["question"], item["reference_answer"], resp.get("answer", ""))
        except Exception as e:
            judge = {"score": 0, "verdict": "错误"}
        scores.append(judge["score"])
        verdicts[judge["verdict"]] = verdicts.get(judge["verdict"], 0) + 1
        if i % 10 == 0:
            print(f"    [{k}] 判分进度 {i}/{len(evalset['answerable'])}", flush=True)

    # 不可答题：拒答率（护栏）
    refused = 0
    for item in evalset["unanswerable"]:
        try:
            resp = eval_rag.chat(item["question"], top_k=k)
            refused += eval_rag.judge_refusal(resp.get("answer", ""))
        except Exception:
            pass

    return {
        "top_k": k,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "verdicts": verdicts,
        "refusal_rate": refused / len(evalset["unanswerable"]) * 100,
    }


if __name__ == "__main__":
    results = []
    for k in CONFIGS:
        print(f"\n=== top_k = {k} ===", flush=True)
        t0 = time.perf_counter()
        r = run_config(k)
        r["time_s"] = round(time.perf_counter() - t0)
        results.append(r)
        print(f"  平均分={r['avg_score']:.2f}  判定={r['verdicts']}  拒答率={r['refusal_rate']:.0f}%", flush=True)

    lines = [
        "# Top-k 实验结果（端到端判分，33 可答 + 10 陷阱）\n",
        "| top_k | 平均分 | 判定分布 | 拒答率 | 用时(s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['top_k']} | {r['avg_score']:.2f} | {r['verdicts']} | {r['refusal_rate']:.0f}% | {r['time_s']} |"
        )
    path = os.path.join(HERE, "exp_topk_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 报告已写入 {path}", flush=True)

    print("\n========== 汇总 ==========", flush=True)
    for r in results:
        print(f"  top_k={r['top_k']}: 平均分 {r['avg_score']:.2f}  拒答 {r['refusal_rate']:.0f}%", flush=True)
