"""
RAG 评测脚本（基于 eval_set.json 的标注数据）
用法:
  python eval_rag.py --retrieval   检索级评测：四种检索模式横向对比（LLM 判定相关性，首次约 3~4 分钟）
  python eval_rag.py --e2e         端到端评测：经 8001 生成 + LLM 判分 + 拒答评测（慢）
  python eval_rag.py --all         全部

指标:
  检索级  hit@5 / hit@10 / MRR@10（LLM 评审员判定检索结果是否真正含回答所需信息）
  端到端  LLM-as-judge 对照参考答案打分（0-10）
  拒答    不可答题的拒答率 + 用两类题的重排分数分布校准拒答阈值

注：LLM 评审/判分只用于线下评测，线上服务链路（8000/8001）不包含任何评审调用。
"""
import argparse
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(errors="replace")

from dotenv import load_dotenv

load_dotenv()

import requests
from openai import OpenAI

SEARCH_API = "http://127.0.0.1:8000"
RAG_API = "http://127.0.0.1:8001"
EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.md")

llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

JUDGE_PROMPT = (
    "你是严格但公正的评分员。请对照参考答案，给候选回答打分。\n"
    '只输出 JSON：{"score": 0到10的整数, "verdict": "正确/部分正确/错误", "reason": "一句话理由"}\n'
    "评分标准：完全回答参考要点=10；主要点覆盖但有遗漏=6~9；沾边但不全=3~5；错误或答非所问=0~2"
)

REFUSAL_JUDGE_PROMPT = (
    "判断候选回答对『知识库没有相关信息』这一情况的处理是否合格。\n"
    "合格（refused=true）的情形：\n"
    "A. 明确说明资料中没有相关内容/不足以回答；或\n"
    "B. 虽然附带了解释或延伸说明，但没有编造具体数据，且明确说明了资料不包含所问信息。\n"
    "不合格（refused=false）：编造了具体数据/事实来作答，或答非所问且未说明资料限制。\n"
    '只输出 JSON：{"refused": true 或 false}'
)

REL_JUDGE_PROMPT = (
    "你是检索相关性评审员。给定一个问题和若干检索到的文档片段（编号 [1]~[N]），"
    "判断哪些片段包含『有助于回答该问题』的信息。\n"
    "标准：包含答案、部分答案、或直接相关的关键数据/事实 → 相关；"
    "仅话题沾边但没有可用于回答的信息（如目录页、客套话）→ 不相关。\n"
    '只输出 JSON：{"relevant": [编号列表，如 [1,3]]}'
)


def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


# ==================== 相关性评审（LLM 判定 + 磁盘缓存） ====================

JUDGE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_judge_cache.json")
_judge_cache = None


def _cache():
    global _judge_cache
    if _judge_cache is None:
        if os.path.exists(JUDGE_CACHE_PATH):
            with open(JUDGE_CACHE_PATH, encoding="utf-8") as f:
                _judge_cache = json.load(f)
        else:
            _judge_cache = {}
    return _judge_cache


def _save_cache():
    if _judge_cache is not None:
        with open(JUDGE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_judge_cache, f, ensure_ascii=False)


def judge_relevance(question, chunks):
    """批量判定 chunks（[(id, text)]）中哪些与问题相关，返回 1-based 编号集合。
    结果按 (问题 + 候选 id 列表) 缓存到磁盘，重复评测不再产生调用。"""
    if not chunks:
        return set()
    cache = _cache()
    key = question + "||" + ",".join(str(c[0]) for c in chunks)
    if key in cache:
        return set(cache[key])

    listing = "\n\n".join(f"[{i}] {t[:400]}" for i, (_, t) in enumerate(chunks, 1))
    resp = llm.chat.completions.create(
        model="qwen-plus", temperature=0.1, max_tokens=200,
        messages=[
            {"role": "system", "content": REL_JUDGE_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n文档片段：\n{listing}"},
        ],
    )
    data = _extract_json(resp.choices[0].message.content)
    rel = set()
    if data and isinstance(data.get("relevant"), list):
        rel = {int(i) for i in data["relevant"] if isinstance(i, (int, float))}
    cache[key] = sorted(rel)
    return rel


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def search(question, mode, top_k=10):
    resp = requests.post(
        f"{SEARCH_API}/search",
        json={"query": question, "search_type": mode, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def chat(question, top_k=5):
    resp = requests.post(
        f"{RAG_API}/chat",
        json={"question": question, "top_k": top_k},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# ==================== 检索级评测 ====================

def retrieval_eval(evalset):
    """检索级评测（LLM 判定相关性口径）：
    各模式检索 top10 → LLM 评审哪些结果真正含有回答所需信息 → hit@5/hit@10/MRR"""
    answerable = evalset["answerable"]
    n = len(answerable)
    modes = ["vector", "keyword", "hybrid", "rerank"]

    # 阶段1：串行检索（避免 Milvus 连接竞争），先收集所有结果
    print("  检索中（4 模式 × 33 题）...")
    collected = {}
    for mode in modes:
        for idx, item in enumerate(answerable):
            try:
                collected[(mode, idx)] = search(item["question"], mode, 10)
            except Exception as e:
                print(f"  [{mode}] 检索失败: {e}")
                collected[(mode, idx)] = []

    # 阶段2：并行 LLM 评审（结果有磁盘缓存，重复评测秒出且零费用）
    print("  LLM 相关性评审（并行；首次较慢，之后走缓存）...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _judge_task(mode, idx):
        item = answerable[idx]
        chunks = [(x["id"], x["text"]) for x in collected[(mode, idx)]]
        return (mode, idx), judge_relevance(item["question"], chunks)

    rel_map = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_judge_task, m, i) for m in modes for i in range(n)]
        done = 0
        for fut in as_completed(futures):
            key, rel = fut.result()
            rel_map[key] = rel
            done += 1
            if done % 20 == 0:
                print(f"    评审进度 {done}/{len(futures)}")
    _save_cache()

    # 阶段3：计算指标
    table = {}
    ans_top1 = []
    for mode in modes:
        hits5 = hits10 = 0
        rrs = []
        for idx in range(n):
            results = collected[(mode, idx)]
            rel = rel_map[(mode, idx)]
            flags = [(i + 1) in rel for i in range(len(results))]
            h5 = any(flags[:5])
            h10 = any(flags[:10])
            rr = next((1.0 / (k + 1) for k, h in enumerate(flags) if h), 0.0)
            hits5 += h5
            hits10 += h10
            rrs.append(rr)
            if mode == "rerank" and results:
                ans_top1.append(results[0]["scores"].get("rerank", 0.0))
        table[mode] = {
            "hit@5": hits5 / n * 100,
            "hit@10": hits10 / n * 100,
            "MRR@10": sum(rrs) / n,
        }
        print(f"  [{mode:<8}] hit@5={table[mode]['hit@5']:.1f}%  hit@10={table[mode]['hit@10']:.1f}%  MRR@10={table[mode]['MRR@10']:.3f}")

    return table, ans_top1


# ==================== 拒答阈值校准 ====================

def threshold_calibration(evalset, ans_top1):
    un_top1 = []
    for item in evalset["unanswerable"]:
        try:
            results = search(item["question"], "rerank", 10)
            if results:
                un_top1.append(results[0]["scores"].get("rerank", 0.0))
        except Exception:
            pass

    print(f"\n  可答题 top1 重排分: 均值={sum(ans_top1)/len(ans_top1):.3f}  最低={min(ans_top1):.3f}")
    print(f"  不可答题 top1 重排分: 均值={sum(un_top1)/len(un_top1):.3f}  最高={max(un_top1):.3f}")

    # 找使分类准确率最高的阈值
    best_t, best_acc = None, -1
    for t in sorted(set([round(s, 3) for s in ans_top1 + un_top1])):
        acc = (sum(1 for s in ans_top1 if s >= t) + sum(1 for s in un_top1 if s < t)) / (len(ans_top1) + len(un_top1))
        if acc > best_acc:
            best_t, best_acc = t, acc

    print(f"  两类分数{'可分' if best_acc > 0.85 else '重叠严重，不可靠'}：最佳阈值 t={best_t}，分类准确率={best_acc*100:.0f}%")
    return {"ans_top1": ans_top1, "un_top1": un_top1, "best_threshold": best_t, "best_accuracy": best_acc}


# ==================== 端到端评测 ====================

def judge_answer(question, reference, answer):
    resp = llm.chat.completions.create(
        model="qwen-plus", temperature=0.1, max_tokens=300,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n参考答案：{reference}\n\n候选回答：{answer[:1500]}"},
        ],
    )
    data = _extract_json(resp.choices[0].message.content)
    if not data:
        return {"score": 0, "verdict": "解析失败", "reason": ""}
    return {"score": int(data.get("score", 0)), "verdict": data.get("verdict", "?"), "reason": data.get("reason", "")}


def judge_refusal(answer):
    resp = llm.chat.completions.create(
        model="qwen-plus", temperature=0.1, max_tokens=100,
        messages=[
            {"role": "system", "content": REFUSAL_JUDGE_PROMPT},
            {"role": "user", "content": f"候选回答：{answer[:800]}"},
        ],
    )
    data = _extract_json(resp.choices[0].message.content)
    return bool(data and data.get("refused"))


def e2e_eval(evalset):
    # 可答题：打分
    scores, verdicts = [], {}
    rows = []
    for i, item in enumerate(evalset["answerable"], 1):
        try:
            resp = chat(item["question"])
            answer = resp.get("answer", "")
            judge = judge_answer(item["question"], item["reference_answer"], answer)
        except Exception as e:
            answer, judge = f"(出错: {e})", {"score": 0, "verdict": "错误", "reason": str(e)}
        scores.append(judge["score"])
        verdicts[judge["verdict"]] = verdicts.get(judge["verdict"], 0) + 1
        rows.append({"question": item["question"], **judge})
        print(f"  [{i}/{len(evalset['answerable'])}] {judge['score']}/10 {judge['verdict']} | {item['question'][:36]}")

    # 不可答题：拒答率
    refused_cnt = 0
    ref_rows = []
    for i, item in enumerate(evalset["unanswerable"], 1):
        try:
            resp = chat(item["question"])
            answer = resp.get("answer", "")
            refused = judge_refusal(answer)
        except Exception as e:
            answer, refused = f"(出错: {e})", False
        refused_cnt += refused
        ref_rows.append({"question": item["question"], "refused": refused})
        print(f"  [拒答 {i}/{len(evalset['unanswerable'])}] {'✓拒答' if refused else '✗未拒答'} | {item['question'][:36]}")

    summary = {
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "verdicts": verdicts,
        "refusal_rate": refused_cnt / len(evalset["unanswerable"]) * 100 if evalset["unanswerable"] else 0,
        "rows": rows,
        "refusal_rows": ref_rows,
    }
    return summary


# ==================== 报告 ====================

def write_report(retrieval_table, calib, e2e_summary):
    lines = ["# RAG 评测报告\n"]
    if retrieval_table:
        lines.append("## 检索级（33 道可答题，LLM 判定相关性口径）\n")
        lines.append("| 检索模式 | hit@5 | hit@10 | MRR@10 |\n|---|---|---|---|")
        for mode, m in retrieval_table.items():
            lines.append(f"| {mode} | {m['hit@5']:.1f}% | {m['hit@10']:.1f}% | {m['MRR@10']:.3f} |")
        lines.append("")
    if calib:
        lines.append("## 拒答阈值校准\n")
        lines.append(f"- 可答题 top1 重排分：均值 {sum(calib['ans_top1'])/len(calib['ans_top1']):.3f}，最低 {min(calib['ans_top1']):.3f}")
        lines.append(f"- 不可答题 top1 重排分：均值 {sum(calib['un_top1'])/len(calib['un_top1']):.3f}，最高 {max(calib['un_top1']):.3f}")
        lines.append(f"- 最佳阈值 t={calib['best_threshold']}，分类准确率 {calib['best_accuracy']*100:.0f}%\n")
    if e2e_summary:
        lines.append("## 端到端\n")
        lines.append(f"- 平均分：{e2e_summary['avg_score']:.1f}/10")
        lines.append(f"- 判定分布：{e2e_summary['verdicts']}")
        lines.append(f"- 拒答率（不可答题）：{e2e_summary['refusal_rate']:.0f}%\n")
        lines.append("### 逐题明细\n")
        lines.append("| 问题 | 分数 | 判定 | 理由 |\n|---|---|---|---|")
        for r in e2e_summary["rows"]:
            lines.append(f"| {r['question'][:40]} | {r['score']} | {r['verdict']} | {r.get('reason','')[:40]} |")
        lines.append("")
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 报告已写入: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not (args.retrieval or args.e2e or args.all):
        parser.print_help()
        sys.exit(0)

    evalset = load_eval_set()
    print(f"评测集: 可答 {len(evalset['answerable'])} 题 / 不可答 {len(evalset['unanswerable'])} 题\n")

    retrieval_table = calib = e2e_summary = None

    if args.retrieval or args.all:
        print("=== 检索级评测（四模式对比）===")
        t0 = time.perf_counter()
        retrieval_table, ans_top1 = retrieval_eval(evalset)
        print(f"（耗时 {time.perf_counter()-t0:.0f}s）")
        print("\n=== 拒答阈值校准 ===")
        calib = threshold_calibration(evalset, ans_top1)

    if args.e2e or args.all:
        print("\n=== 端到端评测（LLM 判分）===")
        t0 = time.perf_counter()
        e2e_summary = e2e_eval(evalset)
        print(f"（耗时 {time.perf_counter()-t0:.0f}s）")

    write_report(retrieval_table, calib, e2e_summary)
    if e2e_summary:
        print(f"\n平均分 {e2e_summary['avg_score']:.1f}/10 | 判定 {e2e_summary['verdicts']} | 拒答率 {e2e_summary['refusal_rate']:.0f}%")
