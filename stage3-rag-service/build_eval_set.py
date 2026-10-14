"""
评测集构建：从知识库文段"反查"生成 问题 + 参考答案 + 答案文段标注
- 采样：按来源分层，只取长度 >400 字符的正文文段（跳过目录类碎段）
- 每题由 LLM 基于文段生成（问题必须能由该文段回答）→ 天然带标注
- 另附 10 道"知识库无法回答"的陷阱题（评测拒答 + 阈值校准用）
输出: eval_set.json
"""
import json
import os
import random
import re
import sys
import time

sys.stdout.reconfigure(errors="replace")

# 复用 stage2 的配置（MySQL 连接参数）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2-milvus"))

from dotenv import load_dotenv

load_dotenv()

import pymysql
from openai import OpenAI
import milvus_config as config

llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

GEN_PROMPT = (
    "你将看到一段知识库文档片段。请完成两件事：\n"
    "1. 基于这段内容写出一个用户可能会问的问题（必须能由这段内容回答，尽量自然口语化）\n"
    "2. 写出该问题的参考答案（严格基于片段内容，不要编造）\n"
    '只输出 JSON：{"question": "...", "answer": "..."}'
)

# 分层采样配额（按来源）
QUOTAS = {"fin.md": 10, "ai_dev.md": 10, "law.md": 8, "金融分析报告.pdf": 5}

SOURCE_DOMAIN = {
    "fin.md": "finance",
    "金融分析报告.pdf": "finance",
    "law.md": "law",
    "ai_dev.md": "ai",
}

# 知识库无法回答的陷阱题（一半"沾边但答不出"，一半完全不相关）
UNANSWERABLE = [
    {"question": "郑州大学市盈率多少", "reason": "实体错配：大学不是上市公司，文中无此实体"},
    {"question": "特斯拉上海工厂去年产量是多少", "reason": "文中无该公司数据"},
    {"question": "如何办理护照", "reason": "办事流程，文中无"},
    {"question": "我的电脑蓝屏了怎么修", "reason": "技术排障，文中无"},
    {"question": "请推荐几部科幻电影", "reason": "推荐类，文中无"},
    {"question": "《红楼梦》的作者是谁", "reason": "常识问答，文中无"},
    {"question": "红烧肉怎么做", "reason": "完全不相关"},
    {"question": "明天天气怎么样", "reason": "闲聊"},
    {"question": "2026年世界杯在哪个国家举办", "reason": "时事，文中无"},
    {"question": "怎么用 Python 写一个贪吃蛇游戏", "reason": "编程教程，文中无"},
]


def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def sample_chunks():
    conn = pymysql.connect(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE, charset="utf8mb4",
    )
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, text, source, section FROM {config.MYSQL_TABLE} "
            f"WHERE CHAR_LENGTH(text) > 400"
        )
        rows = cur.fetchall()
    conn.close()

    by_source = {}
    for r in rows:
        by_source.setdefault(r[2], []).append(r)

    random.seed(42)
    picked = []
    for src, n in QUOTAS.items():
        pool = by_source.get(src, [])
        take = min(n, len(pool))
        picked.extend(random.sample(pool, take))
        print(f"  采样 {src}: {take}/{len(pool)} 段")
    return picked


def gen_qa(chunk_id, text):
    resp = llm.chat.completions.create(
        model="qwen-plus", temperature=0.7, max_tokens=500,
        messages=[
            {"role": "system", "content": GEN_PROMPT},
            {"role": "user", "content": f"文档片段：\n{text[:1500]}"},
        ],
    )
    data = _extract_json(resp.choices[0].message.content)
    if not data or not data.get("question") or not data.get("answer"):
        return None
    return {"question": data["question"].strip(), "answer": data["answer"].strip()}


def main():
    print("=== 采样文段 ===")
    chunks = sample_chunks()
    print(f"共取其 {len(chunks)} 段\n")

    print("=== 逐段生成 问题+参考答案 ===")
    answerable = []
    for i, (cid, text, source, section) in enumerate(chunks, 1):
        qa = None
        for attempt in range(2):
            try:
                qa = gen_qa(cid, text)
                if qa:
                    break
            except Exception as e:
                print(f"  段 {cid} 第 {attempt + 1} 次失败: {e}")
                time.sleep(2)
        if not qa:
            print(f"  段 {cid} 生成失败，跳过")
            continue
        answerable.append({
            "id": len(answerable) + 1,
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "source_ids": [cid],
            "source": source,
            "section": section,
            "domain": SOURCE_DOMAIN.get(source, "general"),
        })
        print(f"  [{i}/{len(chunks)}] id={cid} -> {qa['question'][:40]}")

    unanswerable = [
        {"id": 1000 + i, **item} for i, item in enumerate(UNANSWERABLE, 1)
    ]

    out = {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "meta": {
            "total_answerable": len(answerable),
            "total_unanswerable": len(unanswerable),
            "note": "answerable 由文段反查生成；unanswerable 用于拒答评测与阈值校准",
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print()
    print(f"✓ 评测集已生成: {out_path}")
    print(f"  可答题: {len(answerable)} | 不可答题: {len(unanswerable)}")


if __name__ == "__main__":
    main()
