"""
stage3 第二课：完整版 RAG 服务（端口 8001）
能力：意图识别 / 查询扩展 / 领域过滤 / 拒答约束 / 流式SSE / 多轮对话
架构：8001(本服务) → HTTP → 8000(检索服务) → MySQL + Milvus + ES

运行:  python rag_service.py   （依赖 8000 检索服务已启动）
文档:  http://localhost:8001/docs
"""
import json
import logging
import os
import re
import sys
import time

sys.stdout.reconfigure(errors="replace")

from dotenv import load_dotenv

load_dotenv()

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_service")

SEARCH_API = "http://127.0.0.1:8000"   # stage2 检索服务（127.0.0.1 而非 localhost，避免 IPv6 回退 2 秒延迟）
LLM_MODEL = "qwen-plus"

llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 领域 → 来源文件（领域过滤用；文件在 documents/ 里）
DOMAIN_SOURCES = {
    "ai": ["ai_dev.md"],
    "finance": ["fin.md", "金融分析报告.pdf"],
    "law": ["law.md"],
}
DOMAIN_NAMES = {"law": "法律", "finance": "金融", "ai": "人工智能", "general": "通用"}

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手，请只根据提供的参考资料回答问题。规则：\n"
    "1. 如果参考资料与问题无关、或不足以回答，请明确说明“资料中没有相关内容”，不要编造，也不要用你自己的常识补充；\n"
    "2. 忠实于参考资料的原意；引用到某条资料时可用 [参考N] 标注；\n"
    "3. 回答简洁清晰。"
)

INTENT_PROMPT = (
    "你是意图分类器。请把用户问题分类到以下领域之一：\n"
    "- law（法律）\n- finance（金融）\n- ai（人工智能）\n- general（通用，不属于以上领域）\n\n"
    '只输出一个 JSON，格式：{"domain": "xxx", "confidence": 0.9, "reason": "简短理由"}'
)

EXPANSION_PROMPT = (
    "请为下面的问题生成 3 个语义相同但表述不同的检索查询（用于提升检索召回）。\n"
    "要求：每行一个，不要编号、不要解释，只输出这 3 行。"
)

REWRITE_PROMPT = (
    "根据对话历史，把用户的最新问题改写成一个不依赖上下文、可以独立检索的完整问题。\n"
    "要求：只输出改写后的问题本身，不要任何解释；如果最新问题本身已独立，原样输出。"
)


# ==================== 意图识别 ====================

def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def detect_intent(question):
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.1, max_tokens=200,
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    data = _extract_json(resp.choices[0].message.content)
    if not data or data.get("domain") not in DOMAIN_NAMES:
        return {"domain": "general", "confidence": 0.0, "reason": "解析失败，按通用处理"}
    return {
        "domain": data["domain"],
        "confidence": float(data.get("confidence", 0.5)),
        "reason": data.get("reason", ""),
    }


# ==================== 查询扩展 ====================

def expand_queries(question, n=3):
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.7, max_tokens=200,
        messages=[
            {"role": "system", "content": EXPANSION_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    lines = [l.strip(" -·\t") for l in (resp.choices[0].message.content or "").splitlines()]
    return [l for l in lines if l][:n]


# ==================== 查询改写（多轮指代消解） ====================

def rewrite_question(question, history):
    """把'那它...'这类指代问题改写成独立问题（仅用于检索；生成仍用原问题）"""
    hist_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-6:])
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.1, max_tokens=100,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"对话历史：\n{hist_text}\n\n最新问题：{question}"},
        ],
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question


# ==================== 检索编排 ====================

def search_api(query, top_k, search_type="rerank"):
    try:
        resp = requests.post(
            f"{SEARCH_API}/search",
            json={"query": query, "search_type": search_type, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["results"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="检索服务(8000)未启动，请先运行 stage2-milvus/api_server.py",
        )


def rerank_api(query, ids, top_k):
    """调 8000 的重排端点：对给定候选 id 列表做一次精排（两阶段检索的第二阶段）"""
    try:
        resp = requests.post(
            f"{SEARCH_API}/rerank",
            json={"query": query, "ids": ids, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["results"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="检索服务(8000)未启动，请先运行 stage2-milvus/api_server.py",
        )


MAX_RERANK_CANDIDATES = 40  # 送进重排的候选上限（成本保护）


def retrieve_candidates(question, top_k, enable_expansion):
    """两阶段检索：
    阶段1 多个查询（原问题+变体）各走 hybrid 快速召回 → 按 id 合并去重
    阶段2 合并后的候选只做一次最终重排（旧版是每个查询精排一次，慢约 4 倍）"""
    queries = [question]
    expanded = []
    if enable_expansion:
        try:
            expanded = expand_queries(question)
            queries += expanded
        except Exception as e:
            logger.warning(f"查询扩展失败，跳过: {e}")

    # 无扩展：本来就只需要一次精排
    if len(queries) == 1:
        return search_api(question, top_k, "rerank"), expanded

    # 阶段1：多查询 hybrid 召回（快），合并去重
    merged = {}
    for q in queries:
        for r in search_api(q, top_k * 2, "hybrid"):
            merged.setdefault(r["id"], r)
    if not merged:
        return [], expanded

    candidates = sorted(
        merged.values(), key=lambda x: -x["scores"].get("rrf", 0.0)
    )[:MAX_RERANK_CANDIDATES]

    # 阶段2：一次最终重排
    return rerank_api(question, [c["id"] for c in candidates], top_k * 2), expanded


def apply_domain_filter(candidates, domain):
    """领域过滤：按来源筛；命中不足 3 条则回退不筛"""
    sources = DOMAIN_SOURCES.get(domain)
    if not sources:
        return candidates, False
    filtered = [c for c in candidates if c["source"] in sources]
    if len(filtered) >= 3:
        return filtered, True
    return candidates, False


# ==================== 生成 ====================

def build_messages(question, contexts, history=None):
    if contexts:
        refs = "\n\n".join(
            f"[参考{i + 1}]（{c['source']} · {c['section']}）\n{c['text']}"
            for i, c in enumerate(contexts)
        )
    else:
        refs = "（没有检索到任何参考资料）"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": f"参考资料：\n{refs}\n\n问题：{question}"})
    return msgs


def generate_answer(question, contexts, history=None):
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.3, max_tokens=1500,
        messages=build_messages(question, contexts, history),
    )
    return resp.choices[0].message.content


def generate_answer_stream(question, contexts, history=None):
    stream = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0.3, max_tokens=1500,
        messages=build_messages(question, contexts, history), stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==================== 完整流水线 ====================

def rag_pipeline(question, top_k, enable_intent, enable_expansion, enable_domain_filter):
    timings = {}

    intent = {"domain": "general", "confidence": None, "reason": None}
    t0 = time.perf_counter()
    if enable_intent:
        try:
            intent = detect_intent(question)
        except Exception as e:
            logger.warning(f"意图识别失败: {e}")
    timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    candidates, expanded = retrieve_candidates(question, top_k, enable_expansion)
    timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    filtered = False
    if enable_domain_filter and enable_intent:
        candidates, filtered = apply_domain_filter(candidates, intent["domain"])

    return {
        "contexts": candidates[:top_k],
        "expanded": expanded,
        "intent": intent,
        "domain_filtered": filtered,
        "timings": timings,
    }


def format_sources(contexts):
    return [
        {"id": c["id"], "source": c["source"], "section": c["section"], "scores": c["scores"]}
        for c in contexts
    ]


# ==================== 多轮会话（内存版） ====================

class ConversationManager:
    """学习用内存实现；生产环境应替换为 Redis/MySQL（历史会随进程重启丢失、
    也无法在多实例间共享——对应第 6 课的讨论）"""

    def __init__(self):
        self.conversations = {}

    def add(self, session_id, role, content):
        self.conversations.setdefault(session_id, []).append({"role": role, "content": content})

    def get(self, session_id, limit=10):
        return self.conversations.get(session_id, [])[-limit:]

    def clear(self, session_id):
        self.conversations.pop(session_id, None)


manager = ConversationManager()


# ==================== FastAPI ====================

app = FastAPI(
    title="完整版 RAG 服务",
    description="意图识别 / 查询扩展 / 领域过滤 / 拒答约束 / 流式 / 多轮",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ChatRequest(BaseModel):
    question: str
    # 实验定案：k∈[3,7] 为质量平台期（9.82~9.88，无显著差异），平台期内取偏小值省 token
    top_k: int = Field(4, ge=1, le=20)
    enable_intent: bool = Field(True, description="意图识别")
    enable_expansion: bool = Field(True, description="查询扩展")
    enable_domain_filter: bool = Field(True, description="领域过滤")


class HistoryChatRequest(ChatRequest):
    session_id: str = Field(..., description="会话 ID（多轮）")


class ChatResponse(BaseModel):
    question: str
    rewritten_question: str | None = None   # 多轮时：改写后用于检索的独立问题
    answer: str
    domain: str | None
    intent_confidence: float | None
    domain_filtered: bool
    expanded_queries: list
    sources: list
    retrieved: int
    timings: dict


@app.get("/health")
def health():
    return {"status": "healthy", "service": "full-rag"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = rag_pipeline(
        req.question, req.top_k, req.enable_intent, req.enable_expansion, req.enable_domain_filter
    )
    t0 = time.perf_counter()
    answer = generate_answer(req.question, result["contexts"])
    result["timings"]["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        f"[chat] domain={result['intent']['domain']} 扩展{len(result['expanded'])}条 "
        f"命中{len(result['contexts'])}条 域过滤={result['domain_filtered']} {result['timings']}"
    )
    return ChatResponse(
        question=req.question,
        answer=answer,
        domain=result["intent"]["domain"],
        intent_confidence=result["intent"]["confidence"],
        domain_filtered=result["domain_filtered"],
        expanded_queries=result["expanded"],
        sources=format_sources(result["contexts"]),
        retrieved=len(result["contexts"]),
        timings=result["timings"],
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式：先发 meta（来源/领域），再逐块发答案"""
    result = rag_pipeline(
        req.question, req.top_k, req.enable_intent, req.enable_expansion, req.enable_domain_filter
    )

    def event_stream():
        try:
            meta = {
                "type": "meta",
                "domain": result["intent"]["domain"],
                "intent_confidence": result["intent"]["confidence"],
                "domain_filtered": result["domain_filtered"],
                "expanded_queries": result["expanded"],
                "sources": format_sources(result["contexts"]),
                "timings": result["timings"],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            for piece in generate_answer_stream(req.question, result["contexts"]):
                yield f"data: {json.dumps({'type': 'delta', 'content': piece}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[stream] 出错: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/with-history", response_model=ChatResponse)
def chat_with_history(req: HistoryChatRequest):
    """多轮对话：携带 session_id，自动带上最近历史；
    检索用"改写后的独立问题"（解决"那它..."类指代），生成用原问题+历史"""
    history = manager.get(req.session_id)

    search_question = req.question
    rewrite_ms = 0.0
    if history:
        t0 = time.perf_counter()
        try:
            search_question = rewrite_question(req.question, history)
        except Exception as e:
            logger.warning(f"查询改写失败，用原问题检索: {e}")
        rewrite_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = rag_pipeline(
        search_question, req.top_k, req.enable_intent, req.enable_expansion, req.enable_domain_filter
    )
    result["timings"]["rewrite_ms"] = rewrite_ms

    t0 = time.perf_counter()
    answer = generate_answer(req.question, result["contexts"], history=history)
    result["timings"]["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    manager.add(req.session_id, "user", req.question)
    manager.add(req.session_id, "assistant", answer)

    logger.info(
        f"[chat/with-history] session={req.session_id} 历史{len(history)}条 "
        f"检索用「{search_question}」-> 新增一轮"
    )
    return ChatResponse(
        question=req.question,
        rewritten_question=search_question if search_question != req.question else None,
        answer=answer,
        domain=result["intent"]["domain"],
        intent_confidence=result["intent"]["confidence"],
        domain_filtered=result["domain_filtered"],
        expanded_queries=result["expanded"],
        sources=format_sources(result["contexts"]),
        retrieved=len(result["contexts"]),
        timings=result["timings"],
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("完整版 RAG 服务启动中...")
    print("  API 文档: http://localhost:8001/docs")
    print("  依赖: 检索服务(8000)需先启动")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)
