"""
stage2 第五课：检索服务 API（FastAPI，端口 8000）
把 query_engine 的四种检索包装成 HTTP 服务，供 stage3 的 RAG 服务调用

启动:  python api_server.py        （或 uvicorn api_server:app --port 8000）
文档:  http://localhost:8000/docs  （Swagger 调试界面）
"""
import sys

sys.stdout.reconfigure(errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import store
import query_engine as qe
import milvus_config as config

app = FastAPI(
    title="RAG 检索服务",
    description="四种检索：vector(向量) / keyword(BM25) / hybrid(RRF) / rerank(混合+重排)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="检索问题")
    search_type: str = Field("hybrid", description="检索模式: vector / keyword / hybrid / rerank")
    top_k: int = Field(10, ge=1, le=50, description="返回条数")


class ResultItem(BaseModel):
    id: int
    text: str
    source: str
    section: str
    domain: str = ""   # 领域元数据（供上层领域过滤）
    scores: dict  # 按模式不同: distance / score(BM25) / rrf+各路排名 / rerank+rrf+排名


class SearchResponse(BaseModel):
    query: str
    search_type: str
    count: int
    results: list[ResultItem]


class RerankRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="用于精排的查询（通常是原始问题）")
    ids: list[int] = Field(..., max_length=200, description="候选 chunk id 列表（先快速召回，再送这里精排）")
    top_k: int = Field(10, ge=1, le=50, description="返回条数")


# ==================== 端点 ====================

@app.get("/health")
def health():
    return {"status": "healthy", "service": "rag-search-api"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """统一检索入口：四种模式，结果已回表（含正文）"""
    if req.search_type == "vector":
        raw = qe.vector_search(req.query, req.top_k)
    elif req.search_type == "keyword":
        raw = qe.keyword_search(req.query, req.top_k)
    elif req.search_type == "hybrid":
        raw = qe.hybrid_search(req.query, req.top_k)
    elif req.search_type == "rerank":
        raw = qe.rerank_search(req.query, req.top_k)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的 search_type: {req.search_type}")

    texts = qe.fetch_texts([r["id"] for r in raw])
    results = []
    for r in raw:
        meta = texts.get(r["id"], {})
        results.append(ResultItem(
            id=r["id"],
            text=meta.get("text", ""),
            source=meta.get("source", ""),
            section=meta.get("section", ""),
            domain=meta.get("domain", ""),
            scores={k: v for k, v in r.items() if k != "id"},
        ))
    return SearchResponse(
        query=req.query, search_type=req.search_type, count=len(results), results=results
    )


@app.post("/rerank")
def rerank(req: RerankRequest):
    """对给定的候选 id 列表做一次精排（两阶段检索的第二阶段：快速召回 → 一次性重排）"""
    results = qe.rerank_documents(req.query, req.ids, req.top_k)
    return {"query": req.query, "count": len(results), "results": results}


@app.get("/domains")
def domains():
    """领域表（供 RAG 服务的意图识别动态加载）"""
    return {"domains": store.list_domains()}


@app.get("/stats")
def stats():
    """三库数据量概览"""
    from pymilvus import connections

    conn = store.mysql_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {config.MYSQL_TABLE}")
        mysql_count = cur.fetchone()[0]
    conn.close()

    es_count = store.es_client().count(index=config.ES_INDEX)["count"]

    col = store.milvus_collection()
    milvus_count = col.num_entities
    connections.disconnect("default")

    return {
        "mysql_chunks": mysql_count,
        "es_docs": es_count,
        "milvus_entities": milvus_count,
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("RAG 检索服务启动中...")
    print("  API 文档: http://localhost:8000/docs")
    print("  健康检查: http://localhost:8000/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
