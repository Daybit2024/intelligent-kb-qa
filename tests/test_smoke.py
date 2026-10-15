"""
冒烟测试：
  单元测试（零外部依赖）：评测集完整性、切分函数
  集成测试（服务/数据库在线时执行，离线自动 skip）

运行:  pytest tests/ -v
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================== 单元测试 ====================

def test_eval_set_integrity():
    """评测集结构完整：33 可答 + 10 陷阱，字段齐全"""
    path = os.path.join(ROOT, "stage3-rag-service", "eval_set.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["answerable"]) == 33
    assert len(data["unanswerable"]) == 10
    for item in data["answerable"]:
        assert item["question"] and item["reference_answer"] and item["source_ids"]
    for item in data["unanswerable"]:
        assert item["question"] and item["reason"]


def test_split_by_window():
    """窗口切分：段落聚合到目标长度附近、来源标注正确"""
    from ingest import split_by_window

    content = "\n\n".join(
        f"第{i}段内容，用于验证窗口切分逻辑。" + "测试文本" * 60 for i in range(1, 9)
    )
    chunks = split_by_window(content, "test.md", target=500, overlap_ratio=0.15)
    assert len(chunks) >= 2
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert all(c["source"] == "test.md" for c in chunks)


def test_conversation_manager_trim():
    """会话记忆：写入侧截断——超出上限的旧消息被自动丢弃"""
    from rag_service import ConversationManager

    m = ConversationManager()
    for i in range(20):  # 20 轮 = 40 条消息
        m.add("s1", "user", f"q{i}")
        m.add("s1", "assistant", f"a{i}")

    hist = m.conversations["s1"]
    assert len(hist) == ConversationManager.MAX_HISTORY_MESSAGES
    assert hist[0]["content"] == "q15"  # 剩余窗口 = 最近 5 轮（q15..a19）
    assert hist[-1]["content"] == "a19"


def test_split_by_heading():
    """标题切分：## / ### 层级与 section 命名正确"""
    from ingest import split_by_heading

    content = (
        "# 总标题\n前言内容。\n\n"
        "## 第一章\n第一章内容。\n\n"
        "### 1.1 小节\n小节内容。\n\n"
        "## 第二章\n第二章内容。"
    )
    chunks = split_by_heading(content, "t.md")
    sections = [c["section"] for c in chunks]
    assert "前言" in sections
    assert "第一章" in sections
    assert "第一章 - 1.1 小节" in sections
    assert "第二章" in sections


# ==================== 集成测试（服务在线才跑，离线自动跳过） ====================

def _service_up(url: str) -> bool:
    import requests
    try:
        return requests.get(url, timeout=3).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _service_up("http://127.0.0.1:8000/health"), reason="检索服务(8000)未启动")
def test_search_api_with_domain():
    """检索 API：返回结果携带 domain 元数据（领域过滤的数据基础）"""
    import requests
    resp = requests.post(
        "http://127.0.0.1:8000/search",
        json={"query": "市盈率", "search_type": "keyword", "top_k": 3},
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert all("domain" in item for item in body["results"])


@pytest.mark.skipif(not _service_up("http://127.0.0.1:8000/health"), reason="检索服务(8000)未启动")
def test_domains_endpoint_with_counts():
    """领域表端点：每个领域携带文件数（引用计数，归零领域会被自动清除）"""
    import requests
    resp = requests.get("http://127.0.0.1:8000/domains", timeout=10)
    assert resp.status_code == 200
    domains = resp.json()["domains"]
    assert domains
    assert all("file_count" in d and d["file_count"] >= 1 for d in domains)


@pytest.mark.skipif(not _service_up("http://127.0.0.1:8001/health"), reason="RAG 服务(8001)未启动")
def test_rag_chat_returns_confidence_signal():
    """RAG 服务：响应包含 low_confidence 信号（评测校准阈值的线上透出）"""
    import requests
    resp = requests.post(
        "http://127.0.0.1:8001/chat",
        json={"question": "什么是市盈率", "top_k": 3},
        timeout=300,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert isinstance(body["low_confidence"], bool)
