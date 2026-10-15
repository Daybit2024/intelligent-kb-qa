"""
RAG 知识库助手 — Gradio 聊天前端（端口 7860）
消费 8001 的 SSE 流式接口：逐字回答 + 引用来源面板 + 可调参数

运行:  python -u web_ui.py   →  http://127.0.0.1:7860
依赖:  RAG 服务(8001，依赖检索服务 8000) 需先启动
"""
import json
import os
import sys
import uuid

sys.stdout.reconfigure(errors="replace")

from dotenv import load_dotenv

load_dotenv()

import requests
import gradio as gr

RAG_API = os.getenv("RAG_API", "http://127.0.0.1:8001")  # 容器部署由 compose 注入 http://rag:8001


def format_sources(meta: dict) -> str:
    """把 SSE 的 meta 事件格式化成来源面板 Markdown"""
    parts = []
    if meta.get("low_confidence"):
        parts.append("⚠️ **低置信**：检索相关度普遍较低，回答可能为拒答")
    dom = meta.get("domain") or "通用"
    conf = meta.get("intent_confidence")
    conf_s = f"（置信度 {conf:.2f}）" if isinstance(conf, (int, float)) else ""
    line = f"**领域**：{dom}{conf_s}"
    if meta.get("domain_filtered"):
        line += " · 已领域过滤"
    parts.append(line)

    if meta.get("rewritten_question"):
        parts.append(f"**检索改写**：{meta['rewritten_question']}")
    n_exp = len(meta.get("expanded_queries") or [])
    if n_exp:
        parts.append(f"**查询扩展**：{n_exp} 条变体")

    srcs = meta.get("sources") or []
    if srcs:
        parts.append("**引用来源**：")
        for s in srcs:
            sc = (s.get("scores") or {}).get("rerank")
            sc_s = f" · 重排 {sc:.3f}" if isinstance(sc, (int, float)) else ""
            parts.append(f"- `#{s['id']}` {s['source']} · {s['section']}{sc_s}")

    t = meta.get("timings") or {}
    if t:
        parts.append("**耗时**：" + " / ".join(f"{k.replace('_ms', '')} {v:.0f}ms" for k, v in t.items()))
    return "\n\n".join(parts) if parts else "（无元信息）"


def stream_chat(message, top_k, enable_intent, enable_expansion, session_id=None):
    """生成器：调用 8001 SSE，逐块 yield (答案文本, 来源面板)"""
    payload = {
        "question": message,
        "top_k": int(top_k),
        "enable_intent": enable_intent,
        "enable_expansion": enable_expansion,
    }
    if session_id:
        payload["session_id"] = session_id
    try:
        resp = requests.post(f"{RAG_API}/chat/stream", json=payload, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield "⚠️ RAG 服务(8001) 未启动——请先运行 stage3-rag-service/rag_service.py", ""
        return
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        yield f"⚠️ 服务返回错误（HTTP {e.response.status_code}）：{detail or e}", ""
        return

    answer, sources_md = "", "（检索中…）"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.replace("data: ", "", 1).strip()
        if line == "[DONE]":
            break
        try:
            evt = json.loads(line)
        except Exception:
            continue
        kind = evt.get("type")
        if kind == "meta":
            sources_md = format_sources(evt)
            yield answer, sources_md
        elif kind == "delta":
            answer += evt.get("content", "")
            yield answer, sources_md
        elif kind == "error":
            answer += f"\n\n⚠️ {evt.get('message', '')}"
            yield answer, sources_md
    if not answer:
        yield "（服务未返回内容）", sources_md


def build_demo():
    with gr.Blocks(title="RAG 知识库助手") as demo:
        gr.Markdown(
            "# 📚 RAG 知识库助手\n"
            "多格式文档知识库（PDF/扫描件/Word/Markdown）· 混合检索 + RRF + 重排 · 流式 + 多轮对话"
        )
        session_state = gr.State(value=None)  # 浏览器会话的 session_id（多轮对话用）
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="对话", height=520)
                msg = gr.Textbox(
                    placeholder="输入问题，回车发送（例如：什么是市盈率？）",
                    label="提问", autofocus=True,
                )
                with gr.Row():
                    top_k = gr.Slider(1, 10, value=4, step=1, label="上下文条数 top_k")
                    enable_intent = gr.Checkbox(value=True, label="意图识别")
                    enable_expansion = gr.Checkbox(value=True, label="查询扩展")
                    clear_btn = gr.Button("清空对话")
            with gr.Column(scale=3):
                gr.Markdown("### 🔍 检索详情")
                sources_panel = gr.Markdown("（提问后显示引用来源 / 领域 / 耗时）")

        def submit(message, history, top_k, ei, ee, sid):
            if not (message or "").strip():
                yield history, "（请输入问题）", "", sid
                return
            if not sid:
                sid = uuid.uuid4().hex[:12]  # 新会话
            history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "…"},
            ]
            for partial, src in stream_chat(message, top_k, ei, ee, session_id=sid):
                history[-1]["content"] = partial or "…"
                yield history, src, "", sid

        msg.submit(
            submit,
            [msg, chatbot, top_k, enable_intent, enable_expansion, session_state],
            [chatbot, sources_panel, msg, session_state],
        )
        # 清空对话 = 同时开启新会话（旧上下文丢弃）
        clear_btn.click(
            lambda: ([], "（提问后显示引用来源）", None),
            None, [chatbot, sources_panel, session_state],
        )

    return demo


if __name__ == "__main__":
    try:
        requests.get(f"{RAG_API}/health", timeout=5)
        print("✓ RAG 服务(8001) 在线")
    except Exception:
        print("⚠️ RAG 服务(8001) 未启动！界面能打开，但提问会报错。")
        print("   请先运行: python stage3-rag-service/rag_service.py")

    demo = build_demo()
    demo.queue()
    print("Gradio 界面: http://127.0.0.1:7860")
    # 容器内由 compose 注入 BIND_HOST=0.0.0.0（否则端口映射无法从容器外访问）
    demo.launch(server_name=os.getenv("BIND_HOST", "127.0.0.1"), server_port=7860)
