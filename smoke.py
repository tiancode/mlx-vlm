#!/usr/bin/env python3
"""Check health, model discovery and four GLM inference interfaces."""

import argparse
import base64
import io
import json
import os
import sys

import httpx
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_response_stream(lines):
    declared = {}
    finished = set()
    text = []
    completed = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            continue
        event = json.loads(raw)
        kind = event.get("type", "")
        require(kind not in ("error", "response.failed", "response.incomplete"),
                f"Responses 失败: {event}")
        if kind == "response.output_item.added":
            index, item = event["output_index"], event["item"]
            require(index not in declared, f"重复的 output_index: {index}")
            declared[index] = item
        if kind.endswith(".delta") or kind == "response.output_item.done":
            index = event["output_index"]
            item_id = event.get("item_id") or event.get("item", {}).get("id")
            require(index in declared and declared[index]["id"] == item_id,
                    f"事件未声明或索引不匹配: {kind}")
            if kind == "response.output_item.done":
                finished.add(index)
        if kind == "response.output_text.delta":
            text.append(event["delta"])
        if kind == "response.completed":
            completed = event["response"]
    require(completed is not None and completed.get("status") == "completed",
            "Responses 缺少成功终态")
    output = completed.get("output", [])
    require(len(output) == len(declared) and set(declared) == finished,
            "Responses 声明、结束事件与终态数量不一致")
    for index, item in enumerate(output):
        require(index in declared and item["id"] == declared[index]["id"],
                "Responses 终态索引不一致")
    answer = "".join(text).strip()
    require("4" in answer or "四" in answer, f"Responses 回答不符合预期: {answer!r}")
    return answer


def run_checks(client, model):
    health = client.get("/health")
    health.raise_for_status()
    require(health.json().get("status") in ("ok", "healthy"), "健康检查返回异常状态")
    print("通过: /health", flush=True)

    response = client.get("/v1/models")
    response.raise_for_status()
    require(model in [entry["id"] for entry in response.json()["data"]],
            f"模型列表缺少 {model}")
    print("通过: /v1/models", flush=True)

    def post(path, payload):
        response = client.post(path, json={"model": model, **payload})
        response.raise_for_status()
        return response.json()

    def chat(content):
        result = post("/v1/chat/completions", {
            "max_tokens": 1024, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": content}],
        })
        choice = result["choices"][0]
        require(choice.get("finish_reason") == "stop", "Chat 未正常完成")
        answer = (choice["message"].get("content") or "").strip()
        require(answer, "Chat 正文为空")
        return answer

    answer = chat("Reply with the single word PONG.")
    require("PONG" in answer.upper(), f"Chat 回答不符合预期: {answer!r}")
    print("通过: Chat 文本", flush=True)

    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (20, 90, 220)).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    answer = chat([
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        {"type": "text", "text": "这张图是什么颜色？只答颜色。"},
    ])
    require("蓝" in answer or "blue" in answer.lower(), f"图片颜色识别错误: {answer!r}")
    print("通过: Chat 图片理解", flush=True)

    with client.stream("POST", "/v1/responses", json={
        "model": model, "input": "2 加 2 等于几？一句话。",
        "max_output_tokens": 2000, "stream": True, "store": False,
    }) as response:
        response.raise_for_status()
        check_response_stream(response.iter_lines())
    print("通过: Responses 流式事件及终态", flush=True)

    result = post("/v1/messages", {
        "max_tokens": 1024, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "Reply with the single word PONG."}],
    })
    answer = "".join(item.get("text", "") for item in result.get("content", [])
                     if item.get("type") == "text")
    require(result.get("stop_reason") == "end_turn" and "PONG" in answer.upper(),
            f"Messages 未正常回答: {result}")
    print("通过: Messages", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:1235")
    parser.add_argument("api_key", nargs="?", default=os.environ.get("API_KEY", "happy-coding-axm"))
    parser.add_argument("--model", default=os.environ.get("PUBLIC_NAME", "glm-5.3-flash"))
    args = parser.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {args.api_key}"},
                          timeout=httpx.Timeout(900, connect=10), trust_env=False) as client:
            run_checks(client, args.model)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"检查失败: {error}", file=sys.stderr)
        return 1
    print("全部六项检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
