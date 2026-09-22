#!/usr/bin/env python3
"""统一路由 GLM 与图片 API，映射 GLM 模型字段并适配 Responses 流事件。

图片请求独立转发；GLM 请求固定到启动时的权重路径，避免模型别名触发权重切换。
"""

import json
import os
from contextlib import asynccontextmanager

import httpx
import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from request_lifecycle import RequestLifecycleMiddleware, current_request

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "http://127.0.0.1:1236")
PUBLIC_NAME = os.environ.get("PROXY_MODEL_NAME", "glm-5.3-flash")
UPSTREAM_MODEL = os.environ["PROXY_UPSTREAM_MODEL"]
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("PROXY_PORT", "1235"))
IMAGE_UPSTREAM = os.environ.get("PROXY_IMAGE_UPSTREAM", "")
IMAGE_TIMEOUT = float(os.environ.get("IMAGE_REQUEST_TIMEOUT", "1800")) + 15

# 逐跳头不能转发；Host 由 httpx 按上游地址重建；长度/编码交给 httpx 重算。
DROP_REQUEST = {"host", "content-length", "connection", "keep-alive",
                "transfer-encoding", "upgrade", "proxy-authorization", "te"}
DROP_RESPONSE = {"content-length", "connection", "keep-alive",
                 "transfer-encoding", "upgrade", "content-encoding"}


def _reasoning_text_of(item: dict) -> str:
    parts = item.get("summary") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _with_reasoning_content(item: dict) -> dict:
    """保留上游 summary，并提供 content[].reasoning_text 供客户端读取。"""
    if item.get("type") != "reasoning" or item.get("content"):
        return item
    text = _reasoning_text_of(item)
    if not text:
        return item
    return {**item, "content": [{"type": "reasoning_text", "text": text}]}


def rewrite_request(body: bytes) -> bytes:
    """把顶层 model 换成后端真实 id。不是 JSON 就原样返回。"""
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or "model" not in payload:
        return body
    payload["model"] = UPSTREAM_MODEL
    return json.dumps(payload).encode()


def rewrite_response(body: bytes) -> bytes:
    """后端真实 id 换回短名，并给 reasoning item 补 content。"""
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body

    return json.dumps(rewrite_payload(payload)).encode()


def rewrite_payload(node):
    if isinstance(node, dict):
        node = {
            key: (PUBLIC_NAME if key in ("model", "id") and value == UPSTREAM_MODEL
                  else rewrite_payload(value))
            for key, value in node.items()
        }
        return _with_reasoning_content(node)
    if isinstance(node, list):
        return [rewrite_payload(value) for value in node]
    return node


class ResponsesStreamFixer:
    """适配 mlx-vlm 0.7.1 缺失的 reasoning 声明和结束事件。

    缓冲 message 开场事件，在首个正文 token 前释放；有 reasoning 时将
    message 移到索引 1，使增量事件与 response.completed 的 output 一致。
    """

    # 这两个不算正文，不能用来判定"缓冲窗口结束"
    PRELUDE = {"response.created", "response.in_progress"}

    def __init__(self):
        self._buffered = []
        self._closed = False
        self._reasoning_id = None
        self._reasoning_text = []
        self._reasoning_closed = False
        self._shift = False
        self._message_ids = set()

    def _item(self) -> dict:
        text = "".join(self._reasoning_text)
        return {
            "id": self._reasoning_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}] if text else [],
            "content": [{"type": "reasoning_text", "text": text}] if text else [],
        }

    def _shifted(self, data: dict) -> dict:
        if not self._shift or data.get("output_index") != 0:
            return data
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        item_id = (item or {}).get("id") or data.get("item_id")
        if item_id in self._message_ids:
            return {**data, "output_index": 1}
        return data

    def _close(self):
        """正文开始了：先把 reasoning 收尾，再放出扣住的 message 开场事件。"""
        if self._closed:
            return []
        self._closed = True
        out = []
        if self._reasoning_id is not None and not self._reasoning_closed:
            self._reasoning_closed = True
            text = "".join(self._reasoning_text)
            out.append(("response.reasoning_text.done", {
                "type": "response.reasoning_text.done",
                "item_id": self._reasoning_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            }))
            out.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": self._item(),
            }))
        out.extend((name, self._shifted(data)) for name, data in self._buffered)
        self._buffered = []
        return out

    def feed(self, name, data):
        if not isinstance(data, dict):
            return self._close() + [(name, data)]

        kind = data.get("type") or name or ""
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        item_id = (item or {}).get("id") or data.get("item_id")

        if not self._closed and kind in self.PRELUDE:
            return [(name, data)]

        if not self._closed and kind == "response.output_item.added" \
                and (item or {}).get("type") == "message":
            self._message_ids.add(item_id)
            self._buffered.append((name, data))
            return []
        if not self._closed and kind == "response.content_part.added" \
                and item_id in self._message_ids:
            self._buffered.append((name, data))
            return []

        if kind == "response.reasoning_text.delta":
            out = []
            if self._reasoning_id is None:
                self._reasoning_id = item_id
                self._shift = True
                out.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": item_id, "type": "reasoning",
                             "summary": [], "content": []},
                }))
            self._reasoning_text.append(data.get("delta") or "")
            out.append((name, {**data, "output_index": 0}))
            return out

        # 上游迟到的那个 done：我们已经在 _close() 里按累积文本补发过了，丢掉重复的
        if kind == "response.reasoning_text.done":
            if self._reasoning_closed:
                return []
            if isinstance(data.get("text"), str):
                self._reasoning_text = [data["text"]]
            return self._close()

        if kind == "response.completed" and isinstance(data.get("response"), dict):
            response = data["response"]
            output = response.get("output")
            if isinstance(output, list):
                data = {**data, "response": {
                    **response,
                    "output": [_with_reasoning_content(o) if isinstance(o, dict) else o
                               for o in output],
                }}

        return self._close() + [(name, self._shifted(data))]


def _encode(name, data) -> bytes:
    body = data if isinstance(data, bytes) else json.dumps(rewrite_payload(data)).encode()
    head = b"event: " + name.encode() + b"\n" if name else b""
    return head + b"data: " + body + b"\n\n"


def _render(fixer, block: bytes):
    name = None
    data_lines = []
    for line in block.split(b"\n"):
        if line.startswith(b"event:"):
            name = line[6:].strip().decode()
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        yield block + b"\n\n"
        return
    raw = b"\n".join(data_lines)
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        yield block + b"\n\n"
        return
    for out_name, out_data in fixer.feed(name, payload):
        yield _encode(out_name, out_data)


async def stream_responses(response: httpx.Response):
    """Responses 的 SSE：按事件块解析、补齐、重新编码。"""
    fixer = ResponsesStreamFixer()
    buf = b""
    async for chunk in response.aiter_raw():
        buf = (buf + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            if block.strip():
                for out in _render(fixer, block):
                    yield out
    if buf.strip():
        for out in _render(fixer, buf):
            yield out


async def stream_passthrough(response: httpx.Response):
    """其它 SSE 按完整 data 行改写 JSON 模型字段，保留正文及非 JSON 行。"""
    def rewrite_line(line):
        if not line.startswith(b"data:"):
            return line
        raw = line[5:]
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return line
        ending = b"\r" if line.endswith(b"\r") else b""
        return b"data: " + json.dumps(rewrite_payload(payload)).encode() + ending

    carry = b""
    async for chunk in response.aiter_raw():
        carry += chunk
        if b"\n" not in carry:
            continue
        *lines, carry = carry.split(b"\n")
        yield b"\n".join(rewrite_line(line) for line in lines) + b"\n"
    if carry:
        yield rewrite_line(carry)


@asynccontextmanager
async def lifespan(app):
    # 长 prefill 不设读取超时；图片请求使用独立客户端和有界超时。
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    async with httpx.AsyncClient(base_url=UPSTREAM, timeout=timeout) as client:
        app.state.client = client
        async with httpx.AsyncClient(base_url=IMAGE_UPSTREAM or "http://127.0.0.1:1238",
                                    timeout=httpx.Timeout(IMAGE_TIMEOUT, connect=5)) as image_client:
            app.state.image_client = image_client
            yield


async def proxy_image(request: Request) -> Response:
    if not IMAGE_UPSTREAM:
        return JSONResponse({"error": {"message": "Image service is disabled"}}, status_code=503)
    body = bytearray()
    is_edit = request.url.path.removeprefix("/v1").rstrip("/") == "/images/edits"
    limit = 64 * 1024**2 if is_edit else 65536
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return JSONResponse({"error": {"message": f"Request body exceeds {limit} bytes"}}, status_code=413)
    path = request.url.path
    if path.rstrip("/") in ("/v1/images/health", "/images/health"):
        path = "/health"
    try:
        response = await request.app.state.image_client.request(
            request.method, path, params=request.url.query.encode() or None,
            headers={k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST},
            content=bytes(body),
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": {"message": "Image service timed out"}}, status_code=504)
    except httpx.RequestError:
        return JSONResponse({"error": {"message": "Image service unavailable"}}, status_code=503)
    return Response(response.content, status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE})


async def proxy(request: Request) -> Response:
    if request.url.path.removeprefix("/v1").startswith("/images/"):
        return await proxy_image(request)
    client: httpx.AsyncClient = request.app.state.client
    body = rewrite_request(await request.body())
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST}

    upstream = client.build_request(
        request.method,
        request.url.path,
        params=request.url.query.encode() or None,
        headers=headers,
        content=body,
    )
    response = None
    streaming = False
    try:
        response = await client.send(upstream, stream=True)

        async def close_response():
            # StreamingResponse cancels its AnyIO task group on disconnect.
            # Finish closing the upstream connection even inside that scope.
            with anyio.CancelScope(shield=True):
                await response.aclose()

        owner = current_request.get()
        if owner is not None:
            # Also close if the client vanishes before the body iterator starts.
            owner.async_cleanups.append(close_response)
        out_headers = {
            k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE
        }

        if "text/event-stream" in response.headers.get("content-type", ""):
            is_responses = request.url.path.rstrip("/").endswith("/responses")

            async def body_iter():
                try:
                    source = stream_responses(response) if is_responses else stream_passthrough(response)
                    async for chunk in source:
                        yield chunk
                finally:
                    await close_response()

            streaming = True
            return StreamingResponse(
                body_iter(),
                status_code=response.status_code,
                headers=out_headers,
                media_type=response.headers.get("content-type"),
                background=BackgroundTask(close_response),
            )

        raw = await response.aread()
        if (IMAGE_UPSTREAM and request.method == "GET" and response.status_code == 200
                and request.url.path.rstrip("/") in ("/models", "/v1/models")):
            try:
                images = await request.app.state.image_client.get("/v1/models", headers=headers, timeout=3)
                if images.status_code == 200:
                    payload = json.loads(raw)
                    payload["data"].extend(images.json()["data"])
                    raw = json.dumps(payload).encode()
            except (httpx.RequestError, ValueError, KeyError, TypeError):
                pass  # 图片服务暂不可用不影响 GLM 模型发现。
        return Response(
            content=rewrite_response(raw),
            status_code=response.status_code,
            headers=out_headers,
            media_type=response.headers.get("content-type"),
        )
    finally:
        if response is not None and not streaming:
            await close_response()


app = Starlette(
    lifespan=lifespan,
    routes=[Route("/{path:path}", proxy,
                  methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])],
)
app.add_middleware(
    RequestLifecycleMiddleware,
    cancel_app_on_disconnect=True,
    track_responses=False,
)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=True)
