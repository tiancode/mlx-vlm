"""SSE chunk boundaries, model field mapping and reasoning event ordering."""

import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PROXY_UPSTREAM_MODEL", "/test/model")
import model_proxy as proxy


class Chunks:
    def __init__(self, data, size=1):
        self.data, self.size = data, size

    async def aiter_raw(self):
        for start in range(0, len(self.data), self.size):
            yield self.data[start:start + self.size]


async def collect(source):
    return b"".join([part async for part in source])


class ProxyStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_preserves_body_and_non_json_across_chunks(self):
        event = {"model": proxy.UPSTREAM_MODEL, "choices": [{
            "delta": {"content": f"路径 {proxy.UPSTREAM_MODEL}"}}]}
        data = (b': keepalive\r\n' + b'data: ' + json.dumps(event).encode()
                + b'\r\n\r\ndata: [DONE]\r\n\r\n')
        for size in (1, 7, len(data)):
            out = await collect(proxy.stream_passthrough(Chunks(data, size)))
            self.assertTrue(out.startswith(b': keepalive\r\n'))
            self.assertTrue(out.endswith(b'data: [DONE]\r\n\r\n'))
            mapped = json.loads(out.split(b'data: ', 1)[1].split(b'\r\n')[0])
            self.assertEqual(mapped["model"], proxy.PUBLIC_NAME)
            self.assertEqual(mapped["choices"], event["choices"])

    async def test_responses_reasoning_order_with_crlf_and_split_utf8(self):
        events = [
            {"type": "response.created", "response": {"model": proxy.UPSTREAM_MODEL}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"id": "m", "type": "message"}},
            {"type": "response.reasoning_text.delta", "item_id": "r", "output_index": 0,
             "delta": f"检查 {proxy.UPSTREAM_MODEL}"},
            {"type": "response.output_text.delta", "item_id": "m", "output_index": 0,
             "delta": "四"},
            {"type": "response.reasoning_text.done", "item_id": "r", "output_index": 0,
             "text": f"检查 {proxy.UPSTREAM_MODEL}"},
            {"type": "response.output_item.done", "output_index": 0,
             "item": {"id": "m", "type": "message"}},
            {"type": "response.completed", "response": {"model": proxy.UPSTREAM_MODEL, "output": [
                {"id": "r", "type": "reasoning", "summary": [
                    {"type": "summary_text", "text": f"检查 {proxy.UPSTREAM_MODEL}"}]},
                {"id": "m", "type": "message"}]}},
        ]
        data = b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\r\n\r\n'
                        for e in events)
        out = await collect(proxy.stream_responses(Chunks(data)))
        parsed = [json.loads(line[5:]) for line in out.splitlines() if line.startswith(b'data:')]
        declared = {}
        for event in parsed:
            if event["type"] == "response.output_item.added":
                declared[event["output_index"]] = event["item"]["id"]
            if "item_id" in event:
                self.assertEqual(declared[event["output_index"]], event["item_id"])
        self.assertEqual(declared, {0: "r", 1: "m"})
        self.assertEqual(sum(e["type"] == "response.reasoning_text.done" for e in parsed), 1)
        terminal = parsed[-1]["response"]
        self.assertEqual(terminal["model"], proxy.PUBLIC_NAME)
        self.assertEqual(terminal["output"][0]["content"][0]["text"], f"检查 {proxy.UPSTREAM_MODEL}")

    def test_json_only_rewrites_model_fields(self):
        payload = {"model": proxy.UPSTREAM_MODEL, "data": [{"id": proxy.UPSTREAM_MODEL}],
                   "content": proxy.UPSTREAM_MODEL}
        result = json.loads(proxy.rewrite_response(json.dumps(payload).encode()))
        self.assertEqual(result["model"], proxy.PUBLIC_NAME)
        self.assertEqual(result["data"][0]["id"], proxy.PUBLIC_NAME)
        self.assertEqual(result["content"], proxy.UPSTREAM_MODEL)


if __name__ == "__main__":
    unittest.main()
