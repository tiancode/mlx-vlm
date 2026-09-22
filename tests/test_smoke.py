"""Smoke checks must fail for HTTP errors, wrong answers and incomplete streams."""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import run_checks


class SmokeTests(unittest.TestCase):
    def run_mock(self, fault=None):
        calls = []

        def handle(request):
            path = request.url.path
            calls.append(path)
            payload = json.loads(request.content) if request.content else {}
            if fault == "http":
                return httpx.Response(500, json={"error": "backend failed"})
            if path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            self.assertEqual(payload["model"], "test-model")
            if path == "/v1/chat/completions":
                is_image = isinstance(payload["messages"][0]["content"], list)
                answer = ("red" if fault == "image" else "蓝色") if is_image else "PONG"
                return httpx.Response(200, json={"choices": [{
                    "finish_reason": "stop", "message": {"content": answer}}]})
            if path == "/v1/responses":
                events = [
                    {"type": "response.output_item.added", "output_index": 0,
                     "item": {"id": "m", "type": "message"}},
                    {"type": "response.output_text.delta", "output_index": 0,
                     "item_id": "m", "delta": "4"},
                    {"type": "response.output_item.done", "output_index": 0,
                     "item": {"id": "m", "type": "message"}},
                    {"type": "response.completed", "response": {"status": "completed",
                     "output": [{"id": "m", "type": "message"}]}},
                ]
                if fault == "incomplete":
                    events.pop()
                if fault == "index":
                    events[1]["output_index"] = 1
                return httpx.Response(200, text=''.join('data: ' + json.dumps(e) + '\n\n' for e in events),
                                      headers={"content-type": "text/event-stream"})
            if path == "/v1/messages":
                return httpx.Response(200, json={"stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "PONG"}]})
            self.fail(f"unexpected path: {path}")

        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client, \
             contextlib.redirect_stdout(io.StringIO()):
            run_checks(client, "test-model")
        return calls

    def test_all_six_checks(self):
        self.assertEqual(len(self.run_mock()), 6)

    def test_http_failure(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.run_mock("http")

    def test_wrong_image_answer(self):
        with self.assertRaisesRegex(ValueError, "图片颜色识别错误"):
            self.run_mock("image")

    def test_incomplete_stream(self):
        with self.assertRaisesRegex(ValueError, "缺少成功终态"):
            self.run_mock("incomplete")

    def test_wrong_event_index(self):
        with self.assertRaisesRegex(ValueError, "索引不匹配"):
            self.run_mock("index")


if __name__ == "__main__":
    unittest.main()
