"""No GPU: exercise lazy loading, cancellation, queue bounds and proxy isolation."""
import asyncio
import base64
from contextlib import asynccontextmanager
import json
from io import BytesIO
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_server import create_app, GenerationCancelled
os.environ.setdefault("PROXY_UPSTREAM_MODEL", "/models/glm")
import model_proxy


class FakeEngine:
    loads = 0
    closes = 0
    running = 0
    peak = 0
    cancelled = threading.Event()
    requests = []

    def __init__(self, path):
        type(self).loads += 1

    def close(self):
        type(self).closes += 1

    def generate(self, request, cancel):
        cls = type(self)
        cls.requests.append(request)
        cls.running += 1
        cls.peak = max(cls.peak, cls.running)
        try:
            until = time.monotonic() + (0.5 if request.prompt == "slow" else 0.01)
            while time.monotonic() < until:
                if cancel.wait(0.005):
                    cls.cancelled.set()
                    raise GenerationCancelled()
            return {"data": [{"b64_json": "png", "seed": request.seed}]}
        finally:
            cls.running -= 1


@asynccontextmanager
async def image_client(**kwargs):
    app = create_app("/tmp/qwen-test", "test", engine_factory=FakeEngine, **kwargs)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://image",
                                     headers={"Authorization": "Bearer test"}) as client:
            yield client, app


class ImageServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeEngine.loads = FakeEngine.closes = FakeEngine.running = FakeEngine.peak = 0
        FakeEngine.cancelled = threading.Event()
        FakeEngine.requests = []

    async def test_lazy_load_reuse_idle_unload_and_reload(self):
        async with image_client(idle_timeout=0.05) as (client, app):
            self.assertFalse((await client.get("/health")).json()["loaded"])
            self.assertEqual(FakeEngine.loads, 0)
            for _ in range(2):
                self.assertEqual((await client.post("/v1/images/generations", json={"prompt": "fox"})).status_code, 200)
            self.assertEqual(FakeEngine.loads, 1)
            await asyncio.sleep(0.15)
            self.assertFalse((await client.get("/health")).json()["loaded"])
            self.assertEqual(FakeEngine.closes, 1)
            await client.post("/v1/images/generations", json={"prompt": "fox"})
            self.assertEqual(FakeEngine.loads, 2)

    async def test_auth_and_invalid_requests_never_load_weights(self):
        async with image_client() as (client, app):
            self.assertEqual((await client.get("/health", headers={"Authorization": "wrong"})).status_code, 401)
            self.assertEqual((await client.post("/v1/images/edits", json={})).status_code, 400)
            cases = [({"model": "arbitrary/repo", "prompt": "fox"}, 404),
                     ({"prompt": "fox", "size": "99999x99999"}, 400),
                     ({"prompt": "fox", "response_format": "path", "output_path": "/tmp/file"}, 400),
                     ({"prompt": "fox", "background": "transparent"}, 400),
                     ({"prompt": " "}, 400)]
            for body, status in cases:
                with self.subTest(body=body):
                    self.assertEqual((await client.post("/v1/images/generations", json=body)).status_code, status)
            self.assertEqual(FakeEngine.loads, 0)

    async def test_queue_is_bounded_health_responsive_and_cancel_releases_worker(self):
        async with image_client(queue_size=1) as (client, app):
            first = asyncio.create_task(client.post("/v1/images/generations", json={"prompt": "slow"}))
            while app.state.active is None:
                await asyncio.sleep(0.005)
            second = asyncio.create_task(client.post("/v1/images/generations", json={"prompt": "fox"}))
            while app.state.queue.empty():
                await asyncio.sleep(0.005)
            self.assertTrue((await asyncio.wait_for(client.get("/health"), 0.2)).json()["active"])
            self.assertEqual((await client.post("/v1/images/generations", json={"prompt": "fox"})).status_code, 429)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual((await asyncio.wait_for(second, 2)).status_code, 200)
            self.assertTrue(FakeEngine.cancelled.is_set())
            self.assertEqual(FakeEngine.peak, 1)

    @staticmethod
    def png(color):
        buf = BytesIO()
        Image.new("RGB", (64, 48), color).save(buf, format="PNG")
        return buf.getvalue()

    async def test_edit_single_and_multi_uploads_keep_order_and_share_lazy_engine(self):
        red, blue = self.png("red"), self.png("blue")
        async with image_client(idle_timeout=0.05) as (client, app):
            response = await client.post("/v1/images/edits", data={"prompt": "edit", "steps": "20", "seed": "42"},
                files={"image": ("a.png", red, "image/png")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(FakeEngine.requests[-1].images, [red])
            self.assertEqual(FakeEngine.requests[-1].steps, 20)
            self.assertEqual(FakeEngine.requests[-1].size, "512x512")
            response = await client.post("/v1/images/edits", data={"prompt": "combine"}, files=[
                ("image[]", ("red.png", red, "image/png")), ("image[]", ("blue.png", blue, "image/png"))])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(FakeEngine.requests[-1].images, [red, blue])
            await client.post("/v1/images/generations", json={"prompt": "fox"})
            self.assertEqual(FakeEngine.loads, 1)
            await asyncio.sleep(0.15)
            self.assertFalse((await client.get("/health")).json()["loaded"])

    async def test_edit_json_data_urls_raw_base64_and_max_count(self):
        encoded = base64.b64encode(self.png("red")).decode()
        async with image_client() as (client, app):
            for payload in ({"image": "data:image/png;base64," + encoded}, {"images": [encoded] * 10}):
                response = await client.post("/images/edits", json={"prompt": "edit", **payload})
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(FakeEngine.requests[-1].images), 10)

    async def test_edit_validation_does_not_load_weights(self):
        encoded = base64.b64encode(self.png("red")).decode()
        cases = [{"images": []}, {"images": [encoded] * 11}, {"image": "/etc/passwd"},
                 {"image": "https://example.com/image.png"}, {"image": "ZmFrZQ=="},
                 {"image": encoded, "images": [encoded]}, {"image": encoded, "size": "272x272"},
                 {"image": encoded, "reference_size": 2048}, {"image": encoded, "mask": encoded},
                 {"image": encoded, "model": "other"}]
        async with image_client() as (client, app):
            for payload in cases:
                with self.subTest(keys=list(payload)):
                    response = await client.post("/v1/images/edits", json={"prompt": "edit", **payload})
                    self.assertEqual(response.status_code, 404 if "model" in payload else 400, response.text)
            response = await client.post("/v1/images/edits", content=b"abc", headers={"Content-Type": "text/plain"})
            self.assertEqual(response.status_code, 415)
            response = await client.post("/v1/images/edits", data={"prompt": "edit"},
                files={"mask": ("a.png", self.png("red"), "image/png")})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(FakeEngine.loads, 0)

    async def test_edit_disconnect_releases_worker_for_generation(self):
        async with image_client() as (client, app):
            first = asyncio.create_task(client.post("/v1/images/edits", data={"prompt": "slow"},
                files={"image": ("a.png", self.png("red"), "image/png")}))
            while app.state.active is None:
                await asyncio.sleep(0.005)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual((await client.post("/v1/images/generations", json={"prompt": "fox"})).status_code, 200)
            self.assertTrue(FakeEngine.cancelled.is_set())
            self.assertEqual(FakeEngine.peak, 1)

    async def test_upload_limits_and_damaged_images_are_rejected_before_loading(self):
        png = self.png("red")
        async with image_client() as (client, app):
            with patch("image_server.MAX_EDIT_BODY", 100):
                response = await client.post("/v1/images/edits", content=b"x" * 101,
                    headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 413)
            response = await client.post("/v1/images/edits", data={"prompt": "edit"}, files=[
                ("image[]", (f"{i}.png", png, "image/png")) for i in range(11)])
            self.assertEqual(response.status_code, 400)
            for uploaded in (png[:40], b"invalid image"):
                response = await client.post("/v1/images/edits", data={"prompt": "edit"},
                    files={"image": ("a.png", uploaded, "image/png")})
                self.assertEqual(response.status_code, 400)
            with patch("image_inputs.MAX_IMAGE_PIXELS", 10):
                response = await client.post("/v1/images/edits", data={"prompt": "edit"},
                    files={"image": ("a.png", png, "image/png")})
                self.assertEqual(response.status_code, 400)
            self.assertEqual(FakeEngine.loads, 0)

    async def test_deadline_cancels_gpu_work_and_service_recovers(self):
        async with image_client(timeout=0.08) as (client, app):
            response = await client.post("/v1/images/generations", json={"prompt": "slow"})
            self.assertEqual(response.status_code, 504)
            response = await client.post("/v1/images/generations", json={"prompt": "fox"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(FakeEngine.cancelled.is_set())
            self.assertEqual(FakeEngine.peak, 1)


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def client(self, image_failure=False):
        calls = []
        async def handler(request):
            body = (request.content if request.headers.get("content-type", "").startswith("multipart/")
                    else json.loads(request.content) if request.content else None)
            calls.append((request.url.host, request.url.path, body))
            if request.headers.get("authorization") != "Bearer test":
                return httpx.Response(401, json={"error": "unauthorized"})
            if image_failure and request.url.host == "image":
                raise httpx.ConnectError("offline", request=request)
            if request.url.path == "/v1/models":
                name = model_proxy.UPSTREAM_MODEL if request.url.host == "glm" else "qwen-image-2.1"
                return httpx.Response(200, json={"object": "list", "data": [{"id": name}]})
            if isinstance(body, bytes):
                return httpx.Response(200, content=body, headers={"content-type": request.headers["content-type"]})
            return httpx.Response(200, json=body)
        async with model_proxy.app.router.lifespan_context(model_proxy.app):
            async with httpx.AsyncClient(base_url="http://glm", transport=httpx.MockTransport(handler)) as glm:
                async with httpx.AsyncClient(base_url="http://image", transport=httpx.MockTransport(handler)) as image:
                    model_proxy.app.state.client = glm
                    model_proxy.app.state.image_client = image
                    with patch.object(model_proxy, "IMAGE_UPSTREAM", "http://image"):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(model_proxy.app),
                            base_url="http://proxy", headers={"Authorization": "Bearer test"}) as client:
                            yield client, calls

    async def test_image_routing_and_glm_rewrite_are_isolated(self):
        async with self.client() as (client, calls):
            response = await client.post("/v1/images/generations", json={"model": "qwen-image-2.1", "prompt": "fox"})
            self.assertEqual(response.json()["model"], "qwen-image-2.1")
            self.assertEqual(calls[-1][0], "image")
            await client.post("/v1/chat/completions", json={"model": "old-client-name", "messages": []})
            self.assertEqual(calls[-1][0], "glm")
            self.assertEqual(calls[-1][2]["model"], model_proxy.UPSTREAM_MODEL)
            response = await client.get("/v1/models")
            self.assertEqual([m["id"] for m in response.json()["data"]], [model_proxy.PUBLIC_NAME, "qwen-image-2.1"])

    async def test_auth_failure_does_not_merge_models(self):
        async with self.client() as (client, calls):
            response = await client.get("/v1/models", headers={"Authorization": "wrong"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(len(calls), 1)
            response = await client.post("/v1/images/generations", json={"prompt": "fox"}, headers={"Authorization": "wrong"})
            self.assertEqual(response.status_code, 401)

    async def test_image_outage_leaves_chat_and_discovery_available(self):
        async with self.client(image_failure=True) as (client, calls):
            self.assertEqual((await client.post("/v1/images/generations", json={"prompt": "fox"})).status_code, 503)
            self.assertEqual((await client.post("/v1/chat/completions", json={"model": "glm"})).status_code, 200)
            response = await client.get("/v1/models")
            self.assertEqual(len(response.json()["data"]), 1)

    async def test_edit_multipart_larger_than_text_limit_is_forwarded_unchanged(self):
        body = b'--boundary\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\n\r\n' + b'x' * 100000 + b'\r\n--boundary--\r\n'
        async with self.client() as (client, calls):
            response = await client.post("/v1/images/edits", content=body,
                headers={"Content-Type": "multipart/form-data; boundary=boundary"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, body)
            self.assertEqual(calls[-1], ("image", "/v1/images/edits", body))
            response = await client.post("/v1/images/generations", content=body)
            self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main(verbosity=2)
