"""CPU-only lifecycle tests; no MLX import or model/GPU allocation."""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_lifecycle import (
    RequestOwner, RequestAborted, OwnedIterator, RequestLifecycleMiddleware,
    current_request, active_responses, cancelled_responses,
    bind_generator, bind_cancel_routes,
)


class OwnershipTests(unittest.TestCase):
    def test_cancel_pending_frees_input_and_unblocks_waiter(self):
        owner = RequestOwner()
        request = SimpleNamespace(raw_inputs={"large": object()}, images=[object()], videos=None, audio=None, rqueue=Queue())
        owner.register_pending(request)
        self.assertTrue(owner.cancel("client disconnected"))
        self.assertEqual(request.raw_inputs, {})
        self.assertIsNone(request.images)
        self.assertIsInstance(request.rqueue.get_nowait(), RequestAborted)
        self.assertFalse(owner.claim(request))

    def test_cancel_before_iterator_registration(self):
        owner = RequestOwner()
        owner.cancel("gone")
        iterator = Mock()
        with self.assertRaises(RequestAborted):
            owner.register_iterator(iterator)
        iterator.close.assert_called_once()

    def test_late_iterator_after_handler_exit(self):
        owner = RequestOwner()
        owner.finish()
        iterator = Mock()
        with self.assertRaises(RequestAborted):
            owner.register_iterator(iterator)
        iterator.close.assert_called_once()

    def test_cancel_is_request_scoped(self):
        a, b = RequestOwner(), RequestOwner()
        ia, ib = Mock(_ended=False), Mock(_ended=False)
        a.register_iterator(ia)
        b.register_iterator(ib)
        a.cancel("cancel A")
        ia.close.assert_called_once()
        ib.close.assert_not_called()
        b.check()

    def test_cancelled_eof_is_not_success(self):
        owner = RequestOwner()
        class Iterator:
            def __next__(self):
                owner.cancel("cancel while blocked")
                raise StopIteration
        with self.assertRaises(RequestAborted):
            next(OwnedIterator(Iterator(), owner))

    def test_finished_generation_cannot_be_cancelled(self):
        owner = RequestOwner()
        owner.register_iterator(Mock(_ended=True))
        self.assertFalse(owner.cancel("too late"))
        self.assertFalse(owner.cancelled)

    def test_queued_cancel_not_admitted(self):
        @dataclass
        class Request:
            rqueue: object
            raw_inputs: object
            images: object = None
            videos: object = None
            audio: object = None
            request_id: str = "queued-test"
        class Generator:
            def generate(self):
                return None, Mock(_ended=False)
            def _collect_pending_requests(self):
                return self.queued, False
        bind_generator(Generator, Request)
        owner = RequestOwner()
        token = current_request.set(owner)
        try:
            request = Request(Queue(), {"tokens": [1, 2, 3]})
        finally:
            current_request.reset(token)
        owner.cancel("gone")
        generator = Generator()
        generator.queued = [request]
        self.assertEqual(generator._collect_pending_requests(), ([], False))


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def scope(self):
        return {"type": "http", "method": "POST", "path": "/v1/responses"}

    async def test_proxy_disconnect_cancels_upstream_and_closes(self):
        incoming = asyncio.Queue()
        started = asyncio.Event()
        ended = asyncio.Event()
        close = AsyncMock()
        async def app(scope, receive, send):
            await receive()
            current_request.get().async_cleanups.append(close)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        middleware = RequestLifecycleMiddleware(app, cancel_app_on_disconnect=True)
        task = asyncio.create_task(middleware(self.scope(), incoming.get, AsyncMock()))
        await incoming.put({"type": "http.request", "body": b"{}"})
        await asyncio.wait_for(started.wait(), 1)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)
        self.assertTrue(ended.is_set())
        close.assert_awaited_once()

    async def test_image_edit_disconnect_cancels_upstream(self):
        incoming = asyncio.Queue()
        started = asyncio.Event()
        ended = asyncio.Event()
        async def app(scope, receive, send):
            await receive()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        scope = {**self.scope(), "path": "/v1/images/edits"}
        middleware = RequestLifecycleMiddleware(app, cancel_app_on_disconnect=True)
        task = asyncio.create_task(middleware(scope, incoming.get, AsyncMock()))
        await incoming.put({"type": "http.request", "body": b"image upload"})
        await asyncio.wait_for(started.wait(), 1)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)
        self.assertTrue(ended.is_set())

    async def test_backend_nonstream_disconnect_closes_iterator(self):
        incoming = asyncio.Queue()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        iterator = Mock(_ended=False, close=lambda: cancelled.set())
        async def app(scope, receive, send):
            await receive()
            owner = current_request.get()
            owner.register_iterator(iterator)
            started.set()
            await cancelled.wait()
            with self.assertRaises(RequestAborted):
                owner.check()
        task = asyncio.create_task(RequestLifecycleMiddleware(app)(self.scope(), incoming.get, AsyncMock()))
        await incoming.put({"type": "http.request", "body": b"{}"})
        await asyncio.wait_for(started.wait(), 1)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)

    async def test_split_created_event_is_registered_and_removed(self):
        async def receive():
            await asyncio.Event().wait()
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            frame = b'data: {"type":"response.created","response":{"id":"resp-test"}}\n\n'
            for part in (frame[:19], frame[19:]):
                await send({"type": "http.response.body", "body": part, "more_body": True})
            self.assertIs(active_responses["resp-test"], current_request.get())
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        await RequestLifecycleMiddleware(app)(self.scope(), receive, AsyncMock())
        self.assertNotIn("resp-test", active_responses)

    async def test_two_concurrent_contexts_are_isolated(self):
        owners = []
        async def receive():
            await asyncio.Event().wait()
        async def app(scope, receive, send):
            owner = current_request.get()
            owners.append(owner)
            await asyncio.sleep(0)
            self.assertIs(current_request.get(), owner)
        middleware = RequestLifecycleMiddleware(app)
        await asyncio.gather(*(middleware(self.scope(), receive, AsyncMock()) for _ in range(2)))
        self.assertIsNot(owners[0], owners[1])


class CancelEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_preserves_auth_and_stops_only_target(self):
        import httpx
        from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException
        app = FastAPI()
        async def auth(authorization: str = Header(default="")):
            if authorization != "Bearer test":
                raise HTTPException(401)
        router = APIRouter(dependencies=[Depends(auth)])
        @router.post("/v1/responses/{response_id}/cancel")
        async def cancel(response_id: str):
            raise HTTPException(404)
        @router.get("/v1/responses/{response_id}")
        async def retrieve(response_id: str):
            raise HTTPException(404)
        @router.delete("/v1/responses/{response_id}")
        async def delete(response_id: str):
            raise HTTPException(404)
        app.include_router(router)
        bind_cancel_routes(app)
        owner = RequestOwner()
        iterator = Mock(_ended=False)
        owner.register_iterator(iterator)
        owner.response = {"id": "resp-cancel", "object": "response", "status": "in_progress", "store": True, "output": []}
        active_responses["resp-cancel"] = owner
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                url = "/v1/responses/resp-cancel"
                self.assertEqual((await client.post(url + "/cancel")).status_code, 401)
                iterator.close.assert_not_called()
                client.headers["Authorization"] = "Bearer test"
                response = await client.post(url + "/cancel")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "cancelled")
                iterator.close.assert_called_once()
                owner.finish()
                self.assertEqual((await client.post(url + "/cancel")).json()["status"], "cancelled")
                self.assertEqual((await client.get(url)).json()["status"], "cancelled")
                self.assertTrue((await client.delete(url)).json()["deleted"])
                self.assertEqual((await client.get(url)).status_code, 404)
        finally:
            owner.finish()
            cancelled_responses.pop("resp-cancel", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
