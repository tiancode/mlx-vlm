"""HTTP request ownership for local inference: abort on disconnect or cancel.

The HTTP layer only signals cancellation. All live GPU cache mutation remains
on mlx-vlm's generation thread, at its normal scheduling boundaries.
"""

import asyncio
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import json
import logging
from threading import RLock

current_request = ContextVar("inference_request", default=None)
active_responses = {}
cancelled_responses = OrderedDict()
GENERATION_PATHS = {"/responses", "/chat/completions", "/messages", "/images/generations", "/images/edits"}
logger = logging.getLogger(__name__)


class RequestAborted(RuntimeError):
    pass


class RequestOwner:
    def __init__(self):
        self.lock = RLock()
        self.cancelled = False
        self.finished = False
        self.reason = None
        self.iterator = None
        self.pending = None
        self.response = None
        self.async_cleanups = []

    def check(self):
        with self.lock:
            if self.cancelled or self.finished:
                raise RequestAborted(self.reason or "Request is no longer active")

    def register_iterator(self, iterator):
        with self.lock:
            if self.cancelled or self.finished:
                iterator.close()
                raise RequestAborted(self.reason or "Request is no longer active")
            self.iterator = iterator

    def register_pending(self, request):
        with self.lock:
            self.pending = request
            if self.cancelled or self.finished:
                self._drop_pending()

    def _drop_pending(self):
        if self.pending is None:
            return
        request, self.pending = self.pending, None
        # The request has not been admitted to the GPU thread. Leave a small
        # tombstone in its FIFO; release potentially large preprocessed input.
        request.raw_inputs = {}
        request.images = request.videos = request.audio = None
        request.rqueue.put(RequestAborted(self.reason or "Request aborted before admission"))

    def claim(self, request):
        with self.lock:
            if self.cancelled or self.finished:
                return False
            if self.pending is request:
                self.pending = None
            return True

    def cancel(self, reason):
        with self.lock:
            if self.finished:
                return False
            if self.iterator is not None and getattr(self.iterator, "_ended", False):
                return False
            self.cancelled = True
            self.reason = reason
            self._drop_pending()
            if self.iterator is not None:
                self.iterator.close()
            return True

    def finish(self):
        with self.lock:
            self.finished = True
            self._drop_pending()
            if self.iterator is not None:
                self.iterator.close()
                self.iterator = None
        if self.response is not None:
            response_id = self.response.get("id")
            if active_responses.get(response_id) is self:
                active_responses.pop(response_id, None)


class OwnedIterator:
    def __init__(self, iterator, owner):
        self.iterator = iterator
        self.owner = owner

    def __iter__(self):
        return self

    def __next__(self):
        self.owner.check()
        try:
            token = next(self.iterator)
        except StopIteration:
            # A scheduler cancellation emits EOF. Do not let an aborted request
            # turn that EOF into a successful response.completed event.
            if self.owner.cancelled:
                raise RequestAborted(self.owner.reason) from None
            raise
        self.owner.check()
        return token

    def close(self):
        self.iterator.close()


class RequestLifecycleMiddleware:
    def __init__(self, app, *, cancel_app_on_disconnect=False, track_responses=True):
        self.app = app
        self.cancel_app_on_disconnect = cancel_app_on_disconnect
        self.track_responses = track_responses

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").removeprefix("/v1").rstrip("/")
        if scope["type"] != "http" or scope.get("method") != "POST" or path not in GENERATION_PATHS:
            return await self.app(scope, receive, send)
        owner = RequestOwner()
        context_token = current_request.set(owner)
        inbox = asyncio.Queue(maxsize=1)
        application_task = asyncio.current_task()
        response_is_sse = False
        response_complete = False
        prefix = b""

        async def pump():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    if not response_complete:
                        owner.cancel("Client disconnected")
                        if self.cancel_app_on_disconnect:
                            application_task.cancel()
                    await inbox.put(message)
                    return
                await inbox.put(message)

        async def forward(message):
            nonlocal response_is_sse, response_complete, prefix
            if message["type"] == "http.response.start":
                response_is_sse = any(
                    key.lower() == b"content-type" and b"text/event-stream" in value
                    for key, value in message.get("headers", [])
                )
            elif message["type"] == "http.response.body":
                if self.track_responses and response_is_sse and path == "/responses" and owner.response is None:
                    prefix += message.get("body", b"")
                    while b"\n\n" in prefix:
                        frame, prefix = prefix.split(b"\n\n", 1)
                        for line in frame.splitlines():
                            if not line.startswith(b"data: "):
                                continue
                            try:
                                event = json.loads(line[6:])
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if event.get("type") == "response.created":
                                owner.response = event["response"]
                                active_responses[owner.response["id"]] = owner
                                prefix = b""
                                break
                        if owner.response is not None:
                            break
                if not message.get("more_body", False):
                    response_complete = True
            await send(message)

        watcher = asyncio.create_task(pump())
        try:
            await self.app(scope, inbox.get, forward)
        except asyncio.CancelledError:
            disconnected = owner.cancelled
            owner.cancel("Client disconnected")
            if not self.cancel_app_on_disconnect or not disconnected:
                raise
        finally:
            owner.finish()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await asyncio.gather(
                *(close() for close in owner.async_cleanups), return_exceptions=True
            )
            owner.async_cleanups.clear()
            current_request.reset(context_token)


def bind_generator(generator_class, request_class):
    original_generate = generator_class.generate
    original_init = request_class.__init__
    original_collect = generator_class._collect_pending_requests

    @wraps(original_init)
    def init_request(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._http_owner = current_request.get()
        if self._http_owner is not None:
            self._http_owner.register_pending(self)

    @wraps(original_collect)
    def collect(self, *args, **kwargs):
        requests, stop = original_collect(self, *args, **kwargs)
        admitted = []
        for request in requests:
            owner = getattr(request, "_http_owner", None)
            if owner is None or owner.claim(request):
                admitted.append(request)
            else:
                logger.info("Dropped cancelled queued request=%s", request.request_id)
        return admitted, stop

    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        owner = current_request.get()
        if owner is not None:
            owner.check()
        ctx, iterator = original_generate(self, *args, **kwargs)
        if owner is not None:
            owner.register_iterator(iterator)
            iterator = OwnedIterator(iterator, owner)
        return ctx, iterator

    request_class.__init__ = init_request
    generator_class.generate = generate
    generator_class._collect_pending_requests = collect


def bind_cancel_routes(app):
    from fastapi import HTTPException
    for route in app.routes:
        # Recent FastAPI versions retain included routers instead of flattening
        # them into app.routes. Patch their source routes before startup builds
        # effective dependency graphs, preserving router-level authentication.
        included = getattr(route, "original_router", None)
        if included is not None:
            bind_cancel_routes(included)
            continue
        path = getattr(route, "path", "").removeprefix("/v1")
        methods = getattr(route, "methods", set())
        if path == "/responses/{response_id}" and methods.intersection({"GET", "DELETE"}):
            original = route.dependant.call

            def make_terminal(fallback, delete):
                async def terminal(response_id: str):
                    if response_id not in cancelled_responses:
                        return await fallback(response_id)
                    if delete:
                        cancelled_responses.pop(response_id, None)
                        return {"id": response_id, "object": "response.deleted", "deleted": True}
                    return cancelled_responses[response_id]
                return terminal

            route.endpoint = route.dependant.call = make_terminal(original, "DELETE" in methods)
            continue
        if path != "/responses/{response_id}/cancel" or "POST" not in methods:
            continue
        original = route.dependant.call

        def make_cancel(fallback):
            async def cancel(response_id: str):
                owner = active_responses.get(response_id)
                if owner is None:
                    if response_id in cancelled_responses:
                        return cancelled_responses[response_id]
                    return await fallback(response_id)
                if not owner.cancel("Explicit response cancellation"):
                    raise HTTPException(409, "Response has already finished generating")
                response = dict(owner.response, status="cancelled")
                if response.get("store", True):
                    # Bounded cancellation acknowledgements, without prompt text.
                    cancelled_responses[response_id] = {
                        key: value for key, value in response.items()
                        if key in {"id", "object", "created_at", "status", "model", "output", "usage"}
                    }
                    while len(cancelled_responses) > 256:
                        cancelled_responses.popitem(last=False)
                return response
            return cancel

        route.endpoint = route.dependant.call = make_cancel(original)


def install_backend():
    from mlx_vlm.server.app import app
    from mlx_vlm.server.generation import ResponseGenerator, QueuedGenerationRequest

    if getattr(app.state, "local_request_lifecycle", False):
        return
    bind_generator(ResponseGenerator, QueuedGenerationRequest)
    app.add_middleware(RequestLifecycleMiddleware)
    bind_cancel_routes(app)
    app.state.local_request_lifecycle = True
    logger.warning("Request lifecycle: disconnect and Responses cancel abort actual generation")
