#!/usr/bin/env python3
"""Fixed-model Qwen image API; MLX work stays on one dedicated thread."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import gc
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

from image_inputs import (MAX_EDIT_BODY, MAX_IMAGES, InvalidEditPrompt,
                          decode_base64_image, validate_images)

logger = logging.getLogger("qwen_image")
PUBLIC_NAME = "qwen-image-2.1"


def error(message, status=400, code="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": code, "code": code}}, status_code=status)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = PUBLIC_NAME
    prompt: str = Field(min_length=1, max_length=8192)
    size: str = "1024x1024"
    steps: int = Field(default=40, ge=1, le=80, strict=True)
    seed: int = Field(default_factory=lambda: secrets.randbits(32), ge=0, lt=2**32, strict=True)
    n: Literal[1] = 1
    guidance: float = Field(default=1.0, ge=1.0, le=10.0)
    negative_prompt: str = Field(default=" ", max_length=8192)
    response_format: Literal["b64_json"] = "b64_json"
    output_format: Literal["png"] = "png"
    user: str | None = None

    @model_validator(mode="after")
    def validate_size(self):
        try:
            w, h = (int(part) for part in self.size.split("x"))
        except (ValueError, TypeError):
            raise ValueError("size must be WIDTHxHEIGHT")
        if any(d < 256 or d > 2048 or d % 16 for d in (w, h)):
            raise ValueError("Each dimension must be a multiple of 16 between 256 and 2048")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        return self


class GenerationCancelled(Exception):
    pass


class EditRequest(GenerationRequest):
    size: str = "512x512"
    images: list[bytes] = Field(min_length=1, max_length=MAX_IMAGES, exclude=True, repr=False)
    reference_size: Literal[512, 768, 1024] = 512

    @model_validator(mode="after")
    def validate_edit_size(self):
        if any(int(d) % 32 for d in self.size.split("x")):
            raise ValueError("Editing dimensions must be multiples of 32")
        return self


class CancellableTransformer:
    """Check cancellation at denoising boundaries without changing MLX kernels."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.cancel = threading.Event()
        self.calls = 0

    def __call__(self, *args, **kwargs):
        if self.cancel.is_set():
            raise GenerationCancelled()
        self.calls += 1
        if self.calls == 1 or self.calls % 10 == 0:
            logger.info("Denoising call %d", self.calls)
        return self.transformer(*args, **kwargs)


class ImageEngine:
    def __init__(self, model_path):
        import mlx.core as mx
        from mlx_vlm.generate.image import load_image_generation_model
        self.mx = mx
        mx.set_memory_limit(int(float(os.environ.get("IMAGE_MEMORY_GB", "80")) * 2**30))
        mx.set_cache_limit(1024**3)
        self.model = load_image_generation_model(model_path)
        self.transformer = CancellableTransformer(self.model.pipeline.transformer)
        self.model.pipeline.transformer = self.transformer
        self.editor = None
        logger.info("Image model loaded from %s", model_path)

    def close(self):
        self.editor = None
        self.model = None
        self.transformer = None
        gc.collect()
        self.mx.clear_cache()
        logger.info("Image model unloaded; active memory %.2f GiB", self.mx.get_active_memory() / 2**30)

    def generate(self, request, cancel):
        from mlx_vlm.generate.image import ImageGenerationRequest
        if cancel.is_set():
            raise GenerationCancelled()
        self.transformer.cancel = cancel
        self.transformer.calls = 0
        width, height = map(int, request.size.split("x"))
        result = None
        try:
            if isinstance(request, EditRequest):
                from qwen_image_edit import QwenImageEditor
                if self.editor is None:
                    self.editor = QwenImageEditor(self.model.pipeline, self.transformer.transformer)
                def check_cancel():
                    if cancel.is_set():
                        raise GenerationCancelled()
                encoded = self.editor.generate(request, check_cancel)
            else:
                result = self.model.generate(ImageGenerationRequest(
                    prompt=request.prompt, width=width, height=height,
                    steps=request.steps, seed=request.seed, guidance=request.guidance,
                    extra={"negative_prompt": request.negative_prompt},
                ))
                encoded = result.to_b64_json()
            if cancel.is_set():
                raise GenerationCancelled()
            return {
                "created": int(time.time()), "model": PUBLIC_NAME,
                "data": [{"b64_json": encoded, "mime_type": "image/png",
                          "width": width, "height": height, "seed": request.seed}],
                "size": request.size, "output_format": "png",
            }
        finally:
            result = None
            gc.collect()
            self.mx.clear_cache()


@dataclass
class Job:
    request: GenerationRequest
    future: asyncio.Future
    cancel: threading.Event = field(default_factory=threading.Event)


class BearerAuth:
    def __init__(self, app, api_key):
        self.app = app
        self.expected = ("Bearer " + api_key).encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            if not hmac.compare_digest(headers.get(b"authorization", b""), self.expected):
                return await error("Invalid API key", 401, "authentication_error")(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(model_path, api_key, *, engine_factory=ImageEngine, queue_size=2, timeout=1800, idle_timeout=300):
    if not api_key:
        raise ValueError("IMAGE_API_KEY must be configured")
    if queue_size < 1 or timeout <= 0 or idle_timeout <= 0:
        raise ValueError("Queue size and timeouts must be positive")
    model_path = str(Path(model_path).expanduser().resolve())

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-image")
        queue = asyncio.Queue(maxsize=queue_size)
        app.state.queue = queue
        app.state.active = None
        app.state.loaded = False
        engine = None

        def run(job):
            nonlocal engine
            if job.cancel.is_set():
                raise GenerationCancelled()
            if engine is None:
                engine = engine_factory(model_path)
                app.state.loaded = True
            return engine.generate(job.request, job.cancel)

        def unload():
            nonlocal engine
            if engine is not None:
                engine.close()
                engine = None
                app.state.loaded = False

        try:
            async def worker():
                while True:
                    try:
                        job = await asyncio.wait_for(queue.get(), timeout=idle_timeout if engine is not None else None)
                    except asyncio.TimeoutError:
                        await loop.run_in_executor(executor, unload)
                        continue
                    started = time.monotonic()
                    try:
                        if job.cancel.is_set():
                            continue
                        app.state.active = job
                        value = await loop.run_in_executor(executor, run, job)
                        if not job.future.done():
                            job.future.set_result(value)
                        logger.info("Generated %s steps=%d seed=%d in %.2fs", job.request.size,
                                    job.request.steps, job.request.seed, time.monotonic() - started)
                    except GenerationCancelled:
                        job.future.cancel()
                        logger.info("Image generation cancelled at denoising boundary")
                    except InvalidEditPrompt as exc:
                        if not job.future.done():
                            job.future.set_result(error(str(exc)))
                    except Exception:
                        logger.exception("Image generation failed")
                        if not job.future.done():
                            job.future.set_result(None)
                    finally:
                        app.state.active = None
                        queue.task_done()
                        # Do not retain uploaded images or the last base64 result
                        # during the idle period (or after unloading weights).
                        job = value = None

            task = asyncio.create_task(worker())
            try:
                yield
            finally:
                if app.state.active is not None:
                    app.state.active.cancel.set()
                while not queue.empty():
                    job = queue.get_nowait()
                    job.cancel.set()
                    job.future.cancel()
                    queue.task_done()
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            await loop.run_in_executor(executor, unload)
            executor.shutdown(wait=True, cancel_futures=True)

    async def health(request):
        return JSONResponse({"status": "ok", "model": PUBLIC_NAME, "model_path": model_path,
                             "loaded": request.app.state.loaded, "idle_timeout": idle_timeout,
                             "request_timeout": timeout,
                             "active": request.app.state.active is not None,
                             "queued": request.app.state.queue.qsize()})

    async def models(request):
        return JSONResponse({"object": "list", "data": [{"id": PUBLIC_NAME, "object": "model",
            "created": 0, "owned_by": "local", "loaded": request.app.state.loaded,
            "capabilities": ["image_generation", "image_editing"],
            "max_reference_images": MAX_IMAGES}]})

    async def generate(request: Request):
        # Text-only requests are small; do not buffer arbitrary uploads.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return error("Request body exceeds 64 KiB", 413)
        try:
            params = GenerationRequest.model_validate_json(body)
        except ValidationError as exc:
            return error(json.dumps(exc.errors(include_input=False, include_context=False)))
        return await submit(request, params)

    async def edit(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_EDIT_BODY:
                return error("Editing request body exceeds 64 MiB", 413)
        # The bounded body lets Starlette enforce multipart field/file limits
        # without an unbounded streamed upload reaching the multipart parser.
        request._body = bytes(body)
        del body
        try:
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type == "multipart/form-data":
                params, images = {}, []
                async with request.form(max_files=MAX_IMAGES, max_fields=20, max_part_size=65536) as form:
                    for key, value in form.multi_items():
                        if isinstance(value, UploadFile):
                            if key not in ("image", "image[]"):
                                raise ValueError("Image files must use image or image[]; masks are not supported")
                            images.append(await value.read())
                        elif key in params:
                            raise ValueError(f"Duplicate parameter: {key}")
                        else:
                            params[key] = value
                for key in ("steps", "seed", "n", "reference_size"):
                    if key in params:
                        params[key] = int(params[key])
            elif content_type == "application/json":
                params = json.loads(request._body)
                if not isinstance(params, dict):
                    raise ValueError("Expected a JSON object")
                if "image" in params and "images" in params:
                    raise ValueError("Use either image or images, not both")
                values = params.pop("images", params.pop("image", []))
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list) or not 1 <= len(values) <= MAX_IMAGES:
                    raise ValueError("Provide 1 to 10 images in upload order")
                images = [decode_base64_image(value) for value in values]
                del values
            else:
                return error("Use multipart/form-data or application/json", 415)
            if "images" in params:
                raise ValueError("Use file uploads for multipart images")
            params = EditRequest.model_validate({**params, "images": images})
            if params.model not in (PUBLIC_NAME, model_path):
                return error("Unknown image model; use qwen-image-2.1", 404, "model_not_found")
            await asyncio.to_thread(validate_images, params.images)
        except ValidationError as exc:
            return error(json.dumps(exc.errors(include_input=False, include_context=False)))
        except (ValueError, TypeError, HTTPException) as exc:
            return error(str(exc))
        # Drop raw multipart/base64 copies before waiting in the shared queue.
        request._body = b""
        return await submit(request, params)

    async def submit(request, params):
        if params.model not in (PUBLIC_NAME, model_path):
            return error("Unknown image model; use qwen-image-2.1", 404, "model_not_found")
        job = Job(params, asyncio.get_running_loop().create_future())
        try:
            request.app.state.queue.put_nowait(job)
        except asyncio.QueueFull:
            return error("Image queue is full; retry later", 429, "rate_limit_error")

        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.2)

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait({job.future, watcher}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if job.future in done and not job.future.cancelled():
                result = job.future.result()
                if isinstance(result, JSONResponse):
                    return result
                return JSONResponse(result) if result else error("Image generation failed; see server log", 500, "server_error")
            if watcher in done:
                return error("Client disconnected", 499, "request_cancelled")
            return error("Image request timed out", 504, "timeout")
        finally:
            job.cancel.set()
            job.future.cancel()
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    async def unsupported(request):
        return error("This image operation is not supported", 501, "not_supported")

    app = Starlette(lifespan=lifespan, routes=[
        Route("/health", health), Route("/v1/models", models), Route("/models", models),
        Route("/v1/images/generations", generate, methods=["POST"]),
        Route("/images/generations", generate, methods=["POST"]),
        Route("/v1/images/edits", edit, methods=["POST"]),
        Route("/images/edits", edit, methods=["POST"]),
        Route("/v1/images/{path:path}", unsupported, methods=["POST"]),
        Route("/images/{path:path}", unsupported, methods=["POST"]),
    ])
    app.add_middleware(BearerAuth, api_key=api_key)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    app = create_app(
        os.environ.get("IMAGE_MODEL", str(Path.home() / "models/Qwen-Image-2.1")),
        os.environ.get("IMAGE_API_KEY", ""),
        queue_size=int(os.environ.get("IMAGE_QUEUE_SIZE", "2")),
        timeout=float(os.environ.get("IMAGE_REQUEST_TIMEOUT", "1800")),
        idle_timeout=float(os.environ.get("IMAGE_IDLE_TIMEOUT", "300")),
    )
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("IMAGE_PORT", "1238")),
                timeout_graceful_shutdown=15)
