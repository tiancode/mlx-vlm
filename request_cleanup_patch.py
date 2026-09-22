"""Release completed/cancelled speculative request state on the GPU thread."""

from functools import wraps
import logging

import mlx.core as mx


class OwnedPromptInputs(dict):
    """Server-owned embedding arguments, safe to clear when their request ends."""


def release_prompt_inputs(batch, uids=None):
    resources = vars(batch).get("_local_prompt_inputs", {})
    for uid in list(resources) if uids is None else uids:
        inputs = resources.pop(uid, None)
        if inputs is not None:
            # Clear the shared mapping, including references held by the idle
            # server loop's gen_kwargs/info locals. Caller-owned dicts are never
            # registered here.
            inputs.clear()


def bind_prompt_inputs(generator_class, batch_class):
    original_embed = generator_class._gpu_embed
    original_insert = batch_class.insert

    @wraps(original_embed)
    def embed(self, *args, **kwargs):
        ids, inputs = original_embed(self, *args, **kwargs)
        return ids, OwnedPromptInputs(inputs)

    @wraps(original_insert)
    def insert(self, *args, **kwargs):
        inputs = kwargs.get("prompt_kwargs", args[2] if len(args) > 2 else None)
        uids = original_insert(self, *args, **kwargs)
        resources = vars(self).setdefault("_local_prompt_inputs", {})
        for uid, values in zip(uids, inputs or []):
            if isinstance(values, OwnedPromptInputs):
                resources[uid] = values
        return uids

    generator_class._gpu_embed = embed
    batch_class.insert = insert


def release_speculative_state(batch):
    if len(batch):
        return
    rounds = batch._rounds_iter
    batch._rounds_iter = None
    try:
        if rounds is not None:
            rounds.close()
    finally:
        # Close suspended speculative rounds before dropping their cache. They
        # may still own a transaction that needs its normal generator cleanup.
        batch.prompt_cache = []
        batch.hidden = None
        batch.shared_kv_states = None
        batch.prompt_tokens = None
        batch.first_tokens = None


def release_idle_batch(batch):
    if not batch.has_work:
        release_prompt_inputs(batch)
        # Restore the wired limit and synchronize through upstream's close().
        # The server constructs a new BatchGenerator for the next admission.
        batch.close()
        mx.clear_cache()
        logging.getLogger(__name__).info(
            "GPU idle: active=%.2f GiB allocator_cache=%.2f GiB "
            "(model weights and APC remain resident)",
            mx.get_active_memory() / (1 << 30),
            mx.get_cache_memory() / (1 << 30),
        )


def install():
    from mlx_vlm.generate.ar import BatchGenerator, SpeculativeGenerationBatch
    from mlx_vlm.server.generation import ResponseGenerator

    if getattr(SpeculativeGenerationBatch, "_local_request_cleanup", False):
        return
    original_refresh = SpeculativeGenerationBatch._refresh_uids

    @wraps(original_refresh)
    def refresh(self):
        original_refresh(self)
        release_speculative_state(self)

    SpeculativeGenerationBatch._refresh_uids = refresh
    bind_prompt_inputs(ResponseGenerator, BatchGenerator)

    # Both normal completion and cancellation can leave an idle generator in
    # the server loop. Run cleanup after its GPU operations, on the same thread.
    for method_name in ("next", "remove"):
        original = getattr(BatchGenerator, method_name)

        def wrap(operation, kind):
            @wraps(operation)
            def call(self, *args, **kwargs):
                result = operation(self, *args, **kwargs)
                if kind == "next":
                    release_prompt_inputs(self, [r.uid for r in result[1] if r.finish_reason is not None])
                elif result:
                    release_prompt_inputs(self, [args[0] if args else kwargs["uid"]])
                release_idle_batch(self)
                return result
            return call

        setattr(BatchGenerator, method_name, wrap(original, method_name))

    SpeculativeGenerationBatch._local_request_cleanup = True
    logging.getLogger(__name__).warning(
        "Request cleanup patch: completed/cancelled MTP state and idle wired buffers released"
    )
