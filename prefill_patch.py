"""Avoid materializing unused GLM prefill logits; mlx-vlm 0.7.1 only.

PromptProcessingBatch.prompt_step normally evaluates caches, not logits. The
finite-logit guard would otherwise force an unused [B, chunk, vocab] projection.
Only cache-building chunks with no completed right-padded row may skip it.
Final prefill, finished-row readout, and all decode/verifier calls stay intact.
"""

from contextvars import ContextVar
from functools import wraps
import logging
import os

import mlx.core as mx


_ENABLED = os.environ.get("GLM_PREFILL_SKIP_UNUSED_LOGITS", "1").lower() not in (
    "0", "false", "no",
)
_active_prefill = ContextVar("glm_cache_only_prefill", default=None)


def _unused_logits(model, kwargs):
    batch = _active_prefill.get()
    n = kwargs.get("n_to_process")
    if (
        not _ENABLED
        or batch is None
        or batch.model is not model
        or not isinstance(n, int)
        or n <= 0
        or kwargs.get("skip_logits")
        or kwargs.get("hidden_sink") is not None
    ):
        return False
    # In mixed warm/cold batches prompt_step saves the last logit of each row
    # that ends at this chunk boundary. Preserve that entire forward unchanged.
    if batch._right_pad_per_row is not None:
        end = batch._processed_prompt_columns + n
        if end in batch._suffix_lens:
            return False
    return True


def install():
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.glm5_next.language import LanguageModel

    if getattr(LanguageModel, "_local_prefill_logits", False):
        return
    original_step = PromptProcessingBatch.prompt_step
    original_call = LanguageModel.__call__

    @wraps(original_step)
    def prompt_step(self):
        token = _active_prefill.set(self)
        try:
            return original_step(self)
        finally:
            _active_prefill.reset(token)

    @wraps(original_call)
    def forward(self, *args, **kwargs):
        if not _unused_logits(self, kwargs):
            return original_call(self, *args, **kwargs)
        keep_hidden = kwargs.get("return_hidden", False)
        output = original_call(
            self, *args, **{**kwargs, "skip_logits": True, "return_hidden": True}
        )
        # Check every position's normalized hidden state, rather than silently
        # dropping the numerical guard along with the unused vocabulary head.
        hidden = output.hidden_states[-1]
        if not bool(mx.all(mx.isfinite(hidden)).item()):
            raise FloatingPointError(
                "GLM-5.3-Flash produced non-finite prefill hidden states (NaN/Inf); "
                "generation aborted before sampling. No valid completion was produced."
            )
        if not keep_hidden:
            output.hidden_states = None
        return output

    PromptProcessingBatch.prompt_step = prompt_step
    LanguageModel.__call__ = forward
    LanguageModel._local_prefill_logits = True
    logging.getLogger(__name__).warning(
        "GLM prefill patch: unused vocabulary projection %s; hidden-state guard retained",
        "skipped" if _ENABLED else "unchanged",
    )
