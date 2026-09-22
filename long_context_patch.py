"""GLM 的稀疏注意力地址、logits 与 APC 限长补丁，适配 mlx-vlm 0.7.1。

在模型构造前安装，不修改包或权重。回归见 tests/test_long_context.py，
完整模型验证范围见 diagnostics/README.md。
"""

from functools import wraps
import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)


def widen_sparse_addresses(source):
    """Promote operands BEFORE multiplying, including batch and head strides.

    64 heads * 131073 tokens * 256 elements exceeds INT_MAX. Casting only
    the completed product, or using uint32, cannot support the 1M context.
    Loop counters and token indices stay int32; device offsets use size_t.
    """
    declarations = (
        "uint row_idx",
        "int query_length",
        "int key_length",
        "int query_idx",
        "int batch_head_idx",
        "int batch_idx",
        "int q_head_idx",
        "int kv_head_idx",
        "int indices_offset",
    )
    for declaration in declarations:
        if source.count(declaration + " =") != 1:
            raise RuntimeError(
                "Sparse attention source changed; review the local address patch: "
                + declaration
            )
        source = source.replace(
            declaration + " =", "size_t " + declaration.split()[1] + " ="
        )
    # These casts would narrow the grid index before assigning to size_t.
    source = source.replace("int(row_idx % query_length)", "row_idx % query_length")
    source = source.replace("int(row_idx / query_length)", "row_idx / query_length")
    return source


def require_finite_logits(logits):
    """Check raw model logits, before processors legitimately add -inf masks."""
    if logits is not None and not bool(mx.all(mx.isfinite(logits)).item()):
        raise FloatingPointError(
            "GLM-5.3-Flash produced non-finite logits (NaN/Inf); "
            "generation aborted before sampling. No valid completion was produced."
        )


def install():
    from mlx_vlm import apc
    from mlx_vlm.models import sparse_attention
    from mlx_vlm.models.glm5_next.language import LanguageModel

    if getattr(sparse_attention, "_local_wide_addresses", False):
        return
    sparse_attention._INDEXED_SPARSE_ATTENTION_SOURCE = widen_sparse_addresses(
        sparse_attention._INDEXED_SPARSE_ATTENTION_SOURCE
    )
    sparse_attention._indexed_sparse_attention_kernel.cache_clear()
    sparse_attention._local_wide_addresses = True

    # Old disk checkpoints may already contain state computed by the broken
    # kernel. Keep those files intact, but never mix them with corrected state.
    original_namespace = apc.apc_disk_namespace

    @wraps(original_namespace)
    def namespace(*args, **kwargs):
        return original_namespace(*args, **kwargs) + "#local-wide-address-v1"

    apc.apc_disk_namespace = namespace

    original_call = LanguageModel.__call__

    @wraps(original_call)
    def checked_call(self, *args, **kwargs):
        result = original_call(self, *args, **kwargs)
        require_finite_logits(result.logits)
        return result

    LanguageModel.__call__ = checked_call
    logger.warning(
        "GLM long-context patch: 64-bit sparse addresses; finite-logit check enabled"
    )


def install_apc_limit(manager_class):
    """Optionally limit both memory and disk exact checkpoints.

    Select a shorter intact checkpoint through max_prefix_tokens; never trim a
    recurrent state to pretend it belongs to an earlier prefix.
    APC_EXACT_MAX_TOKENS=0 leaves the configured context budget in control.
    """
    limit = int(os.environ.get("APC_EXACT_MAX_TOKENS", "0"))
    if limit < 0:
        raise ValueError("APC_EXACT_MAX_TOKENS must be >= 0")
    if not limit:
        return
    original_lookup = manager_class.lookup_exact_cache
    original_store = manager_class.store_exact_cache

    @wraps(original_lookup)
    def lookup(
        self, token_ids, extra_hash=0, max_prefix_tokens=None, min_prefix_tokens=0
    ):
        maximum = (
            min(limit, max_prefix_tokens)
            if max_prefix_tokens and max_prefix_tokens > 0
            else limit
        )
        return original_lookup(
            self, token_ids, extra_hash=extra_hash,
            max_prefix_tokens=maximum, min_prefix_tokens=min_prefix_tokens,
        )

    @wraps(original_store)
    def store(self, token_ids, prompt_cache, *, extra_hash=0):
        if len(token_ids) > limit:
            return False
        return original_store(self, token_ids, prompt_cache, extra_hash=extra_hash)

    manager_class.lookup_exact_cache = lookup
    manager_class.store_exact_cache = store
