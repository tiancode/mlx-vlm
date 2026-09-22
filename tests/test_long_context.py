"""Run with the serving venv on a Mac with Metal access (no weights loaded)."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import server_launch
from long_context_patch import require_finite_logits, install_apc_limit
from mlx_vlm import apc
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache, PoolingCache
from mlx_vlm.models.sparse_attention import indexed_sparse_attention


class SparseAttentionTests(unittest.TestCase):
    def test_matches_gathered_reference(self):
        mx.random.seed(42)
        for dtype in (mx.bfloat16, mx.float16):
            q = mx.random.normal((2, 4, 9, 32)).astype(dtype)
            k = mx.random.normal((2, 2, 41, 32)).astype(dtype)
            v = mx.random.normal((2, 2, 41, 64)).astype(dtype)
            idx = mx.array([[[0, 9, 24, 40, -1]] * 8 + [[-1] * 5]] * 2)
            out = indexed_sparse_attention(q, k, v, idx, 32**-0.5)
            refs = []
            for b in range(2):
                rows = []
                for t in range(9):
                    valid = (idx[b, t] >= 0)
                    safe = mx.maximum(idx[b, t], 0)
                    row = mx.fast.scaled_dot_product_attention(
                        q[b:b+1, :, t:t+1], k[b:b+1, :, safe], v[b:b+1, :, safe],
                        scale=32**-0.5, mask=valid[None, None, None],
                    )
                    # The sparse kernel defines an empty selected set as zero.
                    # MLX SDPA's all-false mask returns a nonzero row on 0.32.2.
                    rows.append(mx.where(mx.any(valid), row, mx.zeros_like(row)))
                refs.append(mx.concatenate(rows, axis=2))
            ref = mx.concatenate(refs, axis=0)
            self.assertTrue(bool(mx.all(mx.isfinite(out)).item()))
            self.assertLess(float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()), 0.02)

    @unittest.skipUnless(os.environ.get("RUN_LARGE_METAL_TESTS") == "1", "allocates up to 8.6 GB")
    def test_large_offsets(self):
        # Cross signed 2^31, unsigned 2^32 and the same boundary via batch stride.
        for batch, heads, length in ((1, 64, 131073), (1, 64, 192372), (1, 64, 262145), (2, 32, 131073)):
            with self.subTest(batch=batch, heads=heads, length=length):
                shape = (batch, heads, length, 256)
                # Share K/V storage. Constant by head + batch, so wrong addressing
                # cannot silently pass by retrieving another head's values.
                labels = mx.arange(batch * heads).reshape(batch, heads, 1, 1).astype(mx.bfloat16) + 1
                kv = mx.contiguous(mx.broadcast_to(labels, shape))
                q = mx.zeros((batch, heads, 1, 256), dtype=mx.bfloat16)
                idx = mx.array([[[length - 1, 0, -1]]] * batch, dtype=mx.int32)
                out = indexed_sparse_attention(q, kv, kv, idx, 1.0)
                expected = mx.broadcast_to(labels, out.shape)
                self.assertTrue(bool(mx.all(out == expected).item()))
                del kv, q, idx, out, expected, labels
                mx.clear_cache()


class GuardTests(unittest.TestCase):
    def test_old_disk_namespace_isolated(self):
        self.assertTrue(apc.apc_disk_namespace("/model").endswith("#local-wide-address-v1"))
        self.assertNotEqual(apc.apc_disk_namespace("/model"), apc.apc_disk_namespace("/other"))

    def test_nonfinite_logits(self):
        require_finite_logits(None)
        require_finite_logits(mx.array([[[0.0, 1.0, -2.0]]]))
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(FloatingPointError, "before sampling"):
                require_finite_logits(mx.array([[[0.0, bad]]]))

    def test_empty_batch_cache(self):
        batch = BatchKVCache.merge([KVCache(), KVCache()])
        for i in range(2):
            self.assertTrue(batch.extract(i).empty())

    def test_exact_limit_forwarding(self):
        class Manager:
            def lookup_exact_cache(self, *args, **kwargs):
                return kwargs
            def store_exact_cache(self, *args, **kwargs):
                return True
        with patch.dict(os.environ, {"APC_EXACT_MAX_TOKENS": "64"}):
            install_apc_limit(Manager)
        m = Manager()
        for requested, expected in ((None, 64), (0, 64), (100, 64), (32, 32)):
            self.assertEqual(m.lookup_exact_cache([1] * 100, max_prefix_tokens=requested)["max_prefix_tokens"], expected)
        self.assertTrue(m.store_exact_cache([1] * 64, []))
        self.assertFalse(m.store_exact_cache([1] * 65, []))

    def test_cap_before_eviction(self):
        class Manager:
            disk = None
            memory_max_bytes = 32 * (1 << 30)
            _bytes_per_token = 3.28e6
            def _make_room(self):
                self.observed = self._prefill_reserve_bytes
        m = Manager()
        server_launch._prepare_prefill(m, 101414)
        self.assertEqual(m.observed, 8 * (1 << 30))

    def test_disk_threshold_applies_without_stripping(self):
        cache = [object()]
        with patch.object(server_launch, "_STRIP_DERIVED", False), \
             patch.object(server_launch, "_DISK_MIN_TOKENS", 8), \
             patch.object(server_launch, "_original_save_exact", return_value=True) as save:
            self.assertFalse(apc.DiskBlockStore.save_exact_cache(None, 1, (1,) * 7, 0, cache))
            save.assert_not_called()
            self.assertTrue(apc.DiskBlockStore.save_exact_cache(None, 1, (1,) * 8, 0, cache))
            self.assertIs(save.call_args.args[4], cache)


class SnapshotTests(unittest.TestCase):
    def make_cache(self, length):
        states = ArraysCache(2)
        states[0] = mx.arange(48).reshape(1, 2, 24).astype(mx.bfloat16)
        states[1] = mx.arange(128).reshape(1, 2, 8, 8).astype(mx.float32) / 7
        latent = KVCache()
        latent.update_and_fetch(mx.arange(length * 8).reshape(1, 1, length, 8).astype(mx.bfloat16), mx.zeros((1, 1, length, 0), mx.bfloat16))
        valid = KVCache()
        valid.update_and_fetch(mx.ones((1, 1, length, 1), mx.bool_), mx.zeros((1, 1, length, 0), mx.bool_))
        pool = PoolingCache(4)
        pool.pooled = mx.arange((length // 4) * 8).reshape(1, length // 4, 8).astype(mx.bfloat16)
        pool.accumulate_windows(mx.ones((1, length % 4, 8), mx.bfloat16), mx.full((1, length % 4, 8), 0.5, mx.bfloat16), length - length % 4)
        projected = KVCache()
        projected.update_and_fetch(mx.ones((1, 2, 16, 8), mx.bfloat16), mx.ones((1, 2, 16, 8), mx.bfloat16))
        return [states, CacheList(latent, valid, pool, projected)]

    def assert_state_equal(self, left, right):
        for a, b in zip(left[0].cache, right[0].cache):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(bool(mx.array_equal(a, b).item()))
        for i in (0, 1):
            a, b = left[1][i], right[1][i]
            self.assertEqual(a.offset, b.offset)
            self.assertTrue(bool(mx.array_equal(a.keys[:, :, :a.offset], b.keys[:, :, :b.offset]).item()))
        a, b = left[1][2], right[1][2]
        self.assertEqual(a.remainder, b.remainder)
        for name in ("buf_kv", "buf_gate", "pooled"):
            self.assertTrue(bool(mx.array_equal(getattr(a, name), getattr(b, name)).item()))
        self.assertTrue(right[1][3].empty())

    def test_clone_strips_before_copy_and_preserves_state(self):
        live = self.make_cache(189047)
        with patch.object(server_launch, "_original_clone", wraps=server_launch._original_clone) as clone:
            restored = apc._clone_prompt_cache_for_apc(live, min_capacity_tokens=190000)
            self.assertTrue(clone.call_args.args[0][1][3].empty())
        self.assertFalse(live[1][3].empty())
        self.assert_state_equal(live, restored)
        restored[0][1] = mx.zeros_like(restored[0][1])
        self.assertFalse(bool(mx.array_equal(live[0][1], restored[0][1]).item()))

    def test_disk_roundtrip_long_state(self):
        live = self.make_cache(189047)
        with tempfile.TemporaryDirectory() as directory:
            disk = apc.DiskBlockStore(Path(directory), max_bytes=1 << 28)
            try:
                tokens = tuple(range(189047))
                self.assertTrue(disk.save_exact_cache(123, tokens, 7, live, synchronous=True))
                loaded = disk.load_exact_cache(123)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded[0], tokens)
                self.assertEqual(loaded[1], 7)
                self.assert_state_equal(live, loaded[2])
            finally:
                disk.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
