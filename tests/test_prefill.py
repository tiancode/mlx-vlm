"""Numerical parity of cache-only prefill, including ragged batches (Metal)."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path
import server_launch
import prefill_patch
from mlx_vlm.generate.ar import PromptProcessingBatch
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next import language


def make_model(quantized=False):
    mx.random.seed(42)
    config = TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        kv_lora_rank=32, q_lora_rank=32, qk_nope_head_dim=32, v_head_dim=32,
        linear_num_heads=1, linear_head_dim=128,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "dense"], indexer_types=["full", "full"],
        index_skip_topk_offset=0, index_topk=8, index_kpool=2,
        index_n_heads=2, index_head_dim=32, hc_mult=2,
    )
    model = language.LanguageModel(config)
    model.update(tree_map_with_path(
        lambda path, x: x.astype(mx.bfloat16) if model.cast_predicate(path) else x,
        model.parameters(),
    ))
    if quantized:
        nn.quantize(
            model, group_size=32, bits=8, mode="mxfp8",
            class_predicate=lambda p, m: hasattr(m, "to_quantized") and model.quant_predicate(p, m),
        )
    model.eval()
    mx.eval(model.parameters())
    return model


class PrefillTests(unittest.TestCase):
    def batch(self, model, lengths=(73,), right=False):
        ids = [[i % 250 + 1 for i in range(length)] for length in lengths]
        maximum = max(lengths)
        padding = [maximum - n for n in lengths]
        padded = [
            row + [0] * pad if right else [0] * pad + row
            for row, pad in zip(ids, padding)
        ]
        return PromptProcessingBatch(
            model=model, uids=list(range(len(ids))), input_ids=ids,
            max_tokens=[5] * len(ids),
            inputs_embeds=model.model.embed_tokens(mx.array(padded)),
            prompt_kwargs={}, prefill_step_size=16,
            right_pad_per_row=padding if right else None,
            suffix_lens=list(lengths),
        )

    def run_batch(self, model, enabled, lengths=(73,), right=False):
        head_shapes, sampled_logits, tokens = [], [], []
        original_linear = language.linear

        def linear(module, hidden):
            if module is model.lm_head:
                head_shapes.append(tuple(hidden.shape))
            return original_linear(module, hidden)

        def sampler(logits):
            sampled_logits.append(mx.array(logits))
            return mx.argmax(logits, axis=-1)

        with patch.object(prefill_patch, "_ENABLED", enabled), patch.object(language, "linear", linear):
            batch = self.batch(model, lengths, right)
            while batch.needs_processing():
                self.assertGreater(batch.prompt_step(), 0)
                self.assertIsNone(prefill_patch._active_prefill.get())
            states = [
                (name, mx.array(value) if isinstance(value, mx.array) else value)
                for name, value in tree_flatten([c.state for c in batch.prompt_cache])
            ]
            mx.eval([value for _, value in states if isinstance(value, mx.array)])
            prefill_shapes = list(head_shapes)
            gen = batch.generate(sampler, lambda token: False)
            while len(gen):
                tokens.extend((r.uid, r.token, r.finish_reason) for r in gen.next())
            mx.eval(sampled_logits)
        return states, sampled_logits, tokens, prefill_shapes

    def assert_parity(self, model, lengths=(73,), right=False):
        before = self.run_batch(model, False, lengths, right)
        after = self.run_batch(model, True, lengths, right)
        self.assertEqual(before[2], after[2])
        self.assertEqual(len(before[1]), len(after[1]))
        for a, b in zip(before[1], after[1]):
            self.assertTrue(bool(mx.array_equal(a, b).item()), "sampling logits changed")
        self.assertEqual([name for name, _ in before[0]], [name for name, _ in after[0]])
        for (_, a), (_, b) in zip(before[0], after[0]):
            if isinstance(a, mx.array):
                self.assertTrue(bool(mx.array_equal(a, b).item()), "cache state changed")
            else:
                self.assertEqual(a, b)
        self.assertLess(len(after[3]), len(before[3]))
        return before, after

    def test_bf16_singleton_cache_and_decode_exact(self):
        _, after = self.assert_parity(make_model())
        self.assertEqual(after[3], [])

    def test_mxfp8_singleton_cache_and_decode_exact(self):
        self.assert_parity(make_model(quantized=True))

    def test_left_padded_batch_cache_and_decode_exact(self):
        self.assert_parity(make_model(), lengths=(49, 73))

    def test_right_padded_finished_row_readout_is_preserved(self):
        _, after = self.assert_parity(make_model(), lengths=(16, 73), right=True)
        self.assertEqual(after[3], [(2, 16, 64)])

    def test_hidden_guard_rejects_nonfinite_and_resets_context(self):
        model = make_model()
        for bad in (float("nan"), float("inf"), -float("inf")):
            batch = self.batch(model)
            batch._inputs_embeds = mx.full(batch._inputs_embeds.shape, bad, mx.bfloat16)
            with self.assertRaisesRegex(FloatingPointError, "prefill hidden states.*before sampling"):
                batch.prompt_step()
            self.assertIsNone(prefill_patch._active_prefill.get())

    def test_sampling_logits_guard_still_rejects_nonfinite_head(self):
        model = make_model()
        model.lm_head.weight = mx.full(model.lm_head.weight.shape, float("nan"), mx.bfloat16)
        batch = self.batch(model)
        # The head is unused in these chunks, but it must be checked before
        # sampling the first output token.
        while batch.needs_processing():
            batch.prompt_step()
        with self.assertRaisesRegex(FloatingPointError, "non-finite logits.*before sampling"):
            batch.generate(lambda logits: mx.argmax(logits, axis=-1), lambda token: False)

    def test_decode_and_verifier_calls_outside_prefill_unchanged(self):
        model = make_model()
        for length in (1, 3):
            output = model(mx.ones((1, length), mx.int32), n_to_process=length)
            self.assertEqual(output.logits.shape, (1, length, 256))
        self.assertIsNone(prefill_patch._active_prefill.get())

    def test_nested_context_does_not_change_another_model(self):
        model = make_model()
        marker = prefill_patch._active_prefill.set(SimpleNamespace(model=object()))
        try:
            output = model(mx.ones((1, 3), mx.int32), n_to_process=3)
            self.assertEqual(output.logits.shape, (1, 3, 256))
        finally:
            prefill_patch._active_prefill.reset(marker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
