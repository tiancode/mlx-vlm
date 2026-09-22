import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_vlm.generate.ar import SpeculativeGenerationBatch
from request_cleanup_patch import install, release_idle_batch, bind_prompt_inputs, release_prompt_inputs, OwnedPromptInputs

install()


class CleanupTests(unittest.TestCase):
    def test_owned_embeddings_release_all_server_references(self):
        class Generator:
            def _gpu_embed(self):
                return [1], {"inputs_embeds": object()}
        class Batch:
            def insert(self, prompts, max_tokens=None, prompt_kwargs=None):
                self.forwarded = prompt_kwargs
                return list(range(len(prompts)))
        bind_prompt_inputs(Generator, Batch)
        a = Generator()._gpu_embed()[1]
        b = Generator()._gpu_embed()[1]
        caller_owned = {"inputs_embeds": object()}
        batch = Batch()
        batch.insert([[1], [2], [3]], prompt_kwargs=[a, b, caller_owned])
        self.assertIsInstance(a, OwnedPromptInputs)
        self.assertEqual(len(batch._local_prompt_inputs), 2)
        release_prompt_inputs(batch, [0])
        self.assertEqual(a, {})
        self.assertEqual(batch.forwarded[0], {})
        self.assertTrue(b)
        release_prompt_inputs(batch)
        self.assertEqual(b, {})
        self.assertTrue(caller_owned)
        self.assertEqual(batch._local_prompt_inputs, {})

    def batch(self, count=1, max_tokens=16):
        return SpeculativeGenerationBatch(
            model=None, draft_model=None, draft_kind="mtp", uids=list(range(count)),
            first_tokens=mx.ones((count,), mx.int32), prompt_cache=[object()],
            sampler=None, stop_criteria=lambda token: False,
            max_tokens=[max_tokens] * count, hidden=mx.ones((count, 1, 8)),
            shared_kv_states={"test": mx.ones((count, 1))},
            prompt_tokens=mx.ones((count, 4), mx.int32),
        )

    def assert_released(self, batch):
        self.assertEqual(batch.prompt_cache, [])
        for name in ("hidden", "shared_kv_states", "prompt_tokens", "first_tokens", "_rounds_iter"):
            self.assertIsNone(getattr(batch, name), name)

    def test_first_token_completes_request(self):
        batch = self.batch(max_tokens=1)
        response = batch.next()
        self.assertEqual(response[0].token, 1)
        self.assertEqual(response[0].finish_reason, "length")
        self.assert_released(batch)
        self.assertEqual(batch.next(), [])

    def test_cancel_closes_suspended_rounds(self):
        batch = self.batch()
        closed = []
        def rounds():
            try:
                yield [1], None
            finally:
                closed.append(True)
        batch._rounds_iter = rounds()
        next(batch._rounds_iter)
        batch.filter([])
        self.assertEqual(closed, [True])
        self.assert_released(batch)

    def test_partial_cancel_preserves_other_row(self):
        batch = self.batch(count=2)
        cache = batch.prompt_cache
        batch.filter([1])
        self.assertEqual(batch.uids, [1])
        self.assertIs(batch.prompt_cache, cache)
        self.assertIsNotNone(batch.hidden)
        batch.filter([])
        self.assert_released(batch)

    def test_normal_round_completion(self):
        batch = self.batch(max_tokens=2)
        batch.next()
        def rounds():
            yield [2], None
        batch._rounds_iter = rounds()
        responses = batch.next()
        self.assertEqual(responses[0].token, 2)
        self.assertEqual(responses[0].finish_reason, "length")
        self.assert_released(batch)

    def test_idle_only_releases_wired_limit(self):
        batch = Mock(has_work=True)
        with patch("request_cleanup_patch.mx.clear_cache") as clear:
            release_idle_batch(batch)
            batch.close.assert_not_called()
            clear.assert_not_called()
            batch.has_work = False
            release_idle_batch(batch)
            batch.close.assert_called_once()
            clear.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
