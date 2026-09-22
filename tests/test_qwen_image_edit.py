"""Small MLX numerical checks; run with the image environment and Metal access."""
from pathlib import Path
from importlib.util import find_spec
import sys
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_QWEN_IMAGE = find_spec("mlx_vlm.models.qwen_image") is not None
if HAS_QWEN_IMAGE:
    from qwen_image_edit import EditCondition, EditLayout
    from mlx_vlm.models.qwen_image.transformer import QwenImageTransformer


@unittest.skipUnless(HAS_QWEN_IMAGE, "Run with ~/.venvs/qwen-image/bin/python (mlx-vlm 0.7.2)")
class EditingMathTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.model = QwenImageTransformer(in_channels=4, out_channels=4, num_layers=2,
            num_attention_heads=2, attention_head_dim=16, context_in_dim=12,
            axes_dims_rope=(4, 6, 6))
        # Initial zero-centered text norm would mask prompt differences in a
        # randomly initialized model; these mimic nonzero checkpoint weights.
        self.model.txt_in.text_norm.weight = mx.ones(12) * 0.1
        self.latents = mx.random.normal((1, 4, 4))

    def test_cached_target_matches_existing_text_pipeline_at_two_timesteps(self):
        emb = mx.random.normal((1, 5, 12))
        condition = EditCondition(self.model, emb, [False] * 5, [], [], (2, 2), lambda: None)
        for sigma in (1.0, 0.3):
            t = mx.array([sigma])
            expected = self.model(self.latents, emb, t, (1, 2, 2))
            actual = condition.predict(self.model, self.latents, t, lambda: None)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)

    def test_multiple_references_match_independent_dense_attention(self):
        model = self.model
        emb = mx.random.normal((1, 5, 12))
        refs = [mx.random.normal((1, 4, 4)), mx.random.normal((1, 4, 4))]
        condition = EditCondition(model, emb, [0, 1, 0, 1, 0], refs, [(2, 2), (2, 2)], (2, 2), lambda: None)
        text = model.txt_in(emb)
        h0 = mx.concatenate([text[:, :1], model.img_in(refs[0]), text[:, 2:3],
                             model.img_in(refs[1]), text[:, 4:5], model.img_in(self.latents)], axis=1)
        # Known positions: text 0, image at 1 (+2), text 3, image at 4
        # (+2), text 6, target at 7. Centered 2x2 spatial coordinates.
        frames = [0] + [1]*4 + [3] + [4]*4 + [6] + [7]*4
        ys = [0] + [-1, -1, 0, 0] + [3] + [-1, -1, 0, 0] + [6] + [-1, -1, 0, 0]
        xs = [0] + [-1, 0, -1, 0] + [3] + [-1, 0, -1, 0] + [6] + [-1, 0, -1, 0]
        angles = mx.concatenate([model.pos_embed._gather(a, p)
            for a, p in zip(model.pos_embed._angles, (frames, ys, xs))], axis=-1)
        cos, sin = mx.cos(angles), mx.sin(angles)
        ids = mx.array([-1] + [0]*4 + [-1] + [1]*4 + [-1] + [2]*4)
        indices = mx.arange(15)
        allowed = (indices[:, None] >= indices[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0))
        target = indices >= 11
        for sigma in (1.0, 0.3):
            temb = model.time_text_embed(mx.array([sigma, 0]), h0.dtype)
            mod = model.modulation[0](nn.silu(temb))
            h = h0
            for block in model.transformer_blocks:
                h = block(h, mod, cos, sin, allowed, target)
            expected = model.proj_out(model.norm_out(h, temb, target))[:, -4:]
            actual = condition.predict(model, self.latents, mx.array([sigma]), lambda: None)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)

    def test_grid_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            EditLayout([0, 1, 0], [(4, 4)], (2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
