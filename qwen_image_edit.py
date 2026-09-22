"""Native MLX Qwen-Image-2.1 editing, sharing the text-to-image weights.

Port of the image-conditioned path in huggingface/diffusers PR #14804,
commit 8d3c30bfda9b511c00992f40cff4170a5502814d. Reference image slots expand
fourfold into VAE tokens. Text is causal; each image is bidirectional. The
prefix uses t=0, so its per-layer keys/values can be computed once per request.
Encoded conditions and reference tensors are local to each generate() call.
"""
import base64
from io import BytesIO
import logging
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from mlx_vlm.models.qwen_image.scheduler import FlowMatchEulerDiscreteScheduler
from mlx_vlm.models.qwen_image.text_encoder import IMAGE_PLACEHOLDER, PROMPT_TEMPLATE_TI2I
from mlx_vlm.models.qwen_image.transformer import _apply_rope

from image_inputs import InvalidEditPrompt, prepare_reference

logger = logging.getLogger("qwen_image")


class EditLayout:
    """CPU metadata for interleaving, attention segments and rotary positions."""
    def __init__(self, image_mask, shapes, target_shape):
        image_mask = np.asarray(image_mask, dtype=bool).reshape(-1)
        self.parts = []  # (is_image, encoder_start, encoder_end, reference_index)
        self.segments = []  # (joint_start, joint_end, is_image)
        axes = [[], [], []]
        cursor = position = joint = ref = 0
        while cursor < len(image_mask):
            end = cursor + 1
            is_image = bool(image_mask[cursor])
            while end < len(image_mask) and bool(image_mask[end]) == is_image:
                end += 1
            self.parts.append((is_image, cursor, end, ref))
            if is_image:
                if ref >= len(shapes):
                    raise ValueError("Unexpected image placeholder in prompt")
                h, w = shapes[ref]
                if (end - cursor) * 4 != h * w:
                    raise ValueError("Vision and VAE image grids do not match")
                self._image_positions(axes, position, h, w)
                length = h * w
                position += max(h, w)
                ref += 1
            else:
                length = end - cursor
                for axis in axes:
                    axis.extend(range(position, position + length))
                position += length
            self.segments.append((joint, joint + length, is_image))
            joint += length
            cursor = end
        if ref != len(shapes):
            raise ValueError("Reference image count does not match the prompt")
        self.prefix_len = joint
        self.target_tokens = math.prod(target_shape)
        self._image_positions(axes, position, *target_shape)
        self.axes = axes

    @staticmethod
    def _image_positions(axes, position, h, w):
        axes[0].extend([position] * (h * w))
        axes[1].extend(y for y in range(-(h - h // 2), h // 2) for _ in range(w))
        axes[2].extend(x for _ in range(h) for x in range(-(w - w // 2), w // 2))

    def rope(self, model):
        angles = mx.concatenate([
            mx.array(axis, dtype=mx.float32)[:, None]
            / model.pos_embed.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)[None]
            for axis, dim in zip(self.axes, model.pos_embed.axes_dim)
        ], axis=-1)
        return mx.cos(angles), mx.sin(angles)


def attention(block, x, cos, sin, *, segments=None, cached=None):
    attn = block.attn
    b, s, _ = x.shape
    shape = (b, s, attn.heads, attn.head_dim)
    q = _apply_rope(attn.norm_q(attn.to_q(x).reshape(shape)).transpose(0, 2, 1, 3), cos, sin)
    k = _apply_rope(attn.norm_k(attn.to_k(x).reshape(shape)).transpose(0, 2, 1, 3), cos, sin)
    v = attn.to_v(x).reshape(shape).transpose(0, 2, 1, 3)
    prefix_kv = (k, v)
    if cached is not None:
        k = mx.concatenate([cached[0], k], axis=2)
        v = mx.concatenate([cached[1], v], axis=2)
    scale = 1.0 / math.sqrt(attn.head_dim)
    if segments is None:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    else:
        outputs = []
        for start, end, is_image in segments:
            # Only short text runs need a mask; image blocks attend the whole
            # preceding prefix and themselves. Never allocate a full S x S mask.
            mask = None if is_image else mx.arange(end)[None, :] <= mx.arange(start, end)[:, None]
            outputs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, start:end], k[:, :, :end], v[:, :, :end], scale=scale, mask=mask))
        out = mx.concatenate(outputs, axis=2)
    out = out.transpose(0, 2, 1, 3).reshape(b, s, attn.heads * attn.head_dim)
    return attn.to_out[0](out), prefix_kv


def run_block(block, h, modulation, cos, sin, *, segments=None, cached=None):
    mod1, mod2 = mx.split(modulation, 2, axis=-1)
    x, gate = block._modulate(block.img_norm1(h), mod1, None)
    out, kv = attention(block, x, cos, sin, segments=segments, cached=cached)
    h = h + mx.tanh(gate) * out
    x, gate = block._modulate(block.img_norm2(h), mod2, None)
    return h + mx.tanh(gate) * block.img_mlp(x), kv


class EditCondition:
    def __init__(self, model, embeddings, image_mask, refs, shapes, target_shape, check_cancel):
        if not model.causal_condition:
            raise ValueError("Editing cache requires causal_condition weights")
        layout = EditLayout(image_mask, shapes, target_shape)
        text = model.txt_in(embeddings)
        projected_refs = [model.img_in(ref) for ref in refs]
        h = mx.concatenate([
            projected_refs[ref] if is_image else text[:, start:end]
            for is_image, start, end, ref in layout.parts
        ], axis=1)
        cos, sin = layout.rope(model)
        self.cos, self.sin = cos[layout.prefix_len:], sin[layout.prefix_len:]
        cos, sin = cos[:layout.prefix_len], sin[:layout.prefix_len]
        temb = model.time_text_embed(mx.zeros((1,), dtype=h.dtype), h.dtype)
        modulation = model.modulation[0](nn.silu(temb))
        self.cache = []
        logger.info("Editing prefix: %d tokens, %d references", layout.prefix_len, len(refs))
        for i, block in enumerate(model.transformer_blocks):
            check_cancel()
            h, kv = run_block(block, h, modulation, cos, sin, segments=layout.segments)
            mx.eval(h, *kv)
            self.cache.append(kv)
            if (i + 1) % 8 == 0:
                logger.info("Editing prefix encoded: layer %d", i + 1)

    def predict(self, model, latents, timestep, check_cancel):
        h = model.img_in(latents)
        temb = model.time_text_embed(timestep.astype(h.dtype), h.dtype)
        modulation = model.modulation[0](nn.silu(temb))
        for block, kv in zip(model.transformer_blocks, self.cache):
            check_cancel()
            h, _ = run_block(block, h, modulation, self.cos, self.sin, cached=kv)
        return model.proj_out(model.norm_out(h, temb, None))


class QwenImageEditor:
    def __init__(self, pipeline, transformer):
        self.pipeline = pipeline
        self.transformer = transformer
        encoder = pipeline.text_encoder
        self.processor = Qwen3VLProcessor.from_pretrained(encoder.processor_dir, local_files_only=True)
        # mlx-vlm 0.7.2 loads Qwen3-VL names but omits its Conv3d layout
        # conversion. Text-only prompts do not use this vision projection.
        proj = encoder.model.vision_tower.patch_embed.proj
        if tuple(proj.weight.shape[1:]) == (3, 2, 16, 16):
            proj.weight = proj.weight.transpose(0, 2, 3, 4, 1)
            mx.eval(proj.weight)

    def encode_prompt(self, prompt, images, check_cancel):
        encoder = self.pipeline.text_encoder
        # Do not truncate image placeholders: that silently drops references.
        if len(encoder.tokenizer.encode(prompt, add_special_tokens=False)) > 1024:
            raise InvalidEditPrompt("Editing prompt must be at most 1024 tokens")
        if any(token in prompt for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>")):
            raise InvalidEditPrompt("Prompt must not contain vision control tokens")
        prefix = " ".join(f"<image{i + 1}>{IMAGE_PLACEHOLDER}" for i in range(len(images)))
        rgb_images = []
        for image in images:
            rgb = Image.new("RGB", image.size, "white")
            rgb.paste(image, mask=image.getchannel("A"))
            rgb_images.append(rgb)
        inputs = self.processor(text=[PROMPT_TEMPLATE_TI2I.format(prefix, prompt or " ")],
            images=rgb_images, padding=True, padding_side="left", min_pixels=32**2,
            max_pixels=max(image.width * image.height for image in images) + 1)
        check_cancel()
        hidden = encoder._hidden_states(dict(inputs))[:, encoder.drop_idx:].astype(mx.bfloat16)
        mx.eval(hidden)
        mask = np.asarray(inputs["input_ids"])[0, encoder.drop_idx:] == self.processor.image_token_id
        check_cancel()
        return hidden, mask

    def generate(self, request, check_cancel):
        p = self.pipeline
        width, height = map(int, request.size.split("x"))
        target_shape = (height // 16, width // 16)
        images = [prepare_reference(data, request.reference_size) for data in request.images]
        logger.info("Editing %d images: encoding visual prompt", len(images))
        emb, mask = self.encode_prompt(request.prompt, images, check_cancel)
        refs, shapes = [], []
        for i, image in enumerate(images):
            check_cancel()
            rgba = mx.array(np.asarray(image).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None, :, None]
            mean, _ = p.vae.encode(rgba)
            normalized = (mean - p.latents_mean) / p.latents_std
            ref = normalized.reshape(1, p.z_dim, -1).transpose(0, 2, 1).astype(mx.bfloat16)
            mx.eval(ref)
            refs.append(ref)
            shapes.append((image.height // 16, image.width // 16))
            logger.info("Reference %d VAE encoded: %dx%d", i + 1, image.width, image.height)
        condition = EditCondition(self.transformer, emb, mask, refs, shapes, target_shape, check_cancel)
        negative = None
        if request.guidance > 1:
            neg_emb, neg_mask = self.encode_prompt(request.negative_prompt, images, check_cancel)
            negative = EditCondition(self.transformer, neg_emb, neg_mask, refs, shapes, target_shape, check_cancel)
            del neg_emb, neg_mask
        # Release image encoder intermediates before denoising.
        del emb, mask, refs, images, image, rgba, mean, normalized, ref
        mx.clear_cache()
        mx.random.seed(request.seed)
        tokens = math.prod(target_shape)
        latents = mx.random.normal((1, 1, p.z_dim, *target_shape)).astype(mx.bfloat16)
        latents = latents.reshape(1, p.z_dim, tokens).transpose(0, 2, 1)
        scheduler = FlowMatchEulerDiscreteScheduler(image_seq_len=tokens, num_inference_steps=request.steps)
        for i in range(request.steps):
            check_cancel()
            t = mx.array([float(scheduler.sigmas[i])], dtype=mx.bfloat16)
            pred = condition.predict(self.transformer, latents, t, check_cancel)
            if negative is not None:
                neg_pred = negative.predict(self.transformer, latents, t, check_cancel)
                pred = neg_pred + request.guidance * (pred - neg_pred)
            latents = scheduler.step(noise=pred, step_index=i, latents=latents)
            mx.eval(latents)
            if i == 0 or (i + 1) % 10 == 0:
                logger.info("Editing denoising step %d/%d", i + 1, request.steps)
        check_cancel()
        del condition, negative
        z = latents.transpose(0, 2, 1).reshape(1, p.z_dim, 1, *target_shape).astype(mx.float32)
        pixels = p.vae.decode(z * p.latents_std + p.latents_mean)[0, :3, 0].transpose(1, 2, 0)
        pixels = ((mx.clip(pixels, -1, 1) + 1) * 127.5).astype(mx.uint8)
        mx.eval(pixels)
        check_cancel()
        output = BytesIO()
        Image.fromarray(np.asarray(pixels)).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
