#!/usr/bin/env python3
"""抽查 MXFP8 产物：与源 FP8 权重反量化后的参考值逐张量比较相对误差。

只挑没有被融合/堆叠的张量（o_proj、shared_experts.down_proj），这样源里的
一个权重正好对应产物里的一个权重，误差就只反映 fp8 -> mxfp8 这一步本身。
"""

import json
import sys
from pathlib import Path

import mlx.core as mx

# CPU 抽查避免外置权重的 mmap 缺页阻塞 Metal 命令。
mx.set_default_device(mx.cpu)

from mlx_vlm.fp8 import _dequantize_fp8_weight

SUFFIXES = (".self_attn.o_proj.weight", ".mlp.shared_experts.down_proj.weight")


def load_one(path: Path, index: dict, key: str) -> mx.array:
    return mx.load(str(path / index[key]))[key]


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/model/GLM-5.3-Flash-FineTunning")
    dst = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path.home() / "models/GLM-5.3-Flash-FineTunning-MXFP8"
    sample = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if sample < 1:
        raise SystemExit("抽样数必须为正数")

    config = json.loads((dst / "config.json").read_text())
    quant = config.get("quantization")
    print(f"config.quantization = {quant}")
    if quant != {"group_size": 32, "bits": 8, "mode": "mxfp8"}:
        raise SystemExit("config.json 里的量化参数不是 mxfp8，先跑 finalize_mxfp8.py")

    src_index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    dst_index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]

    # MTP 层单独保存，主模型抽样仅覆盖配置声明的隐藏层。
    num_layers = int(config["text_config"]["num_hidden_layers"])
    def in_base_model(key: str) -> bool:
        marker = ".layers."
        if marker not in key:
            return True
        return int(key.split(marker, 1)[1].split(".", 1)[0]) < num_layers

    candidates = [
        key
        for key in src_index
        if key.endswith(SUFFIXES)
        and f"{key}_scale_inv" in src_index
        and in_base_model(key)
    ]
    step = max(len(candidates) // sample, 1)
    picked = candidates[::step][:sample]
    if not picked:
        raise SystemExit("没有找到可验证的 FP8 张量")

    print(f"{'张量':<62} {'相对误差 RMS':>12} {'最大':>10}")
    worst = 0.0
    for key in picked:
        reference = _dequantize_fp8_weight(
            load_one(src, src_index, key),
            load_one(src, src_index, f"{key}_scale_inv"),
        ).astype(mx.float32)

        out_key = "language_model.model." + key[len("model.language_model.") :]
        prefix = out_key[: -len(".weight")]
        got = mx.dequantize(
            load_one(dst, dst_index, out_key),
            load_one(dst, dst_index, f"{prefix}.scales"),
            group_size=32,
            bits=8,
            mode="mxfp8",
        ).astype(mx.float32)

        if got.shape != reference.shape:
            raise SystemExit(f"{out_key} 形状对不上: {got.shape} vs {reference.shape}")

        scale = mx.sqrt(mx.mean(mx.square(reference)))
        rms = float(mx.sqrt(mx.mean(mx.square(got - reference))) / scale)
        peak = float(mx.max(mx.abs(got - reference)) / mx.max(mx.abs(reference)))
        if not bool(mx.all(mx.isfinite(got)).item()) or not (0 <= rms < 0.05):
            raise SystemExit(f"{out_key} 转换误差不合格或含非有限值: RMS={rms}")
        worst = max(worst, rms)
        print(f"{out_key[-60:]:<62} {rms:>11.4%} {peak:>9.2%}")

    print()
    # 5% 是本地抽查阈值，不能据此保证其它张量或下游任务质量。
    print(f"判定: 通过（抽查最大 RMS {worst:.2%}）")


if __name__ == "__main__":
    main()
