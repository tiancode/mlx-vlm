#!/usr/bin/env python3
"""Validate GLM FP8 shards and fill missing index/processor metadata before conversion."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
    "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


def read_index(path: Path, *, allow_single: bool = False) -> dict:
    files = sorted(path.glob("model-*.safetensors"))
    single = path / "model.safetensors"
    if allow_single and single.exists():
        if files:
            raise ValueError("单文件权重与分片不能混用")
        files = [single]
    else:
        parts = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", file.name) for file in files]
        if any(part is None for part in parts):
            raise ValueError("分片文件名不合法")
        if parts:
            count = int(parts[0][2])
            if (any(int(part[2]) != count for part in parts)
                    or sorted(int(part[1]) for part in parts) != list(range(1, count + 1))):
                raise ValueError("分片不齐全或存在不同批次的分片")
    if not files:
        raise ValueError(f"没有分片: {path}")
    weight_map = {}
    total_size = 0
    for shard in files:
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"截断的 safetensors header: {shard}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= min(shard.stat().st_size - 8, 100_000_000):
                raise ValueError(f"非法 safetensors header: {shard}")
            header = json.loads(stream.read(header_size))
        if not isinstance(header, dict):
            raise ValueError(f"非法 safetensors header: {shard}")
        tensors = []
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(tensor, dict):
                raise ValueError(f"非法张量描述: {shard}: {key}")
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if (not isinstance(shape, list) or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in shape + offsets)
                    or not isinstance(tensor.get("dtype"), str)
                    or tensor.get("dtype") not in DTYPE_BYTES):
                raise ValueError(f"非法张量描述: {shard}: {key}")
            tensors.append((key, tensor))
        end = 0
        entries = sorted(tensors, key=lambda item: tuple(item[1]["data_offsets"]))
        for key, tensor in entries:
            begin, stop = tensor["data_offsets"]
            expected = math.prod(tensor["shape"]) * DTYPE_BYTES[tensor["dtype"]]
            if key in weight_map or begin != end or stop - begin != expected:
                raise ValueError(f"张量重复、截断或布局不合法: {shard}: {key}")
            weight_map[key] = shard.name
            total_size += expected
            end = stop
        if end + 8 + header_size != shard.stat().st_size:
            raise ValueError(f"分片大小不匹配: {shard}")
    if not weight_map:
        raise ValueError(f"没有权重张量: {path}")
    return {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}


def prepare(source: Path, processor_source: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "glm5_next" or quant.get("quant_method") != "fp8"
            or quant.get("fmt") != "e4m3" or quant.get("weight_block_size") != [128, 128]):
        raise ValueError("此转换流程要求 glm5_next、FP8 E4M3、128x128 block 权重")
    index = read_index(source)
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text())
        if existing["weight_map"] != index["weight_map"]:
            raise ValueError("已有权重索引与实际分片不匹配")
    else:
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        print(f"[补齐] {index_path}")

    processor = source / "processor_config.json"
    if not processor.exists():
        base_config = json.loads((processor_source / "config.json").read_text())
        fields = ("model_type", "vision_config", "image_token_id", "video_token_id",
                  "image_start_token_id", "image_end_token_id", "video_start_token_id", "video_end_token_id")
        if any(config.get(key) != base_config.get(key) for key in fields):
            raise ValueError("视觉配置与参考模型不同，不能复用其 processor_config.json")
        shutil.copy2(processor_source / processor.name, processor)
        print(f"[补齐] {processor}（来自 {processor_source}）")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (source / name).is_file():
            raise ValueError(f"缺少微调模型文件: {source / name}")
    print(f"[校验] {len(set(index['weight_map'].values()))} 分片，"
          f"{len(index['weight_map'])} 张量，{index['metadata']['total_size'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--processor-source", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.processor_source)
