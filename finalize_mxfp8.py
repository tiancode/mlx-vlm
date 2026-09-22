#!/usr/bin/env python3
"""验证转换产物，写入 MXFP8 量化配置，并补齐缺失的 tokenizer 与聊天模板。"""

import json
import shutil
import sys
from pathlib import Path
from prepare_checkpoint import read_index

MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
SIDECARS = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "LICENSE")


def check_weights(path: Path) -> tuple[int, int]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        missing = [name for name in set(weight_map.values()) if not (path / name).is_file()]
        if missing:
            raise SystemExit(f"{path} 缺少权重分片: {missing}")
    elif not (path / "model.safetensors").exists():
        raise SystemExit(f"缺少 {index} 或 model.safetensors")
    actual = read_index(path, allow_single=True)["weight_map"]
    if index.exists() and weight_map != actual:
        raise ValueError(f"{path} 权重索引与实际分片不匹配")
    weight_map = actual
    scales = sum(1 for key in weight_map if key.endswith(".scales"))
    if not scales:
        raise SystemExit(f"{path} 没有 MXFP8 scales，拒绝标记为已转换")
    leftover = [key for key in weight_map if key.endswith("weight_scale_inv")]
    if leftover:
        raise SystemExit(
            f"{path} 里还有 {len(leftover)} 个 fp8 block scale，转换没跑完整"
        )
    return len(weight_map), scales


def finalize(src: Path, dst: Path) -> None:
    total, scales = check_weights(dst)

    config_path = dst / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"] = dict(MXFP8)
    config["quantization_config"] = dict(MXFP8)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    for name in SIDECARS:
        source = src / name
        if source.exists() and not (dst / name).exists():
            shutil.copy2(source, dst / name)
            print(f"[补齐] {name}")

    size = sum(f.stat().st_size for f in dst.glob("model*.safetensors"))
    print(f"[完成] {dst}")
    print(f"       张量 {total} 个，其中 {scales} 个带 mxfp8 scale")
    print(f"       权重 {size / 2**30:.1f} GiB")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法: finalize_mxfp8.py <源模型目录> <MXFP8 目录>")
    finalize(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
