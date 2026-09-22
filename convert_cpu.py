#!/usr/bin/env python3
"""在 CPU 上执行 FP8 -> MXFP8 转换，并记录分片写入进度。

避免慢盘 mmap 缺页阻塞单个 Metal 命令；实际耗时取决于存储和计算吞吐。
"""

from pathlib import Path
import time

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_vlm.convert import main  # noqa: E402  (必须在切 device 之后导入)

if __name__ == "__main__":
    save_safetensors = mx.save_safetensors

    def save_with_progress(path, tensors, *args, **kwargs):
        started = time.monotonic()
        size = sum(tensor.nbytes for tensor in tensors.values())
        print(f"[写入] {path} ({size / 2**30:.2f} GiB)", flush=True)
        result = save_safetensors(path, tensors, *args, **kwargs)
        print(f"[已写] {Path(path).name} ({time.monotonic() - started:.1f}s)", flush=True)
        return result

    mx.save_safetensors = save_with_progress
    main()
