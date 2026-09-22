#!/bin/bash
# 独立图片环境；不升级现有 GLM 环境。
set -euo pipefail
cd "$(dirname "$0")"
IMAGE_VENV=${IMAGE_VENV:-$HOME/.venvs/qwen-image}
BASE_PYTHON=${BASE_PYTHON:-$HOME/.venvs/mlx-vlm/bin/python}
if [ ! -x "$IMAGE_VENV/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$IMAGE_VENV"
fi
"$IMAGE_VENV/bin/python" -m pip install -r requirements-image.txt
"$IMAGE_VENV/bin/python" -m pip check
