#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
IMAGE_PYTHON=${IMAGE_PYTHON:-$HOME/.venvs/qwen-image/bin/python}
export IMAGE_MODEL=${IMAGE_MODEL:-$HOME/models/Qwen-Image-2.1}
export IMAGE_PORT=${IMAGE_PORT:-1238}
export IMAGE_API_KEY=${IMAGE_API_KEY:-${API_KEY:-happy-coding-axm}}
export IMAGE_IDLE_TIMEOUT=${IMAGE_IDLE_TIMEOUT:-300}
export IMAGE_REQUEST_TIMEOUT=${IMAGE_REQUEST_TIMEOUT:-1800}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -x "$IMAGE_PYTHON" ] || { echo "请先运行 bash setup_image.sh" >&2; exit 1; }
[ -f "$IMAGE_MODEL/model_index.json" ] || { echo "图片模型不存在: $IMAGE_MODEL" >&2; exit 1; }
"$IMAGE_PYTHON" -c '
import socket,sys
with socket.socket() as probe:
    probe.settimeout(1)
    if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        sys.exit("图片端口已被占用；请先停止现有图片服务。")
' "$IMAGE_PORT"
mkdir -p logs
echo $$ > logs/image-runner.pid
exec "$IMAGE_PYTHON" -u service_runner.py --log logs/image.log -- \
  "$IMAGE_PYTHON" -u image_server.py
