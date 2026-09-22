#!/bin/bash
# 启动并监管统一代理、GLM 后端和按需加载的图片服务。
cd "$(dirname "$0")"

# 默认权重放内置 SSD；DRAFT 为空时关闭 MTP。
MODEL=${MODEL:-$HOME/models/GLM-5.3-Flash-FineTunning-MXFP8}
DRAFT=${DRAFT-${MODEL}-mtp}
# 块长度包含 anchor；2 对应一个草稿 token，auto 使用上游自适应策略。
DRAFT_BLOCK_SIZE=${DRAFT_BLOCK_SIZE:-${MLX_VLM_DRAFT_BLOCK_SIZE:-2}}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-1235}
BACKEND_PORT=${BACKEND_PORT:-1236}
API_KEY=${API_KEY:-happy-coding-axm}
PUBLIC_NAME=${PUBLIC_NAME:-glm-5.3-flash}
IMAGE_ENABLED=${IMAGE_ENABLED:-1}
IMAGE_MODEL=${IMAGE_MODEL:-$HOME/models/Qwen-Image-2.1}
IMAGE_PORT=${IMAGE_PORT:-1238}
IMAGE_PYTHON=${IMAGE_PYTHON:-$HOME/.venvs/qwen-image/bin/python}
IMAGE_IDLE_TIMEOUT=${IMAGE_IDLE_TIMEOUT:-300}
IMAGE_REQUEST_TIMEOUT=${IMAGE_REQUEST_TIMEOUT:-1800}
# prompt 与生成预算共用此上限；0 使用模型声明上限。
CONTEXT=${CONTEXT:-393216}

TOKEN_QUEUE_TIMEOUT=${TOKEN_QUEUE_TIMEOUT:-3000}

case "$DRAFT_BLOCK_SIZE" in
  auto) unset MLX_VLM_DRAFT_BLOCK_SIZE ;;
  ''|*[!0-9]*|0*|1) echo "错误: DRAFT_BLOCK_SIZE 必须为 >=2 的整数或 auto。" >&2; exit 1 ;;
esac

# GLM 使用 exact 快照；此预算不等于推理进程总内存上限。
APC=${APC:-1}
APC_NUM_BLOCKS=${APC_NUM_BLOCKS:-2048}
APC_MEMORY_MAX_GB=${APC_MEMORY_MAX_GB:-32}
APC_DISK_PATH=${APC_DISK_PATH:-$HOME/.cache/mlx-vlm/apc}
APC_DISK_MAX_GB=${APC_DISK_MAX_GB:-200}

case "$HOST" in
  0.0.0.0|127.0.0.1) ;;
  *) echo "错误: HOST 仅支持 0.0.0.0 或 127.0.0.1。" >&2; exit 1 ;;
esac

# Python 环境放 APFS，避免 exFAT 伴生文件干扰包扫描。
PYTHON=${PYTHON:-$HOME/.venvs/mlx-vlm/bin/python}
[ -x "$PYTHON" ] || { echo "venv 不存在: $PYTHON" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "模型不存在: $MODEL（先跑 ./convert.sh）" >&2; exit 1; }
if [ "$IMAGE_ENABLED" = "1" ]; then
  [ -x "$IMAGE_PYTHON" ] || { echo "图片环境不存在，请先 bash setup_image.sh" >&2; exit 1; }
  [ -f "$IMAGE_MODEL/model_index.json" ] || { echo "图片权重不存在: $IMAGE_MODEL" >&2; exit 1; }
fi
"$PYTHON" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
try:
    config = json.loads((root / "config.json").read_text())
    weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    assert config.get("quantization") == {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    assert weights and all((root / name).is_file() for name in set(weights.values()))
except (OSError, ValueError, KeyError, AssertionError) as error:
    sys.exit(f"模型尚未完成转换或文件不完整: {root} ({error})")
' "$MODEL" || exit 1

"$PYTHON" -c '
import socket,sys
with socket.socket() as probe:
    probe.settimeout(1)
    if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        sys.exit("对外端口已被占用；请检查现有服务，切换配置前先停止服务。")
' "$PORT" || exit 1

mkdir -p logs
echo $$ > logs/start.pid

ARGS=(
  --model "$MODEL"
  --host 127.0.0.1
  --port "$BACKEND_PORT"
  --api-key "$API_KEY"

  --max-tokens 65536

  --enable-thinking

  --max-num-seqs 4

  --log-level INFO
)

if [ "$CONTEXT" != "0" ]; then
  ARGS+=(--max-kv-size "$CONTEXT")
fi

if [ -d "$DRAFT" ]; then
  ARGS+=(--draft-model "$DRAFT" --draft-kind mtp)
  if [ "$DRAFT_BLOCK_SIZE" != "auto" ]; then
    ARGS+=(--draft-block-size "$DRAFT_BLOCK_SIZE")
  fi
  echo "自投机解码: $DRAFT (block_size=$DRAFT_BLOCK_SIZE)"
else
  echo "未找到 drafter ($DRAFT)，按普通解码启动。"
fi

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  if [ "$(cat logs/start.pid 2>/dev/null)" = "$$" ]; then rm -f logs/start.pid; fi
  [ ${#PIDS[@]} -eq 0 ] && return 0
  echo "正在停止服务..."
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

show_failure() { # $1=名称 $2=PID $3=日志
  status=0
  wait "$2" || status=$?
  echo "错误: $1 启动失败 (退出码 $status)" >&2
  echo "----- $3 最后 30 行 -----" >&2
  tail -n 30 "$3" >&2
}

# 两次带鉴权探活和模型路径核对后才复用后端。
BACKEND_REUSED=0
backend_model_matches() {
  curl -fs --connect-timeout 1 --max-time 5 \
    -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$BACKEND_PORT/v1/models" |
    "$PYTHON" -c '
import json, sys
try:
    ids = [item["id"] for item in json.load(sys.stdin)["data"]]
except (ValueError, KeyError, TypeError):
    sys.exit("无法读取后端当前加载的模型")
if ids != [sys.argv[1]]:
    sys.exit(f"后端模型不匹配: 当前 {ids}，要求 {sys.argv[1]}。请先 ./stop.sh，再 ./start.sh。")
' "$MODEL"
}
backend_alive() {
  pgrep -f "server_launch[.]py .*--port ${BACKEND_PORT}( |$)" >/dev/null 2>&1 || return 1
  curl -fs --connect-timeout 1 --max-time 3 -o /dev/null \
    -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$BACKEND_PORT/health"
}
if backend_alive && sleep 1 && backend_alive; then
  backend_model_matches || exit 1
  echo "后端 127.0.0.1:$BACKEND_PORT 已在运行，复用（不重新加载模型）。"
  echo "注意: 复用的后端不是本脚本的子进程，Ctrl+C 只停止本次启动的子进程；要全停用 ./stop.sh。"
  BACKEND_REUSED=1
fi

# 外置权重预读可减少 Metal 命令等待慢盘缺页的风险。
PREWARM=${PREWARM:-auto}
if [ "$PREWARM" = "auto" ]; then
  case "$MODEL" in
    /Volumes/*) PREWARM=1 ;;
    *)          PREWARM=0 ;;
  esac
fi

if [ "$PREWARM" = "1" ] && [ "$BACKEND_REUSED" -eq 0 ]; then
  echo "顺序预读外置权重；耗时取决于源盘速度..."
  prewarm_started=$SECONDS
  for f in "$MODEL"/model-*.safetensors "$DRAFT"/model*.safetensors; do
    if [ -f "$f" ]; then
      dd if="$f" of=/dev/null bs=16m 2>/dev/null || { echo "预读失败: $f" >&2; exit 1; }
    fi
  done
  echo "预热完成，用时 $((SECONDS - prewarm_started))s。"
fi

if [ "$BACKEND_REUSED" -eq 0 ]; then
  echo "启动后端 mlx_vlm.server (127.0.0.1:$BACKEND_PORT)..."
  export MLX_VLM_TOKEN_QUEUE_TIMEOUT="$TOKEN_QUEUE_TIMEOUT"
  if [ "$APC" = "1" ]; then
    APC_DISK_ENABLED=${APC_DISK_ENABLED:-0}
    if [ "$APC_DISK_ENABLED" = "1" ]; then
      mkdir -p "$APC_DISK_PATH"
      echo "  前缀缓存: exact 快照模式, 上限 ${APC_MEMORY_MAX_GB} GiB + SSD $APC_DISK_PATH"
      export APC_DISK_PATH APC_DISK_MAX_GB
      export APC_DISK_MIN_TOKENS=${APC_DISK_MIN_TOKENS:-8192}
    else
      echo "  前缀缓存: exact 快照模式, 上限 ${APC_MEMORY_MAX_GB} GiB, 仅内存（不写 SSD）"
    fi
    export APC_ENABLED=1 APC_NUM_BLOCKS APC_MEMORY_MAX_GB APC_DISK_ENABLED
    export APC_STRIP_DERIVED=${APC_STRIP_DERIVED:-1}
    export APC_EXACT_MAX_TOKENS=${APC_EXACT_MAX_TOKENS:-0}
    echo "  exact 快照长度上限: ${APC_EXACT_MAX_TOKENS} token（0 = 不限制）"
  else
    export APC_ENABLED=0
  fi
  export GLM_ALLOCATOR_CACHE_GB=${GLM_ALLOCATOR_CACHE_GB:-2}
  echo "  GPU 分配池上限: ${GLM_ALLOCATOR_CACHE_GB} GiB（0 = 空闲时清空）"
  "$PYTHON" -u service_runner.py --log logs/glm.log -- \
    "$PYTHON" -u server_launch.py "${ARGS[@]}" &
  BACKEND_PID=$!
  PIDS+=("$BACKEND_PID")

  echo "等待模型就绪（进度在 logs/glm.log）..."
  started=$SECONDS
  backend_ready=0
  while [ $((SECONDS - started)) -lt 10800 ]; do
    if curl -fs --connect-timeout 1 --max-time 2 -o /dev/null \
         -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$BACKEND_PORT/health"; then
      backend_model_matches || exit 1
      backend_ready=1
      break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      show_failure "mlx_vlm.server" "$BACKEND_PID" logs/glm.log
      exit 1
    fi
    elapsed=$((SECONDS - started))
    if [ $((elapsed % 60)) -eq 0 ]; then echo "仍在加载 (${elapsed}s)..."; fi
    sleep 1
  done
  [ "$backend_ready" -eq 1 ] || { echo "错误: 后端启动超时；见 logs/glm.log" >&2; exit 1; }
  echo "后端已就绪（用时 $((SECONDS - started))s）。"
fi

PROXY_IMAGE_UPSTREAM=""
if [ "$IMAGE_ENABLED" = "1" ]; then
  image_ready() {
    curl -fs --connect-timeout 1 --max-time 3 \
      -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$IMAGE_PORT/health" |
      "$PYTHON" -c 'import json,sys; from pathlib import Path; d=json.load(sys.stdin); sys.exit(0 if d.get("model_path")==str(Path(sys.argv[1]).expanduser().resolve()) else 1)' "$IMAGE_MODEL" 2>/dev/null
  }
  if image_ready; then
    echo "图片服务 127.0.0.1:$IMAGE_PORT 已在运行，复用。"
  else
    echo "启动图片接口 (127.0.0.1:$IMAGE_PORT)，首次请求加载，空闲 ${IMAGE_IDLE_TIMEOUT}s 卸载。"
    IMAGE_PYTHON="$IMAGE_PYTHON" IMAGE_MODEL="$IMAGE_MODEL" IMAGE_PORT="$IMAGE_PORT" \
    IMAGE_API_KEY="$API_KEY" IMAGE_IDLE_TIMEOUT="$IMAGE_IDLE_TIMEOUT" \
    IMAGE_REQUEST_TIMEOUT="$IMAGE_REQUEST_TIMEOUT" bash ./start_image.sh &
    IMAGE_PID=$!
    PIDS+=("$IMAGE_PID")
    image_ok=0
    for _ in $(seq 1 30); do
      if image_ready; then image_ok=1; break; fi
      if ! kill -0 "$IMAGE_PID" 2>/dev/null; then
        show_failure "image_server" "$IMAGE_PID" logs/image.log
        exit 1
      fi
      sleep 1
    done
    [ "$image_ok" -eq 1 ] || { echo "图片服务启动超时，见 logs/image.log" >&2; exit 1; }
  fi
  PROXY_IMAGE_UPSTREAM="http://127.0.0.1:$IMAGE_PORT"
fi

echo "启动代理 ($HOST:$PORT)..."
PROXY_IMAGE_UPSTREAM="$PROXY_IMAGE_UPSTREAM" \
IMAGE_REQUEST_TIMEOUT="$IMAGE_REQUEST_TIMEOUT" \
PROXY_UPSTREAM="http://127.0.0.1:$BACKEND_PORT" \
PROXY_UPSTREAM_MODEL="$MODEL" \
PROXY_MODEL_NAME="$PUBLIC_NAME" \
PROXY_HOST="$HOST" \
PROXY_PORT="$PORT" \
"$PYTHON" -u service_runner.py --log logs/proxy.log -- \
  "$PYTHON" -u "$PWD/model_proxy.py" &
PROXY_PID=$!
PIDS+=("$PROXY_PID")

proxy_ready=0
for _ in $(seq 1 30); do
  if curl -fs --connect-timeout 1 --max-time 3 -o /dev/null \
       -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models"; then
    echo ""
    proxy_ready=1
    echo "全部就绪:"
    echo "  base_url : http://$HOST:$PORT/v1"
    echo "  model    : $PUBLIC_NAME"
    echo "  api_key  : $API_KEY"
    echo "  上下文   : $CONTEXT token (prefill 超时 ${TOKEN_QUEUE_TIMEOUT}s)"
    echo "  后端     : 127.0.0.1:$BACKEND_PORT  ($MODEL)"
    [ "$IMAGE_ENABLED" = "1" ] && echo "  图片     : qwen-image-2.1，127.0.0.1:${IMAGE_PORT}，按需加载"
    echo "  日志     : logs/glm.log  logs/proxy.log  logs/image.log"
    echo "按 Ctrl+C 停止。"
    break
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    show_failure "model_proxy" "$PROXY_PID" logs/proxy.log
    exit 1
  fi
  sleep 1
done

[ "$proxy_ready" -eq 1 ] || { echo "错误: 代理就绪检查超时；见 logs/proxy.log" >&2; exit 1; }

# 图片进程退出时保留聊天；代理或本次启动的 GLM 退出则结束监管。
while :; do
  LIVE_PIDS=()
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if [ "$pid" = "${IMAGE_PID:-}" ]; then
        echo "图片服务已退出，聊天继续运行；可用 bash start_image.sh 恢复图片接口。" >&2
        continue
      fi
      echo "错误: 服务进程 $pid 意外退出，请检查 logs/ 下的日志。" >&2
      exit 1
    fi
    LIVE_PIDS+=("$pid")
  done
  PIDS=("${LIVE_PIDS[@]}")
  sleep 2
done
