#!/bin/bash
set -u
cd "$(dirname "$0")"

PORT=${PORT:-1235}
BACKEND_PORT=${BACKEND_PORT:-1236}
IMAGE_PORT=${IMAGE_PORT:-1238}
PYTHON=${PYTHON:-$HOME/.venvs/mlx-vlm/bin/python}
failed=0

live_matching() {
  local candidate state
  for candidate in $(pgrep -f "$1" || true); do
    state=$(ps -p "$candidate" -o stat= 2>/dev/null || true)
    [ -n "$state" ] && [[ "$state" != *Z* ]] && echo "$candidate"
  done
}

stop_matching() { # $1=名称 $2=pgrep 模式
  pids=$(live_matching "$2")
  [ -z "$pids" ] && { echo "$1: 未运行"; return 0; }
  echo "$1: 停止 $(echo "$pids" | tr '\n' ' ')"
  kill $pids 2>/dev/null
  for _ in $(seq 1 120); do
    pids=$(live_matching "$2")
    [ -z "$pids" ] && { echo "$1: 已退出"; return 0; }
    sleep 1
  done
  echo "$1: 120 秒未退出，强制结束" >&2
  kill -9 $pids 2>/dev/null || return 1
  sleep 1
  [ -z "$(live_matching "$2")" ]
}

if [ -f logs/start.pid ]; then
  supervisor=$(cat logs/start.pid)
  case "$supervisor" in
    ''|*[!0-9]*) ;;
    *)
      supervisor_command=$(ps -p "$supervisor" -o command= 2>/dev/null || true)
      if [[ "$supervisor_command" = *"bash ./start.sh" ||
            "$supervisor_command" = *"bash start.sh" ||
            "$supervisor_command" = *"bash $PWD/start.sh" ]]; then
        kill "$supervisor" 2>/dev/null
        for _ in $(seq 1 40); do
          state=$(ps -p "$supervisor" -o stat= 2>/dev/null || true)
          if [ -z "$state" ] || [[ "$state" = *Z* ]]; then break; fi
          sleep 1
        done
      fi ;;
  esac
fi
# 新启动的代理使用绝对脚本路径；旧版相对路径代理由记录的 supervisor 关闭。
proxy_pattern=$("$PYTHON" -c 'import re,sys; print("(^| )" + re.escape(sys.argv[1]) + "( |$)")' "$PWD/model_proxy.py") || exit 1
stop_matching "代理" "$proxy_pattern" || failed=1
stop_matching "后端" "server_launch[.]py .*--port ${BACKEND_PORT}( |$)" || failed=1
bash ./stop_image.sh || failed=1

for p in "$PORT" "$BACKEND_PORT" "$IMAGE_PORT"; do
  if "$PYTHON" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$p"; then
    echo "警告: 端口 $p 仍被占用" >&2
    failed=1
  fi
done
if [ "$failed" -eq 0 ]; then
  rm -f logs/start.pid
  echo "全部停止。"
else
  echo "部分服务未能确认停止，请检查上述输出。" >&2
fi
exit "$failed"
