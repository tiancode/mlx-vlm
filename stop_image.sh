#!/bin/bash
# 只停止本工程记录的图片监管进程，不匹配其它 mlx-vlm 服务。
set -u
cd "$(dirname "$0")"
pid_file=logs/image-runner.pid
[ -f "$pid_file" ] || { echo "图片: 未记录运行进程"; exit 0; }
image_pid=$(cat "$pid_file")
case "$image_pid" in
  ''|*[!0-9]*) echo "图片 PID 文件无效" >&2; exit 1 ;;
esac
image_alive() {
  local image_state
  image_state=$(ps -p "$image_pid" -o stat= 2>/dev/null || true)
  # A caller may own the runner and reap it only after this script returns.
  # kill -0 also succeeds for that already-exited zombie process.
  [ -n "$image_state" ] && [[ "$image_state" != *Z* ]]
}
command_line=$(ps -p "$image_pid" -o command= 2>/dev/null || true)
if ! image_alive; then
  rm -f "$pid_file"
  echo "图片: 未运行"
  exit 0
fi
if ! printf '%s\n' "$command_line" | rg -q 'service_runner[.]py .* image_server[.]py$'; then
  echo "图片 PID 已被其它进程复用，未发送信号: $image_pid" >&2
  exit 1
fi
kill "$image_pid"
# service_runner 在 30 秒后兜底停止自己的子进程组。
for _ in $(seq 1 40); do
  if ! image_alive; then
    rm -f "$pid_file"
    echo "图片: 已停止"
    exit 0
  fi
  sleep 1
done
echo "图片: 停止超时，请检查 logs/image.log" >&2
exit 1
