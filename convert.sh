#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

SRC=${SRC:-/Volumes/model/GLM-5.3-Flash-FineTunning}
DST=${DST:-$HOME/models/$(basename "$SRC")-MXFP8}
PROCESSOR_SOURCE=${PROCESSOR_SOURCE:-/Volumes/model/GLM-5.3-Flash}
PYTHON=${PYTHON:-$HOME/.venvs/mlx-vlm/bin/python}

[ -d "$SRC" ] || { echo "源模型不存在: $SRC" >&2; exit 1; }
[ -x "$PYTHON" ] || { echo "venv 不存在: $PYTHON" >&2; exit 1; }
for target in "$DST" "$DST-mtp"; do
  [ ! -e "$target" ] || { echo "输出已存在，拒绝覆盖: $target（请指定新的 DST）" >&2; exit 1; }
done

echo "源: $SRC"
echo "出: $DST"
echo "日志: logs/convert.log"

mkdir -p logs
"$PYTHON" prepare_checkpoint.py "$SRC" --processor-source "$PROCESSOR_SOURCE"
"$PYTHON" -u convert_cpu.py \
  --hf-path "$SRC" \
  --mlx-path "$DST" \
  --mtp 2>&1 | tee -a logs/convert.log

"$PYTHON" finalize_mxfp8.py "$SRC" "$DST" 2>&1 | tee -a logs/convert.log
"$PYTHON" finalize_mxfp8.py "$SRC" "$DST-mtp" 2>&1 | tee -a logs/convert.log
