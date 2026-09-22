#!/bin/bash
# 用法: ./smoke.sh [base_url] [api_key]；任一检查失败即返回非零。
set -eu
PYTHON=${PYTHON:-$HOME/.venvs/mlx-vlm/bin/python}
exec "$PYTHON" "$(dirname "$0")/smoke.py" "$@"
