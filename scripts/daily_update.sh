#!/usr/bin/env bash
#
# Daily refresh wrapper — safe to run from cron or by hand.
#
#   ./scripts/daily_update.sh              # refresh only (you commit manually)
#   AUTO_COMMIT=1 ./scripts/daily_update.sh   # refresh, then commit & push
#
# Pass extra flags straight through, e.g.:
#   ./scripts/daily_update.sh --retrain
#
set -euo pipefail

# Move to the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

# Activate the project virtualenv if present.
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[daily_update] $(date '+%Y-%m-%d %H:%M:%S') starting"
python src/pipeline/run_daily_update.py "$@"

if [[ "${AUTO_COMMIT:-0}" == "1" ]]; then
  echo "[daily_update] committing and pushing"
  git add -A
  if git commit -m "Daily prediction refresh $(date +%F)"; then
    git push
    echo "[daily_update] pushed — Streamlit Cloud will redeploy shortly"
  else
    echo "[daily_update] nothing to commit"
  fi
fi

echo "[daily_update] done"
