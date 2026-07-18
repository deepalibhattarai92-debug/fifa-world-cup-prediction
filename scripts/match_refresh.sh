#!/usr/bin/env bash
#
# Match-aware refresh — safe for cron / GitHub Actions.
#
#   ./scripts/match_refresh.sh              # refresh only when a match window is active
#   AUTO_COMMIT=1 ./scripts/match_refresh.sh  # also commit & push when files change
#   ./scripts/match_refresh.sh --force      # bypass schedule (manual catch-up)
#   ./scripts/match_refresh.sh --status     # show upcoming watch windows
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[match_refresh] $(date '+%Y-%m-%d %H:%M:%S') starting"
python src/pipeline/run_match_refresh.py "$@"

if [[ "${AUTO_COMMIT:-0}" == "1" ]]; then
  echo "[match_refresh] committing and pushing if changed"
  git add -A
  if git commit -m "Match-time prediction refresh $(date +%F-%H%M)"; then
    git push
    echo "[match_refresh] pushed — Streamlit Cloud will redeploy shortly"
  else
    echo "[match_refresh] nothing to commit"
  fi
fi

echo "[match_refresh] done"
