#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ROOT="${CONDA_ROOT:-/media/mldadmin/home/s125mdg38_06/miniconda3}"
CONDA_ENV="${CONDA_ENV:-streamvln}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

EPISODE_ID="${EPISODE_ID:-412}"           # use all for split evaluation
NUM_EPISODES="${NUM_EPISODES:-0}"         # only used with EPISODE_ID=all
SPLIT="${SPLIT:-val_unseen}"
MODEL="${MODEL:-}"
USE_API="${USE_API:-1}"
VISUALIZE="${VISUALIZE:-1}"
ALIGN_START_HEADING="${ALIGN_START_HEADING:-0}"
HISTORY_STRATEGY="${HISTORY_STRATEGY:-uniform}" # uniform/recent/hybrid
HISTORY_FRAMES="${HISTORY_FRAMES:-8}"
HFOV="${HFOV:-90}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_ACTIONS="${MAX_ACTIONS:-500}"
OUTPUT="${OUTPUT:-outputs/eval}"

bool_flag() {
  if [[ "$2" == "1" || "$2" == "true" ]]; then
    printf -- "--%s" "$1"
  else
    printf -- "--no-%s" "$1"
  fi
}

args=(
  --episode-id "$EPISODE_ID"
  --num-episodes "$NUM_EPISODES"
  --split "$SPLIT"
  --history-strategy "$HISTORY_STRATEGY"
  --history-frames "$HISTORY_FRAMES"
  --hfov "$HFOV"
  --max-turns "$MAX_TURNS"
  --max-actions "$MAX_ACTIONS"
  --output "$OUTPUT"
  "$(bool_flag use-api "$USE_API")"
  "$(bool_flag visualize "$VISUALIZE")"
  "$(bool_flag align-start-heading "$ALIGN_START_HEADING")"
)
if [[ -n "$MODEL" ]]; then
  args+=(--model "$MODEL")
fi

python eval.py "${args[@]}" "$@"
