#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHOD="${METHOD:-ppr_linear}"
RUN_NAME="${RUN_NAME:-PPR-Linear}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_cfg/multi_gpu_fp16.yaml}"
RESUME_FROM="${RESUME_FROM:-none}"
RUN_DIR="${PPR_OUTPUT_DIR:-outputs}/$RUN_NAME"

resolve_latest_checkpoint() {
  local run_dir="$1"
  local latest
  latest="$({
    find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint_[0-9]*' \
      -printf '%f\n' 2>/dev/null || true
  } | awk -F_ '$2 ~ /^[0-9]+$/ {print $2 "\t" $0}' | sort -n | tail -n 1 | cut -f2-)"
  [[ -n "$latest" ]] || return 1
  printf '%s/%s\n' "$run_dir" "$latest"
}

RESUME_ARG=""
case "$RESUME_FROM" in
  none|"")
    if [[ -d "$RUN_DIR" ]] && {
      find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint_[0-9]*' -print -quit | grep -q . \
        || [[ -f "$RUN_DIR/pytorch_lora_weights.safetensors" ]];
    }; then
      echo "ERROR: Existing training artifacts found in $RUN_DIR" >&2
      echo "Use RESUME_FROM=auto, provide an explicit checkpoint path, or choose another RUN_NAME/PPR_OUTPUT_DIR." >&2
      exit 1
    fi
    ;;
  auto)
    if ! RESUME_ARG="$(resolve_latest_checkpoint "$RUN_DIR")"; then
      echo "ERROR: No checkpoint_<EPOCH> directory found in $RUN_DIR" >&2
      exit 1
    fi
    ;;
  *)
    RESUME_ARG="$RESUME_FROM"
    if [[ ! -d "$RESUME_ARG" ]]; then
      echo "ERROR: Resume path is not a directory: $RESUME_ARG" >&2
      exit 1
    fi
    ;;
esac

TRAIN_ARGS=(
  --config configs/ppr_sd15.py
  --config.train.method="$METHOD"
  --config.run_name="$RUN_NAME"
)
if [[ -n "$RESUME_ARG" ]]; then
  TRAIN_ARGS+=(--config.resume_from="$RESUME_ARG")
  echo "Resuming from: $RESUME_ARG"
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  train_ppr.py \
  "${TRAIN_ARGS[@]}" \
  "$@"
