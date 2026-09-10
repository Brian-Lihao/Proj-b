#!/usr/bin/env bash
set -euo pipefail

# PPR-Linear tmux launcher.
#
# Replace the uppercase placeholder paths below, or pass the same values as
# environment variables. The tmux session keeps an interactive shell, so
# pressing Ctrl-C stops training without destroying the session.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/PATH/TO/CONDA/ENV/BIN/PYTHON}"
PPR_CACHE_DIR="${PPR_CACHE_DIR:-/PATH/TO/HUGGINGFACE_CACHE}"
PPR_OUTPUT_DIR="${PPR_OUTPUT_DIR:-/PATH/TO/PPR_OUTPUTS}"
LOG_FILE="${LOG_FILE:-/PATH/TO/LOGS/PPR_LINEAR_TRAINING.LOG}"
SESSION_NAME="${SESSION_NAME:-ppr-linear}"
METHOD="${METHOD:-ppr_linear}"
RUN_NAME="${RUN_NAME:-PPR-Linear}"
WANDB_MODE="${WANDB_MODE:-offline}"
RESUME_FROM="${RESUME_FROM:-none}"
OFFLINE_MODELS="${OFFLINE_MODELS:-1}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_linear_tmux.sh [--help]

Environment variables:
  PYTHON_BIN      Python executable of the training environment.
  PPR_CACHE_DIR   Hugging Face cache containing the required models.
  PPR_OUTPUT_DIR  Root directory for run outputs and checkpoints.
  LOG_FILE        Training log file.
  SESSION_NAME    tmux session name (default: ppr-linear).
  METHOD          Training method (default: ppr_linear).
  RUN_NAME        Run directory name (default: PPR-Linear).
  WANDB_MODE      offline (default) | online | disabled.
                  online requires a prior `wandb login`.
  RESUME_FROM     none (default) | auto | /PATH/TO/checkpoint_<EPOCH>
                  auto resumes from the newest checkpoint in the run directory.
                  Resume granularity is one completed epoch: an interrupted
                  epoch is replayed from its start.
  OFFLINE_MODELS  1 (default) forces offline Hugging Face loading; 0 allows downloads.

Examples:
  bash scripts/train_linear_tmux.sh
  WANDB_MODE=online bash scripts/train_linear_tmux.sh
  RESUME_FROM=auto bash scripts/train_linear_tmux.sh
  RESUME_FROM=/PATH/TO/PPR_OUTPUTS/PPR-Linear/checkpoint_3 bash scripts/train_linear_tmux.sh

Session control:
  tmux attach -t SESSION_NAME        # watch training
  Ctrl-C                             # stop training, keep the session
  Ctrl-b d                           # detach, keep training
  tmux kill-session -t SESSION_NAME  # remove the session
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

case "$WANDB_MODE" in
  offline|online|disabled) ;;
  *)
    echo "ERROR: WANDB_MODE must be offline, online, or disabled (got: $WANDB_MODE)" >&2
    exit 1
    ;;
esac

for required_path in "$PYTHON_BIN" "$PPR_CACHE_DIR" "$PPR_OUTPUT_DIR" "$LOG_FILE"; do
  if [[ "$required_path" == /PATH/TO/* ]]; then
    echo "ERROR: Replace all /PATH/TO/... placeholders before launching." >&2
    echo "Run 'bash scripts/train_linear_tmux.sh --help' for details." >&2
    exit 1
  fi
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed or is not available on PATH." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$PPR_CACHE_DIR" ]]; then
  echo "ERROR: Hugging Face cache not found: $PPR_CACHE_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/model_ckpts/sd-v1-5_step-aware_preference_model.bin" ]]; then
  echo "ERROR: Step-aware preference checkpoint is missing." >&2
  exit 1
fi
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

ENV_BIN="$(dirname "$PYTHON_BIN")"
mkdir -p "$PPR_OUTPUT_DIR" "$(dirname "$LOG_FILE")"

# Resumed runs append to the existing log; fresh runs start a new one.
TEE_FLAGS=""
if [[ "$RESUME_FROM" != "none" && "$RESUME_FROM" != "" ]]; then
  TEE_FLAGS="-a"
fi

OFFLINE_EXPORTS=""
if [[ "$OFFLINE_MODELS" == "1" ]]; then
  OFFLINE_EXPORTS='export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; '
fi

printf -v TRAIN_COMMAND \
  '%sexport PATH=%q:"$PATH" PPR_CACHE_DIR=%q PPR_OUTPUT_DIR=%q TOKENIZERS_PARALLELISM=false WANDB_MODE=%q METHOD=%q RUN_NAME=%q RESUME_FROM=%q; unset TRANSFORMERS_CACHE; set -o pipefail; bash scripts/train.sh 2>&1 | tee %s %q; echo "Training exited with status ${PIPESTATUS[0]}."' \
  "$OFFLINE_EXPORTS" "$ENV_BIN" "$PPR_CACHE_DIR" "$PPR_OUTPUT_DIR" "$WANDB_MODE" \
  "$METHOD" "$RUN_NAME" "$RESUME_FROM" "$TEE_FLAGS" "$LOG_FILE"

# Start an interactive shell first, then send the training command into it so
# that Ctrl-C only interrupts training and the session stays alive.
tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR"
tmux set-option -t "=$SESSION_NAME" remain-on-exit on
tmux send-keys -t "=$SESSION_NAME" "$TRAIN_COMMAND" C-m

echo "Started $RUN_NAME ($METHOD) in tmux session: $SESSION_NAME"
echo "W&B mode:  $WANDB_MODE"
echo "Resume:    $RESUME_FROM"
echo "Attach:    tmux attach -t $SESSION_NAME"
echo "Log:       $LOG_FILE"
echo "Output:    $PPR_OUTPUT_DIR/$RUN_NAME"
echo "Stop training but keep the session: press Ctrl-C inside tmux"
echo "Remove the session: tmux kill-session -t $SESSION_NAME"
