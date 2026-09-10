#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHODS=(
  spo
  ppr_linear
  ppr_mid
  ppr_hard
  direct_lpips
  snr_lpips
)
RUN_NAMES=(
  SPO
  PPR-Linear
  PPR-Mid
  PPR-Hard
  Direct-LPIPS
  SNR-LPIPS
)

for index in "${!METHODS[@]}"; do
  echo "Training ${RUN_NAMES[$index]} (${METHODS[$index]})"
  METHOD="${METHODS[$index]}" \
  RUN_NAME="${RUN_NAMES[$index]}" \
    bash scripts/train.sh "$@"
done
