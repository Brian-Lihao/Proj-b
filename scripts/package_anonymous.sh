#!/usr/bin/env bash
set -euo pipefail

# Build a clean archive of this repository for anonymous submission.
#
# The archive excludes version-control data, caches, logs, generated images,
# model weights, and W&B runs. Symbolic links are rejected because they can
# leak absolute paths from the machine that produced them.
#
# The archive format follows the OUTPUT_ARCHIVE extension: .zip produces a ZIP
# file, anything else produces a gzipped tarball.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ARCHIVE="${OUTPUT_ARCHIVE:-${TMPDIR:-/tmp}/ppr-anonymous.tar.gz}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-ppr}"

cd "$ROOT_DIR"

mapfile -t symlinks < <(find . -type l -printf '%p -> %l\n')
if ((${#symlinks[@]} > 0)); then
  echo "ERROR: Refusing to package symbolic links:" >&2
  printf '  %s\n' "${symlinks[@]}" >&2
  exit 1
fi

SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
HOME_DIRS='(/hom''e/|/roo''t/|/User''s/)'
mapfile -t leaks < <(grep -rIlE "$HOME_DIRS" \
  --exclude-dir=.git --exclude-dir=__pycache__ --exclude='*.json' \
  --exclude="$SELF_NAME" . || true)
if ((${#leaks[@]} > 0)); then
  echo "ERROR: Absolute machine paths found in:" >&2
  printf '  %s\n' "${leaks[@]}" >&2
  exit 1
fi

EXCLUDE_GLOBS=(
  '__pycache__'
  '*.py[cod]'
  'wandb'
  'outputs'
  '*.log'
  '*.png'
  '*.jpg'
  '*.bin'
  '*.pt'
  '*.pth'
  '*.ckpt'
  '*.safetensors'
)

rm -f "$OUTPUT_ARCHIVE"

if [[ "$OUTPUT_ARCHIVE" == *.zip ]]; then
  if ! command -v zip >/dev/null 2>&1; then
    echo "ERROR: zip is required to build a .zip archive." >&2
    exit 1
  fi

  STAGING_DIR="$(mktemp -d)"
  trap 'rm -rf "$STAGING_DIR"' EXIT

  TAR_EXCLUDES=(--exclude-vcs)
  for pattern in "${EXCLUDE_GLOBS[@]}"; do
    TAR_EXCLUDES+=(--exclude="$pattern")
  done
  mkdir -p "$STAGING_DIR/$ARCHIVE_ROOT"
  tar --create "${TAR_EXCLUDES[@]}" --file - . \
    | tar --extract --file - --directory "$STAGING_DIR/$ARCHIVE_ROOT"

  (cd "$STAGING_DIR" && zip --recurse-paths --quiet "$OUTPUT_ARCHIVE" "$ARCHIVE_ROOT")

  ARCHIVE_ENTRIES="$(unzip -Z1 "$OUTPUT_ARCHIVE")"
  echo "Created archive: $OUTPUT_ARCHIVE"
  printf '%s\n' "$ARCHIVE_ENTRIES" | head -n 20
  echo "Total entries: $(printf '%s\n' "$ARCHIVE_ENTRIES" | wc -l)"
else
  TAR_EXCLUDES=(--exclude-vcs)
  for pattern in "${EXCLUDE_GLOBS[@]}"; do
    TAR_EXCLUDES+=(--exclude="$pattern")
  done

  tar --create --gzip \
    --file "$OUTPUT_ARCHIVE" \
    "${TAR_EXCLUDES[@]}" \
    --transform "s,^\\.,$ARCHIVE_ROOT," \
    .

  ARCHIVE_ENTRIES="$(tar --list --file "$OUTPUT_ARCHIVE")"
  echo "Created archive: $OUTPUT_ARCHIVE"
  printf '%s\n' "$ARCHIVE_ENTRIES" | head -n 20
  echo "Total entries: $(printf '%s\n' "$ARCHIVE_ENTRIES" | wc -l)"
fi
