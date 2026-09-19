#!/usr/bin/env bash
# Run the GLiNER2.5 backend on the Spark and commit the result. No API key of any kind is read
# on this path -- the model weights are pulled once (anonymously) from Hugging Face into the
# local HF cache; every subsequent run is offline.
#
# Usage (on the Spark):
#   cd ~/projects/jev-email-cascade && git pull && scripts/spark-gliner-run.sh
#   DRY_RUN=1 scripts/spark-gliner-run.sh   # print every command, do nothing
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFLIGHT_LOG="runs/gliner-preflight-${STAMP}.txt"

run() {
  echo "+ $*"
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

echo "=== step 0: sync ==="
run git pull
run uv sync --extra gliner

echo "=== step 1: preflight (device the run will actually use) ==="
if [ "$DRY_RUN" = "1" ]; then
  echo "+ (preflight probe, output would go to ${PREFLIGHT_LOG})"
  DEVICE=cpu
else
  mkdir -p runs
  {
    uname -m
    nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null || echo "no nvidia-smi"
    uv run python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"
  } | tee "$PREFLIGHT_LOG"
  if grep -q "True" "$PREFLIGHT_LOG"; then
    DEVICE=cuda
  else
    echo "CUDA not available to this torch build; continuing on CPU (still a valid, slower row)"
    DEVICE=cpu
  fi
fi
echo "device selected: $DEVICE"

FP16=0
[ "$DEVICE" = "cuda" ] && FP16=1

echo "=== step 2: smoke test (--limit 3) ==="
run env GLINER_DEVICE="$DEVICE" GLINER_FP16="$FP16" uv run cascade run --backend gliner --limit 3

echo "=== step 3: full run (74 emails) ==="
run env GLINER_DEVICE="$DEVICE" GLINER_FP16="$FP16" uv run cascade run --backend gliner

if [ "$DRY_RUN" = "1" ]; then
  echo "=== dry run: stopping before commit ==="
  exit 0
fi

NEW_RUN="$(ls -td runs/*/ | head -1)"
NEW_RUN="${NEW_RUN%/}"
echo "=== step 4: report ==="
uv run cascade report "$NEW_RUN" --markdown > "$NEW_RUN/report.md"

echo "=== step 5: commit and push ==="
git add runs/
git -c user.name="Kaushik Ghosh" -c user.email="kaushikd12@gmail.com" \
  commit -m "Record the GLiNER2.5 run on all 74 labelled emails ($DEVICE)"
git push

echo "done: $NEW_RUN"
