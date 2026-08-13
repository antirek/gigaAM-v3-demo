#!/usr/bin/env bash
set -euo pipefail

MIN_FREE_GB=5
MODEL_SIZE_MB=500
IMAGE_SIZE_MB=2000
TOTAL_NEED_MB=$((MODEL_SIZE_MB + IMAGE_SIZE_MB))

echo "=== Disk space check for GigaAM project ==="
echo ""

df -h / | tail -1 | awk '{printf "Root filesystem: %s used, %s free (%s)\n", $3, $4, $5}'

AVAIL_KB=$(df / | tail -1 | awk '{print $4}')
AVAIL_GB=$(awk "BEGIN {printf \"%.1f\", $AVAIL_KB/1024/1024}")

echo "Free space: ${AVAIL_GB} GB"
echo "Estimated need: ~$((TOTAL_NEED_MB / 1024)) GB (model ~${MODEL_SIZE_MB}MB + Docker image ~${IMAGE_SIZE_MB}MB)"
echo ""

if command -v docker >/dev/null 2>&1; then
  echo "=== Docker disk usage ==="
  docker system df
  RECLAIMABLE=$(docker system df 2>/dev/null | awk '/Images/ {print $4}')
  echo "Reclaimable images: ${RECLAIMABLE:-unknown}"
  echo "Tip: docker image prune -a  # remove unused images"
  echo ""
fi

PROJECT_DATA="$(cd "$(dirname "$0")/.." && pwd)/data/gigaam"
if [ -d "$PROJECT_DATA" ]; then
  DATA_SIZE=$(du -sh "$PROJECT_DATA" 2>/dev/null | awk '{print $1}')
  echo "Project model cache ($PROJECT_DATA): ${DATA_SIZE:-0}"
else
  echo "Project model cache: not created yet (will use ./data/gigaam)"
fi

echo ""
if awk "BEGIN {exit !($AVAIL_KB < $MIN_FREE_GB * 1024 * 1024)}"; then
  echo "WARNING: Less than ${MIN_FREE_GB}GB free. Build/download may fail."
  echo "Consider: docker image prune -a"
  exit 1
fi

if awk "BEGIN {exit !($AVAIL_KB < $TOTAL_NEED_MB * 1024)}"; then
  echo "WARNING: Free space may be insufficient for first build + model download."
  exit 1
fi

echo "OK: Enough disk space for first run."
