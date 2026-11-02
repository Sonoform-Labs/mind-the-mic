#!/usr/bin/env bash
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
REPO="${MTM_DIFFRIR_REPO:-$PKG/data/hearinganythinganywhere}"
[ -d "$REPO" ] || { echo "DiffRIR clone not found at $REPO; run scripts/setup_diffrir.sh first"; exit 1; }
REPO="$(cd "$REPO" && pwd)"
ROOMS="${MTM_ROOMS:-classroomBase dampenedBase hallwayBase complexBase}"
for room in $ROOMS; do
  if [ -f "$REPO/models/$room/errors/errors_multiscale_log_l1.npy" ]; then
    echo "skip $room (already rendered)"
    continue
  fi
  echo "render $room"
  (cd "$REPO" && "$PY" -u train.py "models/$room" "$room" "precomputed/$room" \
      --skip_train --skip_music --skip_binaural)
  echo "$room exit $?"
done
