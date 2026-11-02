#!/usr/bin/env bash
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MTM_DIFFRIR_DIR:-$PKG/data/diffrir}"
BASE="https://zenodo.org/records/11195833/files"
ROOMS="${MTM_ROOMS:-classroomBase dampenedBase hallwayBase complexBase}"
mkdir -p "$DEST"
for r in $ROOMS; do
  if [ -f "$DEST/$r/RIRs.npy" ]; then
    echo "skip $r"
    continue
  fi
  echo "get $r"
  if ! curl -L -f -C - -o "$DEST/$r.zip" "$BASE/$r.zip"; then
    echo "failed $r"
    continue
  fi
  unzip -q -o "$DEST/$r.zip" -d "$DEST" && rm -f "$DEST/$r.zip"
done
for r in $ROOMS; do
  if [ -f "$DEST/$r/RIRs.npy" ]; then echo "ready $r"; else echo "missing $r"; fi
done
