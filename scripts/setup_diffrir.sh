#!/usr/bin/env bash
set -eu
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
REPO="${MTM_DIFFRIR_REPO:-$PKG/data/hearinganythinganywhere}"
COMMIT="9fb146487a82b9cbcf6334320ff5794e688d4a03"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/maswang32/hearinganythinganywhere.git "$REPO"
fi
git -C "$REPO" checkout "$COMMIT"
MTM_DIFFRIR_REPO="$REPO" "$PY" "$PKG/code/diffrir_config.py"
