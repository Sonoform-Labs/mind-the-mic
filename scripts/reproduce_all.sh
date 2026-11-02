#!/usr/bin/env bash
set -eu
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
run() { "$PY" "$PKG/code/run.py" "$@"; }
run m_within_column m_cluster_boot m_pathdiff m_controls m_snr m_grid m_df_bias_check
run m_diffrir_final m_backup_controls m_floor_verify m_reproduce
run m_matched_displacement
bash "$PKG/scripts/render_diffrir.sh"
run m_perposition m_env_perposition m_env_band
run m_figures
