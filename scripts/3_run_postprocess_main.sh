#!/usr/bin/env bash
set -euo pipefail

# Dataset layout preset. This controls how original DWI/bval/bvec files are located under ORIG_ROOT.
# Supported examples: camcan, ukb
LAYOUT="${LAYOUT:-camcan}"

# User-defined paths. Override them from command line, e.g.:
#   ORIG_ROOT=/data/CamCAN PREPROC_ROOT=/data/CamCAN/preprocessed RUN_DIR=./results
ORIG_ROOT="${ORIG_ROOT:-/path/to/original_data}"
PREPROC_ROOT="${PREPROC_ROOT:-/path/to/preprocessed_data}"
RUN_DIR="${RUN_DIR:-/path/to/results}"

# Derived output paths.
PRED_ROOT="${PRED_ROOT:-${RUN_DIR}/dti_shnet_pred}"
EXPORT_ROOT="${EXPORT_ROOT:-${RUN_DIR}/export_dMRI}"
DTI_OUT="${DTI_OUT:-${RUN_DIR}/dti}"
NODDI_OUT="${NODDI_OUT:-${RUN_DIR}/noddi}"

# Reproducibility parameters.
B_IN="${B_IN:-1000}"
B_OUT="${B_OUT:-2000}"
LMAX="${LMAX:-8}"
LAM="${LAM:-0.006}"
TARGET_B="${TARGET_B:-2000}"
B_TOL="${B_TOL:-50}"
ST_CLIP_PCT="${ST_CLIP_PCT:-99.5}"

# Parallel settings.
VIS_WORKERS="${VIS_WORKERS:-4}"
EXPORT_WORKERS="${EXPORT_WORKERS:-4}"
DTI_WORKERS="${DTI_WORKERS:-4}"

# NODDI is slow and requires dmri-amico + MRtrix.
RUN_NODDI="${RUN_NODDI:-0}"
NODDI_WORKERS="${NODDI_WORKERS:-1}"
NODDI_THREADS="${NODDI_THREADS:-8}"

echo "[config] LAYOUT=${LAYOUT}"
echo "[config] ORIG_ROOT=${ORIG_ROOT}"
echo "[config] PREPROC_ROOT=${PREPROC_ROOT}"
echo "[config] RUN_DIR=${RUN_DIR}"
echo "[config] PRED_ROOT=${PRED_ROOT}"
echo "[config] EXPORT_ROOT=${EXPORT_ROOT}"
echo "[config] DTI_OUT=${DTI_OUT}"
echo "[config] NODDI_OUT=${NODDI_OUT}"

python -m dti_shnet.evaluation.visual \
  --pred-root "${PRED_ROOT}" \
  --preproc-root "${PREPROC_ROOT}" \
  --out-csv "${RUN_DIR}/visual_metrics.csv" \
  --b-in "${B_IN}" \
  --b-out "${B_OUT}" \
  --lmax "${LMAX}" \
  --lam "${LAM}" \
  --num-workers "${VIS_WORKERS}"

python -m dti_shnet.export_dwi \
  --layout-preset "${LAYOUT}" \
  --orig-root "${ORIG_ROOT}" \
  --preproc-root "${PREPROC_ROOT}" \
  --pred-root "${PRED_ROOT}" \
  --out-root "${EXPORT_ROOT}" \
  --target-b "${TARGET_B}" \
  --b-tol "${B_TOL}" \
  --st-clip-pct "${ST_CLIP_PCT}" \
  --num-workers "${EXPORT_WORKERS}" \
  --executor process \
  --skip-existing

python -m dti_shnet.evaluation.dti \
  --layout-preset "${LAYOUT}" \
  --orig-root "${ORIG_ROOT}" \
  --preproc-root "${PREPROC_ROOT}" \
  --pred-root "${PRED_ROOT}" \
  --out-dir "${DTI_OUT}" \
  --target-b "${TARGET_B}" \
  --b-tol "${B_TOL}" \
  --st-clip-pct "${ST_CLIP_PCT}" \
  --mask-source mrtrix \
  --mrtrix-clean-scale 2 \
  --mrtrix-nthreads 0 \
  --mrtrix-force 0 \
  --dti-use-all-dw 1 \
  --b1 "${B_IN}" \
  --num-workers "${DTI_WORKERS}"

if [ "${RUN_NODDI}" = "1" ]; then
  python -m dti_shnet.evaluation.noddi \
    --layout-preset "${LAYOUT}" \
    --orig-root "${ORIG_ROOT}" \
    --preproc-root "${PREPROC_ROOT}" \
    --pred-root "${PRED_ROOT}" \
    --out-dir "${NODDI_OUT}" \
    --target-b "${TARGET_B}" \
    --b-tol "${B_TOL}" \
    --st-clip-pct "${ST_CLIP_PCT}" \
    --mask-source mrtrix \
    --mrtrix-clean-scale 2 \
    --mrtrix-nthreads 0 \
    --mrtrix-force 0 \
    --nb-threads "${NODDI_THREADS}" \
    --num-workers "${NODDI_WORKERS}"
fi