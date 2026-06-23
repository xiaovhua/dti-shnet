#!/usr/bin/env bash
set -euo pipefail

# Raw-data root. All templates below are interpreted relative to this directory. Example:
#   ROOT=/data/CamCAN
ROOT="${ROOT:-/path/to/raw_dataset}"

# Optional layout preset. Supported examples: camcan, ukb, flat.
# This only provides default file templates; it does not define any local path.
LAYOUT_PRESET="${LAYOUT_PRESET:-camcan}"

# Output template relative to ROOT. The default produces:
#   ROOT/preprocessed/<subject>/
OUT_DIR_TEMPLATE="${OUT_DIR_TEMPLATE:-preprocessed/{subject}}"

# Optional custom templates. Leave empty to use LAYOUT_PRESET templates. Each template is relative to ROOT and must contain {subject}.
SOURCE_NII_TEMPLATE="${SOURCE_NII_TEMPLATE:-}"
SOURCE_BVAL_TEMPLATE="${SOURCE_BVAL_TEMPLATE:-}"
SOURCE_BVEC_TEMPLATE="${SOURCE_BVEC_TEMPLATE:-}"
TARGET_NII_TEMPLATE="${TARGET_NII_TEMPLATE:-}"
TARGET_BVAL_TEMPLATE="${TARGET_BVAL_TEMPLATE:-}"
TARGET_BVEC_TEMPLATE="${TARGET_BVEC_TEMPLATE:-}"

# Optional subject list. Leave empty to discover subjects from templates.
SUBJECT_LIST="${SUBJECT_LIST:-}"

# Parallelism.
WORKERS="${WORKERS:-8}"

# Reproducibility parameters.
B_IN="${B_IN:-1000}"
B_OUT="${B_OUT:-2000}"
LMAX="${LMAX:-8}"
LAM="${LAM:-0.006}"
B_TOL="${B_TOL:-50}"
ST_CLIP_PCT="${ST_CLIP_PCT:-99.5}"

cmd=(
  python -m dti_shnet.preprocess
  --root "${ROOT}"
  --layout-preset "${LAYOUT_PRESET}"
  --out-dir-template "${OUT_DIR_TEMPLATE}"
  --workers "${WORKERS}"
  --b-in "${B_IN}"
  --b-out "${B_OUT}"
  --b-tol "${B_TOL}"
  --st-clip-pct "${ST_CLIP_PCT}"
  --lmax "${LMAX}"
  --lam "${LAM}"
)

if [ -n "${SOURCE_NII_TEMPLATE}" ]; then
  cmd+=(--source-nii-template "${SOURCE_NII_TEMPLATE}")
fi
if [ -n "${SOURCE_BVAL_TEMPLATE}" ]; then
  cmd+=(--source-bval-template "${SOURCE_BVAL_TEMPLATE}")
fi
if [ -n "${SOURCE_BVEC_TEMPLATE}" ]; then
  cmd+=(--source-bvec-template "${SOURCE_BVEC_TEMPLATE}")
fi
if [ -n "${TARGET_NII_TEMPLATE}" ]; then
  cmd+=(--target-nii-template "${TARGET_NII_TEMPLATE}")
fi
if [ -n "${TARGET_BVAL_TEMPLATE}" ]; then
  cmd+=(--target-bval-template "${TARGET_BVAL_TEMPLATE}")
fi
if [ -n "${TARGET_BVEC_TEMPLATE}" ]; then
  cmd+=(--target-bvec-template "${TARGET_BVEC_TEMPLATE}")
fi
if [ -n "${SUBJECT_LIST}" ]; then
  cmd+=(--subject-list "${SUBJECT_LIST}")
fi

echo "[config] ROOT=${ROOT}"
echo "[config] LAYOUT_PRESET=${LAYOUT_PRESET}"
echo "[config] OUT_DIR_TEMPLATE=${OUT_DIR_TEMPLATE}"
echo "[config] WORKERS=${WORKERS}"

"${cmd[@]}"