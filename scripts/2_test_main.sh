#!/usr/bin/env bash
set -euo pipefail

PREPROC_ROOT=${PREPROC_ROOT:-/path/to/preprocessed}
CKPT=${CKPT:-./results/last.pt}
SAVE_DIR=${SAVE_DIR:-./results}
EVAL_ID_FILE=${EVAL_ID_FILE:-splits/eval_id.txt}

python -m dti_shnet.infer \
  --roots "['${PREPROC_ROOT}']" \
  --ckpt "${CKPT}" \
  --save-dir "${SAVE_DIR}" \
  --eval-id-file "${EVAL_ID_FILE}" \
  --roi 96,96,64 \
  --infer-patch 32,32,32 \
  --overlap 0.5 \
  --use-dti 1 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006
