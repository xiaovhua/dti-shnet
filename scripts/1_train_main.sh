#!/usr/bin/env bash
set -euo pipefail

PREPROC_ROOT=${PREPROC_ROOT:-/path/to/preprocessed}
OUT_DIR=${OUT_DIR:-./results}
EVAL_ID_FILE=${EVAL_ID_FILE:-splits/eval_id.txt}
DTI=${DTI:-1}
SIG=${SIG:-1}

python -m dti_shnet.train \
  --roots "['${PREPROC_ROOT}']" \
  --out-dir "${OUT_DIR}" \
  --eval-id-file "${EVAL_ID_FILE}" \
  --use-dti "${DTI}" \
  --use-sig-loss "${SIG}" \
  --lambda-sig 0.1 \
  --roi 96,96,64 \
  --patch 32,32,32 \
  --dim 64 \
  --batch 4 \
  --epochs 30 \
  --lr 2e-4 \
  --num-workers 8 \
  --pps-train 12 \
  --pps-val 1 \
  --shuffle-patches 0 \
  --seed-per-epoch 1 \
  --amp 1 \
  --seed 0 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006 \
  --canon-n 60
