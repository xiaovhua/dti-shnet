# DTI-SHNet

Official implementation of **DTI-Guided Volumetric Spherical Harmonics Regression for Single-to-Multi-Shell dMRI Synthesis**.

DTI-SHNet synthesizes an unobserved high-b shell from a source single-shell dMRI acquisition. The model learns in the real symmetric spherical harmonics (SH) coefficient domain and reconstructs the predicted coefficients back to target-shell diffusion-weighted signals.

## Overview

```text
Raw dMRI
  ├─ median-b0 normalization
  ├─ source-shell DTI fitting
  │    └─ FA / MD / preprocessing brain mask
  ├─ SH fitting
  │    └─ source and target SH coefficients
  ├─ DTI-SHNet
  │    └─ source SH + FA + MD + mask → target SH
  └─ SH reconstruction
       └─ predicted target-shell signal
```

For the main `lmax=8` setting, the SH coefficient dimension is `K=45`. The final model uses 48 input channels:

```text
45 source SH channels + FA + MD + mask → 45 target SH channels
```

## Repository structure

```text
.
├── dti_shnet/
│   ├── constants.py
│   ├── preprocess.py
│   ├── prepare_shmeta.py
│   ├── dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── export_dwi.py
│   ├── io.py
│   ├── protocol.py
│   ├── sh.py
│   ├── spatial.py
│   ├── models/
│   │   └── unet3d.py
│   └── evaluation/
│       ├── _postprocess_common.py
│       ├── visual.py
│       ├── dti.py
│       └── noddi.py
├── scripts/
│   ├── 0_prepare.sh
│   ├── 1_train_main.sh
│   ├── 2_test_main.sh
│   └── 3_run_postprocess_main.sh
├── splits/
│   └── eval_id.txt
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md
```

## Environment

```bash
conda env create -f environment.yml
conda activate dti-shnet
pip install -e .
```

For DTI and NODDI postprocessing, install MRtrix3 and make sure `dwi2mask` is available on `PATH`.

For NODDI fitting, install AMICO:

```bash
pip install dmri-amico dipy
```

## Data layout and preprocessing

The preprocessing entry point is template-based. Each path template is relative to `--root` and must contain `{subject}`.

### Cam-CAN-style preset

```bash
python -m dti_shnet.preprocess \
  --root /path/to/CamCAN_root \
  --layout-preset camcan \
  --workers 8 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006
```

The preset expands to:

```text
{subject}/Preprocessed_data/dwi_preprocessed_single.nii.gz
{subject}/Preprocessed_data/dwi_preprocessed_single.bval
{subject}/Preprocessed_data/dwi_preprocessed_single.bvec
{subject}/Preprocessed_data/dwi_preprocessed.nii.gz
{subject}/Preprocessed_data/dwi_preprocessed.bval
{subject}/Preprocessed_data/dwi_preprocessed.bvec
```

### UKB-style preset

```bash
python -m dti_shnet.preprocess \
  --root /path/to/UKB_root \
  --layout-preset ukb \
  --workers 8 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006
```

### Custom layout

```bash
python -m dti_shnet.preprocess \
  --root /path/to/root \
  --source-nii-template  "{subject}/source.nii.gz" \
  --source-bval-template "{subject}/source.bval" \
  --source-bvec-template "{subject}/source.bvec" \
  --target-nii-template  "{subject}/target.nii.gz" \
  --target-bval-template "{subject}/target.bval" \
  --target-bvec-template "{subject}/target.bvec" \
  --out-dir-template "preprocessed/{subject}" \
  --workers 8 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006
```

### Preprocessing outputs

Each preprocessed subject folder contains:

| File | Description |
|---|---|
| `signal_source.nii.gz` | S0-normalized source acquisition |
| `signal_target.nii.gz` | S0-normalized full target protocol |
| `signal_dti_baseline.nii.gz` | S0-normalized DTI extrapolation baseline |
| `dwi_dti_baseline.nii.gz` | DTI baseline in physical signal scale |
| `S0.nii.gz` | median b0 image from the source acquisition |
| `brain_mask.nii.gz` | preprocessing mask |
| `FA.nii.gz`, `MD.nii.gz` | source-shell DTI priors |
| `protocol_source.bval/.bvec` | source gradient table |
| `protocol_full.bval/.bvec` | full target gradient table |
| `source_to_full_map.txt` | source-to-full direction matching |
| `shmeta_b1000_b2000_l8_lam0.006.npz` | SH metadata |

Canonical names are used by default. Several earlier file names, such as `St_obs_33.nii.gz`, `St_gt_63.nii.gz`, and `St_syn_63.nii.gz`, are still resolved for compatibility through `dti_shnet.constants.resolve_file`.

## Main training

The main reproducibility setting uses:

```text
roi=96,96,64
patch=32,32,32
dim=64
batch=4
epochs=30
lr=2e-4
num_workers=8
pps_train=12
pps_val=1
shuffle_patches=0
seed_per_epoch=1
amp=1
seed=0
use_dti=1
use_sig_loss=1
lambda_sig=0.1
b_in=1000
b_out=2000
lmax=8
lam=0.006
canon_n=60
```

Run with explicit paths:

```bash
PREPROC_ROOT=/path/to/preprocessed \
OUT_DIR=./results \
bash scripts/1_train_main.sh
```

The script saves `last.pt`, `best.pt`, and `meta.json` under `OUT_DIR`. The main inference script uses `last.pt`.

## Inference

```bash
PREPROC_ROOT=/path/to/preprocessed \
CKPT=./results/last.pt \
SAVE_DIR=./results \
bash scripts/2_test_main.sh
```

By default, inference saves only:

```text
SAVE_DIR/dti_shnet_pred/<subject>/signal_pred.nii.gz
```

`signal_pred.nii.gz` is the normalized target-shell prediction. The ROI prediction is pasted into a full-size target-shell volume, using the preprocessed DTI baseline shell outside the ROI.

Optional inference outputs can be enabled with:

```text
--save-target-shell 1
--save-baseline-shell 1
--save-shell-dwi 1
```

These optional files are not required for the main postprocessing pipeline.

## Postprocessing and evaluation

The verified postprocessing chain is:

```text
signal_pred.nii.gz
  ├─ visual metrics on normalized target-shell signal
  ├─ export original-space full multi-shell DWI
  ├─ DTI maps and delta metrics
  └─ NODDI maps and delta metrics
```

Run visual metrics, full-DWI export, and DTI evaluation:

```bash
ORIG_ROOT=/path/to/original_dataset \
PREPROC_ROOT=/path/to/preprocessed \
RUN_DIR=./results \
LAYOUT=camcan \
bash scripts/3_run_postprocess_main.sh
```

For UKB-style data, use `LAYOUT=ukb` and set `ORIG_ROOT` accordingly.

### Visual metrics

```bash
python -m dti_shnet.evaluation.visual \
  --pred-root ./results/dti_shnet_pred \
  --preproc-root /path/to/preprocessed \
  --out-csv ./results/visual_metrics.csv \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006 \
  --num-workers 4
```

The visual metrics are computed on full arrays without a mask:

```text
PSNR / SSIM / NMSE
```

If `signal_target_shell.nii.gz` is not saved by inference, the evaluator extracts the target shell from `signal_target.nii.gz` using SH metadata.

### Export full multi-shell DWI

```bash
python -m dti_shnet.export_dwi \
  --layout-preset camcan \
  --orig-root /path/to/original_dataset \
  --preproc-root /path/to/preprocessed \
  --pred-root ./results/dti_shnet_pred \
  --out-root ./results/export_dMRI \
  --target-b 2000 \
  --b-tol 50 \
  --st-clip-pct 99.5 \
  --num-workers 4 \
  --executor process \
  --skip-existing
```

Per subject, this writes:

```text
dwi_gt_raw.nii.gz
dwi_gt.nii.gz
dwi_baseline.nii.gz
dwi_pred.nii.gz
export_report.txt
```

`dwi_gt`, `dwi_baseline`, and `dwi_pred` are reconstructed through the same S → St → clipped St → S pipeline using the preprocessed `S0` and `brain_mask`. `dwi_gt_raw` is the untouched original full multi-shell DWI.

### DTI evaluation

```bash
python -m dti_shnet.evaluation.dti \
  --layout-preset camcan \
  --orig-root /path/to/original_dataset \
  --preproc-root /path/to/preprocessed \
  --pred-root ./results/dti_shnet_pred \
  --out-dir ./results/dti \
  --target-b 2000 \
  --b-tol 50 \
  --st-clip-pct 99.5 \
  --mask-source mrtrix \
  --mrtrix-clean-scale 2 \
  --mrtrix-nthreads 0 \
  --mrtrix-force 0 \
  --dti-use-all-dw 1 \
  --b1 1000 \
  --num-workers 4 \
  --skip-existing
```

Outputs:

```text
dti/<subject>/gt/FA.nii.gz
dti/<subject>/gt/MD.nii.gz
dti/<subject>/pred/FA.nii.gz
dti/<subject>/pred/MD.nii.gz
dti/<subject>/brain_mask_used.nii.gz
dti/<subject>/dti_delta_report.txt
dti/dti_summary.csv
```

### NODDI evaluation

NODDI fitting is slow and requires AMICO. It is disabled by default in `scripts/3_run_postprocess_main.sh`. Enable it with `RUN_NODDI=1`.

A safe first test is:

```bash
python -m dti_shnet.evaluation.noddi \
  --layout-preset camcan \
  --orig-root /path/to/original_dataset \
  --preproc-root /path/to/preprocessed \
  --pred-root ./results/dti_shnet_pred \
  --out-dir ./results/noddi \
  --target-b 2000 \
  --b-tol 50 \
  --st-clip-pct 99.5 \
  --mask-source mrtrix \
  --mrtrix-clean-scale 2 \
  --mrtrix-nthreads 0 \
  --mrtrix-force 0 \
  --nb-threads 8 \
  --num-workers 1 \
  --max-subjects 1
```

Full run example:

```bash
RUN_NODDI=1 \
NODDI_WORKERS=1 \
NODDI_THREADS=8 \
ORIG_ROOT=/path/to/original_dataset \
PREPROC_ROOT=/path/to/preprocessed \
RUN_DIR=./results \
LAYOUT=camcan \
bash scripts/3_run_postprocess_main.sh
```

Outputs:

```text
noddi/<subject>/gt/NDI.nii.gz
noddi/<subject>/gt/ODI.nii.gz
noddi/<subject>/gt/ISO.nii.gz
noddi/<subject>/pred/NDI.nii.gz
noddi/<subject>/pred/ODI.nii.gz
noddi/<subject>/pred/ISO.nii.gz
noddi/<subject>/brain_mask_used.nii.gz
noddi/<subject>/noddi_delta_report.txt
noddi/noddi_summary.csv
```

## Output conventions

Main outputs used for the reported pipeline:

```text
signal_pred.nii.gz
visual_metrics.csv
export_dMRI/<subject>/dwi_pred.nii.gz
export_dMRI/<subject>/dwi_gt.nii.gz
dti/dti_summary.csv
noddi/noddi_summary.csv
```

Optional or diagnostic outputs are retained for traceability and compatibility, but are not required for final DTI-SHNet metrics.

## Reproducibility notes

- Visual metrics use full arrays and no mask.
- Signal-consistency loss is unmasked.
- DTI/NODDI fitting and metric evaluation use the mask selected by `--mask-source`; the default is `mrtrix`.
- DWI reconstruction uses the preprocessing `brain_mask` regardless of the fitting mask.
- Main inference uses `last.pt`.
- Default main training parameters match the verified reproducibility setting above.
