# DTI-Guided Volumetric Spherical Harmonics Regression for Single-to-Multi-Shell dMRI Synthesis

<p align="center">
  <a href="https://github.com/xiaovhua/dti-shnet">
    <img src="https://img.shields.io/badge/Code-GitHub-blue?logo=github&style=for-the-badge" alt="code"/>
  </a>
  <a href="https://github.com/xiaovhua/dti-shnet">
    <img src="https://img.shields.io/badge/Paper-MICCAI-red?style=for-the-badge" alt="paper"/>
  </a>
</p>

<!-- Optional: add the method figure after preparing the media folder. -->
<!-- ![method](./media/method.png) -->

### This repository contains the official implementation of **"DTI-Guided Volumetric Spherical Harmonics Regression for Single-to-Multi-Shell dMRI Synthesis"**.

DTI-SHNet synthesizes an unobserved high-b diffusion MRI shell from a source single-shell acquisition. Instead of directly regressing diffusion-weighted images, the model learns a volumetric mapping in the real symmetric spherical harmonics (SH) coefficient domain and reconstructs the predicted target-shell signal from the estimated SH coefficients.

## 🔍 Introduction

DTI-SHNet is designed for single-to-multi-shell dMRI synthesis. Given a source shell, the pipeline performs median-b0 normalization, source-shell DTI fitting, SH coefficient fitting, volumetric SH regression, and SH reconstruction to obtain the predicted target-shell signal.

```text
Raw dMRI
  ├── median-b0 normalization
  ├── source-shell DTI fitting
  │     └── FA / MD / preprocessing brain mask
  ├── SH fitting
  │     └── source-shell and target-shell SH coefficients
  ├── DTI-SHNet
  │     └── source SH + FA + MD + mask → target SH
  └── SH reconstruction
        └── predicted target-shell signal
```

For the main `lmax=8` setting, the SH coefficient dimension is `K=45`. The full DTI-guided model uses 48 input channels:

```text
45 source SH channels + FA + MD + mask → 45 target SH channels
```

**Key highlights:**

- 🧠 SH-domain volumetric regression for single-to-multi-shell dMRI synthesis  
- 🧭 DTI-guided priors using FA, MD, and a brain mask as anatomical/diffusion cues  
- 📦 End-to-end reproducible pipeline from preprocessing to visual, DTI, and NODDI evaluation  

## 📁 Repository Structure

```bash
.
├── dti_shnet/                       # Main package
│   ├── preprocess.py                # Dataset preprocessing and SH metadata generation
│   ├── train.py                     # DTI-SHNet training
│   ├── infer.py                     # Target-shell signal inference
│   ├── export_dwi.py                # Export full multi-shell DWI in original space
│   ├── dataset.py                   # Patch dataset and SH fitting at training time
│   ├── sh.py                        # Real symmetric SH basis and fitting utilities
│   ├── protocol.py                  # bval/bvec and direction utilities
│   ├── spatial.py                   # ROI crop/paste utilities
│   ├── models/
│   │   └── unet3d.py                # 3D U-Net regressor
│   └── evaluation/
│       ├── visual.py                # PSNR / SSIM / NMSE on target-shell signal
│       ├── dti.py                   # FA / MD maps and delta metrics
│       ├── noddi.py                 # AMICO NODDI fitting and delta metrics
│       └── _postprocess_common.py   # Shared reconstruction and mask logic
├── scripts/                         # Reproducible command-line examples
│   ├── 0_prepare.sh                 # Preprocessing
│   ├── 1_train_main.sh              # Main training
│   ├── 2_test_main.sh               # Inference
│   ├── 3_run_postprocess_main.sh    # Visual / export / DTI / optional NODDI
│   └── run_from_scratch.ipynb       # Tutorial for running our codes from scratch
├── splits/
│   └── eval_id.txt                  # Evaluation subject IDs
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/xiaovhua/dti-shnet.git
cd dti-shnet/
```

### 2. Set up environment

```bash
conda env create -f environment.yml
conda activate dti-shnet
pip install -e .
```

For DTI and NODDI postprocessing, please install [MRtrix3](https://www.mrtrix.org/) and make sure `dwi2mask` is available on `PATH`.

For NODDI fitting, install AMICO:

```bash
pip install dmri-amico dipy
```

### 3. Tutorial for training from scratch

If you want to train DTI-SHNet from scratch on your own dataset, you can either follow `scripts/run_from_scratch.ipynb` or follow Sections 4–7 below, from data preparation to postprocessing and evaluation.

### 4. Prepare data

The preprocessing interface is template-based. All templates are interpreted relative to `--root` and should contain `{subject}`.

For a Cam-CAN-style layout, edit [`scripts/0_prepare.sh`](./scripts/0_prepare.sh) or run:

```bash
python -m dti_shnet.preprocess \
  --root /path/to/CamCAN_root \
  --layout-preset camcan \
  --out-dir-template "preprocessed/{subject}" \
  --workers 8 \
  --b-in 1000 \
  --b-out 2000 \
  --lmax 8 \
  --lam 0.006
```

For a custom layout, provide explicit templates:

```bash
python -m dti_shnet.preprocess \
  --root /path/to/dataset_root \
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

Each preprocessed subject folder contains normalized source/target signals, the DTI baseline, `S0`, `brain_mask`, FA/MD priors, protocol files, and SH metadata.

### 5. Train the model

Please set the correct `PREPROC_ROOT` and `OUT_DIR` in [`scripts/1_train_main.sh`](./scripts/1_train_main.sh), then run:

```bash
bash scripts/1_train_main.sh
```

The main reproducibility setting is:

```text
roi=96,96,64       patch=32,32,32      dim=64
batch=4            epochs=30           lr=2e-4
num_workers=8      pps_train=12        pps_val=1
use_dti=1          use_sig_loss=1      lambda_sig=0.1
b_in=1000          b_out=2000          lmax=8
lam=0.006          canon_n=60          optimizer=AdamW
```

Training saves `last.pt`, `best.pt`, and `meta.json` under the output directory. The verified inference pipeline uses `last.pt`.

### 6. Run inference

Please set the correct `PREPROC_ROOT`, `CKPT`, and `SAVE_DIR` in [`scripts/2_test_main.sh`](./scripts/2_test_main.sh), then run:

```bash
bash scripts/2_test_main.sh
```

By default, inference saves only the normalized target-shell prediction:

```text
SAVE_DIR/dti_shnet_pred/<subject>/signal_pred.nii.gz
```

The ROI prediction is pasted into a full-size target-shell volume. Outside the ROI, the preprocessed DTI extrapolation baseline shell is used as the background. Optional target-shell debug outputs can be enabled in `infer.py`, but they are not required for the main evaluation pipeline.

### 7. Postprocess and evaluate

Please set the original data path, preprocessed data path, prediction directory, and dataset layout in [`scripts/3_run_postprocess_main.sh`](./scripts/3_run_postprocess_main.sh), then run:

```bash
bash scripts/3_run_postprocess_main.sh
```

The verified postprocessing chain is:

```text
signal_pred.nii.gz
  ├── visual metrics on normalized target-shell signal
  ├── original-space full multi-shell DWI export
  ├── DTI maps and delta metrics
  └── NODDI maps and delta metrics
```

For manual execution, the main commands are:

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

NODDI fitting is slow and requires AMICO. A safe first test is:

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

## 🎓 Tutorial & Quick Start

A minimal end-to-end workflow is:

```bash
# 1. Preprocess data
bash scripts/0_prepare.sh

# 2. Train DTI-SHNet
bash scripts/1_train_main.sh

# 3. Predict target-shell signal
bash scripts/2_test_main.sh

# 4. Compute visual metrics, export full DWI, and run DTI evaluation
bash scripts/3_run_postprocess_main.sh
```

Please make sure to update all path variables in the scripts before running. `layout-preset` only defines how files are located under the user-provided root directory; it does not define the root path itself.

## 📊 Results

The repository is organized to reproduce the reported visual metrics, DTI-derived metrics, and NODDI-derived metrics from the paper.

Main output files:

```text
signal_pred.nii.gz
visual_metrics.csv
export_DWI/<subject>/dwi_pred.nii.gz
export_DWI/<subject>/dwi_gt.nii.gz
dti/dti_summary.csv
noddi/noddi_summary.csv
```

<!-- Optional: add quantitative result table or figure after release. -->
<!-- ![results](./media/results.png) -->

## 🧪 Reproducibility Notes

- Visual metrics are computed on full arrays without a mask.
- The signal-consistency loss is computed without a mask.
- DWI export reconstructs `dwi_gt`, `dwi_baseline`, and `dwi_pred` using the same `S → St → clipped St → S` pipeline with the preprocessing `S0` and `brain_mask`.
- DTI/NODDI fitting and metric evaluation use the mask selected by `--mask-source`; the default is `mrtrix`.
- The verified inference pipeline uses `last.pt`.
- The main model uses `use_dti=1` and `use_sig_loss=1`.

## 🤝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{li2026dtishnet,
  title     = {DTI-Guided Volumetric Spherical Harmonics Regression for Single-to-Multi-Shell dMRI Synthesis},
  author    = {Li, Binghua and others},
  booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year      = {2026}
}
```

The citation entry will be updated once the official proceedings information is available.

## 📬 Contact

For questions or collaborations, please contact Binghua Li at `b.li.qr@juntendo.ac.jp`.

## 📄 License

Please refer to the repository license file for usage terms.
