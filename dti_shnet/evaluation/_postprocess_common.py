from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:
    from dipy.segment.mask import median_otsu
except Exception:  # pragma: no cover
    median_otsu = None

from ..constants import FILES, resolve_file
from ..io import load_nifti, save_nifti_like, load_bval, load_bvec, write_bval, write_bvec

EPS = 1e-6
B0_THR = 50.0
ST_CLIP_PCT_DEFAULT = 99.5


@dataclass(frozen=True)
class OriginalTemplates:
    orig_nii: str
    orig_bval: str
    orig_bvec: str


ORIGINAL_PRESETS: dict[str, OriginalTemplates] = {
    "flat": OriginalTemplates(
        orig_nii="{subject}/target.nii.gz",
        orig_bval="{subject}/target.bval",
        orig_bvec="{subject}/target.bvec",
    ),
    "cam": OriginalTemplates(
        orig_nii="{subject}/Preprocessed_data/dwi_preprocessed.nii.gz",
        orig_bval="{subject}/Preprocessed_data/dwi_preprocessed.bval",
        orig_bvec="{subject}/Preprocessed_data/dwi_preprocessed.bvec",
    ),
    "camcan": OriginalTemplates(
        orig_nii="{subject}/Preprocessed_data/dwi_preprocessed.nii.gz",
        orig_bval="{subject}/Preprocessed_data/dwi_preprocessed.bval",
        orig_bvec="{subject}/Preprocessed_data/dwi_preprocessed.bvec",
    ),
    "ukb": OriginalTemplates(
        orig_nii="*/multi-shell/{subject}*/data_ud.nii.gz",
        orig_bval="*/multi-shell/{subject}*/dwi.bval",
        orig_bvec="*/multi-shell/{subject}*/bvecs",
    ),
    "thu": OriginalTemplates(
        orig_nii="{subject}/multishell.nii.gz",
        orig_bval="{subject}/multishell.bval",
        orig_bvec="{subject}/multishell.bvec",
    ),
}


def add_original_template_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--orig-root", default="/path/to/original_dataset", help="Root directory of original full multi-shell DWI data.")
    p.add_argument("--layout-preset", choices=sorted(ORIGINAL_PRESETS), default=None,
                   help="Optional original-data path preset. Explicit templates override this.")
    p.add_argument("--dataset", choices=sorted(ORIGINAL_PRESETS), default=None, help=argparse.SUPPRESS)
    p.add_argument("--orig-nii-template", default=None, help="Path template relative to --orig-root; use {subject} as subject-id placeholder.")
    p.add_argument("--orig-bval-template", default=None)
    p.add_argument("--orig-bvec-template", default=None)


def collect_original_templates(args: argparse.Namespace) -> OriginalTemplates:
    preset_name = getattr(args, "layout_preset", None) or getattr(args, "dataset", None)
    preset = ORIGINAL_PRESETS.get(preset_name) if preset_name else None
    nii = getattr(args, "orig_nii_template", None) or (preset.orig_nii if preset else None)
    bval = getattr(args, "orig_bval_template", None) or (preset.orig_bval if preset else None)
    bvec = getattr(args, "orig_bvec_template", None) or (preset.orig_bvec if preset else None)
    missing = [k for k, v in [("orig_nii", nii), ("orig_bval", bval), ("orig_bvec", bvec)] if v is None]
    if missing:
        raise ValueError(
            "Missing original-data path templates: " + ", ".join(missing) +
            ". Use --layout-preset or provide all --orig-*-template arguments."
        )
    return OriginalTemplates(str(nii), str(bval), str(bvec))


def _resolve_template(root: Path, template: str, subject: str, kind: str) -> Path:
    rel = template.format(subject=subject)
    hits = sorted(glob(str(root / rel), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No match for {kind}: root={root} template={template} subject={subject}")
    if len(hits) > 1:
        print(f"[warn] multiple matches for {kind} subject={subject}; using first: {hits[0]}")
    return Path(hits[0])


def locate_original(orig_root: str | Path, subject: str, templates: OriginalTemplates) -> tuple[Path, Path, Path]:
    root = Path(orig_root)
    return (
        _resolve_template(root, templates.orig_nii, subject, "original NIfTI"),
        _resolve_template(root, templates.orig_bval, subject, "original bval"),
        _resolve_template(root, templates.orig_bvec, subject, "original bvec"),
    )


def copy_bval_bvec(src_bval: str | Path, src_bvec: str | Path, dst_bval: str | Path, dst_bvec: str | Path) -> None:
    Path(dst_bval).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_bval), str(dst_bval))
    shutil.copy2(str(src_bvec), str(dst_bvec))


def clip_normalized_by_percentile(data: np.ndarray, mask: np.ndarray, pct: float) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    m = np.asarray(mask) > 0.5
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got {arr.shape}")
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[-1]):
        vol = arr[..., i]
        vals = vol[m]
        vals = vals[np.isfinite(vals)]
        upper = float(np.percentile(vals, pct)) if vals.size else 0.0
        out[..., i] = np.clip(vol, 0.0, upper)
    return out.astype(np.float32)


def preprocess_full_dwi_S_St_S_percentile(dwi_raw: np.ndarray, S0_full: np.ndarray, brain_mask: np.ndarray,
                                           st_clip_pct: float = ST_CLIP_PCT_DEFAULT) -> np.ndarray:
    """Reconstruct DWI by applying the verified S -> St -> clipped St -> S pipeline."""
    if dwi_raw.ndim != 4:
        raise ValueError(f"dwi_raw must be 4D, got {dwi_raw.shape}")
    if S0_full.shape != dwi_raw.shape[:3]:
        raise ValueError(f"S0 shape mismatch: {S0_full.shape} vs {dwi_raw.shape[:3]}")
    if brain_mask.shape != dwi_raw.shape[:3]:
        raise ValueError(f"mask shape mismatch: {brain_mask.shape} vs {dwi_raw.shape[:3]}")
    S0_safe = np.clip(S0_full.astype(np.float32), EPS, None)
    M = (brain_mask > 0.5).astype(np.float32)
    S = np.clip(dwi_raw.astype(np.float32), 0.0, None)
    St = (S / S0_safe[..., None]) * M[..., None]
    St = clip_normalized_by_percentile(St, M, st_clip_pct).astype(np.float32)
    return (St * S0_safe[..., None]).astype(np.float32)


def to_st_percentile(dwi_s: np.ndarray, S0_full: np.ndarray, mask: np.ndarray,
                     st_clip_pct: float = ST_CLIP_PCT_DEFAULT) -> np.ndarray:
    S0_safe = np.clip(S0_full.astype(np.float32), EPS, None)
    M = (mask > 0.5).astype(np.float32)
    St = (np.clip(dwi_s.astype(np.float32), 0.0, None) / S0_safe[..., None]) * M[..., None]
    return clip_normalized_by_percentile(St, M, st_clip_pct).astype(np.float32)


def center_crop_start(full: int, roi: int) -> int:
    return int((full - roi) // 2)


def paste_roi_into_full(full_shape_xyz: tuple[int, int, int], roi_xyzt: np.ndarray, fill_xyzt: np.ndarray):
    X, Y, Z = full_shape_xyz
    rx, ry, rz, T = roi_xyzt.shape
    if fill_xyzt.shape != (X, Y, Z, T):
        raise ValueError(f"fill shape mismatch: {fill_xyzt.shape} vs {(X, Y, Z, T)}")
    sx, sy, sz = center_crop_start(X, rx), center_crop_start(Y, ry), center_crop_start(Z, rz)
    ex, ey, ez = sx + rx, sy + ry, sz + rz
    if not (0 <= sx < ex <= X and 0 <= sy < ey <= Y and 0 <= sz < ez <= Z):
        raise ValueError(f"ROI does not fit full={full_shape_xyz} roi={(rx, ry, rz)}")
    out = fill_xyzt.copy()
    out[sx:ex, sy:ey, sz:ez, :] = roi_xyzt
    return out, (sx, sy, sz, ex, ey, ez)


def load_preproc_mask(pre_dir: str | Path) -> np.ndarray:
    _, M = load_nifti(resolve_file(pre_dir, FILES.brain_mask))
    if M.ndim == 4:
        M = M[..., 0]
    return (M > 0.5).astype(np.uint8)


def load_mask_file(path: str | Path) -> np.ndarray:
    _, M = load_nifti(path)
    if M.ndim == 4:
        M = M[..., 0]
    return (M > 0.5).astype(np.uint8)


def validate_mask_nonempty(mask: np.ndarray, desc: str) -> int:
    n = int(np.asarray(mask).sum())
    if n <= 0:
        raise RuntimeError(f"empty brain mask: {desc}")
    return n


def mask_voxels(path: str | Path) -> int:
    return int(load_mask_file(path).sum())


def _subprocess_env_with_home() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("HOME"):
        home = env.get("USERPROFILE")
        if not home:
            home = (env.get("HOMEDRIVE", "") + env.get("HOMEPATH", "")) or os.path.expanduser("~")
        if home and home != "~":
            env["HOME"] = home
    return env


def run_mrtrix_dwi2mask(dwi_path: str | Path, bvec_path: str | Path, bval_path: str | Path,
                        out_mask_path: str | Path, args: argparse.Namespace) -> None:
    cmd = [
        getattr(args, "mrtrix_dwi2mask_cmd", "dwi2mask"),
        str(dwi_path),
        str(out_mask_path),
        "-fslgrad",
        str(bvec_path),
        str(bval_path),
        "-clean_scale",
        str(int(getattr(args, "mrtrix_clean_scale", 2))),
        "-nthreads",
        str(int(getattr(args, "mrtrix_nthreads", 0))),
    ]
    if int(getattr(args, "mrtrix_quiet", 1)) == 1:
        cmd.append("-quiet")
    if int(getattr(args, "mrtrix_force", 0)) == 1:
        cmd.append("-force")
    print("[mrtrix] running:", " ".join('"%s"' % c if " " in str(c) else str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=_subprocess_env_with_home())


def otsu_threshold(vol: np.ndarray) -> float:
    vals = np.asarray(vol, dtype=np.float32)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.0
    lo, hi = np.percentile(vals, [1.0, 99.5])
    vals = vals[(vals >= lo) & (vals <= hi)]
    if vals.size == 0:
        return float(lo)
    hist, edges = np.histogram(vals, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist).astype(np.float64)
    w2 = np.cumsum(hist[::-1]).astype(np.float64)[::-1]
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    score = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[:-1][int(np.argmax(score))])


def clean_mask(mask: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    m = np.asarray(mask) > 0
    if ndi is None:
        return m.astype(np.uint8)
    lab, n = ndi.label(m)
    if n > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(np.argmax(sizes))
    m = ndi.binary_fill_holes(m)
    if dilate_iter > 0:
        m = ndi.binary_dilation(m, iterations=int(dilate_iter))
    return m.astype(np.uint8)


def make_dipy_mask(dwi_raw: np.ndarray, bvals: np.ndarray, b0_thr: float, median_radius: int, numpass: int,
                   dilate: int, keep_largest: bool):
    if median_otsu is None:
        raise RuntimeError("DIPY is not installed. Use --mask-source preproc or install dipy.")
    b0_idx = np.where(bvals <= b0_thr)[0]
    if b0_idx.size == 0:
        b0_idx = np.where(bvals <= B0_THR)[0]
    if b0_idx.size == 0:
        raise RuntimeError(f"no b0 volume found with b <= {b0_thr} or <= {B0_THR}")
    mean_b0 = dwi_raw[..., b0_idx].mean(axis=3).astype(np.float32)
    _, mask = median_otsu(mean_b0, median_radius=int(median_radius), numpass=int(numpass), autocrop=False, dilate=int(dilate))
    mask = np.asarray(mask) > 0
    if keep_largest:
        mask = clean_mask(mask, dilate_iter=0)
    return mask.astype(np.uint8), f"dipy; b0_idx={b0_idx.tolist()}; median_radius={median_radius}; numpass={numpass}; dilate={dilate}"


def get_fit_mask(pre_dir: str | Path, dwi_raw: np.ndarray, bvals: np.ndarray, orig_nii: str | Path,
                 orig_bval: str | Path, orig_bvec: str | Path, args: argparse.Namespace):
    source = getattr(args, "mask_source", "mrtrix")
    if source == "preproc":
        M = load_preproc_mask(pre_dir)
        return M, f"preproc; voxels={validate_mask_nonempty(M, 'preproc')}"
    if source == "mrtrix":
        mask_path = Path(pre_dir) / getattr(args, "mrtrix_mask_name", "brain_mask_mrtrix.nii.gz")
        need_run = int(getattr(args, "mrtrix_force", 0)) == 1 or (not mask_path.exists())
        if (not need_run) and mask_path.exists():
            try:
                if mask_voxels(mask_path) <= 0:
                    need_run = True
            except Exception:
                need_run = True
        if need_run:
            if mask_path.exists():
                try:
                    mask_path.unlink()
                except Exception:
                    pass
            run_mrtrix_dwi2mask(orig_nii, orig_bvec, orig_bval, mask_path, args)
        M = load_mask_file(mask_path)
        return M, f"mrtrix; mask={mask_path}; voxels={validate_mask_nonempty(M, str(mask_path))}"
    if source == "dipy":
        return make_dipy_mask(
            dwi_raw, bvals, getattr(args, "b0_thr", 50.0),
            getattr(args, "dipy_median_radius", 4), getattr(args, "dipy_numpass", 4),
            getattr(args, "dipy_dilate", 1), bool(getattr(args, "dipy_keep_largest", 1)),
        )
    b0_idx = np.where(bvals <= getattr(args, "b0_thr", 50.0))[0]
    if b0_idx.size == 0:
        b0_idx = np.where(bvals <= B0_THR)[0]
    if b0_idx.size == 0:
        raise RuntimeError("no b0 volume found")
    b0 = dwi_raw[..., b0_idx].mean(axis=3)
    M = b0 > otsu_threshold(b0)
    if source == "auto_b0_clean":
        M = clean_mask(M, dilate_iter=getattr(args, "mask_dilate", 1))
    return M.astype(np.uint8), f"{source}; b0_idx={b0_idx.tolist()}"


def resolve_prediction_signal(pred_dir: str | Path, pred_name: str = FILES.prediction_signal) -> Path:
    pred_dir = Path(pred_dir)
    candidates = [pred_dir / pred_name]
    # Compatibility with earlier prediction folder names.
    if pred_name != "St_pred.nii.gz":
        candidates.append(pred_dir / "St_pred.nii.gz")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing prediction in {pred_dir}; tried {[p.name for p in candidates]}")


def build_dwi_triplet(subject: str, args: argparse.Namespace, *, need_fit_mask: bool = True):
    """Build original-space dwi_gt, dwi_baseline, and dwi_pred.

    The reconstruction mask is always the preprocessing brain_mask. The fitting
    mask is controlled by --mask-source and is used only for DTI/NODDI fitting and
    metric evaluation.
    """
    templates = collect_original_templates(args)
    pre_dir = Path(args.preproc_root) / subject
    pred_dir = Path(args.pred_root) / subject
    if not pre_dir.is_dir():
        raise FileNotFoundError(f"missing preprocessed dir: {pre_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"missing prediction dir: {pred_dir}")

    orig_nii, orig_bval, orig_bvec = locate_original(args.orig_root, subject, templates)
    orig_img, dwi_raw = load_nifti(orig_nii)
    bvals = load_bval(orig_bval).astype(np.float32)
    bvecs = load_bvec(orig_bvec).astype(np.float32)
    if dwi_raw.ndim != 4:
        raise ValueError(f"[{subject}] original DWI must be 4D, got {dwi_raw.shape}")
    X, Y, Z, N = dwi_raw.shape
    if bvals.shape[0] != N or bvecs.shape[0] != N:
        raise ValueError(f"[{subject}] bval/bvec mismatch with DWI N={N}")
    idx_t = np.where(np.abs(bvals - float(args.target_b)) <= float(args.b_tol))[0]
    if idx_t.size == 0:
        raise RuntimeError(f"[{subject}] no target shell b≈{args.target_b}±{args.b_tol}")

    _, S0 = load_nifti(resolve_file(pre_dir, FILES.s0))
    if S0.ndim == 4:
        S0 = S0[..., 0]
    recon_mask = load_preproc_mask(pre_dir)
    if S0.shape != (X, Y, Z) or recon_mask.shape != (X, Y, Z):
        raise ValueError(f"[{subject}] preproc S0/mask shape mismatch with original DWI")

    dwi_gt = preprocess_full_dwi_S_St_S_percentile(dwi_raw, S0, recon_mask, float(args.st_clip_pct))

    _, Stsyn = load_nifti(resolve_file(pre_dir, FILES.dti_signal))
    if Stsyn.ndim != 4 or Stsyn.shape != (X, Y, Z, N):
        raise ValueError(f"[{subject}] DTI baseline signal shape mismatch: {Stsyn.shape} vs {(X, Y, Z, N)}")
    S0_safe = np.clip(S0.astype(np.float32), EPS, None)
    St_base_b = np.clip(Stsyn[..., idx_t], 0.0, None).astype(np.float32)
    S_base_b = (St_base_b * S0_safe[..., None]).astype(np.float32)
    dwi_baseline = dwi_gt.copy()
    dwi_baseline[..., idx_t] = S_base_b

    pred_path = resolve_prediction_signal(pred_dir, getattr(args, "pred_name", FILES.prediction_signal))
    _, st_pred = load_nifti(pred_path)
    if st_pred.ndim == 3:
        st_pred = st_pred[..., None]
    if st_pred.ndim != 4:
        raise ValueError(f"[{subject}] bad prediction shape {st_pred.shape}")
    if st_pred.shape[-1] != idx_t.size:
        raise ValueError(f"[{subject}] prediction T={st_pred.shape[-1]} != target volumes={idx_t.size}")
    if st_pred.shape[:3] == (X, Y, Z):
        St_pred_b = np.clip(st_pred, 0.0, None).astype(np.float32)
        roi_box = (0, 0, 0, X, Y, Z)
    else:
        St_pred_b, roi_box = paste_roi_into_full((X, Y, Z), np.clip(st_pred, 0.0, None).astype(np.float32), St_base_b)
    S_pred_b = (St_pred_b * S0_safe[..., None]).astype(np.float32)
    dwi_pred = dwi_gt.copy()
    dwi_pred[..., idx_t] = S_pred_b

    if need_fit_mask:
        fit_mask, fit_mask_desc = get_fit_mask(pre_dir, dwi_raw, bvals, orig_nii, orig_bval, orig_bvec, args)
    else:
        fit_mask, fit_mask_desc = recon_mask, "preproc; not used for fitting"

    meta = {
        "orig_nii": str(orig_nii),
        "orig_bval": str(orig_bval),
        "orig_bvec": str(orig_bvec),
        "target_b": float(args.target_b),
        "b_tol": float(args.b_tol),
        "st_clip_pct": float(args.st_clip_pct),
        "idx_target": idx_t.tolist(),
        "prediction_path": str(pred_path),
        "prediction_shape": tuple(st_pred.shape),
        "full_shape": (X, Y, Z),
        "roi_box": roi_box,
        "recon_mask": "preproc",
        "fit_mask": fit_mask_desc,
        "note": "dwi_gt/dwi_baseline/dwi_pred are reconstructed consistently via S->St->S using preproc S0 + brain_mask and percentile clipping.",
    }
    return orig_img, dwi_gt, dwi_baseline, dwi_pred, S0.astype(np.float32), bvals, bvecs, fit_mask, meta


def add_mask_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mask-source", "--mask_source", dest="mask_source", choices=["preproc", "mrtrix", "dipy", "auto_b0", "auto_b0_clean"], default="mrtrix")
    p.add_argument("--b0-thr", "--b0_thr", dest="b0_thr", type=float, default=50.0)
    p.add_argument("--mrtrix-dwi2mask-cmd", "--mrtrix_dwi2mask_cmd", dest="mrtrix_dwi2mask_cmd", default="dwi2mask")
    p.add_argument("--mrtrix-mask-name", "--mrtrix_mask_name", dest="mrtrix_mask_name", default="brain_mask_mrtrix.nii.gz")
    p.add_argument("--mrtrix-clean-scale", "--mrtrix_clean_scale", dest="mrtrix_clean_scale", type=int, default=2)
    p.add_argument("--mrtrix-nthreads", "--mrtrix_nthreads", dest="mrtrix_nthreads", type=int, default=0)
    p.add_argument("--mrtrix-force", "--mrtrix_force", dest="mrtrix_force", type=int, default=0)
    p.add_argument("--mrtrix-quiet", "--mrtrix_quiet", dest="mrtrix_quiet", type=int, default=1)
    p.add_argument("--dipy-median-radius", "--dipy_median_radius", dest="dipy_median_radius", type=int, default=4)
    p.add_argument("--dipy-numpass", "--dipy_numpass", dest="dipy_numpass", type=int, default=4)
    p.add_argument("--dipy-dilate", "--dipy_dilate", dest="dipy_dilate", type=int, default=1)
    p.add_argument("--dipy-keep-largest", "--dipy_keep_largest", dest="dipy_keep_largest", type=int, default=1)
    p.add_argument("--mask-dilate", "--mask_dilate", dest="mask_dilate", type=int, default=1)
