from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from .constants import FILES, resolve_file, read_id_list, id_in_path
from .io import list_subject_dirs, load_nifti, save_nifti_like, take_volumes
from .protocol import load_shmeta
from .sh import build_sh_regressor, real_sym_sh_basis, n_sh_coeffs
from .spatial import center_crop_pad_3d, center_crop_pad_4d, center_slices
from .models.unet3d import UNet3DReg

EPS = 1e-6


def parse_tuple3(s: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in s.split(","))  # type: ignore[return-value]


def compute_coeff_volume(signal_roi: np.ndarray, mask_roi: np.ndarray, P: np.ndarray) -> np.ndarray:
    rx, ry, rz, n = signal_roi.shape
    K = P.shape[0]
    idx = np.flatnonzero(mask_roi.reshape(-1))
    flat = np.zeros((K, rx * ry * rz), dtype=np.float32)
    if idx.size == 0:
        return flat.reshape(K, rx, ry, rz)
    S = signal_roi[mask_roi].reshape(idx.size, n).astype(np.float32)
    C = (P @ S.T).T.astype(np.float32)
    flat[:, idx] = C.T
    return flat.reshape(K, rx, ry, rz)


@torch.no_grad()
def sliding_window_infer(model: torch.nn.Module, x: torch.Tensor, patch: tuple[int, int, int], overlap: float = 0.5):
    model.eval()
    B, C, X, Y, Z = x.shape
    px, py, pz = [min(int(a), int(b)) for a, b in zip(patch, (X, Y, Z))]
    sx = max(1, int(px * (1.0 - overlap)))
    sy = max(1, int(py * (1.0 - overlap)))
    sz = max(1, int(pz * (1.0 - overlap)))
    x_starts = list(range(0, max(X - px + 1, 1), sx))
    y_starts = list(range(0, max(Y - py + 1, 1), sy))
    z_starts = list(range(0, max(Z - pz + 1, 1), sz))
    if x_starts[-1] != X - px:
        x_starts.append(X - px)
    if y_starts[-1] != Y - py:
        y_starts.append(Y - py)
    if z_starts[-1] != Z - pz:
        z_starts.append(Z - pz)

    out = None
    cnt = torch.zeros((B, 1, X, Y, Z), device=x.device, dtype=x.dtype)
    for xs in x_starts:
        for ys in y_starts:
            for zs in z_starts:
                xb = x[:, :, xs:xs + px, ys:ys + py, zs:zs + pz]
                pb = model(xb)
                if out is None:
                    out = torch.zeros((B, pb.shape[1], X, Y, Z), device=x.device, dtype=pb.dtype)
                out[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += pb
                cnt[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += 1.0
    if out is None:
        raise RuntimeError("empty inference")
    return out / torch.clamp(cnt, min=1e-6)


def subject_list(roots, eval_id_file: str | Path):
    ids = read_id_list(eval_id_file)
    subjs = list_subject_dirs(roots)
    return sorted([s for s in subjs if id_in_path(ids, s)])


def main():
    ap = argparse.ArgumentParser(
        description="Run DTI-SHNet inference. By default only signal_pred.nii.gz is saved."
    )
    ap.add_argument("--roots", required=True, help="Python list of preprocessed roots")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path; use last.pt for the main reported setting.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--eval-id-file", default="splits/eval_id.txt")
    ap.add_argument("--roi", default="96,96,64")
    ap.add_argument("--infer-patch", default="32,32,32")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--b-in", type=int, default=1000)
    ap.add_argument("--b-out", type=int, default=2000)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.006)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-dti", type=int, default=1, choices=[0, 1])
    ap.add_argument("--save-target-shell", type=int, default=0,
                    help="1: additionally save signal_target_shell.nii.gz.")
    ap.add_argument("--save-baseline-shell", type=int, default=0,
                    help="1: additionally save signal_dti_baseline_shell.nii.gz.")
    ap.add_argument("--save-shell-dwi", type=int, default=0,
                    help="1: additionally save target-shell-only physical DWI files. These are not full multi-shell DWI.")
    args = ap.parse_args()

    roots = ast.literal_eval(args.roots)
    roi = parse_tuple3(args.roi)
    infer_patch = parse_tuple3(args.infer_patch)
    save_root = Path(args.save_dir) / "dti_shnet_pred"
    save_root.mkdir(parents=True, exist_ok=True)

    K = n_sh_coeffs(args.lmax)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    meta = ck.get("meta", {})
    in_ch = K + 3 if int(args.use_dti) == 1 else K
    base_dim = int(meta.get("dim", args.dim))
    model = UNet3DReg(in_channels=in_ch, out_channels=K, base=base_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    subjs = subject_list(roots, args.eval_id_file)
    used = []
    for sd in tqdm(subjs, desc="infer", ncols=110):
        shmeta = load_shmeta(sd, args.b_in, args.b_out, args.lmax, args.lam)
        if shmeta is None:
            print(f"[skip] missing shmeta: {sd}")
            continue
        in_i, in_dirs, out_idx, out_dirs = shmeta

        mask_img, mask_full = load_nifti(resolve_file(sd, FILES.brain_mask))
        if mask_full.ndim == 4 and mask_full.shape[-1] == 1:
            mask_full = mask_full[..., 0]
        mask_full = mask_full > 0.5
        mask_roi = center_crop_pad_3d(mask_full.astype(np.float32), roi, pad_value=0.0) > 0.5

        s0_img, S0_full = load_nifti(resolve_file(sd, FILES.s0))
        if S0_full.ndim == 4 and S0_full.shape[-1] == 1:
            S0_full = S0_full[..., 0]
        S0_full = S0_full.astype(np.float32)
        S0e_full = np.clip(S0_full, EPS, None)

        img_src = nib.load(str(resolve_file(sd, FILES.source_signal)))
        signal_source_full = take_volumes(img_src, in_i).astype(np.float32)
        signal_source_roi = center_crop_pad_4d(signal_source_full, roi, pad_value=0.0)

        img_tgt = nib.load(str(resolve_file(sd, FILES.target_signal)))
        signal_target_full = take_volumes(img_tgt, out_idx).astype(np.float32)

        p_base = resolve_file(sd, FILES.dti_signal, required=False)
        if p_base is not None:
            img_base = nib.load(str(p_base))
            signal_base_full = take_volumes(img_base, out_idx).astype(np.float32)
        else:
            signal_base_full = np.zeros_like(signal_target_full, dtype=np.float32)

        reg_in = build_sh_regressor(in_dirs, lmax=args.lmax, lam=args.lam)
        C_in = compute_coeff_volume(signal_source_roi, mask_roi, reg_in.P)
        if int(args.use_dti) == 1:
            _, FA_full = load_nifti(resolve_file(sd, FILES.fa))
            _, MD_full = load_nifti(resolve_file(sd, FILES.md))
            if FA_full.ndim == 4 and FA_full.shape[-1] == 1:
                FA_full = FA_full[..., 0]
            if MD_full.ndim == 4 and MD_full.shape[-1] == 1:
                MD_full = MD_full[..., 0]
            FA_roi = center_crop_pad_3d(FA_full.astype(np.float32), roi, pad_value=0.0)
            MD_roi = center_crop_pad_3d(MD_full.astype(np.float32), roi, pad_value=0.0)
            FA_roi = 2.0 * np.clip(FA_roi, 0.0, 1.0) - 1.0
            MD_roi = 2.0 * (np.clip(MD_roi, 0.0, 0.003) / 0.003) - 1.0
            x = np.concatenate([C_in, FA_roi[None], MD_roi[None], mask_roi.astype(np.float32)[None]], axis=0)
        else:
            x = C_in

        xt = torch.from_numpy(np.ascontiguousarray(x[None].astype(np.float32))).to(device)
        pred_coef = sliding_window_infer(model, xt, patch=infer_patch, overlap=args.overlap)[0].detach().cpu().numpy().astype(np.float32)

        B_out = real_sym_sh_basis(out_dirs, args.lmax).astype(np.float32)
        rx, ry, rz = roi
        V = rx * ry * rz
        signal_pred = (B_out @ pred_coef.reshape(K, V)).T.reshape(rx, ry, rz, -1).astype(np.float32)
        signal_pred = np.clip(signal_pred, 0.0, None)

        full_sl, roi_sl = center_slices(signal_target_full.shape[:3], roi)
        signal_pred_full = signal_base_full.copy()
        signal_pred_full[full_sl + (slice(None),)] = signal_pred[roi_sl + (slice(None),)]

        out_sub = save_root / sd.name
        out_sub.mkdir(parents=True, exist_ok=True)
        save_nifti_like(signal_pred_full, mask_img, out_sub / FILES.prediction_signal)

        if int(args.save_target_shell) == 1:
            save_nifti_like(signal_target_full, mask_img, out_sub / FILES.target_shell_signal)
        if int(args.save_baseline_shell) == 1:
            save_nifti_like(signal_base_full, mask_img, out_sub / FILES.baseline_shell_signal)
        if int(args.save_shell_dwi) == 1:
            save_nifti_like(signal_pred_full * S0e_full[..., None], mask_img, out_sub / FILES.prediction_dwi)
            if int(args.save_target_shell) == 1:
                save_nifti_like(signal_target_full * S0e_full[..., None], mask_img, out_sub / FILES.target_shell_dwi)
            if int(args.save_baseline_shell) == 1:
                save_nifti_like(signal_base_full * S0e_full[..., None], mask_img, out_sub / FILES.baseline_shell_dwi)
        used.append(sd.name)

    summary = {
        "n_subjects": len(used),
        "subjects": used,
        "ckpt": str(args.ckpt),
        "checkpoint_policy": "last.pt for the main reported setting",
        "saved_by_default": [FILES.prediction_signal],
    }
    (save_root / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
