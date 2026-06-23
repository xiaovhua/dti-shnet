from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..io import save_nifti_like
from ._postprocess_common import (
    EPS,
    B0_THR,
    ST_CLIP_PCT_DEFAULT,
    add_original_template_args,
    add_mask_args,
    build_dwi_triplet,
    to_st_percentile,
)


def norm_bvecs(bvals: np.ndarray, bvecs: np.ndarray, b0_thr: float = B0_THR) -> np.ndarray:
    B = bvecs.astype(np.float32).copy()
    for i, b in enumerate(bvals):
        if b <= b0_thr:
            B[i] = 0.0
        else:
            n = float(np.linalg.norm(B[i])) + 1e-8
            B[i] /= n
    return B


def design_matrix(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    g = bvecs.astype(np.float32)
    b = bvals.astype(np.float32)[:, None]
    A = np.stack([
        b[:, 0] * g[:, 0] * g[:, 0],
        b[:, 0] * g[:, 1] * g[:, 1],
        b[:, 0] * g[:, 2] * g[:, 2],
        2.0 * b[:, 0] * g[:, 0] * g[:, 1],
        2.0 * b[:, 0] * g[:, 0] * g[:, 2],
        2.0 * b[:, 0] * g[:, 1] * g[:, 2],
    ], axis=1).astype(np.float32)
    return A


def fit_dti_fa_md_from_st(St_xyzt: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, brain_mask: np.ndarray,
                          use_idx: np.ndarray, ridge: float = 1e-6):
    X, Y, Z, T = St_xyzt.shape
    V = X * Y * Z
    M = (brain_mask > 0.5).reshape(V).astype(np.float32)
    if int(M.sum()) == 0:
        raise RuntimeError("empty brain mask")

    bvecs_n = norm_bvecs(bvals, bvecs)
    A = design_matrix(bvals[use_idx], bvecs_n[use_idx])
    AtA = (A.T @ A).astype(np.float32) + ridge * np.eye(6, dtype=np.float32)
    P = np.linalg.solve(AtA, A.T).astype(np.float32)

    St = St_xyzt.reshape(V, T).astype(np.float32)
    Yobs = -np.log(np.clip(St[:, use_idx], EPS, 1.0)).astype(np.float32)
    Yobs[M < 0.5, :] = 0.0
    TH = (P @ Yobs.T).T

    D = np.zeros((V, 3, 3), dtype=np.float32)
    D[:, 0, 0] = TH[:, 0]
    D[:, 1, 1] = TH[:, 1]
    D[:, 2, 2] = TH[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = TH[:, 3]
    D[:, 0, 2] = D[:, 2, 0] = TH[:, 4]
    D[:, 1, 2] = D[:, 2, 1] = TH[:, 5]

    w, _ = np.linalg.eigh(D)
    lam1 = w[:, 2]
    lam2 = w[:, 1]
    lam3 = w[:, 0]
    md = (lam1 + lam2 + lam3) / 3.0
    denom = np.sqrt(lam1 * lam1 + lam2 * lam2 + lam3 * lam3) + EPS
    fa = np.sqrt(1.5) * np.sqrt((lam1 - md) ** 2 + (lam2 - md) ** 2 + (lam3 - md) ** 2) / denom
    fa = np.clip(fa, 0.0, 1.0)
    fa[M < 0.5] = np.nan
    md[M < 0.5] = np.nan
    return fa.reshape(X, Y, Z).astype(np.float32), md.reshape(X, Y, Z).astype(np.float32)


def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def process_one_subject(subj: str, args: argparse.Namespace):
    orig_img, dwi_gt, dwi_baseline, dwi_pred, S0, bvals, bvecs, mask, meta = build_dwi_triplet(subj, args, need_fit_mask=True)

    # Refit tensor from normalized St after the same percentile clipping used in reconstruction.
    St_gt_all = to_st_percentile(dwi_gt, S0, mask, st_clip_pct=float(args.st_clip_pct))
    St_pr_all = to_st_percentile(dwi_pred, S0, mask, st_clip_pct=float(args.st_clip_pct))

    if int(args.dti_use_all_dw) == 1:
        use_idx = np.where(bvals > B0_THR)[0]
        dti_desc = f"all_dw_b_gt_{B0_THR:g}"
    else:
        use_idx = np.where((bvals > B0_THR) & (np.abs(bvals - float(args.b1)) <= float(args.b_tol)))[0]
        dti_desc = f"b≈{args.b1:g}±{args.b_tol:g}"
    if use_idx.size < 6:
        raise RuntimeError(f"[{subj}] DTI insufficient directions: {use_idx.size}")

    fa_gt, md_gt = fit_dti_fa_md_from_st(St_gt_all, bvals, bvecs, mask, use_idx)
    fa_pr, md_pr = fit_dti_fa_md_from_st(St_pr_all, bvals, bvecs, mask, use_idx)
    delta_fa = masked_mae(fa_pr, fa_gt, mask)
    delta_md = masked_mae(md_pr, md_gt, mask)

    if args.out_dir:
        subj_out = Path(args.out_dir) / subj
        gt_out = subj_out / "gt"
        pr_out = subj_out / "pred"
        gt_out.mkdir(parents=True, exist_ok=True)
        pr_out.mkdir(parents=True, exist_ok=True)
        save_nifti_like(fa_gt, orig_img, gt_out / "FA.nii.gz")
        save_nifti_like(md_gt, orig_img, gt_out / "MD.nii.gz")
        save_nifti_like(fa_pr, orig_img, pr_out / "FA.nii.gz")
        save_nifti_like(md_pr, orig_img, pr_out / "MD.nii.gz")
        save_nifti_like(mask.astype(np.uint8), orig_img, subj_out / "brain_mask_used.nii.gz", dtype=np.uint8)
        if args.save_dwi:
            save_nifti_like(dwi_gt, orig_img, subj_out / "dwi_gt.nii.gz")
            save_nifti_like(dwi_pred, orig_img, subj_out / "dwi_pred.nii.gz")
        report = [
            f"subj: {subj}",
            f"delta_FA: {delta_fa:.8g}",
            f"delta_MD: {delta_md:.8g}",
            f"dti_use_idx_len: {int(use_idx.size)}",
            f"dti_use_idx: {use_idx.tolist()}",
            f"dti_desc: {dti_desc}",
            "",
        ]
        for k, v in meta.items():
            report.append(f"{k}: {v}")
        (subj_out / "dti_delta_report.txt").write_text("\n".join(report), encoding="utf-8")

    return {"subj": subj, "delta_FA": float(delta_fa), "delta_MD": float(delta_md), "n_dti_vols": int(use_idx.size)}


def _task(payload):
    subj, args = payload
    if args.skip_existing and args.out_dir and (Path(args.out_dir) / subj / "dti_delta_report.txt").exists():
        return {"status": "skip", "message": f"[skip] {subj} existing report", "result": None}
    try:
        res = process_one_subject(subj, args)
        return {"status": "ok", "message": f"[OK] {subj}: ΔFA={res['delta_FA']:.6g} ΔMD={res['delta_MD']:.6g} n={res['n_dti_vols']}", "result": res}
    except Exception as exc:
        return {"status": "err", "message": f"[ERR] {subj}: {exc!r}", "result": None}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute FA/MD maps and delta metrics from reconstructed full dWI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_original_template_args(ap)
    ap.add_argument("--preproc-root", "--preproc_root", dest="preproc_root", default="/path/to/preprocessed")
    ap.add_argument("--pred-root", "--pred_root", dest="pred_root", default="/path/to/dti_shnet_pred")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="/path/to/dti")
    ap.add_argument("--pred-name", default="signal_pred.nii.gz")
    ap.add_argument("--target-b", "--target_b", dest="target_b", type=float, default=2000.0)
    ap.add_argument("--b-tol", "--b_tol", dest="b_tol", type=float, default=50.0)
    ap.add_argument("--st-clip-pct", "--st_clip_pct", dest="st_clip_pct", type=float, default=ST_CLIP_PCT_DEFAULT)
    ap.add_argument("--dti-use-all-dw", "--dti_use_all_dw", dest="dti_use_all_dw", type=int, default=1,
                    help="1: fit tensor using all b>0 vols; 0: use only b≈b1.")
    ap.add_argument("--b1", type=float, default=1000.0)
    add_mask_args(ap)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--max-subjects", "--max_subjects", dest="max_subjects", type=int, default=0)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    ap.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true")
    ap.add_argument("--save-dwi", "--save_dwi", dest="save_dwi", action="store_true")
    return ap.parse_args()


def mean_std(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        pre = sorted(d.name for d in Path(args.preproc_root).iterdir() if d.is_dir())
        prd = sorted(d.name for d in Path(args.pred_root).iterdir() if d.is_dir())
        subjects = sorted(set(pre) & set(prd))
    if args.max_subjects > 0:
        subjects = subjects[:args.max_subjects]
    if not subjects:
        raise RuntimeError("no subjects found")

    print(f"[info] subjects={len(subjects)} target_b={args.target_b:g}")
    print(f"[info] pred_root={args.pred_root}")
    print(f"[info] orig_root={args.orig_root}")
    print(f"[info] preproc_root={args.preproc_root}")
    print(f"[info] mask_source_for_fit={args.mask_source}")
    print("[info] reconstruction_mask=preproc")
    print(f"[info] out_dir={args.out_dir}")

    results = []
    ok = bad = 0
    workers = max(1, int(args.num_workers))
    if workers == 1:
        for subj in subjects:
            item = _task((subj, args))
            print(item["message"])
            if item["status"] == "ok":
                results.append(item["result"]); ok += 1
            elif item["status"] != "skip":
                bad += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_task, (subj, args)) for subj in subjects]
            for fut in futures:
                item = fut.result()
                print(item["message"])
                if item["status"] == "ok":
                    results.append(item["result"]); ok += 1
                elif item["status"] != "skip":
                    bad += 1
    if not results:
        raise RuntimeError(f"no valid subjects; bad={bad}")

    m_fa, s_fa = mean_std([r["delta_FA"] for r in results])
    m_md, s_md = mean_std([r["delta_MD"] for r in results])
    print("\n========== Summary: DTI delta metrics ==========")
    print(f"ΔFA: {m_fa:.6g} ± {s_fa:.6g}")
    print(f"ΔMD: {m_md:.6g} ± {s_md:.6g}")
    print(f"[done] ok={ok} bad={bad}")

    csv_path = out_dir / "dti_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subj", "delta_FA", "delta_MD", "n_dti_vols"])
        for r in results:
            w.writerow([r["subj"], f"{r['delta_FA']:.8g}", f"{r['delta_MD']:.8g}", r["n_dti_vols"]])
        w.writerow(["MEAN", f"{m_fa:.8g}", f"{m_md:.8g}", ""])
        w.writerow(["STD", f"{s_fa:.8g}", f"{s_md:.8g}", ""])
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
