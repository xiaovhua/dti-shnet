from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..constants import FILES, resolve_file
from ..io import load_nifti, take_volumes
from ..protocol import load_shmeta


def nmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum((gt - pred) ** 2) / max(float(np.sum(gt ** 2)), eps))


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float, eps: float = 1e-12) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    return float(20.0 * np.log10(max(float(data_range), eps)) - 10.0 * np.log10(max(mse, eps)))


def mean_ssim_3d(pred: np.ndarray, gt: np.ndarray, data_range: float) -> float:
    """Mean 2D-slice SSIM over z-slices and target-shell volumes."""
    if pred.ndim == 3:
        pred = pred[..., None]
        gt = gt[..., None]
    vals = []
    _, _, Z, T = pred.shape
    for t in range(T):
        for z in range(Z):
            ax = gt[:, :, z, t].astype(np.float32)
            bx = pred[:, :, z, t].astype(np.float32)
            vals.append(ssim(ax, bx, data_range=data_range))
    return float(np.mean(vals)) if vals else float("nan")


def load_gt_target_shell(pred_dir: Path, preproc_root: Path | None, args: argparse.Namespace) -> np.ndarray:
    # Prefer a saved target-shell file; otherwise extract the target shell from the preprocessed full protocol.
    saved = pred_dir / args.gt_name
    if saved.exists():
        _, gt = load_nifti(saved)
        return gt.astype(np.float32)

    if preproc_root is None:
        raise FileNotFoundError(f"{saved} not found and --preproc-root was not provided")

    pre_dir = preproc_root / pred_dir.name
    shmeta = load_shmeta(pre_dir, args.b_in, args.b_out, args.lmax, args.lam)
    if shmeta is None:
        raise FileNotFoundError(f"missing shmeta for {pre_dir}")
    _, _, out_idx, _ = shmeta
    img_tgt = nib.load(str(resolve_file(pre_dir, FILES.target_signal)))
    return take_volumes(img_tgt, out_idx).astype(np.float32)


def _subject_task(payload: tuple[str, argparse.Namespace]) -> dict:
    subject, args = payload
    pred_dir = Path(args.pred_root) / subject
    p_pred = pred_dir / args.pred_name

    if not p_pred.exists():
        return {"status": "skip", "subject": subject, "message": f"[skip] {subject}: missing {args.pred_name}"}

    try:
        preproc_root = Path(args.preproc_root) if args.preproc_root else None
        _, pred = load_nifti(p_pred)
        gt = load_gt_target_shell(pred_dir, preproc_root, args)

        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)

        if pred.shape != gt.shape:
            return {
                "status": "skip",
                "subject": subject,
                "message": f"[skip] {subject}: shape mismatch pred={pred.shape} gt={gt.shape}",
            }

        finite_gt = gt[np.isfinite(gt)]
        data_range = float(np.percentile(finite_gt, float(args.data_range_pct))) if finite_gt.size else 1.0
        data_range = max(data_range, 1e-6)

        row = {
            "subject": subject,
            "psnr": psnr(pred, gt, data_range),
            "ssim": mean_ssim_3d(pred, gt, data_range),
            "nmse": nmse(pred, gt),
        }
        return {"status": "ok", "subject": subject, "row": row, "message": f"[OK] {subject}"}
    except Exception as exc:
        return {"status": "err", "subject": subject, "message": f"[ERR] {subject}: {exc!r}"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate PSNR/SSIM/NMSE on unmasked normalized target-shell signal arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pred-root", "--pred_root", dest="pred_root", required=True)
    ap.add_argument(
        "--preproc-root",
        "--preproc_root",
        dest="preproc_root",
        default="",
        help="Needed when infer.py only saved signal_pred.nii.gz.",
    )
    ap.add_argument("--out-csv", "--out_csv", dest="out_csv", required=True)
    ap.add_argument("--pred-name", default=FILES.prediction_signal)
    ap.add_argument("--gt-name", default=FILES.target_shell_signal)
    ap.add_argument("--b-in", type=int, default=1000)
    ap.add_argument("--b-out", type=int, default=2000)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.006)
    ap.add_argument("--data-range-pct", type=float, default=99.9)
    ap.add_argument("--subjects", default="", help="Comma-separated subject ids; empty means all subjects under pred-root.")
    ap.add_argument("--max-subjects", "--max_subjects", dest="max_subjects", type=int, default=0)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    return ap.parse_args()


def _collect_subjects(args: argparse.Namespace) -> list[str]:
    pred_root = Path(args.pred_root)
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = sorted(d.name for d in pred_root.iterdir() if d.is_dir())
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    return subjects


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "psnr", "ssim", "nmse"])
        w.writeheader()
        w.writerows(rows)
        if rows:
            vals = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in ["psnr", "ssim", "nmse"]}
            w.writerow({"subject": "MEAN", **{k: float(np.nanmean(v)) for k, v in vals.items()}})
            w.writerow({"subject": "STD", **{k: float(np.nanstd(v)) for k, v in vals.items()}})


def main() -> None:
    args = parse_args()
    subjects = _collect_subjects(args)
    if not subjects:
        raise RuntimeError(f"No subjects found under {args.pred_root}")

    print(f"[info] subjects={len(subjects)} num_workers={max(1, int(args.num_workers))}")
    print(f"[info] pred_root={args.pred_root}")
    print(f"[info] preproc_root={args.preproc_root if args.preproc_root else '<not provided>'}")
    print("[info] metrics are computed on full arrays without mask")

    rows: list[dict] = []
    ok = bad = skip = 0
    payloads = [(s, args) for s in subjects]
    workers = max(1, int(args.num_workers))

    if workers == 1:
        results = [_subject_task(p) for p in payloads]
    else:
        # map preserves input order, so the CSV order is stable.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_subject_task, payloads))

    for item in results:
        print(item["message"])
        if item["status"] == "ok":
            rows.append(item["row"])
            ok += 1
        elif item["status"] == "skip":
            skip += 1
        else:
            bad += 1

    out_csv = Path(args.out_csv)
    _write_csv(rows, out_csv)

    if rows:
        for k in ["psnr", "ssim", "nmse"]:
            vals = np.asarray([r[k] for r in rows], dtype=np.float64)
            print(f"{k}: {np.nanmean(vals):.6f} ± {np.nanstd(vals):.6f}")
    print(f"[saved] {out_csv} ok={ok} skip={skip} bad={bad}")


if __name__ == "__main__":
    main()
