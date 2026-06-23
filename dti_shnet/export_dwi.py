from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from .io import save_nifti_like
from .evaluation._postprocess_common import (
    ST_CLIP_PCT_DEFAULT,
    add_original_template_args,
    build_dwi_triplet,
    copy_bval_bvec,
)


def _subject_task(payload):
    subj, args = payload
    try:
        print(f"[start] {subj}", flush=True)
        orig_img, dwi_gt, dwi_baseline, dwi_pred, _S0, _bvals, _bvecs, _fit_mask, meta = build_dwi_triplet(
            subj, args, need_fit_mask=False
        )
        out = Path(args.out_root) / subj
        out.mkdir(parents=True, exist_ok=True)

        # dwi_baseline is retained for baseline comparison and ROI-size prediction compatibility.
        save_nifti_like(dwi_gt, orig_img, out / "dwi_gt.nii.gz")
        save_nifti_like(dwi_baseline, orig_img, out / "dwi_baseline.nii.gz")
        save_nifti_like(dwi_pred, orig_img, out / "dwi_pred.nii.gz")

        # Untouched original multi-shell DWI, saved for traceability.
        from .io import load_nifti

        _, raw = load_nifti(meta["orig_nii"])
        save_nifti_like(raw, orig_img, out / "dwi_gt_raw.nii.gz")

        for stem in ["dwi_gt_raw", "dwi_gt", "dwi_baseline", "dwi_pred"]:
            copy_bval_bvec(meta["orig_bval"], meta["orig_bvec"], out / f"{stem}.bval", out / f"{stem}.bvec")

        report = [
            f"subj: {subj}",
            f"orig_nii: {meta['orig_nii']}",
            f"orig_bval: {meta['orig_bval']}",
            f"orig_bvec: {meta['orig_bvec']}",
            f"target_b: {meta['target_b']:g}",
            f"b_tol: {meta['b_tol']:g}",
            f"st_clip_pct: {meta['st_clip_pct']:g}",
            f"idx_target (len={len(meta['idx_target'])}): {meta['idx_target']}",
            f"roi_pred_path: {meta['prediction_path']}",
            f"roi_shape: {meta['prediction_shape']} (x,y,z,t)",
            f"full_shape: {meta['full_shape']}",
            f"roi_box (sx,sy,sz,ex,ey,ez): {meta['roi_box']}",
            "",
            "Reconstruction:",
            "  dwi_gt/dwi_baseline/dwi_pred are reconstructed consistently via S->St->S",
            "  using preproc S0 + brain_mask and percentile clipping.",
            "  Steps: clip(S,0,inf) -> St=S/S0 -> *mask -> percentile_clip -> S_rec=St*S0",
            "  dwi_gt_raw is the untouched original multi-shell.",
        ]
        (out / "export_report.txt").write_text("\n".join(report), encoding="utf-8")
        return {"status": "ok", "subject": subj, "message": f"[OK] {subj}"}
    except Exception as exc:
        return {"status": "err", "subject": subj, "message": f"[ERR] {subj}: {exc!r}"}


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Export original-space full multi-shell DWI from signal_pred.nii.gz."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_original_template_args(ap)
    ap.add_argument("--preproc-root", "--preproc_root", dest="preproc_root", default="/path/to/preprocessed")
    ap.add_argument("--pred-root", "--pred_root", dest="pred_root", default="/path/to/dti_shnet_pred")
    ap.add_argument("--out-root", "--out_root", dest="out_root", default="/path/to/exported_dwi")
    ap.add_argument(
        "--pred-name",
        default="signal_pred.nii.gz",
        help="Prediction filename under pred-root/subject. St_pred.nii.gz is also accepted for compatibility.",
    )
    ap.add_argument("--target-b", "--target_b", dest="target_b", type=float, default=2000.0)
    ap.add_argument("--b-tol", "--b_tol", dest="b_tol", type=float, default=50.0)
    ap.add_argument("--st-clip-pct", "--st_clip_pct", dest="st_clip_pct", type=float, default=ST_CLIP_PCT_DEFAULT)
    ap.add_argument("--subjects", default="", help="Comma-separated subject ids; empty means intersection of preproc-root and pred-root.")
    ap.add_argument("--max-subjects", "--max_subjects", dest="max_subjects", type=int, default=0)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    ap.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help=(
            "Subject-level parallel backend. 'process' is often faster for gzip/NIfTI + NumPy work; "
            "use 'thread' if RAM is tight or process spawning is problematic on Windows."
        ),
    )
    ap.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true")
    # Dummy args used by common.build_dwi_triplet when need_fit_mask=False.
    ap.set_defaults(mask_source="preproc")
    return ap.parse_args()


def _collect_subjects(args):
    preproc_root = Path(args.preproc_root)
    pred_root = Path(args.pred_root)
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        pre = sorted(d.name for d in preproc_root.iterdir() if d.is_dir())
        prd = sorted(d.name for d in pred_root.iterdir() if d.is_dir())
        subjects = sorted(set(pre) & set(prd))
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    return subjects


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = _collect_subjects(args)
    if not subjects:
        raise RuntimeError("No common subjects found between preproc-root and pred-root.")

    payloads = []
    for subj in subjects:
        if args.skip_existing and (out_root / subj / "export_report.txt").exists():
            print(f"[skip] {subj}", flush=True)
            continue
        payloads.append((subj, args))

    workers = max(1, int(args.num_workers))
    print(f"[info] subjects={len(subjects)} submitted={len(payloads)} target_b={args.target_b:g} st_clip_pct={args.st_clip_pct:g}", flush=True)
    print(f"[info] orig_root={args.orig_root}", flush=True)
    print(f"[info] preproc_root={args.preproc_root}", flush=True)
    print(f"[info] pred_root={args.pred_root}", flush=True)
    print(f"[info] out_root={args.out_root}", flush=True)
    print(f"[info] num_workers={workers} executor={args.executor}", flush=True)
    print("[info] export writes 4 full 4D NIfTI files per subject; gzip writing can be slow.", flush=True)

    if not payloads:
        print("[done] nothing to do", flush=True)
        return

    ok = bad = 0
    Executor = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    try:
        with Executor(max_workers=workers) as ex:
            futures = [ex.submit(_subject_task, p) for p in payloads]
            total = len(futures)
            for i, fut in enumerate(as_completed(futures), start=1):
                item = fut.result()
                print(f"[{i}/{total}] {item['message']}", flush=True)
                if item["status"] == "ok":
                    ok += 1
                else:
                    bad += 1
    except KeyboardInterrupt:
        print("\n[interrupt] export_dwi interrupted by user. Use --skip-existing to resume completed subjects.", flush=True)
        raise

    print(f"[done] ok={ok} bad={bad}", flush=True)


if __name__ == "__main__":
    main()
