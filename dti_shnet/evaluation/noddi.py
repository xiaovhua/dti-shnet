from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np

try:
    import amico
except Exception as exc:  # pragma: no cover
    amico = None
    AMICO_IMPORT_ERROR = exc
else:
    AMICO_IMPORT_ERROR = None

from ..io import save_nifti_like, write_bval, write_bvec, load_nifti
from ._postprocess_common import (
    ST_CLIP_PCT_DEFAULT,
    add_original_template_args,
    add_mask_args,
    build_dwi_triplet,
)


def ensure_amico() -> None:
    if amico is None:
        raise RuntimeError(
            "Cannot import AMICO. Install the same NODDI backend used in the original experiments:\n"
            "  pip uninstall -y amico\n"
            "  pip install dmri-amico\n"
            f"Import error was: {AMICO_IMPORT_ERROR!r}"
        )


def call_amico_setup_once() -> None:
    try:
        import amico.core
        amico.core.setup()
    except Exception as exc:
        print(f"[warn] amico.core.setup() failed or was unnecessary: {exc}")


def find_map(root: str | Path, patterns: list[str]):
    root = str(root)
    for pat in patterns:
        hits = sorted(glob(os.path.join(root, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def find_noddi_maps(root: str | Path):
    ndi_path = find_map(root, [
        "FIT_ICVF.nii.gz", "fit_ICVF.nii.gz", "FIT_NDI.nii.gz", "fit_NDI.nii.gz",
        "*ICVF*.nii.gz", "*NDI*.nii.gz",
    ])
    odi_path = find_map(root, [
        "FIT_OD.nii.gz", "fit_OD.nii.gz", "FIT_ODI.nii.gz", "fit_ODI.nii.gz",
        "*ODI*.nii.gz", "*OD*.nii.gz",
    ])
    iso_path = find_map(root, [
        "FIT_ISOVF.nii.gz", "fit_ISOVF.nii.gz", "FIT_FWF.nii.gz", "fit_FWF.nii.gz",
        "*ISOVF*.nii.gz", "*ISO*.nii.gz", "*FWF*.nii.gz",
    ])
    if not (ndi_path and odi_path and iso_path):
        all_maps = sorted(glob(os.path.join(str(root), "**", "*.nii*"), recursive=True))
        raise FileNotFoundError(
            f"cannot find NODDI maps under {root}\nNDI={ndi_path}\nODI={odi_path}\nISO/FWF={iso_path}\nfound={all_maps[:30]}"
        )
    return {"NDI": ndi_path, "ODI": odi_path, "ISO": iso_path}


def run_amico_noddi(dwi: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray, like_img,
                    work_parent: str | Path, subject_name: str, output_name: str, b0_thr: float, nb_threads: int):
    ensure_amico()
    os.environ["OMP_NUM_THREADS"] = str(nb_threads)
    os.environ["MKL_NUM_THREADS"] = str(nb_threads)

    study_dir = os.path.abspath(str(work_parent))
    subj_dir = os.path.join(study_dir, subject_name)
    os.makedirs(subj_dir, exist_ok=True)
    save_nifti_like(dwi, like_img, os.path.join(subj_dir, "dwi.nii.gz"), dtype=np.float32)
    save_nifti_like(mask.astype(np.uint8), like_img, os.path.join(subj_dir, "mask.nii.gz"), dtype=np.uint8)
    write_bval(os.path.join(subj_dir, "dwi.bval"), bvals)
    write_bvec(os.path.join(subj_dir, "dwi.bvec"), bvecs)

    cwd = os.getcwd()
    try:
        os.chdir(study_dir)
        scheme = amico.util.fsl2scheme(os.path.join(subject_name, "dwi.bval"), os.path.join(subject_name, "dwi.bvec"))
        ae = amico.Evaluation(study_dir, subject_name, output_path=output_name)
        ae.load_data(dwi_filename="dwi.nii.gz", scheme_filename=os.path.basename(scheme), mask_filename="mask.nii.gz", b0_thr=float(b0_thr))
        ae.set_model("NODDI")
        ae.generate_kernels()
        ae.CONFIG["solver_params"]["numThreads"] = int(nb_threads)
        ae.load_kernels()
        ae.fit()
        ae.save_results()
    finally:
        os.chdir(cwd)

    out_dir = os.path.join(study_dir, output_name)
    paths = find_noddi_maps(out_dir)
    maps = {}
    for key, path in paths.items():
        _, arr = load_nifti(path)
        maps[key] = arr.astype(np.float32)
    return maps, paths, out_dir


def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def copy_and_standardize_maps(paths: dict[str, str], dst_dir: str | Path) -> None:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for key, src in paths.items():
        shutil.copy2(src, dst_dir / f"{key}.nii.gz")


def process_one_subject(subj: str, args: argparse.Namespace):
    t0 = time.time()
    orig_img, dwi_gt, dwi_baseline, dwi_pred, S0, bvals, bvecs, fit_mask, meta = build_dwi_triplet(subj, args, need_fit_mask=True)
    work_parent = tempfile.mkdtemp(prefix=f"amico_{subj}_", dir=args.tmp_dir if args.tmp_dir else None)
    try:
        maps_gt, paths_gt, _ = run_amico_noddi(
            dwi_gt, bvals, bvecs, fit_mask, orig_img,
            work_parent=work_parent, subject_name="gt", output_name="NODDI_gt",
            b0_thr=float(args.b0_thr), nb_threads=int(args.nb_threads),
        )
        maps_pr, paths_pr, _ = run_amico_noddi(
            dwi_pred, bvals, bvecs, fit_mask, orig_img,
            work_parent=work_parent, subject_name="pred", output_name="NODDI_pred",
            b0_thr=float(args.b0_thr), nb_threads=int(args.nb_threads),
        )
        delta_ndi = masked_mae(maps_pr["NDI"], maps_gt["NDI"], fit_mask)
        delta_odi = masked_mae(maps_pr["ODI"], maps_gt["ODI"], fit_mask)
        delta_iso = masked_mae(maps_pr["ISO"], maps_gt["ISO"], fit_mask)

        if args.out_dir:
            subj_out = Path(args.out_dir) / subj
            subj_out.mkdir(parents=True, exist_ok=True)
            copy_and_standardize_maps(paths_gt, subj_out / "gt")
            copy_and_standardize_maps(paths_pr, subj_out / "pred")
            save_nifti_like(fit_mask.astype(np.uint8), orig_img, subj_out / "brain_mask_used.nii.gz", dtype=np.uint8)
            if args.save_dwi:
                save_nifti_like(dwi_gt, orig_img, subj_out / "dwi_gt.nii.gz")
                save_nifti_like(dwi_pred, orig_img, subj_out / "dwi_pred.nii.gz")
                write_bval(subj_out / "dwi.bval", bvals)
                write_bvec(subj_out / "dwi.bvec", bvecs)
            report = [
                f"subj: {subj}",
                f"delta_NDI: {delta_ndi:.8g}",
                f"delta_ODI: {delta_odi:.8g}",
                f"delta_ISO: {delta_iso:.8g}",
                f"time_sec: {time.time() - t0:.2f}",
                "",
            ]
            for k, v in meta.items():
                report.append(f"{k}: {v}")
            (subj_out / "noddi_delta_report.txt").write_text("\n".join(report), encoding="utf-8")
        return {"subj": subj, "delta_NDI": float(delta_ndi), "delta_ODI": float(delta_odi), "delta_ISO": float(delta_iso), "time_sec": time.time() - t0}
    finally:
        if not args.keep_work:
            shutil.rmtree(work_parent, ignore_errors=True)
        else:
            print(f"[keep] AMICO work dir: {work_parent}")


def _task(payload):
    subj, args = payload
    if args.skip_existing and args.out_dir and (Path(args.out_dir) / subj / "noddi_delta_report.txt").exists():
        return {"status": "skip", "message": f"[skip] {subj} existing report", "result": None}
    try:
        ensure_amico()
        if args.run_amico_setup:
            call_amico_setup_once()
        res = process_one_subject(subj, args)
        return {"status": "ok", "message": f"[OK] {subj}: ΔNDI={res['delta_NDI']:.6g} ΔODI={res['delta_ODI']:.6g} ΔISO={res['delta_ISO']:.6g} time={res['time_sec']:.1f}s", "result": res}
    except Exception as exc:
        return {"status": "err", "message": f"[ERR] {subj}: {exc!r}", "result": None}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fit NODDI with AMICO and compute NDI/ODI/ISO delta metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_original_template_args(ap)
    ap.add_argument("--preproc-root", "--preproc_root", dest="preproc_root", default="/path/to/preprocessed")
    ap.add_argument("--pred-root", "--pred_root", dest="pred_root", default="/path/to/dti_shnet_pred")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="/path/to/noddi")
    ap.add_argument("--pred-name", default="signal_pred.nii.gz")
    ap.add_argument("--target-b", "--target_b", dest="target_b", type=float, default=2000.0)
    ap.add_argument("--b-tol", "--b_tol", dest="b_tol", type=float, default=50.0)
    ap.add_argument("--st-clip-pct", "--st_clip_pct", dest="st_clip_pct", type=float, default=ST_CLIP_PCT_DEFAULT)
    add_mask_args(ap)
    ap.add_argument("--tmp-dir", "--tmp_dir", dest="tmp_dir", default="")
    ap.add_argument("--nb-threads", "--nb_threads", dest="nb_threads", type=int, default=8, help="AMICO internal threads per subject.")
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4, help="Subject-level workers.")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--max-subjects", "--max_subjects", dest="max_subjects", type=int, default=0)
    ap.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true")
    ap.add_argument("--keep-work", "--keep_work", dest="keep_work", action="store_true")
    ap.add_argument("--save-dwi", "--save_dwi", dest="save_dwi", action="store_true")
    ap.add_argument("--run-amico-setup", "--run_amico_setup", dest="run_amico_setup", action="store_true")
    return ap.parse_args()


def mean_std(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main():
    args = parse_args()
    ensure_amico()
    if args.run_amico_setup:
        call_amico_setup_once()
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.tmp_dir:
        Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

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
    workers = max(1, int(args.num_workers))
    print(f"[info] num_workers={workers} nb_threads_per_subject={int(args.nb_threads)}")
    if workers > 1 and int(args.nb_threads) > 1:
        print(f"[warn] total AMICO threads may reach {workers * int(args.nb_threads)}; reduce --nb-threads if needed.")

    results = []
    ok = bad = 0
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

    m_ndi, s_ndi = mean_std([r["delta_NDI"] for r in results])
    m_odi, s_odi = mean_std([r["delta_ODI"] for r in results])
    m_iso, s_iso = mean_std([r["delta_ISO"] for r in results])
    print("\n========== Summary: NODDI delta metrics ==========")
    print(f"ΔNDI: {m_ndi:.6g} ± {s_ndi:.6g}")
    print(f"ΔODI: {m_odi:.6g} ± {s_odi:.6g}")
    print(f"ΔISO: {m_iso:.6g} ± {s_iso:.6g}")
    print(f"[done] ok={ok} bad={bad}")

    csv_path = Path(args.out_dir) / "noddi_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subj", "delta_NDI", "delta_ODI", "delta_ISO", "time_sec"])
        for r in results:
            w.writerow([r["subj"], f"{r['delta_NDI']:.8g}", f"{r['delta_ODI']:.8g}", f"{r['delta_ISO']:.8g}", f"{r['time_sec']:.3f}"])
        w.writerow(["MEAN", f"{m_ndi:.8g}", f"{m_odi:.8g}", f"{m_iso:.8g}", ""])
        w.writerow(["STD", f"{s_ndi:.8g}", f"{s_odi:.8g}", f"{s_iso:.8g}", ""])
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
