from __future__ import annotations

import argparse
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
import re
import traceback

import nibabel as nib
import numpy as np
from tqdm import tqdm

from .constants import FILES
from .io import load_bval, load_bvec, save_nifti_like, write_bval, write_bvec
from .protocol import (
    B0_THR,
    DEFAULT_B_TOL,
    DEFAULT_DOT_THR,
    design_matrix,
    norm_bvecs,
    match_source_to_full_protocol,
    write_mapping_txt,
    build_shmeta_for_subject,
)

EPS = 1e-6
RIDGE = 1e-6
S0_MASK_PCT = 60.0
S0_MASK_MIN = 20.0
ST_CLIP_PCT = 99.5


@dataclass(frozen=True)
class LayoutTemplates:
    """Relative path templates for one raw-data layout.

    All templates are interpreted relative to ``--root``.  The special token
    ``{subject}`` identifies the subject directory/name and is replaced at
    runtime.  Other glob wildcards such as ``*`` and ``**`` are allowed.

    This keeps the preprocessing IO generic: UKB, Cam-CAN, and any local layout
    are simply different sets of templates, not different code paths.
    """

    source_nii: str
    target_nii: str
    source_bval: str
    target_bval: str
    source_bvec: str
    target_bvec: str
    out_dir: str = "preprocessed/{subject}"


LAYOUT_PRESETS: dict[str, LayoutTemplates] = {
    # Convenience presets only.  The implementation below is fully template-based.
    "flat": LayoutTemplates(
        source_nii="{subject}/source.nii.gz",
        target_nii="{subject}/target.nii.gz",
        source_bval="{subject}/source.bval",
        target_bval="{subject}/target.bval",
        source_bvec="{subject}/source.bvec",
        target_bvec="{subject}/target.bvec",
    ),
    "camcan": LayoutTemplates(
        source_nii="{subject}/Preprocessed_data/dwi_preprocessed_single.nii.gz",
        target_nii="{subject}/Preprocessed_data/dwi_preprocessed.nii.gz",
        source_bval="{subject}/Preprocessed_data/dwi_preprocessed_single.bval",
        target_bval="{subject}/Preprocessed_data/dwi_preprocessed.bval",
        source_bvec="{subject}/Preprocessed_data/dwi_preprocessed_single.bvec",
        target_bvec="{subject}/Preprocessed_data/dwi_preprocessed.bvec",
    ),
    "cam": LayoutTemplates(
        source_nii="{subject}/Preprocessed_data/dwi_preprocessed_single.nii.gz",
        target_nii="{subject}/Preprocessed_data/dwi_preprocessed.nii.gz",
        source_bval="{subject}/Preprocessed_data/dwi_preprocessed_single.bval",
        target_bval="{subject}/Preprocessed_data/dwi_preprocessed.bval",
        source_bvec="{subject}/Preprocessed_data/dwi_preprocessed_single.bvec",
        target_bvec="{subject}/Preprocessed_data/dwi_preprocessed.bvec",
    ),
    "ukb": LayoutTemplates(
        source_nii="*/single-shell/{subject}/data_ss_b0_b1000.nii.gz",
        target_nii="*/multi-shell/{subject}/data_ud.nii.gz",
        source_bval="*/single-shell/{subject}/dwi_ss.bval",
        target_bval="*/multi-shell/{subject}/dwi.bval",
        source_bvec="*/single-shell/{subject}/dwi_ss.bvec",
        target_bvec="*/multi-shell/{subject}/bvecs",
    ),
}


def make_brain_mask_from_s0(
    S0: np.ndarray,
    *,
    min_thr: float = S0_MASK_MIN,
    pct: float = S0_MASK_PCT,
) -> np.ndarray:
    S0_pos = np.clip(S0, 0.0, None)
    thr_pct = float(np.percentile(S0_pos[S0_pos > 0], pct)) if np.any(S0_pos > 0) else float(min_thr)
    thr = max(float(min_thr), thr_pct)
    return (S0_pos >= thr).astype(np.float32)


def clip_normalized_by_percentile(data: np.ndarray, mask: np.ndarray, pct: float = ST_CLIP_PCT) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    m = np.asarray(mask).reshape(-1) > 0.5
    if arr.ndim == 4:
        X, Y, Z, N = arr.shape
        flat = arr.reshape(-1, N)
        out = np.zeros_like(flat, dtype=np.float32)
        for i in range(N):
            vals = flat[:, i][m]
            vals = vals[np.isfinite(vals)]
            upper = float(np.percentile(vals, pct)) if vals.size > 0 else 0.0
            out[:, i] = np.clip(flat[:, i], 0.0, upper)
        return out.reshape(X, Y, Z, N)
    if arr.ndim == 2:
        V, N = arr.shape
        if m.shape[0] != V:
            raise ValueError(f"mask/data mismatch: {m.shape[0]} vs {V}")
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(N):
            vals = arr[:, i][m]
            vals = vals[np.isfinite(vals)]
            upper = float(np.percentile(vals, pct)) if vals.size > 0 else 0.0
            out[:, i] = np.clip(arr[:, i], 0.0, upper)
        return out
    raise ValueError(f"Unsupported ndim={arr.ndim}; expected 2D or 4D")


def dti_baseline_and_priors(
    img_source: nib.Nifti1Image,
    bval_source: np.ndarray,
    bvec_source: np.ndarray,
    bval_full: np.ndarray,
    bvec_full: np.ndarray,
    *,
    b0_thr: float = B0_THR,
    b_tol: float = DEFAULT_B_TOL,
    dot_thr: float = DEFAULT_DOT_THR,
    override_observed: bool = True,
    st_clip_pct: float = ST_CLIP_PCT,
):
    S4d = img_source.get_fdata(dtype=np.float32).astype(np.float32)
    X, Y, Z, Nsrc = S4d.shape
    V = X * Y * Z
    S = np.clip(S4d.reshape(V, Nsrc), 0.0, None)

    bvec_source_n = norm_bvecs(bval_source, bvec_source, b0_thr=b0_thr)
    bvec_full_n = norm_bvecs(bval_full, bvec_full, b0_thr=b0_thr)

    is_b0 = bval_source <= float(b0_thr)
    if int(is_b0.sum()) == 0:
        raise RuntimeError("No b0 frames found in source dMRI")
    S0 = np.median(S[:, is_b0], axis=1).astype(np.float32)
    S0[S0 < EPS] = EPS
    S0_vol = S0.reshape(X, Y, Z)
    mask = make_brain_mask_from_s0(S0_vol)
    Mflat = mask.reshape(V, 1)

    source_signal = (S / S0[:, None]) * Mflat
    source_signal = clip_normalized_by_percentile(source_signal, Mflat[:, 0], pct=st_clip_pct).astype(np.float32)

    idx_dw = np.where(~is_b0)[0]
    if idx_dw.size < 6:
        raise RuntimeError(f"Not enough DW directions to fit tensor: {idx_dw.size}")
    A = design_matrix(bval_source[idx_dw], bvec_source_n[idx_dw])
    AtA = (A.T @ A).astype(np.float32) + RIDGE * np.eye(6, dtype=np.float32)
    P = np.linalg.solve(AtA, A.T).astype(np.float32)

    Yobs = -np.log(np.clip(source_signal[:, idx_dw], EPS, 1.0)).astype(np.float32)
    Yobs[Mflat[:, 0] < 0.5, :] = 0.0
    tensor6 = (P @ Yobs.T).T.astype(np.float32)

    A_full = design_matrix(bval_full, bvec_full_n)
    Ypred = (A_full @ tensor6.T).T
    signal_dti = np.exp(-np.clip(Ypred, -50.0, 50.0)).astype(np.float32)
    signal_dti *= Mflat
    signal_dti = clip_normalized_by_percentile(signal_dti, Mflat[:, 0], pct=st_clip_pct).astype(np.float32)

    mapping, report = match_source_to_full_protocol(
        bval_source=bval_source,
        bvec_source=bvec_source_n,
        bval_full=bval_full,
        bvec_full=bvec_full_n,
        b_tol=b_tol,
        dot_thr=dot_thr,
        b0_thr=b0_thr,
    )

    if override_observed:
        for i in range(Nsrc):
            j = int(mapping[i])
            if 0 <= j < signal_dti.shape[1]:
                signal_dti[:, j] = source_signal[:, i]

    D = np.zeros((V, 3, 3), dtype=np.float32)
    D[:, 0, 0] = tensor6[:, 0]
    D[:, 1, 1] = tensor6[:, 1]
    D[:, 2, 2] = tensor6[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = tensor6[:, 3]
    D[:, 0, 2] = D[:, 2, 0] = tensor6[:, 4]
    D[:, 1, 2] = D[:, 2, 1] = tensor6[:, 5]
    FA = np.zeros(V, dtype=np.float32)
    MD = np.zeros(V, dtype=np.float32)
    mask_flat = Mflat[:, 0] > 0.5
    for v in np.flatnonzero(mask_flat):
        w, _ = np.linalg.eigh(D[v])
        lam1, lam2, lam3 = float(w[2]), float(w[1]), float(w[0])
        md = (lam1 + lam2 + lam3) / 3.0
        MD[v] = md
        denom = np.sqrt(lam1 * lam1 + lam2 * lam2 + lam3 * lam3) + EPS
        FA[v] = np.sqrt(1.5) * np.sqrt((lam1 - md) ** 2 + (lam2 - md) ** 2 + (lam3 - md) ** 2) / denom

    dwi_dti = (signal_dti * S0[:, None]).reshape(X, Y, Z, -1).astype(np.float32)
    return {
        "S0": S0_vol.astype(np.float32),
        "brain_mask": mask.astype(np.float32),
        "signal_source": source_signal.reshape(X, Y, Z, Nsrc).astype(np.float32),
        "signal_dti": signal_dti.reshape(X, Y, Z, -1).astype(np.float32),
        "dwi_dti": dwi_dti,
        "FA": np.clip(FA.reshape(X, Y, Z), 0.0, 1.0).astype(np.float32),
        "MD": MD.reshape(X, Y, Z).astype(np.float32),
        "mapping": mapping,
        "report": report,
    }


def preprocess_one_subject(
    source_nii: str | Path,
    target_nii: str | Path,
    source_bval: str | Path,
    target_bval: str | Path,
    source_bvec: str | Path,
    target_bvec: str | Path,
    out_dir: str | Path,
    *,
    b_in: float = 1000.0,
    b_out: float = 2000.0,
    b_tol: float = DEFAULT_B_TOL,
    dot_thr: float = DEFAULT_DOT_THR,
    st_clip_pct: float = ST_CLIP_PCT,
    override_observed: bool = True,
    make_shmeta: bool = True,
    lmax: int = 8,
    lam: float = 0.006,
    overwrite_shmeta: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_source = nib.load(str(source_nii))
    img_target = nib.load(str(target_nii))
    bval_source = load_bval(source_bval)
    bvec_source = load_bvec(source_bvec)
    bval_target = load_bval(target_bval)
    bvec_target = load_bvec(target_bvec)
    if bvec_source.shape[0] != bval_source.shape[0]:
        raise ValueError("source bval/bvec length mismatch")
    if bvec_target.shape[0] != bval_target.shape[0]:
        raise ValueError("target bval/bvec length mismatch")

    out = dti_baseline_and_priors(
        img_source,
        bval_source,
        bvec_source,
        bval_target,
        bvec_target,
        b_tol=b_tol,
        dot_thr=dot_thr,
        override_observed=override_observed,
        st_clip_pct=st_clip_pct,
    )

    save_nifti_like(out["signal_source"], img_source, out_dir / FILES.source_signal)
    save_nifti_like(out["signal_dti"], img_source, out_dir / FILES.dti_signal)
    save_nifti_like(out["dwi_dti"], img_source, out_dir / FILES.dti_dwi)
    save_nifti_like(out["S0"], img_source, out_dir / FILES.s0)
    save_nifti_like(out["brain_mask"], img_source, out_dir / FILES.brain_mask)
    save_nifti_like(out["FA"], img_source, out_dir / FILES.fa)
    save_nifti_like(out["MD"], img_source, out_dir / FILES.md)
    write_bval(out_dir / FILES.protocol_bval, bval_target)
    write_bvec(out_dir / FILES.protocol_bvec, norm_bvecs(bval_target, bvec_target))
    write_bval(out_dir / FILES.source_bval, bval_source)
    write_bvec(out_dir / FILES.source_bvec, norm_bvecs(bval_source, bvec_source))
    write_mapping_txt(out_dir / FILES.source_to_full_map, out["mapping"], out["report"])

    S_target = img_target.get_fdata(dtype=np.float32).astype(np.float32)
    S0e = np.clip(out["S0"], EPS, None).astype(np.float32)
    mask = (out["brain_mask"] > 0).astype(np.float32)
    signal_target = (np.clip(S_target, 0.0, None) / S0e[..., None]) * mask[..., None]
    signal_target = clip_normalized_by_percentile(signal_target, mask, pct=st_clip_pct).astype(np.float32)
    save_nifti_like(signal_target, img_source, out_dir / FILES.target_signal)

    if make_shmeta:
        build_shmeta_for_subject(
            out_dir,
            b_in=b_in,
            b_out=b_out,
            b_tol=b_tol,
            lmax=lmax,
            lam=lam,
            overwrite=overwrite_shmeta,
        )


def _read_subject_list(path: str | Path | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"subject list not found: {p}")
    out = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def _template_parts(template: str) -> tuple[str, ...]:
    return Path(template).parts


def _subject_part_index(template: str) -> int | None:
    parts = _template_parts(template)
    for i, p in enumerate(parts):
        if p == "{subject}":
            return i
    return None


def _discover_subjects(root: Path, source_template: str, subject_list: str | Path | None = None) -> list[str]:
    explicit = _read_subject_list(subject_list)
    if explicit is not None:
        return sorted(set(explicit))

    idx = _subject_part_index(source_template)
    if idx is None:
        raise ValueError(
            "Cannot discover subjects because --source-nii-template does not contain "
            "a path component exactly equal to '{subject}'. Provide --subject-list, "
            "or use a template such as '{subject}/source.nii.gz'."
        )
    glob_pattern = source_template.replace("{subject}", "*")
    matches = sorted(glob(str(root / glob_pattern), recursive="**" in glob_pattern))
    subjects: set[str] = set()
    for m in matches:
        try:
            rel_parts = Path(m).resolve().relative_to(root.resolve()).parts
        except ValueError:
            rel_parts = Path(m).parts
        if len(rel_parts) > idx:
            subjects.add(rel_parts[idx])
    return sorted(subjects)


def _resolve_template(root: Path, template: str, subject: str, *, kind: str) -> Path:
    rel = template.format(subject=subject)
    matches = sorted(glob(str(root / rel), recursive="**" in rel))
    if not matches:
        raise FileNotFoundError(f"No match for {kind}: root={root} template={template} subject={subject}")
    if len(matches) > 1:
        print(f"[warn] multiple matches for {kind} subject={subject}; using {matches[0]}")
    return Path(matches[0])


def _output_subject_id(raw_subject: str, regex: str | None) -> str:
    if not regex:
        return raw_subject
    m = re.search(regex, raw_subject)
    if not m:
        return raw_subject
    return m.group(1) if m.groups() else m.group(0)


def _collect_templates(args: argparse.Namespace) -> LayoutTemplates:
    preset_name = args.layout_preset or args.dataset
    preset = LAYOUT_PRESETS.get(preset_name) if preset_name else None

    def get(name: str) -> str | None:
        value = getattr(args, name)
        if value:
            return value
        if preset is not None:
            return getattr(preset, name)
        return None

    fields = {
        "source_nii": get("source_nii"),
        "target_nii": get("target_nii"),
        "source_bval": get("source_bval"),
        "target_bval": get("target_bval"),
        "source_bvec": get("source_bvec"),
        "target_bvec": get("target_bvec"),
        "out_dir": args.out_dir_template or (preset.out_dir if preset is not None else "preprocessed/{subject}"),
    }
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        raise ValueError(
            "Missing layout templates: "
            + ", ".join(missing)
            + ". Provide --layout-preset or all --*-template arguments."
        )
    return LayoutTemplates(**fields)  # type: ignore[arg-type]


def build_pairs_from_templates(
    root: str | Path,
    templates: LayoutTemplates,
    *,
    subject_list: str | Path | None = None,
    output_subject_regex: str | None = None,
) -> list[tuple[str, str, str, str, str, str, str]]:
    root = Path(root)
    raw_subjects = _discover_subjects(root, templates.source_nii, subject_list)
    pairs: list[tuple[str, str, str, str, str, str, str]] = []
    used_out: dict[str, str] = {}
    for raw_subj in raw_subjects:
        try:
            out_subj = _output_subject_id(raw_subj, output_subject_regex)
            if out_subj in used_out and used_out[out_subj] != raw_subj:
                raise ValueError(f"duplicate output subject id {out_subj!r} from {raw_subj!r} and {used_out[out_subj]!r}")
            used_out[out_subj] = raw_subj
            source_nii = _resolve_template(root, templates.source_nii, raw_subj, kind="source nii")
            target_nii = _resolve_template(root, templates.target_nii, raw_subj, kind="target nii")
            source_bval = _resolve_template(root, templates.source_bval, raw_subj, kind="source bval")
            target_bval = _resolve_template(root, templates.target_bval, raw_subj, kind="target bval")
            source_bvec = _resolve_template(root, templates.source_bvec, raw_subj, kind="source bvec")
            target_bvec = _resolve_template(root, templates.target_bvec, raw_subj, kind="target bvec")
            out_dir = root / templates.out_dir.format(subject=out_subj, raw_subject=raw_subj)
            pairs.append((
                str(source_nii),
                str(target_nii),
                str(source_bval),
                str(target_bval),
                str(source_bvec),
                str(target_bvec),
                str(out_dir),
            ))
        except Exception as e:
            print(f"[skip] subject={raw_subj}: {e}")
    return sorted(pairs, key=lambda x: x[-1])


# Compatibility wrapper for preset-based calls. Template-based CLI is preferred.
def build_pairs(dataset: str, root: str | Path):
    preset = LAYOUT_PRESETS.get(dataset)
    if preset is None:
        raise ValueError(f"Unknown layout preset: {dataset}")
    return build_pairs_from_templates(root, preset)


def _process_item(args_tuple):
    item, kwargs = args_tuple
    try:
        preprocess_one_subject(*item, **kwargs)
        return None
    except Exception as e:
        return f"{item[-1]}: {repr(e)}\n{traceback.format_exc()}"


def _print_presets() -> None:
    for name, p in LAYOUT_PRESETS.items():
        print(f"[{name}]")
        print(f"  --source-nii-template  {p.source_nii}")
        print(f"  --target-nii-template  {p.target_nii}")
        print(f"  --source-bval-template {p.source_bval}")
        print(f"  --target-bval-template {p.target_bval}")
        print(f"  --source-bvec-template {p.source_bvec}")
        print(f"  --target-bvec-template {p.target_bvec}")
        print(f"  --out-dir-template     {p.out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Prepare DTI-SHNet subject folders using template-based data IO")
    ap.add_argument("--root", required=True, help="Raw-data root. All templates are relative to this directory.")
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--layout-preset", choices=sorted(LAYOUT_PRESETS), default=None,
                    help="Optional convenience preset. Explicit templates override preset values.")
    ap.add_argument("--dataset", choices=sorted(LAYOUT_PRESETS), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--list-layout-presets", action="store_true")

    ap.add_argument("--source-nii-template", dest="source_nii", default=None)
    ap.add_argument("--target-nii-template", dest="target_nii", default=None)
    ap.add_argument("--source-bval-template", dest="source_bval", default=None)
    ap.add_argument("--target-bval-template", dest="target_bval", default=None)
    ap.add_argument("--source-bvec-template", dest="source_bvec", default=None)
    ap.add_argument("--target-bvec-template", dest="target_bvec", default=None)
    ap.add_argument("--out-dir-template", default="preprocessed/{subject}",
                    help="Output path relative to --root. Supports {subject} and {raw_subject}.")
    ap.add_argument("--subject-list", default=None,
                    help="Optional text file of raw subject IDs. If omitted, subjects are discovered from source template.")
    ap.add_argument("--output-subject-regex", default=None,
                    help="Optional regex applied to raw subject ID; first capture group becomes the output subject ID.")

    ap.add_argument("--b-in", type=float, default=1000.0)
    ap.add_argument("--b-out", type=float, default=2000.0)
    ap.add_argument("--b-tol", type=float, default=DEFAULT_B_TOL)
    ap.add_argument("--dot-thr", type=float, default=DEFAULT_DOT_THR)
    ap.add_argument("--st-clip-pct", type=float, default=ST_CLIP_PCT)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.006)
    ap.add_argument("--no-override", action="store_true")
    ap.add_argument("--no-shmeta", action="store_true")
    ap.add_argument("--overwrite-shmeta", action="store_true")
    args = ap.parse_args()

    if args.list_layout_presets:
        _print_presets()
        return

    templates = _collect_templates(args)
    pairs = build_pairs_from_templates(
        args.root,
        templates,
        subject_list=args.subject_list,
        output_subject_regex=args.output_subject_regex,
    )
    preset_msg = args.layout_preset or args.dataset or "custom"
    print(f"[info] layout={preset_msg} pairs={len(pairs)} root={args.root}")
    if not pairs:
        raise SystemExit("No valid subject pairs found.")

    kwargs = dict(
        b_in=args.b_in,
        b_out=args.b_out,
        b_tol=args.b_tol,
        dot_thr=args.dot_thr,
        st_clip_pct=args.st_clip_pct,
        override_observed=not args.no_override,
        make_shmeta=not args.no_shmeta,
        lmax=args.lmax,
        lam=args.lam,
        overwrite_shmeta=args.overwrite_shmeta,
    )
    errors = []
    if args.workers <= 1:
        for item in tqdm(pairs, ncols=120):
            err = _process_item((item, kwargs))
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for err in tqdm(ex.map(_process_item, [(it, kwargs) for it in pairs]), total=len(pairs), ncols=120):
                if err:
                    errors.append(err)
    if errors:
        print(f"[done with errors] {len(errors)} failed")
        for e in errors[:20]:
            print(e)
    else:
        print("[OK] all subjects processed")


if __name__ == "__main__":
    main()
