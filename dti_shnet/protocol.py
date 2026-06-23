from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np

from .constants import FILES, canonical_path, resolve_file, shmeta_name
from .io import load_bval, load_bvec

B0_THR = 50.0
DEFAULT_B_TOL = 50.0
DEFAULT_DOT_THR = 0.985
DIR_COS_TOL_DEFAULT = 0.999


def norm_bvecs(bvals: np.ndarray, bvecs: np.ndarray, b0_thr: float = B0_THR) -> np.ndarray:
    B = np.asarray(bvecs, dtype=np.float64).copy()
    for i, b in enumerate(np.asarray(bvals).reshape(-1)):
        if float(b) <= float(b0_thr):
            B[i] = 0.0
        else:
            n = float(np.linalg.norm(B[i])) + 1e-8
            B[i] /= n
    return B.astype(np.float32)


def design_matrix(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    g = np.asarray(bvecs, dtype=np.float32)
    b = np.asarray(bvals, dtype=np.float32).reshape(-1, 1)
    A = np.stack([
        b[:, 0] * g[:, 0] * g[:, 0],
        b[:, 0] * g[:, 1] * g[:, 1],
        b[:, 0] * g[:, 2] * g[:, 2],
        2.0 * b[:, 0] * g[:, 0] * g[:, 1],
        2.0 * b[:, 0] * g[:, 0] * g[:, 2],
        2.0 * b[:, 0] * g[:, 1] * g[:, 2],
    ], axis=1).astype(np.float32)
    return A


@dataclass
class MatchReport:
    n_source: int
    n_target: int
    n_matched: int
    n_unmatched: int
    mean_abs_dot: float
    min_abs_dot: float
    b_tol: float
    dot_thr: float


def match_source_to_full_protocol(
    bval_source: np.ndarray,
    bvec_source: np.ndarray,
    bval_full: np.ndarray,
    bvec_full: np.ndarray,
    *,
    b_tol: float = DEFAULT_B_TOL,
    dot_thr: float = DEFAULT_DOT_THR,
    b0_thr: float = B0_THR,
) -> tuple[np.ndarray, MatchReport]:
    Ns = int(bval_source.shape[0])
    Nt = int(bval_full.shape[0])
    mapping = -np.ones((Ns,), dtype=np.int32)
    absdots: list[float] = []

    is_b0_s = bval_source <= b0_thr
    is_b0_t = bval_full <= b0_thr
    b0_s_idx = np.where(is_b0_s)[0]
    b0_t_idx = np.where(is_b0_t)[0]
    for k, i in enumerate(b0_s_idx):
        if k < len(b0_t_idx):
            mapping[i] = int(b0_t_idx[k])

    for i in range(Ns):
        if is_b0_s[i]:
            continue
        bi = float(bval_source[i])
        gi = bvec_source[i].astype(np.float32)
        if np.linalg.norm(gi) < 1e-6:
            continue
        cand = np.where(np.abs(bval_full - bi) <= float(b_tol))[0]
        if cand.size == 0:
            continue
        Gc = bvec_full[cand].astype(np.float32)
        norms = np.linalg.norm(Gc, axis=1)
        valid = norms > 1e-6
        if not np.any(valid):
            continue
        cand = cand[valid]
        Gc = Gc[valid] / (norms[valid][:, None] + 1e-8)
        dots = np.abs(Gc @ gi)
        best_i = int(np.argmax(dots))
        if float(dots[best_i]) >= float(dot_thr):
            mapping[i] = int(cand[best_i])
            absdots.append(float(dots[best_i]))

    n_matched = int(np.sum(mapping >= 0))
    rep = MatchReport(
        n_source=Ns,
        n_target=Nt,
        n_matched=n_matched,
        n_unmatched=int(Ns - n_matched),
        mean_abs_dot=float(np.mean(absdots)) if absdots else 0.0,
        min_abs_dot=float(np.min(absdots)) if absdots else 0.0,
        b_tol=float(b_tol),
        dot_thr=float(dot_thr),
    )
    return mapping, rep


def write_mapping_txt(path: str | Path, mapping: np.ndarray, rep: MatchReport | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if rep is not None:
            f.write("source_index -> full_protocol_index (-1 means unmatched)\n")
            f.write(f"n_source={rep.n_source} n_target={rep.n_target} matched={rep.n_matched} unmatched={rep.n_unmatched}\n")
            f.write(f"b_tol={rep.b_tol} dot_thr={rep.dot_thr} mean_abs_dot={rep.mean_abs_dot:.6f} min_abs_dot={rep.min_abs_dot:.6f}\n\n")
        for i, j in enumerate(mapping.tolist()):
            f.write(f"{i} -> {int(j)}\n")


def parse_mapping_txt(path: str | Path) -> np.ndarray:
    pat = re.compile(r"^(\d+)\s*->\s*(-?\d+)\s*$")
    pairs: list[tuple[int, int]] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.match(ln.strip())
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    if not pairs:
        raise ValueError(f"No mapping lines parsed from {path}")
    pairs.sort(key=lambda x: x[0])
    mapping = -np.ones((pairs[-1][0] + 1,), dtype=np.int64)
    for i, j in pairs:
        mapping[i] = j
    return mapping


def build_shmeta_for_subject(
    subject_dir: str | Path,
    *,
    b_in: float = 1000.0,
    b_out: float = 2000.0,
    b_tol: float = DEFAULT_B_TOL,
    lmax: int = 8,
    lam: float = 0.006,
    overwrite: bool = False,
) -> Path:
    sd = Path(subject_dir)
    p_bval = resolve_file(sd, FILES.protocol_bval)
    p_bvec = resolve_file(sd, FILES.protocol_bvec)
    p_map = resolve_file(sd, FILES.source_to_full_map)

    # Resolve these to assert the subject is trainable.
    resolve_file(sd, FILES.source_signal)
    resolve_file(sd, FILES.target_signal)

    outp = sd / shmeta_name(b_in, b_out, lmax, lam)
    if outp.exists() and not overwrite:
        return outp

    bval = load_bval(p_bval)
    bvec = norm_bvecs(bval, load_bvec(p_bvec))
    mapping = parse_mapping_txt(p_map)

    in_i: list[int] = []
    in_dirs: list[np.ndarray] = []
    for i in range(mapping.shape[0]):
        j = int(mapping[i])
        if j < 0 or j >= bval.shape[0]:
            continue
        if abs(float(bval[j]) - float(b_in)) <= float(b_tol):
            in_i.append(i)
            in_dirs.append(bvec[j])
    if len(in_i) < 6:
        raise RuntimeError(f"not enough input frames for b={b_in}: {len(in_i)} in {sd}")

    out_idx = np.where(np.abs(bval - float(b_out)) <= float(b_tol))[0].astype(np.int64)
    if out_idx.size < 6:
        raise RuntimeError(f"not enough output frames for b={b_out}: {out_idx.size} in {sd}")
    out_dirs = bvec[out_idx]

    np.savez_compressed(
        outp,
        in_i=np.asarray(in_i, dtype=np.int64),
        in_dirs=np.asarray(in_dirs, dtype=np.float32),
        out_idx=out_idx.astype(np.int64),
        out_dirs=out_dirs.astype(np.float32),
        meta=json.dumps({"b_in": float(b_in), "b_out": float(b_out), "b_tol": float(b_tol), "lmax": int(lmax), "lam": float(lam)}),
    )
    return outp


def load_shmeta(subject_dir: str | Path, b_in: int, b_out: int, lmax: int, lam: float):
    p = Path(subject_dir) / shmeta_name(b_in, b_out, lmax, lam)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as f:
        return (
            f["in_i"].astype(np.int64),
            f["in_dirs"].astype(np.float32),
            f["out_idx"].astype(np.int64),
            f["out_dirs"].astype(np.float32),
        )
