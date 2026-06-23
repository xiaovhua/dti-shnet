from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import nibabel as nib


def list_subject_dirs(roots: Sequence[str | Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for d in rp.iterdir():
            if d.is_dir():
                out.append(d)
    return sorted(out, key=lambda p: (str(p.parent), p.name))


def load_nifti(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data.astype(np.float32)


def save_nifti_like(data: np.ndarray, like_img: nib.Nifti1Image, out_path: str | Path, *, dtype=np.float32) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = like_img.header.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), like_img.affine, hdr)
    nib.save(img, str(out_path))


def load_bval(path: str | Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float64).reshape(-1)


def load_bvec(path: str | Path) -> np.ndarray:
    rows = [ln.strip().split() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    B = np.asarray(rows, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(f"bvec file malformed: {path} -> {B.shape}")
    if B.shape[0] == 3 and B.shape[1] >= 6:
        B = B.T
    if B.shape[1] != 3:
        raise ValueError(f"bvec must be Nx3 or 3xN, got {B.shape}: {path}")
    return B.astype(np.float64)


def write_bval(path: str | Path, bvals: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), np.asarray(bvals, dtype=np.float64).reshape(1, -1), fmt="%.8g")


def write_bvec(path: str | Path, bvecs: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError(f"bvecs should be Nx3, got {bvecs.shape}")
    np.savetxt(str(path), bvecs.T, fmt="%.8g")


def take_volumes(img: nib.Nifti1Image, idx: np.ndarray) -> np.ndarray:
    """Read selected 4D frames without relying on nibabel fancy indexing."""
    proxy = img.dataobj
    vols = [np.asarray(proxy[..., int(i)], dtype=np.float32) for i in idx.tolist()]
    return np.stack(vols, axis=-1).astype(np.float32)
