from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .constants import FILES, read_id_list, id_in_path, resolve_file
from .io import list_subject_dirs, load_nifti
from .protocol import load_shmeta
from .sh import build_sh_regressor, n_sh_coeffs
from .spatial import center_crop_pad_3d, center_crop_pad_4d

EPS = 1e-6


@dataclass
class SubjectCache:
    mask: np.ndarray
    fa: np.ndarray
    md: np.ndarray
    P_in: np.ndarray
    P_out: np.ndarray
    C_in: Optional[np.ndarray] = None
    C_out: Optional[np.ndarray] = None


class _LRU:
    def __init__(self, max_items: int = 1):
        self.max_items = int(max_items)
        self.d: dict[str, SubjectCache] = {}

    def get(self, key: str) -> Optional[SubjectCache]:
        return self.d.get(key)

    def put(self, key: str, value: SubjectCache) -> None:
        if key in self.d:
            self.d.pop(key)
        self.d[key] = value
        while len(self.d) > self.max_items:
            oldest = next(iter(self.d.keys()))
            self.d.pop(oldest)


def _compute_coeff_vol(signal: np.ndarray, mask: np.ndarray, P: np.ndarray) -> np.ndarray:
    X, Y, Z, N = signal.shape
    K = P.shape[0]
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return np.zeros((K, X, Y, Z), dtype=np.float16)
    S = signal[mask].reshape(idx.size, N).astype(np.float32)
    C = (P @ S.T).T
    flat = np.zeros((K, X * Y * Z), dtype=np.float16)
    flat[:, idx] = C.T.astype(np.float16)
    return flat.reshape(K, X, Y, Z)


def _scale_fa_md(subject_dir: Path, roi: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    _, FA = load_nifti(resolve_file(subject_dir, FILES.fa))
    _, MD = load_nifti(resolve_file(subject_dir, FILES.md))
    if FA.ndim == 4 and FA.shape[-1] == 1:
        FA = FA[..., 0]
    if MD.ndim == 4 and MD.shape[-1] == 1:
        MD = MD[..., 0]
    FA = center_crop_pad_3d(FA.astype(np.float32), roi, pad_value=0.0)
    MD = center_crop_pad_3d(MD.astype(np.float32), roi, pad_value=0.0)
    FA = 2.0 * np.clip(FA, 0.0, 1.0) - 1.0
    MD = np.clip(MD, 0.0, 0.003) / 0.003
    MD = 2.0 * MD - 1.0
    return FA.astype(np.float32), MD.astype(np.float32)


class DTISHNetPatchDataset(Dataset):
    """Patch dataset matching the original DTI-SHNet implementation.

    Source/target normalized signal volumes are converted to SH coefficient
    volumes on the fly.  Coefficients are fitted only inside the preprocessing
    mask.  If ``use_dti=True``, the dataset returns C_source + FA + MD; the
    training loop appends the mask as an additional input channel, reproducing
    the original final model (K+3 input channels for lmax=8 -> 48 channels).
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        eval_id_file: str | Path = "eval_id.txt",
        roi: tuple[int, int, int] = (96, 96, 64),
        patch: tuple[int, int, int] = (32, 32, 32),
        b_in: int = 1000,
        b_out: int = 2000,
        lmax: int = 8,
        lam: float = 0.006,
        split: str = "train",
        use_dti: bool = True,
        patches_per_subject: int = 8,
        seed: int = 0,
        cache_subjects: int = 1,
    ):
        self.roi = tuple(int(x) for x in roi)
        self.patch = tuple(int(x) for x in patch)
        self.b_in = int(b_in)
        self.b_out = int(b_out)
        self.lmax = int(lmax)
        self.lam = float(lam)
        self.use_dti = bool(use_dti)
        self.pps = int(patches_per_subject)
        self.seed = int(seed)
        self.cache = _LRU(cache_subjects)
        self.K = n_sh_coeffs(self.lmax)

        eval_ids = read_id_list(eval_id_file)
        subjs = list_subject_dirs(roots)
        if split == "train":
            subjs = [s for s in subjs if not id_in_path(eval_ids, s)]
        elif split in {"val", "test"}:
            subjs = [s for s in subjs if id_in_path(eval_ids, s)]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split={split}")

        valid = []
        for sd in subjs:
            try:
                resolve_file(sd, FILES.brain_mask)
                resolve_file(sd, FILES.source_signal)
                resolve_file(sd, FILES.target_signal)
                if self.use_dti:
                    resolve_file(sd, FILES.fa)
                    resolve_file(sd, FILES.md)
                if load_shmeta(sd, self.b_in, self.b_out, self.lmax, self.lam) is None:
                    continue
                valid.append(sd)
            except FileNotFoundError:
                continue
        self.valid_subjs = sorted(valid)
        print(f"[dataset] split={split} candidates={len(subjs)} valid={len(self.valid_subjs)} K={self.K}")

    def __len__(self) -> int:
        return len(self.valid_subjs) * self.pps

    def _load_subject(self, sd: Path) -> SubjectCache:
        key = str(sd)
        got = self.cache.get(key)
        if got is not None:
            return got

        shmeta = load_shmeta(sd, self.b_in, self.b_out, self.lmax, self.lam)
        if shmeta is None:
            raise RuntimeError(f"Missing shmeta for {sd}")
        in_i, in_dirs, out_idx, out_dirs = shmeta

        _, mask = load_nifti(resolve_file(sd, FILES.brain_mask))
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        mask = center_crop_pad_3d((mask > 0.5).astype(np.float32), self.roi, pad_value=0.0) > 0.5

        if self.use_dti:
            FA, MD = _scale_fa_md(sd, self.roi)
        else:
            FA = np.zeros(self.roi, dtype=np.float32)
            MD = np.zeros(self.roi, dtype=np.float32)

        img_src = nib.load(str(resolve_file(sd, FILES.source_signal)))
        src_all = np.asarray(img_src.dataobj, dtype=np.float32)
        src = center_crop_pad_4d(src_all[..., in_i], self.roi, pad_value=0.0)
        src = np.clip(src, 0.0, None).astype(np.float32)

        img_tgt = nib.load(str(resolve_file(sd, FILES.target_signal)))
        tgt_all = np.asarray(img_tgt.dataobj, dtype=np.float32)
        tgt = center_crop_pad_4d(tgt_all[..., out_idx], self.roi, pad_value=0.0)
        tgt = np.clip(tgt, 0.0, None).astype(np.float32)

        reg_in = build_sh_regressor(in_dirs, lmax=self.lmax, lam=self.lam)
        reg_out = build_sh_regressor(out_dirs, lmax=self.lmax, lam=self.lam)

        cache = SubjectCache(mask=mask, fa=FA, md=MD, P_in=reg_in.P, P_out=reg_out.P)
        cache.C_in = _compute_coeff_vol(src, mask, cache.P_in)
        cache.C_out = _compute_coeff_vol(tgt, mask, cache.P_out)
        self.cache.put(key, cache)
        return cache

    def __getitem__(self, idx: int):
        subj_i = idx // self.pps
        sd = self.valid_subjs[subj_i]
        cache = self._load_subject(sd)
        assert cache.C_in is not None and cache.C_out is not None

        K, X, Y, Z = cache.C_in.shape
        px, py, pz = self.patch
        px, py, pz = min(px, X), min(py, Y), min(pz, Z)
        rng = np.random.default_rng(self.seed + idx * 10007)
        x0 = int(rng.integers(0, X - px + 1)) if X > px else 0
        y0 = int(rng.integers(0, Y - py + 1)) if Y > py else 0
        z0 = int(rng.integers(0, Z - pz + 1)) if Z > pz else 0

        Cin = cache.C_in[:, x0:x0+px, y0:y0+py, z0:z0+pz].astype(np.float32)
        Cout = cache.C_out[:, x0:x0+px, y0:y0+py, z0:z0+pz].astype(np.float32)
        m = cache.mask[x0:x0+px, y0:y0+py, z0:z0+pz].astype(np.float32)[None]

        if self.use_dti:
            fa = cache.fa[x0:x0+px, y0:y0+py, z0:z0+pz].astype(np.float32)[None]
            md = cache.md[x0:x0+px, y0:y0+py, z0:z0+pz].astype(np.float32)[None]
            x = np.concatenate([Cin, fa, md], axis=0).astype(np.float32)
        else:
            x = Cin.astype(np.float32)

        return (
            torch.from_numpy(np.ascontiguousarray(x)).clone(),
            torch.from_numpy(np.ascontiguousarray(Cout)).clone(),
            torch.from_numpy(np.ascontiguousarray(m)).clone(),
        )
