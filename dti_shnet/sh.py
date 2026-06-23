from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
try:
    from scipy.special import sph_harm_y as _sph_harm_y
    def _sph_harm(m, l, phi, theta):
        # scipy sph_harm_y uses (n, m, theta, phi)
        return _sph_harm_y(l, m, theta, phi)
except Exception:  # scipy<1.15
    from scipy.special import sph_harm as _old_sph_harm
    def _sph_harm(m, l, phi, theta):
        return _old_sph_harm(m, l, phi, theta)


def _dirs_to_angles(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError(f"directions must be Nx3, got {dirs.shape}")
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.clip(r, 1e-12, None)
    x, y, z = x / r, y / r, z / r
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    return theta, phi


def n_sh_coeffs(lmax: int) -> int:
    if int(lmax) < 0 or int(lmax) % 2 != 0:
        raise ValueError(f"lmax must be a non-negative even integer, got {lmax}")
    return sum((2 * l + 1) for l in range(0, int(lmax) + 1, 2))


def real_sym_sh_basis(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """Real symmetric SH basis used in the original experiments.

    Coefficient order is l=0,2,...,lmax and m=-l,...,l.
    """
    theta, phi = _dirs_to_angles(dirs)
    rows = []
    for l in range(0, int(lmax) + 1, 2):
        for m in range(-l, l + 1):
            Y = _sph_harm(m, l, phi, theta)
            if m == 0:
                Yr = Y.real
            elif m > 0:
                Yr = np.sqrt(2.0) * ((-1.0) ** m) * Y.real
            else:
                Yr = np.sqrt(2.0) * ((-1.0) ** m) * Y.imag
            rows.append(Yr.astype(np.float64))
    return np.stack(rows, axis=1).astype(np.float32)


def laplace_beltrami_diag(lmax: int) -> np.ndarray:
    diag = []
    for l in range(0, int(lmax) + 1, 2):
        val = float(l * (l + 1))
        for _ in range(2 * l + 1):
            diag.append(val)
    return np.asarray(diag, dtype=np.float64)


@dataclass
class SHRegressor:
    P: np.ndarray
    B: np.ndarray
    lmax: int
    lam: float


def build_sh_regressor(dirs: np.ndarray, lmax: int, lam: float) -> SHRegressor:
    B = real_sym_sh_basis(dirs, lmax).astype(np.float64)
    L = np.diag(laplace_beltrami_diag(lmax))
    A = B.T @ B + float(lam) * L
    P = np.linalg.solve(A, B.T)
    return SHRegressor(P=P.astype(np.float32), B=B.astype(np.float32), lmax=int(lmax), lam=float(lam))
