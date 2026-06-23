from __future__ import annotations
from typing import Tuple
import numpy as np


def center_crop_pad_3d(vol: np.ndarray, roi: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    assert vol.ndim == 3, vol.shape
    rx, ry, rz = [int(v) for v in roi]
    X, Y, Z = vol.shape
    cx, cy, cz = X // 2, Y // 2, Z // 2
    x0, y0, z0 = cx - rx // 2, cy - ry // 2, cz - rz // 2
    x1, y1, z1 = x0 + rx, y0 + ry, z0 + rz

    sx0, sy0, sz0 = max(x0, 0), max(y0, 0), max(z0, 0)
    sx1, sy1, sz1 = min(x1, X), min(y1, Y), min(z1, Z)
    dx0, dy0, dz0 = sx0 - x0, sy0 - y0, sz0 - z0
    dx1, dy1, dz1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0), dz0 + (sz1 - sz0)

    out = np.full((rx, ry, rz), pad_value, dtype=vol.dtype)
    out[dx0:dx1, dy0:dy1, dz0:dz1] = vol[sx0:sx1, sy0:sy1, sz0:sz1]
    return out


def center_crop_pad_4d(vol4: np.ndarray, roi: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    assert vol4.ndim == 4, vol4.shape
    rx, ry, rz = [int(v) for v in roi]
    X, Y, Z, T = vol4.shape
    out = np.empty((rx, ry, rz, T), dtype=vol4.dtype)
    for t in range(T):
        out[..., t] = center_crop_pad_3d(vol4[..., t], (rx, ry, rz), pad_value)
    return out


def center_slices(full_shape: tuple[int, int, int], roi: tuple[int, int, int]):
    X, Y, Z = [int(v) for v in full_shape]
    rx, ry, rz = [int(v) for v in roi]
    cx, cy, cz = X // 2, Y // 2, Z // 2
    x0, y0, z0 = cx - rx // 2, cy - ry // 2, cz - rz // 2
    x1, y1, z1 = x0 + rx, y0 + ry, z0 + rz

    sx0, sy0, sz0 = max(x0, 0), max(y0, 0), max(z0, 0)
    sx1, sy1, sz1 = min(x1, X), min(y1, Y), min(z1, Z)
    dx0, dy0, dz0 = sx0 - x0, sy0 - y0, sz0 - z0
    dx1, dy1, dz1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0), dz0 + (sz1 - sz0)
    full_sl = (slice(sx0, sx1), slice(sy0, sy1), slice(sz0, sz1))
    roi_sl = (slice(dx0, dx1), slice(dy0, dy1), slice(dz0, dz1))
    return full_sl, roi_sl
