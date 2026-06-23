"""Canonical file names and compatibility aliases for DTI-SHNet.

New preprocessing outputs use semantic names. Legacy names are resolved only
through :func:`resolve_file`, so the rest of the code can remain name-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileNames:
    source_signal: str = "signal_source.nii.gz"
    target_signal: str = "signal_target.nii.gz"
    dti_signal: str = "signal_dti_baseline.nii.gz"
    dti_dwi: str = "dwi_dti_baseline.nii.gz"
    prediction_signal: str = "signal_pred.nii.gz"
    prediction_dwi: str = "dwi_pred.nii.gz"
    target_shell_signal: str = "signal_target_shell.nii.gz"
    target_shell_dwi: str = "dwi_target_shell.nii.gz"
    baseline_shell_signal: str = "signal_dti_baseline_shell.nii.gz"
    baseline_shell_dwi: str = "dwi_dti_baseline_shell.nii.gz"
    s0: str = "S0.nii.gz"
    brain_mask: str = "brain_mask.nii.gz"
    fa: str = "FA.nii.gz"
    md: str = "MD.nii.gz"
    protocol_bval: str = "protocol_full.bval"
    protocol_bvec: str = "protocol_full.bvec"
    source_bval: str = "protocol_source.bval"
    source_bvec: str = "protocol_source.bvec"
    source_to_full_map: str = "source_to_full_map.txt"


FILES = FileNames()

LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    FILES.source_signal: ("St_obs_33.nii.gz",),
    FILES.target_signal: ("St_gt_63.nii.gz",),
    FILES.dti_signal: ("St_syn_63.nii.gz",),
    FILES.dti_dwi: ("synth_multi_63.nii.gz",),
    FILES.protocol_bval: ("synth_multi_63.bval",),
    FILES.protocol_bvec: ("synth_multi_63.bvec",),
    FILES.source_to_full_map: ("single_to_target_map.txt",),
}


def canonical_path(subject_dir: str | Path, filename: str) -> Path:
    return Path(subject_dir) / filename


def candidates(subject_dir: str | Path, filename: str) -> list[Path]:
    sd = Path(subject_dir)
    return [sd / filename] + [sd / x for x in LEGACY_ALIASES.get(filename, ())]


def resolve_file(subject_dir: str | Path, filename: str, *, required: bool = True) -> Path | None:
    """Return an existing canonical file path, falling back to supported aliases."""
    for p in candidates(subject_dir, filename):
        if p.exists():
            return p
    if required:
        tried = ", ".join(str(p.name) for p in candidates(subject_dir, filename))
        raise FileNotFoundError(f"Missing required file in {subject_dir}: tried {tried}")
    return None


def shmeta_name(b_in: int | float, b_out: int | float, lmax: int, lam: float) -> str:
    return f"shmeta_b{int(b_in)}_b{int(b_out)}_l{int(lmax)}_lam{lam}.npz"


def read_id_list(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def id_in_path(ids: Iterable[str], path: str | Path) -> bool:
    s = str(path)
    return any(eid and eid in s for eid in ids)
