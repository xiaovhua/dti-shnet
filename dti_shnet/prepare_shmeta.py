from __future__ import annotations

import argparse
import ast
from pathlib import Path
from tqdm import tqdm
from .io import list_subject_dirs
from .protocol import build_shmeta_for_subject


def main():
    ap = argparse.ArgumentParser(description="Build SH metadata files for existing preprocessed subject folders")
    ap.add_argument("--roots", required=True, help="Python list of preprocessed roots")
    ap.add_argument("--b-in", type=float, default=1000.0)
    ap.add_argument("--b-out", type=float, default=2000.0)
    ap.add_argument("--b-tol", type=float, default=50.0)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.006)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    roots = ast.literal_eval(args.roots)
    errors = []
    for sd in tqdm(list_subject_dirs(roots), ncols=120):
        try:
            build_shmeta_for_subject(sd, b_in=args.b_in, b_out=args.b_out, b_tol=args.b_tol, lmax=args.lmax, lam=args.lam, overwrite=args.overwrite)
        except Exception as e:
            errors.append(f"{sd}: {repr(e)}")
    print(f"[done] errors={len(errors)}")
    for e in errors[:30]:
        print(e)

if __name__ == "__main__":
    main()
