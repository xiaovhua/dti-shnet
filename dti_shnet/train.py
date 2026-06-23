from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DTISHNetPatchDataset
from .models.unet3d import UNet3DReg
from .sh import real_sym_sh_basis, n_sh_coeffs


def parse_tuple3(s: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in s.split(","))  # type: ignore[return-value]


def collate(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    m = torch.stack([b[2] for b in batch], dim=0)
    return x, y, m


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad * quad + delta * lin


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2 * (q2*q2 + q3*q3), 2 * (q1*q2 - q3*q4), 2 * (q1*q3 + q2*q4)],
        [2 * (q1*q2 + q3*q4), 1 - 2 * (q1*q1 + q3*q3), 2 * (q2*q3 - q1*q4)],
        [2 * (q1*q3 - q2*q4), 2 * (q2*q3 + q1*q4), 1 - 2 * (q1*q1 + q2*q2)],
    ], dtype=np.float32)


def fibonacci_sphere(n: int, rng: np.random.Generator, random_rotate: bool = True) -> np.ndarray:
    rnd = 1.0
    pts = []
    offset = 2.0 / int(n)
    inc = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(int(n)):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(0.0, 1 - y * y))
        phi = (i + rnd) * inc
        pts.append([np.cos(phi) * r, y, np.sin(phi) * r])
    P = np.asarray(pts, dtype=np.float32)
    if random_rotate:
        P = (P @ _random_rotation_matrix(rng).T).astype(np.float32)
    return P


def main():
    ap = argparse.ArgumentParser(
        description="Train DTI-SHNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--roots", required=True, help="Python list of preprocessed roots, e.g. \"['/data/preprocessed']\"")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-id-file", default="splits/eval_id.txt")
    ap.add_argument("--roi", default="96,96,64")
    ap.add_argument("--patch", default="32,32,32")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pps-train", type=int, default=12)
    ap.add_argument("--pps-val", type=int, default=1)
    ap.add_argument("--shuffle-patches", type=int, default=0, choices=[0, 1])
    ap.add_argument("--seed-per-epoch", type=int, default=1, choices=[0, 1])
    ap.add_argument("--amp", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-dti", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use-sig-loss", type=int, default=1, choices=[0, 1])
    ap.add_argument("--b-in", type=int, default=1000)
    ap.add_argument("--b-out", type=int, default=2000)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.006)
    ap.add_argument("--lambda-sig", type=float, default=0.1)
    ap.add_argument("--canon-n", type=int, default=60)
    args = ap.parse_args()

    roots = ast.literal_eval(args.roots)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_tuple3(args.roi)
    patch = parse_tuple3(args.patch)
    rng = np.random.default_rng(args.seed)

    if int(args.use_sig_loss) == 0:
        args.lambda_sig = 0.0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_tr = DTISHNetPatchDataset(
        roots, eval_id_file=args.eval_id_file, roi=roi, patch=patch,
        b_in=args.b_in, b_out=args.b_out, lmax=args.lmax, lam=args.lam,
        split="train", patches_per_subject=args.pps_train, seed=args.seed,
        cache_subjects=1, use_dti=bool(args.use_dti),
    )
    ds_va = DTISHNetPatchDataset(
        roots, eval_id_file=args.eval_id_file, roi=roi, patch=patch,
        b_in=args.b_in, b_out=args.b_out, lmax=args.lmax, lam=args.lam,
        split="val", patches_per_subject=args.pps_val, seed=args.seed,
        cache_subjects=1, use_dti=bool(args.use_dti),
    )
    if len(ds_tr) == 0:
        raise RuntimeError("No training data found. Run preprocessing and make sure eval_id.txt is correct.")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=bool(args.shuffle_patches), num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True, drop_last=False, collate_fn=collate)

    K = n_sh_coeffs(args.lmax)
    in_ch = K + 3 if bool(args.use_dti) else K
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3DReg(in_channels=in_ch, out_channels=K, base=args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    meta = vars(args).copy()
    meta.update({"roots": roots, "roi": roi, "patch": patch, "K": K, "in_channels": in_ch, "optimizer": "AdamW", "device": str(device)})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cfg] device={device} K={K} in_ch={in_ch} use_dti={args.use_dti} lambda_sig={args.lambda_sig} steps={len(dl_tr)}")

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        if int(args.seed_per_epoch) == 1:
            ds_tr.seed = int(args.seed) + ep * 1000003
        model.train()
        tr, ntr = 0.0, 0
        for x, y, m in tqdm(dl_tr, desc=f"train ep{ep}", ncols=110):
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            if bool(args.use_dti):
                x = torch.cat([x, m.float()], dim=1)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                # Coefficient loss: mask-weighted difference, averaged over the full patch tensor.
                loss_coef = huber((pred - y) * m, delta=1.0).mean()
                if float(args.lambda_sig) > 0:
                    Bsz, Kc, px, py, pz = pred.shape
                    V = px * py * pz
                    pred_vk = pred.reshape(Bsz, Kc, V).permute(0, 2, 1).reshape(Bsz * V, Kc)
                    gt_vk = y.reshape(Bsz, Kc, V).permute(0, 2, 1).reshape(Bsz * V, Kc)
                    dirs = fibonacci_sphere(args.canon_n, rng=rng, random_rotate=True)
                    Bcan = torch.from_numpy(real_sym_sh_basis(dirs, args.lmax).astype(np.float32)).to(device)
                    Sp = (Bcan @ pred_vk.T).T
                    Sg = (Bcan @ gt_vk.T).T
                    # Signal-consistency loss is unmasked and averaged over batch, voxels, and canonical directions.
                    loss_sig = torch.mean(torch.abs(Sp - Sg))
                else:
                    loss_sig = torch.tensor(0.0, device=device)
                loss = loss_coef + float(args.lambda_sig) * loss_sig
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr += float(loss.item()) * x.shape[0]
            ntr += x.shape[0]
        tr /= max(ntr, 1)

        model.eval()
        va, nva = 0.0, 0
        with torch.no_grad():
            for x, y, m in tqdm(dl_va, desc=f"val ep{ep}", ncols=110):
                x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
                if bool(args.use_dti):
                    x = torch.cat([x, m.float()], dim=1)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(x)
                    loss = huber((pred - y) * m, delta=1.0).mean()
                va += float(loss.item()) * x.shape[0]
                nva += x.shape[0]
        va /= max(nva, 1)
        ck = {"model": model.state_dict(), "meta": meta, "epoch": ep, "train": tr, "val": va}
        torch.save(ck, out_dir / "last.pt")
        if va < best:
            best = va
            torch.save(ck, out_dir / "best.pt")
        print(f"[ep {ep}] train={tr:.6f} val={va:.6f} best={best:.6f}")
    print(f"[done] best={best:.6f}; final checkpoint={out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
