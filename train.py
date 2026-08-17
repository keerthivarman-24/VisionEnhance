import argparse
import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import RestorationDataset
from losses import CompositeLoss
from model import RestorationNet


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def compute_ssim_metric(pred, target, ssim_loss_module):
    with torch.no_grad():
        return 1.0 - ssim_loss_module(pred, target).item()


class EMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(m_param.detach(), alpha=1 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for deg, gt in val_loader:
        deg, gt = deg.to(device), gt.to(device)
        pred = torch.clamp(model(deg), 0.0, 1.0)
        total_psnr += compute_psnr(pred, gt)
        total_ssim += compute_ssim_metric(pred, gt, criterion.ssim)
        n += 1
    return total_psnr / max(1, n), total_ssim / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=64,
                         help="Patch size in DEGRADED-image pixels; GT patch is this * upscale")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp", action="store_true",
                         help="Enable fp16 mixed precision. OFF by default -- fp16 "
                              "overflow is the root cause of every NaN seen so far "
                              "in this pipeline; fp32 trades some speed for "
                              "guaranteed numerical stability, which matters more "
                              "under a hard deadline. Only enable this if training "
                              "time genuinely becomes the bottleneck.")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=14)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--redegradation_prob", type=float, default=0.3,
                         help="Probability of synthesizing a fresh degraded patch "
                              "from GT during training, on top of the fixed pairs. "
                              "Was accidentally hardcoded to 0.0 in earlier runs -- "
                              "this is the mechanism meant to help generalization "
                              "to noise/content the fixed training pairs don't cover.")
    parser.add_argument("--norm_max", type=float, default=None,
                         help="Pixel-value normalization constant. Run "
                              "`python dataset.py --root <path> --inspect` first to "
                              "determine this -- see dataset.py docstring.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_train_view = RestorationDataset(
        args.data_root,
        patch_size_degraded=args.patch_size,
        train=True,
        redegradation_prob=args.redegradation_prob,
        norm_max=args.norm_max,
    )
    full_val_view = RestorationDataset(args.data_root, patch_size_degraded=args.patch_size,
                                        train=False, norm_max=args.norm_max)

    n = len(full_train_view)
    n_val = max(1, int(n * args.val_split))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train_view, train_idx)
    val_ds = Subset(full_val_view, val_idx)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = RestorationNet(width=args.width, num_blocks=args.num_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, max(1, total_steps // 20))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=0.999)

    start_epoch = 0
    best_ssim = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {}
        for i, (deg, gt) in enumerate(train_loader):
            deg, gt = deg.to(device, non_blocking=True), gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(deg)
                pred = torch.clamp(pred, 0.0, 1.0)
                loss, parts = criterion(pred, gt)

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at epoch {epoch} step {i + 1} "
                      f"-- skipping this batch, not stepping the optimizer")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"WARNING: non-finite GRADIENT at epoch {epoch} step {i + 1} "
                      f"(loss was finite) -- skipping this batch")
                weights_ok = all(torch.isfinite(p).all() for p in model.parameters())
                print(f"  Weight integrity check: "
                      f"{'OK, still finite' if weights_ok else 'CORRUPTED -- weights are already NaN, kill this run'}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v

            if (i + 1) % args.log_every == 0:
                avg = {k: v / (i + 1) for k, v in running.items()}
                lr_now = scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {i + 1}/{len(train_loader)} "
                      f"loss={avg['total']:.4f} (char={avg['charbonnier']:.4f} "
                      f"ssim={avg['ssim']:.4f} grad={avg['gradient']:.4f} "
                      f"fft={avg['fft']:.4f}) lr={lr_now:.2e}")

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

        if (epoch + 1) % args.val_every == 0:
            val_psnr, val_ssim = validate(ema.shadow, val_loader, criterion, device)
            print(f"[val] epoch {epoch} PSNR={val_psnr:.2f} dB  SSIM={val_ssim:.4f}")

            ckpt = {
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_ssim": best_ssim,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                ckpt["best_ssim"] = best_ssim
                torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
                print(f"New best SSIM: {best_ssim:.4f} -- saved best.pt")


if __name__ == "__main__":
    main()