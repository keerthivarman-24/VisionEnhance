"""
Standalone evaluation script -- Track 1 (KLA) submission. Works on the .npy
Test_NoisyLR format.

Usage (minimum, matches the hackathon spec exactly):
    python inference.py --input_dir dataset/Test_NoisyLR --output_dir ./restored_outputs

Optional overrides:
    --checkpoint <path>       default: checkpoints/best.pt
    --norm_max <float>        overrides the value stored in the checkpoint
    --width / --num_blocks    overrides the values stored in the checkpoint
    --save_png                also write an 8-bit PNG preview alongside each .npy
    --save_comparison         also save a 2-panel (or 3-panel with GT) comparison
                               plot per image, under <output_dir>/comparison_plots/
    --gt_dir <path>           optional GT folder (for --save_comparison 3-panel view;
                               only meaningful on your own held-out split, not the
                               real Test_NoisyLR which has no GT)
"""

import argparse
import os
import time

import numpy as np
import torch

from dataset import load_raw
from model import RestorationNet


def save_outputs(pred_arr, scale, out_path_npy, save_png, out_path_png=None):
    # pred_arr is in normalized [0,1] space (clamped) -- scale back to the same
    # numeric range the input .npy files were in, so outputs are directly
    # comparable to held-out GT .npy files by KLA's own eval script.
    restored = pred_arr * scale
    np.save(out_path_npy, restored.astype(np.float32))

    if save_png:
        from PIL import Image
        preview = np.clip(pred_arr, 0.0, 1.0)
        Image.fromarray((preview * 255).astype(np.uint8)).save(out_path_png)


def save_comparison_plot(input_arr, pred_arr, out_path, gt_arr=None):
    import matplotlib
    matplotlib.use("Agg")  # headless, no display needed
    import matplotlib.pyplot as plt

    panels = [("Input (low resolution)", input_arr), ("Generated (upscaled)", pred_arr)]
    if gt_arr is not None:
        panels.insert(0, ("Ground truth", gt_arr))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), cmap="gray")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_comparison", action="store_true")
    parser.add_argument("--gt_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(args.output_dir, "comparison_plots"), exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    width = args.width if args.width is not None else ckpt_args.get("width", 32)
    num_blocks = args.num_blocks if args.num_blocks is not None else ckpt_args.get("num_blocks", 14)
    norm_max = args.norm_max if args.norm_max is not None else ckpt_args.get("norm_max", None)

    if norm_max is None:
        raise ValueError(
            "No norm_max found in the checkpoint and none passed via --norm_max.\n"
            "Refusing to silently auto-detect a per-image scale here -- that's what "
            "produces garbage output on test images, since there's no ground truth "
            "to anchor the guess to at inference time.\n"
            "Fix: retrain with an explicit --norm_max, or pass it here directly."
        )

    model = RestorationNet(width=width, num_blocks=num_blocks).to(device)
    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (width={width}, num_blocks={num_blocks}, "
          f"norm_max={norm_max}, device={device})")

    files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
    print(f"Found {len(files)} .npy images in {args.input_dir}")

    total_time = 0.0
    with torch.no_grad():
        for fname in files:
            # Timer starts here: covers disk read, preprocessing, CPU->GPU transfer,
            # model execution, GPU->CPU transfer, post-processing, and saving --
            # matching KLA's stated runtime definition exactly, not just the
            # forward pass.
            t0 = time.time()

            in_path = os.path.join(args.input_dir, fname)
            raw = load_raw(in_path)
            arr = raw / norm_max
            # Diagnosed cliff: inputs with max > ~1.50 reliably collapse the model
            # into degenerate black output -- training overshoot only ever reached
            # ~1.3-1.4. Clip to the empirically-safe range as a stopgap until a
            # retrain with redegradation active removes the underlying cause.
            if arr.max() > 1.45:
                arr = np.clip(arr, -0.1, 1.45)

            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

            pred = model(x)
            if device == "cuda":
                torch.cuda.synchronize()

            pred = torch.clamp(pred, 0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy()

            stem = os.path.splitext(fname)[0]
            out_npy = os.path.join(args.output_dir, f"{stem}.npy")
            out_png = os.path.join(args.output_dir, f"{stem}.png") if args.save_png else None
            save_outputs(pred, norm_max, out_npy, args.save_png, out_png)

            elapsed = time.time() - t0
            total_time += elapsed

            # Comparison plots are a reporting convenience, not part of the
            # required pipeline -- generated AFTER the timer stops so they don't
            # inflate the reported throughput number.
            if args.save_comparison:
                gt_arr = None
                if args.gt_dir:
                    gt_path = os.path.join(args.gt_dir, fname)
                    if os.path.isfile(gt_path):
                        gt_arr = load_raw(gt_path) / norm_max
                plot_path = os.path.join(args.output_dir, "comparison_plots", f"{stem}.png")
                save_comparison_plot(arr, pred, plot_path, gt_arr=gt_arr)

    if files:
        print(f"Done. {len(files)} images restored. "
              f"Avg END-TO-END time (disk read -> preprocess -> GPU -> model -> "
              f"CPU -> save): {total_time / len(files) * 1000:.1f} ms/image "
              f"(total {total_time:.2f}s)")
    else:
        print("No .npy files found in --input_dir -- nothing to do.")


if __name__ == "__main__":
    main()