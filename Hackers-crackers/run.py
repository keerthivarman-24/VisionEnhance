"""KLA Track 1 submission inference script.

Required usage after renaming this file to run.py:
    python run.py <input-dir> <output-dir>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from model import RestorationNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore every .npy image in an input directory."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def find_checkpoint():
    """Find the bundled model checkpoint relative to this script."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "models" / "best.pt",
        script_dir / "models" / "model.pt",
    ]

    for checkpoint_path in candidates:
        if checkpoint_path.is_file():
            return checkpoint_path

    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Model checkpoint was not found. Expected {expected}"
    )


def load_input(path):
    """Load and validate one grayscale NumPy input."""
    array = np.load(path, allow_pickle=False)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(
            f"{path.name}: expected shape (H, W) or (H, W, 1), "
            f"but received {array.shape}"
        )

    array = np.asarray(array, dtype=np.float32)

    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: input contains NaN or Inf values")

    return array


def load_model(checkpoint_path, device):
    """Construct the trained architecture and load EMA or raw model weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    width = int(checkpoint_args.get("width", 32))
    num_blocks = int(checkpoint_args.get("num_blocks", 14))
    refine_blocks = int(checkpoint_args.get("refine_blocks", 2))
    upscale = int(checkpoint_args.get("upscale", 2))

    norm_max = checkpoint_args.get("norm_max", 1.0)
    if norm_max is None:
        norm_max = 1.0
    norm_max = float(norm_max)

    if norm_max <= 0:
        raise ValueError(f"Invalid norm_max stored in checkpoint: {norm_max}")

    if upscale != 2:
        raise ValueError(
            f"The submitted model must use 2x upscaling, but checkpoint uses {upscale}x"
        )

    model = RestorationNet(
        in_channels=1,
        width=width,
        num_blocks=num_blocks,
        refine_blocks=refine_blocks,
        upscale=upscale,
    ).to(device)

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, norm_max, upscale


def validate_output(prediction, input_shape, upscale, filename):
    """Validate all organizer-required output properties."""
    expected_shape = (
        input_shape[0] * upscale,
        input_shape[1] * upscale,
    )

    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"{filename}: expected output shape {expected_shape}, "
            f"but received {prediction.shape}"
        )

    if prediction.dtype != np.float32:
        raise RuntimeError(
            f"{filename}: expected float32 output, but received {prediction.dtype}"
        )

    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{filename}: output contains NaN or Inf values")

    minimum = float(prediction.min())
    maximum = float(prediction.max())

    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(
            f"{filename}: output range [{minimum}, {maximum}] is outside [0, 1]"
        )


def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".npy"
    )

    if not input_files:
        raise RuntimeError(f"No .npy files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_checkpoint()
    model, norm_max, upscale = load_model(checkpoint_path, device)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input files: {len(input_files)}")

    with torch.inference_mode():
        for index, input_path in enumerate(input_files, start=1):
            raw = load_input(input_path)
            input_shape = raw.shape

            # The supplied KLA arrays use norm_max=1.0. Valid noisy overshoot is
            # preserved. No per-image normalization or input clipping is applied.
            normalized = raw / norm_max

            tensor = (
                torch.from_numpy(normalized)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            prediction = model(tensor)
            prediction = prediction.squeeze(0).squeeze(0)
            prediction = prediction.float().cpu().numpy()

            # The organizer requires restored output values within [0, 1].
            prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32)

            validate_output(
                prediction,
                input_shape,
                upscale,
                input_path.name,
            )

            output_path = args.output_dir / input_path.name
            np.save(output_path, prediction, allow_pickle=False)

            print(f"[{index}/{len(input_files)}] {input_path.name}")

    print(f"Completed. Restored files saved to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
