import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_raw(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale array, got "
            f"shape={arr.shape}: {path}"
        )

    if not np.isfinite(arr).all():
        bad_count = int(
            arr.size - np.isfinite(arr).sum()
        )

        raise ValueError(
            f"{path} contains {bad_count} NaN/Inf values"
        )

    return arr


def auto_detect_scale(max_val):
    if max_val <= 1.5:
        return 1.0
    for bucket in (255.0, 1023.0, 4095.0, 16383.0, 65535.0):
        if max_val <= bucket * 1.05:
            return bucket
    return max_val


def load_grayscale_normalized(path, norm_max=None):
    arr = load_raw(path)
    scale = norm_max if norm_max is not None else auto_detect_scale(arr.max())
    return arr / scale

class RestorationDataset(Dataset):
    def __init__(self, root, degraded_subdir="NoisyLR", gt_subdir="GT",
                 patch_size_degraded=64, upscale=2, train=True,
                 redegradation_prob=0.0, norm_max=None):
        self.upscale = upscale
        self.patch_size_degraded = patch_size_degraded
        self.train = train
        self.redegradation_prob = redegradation_prob if train else 0.0
        self.norm_max = norm_max

        degraded_dir = os.path.join(root, degraded_subdir)
        gt_dir = os.path.join(root, gt_subdir)

        degraded_files = sorted(glob.glob(os.path.join(degraded_dir, "*")))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*")))

        if len(degraded_files) == 0 or len(gt_files) == 0:
            raise FileNotFoundError(
                f"Found {len(degraded_files)} files in '{degraded_dir}' and "
                f"{len(gt_files)} in '{gt_dir}'. Check --root and the subdir names "
                f"(degraded_subdir / gt_subdir) match your actual dataset layout."
            )
        if len(degraded_files) != len(gt_files):
            raise ValueError(
                f"Mismatched pair counts: {len(degraded_files)} degraded vs "
                f"{len(gt_files)} ground truth in {root}."
            )

        deg_stems = [os.path.splitext(os.path.basename(f))[0] for f in degraded_files]
        gt_stems = [os.path.splitext(os.path.basename(f))[0] for f in gt_files]

        if set(deg_stems) == set(gt_stems):
            self.degraded_files = sorted(degraded_files,
                                          key=lambda f: os.path.splitext(os.path.basename(f))[0])
            self.gt_files = sorted(gt_files,
                                    key=lambda f: os.path.splitext(os.path.basename(f))[0])
        else:
            print(f"[RestorationDataset] WARNING: filenames don't match between "
                  f"'{degraded_subdir}' and '{gt_subdir}' -- falling back to "
                  f"sorted-order pairing. Verify this is correct for your dataset!")
            self.degraded_files = degraded_files
            self.gt_files = gt_files

        print(f"[RestorationDataset] {len(self.degraded_files)} pairs from {root} "
              f"({'train' if train else 'val'} mode)")

    def __len__(self):
        return len(self.degraded_files)

    @staticmethod
    def _dihedral_augment(deg, gt):
        if random.random() < 0.5:
            deg, gt = np.fliplr(deg), np.fliplr(gt)
        if random.random() < 0.5:
            deg, gt = np.flipud(deg), np.flipud(gt)
        k = random.randint(0, 3)
        if k:
            deg, gt = np.rot90(deg, k), np.rot90(gt, k)
        return np.ascontiguousarray(deg), np.ascontiguousarray(gt)

    def _apply_synthetic_redegradation(self, gt_patch):
        h, w = gt_patch.shape
        sigma = random.uniform(0.05, 0.20)
        speckled = gt_patch * (1 + np.random.normal(0, sigma, gt_patch.shape))
        speckled += np.random.normal(0, random.uniform(0.01, 0.05), gt_patch.shape)
        speckled = np.clip(speckled, 0.0, 1.0).astype(np.float32)

        pil_img = Image.fromarray((speckled * 255).astype(np.uint8))
        small = pil_img.resize((w // self.upscale, h // self.upscale), Image.BOX)
        return np.array(small, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        deg_raw = load_raw(self.degraded_files[idx])
        gt_raw = load_raw(self.gt_files[idx])

        scale = self.norm_max if self.norm_max is not None else auto_detect_scale(gt_raw.max())
        deg_img = deg_raw / scale
        gt_img = gt_raw / scale

        ps = self.patch_size_degraded
        if self.train:
            dh, dw = deg_img.shape
            if dh < ps or dw < ps:
                raise ValueError(
                    f"Degraded image at index {idx} ({dh}x{dw}) is smaller than "
                    f"patch_size_degraded={ps}."
                )
            top = random.randint(0, dh - ps)
            left = random.randint(0, dw - ps)
            deg_patch = deg_img[top:top + ps, left:left + ps]
            gt_patch = gt_img[top * self.upscale:(top + ps) * self.upscale,
                               left * self.upscale:(left + ps) * self.upscale]

            if random.random() < self.redegradation_prob:
                deg_patch = self._apply_synthetic_redegradation(gt_patch)

            deg_patch, gt_patch = self._dihedral_augment(deg_patch, gt_patch)
        else:
            deg_patch, gt_patch = deg_img, gt_img

        deg_t = torch.from_numpy(deg_patch.copy()).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_patch.copy()).unsqueeze(0).float()
        return deg_t, gt_t


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--inspect", action="store_true",
                         help="Print raw dtype/min/max stats for a few sample files "
                              "before deciding on --norm_max")
    args = parser.parse_args()

    if args.inspect:
        for label, subdir in [("GT", "GT"), ("NoisyLR", "NoisyLR")]:
            files = sorted(glob.glob(os.path.join(args.root, subdir, "*")))[:5]
            print(f"\n{label} ({len(files)} sample files shown):")
            for f in files:
                arr = np.load(f)
                print(f"  {os.path.basename(f)}: dtype={arr.dtype}, shape={arr.shape}, "
                      f"min={arr.min():.4f}, max={arr.max():.4f}")
        print("\nPick --norm_max based on these stats (e.g. max~4095 -> --norm_max 4095).")
        print("If GT and NoisyLR maxes land in DIFFERENT buckets, that's exactly the "
              "overshoot the brief describes -- use GT's bucket as norm_max for both.")

    ds = RestorationDataset(args.root, train=True, norm_max=args.norm_max)
    deg, gt = ds[0]
    print(f"\nDegraded patch: {tuple(deg.shape)}, range [{deg.min():.3f}, {deg.max():.3f}]")
    print(f"GT patch:       {tuple(gt.shape)}, range [{gt.min():.3f}, {gt.max():.3f}]")

    ds_val = RestorationDataset(args.root, train=False, norm_max=args.norm_max)
    deg_full, gt_full = ds_val[0]
    print(f"Full degraded image: {tuple(deg_full.shape)}")
    print(f"Full GT image:       {tuple(gt_full.shape)}")