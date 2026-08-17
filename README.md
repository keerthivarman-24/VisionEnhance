# KLA Track 1: Joint Image Denoising and 2x Super-Resolution
This project implements a PyTorch-based image restoration model for the KLA problem statement in the SEMICON India Hackathon 2026.


## 2. Backbone: NAFNet-style restoration body + PixelShuffle head

**Architecture:**

```
Input (1×H×W, degraded)
   │
3×3 Conv → C channels (C=32 to start, 48 if VRAM allows)
   │
[NAFBlock] × N   (N=12–16)   ← all restoration happens HERE, at input resolution
   │
3×3 Conv → 4C channels
   │
PixelShuffle(×2)             ← single upsample, sub-pixel conv (Shi et al., ESPCN-style)
   │
[NAFBlock] × 2               ← light refinement at output resolution (fix upsample artifacts)
   │
3×3 Conv → 1 channel (output)
   │
   + Bicubic-upsampled(input)   ← long residual skip, network only learns the correction
   │
Restored output (1×2H×2W)
```

**NAFBlock internals**:
- Block A (spatial mixing): LayerNorm → 1×1 Conv → depthwise 3×3 Conv → **SimpleGate**
  (split channels in half, elementwise multiply — replaces GELU) → **Simplified Channel
  Attention** (global avg pool → 1×1 conv, no sigmoid) → 1×1 Conv → residual add
- Block B (channel mixing / FFN): LayerNorm → 1×1 Conv (expand) → SimpleGate → 1×1 Conv
  (project) → residual add

No BatchNorm anywhere (LayerNorm only)

---

## 3.Project Structure

Place the project files in one directory:

```text
KLA-Restoration/
├── dataset.py
├── losses.py
├── model.py
├── train.py
├── standalone.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── NoisyLR/
│   │   ├── sample_0001.npy
│   │   ├── sample_0002.npy
│   │   └── ...
│   ├── GT/
│   │   ├── sample_0001.npy
│   │   ├── sample_0002.npy
│   │   └── ...
│   └── Test_NoisyLR/
│       ├── test_0001.npy
│       ├── test_0002.npy
│       └── ...
├── checkpoints/
└── restored_outputs/
```

> **Notice**: The filename stems in `NoisyLR` and `GT` must match for training.

```text
NoisyLR/sample_0001.npy
GT/sample_0001.npy
```
---

## 4. Dataset Intensity Range

The supplied arrays are already stored as floating-point values.

Observed ranges:

```text
GT:      0.0 to 1.0
NoisyLR: approximately -0.2786 to 2.1580
```

Negative values and values above `1.0` in NoisyLR are valid noise overshoot. They must not be divided by `255` or clipped in the dataset loader.

For the current code, always pass:

```text
--norm_max 1.0
```

This makes the dataset and inference pipelines preserve the supplied numerical scale.

> **Important:** Do not use automatic per-image normalization. It can scale different NoisyLR samples differently and invalidate the learned input-to-target mapping.

---

## 5. Requirements

Recommended configuration:

- Python 3.10 or newer
- PyTorch 2.1 or newer
- NVIDIA CUDA GPU recommended
- NumPy
- Pillow
- Matplotlib

---

## 6. Environment Setup

### Linux or macOS

```bash
cd KLA-Restoration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
cd KLA-Restoration
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
---

## 5. Install a CUDA-Enabled PyTorch Build

The command in `requirements.txt` installs the PyTorch build selected by pip. For GPU training, install a PyTorch build compatible with the installed NVIDIA driver and CUDA environment.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU execution is supported, but training will be significantly slower.

---

## 6. Dataset Verification

Before training, inspect the files and numerical ranges:

```bash
python dataset.py \
  --root ./dataset \
  --norm_max 1.0 \
  --inspect
```

Windows PowerShell:

```powershell
python dataset.py `
  --root ./dataset `
  --norm_max 1.0 `
  --inspect
```

Confirm that:

- Files are 2D grayscale NumPy arrays.
- GT values are in `[0,1]`.
- NoisyLR may contain negative values and values above `1`.
- No file contains NaN or Inf values.
- Every GT image is exactly 2x the corresponding NoisyLR image.
- NoisyLR and GT files are correctly paired.

If the program warns that filenames do not match and sorted-order pairing is being used, manually verify every pair before training. Matching filename stems are strongly recommended.

---

## 7. Recommended Stable Training Command

Start in full float32 precision. Do not pass `--amp` for the first run.

```bash
python train.py \
  --data_root ./dataset \
  --out_dir ./checkpoints \
  --norm_max 1.0 \
  --redegradation_prob 0.0 \
  --epochs 100 \
  --batch_size 8 \
  --patch_size 64 \
  --lr 5e-5 \
  --width 32 \
  --num_blocks 8 \
  --val_split 0.05 \
  --num_workers 0 \
  --val_every 1 \
  --log_every 50
```

Windows PowerShell:

```powershell
python train.py `
  --data_root ./dataset `
  --out_dir ./checkpoints `
  --norm_max 1.0 `
  --redegradation_prob 0.0 `
  --epochs 100 `
  --batch_size 8 `
  --patch_size 64 `
  --lr 5e-5 `
  --width 32 `
  --num_blocks 8 `
  --val_split 0.05 `
  --num_workers 0 `
  --val_every 1 `
  --log_every 50
```

After stable training is confirmed, increase the model depth:

```text
--num_blocks 14
```

### Why these settings are recommended

- `--norm_max 1.0` preserves the supplied floating-point scale.
- `--redegradation_prob 0.0` uses only the official paired degradation during the baseline run.
- Omitting `--amp` avoids float16 overflow while debugging.
- `--lr 5e-5` is a conservative learning rate for stable NAFBlock training.
- `--num_workers 0` simplifies initial debugging and is compatible with Windows.
- Eight blocks provide a faster baseline before moving to fourteen blocks.

---
## 8. Training Outputs

The training script writes:

```text
checkpoints/
├── latest.pt
└── best.pt
```

`latest.pt` contains the most recently validated state.

`best.pt` contains the checkpoint with the highest validation SSIM observed during training.

Each checkpoint contains:

- Raw model weights
- EMA model weights
- Optimizer state
- Learning-rate scheduler state
- Current epoch
- Best validation SSIM
- Training arguments

The standalone inference script prefers the EMA weights when available.

---

## 9. Resume Training

Resume from a valid finite checkpoint:

```bash
python train.py \
  --data_root ./dataset \
  --out_dir ./checkpoints \
  --norm_max 1.0 \
  --redegradation_prob 0.0 \
  --epochs 150 \
  --batch_size 8 \
  --patch_size 64 \
  --lr 5e-5 \
  --width 32 \
  --num_blocks 8 \
  --num_workers 0 \
  --resume ./checkpoints/latest.pt
```

Do not resume a checkpoint saved after the loss or model parameters became NaN.

The architecture options used while resuming must match the checkpoint, especially:

```text
--width
--num_blocks
```

---

## 10. Standalone Inference

Run inference over the test directory:

```bash
python standalone.py \
  --input_dir ./dataset/Test_NoisyLR \
  --output_dir ./restored_outputs \
  --checkpoint ./checkpoints/best.pt \
  --norm_max 1.0
```

Windows PowerShell:

```powershell
python standalone.py `
  --input_dir ./dataset/Test_NoisyLR `
  --output_dir ./restored_outputs `
  --checkpoint ./checkpoints/best.pt `
  --norm_max 1.0
```

You can download pre-trained model [here](https://github.com/keerthivarman-24/VisionEnhance/releases/tag/v1.0)

```bash
python standalone.py --input_dir ./dataset/Test_NoisyLR --output_dir .restored_outputs --checkpoint model.pt --norm_max 1.0
```


The restored `.npy` files are saved with the same filenames:

```text
restored_outputs/
├── test_0001.npy
├── test_0002.npy
└── ...
```

Expected output properties:

```text
Data type: float32
Intensity range after inference clamping: [0,1]
Height: 2 x input height
Width: 2 x input width
Filename: same as the input filename
```

---

## 11. Save PNG Previews

To save an 8-bit PNG preview alongside each restored `.npy` output:

```bash
python standalone.py \
  --input_dir ./dataset/Test_NoisyLR \
  --output_dir ./restored_outputs \
  --checkpoint ./checkpoints/best.pt \
  --norm_max 1.0 \
  --save_png
```

PNG files are intended only for visual inspection. Use the restored `.npy` files for official numerical evaluation or submission unless the organizer specifies otherwise.

---

## 12. Save Comparison Plots

For NoisyLR and generated output comparisons:

```bash
python standalone.py \
  --input_dir ./dataset/Test_NoisyLR \
  --output_dir ./restored_outputs \
  --checkpoint ./checkpoints/best.pt \
  --norm_max 1.0 \
  --save_comparison
```

For a held-out local validation set with GT available:

```bash
python standalone.py \
  --input_dir ./dataset/Validation_NoisyLR \
  --gt_dir ./dataset/Validation_GT \
  --output_dir ./validation_outputs \
  --checkpoint ./checkpoints/best.pt \
  --norm_max 1.0 \
  --save_png \
  --save_comparison
```

Comparison figures are written to:

```text
validation_outputs/comparison_plots/
```

Comparison plots are generated after the reported inference timer stops.

---