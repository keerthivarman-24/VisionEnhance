# KLA Track 1: AI-Based Restoration of Degraded Images

This submission provides an offline PyTorch inference pipeline for restoring degraded grayscale semiconductor images. The trained model performs joint denoising and fixed **2x super-resolution** on NumPy `.npy` inputs.

## Submission Structure

```text
team_name/
├── run.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── model.pt
```

`model.py` is a supporting source file required by `run.py`. The trained model weights are included locally under `models/model.pt`.

## System Requirements

- Python 3.10 or newer
- NVIDIA GPU recommended
- CUDA-compatible PyTorch installation
- No internet access is required during inference
- No API keys, external model downloads, or user interaction are required

The program automatically uses an NVIDIA GPU when CUDA is available. CPU inference is supported as a fallback but will be slower.

## Dependencies

The required Python packages are listed in `requirements.txt`.

Install them with:

```bash
python -m pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
numpy==1.26.4
torch==2.2.2
```

If the evaluation machine uses a different CUDA runtime, install the compatible build of the same PyTorch version before running the submission. No packages are downloaded by `run.py`.

## Required Execution Command

Run the solution using exactly:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

Windows PowerShell example:

```powershell
python run.py .\Test_NoisyLR .\restored_outputs
```

The output directory is created automatically if it does not exist.

## Input Format

The input directory may contain one or more `.npy` files. Non-`.npy` files are ignored.

Each input must be:

- A grayscale NumPy array
- Shape `(H, W)` or `(H, W, 1)`
- Numeric and convertible to `float32`
- Free from NaN and Inf values

The supplied noisy arrays may contain valid degradation overshoot below `0` or above `1`. The inference pipeline does not perform per-image normalization or input clipping.

## Output Format

For every input `.npy` file, `run.py` creates exactly one restored `.npy` file in the output directory.

Each output:

- Has the same filename as the corresponding input
- Is a grayscale NumPy array with shape `(2H, 2W)`
- Uses `float32`
- Contains values within `[0,1]`
- Contains no NaN or Inf values

Example:

```text
Input:  input/sample_001.npy   shape=(256, 256)
Output: output/sample_001.npy  shape=(512, 512)
```

## Model Details

The restoration network uses:

- NAFNet-style restoration blocks
- LayerNorm2d
- SimpleGate feature mixing
- Simplified channel attention
- PixelShuffle for fixed 2x upscaling
- Bicubic long residual connection
- EMA model weights for inference when available

The checkpoint configuration is read automatically from `models/best.pt`. The checkpoint must match the included `model.py` architecture.

## Offline Operation

The submission is self-contained. During execution, `run.py` does not:

- Access the internet
- Call external APIs
- Require API keys
- Download model weights
- Request interactive input
- Require manual path configuration

All necessary model weights and supporting code are included in the submission folder.

## Output Validation

`run.py` validates the following before saving each result:

- Input shape is `(H, W)` or `(H, W, 1)`
- Input contains no NaN or Inf
- Output resolution is exactly 2x the input resolution
- Output data type is `float32`
- Output contains no NaN or Inf
- Output values are within `[0,1]`

If a requirement is violated, the script stops with a clear error message and a non-zero exit status.

## Environment Verification

To confirm Python, NumPy, PyTorch, CUDA, and the bundled checkpoint are available, run:

```bash
python -c "import numpy, torch; from pathlib import Path; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Checkpoint:', Path('models/best.pt').is_file())"
```

## Final Submission Check

Before creating the final archive, verify that:

- `run.py` works with positional input and output directory arguments
- `model.py` is included
- `models/best.pt` is included
- Every input produces one same-name output
- Every output is exactly 2x the input resolution
- Every output is `float32`
- Every output is finite and within `[0,1]`
- The solution works on an NVIDIA GPU
- No internet access or additional downloads are needed
- Test data, generated previews, caches, and temporary outputs are removed

## Reproducible Test

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

A successful verification ends with:

```text
VERIFICATION PASSED
```
