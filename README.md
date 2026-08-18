# KLA Track 1: AI-Based Restoration of Degraded Images

This project implements a PyTorch-based image restoration model for the KLA problem statement in the SEMICON India Hackathon 2026.

This submission provides a self-contained, offline PyTorch inference pipeline for restoring degraded grayscale semiconductor images. The trained model performs joint denoising and fixed **2x super-resolution** on NumPy `.npy` inputs.

## 1. Submission Structure

The final submission directory must use the following structure:

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

File descriptions:

- `run.py`: Required inference entry point. Reads all `.npy` files from the input directory and saves restored `.npy` files to the output directory.
- `model.py`: Defines the NAFNet-style restoration network used to load the trained checkpoint.
- `requirements.txt`: Lists the Python dependencies and tested versions.
- `README.md`: Contains setup, execution, input, output, and troubleshooting instructions.
- `.gitignore`: Excludes local caches, datasets, temporary outputs, and development checkpoints.
- `models/model.pt`: Contains the trained model checkpoint and is loaded automatically by `run.py`.

> **Important:** Keep `run.py`, `model.py`, and the `models` directory in the locations shown above. No source-code edits or manual checkpoint path configuration should be required during evaluation.

## 2. System Requirements

Recommended evaluation environment:

- Python 3.10 or newer
- NVIDIA GPU recommended
- CUDA-compatible NVIDIA driver
- CUDA-compatible PyTorch installation
- Sufficient memory to hold one input image and its 2x restored output

The program automatically uses an NVIDIA GPU when CUDA is available. CPU execution is supported as a fallback, but inference will be slower.

The solution does not require:

- Internet access during inference
- API keys
- External APIs
- Additional model downloads
- Interactive user input
- Manual source-code changes

## 3. Environment Setup

### 3.1 Open a terminal in the submission directory

```bash
cd Hackers-crackers
```

The current directory should contain `run.py`, `model.py`, `requirements.txt`, `README.md`, and the `models` directory.

### 3.2 Create a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3.3 Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3.4 Install dependencies

```bash
python -m pip install -r requirements.txt
```

The tested dependency versions are:

```text
numpy==2.3.3
torch==2.8.0
```

If the evaluation system uses a different CUDA runtime, install the CUDA-compatible build of the listed PyTorch version before running the submission. The `run.py` program itself does not download packages or model files.

## 4. Verify the Environment

Run the following command from the submission directory:

```bash
python -c "import numpy, torch; from pathlib import Path; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Checkpoint found:', Path('models/model.pt').is_file())"
```

Expected checks:

- NumPy imports successfully.
- PyTorch imports successfully.
- `CUDA available` is `True` on an NVIDIA GPU system.
- `Checkpoint found` is `True`.

If `Checkpoint found` is `False`, confirm that the checkpoint is located at:

```text
models/model.pt
```

## 5. Required Execution Command

The solution must be run using exactly two positional arguments:

```bash
python run.py <input-dir> <output-dir>
```

Example on Linux or macOS:

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

Example on Windows PowerShell:

```powershell
python run.py .\Test_NoisyLR .\restored_outputs
```

Argument descriptions:

- `<input-dir>`: Directory containing degraded `.npy` input files.
- `<output-dir>`: Directory where restored `.npy` files will be written.

The output directory is created automatically if it does not already exist.

No additional flags, checkpoint arguments, API keys, or manual configuration are required.

## 6. Input Format

The input directory may contain one or more `.npy` files. Files with other extensions are ignored.

Each input must satisfy the following conditions:

- The input is a NumPy array.
- The input is a grayscale image.
- The shape is `(H, W)` or `(H, W, 1)`.
- The values are numeric and convertible to `float32`.
- The array contains no NaN or Inf values.

Examples of accepted shapes:

```text
(128, 128)
(256, 256)
(128, 128, 1)
(256, 256, 1)
```

The supplied degraded images may contain valid noise overshoot below `0` or above `1`. The inference pipeline preserves this input distribution and does not apply per-image min-max normalization or arbitrary input clipping.

## 7. Output Format

For every input `.npy` file, `run.py` creates exactly one restored `.npy` file.

Each output:

- Has the same filename as the corresponding input.
- Is a grayscale NumPy array.
- Has shape `(2H, 2W)`.
- Uses NumPy `float32`.
- Contains values within `[0,1]`.
- Contains no NaN or Inf values.

Example:

```text
Input file:  input/sample_001.npy
Input shape: (256, 256)

Output file:  output/sample_001.npy
Output shape: (512, 512)
Output dtype: float32
Output range: [0,1]
```

For an input shaped `(H, W, 1)`, the saved output is a two-dimensional grayscale array shaped `(2H, 2W)`.

## 8. Model Details

The restoration network uses:

- NAFNet-style restoration blocks
- `LayerNorm2d`
- SimpleGate feature mixing
- Simplified channel attention
- ICNR-initialized PixelShuffle for fixed 2x upscaling
- High-resolution refinement blocks
- Bicubic long residual connection
- EMA weights for inference when available in the checkpoint

The model performs most restoration processing at the input resolution and upsamples once near the output. This design reduces computation while preserving a fixed 2x output scale.

The checkpoint must match the architecture defined in `model.py`. Do not modify `model.py` after generating `models/model.pt` unless the model is retrained and a compatible checkpoint is exported.

## 9. Offline Operation

The submission is self-contained. During execution, `run.py` does not:

- Access the internet
- Call external services
- Require credentials or API keys
- Download model weights
- Download configuration files
- Ask for interactive input
- Require editing paths inside the script

All required source code and trained weights are included in the submission directory.

## 10. Built-in Validation

Before saving each result, `run.py` checks that:

- The input shape is `(H, W)` or `(H, W, 1)`.
- The input contains no NaN or Inf values.
- The model output is exactly 2x the input height and width.
- The output is converted to `float32`.
- The output contains no NaN or Inf values.
- The output values are within `[0,1]`.

If a requirement is violated, the program prints a clear error and exits with a non-zero status instead of silently saving an invalid result.

## 11. Step-by-Step Test Procedure

### Step 1: Prepare a test input directory

Create or select a directory containing one or more degraded `.npy` files:

```text
Test_NoisyLR/
├── sample_001.npy
├── sample_002.npy
└── sample_003.npy
```

### Step 2: Run inference

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

### Step 3: Confirm the output directory

Expected structure:

```text
restored_outputs/
├── sample_001.npy
├── sample_002.npy
└── sample_003.npy
```

The number and names of output `.npy` files should match the input `.npy` files exactly.

### Step 4: Inspect one output manually

```bash
python -c "import numpy as np; x=np.load('./restored_outputs/sample_001.npy', allow_pickle=False); print('shape:', x.shape); print('dtype:', x.dtype); print('min:', float(x.min())); print('max:', float(x.max())); print('finite:', np.isfinite(x).all())"
```

Expected result:

```text
shape: exactly twice the corresponding input dimensions
dtype: float32
min: greater than or equal to 0.0
max: less than or equal to 1.0
finite: True
```

## 12. Troubleshooting

### Error: model checkpoint not found

Confirm the file exists at:

```text
models/model.pt
```

Run:

```bash
python -c "from pathlib import Path; print(Path('models/model.pt').resolve()); print(Path('models/model.pt').is_file())"
```

### Error: no `.npy` files found

Confirm the input argument points directly to the directory containing `.npy` files:

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

The current implementation reads files directly inside the specified directory and does not recursively scan nested subdirectories.

### Error: invalid input shape

Each input must be shaped `(H, W)` or `(H, W, 1)`. RGB arrays such as `(H, W, 3)` are not accepted.

Inspect an input with:

```bash
python -c "import numpy as np; x=np.load('./Test_NoisyLR/sample_001.npy', allow_pickle=False); print(x.shape, x.dtype)"
```

### Error: CUDA out of memory

The program processes images one at a time. Close other GPU applications and retry. If necessary, run on a GPU with more memory or use the CPU fallback.

### CUDA is not detected

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If CUDA is unavailable, confirm that the installed PyTorch build supports the NVIDIA driver and CUDA environment.

### Output filename or count mismatch

Delete the old output directory and run inference again. Check that the input directory contains only the intended `.npy` files.

### Output contains NaN or Inf

`run.py` rejects non-finite outputs. Confirm that the input file itself contains only finite values and that `models/model.pt` is the tested final checkpoint.

## 13. Reproducible Final Test

Run the following command from the submission root:

```bash
python run.py ./Test_NoisyLR ./restored_outputs
```

Then inspect the output count:

```bash
python -c "from pathlib import Path; a=list(Path('./Test_NoisyLR').glob('*.npy')); b=list(Path('./restored_outputs').glob('*.npy')); print('inputs:', len(a)); print('outputs:', len(b)); print('count matches:', len(a)==len(b))"
```

A successful final test should confirm:

```text
count matches: True
```
