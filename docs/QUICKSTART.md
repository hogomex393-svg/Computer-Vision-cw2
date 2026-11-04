# Quickstart

Run commands from the repository root. Training-side scripts require the locked Python environment. `nano.py` and `runtime.py` are deliberately standalone and compatible in syntax with Python 3.6; Nano does not need the training stack.

## Installation

For Windows PowerShell:

```powershell
py -3.13 -m venv F:\PyLibs\cv_cw2_lib
& F:\PyLibs\cv_cw2_lib\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:TORCH_HOME='F:\Models\computer_vision\cw2\torch'
$env:HF_HOME='F:\Models\computer_vision\cw2\huggingface'
python project.py env --out artifacts/environment.json
```

For a CPU-only machine, install requirements-cpu.txt instead. Do not install both variants together. requirements.lock.txt is the measured full CUDA environment snapshot; requirements-base.lock.txt records shared dependencies.

Linux example:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest tests.py -q --basetemp artifacts/tests
```

The full clean Windows CPU installation is recorded in VALIDATION.md when complete. Other OS builds require their own verification; no cross-platform pass is implied.

## Data and audit

The official Google Drive archives are train.zip (4.2G), validation.zip (883M), test.zip (3.7G). Alternative author-provided Baidu entries are in SOURCES.md/project page. Target layout:

```text
Trans10K/train/images + masks
Trans10K/validation/easy/images + masks
Trans10K/validation/hard/images + masks
Trans10K/test/easy/images + masks
Trans10K/test/hard/images + masks
```

`python project.py audit --root DATA --out data/manifests --encoding trans10k_v1 --geometry-overrides configs/trans10k_v1_geometry.json` checks every image/mask pair, dimensions, actual label values, broken files, official counts and decoded image duplicates. Cross-split duplicates fail; do not silently remove public test samples or edit official splits to force success. Inspect the written audit.json first.

`python project.py cache --root DATA --manifest data/manifests/train.json --out CACHE --size 384` creates memory-mapped resized RGB and integer labels. Cache identity must match the audited manifest and input size. The cached and uncached augmentations are tested for exact agreement. Original-resolution evaluation always uses original images and masks.

## Training and resuming

```bash
python project.py train --config configs/T.json --root DATA --manifests data/manifests --cache CACHE --out runs/formal/T --device cuda --stop-after-epoch 1
python project.py train --config configs/T.json --root DATA --manifests data/manifests --cache CACHE --out runs/formal/T --device cuda --resume
python project.py pipeline --root DATA --manifests data/manifests --cache CACHE --out runs/formal --device cuda --calibrate
```

The first command is a full-epoch timing checkpoint within the configured 20-epoch budget, not a replacement for formal training. Resume uses the identical source/config/manifest identity. Changed code or config requires a new directory. Concurrent writers to the same run are refused.

The pipeline runs T, S0, optional validation-only calibration, then S1/S2/S3 sequentially. Calibration tests two boundary weights and two paired temperature/KD-weight settings for three full epochs each. Student final experiments still use the same 20-epoch budget and initialization. T must beat S0 on validation before formal KD can proceed.

## Evaluation, export and demonstration

```bash
python project.py evaluate --checkpoint runs/formal/S0/best.pt --root DATA --manifest data/manifests/test.json --out runs/formal/S0/manual_test.json --device cuda
python project.py export --checkpoint runs/formal/S3/best.pt --out artifacts/S3_384.onnx --size 384 --opset 11
python project.py demo --model artifacts/S3_384.onnx --source sample.jpg --out artifacts/image_demo
python project.py demo --model artifacts/S3_384.onnx --source recording.mp4 --out artifacts/video_demo
python project.py demo --model artifacts/S3_384.onnx --source 0 --out artifacts/camera_demo --show
python project.py compare --baseline runs/formal/S0/best.pt --final runs/formal/S3/best.pt --root DATA --manifest data/manifests/test.json --out artifacts/comparison --device cuda
```

Replace S3 with the validation-selected experiment from runs/formal/selection.json. Camera 0 must exist; missing sources raise an error. For the same video, run S0 then the selected model into different output directories; no simultaneous GPU inference is required. Input playback is marked separately from live camera.

Only load trusted project checkpoints: optimizer/RNG resume files use PyTorch pickle serialization. ONNX metadata/checksums travel with the ONNX file. The export tests floating logits and mask agreement, not just file existence.

## Evaluation caches

For the full GPU run, build a clean validation cache and a test cache for all appearance conditions once:

```bash
python project.py cache --root DATA --manifest data/manifests/validation.json --out VAL_CACHE --size 384
python project.py cache --root DATA --manifest data/manifests/test.json --out TEST_CACHE --size 384 --conditions clean brightness blur jpeg
```

Pass `--validation-cache VAL_CACHE` to train/pipeline and `--test-cache TEST_CACHE` to pipeline. Standalone evaluate accepts `--cache TEST_CACHE`. The cached and uncached inputs are identical; original-resolution target masks and metrics are unchanged. Expect additional storage for uint8 memory-mapped arrays; no extra network download is involved.
