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

`python project.py audit --root DATA --out data/manifests --encoding trans10k_v1 --geometry-overrides configs/trans10k_v1_geometry.json` checks every image/mask pair, dimensions, actual label values, broken files, official counts and decoded image duplicates. Cross-split duplicates fail. The actual official archives contain 11 such groups; after inspecting audit.json, run `python project.py prepare-splits --audit data/manifests --out data/clean_manifests`. This refuses other audit failures, keeps the original audit intact, and produces a separately named 4990/999/4428 protocol with the full test set retained.

`python project.py cache --root DATA --manifest data/clean_manifests/train.json --out CACHE --size 384` creates memory-mapped resized RGB and integer labels. Cache identity must match the audited manifest and input size. The cached and uncached augmentations are tested for exact agreement. Original-resolution evaluation always uses original images and masks.

## Training and resuming

```bash
python project.py train --config configs/T.json --root DATA --manifests data/clean_manifests --cache CACHE --out runs/final/T --device cuda --stop-after-epoch 1
python project.py train --config configs/T.json --root DATA --manifests data/clean_manifests --cache CACHE --out runs/final/T --device cuda --resume
python project.py pipeline --root DATA --manifests data/clean_manifests --cache CACHE --out runs/final --device cuda --calibrate
```

The first command is a full-epoch timing checkpoint within the configured 20-epoch budget, not a replacement for formal training. Resume uses the identical source/config/manifest identity. Changed code or config requires a new directory. Concurrent writers to the same run are refused.

The pipeline runs T, S0, optional validation-only calibration, then S1/S2/S3 sequentially. Calibration tests two boundary weights and two paired temperature/KD-weight settings for three full epochs each. Student final experiments still use the same 20-epoch budget and initialization. T must beat S0 on validation before formal KD can proceed.

## Evaluation, export and demonstration

```bash
python project.py evaluate --checkpoint runs/final/S0/best.pt --root DATA --manifest data/clean_manifests/test.json --out runs/final/S0/manual_test.json --device cuda
python project.py export --checkpoint runs/final/S3/best.pt --out artifacts/S3_384.onnx --size 384 --opset 11
python project.py demo --model artifacts/S3_384.onnx --source sample.jpg --out artifacts/image_demo
python project.py demo --model artifacts/S3_384.onnx --source recording.mp4 --out artifacts/video_demo
python project.py demo --model artifacts/S3_384.onnx --source 0 --out artifacts/camera_demo --show
python project.py compare --baseline runs/final/S0/best.pt --final runs/final/S3/best.pt --root DATA --manifest data/clean_manifests/test.json --out artifacts/comparison --device cuda
```

Replace S3 with the validation-selected experiment from runs/final/selection.json. Camera 0 must exist; missing sources raise an error. For the same video, run S0 then the selected model into different output directories; no simultaneous GPU inference is required. Input playback is marked separately from live camera.

Only load trusted project checkpoints: optimizer/RNG resume files use PyTorch pickle serialization. ONNX metadata/checksums travel with the ONNX file. The export tests floating logits and mask agreement, not just file existence.

## Evaluation caches

For the full GPU run, build a clean validation cache and a test cache for all appearance conditions once:

```bash
python project.py cache --root DATA --manifest data/clean_manifests/validation.json --out VAL_CACHE --size 384
python project.py cache --root DATA --manifest data/clean_manifests/test.json --out TEST_CACHE --size 384 --conditions clean brightness blur jpeg
```

Pass `--validation-cache VAL_CACHE` to train/pipeline and `--test-cache TEST_CACHE` to pipeline. Standalone evaluate accepts `--cache TEST_CACHE`. The cached and uncached inputs are identical; original-resolution target masks and metrics are unchanged. Expect additional storage for uint8 memory-mapped arrays; no extra network download is involved.

## Native-resolution target cache and progress

```bash
python project.py cache-targets --root DATA --manifest data/clean_manifests/validation.json --cache VAL_CACHE
python project.py cache-targets --root DATA --manifest data/clean_manifests/test.json --cache TEST_CACHE
```

These lossless 2-bit caches preserve labels 0/1/2/255 at original resolution. They use about 14 GiB for this dataset, rather than about 55 GiB as unpacked uint8. Original-pixel evaluation is unchanged. CUDA statistics and CPU reference statistics were checked for exact agreement on the entire 999-image validation set.

The pipeline updates docs/PROGRESS.md and RUN_ROOT/CURRENT_STAGE.json at training/validation boundaries and each evaluation/export phase. run.json contains the actual process PID and completed epochs. Check the process command and logs before resuming; do not start a second writer. Historical runs/formal/T is the one-epoch performance pilot, not the final 20-epoch series.

Pipeline recovery reuses evaluations only when run identity, checkpoint digest, manifest digest, corruption and complete sample count match. Exports require matching model/reference digests and fixed shape. Figure rendering selects fixed examples and global failures from the completed clean-test per-image results, and re-infers only that figure subset; full benchmark evaluation remains unchanged.

## Training precision and completion

Student configurations now use BF16 AMP on supported CUDA devices after a reproduced FP16 forward overflow; see NUMERICAL_FIX.md. All four students and calibration candidates must use the same setting. The teacher may retain its original verified training protocol. To reuse an independently completed teacher in a new run, pass `--teacher-run PREVIOUS_RUN/T`; the files, config, source identities and manifest/checkpoint provenance are verified and preserved. Do not reuse the older FP16 S0 as the BF16 baseline.

`delivery.py` checks all five completed experiments, exact full-test coverage, validation-only selection and exports, renders the documented validation still-image replay, and prepares result figures and report text. `--upload` additionally performs normal commits to the designated branch/repository, a clean remote CPU check, archive validation and collaborator-only draft release upload. It never publishes a public release. `--wait` waits for the specified pipeline to complete and fails if that pipeline fails. Required arguments include `--runs`, `--manifests`, `--video`, `--video-sources`, `--out`; optional tools are `--ffmpeg` and `--cpu-python`. The provided replay must be labeled as a validation still-image sequence, not live footage.

Standalone packaging now also requires `--demo-dir DELIVERY/demo`, produced by the completion tool. No CUDA training stack is required on Nano.
