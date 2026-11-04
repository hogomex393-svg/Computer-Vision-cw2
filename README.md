# Transparent segmentation for Jetson Nano

RGB semantic segmentation of background, transparent things and transparent stuff. The deployment student is MobileNetV3-Large with a lightweight multiscale decoder. Training experiments add boundary supervision and SegFormer-B2 probability distillation independently. Known techniques are combined and evaluated for edge deployment; this is not a claim of a novel architecture.

**Current measured status is maintained in [docs/PROGRESS.md](docs/PROGRESS.md).** Synthetic smoke results are not public-dataset accuracy. Nano code and host tests do not constitute physical Nano validation. Formal metrics remain pending until actual run receipts exist.

## Fast CPU verification

Use Python 3.13 in an isolated environment. The tested training GPU pair is PyTorch 2.6.0 and torchvision 0.21.0 with CUDA 12.4. CPU wheels are provided separately.

```bash
git clone --branch codex/transparent-segmentation https://github.com/hogomex393-svg/Computer-Vision-cw2.git
cd Computer-Vision-cw2
python -m venv .venv
# Activate .venv using your shell, or invoke its Python directly.
python -m pip install -r requirements-cpu.txt
python -m pytest tests.py -q --basetemp artifacts/tests
python project.py synthetic --root data/smoke
python project.py audit --root data/smoke --out data/smoke_manifests --encoding indexed --synthetic
python project.py train --config configs/smoke.json --root data/smoke --manifests data/smoke_manifests --out runs/smoke --device cpu
python project.py export --checkpoint runs/smoke/best.pt --out artifacts/student_smoke.onnx --smoke
python project.py demo --model artifacts/student_smoke.onnx --source data/smoke/test/images --out artifacts/smoke_demo --smoke
```

On this user's computer, virtual environments and caches are on F: rather than the repository. The commands are portable; replace .venv with your chosen location. Headless inference is the default; --show enables OpenCV windows.

## Real data and experiments

Download Trans10K v1 only from the [author's project page](https://xieenze.github.io/projects/TransLAB/TransLAB.html), or run `download_data.py --root YOUR_DATA_ROOT`. It downloads approximately 8.8 GB, validates ZIP magic/CRC and writes SHA256 receipts. Login/permission responses fail clearly. Do not point this command at unrelated directories.

```bash
python -m pip install -r requirements.txt
python download_data.py --root data/Trans10K
python project.py audit --root data/Trans10K --out data/manifests --encoding trans10k_v1 --geometry-overrides configs/trans10k_v1_geometry.json
python project.py prepare-splits --audit data/manifests --out data/clean_manifests
python project.py cache --root data/Trans10K --manifest data/clean_manifests/train.json --out data/train_cache384 --size 384
python project.py cache --root data/Trans10K --manifest data/clean_manifests/validation.json --out data/val_cache384 --size 384
python project.py cache --root data/Trans10K --manifest data/clean_manifests/test.json --out data/test_cache384 --size 384 --conditions clean brightness blur jpeg
python project.py pipeline --root data/Trans10K --manifests data/clean_manifests --cache data/train_cache384 --validation-cache data/val_cache384 --test-cache data/test_cache384 --out runs/final --device cuda --calibrate
python report.py --runs runs/final --out results
```

The initial strict audit intentionally fails on the documented official cross-split duplicates. Only after checking that no other errors remain, prepare-splits creates the verified 4990/999/4428 leak-free-v1 protocol while preserving all 4428 test images. See DATA_AUDIT.md.

The official raw white mask value is transparent stuff, **not ignore**. See [protocol](docs/EXPERIMENTS.md) and the SHA-bound [training geometry corrections](docs/DATA_AUDIT.md). Public images, pretrained weights and trained binaries are excluded from ordinary Git commits. Official initializations are [Torchvision MobileNetV3 V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html) and [NVIDIA SegFormer B2](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512). Their caches can be set with TORCH_HOME and HF_HOME before training.

Trained artifact availability, hashes and group access are recorded in [handoff](docs/HANDOFF.md). If no downloadable trained model is listed there, use smoke only to verify functionality or execute the full real-data pipeline; do not portray random/general pretrained weights as a trained transparent-object model.

## Project entry points

- `project.py`: audit/cache/train/evaluate/export/demo/compare/pipeline/environment.
- `download_data.py`: official source-aware ZIP download and validation.
- `nano.py`: target environment, engine build, TensorRT runtime, sustained benchmark and size/precision matrix.
- `report.py`: evidence-driven result summaries and plots.
- `tests.py`: data geometry, ignore labels, numerical metrics, loss degeneracy, gradients, resume, export and runtime regression.

Read [QUICKSTART](docs/QUICKSTART.md), [JETSON](docs/JETSON.md), [VALIDATION](docs/VALIDATION.md), [REPORT_DRAFT](docs/REPORT_DRAFT.md) and [DEMO_SCRIPT](docs/DEMO_SCRIPT.md). The attached execution requirements are preserved in docs/TASK_BRIEF.md.
