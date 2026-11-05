# Experiment protocol

T: SegFormer-B2 initialized from NVIDIA's ADE20K model with a newly initialized three-class head. Train on Trans10K v1; general pretrained weights are not a result.

S0: MobileNetV3-Large ImageNet V2, 64-channel pyramid decoder, CE plus foreground Dice.

S1: S0 with boundary auxiliary loss. S2: S0 with output KL distillation. S3: both. Main network initialization is identical for the four students at the same seed; auxiliary parameters are constructed afterward. No multi-level feature distillation is used.

The formal budget is 20 complete epochs per model at seed 42, input 384, batch 4. All students share AdamW learning rate 3e-4, weight decay 0.01, cosine schedule, horizontal flip, brightness factor [0.8,1.2], frozen BatchNorm running statistics and gradient clipping at 5. Teacher LR is 6e-5. The completed teacher used FP16 AMP. After a reproduced student forward overflow, all four students and calibration candidates use BF16 AMP with FP32 loss reductions, and restart in runs/final_bf16. See NUMERICAL_FIX.md. No test-set tuning or silent step truncation is allowed. CPU runs require explicit smoke mode.

The decoder channel count stays 64. Boundary band dilation radius is 1 pixel at training resolution, invalid neighbors excluded. Initial lambda_boundary=0.2, lambda_kd=0.5, temperature=4. Validation-only calibration candidates and their selected values, if run, are stored under the calibration directory; they must not alter test protocol or main training budgets.

Model selection: maximum whole-validation-set three-class mIoU. `best.pt` and `last.pt` are distinct. Final deployment selection uses validation scores only, with test metrics read afterward. If T is not better than S0, formal KD fails closed and training/data must be investigated.

Resume is supported at completed epoch boundaries. Checkpoints preserve optimizer, scheduler, scaler, epoch/update count, Python/NumPy/Torch/CUDA random states and complete config. Data order is seeded per epoch; augmentation per seed/epoch/index. Tested CPU resume matches an uninterrupted run exactly. This does not promise cross-platform/CUDA-bitwise reproducibility. An interrupted epoch is replayed from its previous checkpoint; partial epoch logs do not constitute completed training.

Every run stores source hashes, config, train/validation manifest hashes, status and actual batch. Changing the training code/config/data requires a new output directory. `report.py` rejects mixed student protocols, stale evaluations and smoke results.

Evaluation restores floating logits to the original image grid before argmax. mIoU excludes only categories with zero union; background is included in mIoU and per-class values are always available. Merged foreground IoU/Dice treat both transparent classes as foreground. Boundary F1 is micro-aggregated across classes 1/2 using a 2-pixel Chebyshev tolerance at original resolution; ignore neighborhoods are excluded; empty/empty returns null.

Robustness: clean baseline; brightness x0.6; 7x7 Gaussian blur sigma=1.5; JPEG quality 30; seed 42 plus image index. No ground-truth transformation is needed for these appearance-only perturbations. Public benchmark and self-captured material must be reported separately.

Visual examples include fixed first manifest indices plus objectively selected lowest final-model per-image mIoU failures. Selection indices/rule are saved. Formal outputs are read from JSON; no fabricated numbers or hand-filled favorable values.

Actual completion, checkpoints and measured metrics are listed in docs/PROGRESS.md and generated results/RESULTS.md. Until these exist, formal results are pending.

## Actual data protocol revision

The official source index is 5000/1000/4428. Full decoded-image auditing found 11 cross-split duplicate groups. Actual T/S0/S1/S2/S3 runs use the explicit leak-free-v1 derived index: 4990 train, 999 validation, 4428 test. All original test images remain; ten train and one validation examples are excluded from experiment manifests only. Eighteen training mask rotations are SHA-bound and documented in DATA_AUDIT.md. Raw data and source audit remain intact. This is a disclosed data-integrity adaptation, not an unchanged-original-protocol benchmark.

Image decoding uses the raw stored pixel grid consistently: Pillow loading and OpenCV image demo both ignore automatic EXIF orientation. Known training-mask rotations are applied explicitly from the audited manifest.
