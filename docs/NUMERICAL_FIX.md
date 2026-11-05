# Student training precision revision

The original FP16 calibration failed twice. The second production failure was captured at batch 436 of `kd_T4_W05`, before its first completed epoch. Inputs, teacher logits and student parameters were finite. Student FP16 logits were not finite, while the same saved student and inputs produced finite FP32 logits.

Forward hooks reproduced the first invalid operation: `encoder.16.0`, the final MobileNetV3 expansion convolution. Its FP32 output reached absolute magnitude 67143.6796875, beyond FP16's maximum finite value of 65504. BatchNorm and Hardswish then propagated invalid values into the decoder. Lowering the gradient scale cannot repair this forward overflow. The earlier independent replay's lack of failure was insufficient evidence of a fix.

The RTX 4060 reports native BF16 support. Replaying the actual saved failure in BF16 produced finite student/teacher logits, total loss 3.0291547775268555, finite student gradients and no teacher gradients. This validates the numerical counterexample, not the final model's accuracy. Evidence: results/fp16_overflow_diagnosis.json and results/bf16_failure_replay.json. The original tensors and failed run remain local.

The minimal training change makes AMP dtype explicit, supports BF16 without FP16 gradient scaling, rejects unsupported/unknown precision, and fails on nonfinite unscaled gradients. Model architecture, losses, data, optimizer settings, budgets and FP32 evaluation remain unchanged. All four student configurations now use the same BF16 protocol and restart from the same original pretrained initialization in `runs/final_bf16`. All four calibration candidates are rerun there too. Older FP16 student and calibration results are preserved in `runs/final` and are excluded from the final controlled comparison.

The completed 20-epoch teacher is reused as an independently trained teacher. The pipeline verifies its original config, identity, train/validation manifests, best/last checkpoints and validation receipt, copies its files byte for byte, and records `T/teacher_reuse.json`. Its original source hashes and training commit are retained; it is never relabeled as a BF16-trained teacher. The student comparison rejects mixed AMP dtypes and source identities.

BF16 is a host training choice. Export still uses FP32 weights and logits. Nano deployment remains TensorRT FP32/FP16, with mandatory target numerical validation; BF16 is not required on the Nano. Actual board performance is still unverified.

Source: [PyTorch 2.6 AMP documentation](https://docs.pytorch.org/docs/2.6/amp.html) and [gradient scaling examples](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html). Hardware range failure and the successful counterexample replay were measured locally.

Storage note after user-approved cleanup: old FP16 checkpoint files were removed, while their JSON/log records remain under runs/final. The captured failure tensor still exists as artifacts/kd_production_failure_batch.pt. All completed BF16 checkpoints and formal results remain intact.
