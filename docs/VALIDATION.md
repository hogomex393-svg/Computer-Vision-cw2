# Validation log

Only executed checks belong here. Code existence is not a pass.

- Initial CUDA probe in existing environment: PyTorch 2.6.0+cu124, CUDA tensor allocation passed; RTX 4060 Laptop 8GB.
- Git ls-remote target: exit 0, no refs (empty repository).
- Trans10K, training, ONNX, runtime and Nano checks: pending implementation.

## Milestone 2

1. First regression: 12 passed, 1 failed. Cause: LogitsOnly wrapper default train state restored BatchNorm training after ONNX export. Fixed by setting wrapper eval explicitly; added state assertion.
2. Second regression: ONNX alignment passed; Unicode Windows image path failed. Fixed byte-based image decode/encode and relative video paths.
3. Full regression: 13 passed in 19.41 seconds, artifacts/pytest.xml.
4. GPU smoke: 60 epochs / 180 updates, runs/smoke/epochs.jsonl and run.json. Synthetic-only loss reduced from 2.088117 to 0.074373; validation mIoU best 0.947899.
5. `project.py export --checkpoint runs/smoke/best.pt --out artifacts/student_smoke.onnx --smoke`: exit 0, max absolute logit error <=3.82e-6, mask agreement=1.0.
6. `project.py demo --model artifacts/student_smoke.onnx --source data/smoke/test/images --out artifacts/smoke_demo --smoke`: exit 0, four image outputs generated; one representative visually inspected.
7. `nano.py env --out artifacts/nano_host`: exit 0; correctly reports no target TensorRT, tegrastats or JetPack. Not a board validation.
8. Engineering state validator passed after adding item statements.

## Milestone 3 in progress

- Large FP16 reduction regression added; loss reductions explicitly cast to FP32.
- Full regression after cache/normalization fixes: 14 passed in 19.78 seconds.
- Separate CPU environment installed from official package indexes: 14 passed in 40.57 seconds; latest after runtime/reference updates 14 passed in 51.79 seconds.
- Formal-summary negative check: all absent experiments remain pending, no fabricated metrics.
- Official archives passed CRC and have SHA256 receipts on F:. Test download resumed from byte 2841640960 after a broken connection, with Content-Range checked before append.
- GPU shape probe: T 0.1204 s/step, S3 0.07135 s/step, batch 4, 384 input; random inputs and no quality claim.

- GitHub authentication completed via user-approved device flow. Working branch remote SHA 7b694c31cbf16f02f1d91d2bf475b80b44b524d0 verified with ls-remote. No force push.
- First fresh remote checkout: 10 passed, 4 setup errors because artifacts/ was absent in a clean clone. Fixed tests to initialize their output parent. Reverification follows.
- Unicode video filename and output folder: eight frames processed successfully, output video created.

- Remote fresh checkout after initialization fix: 14 passed in 25.28 seconds, artifacts/remote_pytest.xml (remote 1f4476a).
- Full dimension inventory: 18 mismatches, all in train. All 18 were visually checked in both rotation directions. Explicit SHA-bound recipes preserve original files and official split membership.
- Geometry repair, hash refusal, journal reuse and truncated-tail recovery regression added. Full suite: 15 passed in 22.02 seconds.
- Validation/corruption cache equivalence tested for clean/brightness/blur/JPEG; full suite 15 passed in 33.74 seconds.
- 384x384 smoke export: max logit error 1.24e-5 and mask agreement 1.0; artifacts/student_smoke_384.json.

- EXIF orientation consistency added: raw JPEG pixel grid matches between dataset and demo. Full regression: 16 passed in 62.17 seconds while dataset caches were being prepared.
- Derived split validation passed: source 5000/1000/4428, effective 4990/999/4428, zero exact cross-split intersections, test set retained in full. Raw audit remains failed for its known cross-split duplicates; no assertions were disabled.

- Actual teacher full first epoch (locked source f6d7364, leak-free-v1): 1248 minibatches, loss 0.3784324, whole-validation mIoU 0.8733769, epoch 1237.84 seconds. runs/formal/T/last.pt and best.pt exist; run status interrupted_at_epoch_boundary by the requested one-epoch timing gate. This is NOT completion of the configured 20 epochs or a test-set result.
- Original-resolution evaluation profiling is now in progress; artifacts/evaluation_profile.txt will contain measured hotspots.

## Evaluation performance correction

The 10-image profile identified duplicated boundary computation and original mask decoding as major costs. Per-image mIoU now reuses the computed confusion matrix. CUDA integer statistics retain the same shared CPU geometric restoration, and original labels use a lossless 2-bit cache. No pixels, classes or samples are omitted.

The existing one-epoch teacher was re-evaluated on all 999 validation images: 102.868 seconds, 999 GPU-statistics samples. mIoU, per-class IoUs, foreground IoU/Dice, boundary F1, confusion matrix, valid-pixel count and sample count match the previous CPU results exactly. Receipt: artifacts/optimized_validation_equivalence.json.

Full regression after metric/cache changes: 20 passed in 9.27 seconds. Automatic progress preservation regression added; full suite 21 passed in 10.46 seconds.

Historical runs/formal/T remains a one-epoch timing pilot. Because code identity changed, the final series starts in runs/final rather than bypassing resume guards.

- Non-smoke demo labels now identify IMAGE/REPLAY/LIVE. Torch demo removes the training-only boundary head; all four synthetic demo masks matched ONNX exactly.
- Nano environment now records device model/RAM/L4T; host probe correctly reports target unavailable. Python 3.6 syntax and progress preservation checks passed. Physical API execution remains pending.
- Packaging rejects running or incomplete experiment sets before creating output. Guard regression passed. Draft artifact publisher is implemented and compiled; actual archive/upload/privacy-gate validation remains pending until formal model package creation.

## Teacher formal completion

T completed all 20 epochs / 24960 minibatches. Best whole-validation mIoU=0.9239955208. Total run time=5991.46 seconds. This is validation, not test accuracy; full test evaluation remains queued after all model/hyperparameter selections. Model states remain in runs/final/T. Small receipts are results/T_training.json, T_validation.json and T_epochs.jsonl.

Training source hashes still match the recorded a2ec4a2 implementation. Subsequent changes are limited to delivery/Nano/caller orchestration and documentation.

- Latest complete clean CPU suite: 23 passed in 25.37 seconds (artifacts/final_cpu_pytest.xml).
- Delivery orchestration rejects stale checkpoint/data/corruption receipts and incomplete per-image results; focused checks passed.
- Figure selection uses fixed first entries plus globally lowest scores from the complete clean-test result; shuffled result-order regression passed.
- Core training source hashes remain unchanged while the active pipeline runs.
## 2026-11-05 delivery validation and KD recovery

The complete T and S0 runs each contain 20 epochs. T best validation mIoU is 0.9239955207704398; S0 is 0.8775038527093654. These are validation measurements on leak-free-v1, not final test results.

The first KD calibration failed on nonfinite loss before completing its first epoch. Its run record is preserved under runs/failed_kd_20261105_003541. An independent 1248-batch replay did not reproduce the failure; root cause is still undetermined. The same-code production retry uses local.retry_pipeline.py solely to collect tensor diagnostics after an exception. No training result is accepted by suppressing nonfinite checks.

Fixed the release validator's omission of the required manifests directory from its allowed paths. In the independent CPU environment, `python -m pytest tests.py -q -k "release_bundle or nano_report" --basetemp artifacts/pytest_delivery_20261105 --junitxml artifacts/pytest_delivery_20261105.xml` returned 0: two checks passed in 6.93 seconds. The bundle fixture includes all expected paths and refuses modified manifest bytes and unrelated archive entries. Nano result ingestion refuses unverified hardware. No draft release or model upload has yet been claimed.

## BF16 correction and complete delivery checks

The captured production batch reproduces FP16 overflow at encoder.16.0 (FP32 magnitude 67143.6796875). BF16 replay has finite output, loss and every student gradient; the teacher remains gradient-free. CPU full regression: 27 passed in 13.15 s, artifacts/pytest_bf16_cpu.xml. Delivery guards and archive checks: 3 passed in 6.95 s, artifacts/pytest_final_delivery.xml.

The new replay generator was exercised using the existing trained FP16 S0 as both comparison slots. All 90 source frames passed through both paths, H.264 outputs and provenance digests were produced in artifacts/delivery_replay_check. This verifies the baseline-selected edge case and real video processing, not completion of the new BF16 experiment series.

Independent remote checkout at 0966f81e57b8e06e9ffc71f5093a9dea932aa9c6 passed all 28 tests in 25.71 s using the separately installed CPU environment. Receipt: ../tmp/cw2-0966f81-check/artifacts/remote_bf16_pytest.xml. Subsequent Nano output guarding was checked without hardware: nonfinite engine output is refused before argmax, and Python 3.6 syntax remains valid. Real TensorRT execution is still pending.
