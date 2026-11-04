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
