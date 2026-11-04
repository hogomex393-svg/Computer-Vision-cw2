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
6. `project.py demo --model artifacts/student_smoke.onnx --source data/smoke/test/images --out artifacts/smoke_demo --smoke`: exit 0, four image outputs visually inspected.
7. `nano.py env --out artifacts/nano_host`: exit 0; correctly reports no target TensorRT, tegrastats or JetPack. Not a board validation.
8. Engineering state validator passed after adding item statements.
