# Progress

## Verified milestone 2

- Core data/model/loss/metric/training/export/runtime code implemented.
- `python -m pytest tests.py -q --basetemp artifacts/tests --junitxml=artifacts/pytest.xml`: 13 passed, 19.41 seconds.
- Real SegFormer B0 forward and teacher-freeze gradient test passed (random weights, no accuracy claim).
- CPU epoch-boundary resume matched uninterrupted training exactly on the tested platform.
- CUDA synthetic run: 60 epochs / 180 updates completed. Loss 2.0881 -> 0.07437; synthetic validation mIoU 0.94790. This is colored-shape smoke data, not transparent-object accuracy.
- Trained smoke export opset 11: max absolute difference 3.82e-6, all tested mask agreements 1.0; auxiliary head removal exact.
- Headless images, video, bounded async capture and missing-source failures passed.
- Nano host environment probe correctly reports target unavailable. Python 3.6 syntax check passes.

## Current work

Official Trans10K download and full audit; official pretrained weights and GPU budget probe; clean-environment verification; documentation and formal experiment orchestration.

## Local paths

Project interpreter: F:/PyLibs/cv_cw2_lib/Scripts/python.exe
Data: F:/Datasets/computer_vision/Trans10K
Weights/cache: F:/Models/computer_vision/cw2
Smoke outputs: runs/smoke, artifacts/student_smoke.onnx, artifacts/smoke_demo

## External boundaries

The user authorized proceeding with all feasible downloads and computation. GPU is available. No Nano connection is provided; no LAN scanning. GitHub target is public; only project code/docs/small results may be published. No formal trained transparent model exists yet.

## Resume

Read docs/VALIDATION.md and docs/work_state.json. Inspect Git status. Do not overwrite completed runs with changed code/config identities; create a new run directory. Do not mistake smoke weights for real weights.
