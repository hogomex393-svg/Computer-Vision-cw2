# Execution contract

Source: user-supplied first member execution brief, received 2026-11-04. User explicitly authorized implementation, ordinary fixes, local commits and upload to the named repository. No paid compute or APIs.

R01: Explicit Trans10K v1 audit, strict label mapping, splits and leakage detection. Evidence: audit receipt and adversarial tests.

R02: SegFormer B2 teacher (documented B0 fallback), MobileNetV3-Large pyramid decoder, independent boundary/KL branches. Evidence: architecture checks, gradients and logits equivalence.

R03: T/S0/S1/S2/S3 configs, comparable initialization/budget, AMP, epoch resume and provenance. Evidence: training, overfit, resume and failure-path tests.

R04: Original-resolution segmentation and boundary metrics, seeded robustness and honest image/report aggregation. Evidence: manually solvable metrics and saved evaluations.

R05: Fixed-shape logits-only ONNX 11/12, no teacher/auxiliary head, actual ORT comparison. Evidence: numerical errors and mask agreement.

R06: Nano JetPack 4.6.x/TensorRT 8.x environment/build/runner, independent runtime, sync/async bounded queues, profiling and sustained matrix. Evidence: host syntax/integration; target hardware explicitly pending.

R07: Image/directory/video/camera OpenCV demo and truthful smoke labels. Evidence: headless output and missing-camera error.

R08: Required documents, portable install/quickstart, progress, commits and verified remote branch. No data, credentials or large binaries in Git.

Non-goals: instance segmentation, counting, depth, GANs, heavy web frontends, paid cloud and claims of novelty for known boundary/distillation techniques.

Completion levels remain separate: implemented, host-tested, real-data-trained, ONNX-verified, Nano-verified, remote-verified. Missing external conditions cannot be represented as success.
