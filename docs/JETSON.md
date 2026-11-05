# Jetson Nano deployment

Implementation complete; original Nano hardware validation is pending. Host syntax and non-hardware paths do not verify TensorRT parser/API support on the board.

Target: original Jetson Nano 4GB, JetPack 4.6.x, its supplied TensorRT 8.x (commonly 8.2.x), Python 3.6. Do not replace this with Orin Nano, JetPack 6 or TensorRT 10/11 instructions. Engines must be built on the destination Nano. PC engine files are not portable.

## Inspect first

Copy `nano.py`, `runtime.py`, exported ONNX files and their JSON receipts. Inspect the board's existing packages; do not replace its CUDA/JetPack drivers or pip-install a modern x86 TensorRT package.

```bash
python3 nano.py env --out nano_results/environment
python3 -c "import tensorrt, pycuda.driver, cv2, numpy; print(tensorrt.__version__, cv2.__version__, numpy.__version__)"
```

Use the system TensorRT and OpenCV packages. Python 3.6 commonly pairs with NumPy 1.19.5; verify the actual installed bindings before any changes. PyCUDA must be installed for the board's CUDA/Python ABI, preferably an existing JetPack-compatible installation. The host's training requirements are not Nano requirements.

## Build and run

```bash
python3 nano.py build --onnx models/S3_384.onnx --engine models/S3_384_fp32.engine --precision fp32
python3 nano.py build --onnx models/S3_384.onnx --engine models/S3_384_fp16.engine --precision fp16
python3 nano.py verify --engine models/S3_384_fp16.engine --reference models/S3_384.reference.npz --out nano_results/alignment
python3 nano.py demo --engine models/S3_384_fp16.engine --source demo.mp4 --out nano_results/playback
python3 nano.py demo --engine models/S3_384_fp16.engine --source 0 --out nano_results/live --show
```

Use the model name selected on validation, not necessarily S3. The exported logits are fixed [1,3,H,H]. Default ONNX opset 11 is host-tested; opset 12 is also exposed as a candidate. The board parser decides the final supported combination. Parser errors are raised with every recorded TensorRT diagnostic. Do not claim compatibility until build and execution pass on the real board.

FP16 can retain some operations in FP32 as chosen by TensorRT. INT8 is not required. If build memory is insufficient, try `--workspace-mb 128`, close unrelated board applications and compare the 256 model. Record each change; do not silently substitute input size in a benchmark.

The runner rejects nonfinite logits before argmax or rendering. The host training overflow documented in NUMERICAL_FIX.md makes target FP16 verification especially important: BF16 training does not prove TensorRT FP16 compatibility. If FP16 produces invalid values or fails alignment, preserve the evidence and use the independently built FP32 engine while diagnosing the affected layer. Do not report the failed FP16 case as a speed/accuracy success.

## Timing and resources

```bash
python3 nano.py benchmark --engine models/S3_384_fp16.engine --source demo.mp4 --out nano_results/sync --duration 600
python3 nano.py benchmark --engine models/S3_384_fp16.engine --source 0 --out nano_results/async --duration 600 --async-mode
python3 nano.py matrix --models-dir models --model-prefix S3 --source demo.mp4 --out nano_results/matrix --duration 600
```

The matrix performs 256/384/512 x FP32/FP16, building each engine on the Nano and running ten minutes per combination. Sync playback loops the local video to reach the requested duration. Async capture keeps at most one queued frame and reports discarded frames and frame age. Short/as-fast-as-possible video playback is not a substitute for a ten-minute live asynchronous run; an early-ended source fails the sustained-duration check.

Warmup is excluded. CUDA events measure pure engine execution, excluding transfers; application timing includes capture, preprocessing, inference, postprocessing and writing the comparison video. Saved FPS, mean/P50/P95, wall processing FPS and camera-reported FPS are different quantities. Never quote the camera's nominal FPS as model performance.

tegrastats is collected when present. RAM, GPU utilization, temperatures and POM_5V_IN are summarized; unavailable values stay unavailable. `nvpmodel -q` is recorded, never changed automatically. Keep the raw log and environment receipt with every benchmark. The runner explicitly owns/pushes/pops/releases its CUDA context, synchronizes before reading host buffers and releases buffers on close.

## Board acceptance still required

1. Actual JetPack/TensorRT/Python/PyCUDA versions and physical Nano model.
2. Parse/build for opset 11; if unsupported, export 12 and record which combination works.
3. FP32/FP16 engine output agreement on the same inputs against ONNX reference, including original-grid masks.
4. Pure inference and end-to-end performance, memory, power mode, temperature and ten-minute stability.
5. Real camera and local-video demonstrations; foreground overlay alignment and saved video readability.

If a board failure occurs, preserve the command, complete error and environment JSON. These are implementation/compatibility issues to diagnose, not instructions for a teammate to implement missing functions.

The environment receipt also records the device-tree model, physical RAM and L4T release; an aarch64 machine alone is not treated as an original Nano 4GB. Reference NPZ files are bound to the engine source by SHA256.

Frame age is measured from completion of OpenCV capture/read, not from a sensor exposure timestamp. Driver buffering can add latency outside that measurement. Saved comparison videos retain the source playback FPS; measured processing throughput comes from timings/benchmark JSON, not the video's nominal FPS. Non-smoke demos visibly label IMAGE, REPLAY or LIVE.

For the baseline comparison, the model package includes S0_384.onnx in addition to the selected model. Build S0_384_fp16.engine, process the same local video into a separate output directory, then run the selected model. Keep both timing JSON files and label both clips as replay.
