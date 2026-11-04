# Environment

Host verified: Windows 11 build 26300, Python 3.13.3, NVIDIA RTX 4060 Laptop 8GB, NVIDIA driver 596.36. CUDA allocation and training were executed successfully.

Project environment: `F:/PyLibs/cv_cw2_lib/Scripts/python.exe`. It is a separate virtual environment. Immutable existing torch/torchvision dependency files were reused through filesystem hard links; their metadata was copied. The original environment was not upgraded or edited. New dependencies were installed only into this project environment. A separately installed CPU environment is being used to check reproducibility independent of those reused files.

Verified versions: torch 2.6.0+cu124, torchvision 0.21.0+cu124, Transformers 4.49.0, ONNX 1.18.0, ONNX Runtime 1.23.2, OpenCV 4.11.0, NumPy 2.4.4, Pillow 12.2.0. Full captured versions: requirements.lock.txt. Portable CPU/CUDA installation inputs are separate.

Local data: `F:/Datasets/computer_vision/Trans10K`. Local model caches: `F:/Models/computer_vision/cw2`. Do not use these personal paths in another machine's configuration; supply --root, --manifests and cache environment variables.

GPU shape probe: B2 training, 4x3x384x384, approximately 0.120 seconds per step and 2.71 GB peak allocated memory. S3 including teacher forward approximately 0.071 seconds per step and 1.79 GB. These short random-input probes exclude disk decoding and full validation, and are not FPS claims or accuracy evidence. Both models were present during the probe, making memory measurements conservative for isolated T training.

Target Nano: no SSH address or connected device supplied. No arbitrary network scan was performed. Nano Python syntax is planned for 3.6, JetPack 4.6.x and TensorRT 8.x; actual board packages and performance remain pending.

Git: installed, local commit identity configured. Provided remote is public and initially empty; API grants push. Work branch: codex/transparent-segmentation. No visibility or repository settings were changed.
