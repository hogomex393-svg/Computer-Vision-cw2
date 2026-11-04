# Transparent Object Segmentation for an Original Jetson Nano

This is a working report draft. Only measured values from the result receipts may replace pending fields. The system is semantic segmentation, not instance segmentation or object counting.

## Objective and scope

Given one RGB image, predict background, transparent things and transparent stuff at every original-resolution pixel. Transparent surfaces inherit background appearance and can have weak or misleading edges. The project studies edge deployment of a lightweight segmentation model with two established training improvements, rather than claiming an original segmentation architecture.

The teacher is SegFormer-B2, a hierarchical Mix Transformer encoder with an all-MLP fusion decoder. The teacher starts from NVIDIA's ADE20K weights and receives a new three-class classifier, then must be fine-tuned on Trans10K. MobileNetV3-Large supplies the student's approximately 1/4, 1/8, 1/16 and 1/32 features. Learned 1x1 projections map them to 64 channels; bilinear upsampling, summation and depthwise/pointwise convolution fuse the pyramid. A three-channel classifier produces logits. The measured three-class B2 implementation has 27,348,931 parameters; the student inference network has 3,064,819. The auxiliary boundary head adds 18,497 parameters during training and is removed for deployment.

## Data and integrity

The official Trans10K v1 source index contains 5000 training, 1000 validation and 4428 test images. Exact decoded-image audit discovered 11 cross-split duplicate groups. The actual experiment protocol, explicitly named leak-free-v1, uses 4990 training and 999 validation images while preserving the full 4428-image test set. Ten training and one validation entries are excluded only from derived manifests to prevent overlap; original files and source indices remain intact. The foreground labels represent things such as bottles and stuff such as glass walls. The official RGB mask channel means 0/85/255 map to internal classes 0/1/2; internal 255 is reserved for ignore/padding. Unknown encodings fail rather than becoming background. Trans10K v2 reuses images with a finer taxonomy and is not an independent cross-dataset evaluation.

The audit records pairing, resolution, raw colors, class populations, damaged files, official counts and duplicates. Eighteen training masks have visually confirmed 90-degree orientation errors; an explicit SHA-bound recipe corrects them without changing source files, labels or split membership. Sixteen rotate clockwise and two counterclockwise. Validation/test geometry is unchanged. See DATA_AUDIT.md and the diagnostic sheets. Keep the actual audit receipt with the report. Any self-captured material is grouped by object/session before splitting, with adjacent video frames in one group; it is reported separately from the public benchmark.

All inputs use RGB, ImageNet normalization and square letterboxing. Mask interpolation is nearest-neighbor, with ignore padding. A deterministic cache retains resized training images and labels without changing augmentation results. Evaluation upsamples floating logits to the network input grid, removes the exact padding and resizes to the original grid before argmax.

## Losses

Let V denote valid pixels. The segmentation term is cross-entropy averaged over V plus Dice averaged over foreground classes 1 and 2:

`Lseg = CE_valid + mean_c(1 - (2 sum(p_c*y_c) + eps)/(sum(p_c) + sum(y_c) + eps))`.

For an absent foreground class, the corresponding term is its mean predicted foreground probability over valid pixels, explicitly penalizing false positives. An all-ignore batch returns a differentiable zero. FP32 reductions avoid FP16 pixel-sum overflow.

The auxiliary boundary target marks unequal adjacent valid semantic labels, dilates by a fixed radius of one training pixel and excludes invalid neighborhoods. A light decoder feature head predicts the boundary. Positive and negative BCE contributions are balanced when both exist; single-class targets use ordinary BCE. The head is training-only and has no dependency path into main logits.

Output distillation uses frozen eval-mode teacher probabilities on the exact same geometrically transformed images:

`Lkd = T^2 / |V| * sum_{i in V} KL(softmax(z_teacher/T) || softmax(z_student/T))`.

The teacher's output is bilinearly aligned before KL. Different normalization, if configured, is converted on the same image tensor. The total is `Lseg + lambda_b*Lboundary + lambda_kd*Lkd`. S0 disables both, S1 enables boundary only, S2 distillation only, and S3 both. Disabled branches do not instantiate unused teacher dependencies or contribute losses.

## Experimental controls

Use docs/EXPERIMENTS.md and the actual effective config files. Student initialization, input size, decoder width, data, augmentation, optimizer, seed, budget and selection rule are controlled. Small calibration choices use validation only and are recorded independently from the complete final runs. Teacher validation must exceed S0 before formal distillation.

Results to insert from generated JSON: T/S0/S1/S2/S3 mIoU, each class IoU, merged foreground IoU/Dice, original-resolution boundary F1 and fixed corruption scores. Include unsuccessful/no-gain ablations as observed. No values are supplied here before measurement.

## Deployment and optimization

Export only the student inference graph with fixed batch=1 and size 256/384/512; the main candidate uses ONNX opset 11. Removing the auxiliary head must preserve logits exactly in eval mode. Host export checks maximum/mean absolute logit differences and argmax agreement using ONNX Runtime; the companion NPZ permits the same check on Nano.

Nano optimization compares TensorRT FP32 versus FP16 and three input resolutions. Engines are generated on the original Nano with its JetPack 4.6.x/TensorRT 8.x installation. Resource and latency measurements require that actual board. Accuracy-speed tradeoffs are reported separately from the two training improvements.

Pure engine execution uses synchronized CUDA events. End-to-end latency includes capture, preprocessing, inference, restoration, overlay and output. A bounded asynchronous queue prioritizes the newest frame and explicitly records discarded frames and frame age. Warmup is excluded; mean, P50, P95, processing FPS, power mode, RAM, GPU utilization and temperatures are retained. Missing telemetry stays unavailable.

## Current engineering evidence

Host regressions cover masks, coordinates, manually computable metrics, degenerate losses, teacher gradients, exact epoch-boundary resume, auxiliary removal, ONNX alignment and image/video/headless operation. A separately installed CPU environment runs the same checks. Synthetic colored-shape overfit results are functional diagnostics only. Formal experiment and Nano sections must be updated from docs/PROGRESS.md/results after execution, not inferred from this implementation description.

## Limitations and conclusions pending measurements

RGB alone cannot fully disambiguate every transparent surface; mirrors, thin edges, clutter and severe reflections can fail. The teacher is heavier than the deployment student. Edge constraints, board thermal state and scene domain shift must be measured. This report must distinguish implemented code, real-data model quality and physical-board validation.

References: [TransLAB/Trans10K](https://xieenze.github.io/projects/TransLAB/TransLAB.html), [Trans2Seg](https://github.com/xieenze/Trans2Seg), [SegFormer](https://github.com/NVlabs/SegFormer), [MobileNetV3 implementation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), [TransCues](https://arxiv.org/abs/2512.07034). Detailed source/version decisions are in docs/SOURCES.md.
