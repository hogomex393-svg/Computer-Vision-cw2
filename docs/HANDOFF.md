# 组员交接与唯一待办

本机 L3 工作已完成：训练、完整测试、ONNX、演示回放、报告、模型包和 GitHub 交付均有实际证据。原版 Nano 实测和现场录制尚未进行，交由两位组员接手。

## 取得工程与模型

源码与报告：[工作分支](https://github.com/hogomex393-svg/Computer-Vision-cw2/tree/codex/transparent-segmentation)。模型包使用的源码版本为 `34530d7915c3691b606653d6158a23111fe45248`；分支上的后续提交只补充交接说明和最终复核记录。

模型包：[cw2_models_bf16_v2.zip](https://github.com/hogomex393-svg/Computer-Vision-cw2/releases/tag/untagged-249e98c24b10a87ef648)，228205346 字节，约 228 MB。需使用有本仓库推送权限的 GitHub 账号登录。保持草稿状态，不点击 Publish。v1 保留作历史记录，交付使用 v2；v2 修正了回放命令路径，模型、测试和视频内容未变。

SHA256：`b0f1b57f60c25cc8c54d8284ee97b0b8073b8b58730318cb6392106428eb4246`。已从 GitHub 完整下载并核对这个摘要，匿名访问被拒绝。下载到本机后可用 PowerShell `Get-FileHash cw2_models_bf16_v2.zip -Algorithm SHA256` 再核对。

将包解压到单独目录，在包根目录运行 `nano.py`。包中有五个最佳推理 checkpoint、S1 的 256/384/512 ONNX、S0 的 384 ONNX、数值对齐参考、完整实验记录和数据清单，以及 `demo/input_replay.mp4`、`demo/S0.mp4`、`demo/final.mp4`。完整优化器和随机状态的续跑 checkpoint 保留在训练电脑的 `runs/final_bf16/*/last.pt`。

## 已完成的真实结果

T、S0、S1、S2、S3 都完成 20 轮。每个模型都在全部 4428 张测试图上完成 clean、brightness、blur、jpeg 四种条件评价，恢复到原图尺寸计算。使用明确披露的 leak-free-v1 划分 4990/999/4428；原始数据与 18 个训练标签方向修正证据均保留。

部署选择在测试前按验证集锁定为 S1。干净测试 mIoU：T 92.3477%，S0 87.4782%，S1 87.5862%，S2 87.4738%，S3 87.6888%。S1 相对 S0 仅提高 0.1079 个百分点；S3 测试略高也不据此更换已锁定模型。单个 seed、收益小和失败案例都在报告中如实说明。

四个 ONNX 导出均通过 PyTorch/ONNX Runtime 数值检查，移除边界辅助头后主输出不变；全部参考掩膜一致率为 1.000，最大 logits 误差为 0.00003910。独立远端检出的 CPU 回归 29 项通过。

三段 MP4 都是同一组验证静态图组成的 90 帧、9 秒回放。它们验证了真实训练模型的读视频、分割和保存链路，不是摄像头实拍，也不是 Nano 性能。查看 `results/final/FINAL_REVIEW.json`、`results/final/results.json`、`docs/REPORT_DRAFT.md` 和 `docs/VALIDATION.md` 获取证据。

## 组员唯一待办

1. 板端负责人：在原版 Nano 4GB 上核对 JetPack 4.6.x、TensorRT 8.x、Python/PyCUDA/OpenCV。包根目录依次执行下面命令，先用附带回放，不依赖摄像头。然后按 JETSON.md 完成 256/384/512、FP32/FP16、同步/异步、十分钟稳定性和资源记录。预期产物为环境 JSON、引擎、对齐 JSON、matrix/benchmark JSON 和 tegrastats 日志。
2. 现场负责人：准备透明物件、三种背景和无隐私摄像头素材。按 DEMO_SCRIPT.md 拍同一视频的 S0/S1 对照、真实 Nano 工作画面和真实失败案例，记录来源与场景。预期产物为原始录像、最终成片和素材说明。

```bash
python3 nano.py env --out nano_results/environment
python3 nano.py build --onnx models/S1_384.onnx --engine models/S1_384_fp32.engine --precision fp32
python3 nano.py build --onnx models/S1_384.onnx --engine models/S1_384_fp16.engine --precision fp16
python3 nano.py verify --engine models/S1_384_fp16.engine --reference models/S1_384.reference.npz --out nano_results/alignment
python3 nano.py demo --engine models/S1_384_fp16.engine --source demo/input_replay.mp4 --out nano_results/playback
python3 nano.py matrix --models-dir models --model-prefix S1 --source demo/input_replay.mp4 --out nano_results/matrix --duration 600
```

这些命令的包内输入路径已核对。TensorRT 引擎仍必须在目标板生成并验收；主机没有伪造板端数据。FP16 若出现非有限值或对齐失败，保留错误并使用独立构建的 FP32 引擎继续诊断。
