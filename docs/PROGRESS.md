<!-- CV_CW2_AUTO_PROGRESS_BEGIN -->
## 自动运行进度

更新时间：2026-11-05T00:04:27+08:00
当前阶段：S0：验证第 16 轮
运行目录：runs\final
以下为最近检查点；恢复前仍需核对进程和日志。

T：completed，完整轮次 20/20，最佳验证 mIoU 0.923996。
S0：running，完整轮次 15/20，最佳验证 mIoU 0.874818。
S1：pending，完整轮次 0/pending。
S2：pending，完整轮次 0/pending。
S3：pending，完整轮次 0/pending。

模型训练、测试、导出与真实 Nano 验收是不同状态。
<!-- CV_CW2_AUTO_PROGRESS_END -->









































































# 当前进度

## 已完成并验证

- 工程已实现数据审计、标签转换、学生/教师、独立边界与蒸馏、训练恢复、原图评价、ONNX、图像/视频演示和 Nano 工具。
- 项目 CUDA 环境和从公开源新安装的独立 CPU 环境均通过关键回归；最新完整干净 CPU 回归 23 项通过，25.37 秒。日志 artifacts/final_cpu_pytest.xml。
- 合成样本 GPU 训练完成 60 轮、180 次更新，损失 2.0881 降至 0.07437。它只证明链路有效，不代表透明物体准确率。
- 合成样本学生 ONNX opset 11 与 PyTorch 对齐，最大绝对误差小于 3.82e-6，测试掩膜一致率 1.0。辅助头移除后主输出完全一致。
- 官方 Trans10K 三个压缩包全部下载、CRC 校验、解压完成。测试集中断后通过 HTTP Range 续传，不是残缺数据。
- NVIDIA SegFormer-B2 和 Torchvision MobileNetV3 官方预训练权重已取得。384 输入、batch 4 在 RTX 4060 8GB 上通过显存/速度探测。
- 原任务书、需求契约和实现/交接文档均已保存。工作分支已上传并完成独立远端检出验证；最新已推送的主训练实现为 a2ec4a2。

## 正在进行

已逐张核对并修正 18 个训练掩膜方向异常，配方和 SHA256 在 configs/trans10k_v1_geometry.json，检查图位于 results/geometry_review_*.jpg。验证和测试集不做此类修正。完整审计已完成：几何与标签检查通过，但官方源有 11 组跨集合完全重复。原始审计保留失败记录；独立生成 leak-free-v1 清单并验证交集为空，实际训练/验证/测试为 4990/999/4428，完整测试集保留。训练、验证和四种条件的测试缓存均已完成。最新完整回归 16 项通过。T 首个完整 epoch 已完成，loss=0.3784324，validation mIoU=0.8733769，耗时 1237.84 秒。checkpoint 在 runs/formal/T；状态是预定的 epoch 边界停止，不是 20 轮完成。性能优化已验证：同一 checkpoint 的全部 999 张验证图只用 102.868 秒，所有原有指标、混淆矩阵和有效像素总数完全一致。原目录保留为性能 pilot；新的正式系列使用 runs/final，保持全部模型各 20 轮。训练会自动更新本文件顶部和 runs/final/CURRENT_STAGE.json。

正式训练预算为 T 与 S0～S3 各 20 个完整 epoch、seed 42；正式学生保持相同初始化和训练协议。先完成 T 的首轮计时，再恢复完整预算。验证集小范围校准单独记录，不使用测试集调参。

## 下一步

1. 以本文件顶部自动进度、runs/final/CURRENT_STAGE.json 和实际进程为准；不重复启动已经在运行的流水线。
2. 当前正式预算 T、S0～S3 各 20 轮；验证集校准为四个独立候选各 3 轮。日志在 artifacts/final_pipeline.stdout.log 与 stderr.log。
3. 完成 S0/S1/S2/S3、鲁棒性、正式 ONNX、报告图表和模型包。
4. 上传工作分支，核对远端，再从新检出目录运行 CPU 快速验证。

## 本机位置

工程解释器 F:/PyLibs/cv_cw2_lib/Scripts/python.exe；干净 CPU 解释器 F:/PyLibs/cv_cw2_clean_lib/Scripts/python.exe。
数据 F:/Datasets/computer_vision/Trans10K；训练缓存 F:/Datasets/computer_vision/Trans10K_train_clean384，验证缓存 F:/Datasets/computer_vision/Trans10K_val_clean384，测试缓存 F:/Datasets/computer_vision/Trans10K_test384。实际清单 data/clean_manifests。
权重缓存 F:/Models/computer_vision/cw2。训练时设置 TORCH_HOME 与 HF_HOME 到该目录，避免写 C 盘缓存。
合成结果 runs/smoke、artifacts/student_smoke.onnx、artifacts/smoke_demo。

## 仍待真实条件

GitHub 正确账号已登录；工作分支已上传并核对远端提交 7b694c31cbf16f02f1d91d2bf475b80b44b524d0。首次远端干净检出发现测试输出父目录缺失，已修复并复验通过 14 项；此后本地几何与缓存新增检查达到 15 项通过。

未提供真实 Nano；板端 API、引擎、速度、温度、功耗、摄像头和现场录制仍待实机验证。仓库公开，不能创建私有 Release；正式模型包完成后再处理私有组内存储，不把大文件硬塞 Git。

## 恢复规则

先读本文件、docs/VALIDATION.md、docs/work_state.json，再查 Git 状态与活动进程。不要重建工程。训练源码、配置、清单身份变化须使用新 run 目录，不能覆盖已有实验。不得把仍在执行的正式训练写成已完成。
