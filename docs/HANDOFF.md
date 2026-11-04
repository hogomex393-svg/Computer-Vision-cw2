# 组员交接与唯一待办

本文件随实际运行状态更新。先看 docs/PROGRESS.md，不要把下面“代码已实现”理解为“真实实验全部完成”。

## 当前交接状态

工程代码：数据审计、MobileNetV3 学生、SegFormer 教师、边界监督、蒸馏、训练/恢复、原图评价、ONNX、OpenCV 演示和 Nano 工具已经实现。主机及干净 CPU 环境的实际测试见 docs/VALIDATION.md。

真实数据：官方 train/validation/test 全部下载、CRC 校验并解压，逐张审计完成。18 个训练标签方向异常已显式修正；11 组源数据跨集合重复保留在原始失败报告中。实际实验使用独立验收通过的 leak-free-v1 清单 4990/999/4428，全部测试图保留。

正式训练：T、S0、S1、S2、S3 目前 pending。已有 runs/smoke 仅为合成样本功能验证。正式 checkpoint、日志和结果将在完成后写入本文件。

模型导出：合成样本学生已经完成 PyTorch/ONNX Runtime 对齐，不能作为真实透明物体模型。正式模型导出 pending。

Nano：环境检查、引擎构建、FP32/FP16 runner、对齐检查、视频/摄像头、计时/资源采集和十分钟矩阵脚本已实现；没有真实板卡连接，硬件验证 pending。

GitHub：目标 https://github.com/hogomex393-svg/Computer-Vision-cw2 ，工作分支 codex/transparent-segmentation。DzAsy11 已完成官方登录；工作分支已上传，远端提交 7b694c31cbf16f02f1d91d2bf475b80b44b524d0 已核对。后续正常提交继续更新。仓库公开，不自动把大模型、原始数据或原始录像放进提交。

## 另外两位组员的最短接手路径

1. 板端负责人：取得正式模型包后，先 `python3 nano.py env --out nano_results/environment`，提供实际 Nano/JetPack/TensorRT/Python 版本。按 docs/JETSON.md 构建引擎、验证 NPZ 对齐，跑本地视频，再跑矩阵。预期产物为 engine、alignment.json、benchmark.json、matrix.json、环境与 tegrastats 日志。这些是实机验收工作，核心 runner 已实现。
2. 现场负责人：准备透明物件、三种背景、摄像头与无隐私回放素材，按 docs/DEMO_SCRIPT.md 录制。保留真实失败案例、同一视频的 S0/最终模型比较和现场 Nano 镜头。预期产物为原始录像、成片及素材来源记录。

## 尚需外部条件

目前缺少真实 Nano 及现场物件/摄像头；GitHub 认证已解决。GPU 已在本机可用，官方数据不需账号登录，普通代码问题由 Codex 继续修复。GitHub 推送与模型包分享的权限会在实际操作时核实；出现确实不可替代的条件才补到这里。

不要从聊天里发送密码。若使用私有组内存储交付模型，只需提供获得授权的存储位置或成员访问方式；公开 GitHub 仓库不能提供私有 Release。
