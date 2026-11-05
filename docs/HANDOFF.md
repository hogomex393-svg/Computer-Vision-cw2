# 组员交接与唯一待办

本文件随实际运行状态更新。先看 docs/PROGRESS.md，不要把下面“代码已实现”理解为“真实实验全部完成”。

## 当前交接状态

工程代码：数据审计、MobileNetV3 学生、SegFormer 教师、边界监督、蒸馏、训练/恢复、原图评价、ONNX、OpenCV 演示和 Nano 工具已经实现。主机及干净 CPU 环境的实际测试见 docs/VALIDATION.md。

真实数据：官方 train/validation/test 全部下载、CRC 校验并解压，逐张审计完成。18 个训练标签方向异常已显式修正；11 组源数据跨集合重复保留在原始失败报告中。实际实验使用独立验收通过的 leak-free-v1 清单 4990/999/4428，全部测试图保留。

正式训练：runs/final 中 T、S0 各完成 20 轮，最佳验证 mIoU 分别为 0.9239955207704398、0.8775038527093654。checkpoint 和 epochs.jsonl 在各自目录，小型回执在 results。两项边界校准各完成 3 轮；蒸馏校准因非有限损失曾停止，失败记录已保留，2026-11-05 08:05 按原配置重新运行并捕获异常现场。S1、S2、S3 及全部最终测试尚未完成，最新阶段以 PROGRESS 顶部为准。runs/formal/T 是性能 pilot，runs/smoke 是合成功能验证。

模型导出：真实 S0 的 runs/final/exports/S0_384.onnx 已通过 PyTorch/ONNX Runtime 对齐，最大绝对误差小于 4.08e-5、三组掩膜一致率 1.0。验证图片和九秒静态图回放已跑通；最终学生和完整模型包仍待正式实验结束。

Nano：环境检查、引擎构建、FP32/FP16 runner、对齐检查、视频/摄像头、计时/资源采集和十分钟矩阵脚本已实现；没有真实板卡连接，硬件验证 pending。

GitHub：目标 https://github.com/hogomex393-svg/Computer-Vision-cw2 ，工作分支 codex/transparent-segmentation。DzAsy11 已完成官方登录；工作分支已上传，远端提交 7b694c31cbf16f02f1d91d2bf475b80b44b524d0 已核对。后续正常提交继续更新。仓库公开，不自动把大模型、原始数据或原始录像放进提交。

## 另外两位组员的最短接手路径

1. 板端负责人：取得正式模型包后，先 `python3 nano.py env --out nano_results/environment`，提供实际 Nano/JetPack/TensorRT/Python 版本。按 docs/JETSON.md 构建引擎、验证 NPZ 对齐，跑本地视频，再跑矩阵。预期产物为 engine、alignment.json、benchmark.json、matrix.json、环境与 tegrastats 日志。这些是实机验收工作，核心 runner 已实现。
2. 现场负责人：准备透明物件、三种背景、摄像头与无隐私回放素材，按 docs/DEMO_SCRIPT.md 录制。保留真实失败案例、同一视频的 S0/最终模型比较和现场 Nano 镜头。预期产物为原始录像、成片及素材来源记录。

## 尚需外部条件

目前缺少真实 Nano 及现场物件/摄像头；GitHub 认证已解决。GPU 已在本机可用，官方数据不需账号登录，普通代码问题由 Codex 继续修复。GitHub 推送与模型包分享的权限会在实际操作时核实；出现确实不可替代的条件才补到这里。

不要从聊天里发送密码。若使用私有组内存储交付模型，只需提供获得授权的存储位置或成员访问方式；公开 GitHub 仓库的已发布 Release 公开可见；本项目计划使用未发布的草稿 Release。GitHub 官方 API 文档说明，草稿仅向有推送权限的成员列出，上传工具还会先用无敏感内容的清单实际验证匿名访问被拒绝，再上传模型。

## 模型包生成和组内交付

正式流水线成功后使用：

```bash
python package_artifacts.py --runs runs/final --manifests data/clean_manifests --out artifacts/cw2_models_v1
python publish_artifacts.py --archive artifacts/cw2_models_v1.zip --check-only
python publish_artifacts.py --archive artifacts/cw2_models_v1.zip --account DzAsy11 --tag cw2-host-v1
```

最后一条仅创建草稿，不发布公开 Release。它使用系统 Git 已登录账号，不写入或输出凭据；固定操作本任务仓库，核对账号与推送权限。先验证草稿和清单不能匿名访问，再上传模型并核对 GitHub 服务端 SHA256。真正成功的链接与回执要等实际执行后填写，不能把工具写好当成上传完成。

组员应使用有本仓库推送权限的 GitHub 账号登录草稿页下载。不要点击 Publish：发布会改变可见范围。模型包包含最佳推理 checkpoint、ONNX/数值对齐参考、实验日志和清单；完整优化器/RNG 续跑状态仍在本机各实验 last.pt。

来源：[GitHub Releases API](https://docs.github.com/en/rest/releases/releases#list-releases)。
