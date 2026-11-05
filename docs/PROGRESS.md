<!-- ACTUAL_DELIVERY_BEGIN -->
## 主机实验与模型交付已完成

T、S0、S1、S2、S3 均完成 20 轮；全部模型完成 4428 张原图上的四种条件测试。最终学生按验证集选择为 S1，正式模型和同一输入的 S0/最终模型演示已打包。

模型包：[cw2_models_bf16_v1.zip](https://github.com/hogomex393-svg/Computer-Vision-cw2/releases/tag/untagged-a5fe408ef7f4e9df4648)。需要有仓库推送权限的 GitHub 账号登录；保持 Release 为草稿，不点击 Publish。

体积：228205341 字节。SHA256：`630a471809b1befc786e8bab8354a5fa9da9f0871a726789e81e4bec2b6cb86d`。包内源码版本：`ee76dcda308a581c25804a5b7ddceb1e7e7de6aa`。

模型包含五个最佳 checkpoint、最终学生 256/384/512 ONNX、S0 384 ONNX、对齐参考、实验记录、数据清单和验证静态图回放。已实际验证 GitHub 上传摘要、匿名访问拒绝和远端检出的 CPU 回归。

仍需组员：按 JETSON.md 在原版 Nano 构建引擎、对齐和十分钟性能矩阵；按 DEMO_SCRIPT.md 做实拍。本机回放不代表 Nano 实测。
<!-- ACTUAL_DELIVERY_END -->

<!-- CV_CW2_AUTO_PROGRESS_BEGIN -->
## 自动运行进度

更新时间：2026-11-05T15:58:33+08:00
当前阶段：主机训练、测试、导出流水线完成；模型包交付与 Nano 验收另行记录
运行目录：runs\final_bf16
以下为最近检查点；恢复前仍需核对进程和日志。

T：completed，完整轮次 20/20，最佳验证 mIoU 0.923996。
S0：completed，完整轮次 20/20，最佳验证 mIoU 0.872828。
S1：completed，完整轮次 20/20，最佳验证 mIoU 0.884096。
S2：completed，完整轮次 20/20，最佳验证 mIoU 0.877203。
S3：completed，完整轮次 20/20，最佳验证 mIoU 0.881062。

模型训练、测试、导出与真实 Nano 验收是不同状态。
<!-- CV_CW2_AUTO_PROGRESS_END -->
# 当前任务与恢复入口

用户要求完成剩余训练、测试、模型打包并整理上传 GitHub，普通问题自主解决。活动训练目录为 runs/final_bf16，顶部是自动更新的实际状态。工程 L3 主机交付仍在进行，不能称全部完成。

## 当前执行链

1. 隐藏 GPU 流水线按 T、S0、参数校准、S1、S2、S3、全部模型完整测试、ONNX、案例图顺序运行。T 从已完成的 runs/final/T 原样核验复制；四个学生统一 BF16、seed 42、384 输入、每组 20 轮。训练源码于 6902a18 冻结，后续改动只在交付工具和文档。
2. 隐藏 delivery.py 等待该流水线成功。它核对五组实验、每张测试图覆盖、全部条件、验证集选模及导出，再生成同一验证静态图回放的 S0/最终输出、报告与小型图表。
3. --upload 明确授权的收尾仅操作指定 GitHub 仓库与工作分支：正常提交、远端核对、独立检出 CPU 测试、模型和演示包内部校验、草稿 Release 匿名拒绝检查、上传摘要核验，最后更新交接首页和工作状态。失败会记录并停止，不写成成功。

训练日志 artifacts/final_bf16.stdout.log / stderr.log，启动回执 artifacts/final_bf16_launch.json。交付日志 artifacts/delivery_bf16.stdout.log / stderr.log，状态 artifacts/delivery_bf16/delivery_state.json，启动回执 artifacts/delivery_bf16_launch.json。恢复前必须检查这些文件和实际进程，禁止重复启动。

## 已验证事实

官方 Trans10K 三个压缩包全部下载、CRC 校验、解压、逐张审计。18 个训练标签方向修正不改原始文件。11 组跨集合完全重复导致采用独立 leak-free-v1 清单 4990/999/4428；完整测试集保留。各条件缓存和原图掩膜无损缓存已完成。

教师完成 20 轮，最佳验证 mIoU=0.9239955207704398。旧 FP16 S0 完成 20 轮，0.8775038527093654 仅作为历史记录，不与新 BF16 学生混入同一消融。新 S0 首轮后导出预检通过，三组掩膜一致率 1.0，最大误差 1.15e-5 以下；不是最终模型验收。

FP16 蒸馏失败已定位到 encoder.16.0 前向溢出，FP32 中间幅度 67143.6796875 大于 FP16 最大值 65504。保存的实际失败批次在 BF16 下前向与梯度均有限。四个学生及全部校准因此在新目录统一重跑。完整证据与论文报告解释在 NUMERICAL_FIX.md；旧运行和失败张量保留在 runs/final。

完整 CPU 回归 27 项通过；后续交付守卫检查通过。真实 90 帧回放生成已用旧训练 S0 验证，包含 H.264 输出、计时和来源 SHA。GitHub DzAsy11 的 API 身份和 push 权限已实际通过，凭据不进入文件。

## 恢复命令

GPU Python F:/PyLibs/cv_cw2_lib/Scripts/python.exe；独立 CPU Python F:/PyLibs/cv_cw2_clean_lib/Scripts/python.exe。TORCH_HOME=F:/Models/computer_vision/cw2/torch，HF_HOME=F:/Models/computer_vision/cw2/huggingface，HF_HUB_OFFLINE=1。

训练完整参数在 artifacts/final_bf16_launch.json，包含 --out runs/final_bf16 --teacher-run runs/final/T --calibrate。交付完整参数在 artifacts/delivery_bf16_launch.json。确认没有活动进程后才恢复。模型/数据/核心训练代码身份改变不能绕过 guard；第一轮失败且无 last.pt 时先保留失败目录。

## 真实外部条件

Nano 未提供连接；目标板引擎、FP32/FP16 数值、速度/资源/十分钟矩阵及实拍交给两位组员。主机模型和回放不冒充 Nano 实测。

模型包和下载链接须以 delivery_state=completed 及 results/final/delivery.json 为准。仓库公开，模型仅进协作者草稿 Release，不发布公开 Release。课程原始任务书在 TASK_BRIEF.md，需求及版本在 REQUIREMENTS.md/work_state.json。Git 历史保留各阶段状态。

## 自动续查

本聊天已建立每 15 分钟续查，ID cw2，名称“完成计算机视觉 CW2 交付”。正常运行无须重复汇报；失败时继续修复，全部交付后再做实际图表/视频/包内容/远端复核并停用续查。不要把 delivery.py 的 completed 单独当成人工最终复核完成；Nano 保持未实测。收据 artifacts/continuation_automation.json。
