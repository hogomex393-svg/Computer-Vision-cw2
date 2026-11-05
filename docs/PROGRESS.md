<!-- CV_CW2_AUTO_PROGRESS_BEGIN -->
## 自动运行进度

更新时间：2026-11-05T08:50:13+08:00
当前阶段：S0：验证第 5 轮
运行目录：runs\final_bf16
以下为最近检查点；恢复前仍需核对进程和日志。

T：completed，完整轮次 20/20，最佳验证 mIoU 0.923996。
S0：running，完整轮次 4/20，最佳验证 mIoU 0.836061。
S1：pending，完整轮次 0/pending。
S2：pending，完整轮次 0/pending。
S3：pending，完整轮次 0/pending。

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
