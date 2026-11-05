<!-- CV_CW2_AUTO_PROGRESS_BEGIN -->
## 自动运行进度

更新时间：2026-11-05T08:07:58+08:00
当前阶段：calibration_kd_T2_W025：验证第 1 轮
运行目录：runs\final
以下为最近检查点；恢复前仍需核对进程和日志。

T：completed，完整轮次 20/20，最佳验证 mIoU 0.923996。
S0：completed，完整轮次 20/20，最佳验证 mIoU 0.877504。
S1：pending，完整轮次 0/pending。
S2：pending，完整轮次 0/pending。
S3：pending，完整轮次 0/pending。

模型训练、测试、导出与真实 Nano 验收是不同状态。
<!-- CV_CW2_AUTO_PROGRESS_END -->


# 项目进度与恢复入口

## 已完成

数据审计、模型与损失、训练恢复、原图评价、ONNX、OpenCV 演示和 Nano 工具均已实现。独立 CPU 环境的完整回归已有 23 项通过；2026-11-05 新增模型包完整性与 Nano 结果真实性检查通过，详见 VALIDATION.md。

官方 Trans10K 三个压缩包全部下载、CRC 校验并解压。18 个训练掩膜方向异常按 SHA 绑定配方在加载时修正，原始数据不改。官方划分有 11 组跨集合重复；实际实验采用 leak-free-v1 清单 4990/999/4428，保留完整测试集。训练、验证、四种测试条件的缓存以及原图掩膜无损缓存均已完成。

正式系列 runs/final 中，T 和 S0 都完成 20 个完整 epoch。最佳验证 mIoU：T=0.9239955207704398，S0=0.8775038527093654。边界校准两个候选均完成 3 轮，lambda=0.1/0.2 对应 0.8239747173978853/0.8259085263637763；最终参数须等蒸馏候选也完成再统一选择。

真实 S0 的 384 ONNX 已导出并通过数值对齐，最大误差小于 4.08e-5，三组掩膜一致率均为 1.0。真实验证图片预览 artifacts/S0_validation_preview.png；九秒验证静态图回放 artifacts/S0_validation_replay.mp4。回放不代表实时摄像头或 Nano 性能。

GitHub 工作分支已实际推送。当前已核对的远端版本 be6f3d0；训练核心源码身份为 a2ec4a2 对应的五个文件 SHA。后续正常提交不覆盖已有历史。

## 当前恢复点

2026-11-05 00:35:41，第一项蒸馏校准在第一个 epoch 出现 Nonfinite training loss，流程按设计失败退出。失败记录完整保留在 runs/failed_kd_20261105_003541，原日志 artifacts/final_pipeline.stdout.log 与 stderr.log。没有伪造完成轮次，也未删除已经完成的 T、S0 和边界校准。

独立诊断重放全部 1248 个训练批次未复现，证据 artifacts/kd_failure_diagnosis.json；根因尚未确认。08:05 使用相同训练源码、配置、数据重新执行正式 pipeline。local.retry_pipeline.py 只在实际异常退出后读取失败栈并保存张量现场，不改变训练计算；启动回执 artifacts/final_retry_launch.json。运行状态以顶部自动记录、run.json、实际进程和日志为准。

当前日志：artifacts/final_pipeline_retry.stdout.log / stderr.log。若再次异常，检查 artifacts/kd_production_failure.json 与 kd_production_failure_batch.pt。禁止在没有证据时将重试称为根因修复。

## 下一步

1. 完成蒸馏校准，继续 S1、S2、S3 各 20 轮。全部学生的主干初始化、数据、预算及选模协议保持一致。
2. 全部五组在 4428 张测试图上完成干净/亮度/模糊/JPEG 评价；最终学生只按验证集选择。
3. 完成最终学生 256/384/512 与 S0 384 导出、案例图、演示和报告；模型包通过内容与 SHA 核验后上传仅协作者可访问的草稿 Release。
4. 正常提交并核对远端，在新的检出目录运行 CPU 验证。

## 运行位置与命令

GPU Python：F:/PyLibs/cv_cw2_lib/Scripts/python.exe。独立 CPU Python：F:/PyLibs/cv_cw2_clean_lib/Scripts/python.exe。
原始数据 F:/Datasets/computer_vision/Trans10K。缓存分别为 Trans10K_train_clean384、Trans10K_val_clean384、Trans10K_test384。实际清单 data/clean_manifests。
权重缓存通过 TORCH_HOME=F:/Models/computer_vision/cw2/torch、HF_HOME=F:/Models/computer_vision/cw2/huggingface 指定；已有权重时设置 HF_HUB_OFFLINE=1。

```powershell
python project.py pipeline --root F:/Datasets/computer_vision/Trans10K --manifests data/clean_manifests --cache F:/Datasets/computer_vision/Trans10K_train_clean384 --validation-cache F:/Datasets/computer_vision/Trans10K_val_clean384 --test-cache F:/Datasets/computer_vision/Trans10K_test384 --out runs/final --device cuda --calibrate
```

恢复前先检查进程，不能重复启动。完成的实验按身份校验后复用；未完成实验只能从已保存 epoch 边界恢复。若失败发生在第一个 epoch 且无 last.pt，先保留失败目录，不能假装存在可恢复 checkpoint。训练核心源码、配置或清单身份变化须用新目录，不绕过身份核对。

runs/formal/T 是一个 epoch 的性能 pilot，runs/smoke 是合成功能验证，两者不属于正式系列。原图评价优化已在完整 999 张验证图上核实所有指标、混淆矩阵、有效像素与旧实现完全一致，证据 results/optimized_validation_equivalence.json。

## 外部条件与交付边界

GitHub 认证、GPU 和数据已解决。缺少真实原版 Nano 连接及现场物件/摄像头，板端速度、温度、功耗、实拍和最终录像仍需组员完成；不能用 PC 或回放代替。

源码和小型结果已上传；正式模型包尚未生成、上传。目标仓库公开，采用草稿 Release 组内交付，先用非敏感清单核实匿名拒绝，再上传并核对模型摘要。不能将草稿发布为公开 Release。具体接手步骤见 HANDOFF.md。
