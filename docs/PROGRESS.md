# 最终进度

最终内容复核时间：2026-11-05T16:22:56+08:00。本机 L3 交付完成，训练、测试及自动交付进程均已退出，不再运行本项目的 GPU 计算。定时续查在本轮最终远端核对后结束。

五个模型各完成 20 轮；全部 20 个模型/条件组合各覆盖 4428 张原图。最终选择为 S1，依据验证集、在测试前锁定。完整结果和局限见 REPORT_DRAFT.md 与 results/final。

四个正式 ONNX、六个固定案例、六个全局失败案例、两张结果图及三段回放均已检查。远端干净检出 29 项测试通过。v2 模型包已上传协作者草稿 Release，完整下载字节数和 SHA256 与本机一致，匿名访问拒绝。

当前交付：[cw2_models_bf16_v2.zip](https://github.com/hogomex393-svg/Computer-Vision-cw2/releases/tag/untagged-249e98c24b10a87ef648)。228205346 字节，SHA256 `b0f1b57f60c25cc8c54d8284ee97b0b8073b8b58730318cb6392106428eb4246`。包内源码版本 `34530d7915c3691b606653d6158a23111fe45248`。本机文件 `artifacts/delivery_bf16/cw2_models_bf16_v2.zip`，演示位于 `artifacts/delivery_bf16/demo`。v1 保留历史，不再作为最新交付。

## 尚待外部条件

仅剩组员的原版 Nano 实测和现场拍摄。主机完成不等于板端验收。具体责任、命令和预期产物统一见 HANDOFF.md，不另列重复任务。

## 可恢复证据

正式实验：`runs/final_bf16`；精度修复前的旧实验与失败证据：`runs/final`。旧 FP16 S0 不混入正式 BF16 消融。教师复用收据：`runs/final_bf16/T/teacher_reuse.json`。数据协议和精度修订：DATA_AUDIT.md、NUMERICAL_FIX.md、REQUIREMENTS.md、work_state.json。

最终复核：`results/final/FINAL_REVIEW.json`、`results/final/delivery.json`。完整测试/导出状态：`runs/final_bf16/pipeline.json`。自动交付历史日志：`artifacts/final_bf16.*.log`、`artifacts/delivery_bf16.*.log`。最终完整下载验证：`artifacts/final_remote_review.json`。

GPU Python：`F:/PyLibs/cv_cw2_lib/Scripts/python.exe`；独立 CPU Python：`F:/PyLibs/cv_cw2_clean_lib/Scripts/python.exe`。原始数据：`F:/Datasets/computer_vision/Trans10K`，实际清单：`data/clean_manifests`。权重缓存由 TORCH_HOME/HF_HOME 指向 `F:/Models/computer_vision/cw2`。

需要重做新实验时使用新 run 目录，保持配置和来源可追溯。不要因旧 PID 或历史日志重新启动已经完成的流水线。15:07 曾为缓解前台卡顿将测试进程降为 BelowNormal；该进程已结束，没有改动其他服务或系统全局优先级。
