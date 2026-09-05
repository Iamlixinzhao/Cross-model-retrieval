# 给服务器运行 agent 的交接

## 任务边界

代码已写入原仓库 `pcme-rebuild` 分支下的 `pcme_rebuild/`。当前没有真实数据训练结果；不要把本地单元测试或非零 sigma 当成研究假设得到验证。作者希望先低资源快速验证，再决定是否多 GPU 扩展。不要直接启动全量训练或先大范围搜索。

默认采用小规模图文 CLIP + PCME-inspired loss。若服务器只有 MSRVTT/ImageBind 特征，可先用 `import-legacy` 做 pooled ablation，并明确报告其限制。没有原始媒体 ID 就不能安全导入，不要以行号编造 ID。先完成数据与设备预检，再运行。

完整命令在 [README](../README.md)。命令均从 `Cross-model-retrieval/pcme_rebuild` 执行。

## 执行顺序

1. 检查 GPU、驱动、CUDA PyTorch 和实际数据路径；安装项目。保存 Python、torch、transformers、CUDA 版本与 GPU 型号。
2. 执行 `python -m pytest -q`。服务器还必须执行 `PCME_TEST_DDP=1 python -m pytest -q tests/test_distributed.py`；这是两进程 CPU/Gloo 全局概率损失梯度等价测试，不消耗训练数据。若默认网卡不适用，可以指定服务器允许的通信接口。不要把 skipped 当 passed。
3. 准备固定的 1000 train / 200 val / 200 test media 开发划分，每个 media 保留所有 captions。若数据不足，明确报告实际数量。不在不同 split 重新编号 media ID。
4. 运行 `preflight`；确认 captions-per-media 分布、train/val 无重叠、局部 token 是否确实存在。缓存必须来自同一 backbone/预处理。
5. 按 README 完成 deterministic、PCME beta=0、PCME beta=1e-4 三个实验。第一轮用相同 seed，4 个 MC draws、128 维。报告每 epoch 时间、GPU 峰值显存、loss、sigma 分位数与验证 Recall。训练器记录 epoch 时间但不自动测量峰值显存，运行方另行记录。
6. 对三个 checkpoint 做固定 validation gallery 的诊断。若 PCME 明显落后原 backbone 或数值异常，先检查标签、损失尺度、KL；用小 beta sweep，再考虑 token 路径/最后一层微调，避免直接增加 GPU。
7. 冻结选定教师；用 `evaluate --export-only` 导出 train，用 `evaluate --export` 导出 val。两个文件必须来自**同一 checkpoint**。
8. 用 train 拟合 PolySim，val 选 degree/nodes/可选 μ residual。先不启用 μ residual，记录 PDF 质量和教师分数拟合；不得修改教师以“帮助”PolySim。
9. 只有开发结果值得继续时，增加训练 seeds 17/18/19，随后切换在线微调及多 GPU。变更教师就重新导出、重新拟合 A。
10. 固定模型和超参数后才解封 test；禁止用 test 定义 Chebyshev 区间、早停、选择 beta 或 degree。

## 判断标准

| 检查 | 应观察什么 | 不能得出的结论 |
| --- | --- | --- |
| 数值正确性 | loss/梯度有限，sigma 不持续触 guard，MC 与独立公式一致 | 通过单元测试不等于能训练好 |
| 不确定性信息 | 不同输入的 sigma 有差异；zero/constant/shuffled 干预会影响 held-out 分数或排序 | sigma 非零不是充分证据 |
| 检索作用 | PCME 对比 deterministic/raw backbone；看干预影响是否超过 MC 自身波动 | 不能保证 PCME 必定提高 Recall |
| 选择性检索 | 高不确定样本的错误关联；剔除它们后 Recall 是否改善 | 这不是完备的语义校准或 OOD 验证 |
| PDF 近似 | 尾质量、节点加倍变化、真 PDF 网格积分质量误差、负密度占比 | 网格漏掉窄峰时不能引用小 L1；逐维误差不是联合误差 |
| PCME 替代性 | 正/负分别 MAE、balanced MAE、Spearman、top-1 agreement、Recall 差与 MC repeatability | 不能用大量接近 0 的负样本压低全局 MAE 来宣称成功 |
| μ residual | 先无 residual；再显式消融，记录 gamma 和 sigma 干预 | 不能让均值分支替代全部信息还宣称保留不确定性 |
| 扩展性 | 全局 batch 数、DDP 梯度等价、实际时间/显存 | GPU 数增加不自动带来线性加速 |

可预先约定探索性目标，例如 PolySim 双向 Recall@1 距 PCME 各不超过 1–2 个百分点、top-1 agreement 达到 95%，但 200-item 验证集的统计粒度有限，这些不是数学保证或本项目已通过的 gate。优先对照多 MC seed 的重复误差，并在更大验证集和多训练 seed 下复核。

不要设置 `sigma>某常数` 作为成功标准；不同维度下总噪声 `sqrt(sum(sigma²))` 才能和单位范数 μ 的尺度一起解释。`uncertainty_error_spearman=null` 表示错误/不确定性为常量或不足以定义相关系数，不是 0。

## 应交付的运行结果

- 每个 run 的 config.json、metadata.json、metrics.jsonl、best/last checkpoint。
- 三个教师实验的 validation 诊断，含所有 MC seeds 和 sigma 干预。
- 不同 degree 的 train/val 密度报告与 held-out surrogate 对照；若失败，保留失败结果。
- 原始 media/caption 数量、划分来源、随机种子、源文件 hash、资源消耗。
- 一个小表：base cosine / deterministic / PCME no-KL / PCME / PolySim 的双向 R@1、R@5、R@10 与误差范围。训练 loss 不同归约之间不能直接比较数值大小。

不要覆盖原 ICCAD 结果。仅输出新实验目录；在用户审阅真实实验结果前，不把原论文的结果替换成推测数字。

## 已知限制与待测内容

- 本地无服务器 GPU、真实数据与预训练 checkpoint；未执行真实训练、CUDA/NCCL 多卡训练或 epoch 续训实验。
- token cache 生成在内存拼接，不适合无限规模；full COCO 建议在线模型或外部预处理。
- 完整检索矩阵保留在 CPU；默认最多 10M pairs。`--max-pairs` 增大前先估算 RAM/计算量。
- 原始在线输入只支持图像/文本；视频/音频可接缓存特征，但没有可直接运行的原始视频/音频 backbone 微调。
- 没有语义 pseudo positives、随机图像增广或 text dropout；这是一个可解释的轻量基线，并非复现官方完整增强方案。
- 原始 PCME public implementation 的 `logsigma` 采样和 KL 约定有不一致；本项目采用统一数学约定，不能直接导入旧权重。
- warm start 从 local-token cache → online 时载入概率 head/matching 参数并核对 CLIP 来源；online → online 时同时载入已微调 backbone，保留其权重。优化器/调度器会重置；需要完整续训时使用 `--resume`。online → cache 被拒绝，避免将旧 backbone 特征混用到微调后的 head。
