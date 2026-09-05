# 研究与实现审计

审计对象：用户提供的 `ICCAD2026_PolySim (1).pdf`，以及原仓库 commit `29c637543fe10b500f005445eca1c6ccff5476d6`。没有读取服务器 checkpoint 或训练日志，因此以下不能用来推断服务器实际运行的是哪个版本，也不能量化论文结果会变化多少。

## 原项目中的可确认差异

| 位置 | 当前代码事实 | 对实验含义的影响 |
| --- | --- | --- |
| `train_pcme_projector.py::pcme_loss_monte_carlo` | 每次从高斯采样、L2 归一化，做双向 cosine InfoNCE，再平均 loss | 不等同于 PCME 的 log of mean matching probability；归一化后采样不再是高斯 |
| 同函数标签 | 仅以 batch 对角线为正样本 | 当同媒体的多个 captions 共存时会误标 false negatives |
| `generate_train_embeddings.py::get_train_split` | 只取 `video_to_captions[vid][0]` | 丢弃了同媒体的其余描述；此结论仅适用于该入口，不能概括仓库所有数据管线 |
| `ProbabilisticProjector.forward` | `logvar.clamp(-5,2)` | 该版本 `sigma>=exp(-2.5)≈0.0821`，数学上不可能输出真正接近 0 的 σ；需要核对服务器脚本、checkpoint 及记录的是 sigma、variance 还是 logvar |
| 方差正则开关 | 可选 variance penalty、target、bounds 或 KL | 不同设置作用不同；不能把 σ 塌缩单独归因于冻结 backbone，也不能用目标方差证明学到语义不确定性 |
| `train_cheb_projector_v2.py` | 可学习 Gaussian projector 和 A，共同优化检索 loss | 改善 Recall 不能独立证明对一个固定 PCME 教师的保真度 |
| `OrderBilinearSimilarity` | 仅将 A 对称化，二次型负值用 clamp 参与“归一化” | 对称不保证半正定；该范数未必是合法的 A 范数 |

原文件中的音频分支会构建但训练数据循环只使用 text/video；新项目以一对模态为明确的训练单位，不对未训练的第三模态报告结果。

## PCME 原始设计的依据与本项目选择

PCME 论文明确包含独立的 mean/uncertainty 分支、局部注意力、概率匹配损失及防塌缩 KL；官方配置还有 backbone 预训练/微调阶段，因此“冻结 backbone + 两个分支”本身不足以判错。

- [PCME 论文，CVPR 2021](https://arxiv.org/abs/2101.05068)
- [官方代码，固定 commit](https://github.com/naver-ai/pcme/tree/4947684954ffa7884680424bf019f14b2ab43ee2)
- [概率损失与正则](https://github.com/naver-ai/pcme/blob/4947684954ffa7884680424bf019f14b2ab43ee2/criterions/probemb.py)
- [图像编码器](https://github.com/naver-ai/pcme/blob/4947684954ffa7884680424bf019f14b2ab43ee2/models/image_encoder.py)
- [不确定性分支](https://github.com/naver-ai/pcme/blob/4947684954ffa7884680424bf019f14b2ab43ee2/models/uncertainty_module.py)
- [采样工具](https://github.com/naver-ai/pcme/blob/4947684954ffa7884680424bf019f14b2ab43ee2/utils/tensor_utils.py)
- [COCO 配置](https://github.com/naver-ai/pcme/blob/4947684954ffa7884680424bf019f14b2ab43ee2/config/coco/pcme_coco.yaml)

有两个不能机械复制的细节：官方采样函数对名为 `logsigma` 的变量使用 `exp(logsigma)` 作为标准差，而 KL 公式却用到了类似 log-variance 的 `1 + logsigma - mu² - exp(logsigma)`。本项目遵循一致的数学定义 `logvar=log(sigma²)`，使用解析 KL，**不是复刻该变量约定的不一致**。官方概率函数 `exp(l)/(exp(l)+exp(-l))` 等于标准 `sigmoid(2l)`；本项目将 factor 显式配置。

默认 balanced NLL、每样本平均 KL、CLIP token 编码器、真实 mu 的解析 KL 与官方求和归约、ResNet/GloVe 架构及采样均值输入 KL 不同。因此官方超参数不能直接视为本项目的复现配置。本项目为独立实现，没有导入或转换官方 checkpoint。

## PolySim 理论需要分别成立的环节

1. 在有限区间上，用 Chebyshev 展开近似每维 Gaussian PDF 是合理的数值方案，但低阶能否近似窄峰取决于 sigma、区间和分辨率。多项式存在不代表任意固定低阶就足够精确。
2. 对角高斯联合 PDF 是各维 PDF 的**乘积**。逐维系数拼接后的加性双线性 score 不是该乘积的自动等价表达。
3. PCME 的 teacher score 是 `E[sigmoid(b-a||z_t-z_m||)]`；它一般不同于高斯密度 overlap、expected squared distance 或 ELK。拟合 PDF 不直接提供 teacher score 误差界。
4. 本项目对固定教师拟合共享阶数矩阵 A。这是受限的 surrogate 类；在 held-out 数据上可能失败，需要通过 score/ranking 指标确认，不能从训练 Recall 改善直接推论成立。
5. 实对称 A 的同侧 `A^(1/2)` 欧氏内积映射需要 A 半正定。新代码采用 `Ct @ A` 与 `Cm` 的非对称点积变换，任意实 A 都成立。它没有复制旧实现的负二次型 clamp 归一化。
6. `K²` 个非线性样本对比较不等同于实际 crossbar 的 `K²` 次物理 pass；tiling、并行阵列、缓存和外设都影响硬件代价。本初版只导出代数 MVM，不报告未经测量的能耗或加速比。

这些是验证路径的修正，不是已得到的论文更正实验结果。是否需要重做哪张表、如何解释差异，应依据运行 agent 产出的完整证据再判断。

## 技术接口来源

- [Transformers CLIP 模型接口](https://huggingface.co/docs/transformers/en/model_doc/clip)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch 可微 all_gather 实现](https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py)

第三方 API 有版本差异；附带离线 tiny CLIP 前向/反向与缓存一致性测试，无需下载权重。
