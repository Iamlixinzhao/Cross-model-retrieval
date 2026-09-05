# PCME 重建与 PolySim 独立验证

这是原 ICCAD 项目下的一个**独立、可安装的新项目**。目标是先建立可审计的概率检索模型，再冻结这个模型，检验 Chebyshev 表示是否能替代其 Monte Carlo 匹配。原仓库代码和历史结果保持原样。

**实现状态：代码及本地数学/梯度测试；尚无真实数据训练结果。** 配置是起点，不是验证过的最优超参数，也不承诺 PCME 或 PolySim 优于 CLIP。具体来源和原项目问题见 [AUDIT.md](docs/AUDIT.md)；服务器执行与判断标准见 [AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md)；已测/未测范围见 [TESTING.md](docs/TESTING.md)。

## 设计与边界

| 路径 | 输入与训练内容 | 用途 |
| --- | --- | --- |
| `quick_cache`，local_tokens | 缓存的 CLIP patch/token；训练独立 μ、logvar 注意力分支 | 推荐第一步；backbone 只运行一次 |
| `quick_cache`，pooled_ablation | 现有 ImageBind/CLIP/CLAP 最终向量；训练概率分支 | 最省资源的损失排查；无法恢复局部语义信息 |
| `quick_clip` | 原始图像和全部 captions；独立概率分支 + CLIP 两个 tower 最后 1 层微调 | 验证冻结 backbone 的限制 |
| `multigpu_clip` | 同一训练器，DDP 全局可微负样本，最后 2 层微调 | 小规模实验通过后扩展 |
| `distill` | 固定 PCME 导出的 μ/logvar；仅学习阶数交互矩阵 A 和 bias，可选 μ residual | 独立检验 PolySim，不让教师为多项式重新改变 σ |

这是一套 **PCME-inspired 实现**，不是 CVPR 2021 ResNet152/GloVe/GRU 的逐行复刻，也不是 PCME++。第一版在线 backbone 支持 CLIP 图文；缓存接口可接已有视频/音频特征。未实现原始视频/音频 backbone 微调、CrossSim、量化和真实硬件时延测量。

核心约定：

- 全部模块使用 `logvar = log(sigma²)`，采样 `mu + exp(0.5*logvar)*epsilon`。
- μ 做 L2 归一化；σ 分支不做 LayerNorm、L2、sigmoid；高斯采样之后**不再归一化**。
- 匹配概率为 `p = mean_{k,l} sigmoid(b - a*||z_t[k]-z_m[l]||)`，a 始终为正。先平均概率再取 log，不能替换为平均 InfoNCE 或平均 BCE。
- KL 使用到 N(0,I) 的解析 KL：**维度求和，样本求平均**。默认 `beta=1e-4`，没有“把 σ 拉到 0.3”之类目标。数值 guard 是 `logvar∈[-30,10]`，会报告触边比例。
- 每个 batch 按唯一 media ID 抽样，每个 media 默认随机取 2 条 caption；所有同 ID 配对都是正样本。验证使用全部 captions、去重后的 media gallery。
- 默认正/负 NLL 分别平均后各占 0.5，以减轻快速实验和 DDP batch 改变带来的类比例变化。`pair_reduction=all_pairs` 提供另一种归约；两者不是相同实验。
- `sigmoid_factor=2` 可使用官方代码的双 logit 约定；默认 1 对应论文通常的 sigmoid 写法。不要直接搬用官方 a/b 数值及 beta 并假定损失缩放一致。
- 可选 uniformity 默认关闭；打开后在每 rank 的本地样本上计算，改变 world size 会改变这个辅助项的采样范围。

## 1. 安装

在服务器克隆 `pcme-rebuild` 分支，进入这个子目录：

```bash
git clone --branch pcme-rebuild https://github.com/Iamlixinzhao/Cross-model-retrieval.git
cd Cross-model-retrieval/pcme_rebuild
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先根据服务器驱动/GPU安装合适的 CUDA PyTorch，再安装本项目。
python -m pip install -e '.[clip,test]'
python -m pytest -q
```

不需要原项目的 ImageBind 绝对路径。纯缓存路径只需 `pip install -e '.[test]'`。建议 Python 3.10–3.12；依赖范围见 `pyproject.toml`。CLIP 权重首次下载需要服务器联网，也可把 `clip_name` 改成本地 Hugging Face 模型目录。Blackwell 的 PyTorch/CUDA wheel 请由运行 agent 根据驱动选择，勿照搬本地 CPU 环境。

## 2. 推荐：小规模图文数据 + 局部特征缓存

使用服务器已有 Flickr30k 或 COCO 图像及 Karpathy 格式标注。数据不会自动下载；不混用 1k/5k benchmark 划分。也可自行提供 JSONL，每行如下，image 建议绝对路径：

```json
{"id":"flickr30k:123","image":"/datasets/flickr30k/123.jpg","caption":"A dog runs in the grass.","split":"train"}
```

同图不同 caption 保留为多行；同 ID 必须在同一个 split；路径别名检查会拒绝同文件跨 split。ID 必须是跨 split 一致的原始 ID，不能分别从 0 重新编号。工具检查路径和 ID，不能发现不同文件名下内容相同的图片，运行方须使用可信划分。

```bash
python -m pcme_rebuild.prepare karpathy \
  --annotations /datasets/flickr30k/dataset_flickr30k.json \
  --images-root /datasets/flickr30k/images --namespace flickr30k \
  --output data/full.jsonl

python -m pcme_rebuild.prepare subset --manifest data/full.jsonl \
  --train 1000 --val 200 --test 200 --seed 17 --output data/quick.jsonl

for split in train val test; do
  python -m pcme_rebuild.prepare cache-clip --manifest data/quick.jsonl \
    --split "$split" --batch-size 8 --device cuda --output "data/quick/$split.pt"
done

python -m pcme_rebuild.preflight --config configs/quick_cache.json
```

COCO `restval` 默认不加入训练；需要时显式添加 `--include-restval`。subset 按 **media ID** 抽样并保留全部 captions。1000/200/200 是资源受限的开发划分，不是标准 benchmark 结果。所有配置路径相对于当前工作目录。

## 3. 先运行三个可解释的对照

```bash
python -m pcme_rebuild.train --config configs/quick_cache.json \
  --output runs/deterministic --set 'objective="deterministic"'

python -m pcme_rebuild.train --config configs/quick_cache.json \
  --output runs/pcme_no_kl --set kl_beta=0

python -m pcme_rebuild.train --config configs/quick_cache.json \
  --output runs/pcme
```

三个实验使用同一划分、backbone、维度和 seed。日志同时记录 KL、匹配损失、sigma 分位数、噪声半径、数值 guard 触边和验证 Recall。另有 `base_cosine` 作为未经训练的原 backbone 基线。`best.pt` 按固定 MC seed 的验证双向平均 R@1 选择；`last.pt` 包含优化器、调度器和逐 rank RNG，用于同配置同 world size 的 epoch 边界续训：

```bash
python -m pcme_rebuild.train --config configs/quick_cache.json \
  --output runs/pcme --resume runs/pcme/last.pt
```

续训不是增训新的总 epoch 数；若要改变阶段/epoch/world size，使用 `--init-from` 新建实验。不要将旧项目 projector checkpoint 直接载入本项目，参数语义和架构不同。

## 4. 检查 σ 是否有用，再决定是否扩大训练

```bash
python -m pcme_rebuild.evaluate --checkpoint runs/pcme/best.pt --split val \
  --samples 16 --seeds 123 124 125 --pair-chunk 128 \
  --output runs/pcme/diagnostics_val.json --export runs/pcme/val_gaussians.pt

# 训练集只导出教师，避免不必要的全量两两 MC 检索。
python -m pcme_rebuild.evaluate --checkpoint runs/pcme/best.pt --split train \
  --export runs/pcme/train_gaussians.pt --export-only
```

对 deterministic/no-KL checkpoint 运行同样的验证命令，输出到各自目录。`sigma_zero`、`sigma_constant`（保留每维平均方差）、`sigma_shuffled`（打乱样本与方差对应关系）都会报告 Recall 和分数变化。干预与原模型使用相同 MC 随机数，减少比较噪声。

**σ 非零不等于学到了不确定性。** 看干预是否改变匹配/排序、选择性检索是否随拒绝高不确定样本而改善，并与重复 MC 的差异比较。正式结论需要多训练 seed 和不完整语义标注的局限说明。ID 不同的 caption/media 可能在语义上仍匹配；这里不自动产生未经验证的 pseudo positives。

若需要 beta 搜索，先比较 `0, 1e-5, 1e-4, 1e-3`；使用验证集决定，不访问 test。不要单看平均 σ，128 维下每维 σ=.05 的整体噪声半径约 .566。

## 5. 冻结 PCME，检验 PolySim

```bash
python -m pcme_rebuild.distill fit \
  --train runs/pcme/train_gaussians.pt --val runs/pcme/val_gaussians.pt \
  --degree 5 --nodes 128 --samples 16 --output runs/polysim_d5

python -m pcme_rebuild.distill evaluate \
  --checkpoint runs/polysim_d5/best.pt --embeddings runs/pcme/val_gaussians.pt \
  --samples 32 --seeds 123 124 125 --output runs/polysim_d5/val_report.json \
  --export-mvm runs/polysim_d5/mvm_val.pt
```

对 degree=3/5/9/15 用各自独立目录比较。degree=5 包含 T0…T5 共 6 个系数，不是只用 T1/T3/T5。节点数必须大于 degree；窄高斯需要更多节点和更高 degree。默认 **不使用 μ residual**，先检验概率表示本身；再用 `--mu-residual` 作为显式消融。

区间仅从训练导出的 μ/σ 拟合，val/test 共用；包含标准化坐标变换的 Jacobian 和正确的 c0 半权重。报告有：边界外质量、联合盒子内保留质量、负密度占比、网格积分误差、节点数加倍后的系数变化、教师分数 MAE/Spearman、MC 自身重复误差、top-1 排序一致性与双向 Recall。

**逐维 PDF 拟合 ≠ 联合密度拟合 ≠ PCME 匹配概率拟合。** 这里的 order-bilinear 是受限的可学习 surrogate，其成败由 held-out 分数与排序验证决定。低阶失败也是有效实验结果。

MVM 使用 `query=[flatten(Ct A), gamma*mu_t, bias]`，`database=[flatten(Cm), mu_m, 1]`，点积精确等于该 surrogate 的 logit。A 可为不定矩阵，不取平方根；排序无需 sigmoid。导出是软件代数等价性，不代表一次物理 crossbar cycle 或已测硬件加速。

## 6. 完成验证后，再做在线微调或多 GPU

从 **local_tokens 缓存**模型 warm start；pooled cache 的 token 维度不同，不能冒充局部特征模型载入：

```bash
python -m pcme_rebuild.train --config configs/quick_clip.json \
  --init-from runs/pcme/best.pt --output runs/pcme_clip

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m pcme_rebuild.train --config configs/multigpu_clip.json \
  --init-from runs/pcme_clip/best.pt --output runs/pcme_full
```

batch_size 是每 GPU 的 media 数，captions 默认乘 2；全局候选会跨 rank 汇集且保留梯度。两方向都是本 rank anchors 对全局 candidates，DDP 平均梯度得到全局目标。训练丢弃不足一个全局 batch 的尾部；验证不丢样本。小 batch 不等价于通过梯度累积获得更多负样本；第一版不提供梯度累积。

激活 checkpointing 只在在线微调模式使用；概率计算强制 float32，encoder 可用 bf16。显存不足先降低 batch_size/pair_chunk；降低 train_samples 会改变 MC 方差，须记录。DDP 可扩展编码器训练，但全局配对成本随总 batch 增长；不承诺线性加速。

先固定方法/超参数，再导出和评估 test。扩展到不同 PCME checkpoint 后，需要重新导出 train/val Gaussian、重新拟合 PolySim，不能继续复用旧教师的 A。

## 7. 复用旧 ImageBind / 视频 / 音频向量

```bash
python -m pcme_rebuild.prepare import-legacy \
  --text /data/train/emb_text.pt --media /data/train/emb_video.pt \
  --text-ids /data/train/text_ids.json --media-ids /data/train/video_ids.json \
  --split train --output data/quick/train.pt
```

两个 ID 文件都是字符串 JSON 数组，逐行对应各自特征，例如 `["msrvtt:video12","msrvtt:video12","msrvtt:video19"]`。重复 media ID 的特征必须相同才会去重；文本全部保留。val/test 单独导入。禁止凭行号猜 ID；如果旧缓存只留了一条 caption，新代码无法补回不存在的描述，需要重新提取完整数据。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `data.py`, `prepare.py`, `preflight.py` | 数据格式、split 防泄漏、ID 分组、缓存和服务器预检 |
| `model.py`, `probability.py` | 局部概率编码器、概率匹配、解析 KL |
| `train.py` | 单卡/DDP、验证选模、epoch 续训 |
| `evaluate.py` | 双向检索、σ 干预、Gaussian 导出 |
| `polynomial.py`, `distill.py` | 密度拟合、冻结教师蒸馏、MVM 导出 |
| `tests/` | 公式、数值、ID、局部注意力、CLIP 接口、跨进程梯度等价性 |

大数据注意：token cache 是 CPU 内存映射的 `.pt`，缓存生成阶段仍在 CPU 拼接张量；不是无限容量流式 ETL。完整评分矩阵保留在 CPU，默认最多 10M pairs，可显式调高 `--max-pairs`；正式 COCO 5k×25k 请先规划 RAM 与 K² 计算量。完整图像数据训练可使用在线 CLIP 避免大型 token 缓存。
