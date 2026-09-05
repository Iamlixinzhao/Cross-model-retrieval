# 本地验证记录

日期：2026-09-05。仅进行软件正确性检查，没有真实数据训练、权重下载或服务器访问。

环境：Python 3.12，torch 2.14.0+cpu，transformers 4.57.6，numpy 2.3.5，Pillow 12.3.0。该环境是本地检查环境，不是服务器 CUDA 环境的安装清单。

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
13 passed, 1 skipped
```

通过范围：

- 概率平均与 log 的顺序，与独立广播公式的值/梯度对照。
- K×K 样本对、极端 logit 数值稳定性、多正样本。
- 解析 Gaussian KL 与 torch.distributions 对照，KL 对方差塌缩的梯度方向。
- 采样均值/方差、采样后没有单位球归一化。
- attention padding mask、mean/variance 局部注意力梯度。
- caption 分组、重复 gallery 与 split 泄漏拒绝、双向全正样本检索。
- Chebyshev 密度重建、坐标 Jacobian、未见尾部检测。
- 不定对称 A、负 gamma 和非零 bias 下的代数 MVM 等价性。
- 无联网的随机 tiny CLIP：online/cache 前向一致性、冻结层无梯度、最后一层有梯度、激活 checkpointing 接口。

同时通过 `compileall` 与训练/蒸馏 CLI help 检查。

跳过范围：`test_distributed.py` 需要显式 `PCME_TEST_DDP=1`。曾尝试执行，Gloo 初始化因本地接口操作 `Operation not permitted` 失败；指定 loopback 后仍失败。失败发生在通信初始化，尚未进入梯度比较。测试保留，不修改断言来隐藏失败。服务器需执行：

```bash
PCME_TEST_DDP=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_distributed.py
```

未验证：CUDA/NCCL、真实 CLIP 权重/数据训练收敛、完整训练 CLI 的 checkpoint 续训、多卡吞吐/显存、真实数据 teacher/surrogate 精度。代码中包含这些运行入口，具体实验交由服务器 agent 按交接文档执行。
