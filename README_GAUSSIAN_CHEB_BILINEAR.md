# Gaussian-Chebyshev Bilinear Retrieval

这份文档只解释当前保留下来的方法：

- `train_cheb_projector_v2.py`
- `measure_cheb_v2.py`

也就是：

1. 先把 ImageBind embedding 映射成一个对角 Gaussian
2. 再把每一维 Gaussian 的 PDF 用 Chebyshev 系数表示
3. 最后不用普通点积，而是在系数空间里学习一个 **双线性 kernel**

本文重点解释：

- 训练时到底在学什么
- 相似度是怎么定义的
- 为什么这个方法比之前的 `coeff dot` 更合理

---

## 1. 方法总览

对于一条文本 embedding `x_t` 和一条视频 embedding `x_v`，当前方法的流程是：

1. `x_t` 经过 `text_proj` 得到 `(mu_t, logvar_t)`
2. `x_v` 经过 `video_proj` 得到 `(mu_v, logvar_v)`
3. 对每个维度的 1D Gaussian PDF 做 Chebyshev 展开，得到系数张量
4. 在系数空间里用一个可学习的双线性 kernel 算相似度
5. 用 multi-positive NCE 训练这个相似度

记号上可写成：

$$
x_t \rightarrow (\mu_t, \log \sigma_t^2) \rightarrow C_t
$$
$$
x_v \rightarrow (\mu_v, \log \sigma_v^2) \rightarrow C_v
$$
$$
s(t,v) = \text{BilinearKernel}(C_t, C_v; \mu_t, \mu_v)
$$

其中：

- `x_t, x_v \in R^D`
- `mu_t, mu_v \in R^D`
- `logvar_t, logvar_v \in R^D`
- `C_t, C_v \in R^{D \times K}`
- `K = cheb_order + 1`

---

## 2. Gaussian Projector

### 2.1 输入

输入是 ImageBind embedding：

$$
x \in \mathbb{R}^D
$$

训练时先做 L2 归一化：

$$
\tilde{x} = \frac{x}{\|x\|_2}
$$

### 2.2 两个头：`mu_net` 和 `logvar_net`

每个模态各自有一个 projector：

- 文本：`text_proj`
- 视频：`video_proj`

每个 projector 都有两条 MLP 分支：

$$
f_\mu(\tilde{x}), \quad f_{\log\sigma^2}(\tilde{x})
$$

对应代码里的：

- `mu_net`
- `logvar_net`

### 2.3 `mu` 的定义

先做残差：

$$
\mu' = \tilde{x} + f_\mu(\tilde{x})
$$

如果开启 `--use_ln`，会再经过 LayerNorm。

当前代码支持两种 `mu` 约束方式，但推荐并常用的是：

$$
\mu = \frac{\mu'}{\|\mu'\|_2}
$$

也就是 `--mu_on_sphere`，让 `mu` 落在单位球面上，和 PCME 的做法一致。

如果不用 `mu_on_sphere`，则会用：

$$
\mu = \tanh(\mu')
$$

把 `mu` 限制到 `[-1,1]`。

### 2.4 `logvar` 的定义

另一条分支输出对角 Gaussian 的 log-variance：

$$
\log \sigma^2 = \operatorname{clip}(f_{\log\sigma^2}(\tilde{x}), -5, 2)
$$

于是：

$$
\sigma = \exp\left(\frac{1}{2}\log \sigma^2\right)
$$

每个维度都有自己的 `sigma_d`。

所以 projector 的输出就是一个对角 Gaussian：

$$
q(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))
$$

---

## 3. Gaussian 到 Chebyshev 系数

### 3.1 每个维度是一维 Gaussian

对任意一个样本、任意一个维度 `d`，我们取：

$$
\mu_d,\sigma_d
$$

定义这个维度上的 1D Gaussian PDF：

$$
g_d(u) =
\frac{1}{\sigma_d \sqrt{2\pi}}
\exp\left(
-\frac{(u-\mu_d)^2}{2\sigma_d^2}
\right)
$$

这里的 `u` 不是 embedding 维度索引，而是函数自变量。  
我们在固定区间

$$
[u_{\min}, u_{\max}]
$$

上展开这个函数。

---

## 4. Chebyshev 展开

### 4.1 标准区间

Chebyshev 多项式定义在 `[-1,1]` 上，所以先把区间 `[u_min, u_max]` 映射到 `[-1,1]`。

我们使用第一类 Chebyshev 节点：

$$
\theta_m = \frac{\pi (m+1/2)}{M}, \quad m=0,\dots,M-1
$$

$$
x_m = \cos(\theta_m)
$$

再映射回 `u` 空间：

$$
u_m = \frac{u_{\max}-u_{\min}}{2} x_m + \frac{u_{\max}+u_{\min}}{2}
$$

这里：

- `M = num_nodes`

### 4.2 基函数

当前实现里，Chebyshev 基函数主要用 **递推定义** 来构造：

$$
T_0(x)=1,\qquad T_1(x)=x
$$

$$
T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)
$$

这和解析定义

$$
T_k(x) = \cos(k \arccos x)
$$

是等价的，但代码里使用递推形式更直接，也更符合“只靠乘法与加法逐阶生成”的思路。

其中：

- `K = cheb_order + 1`

### 4.3 离散投影得到系数

我们在节点上采样 Gaussian：

$$
g_d(u_m)
$$

然后用离散投影近似系数：

$$
c_{d,k} \approx \frac{2}{M}\sum_{m=0}^{M-1} g_d(u_m)T_k(x_m)
$$

这里的 `T_k(x_m)` 在代码中不是通过 `cos(k\theta_m)` 直接计算，而是先在节点 `x_m` 上用递推式

$$
T_0=1,\quad T_1=x,\quad T_{n+1}=2xT_n-T_{n-1}
$$

逐阶生成一张 basis table，再与采样到的 Gaussian PDF 做离散投影。

对 `k=0` 的项，代码里额外乘了 `0.5`，对应经典 Chebyshev 展开的常数项约定：

$$
c_{d,0} \leftarrow \frac{1}{2}c_{d,0}
$$

### 4.4 阶系数门控

模型对不同阶再乘一个可学习门控：

$$
\tilde{c}_{d,k} = g_k \, c_{d,k}
$$

其中：

$$
g_k \in \mathbb{R}
$$

由 `order_gates` 学习。

于是每个样本最终得到：

$$
C \in \mathbb{R}^{D \times K}
$$

第 `d` 行是该维 Gaussian 的 Chebyshev 系数。

---

## 5. 为什么不能直接 `coeff dot`

之前失败的做法是：

$$
\phi = \operatorname{vec}(C)
$$
$$
s(x_i,x_j) = \phi_i^\top \phi_j
$$

这相当于把 Chebyshev 系数当普通 embedding。

问题在于：

1. Chebyshev 系数是**函数表示**
2. 不是天然的 retrieval embedding
3. 裸点积假设不同阶之间彼此独立
4. 也没有让模型学习“哪一阶和哪一阶应该交互”

所以后来改成了 **双线性 kernel**。

---

## 6. 双线性 kernel

这是当前方法最关键的一步。

### 6.1 基本形式

记两个样本的系数张量分别为：

$$
C_i, C_j \in \mathbb{R}^{D \times K}
$$

其中第 `d` 维的系数向量记作：

$$
c_i^{(d)} \in \mathbb{R}^K,\quad c_j^{(d)} \in \mathbb{R}^K
$$

定义一个可学习的阶间交互矩阵：

$$
A \in \mathbb{R}^{K \times K}
$$

那么双线性相似度定义为：

$$
s_{\text{coeff}}(i,j)
=
\sum_{d=1}^{D}
\left(c_i^{(d)}\right)^\top
A
c_j^{(d)}
$$

这比普通点积更一般：

- 如果 `A = I`，就退化成每一阶独立点积
- 如果 `A` 可学习，就允许：
  - `T0` 和 `T1` 交互
  - `T1` 和 `T3` 交互
  - `T2` 和 `T3` 交互
  - 等等

### 6.2 对称化

代码里实际上使用：

$$
A_{\text{eff}} = \frac{A + A^\top}{2}
$$

这样 kernel 更稳定，也更像一个对称的相似度形式。

### 6.3 归一化

如果开启 kernel normalization，则进一步做类似 cosine 的归一化：

$$
\hat{s}_{\text{coeff}}(i,j)
=
\frac{s_{\text{coeff}}(i,j)}
\sqrt{s_{\text{coeff}}(i,i)}\sqrt{s_{\text{coeff}}(j,j)}}
$$

更准确地说，是对 `A` 诱导的二次型做归一化。

这一步的作用是：

- 防止某些样本因为系数整体幅值大而主导相似度
- 让相似度更接近“角度式比较”

### 6.4 `mu` 残差项

当前方法还可以加一个 `mu` 的残差相似度：

$$
s_\mu(i,j) =
\left\langle
\frac{\mu_i}{\|\mu_i\|},
\frac{\mu_j}{\|\mu_j\|}
\right\rangle
$$

最终相似度为：

$$
s(i,j)=
\hat{s}_{\text{coeff}}(i,j)
\;+\;
\lambda s_\mu(i,j)
$$

其中：

$$
\lambda
$$

是可学习参数 `mu_weight`。

这一步的直觉是：

- Chebyshev 系数描述 Gaussian 的函数形状
- `mu` 仍然保留了 PCME / ImageBind 的主语义方向
- 两者相加，可以让函数表示和语义主干同时发挥作用

---

## 7. 训练目标

### 7.1 相似度矩阵

对于一个 batch，得到：

$$
S \in \mathbb{R}^{B \times B}
$$

其中：

$$
S_{ij} = s(x^t_i, x^v_j)
$$

### 7.2 多正样本 NCE

由于一个视频可能对应多个 caption，所以不是单一对角线正样本，而是用 `vid_ids` 构造正样本掩码：

$$
M_{ij} =
\mathbf{1}[\text{vid\_id}_i = \text{vid\_id}_j]
$$

先做温度缩放：

$$
\tilde{S}_{ij} = \frac{S_{ij}}{\tau}
$$

对每一行的 text-to-video 损失：

$$
\mathcal{L}_{t2v}
=
-\frac{1}{B}
\sum_i
\left[
\log \sum_{j: M_{ij}=1} \exp(\tilde{S}_{ij})
-
\log \sum_j \exp(\tilde{S}_{ij})
\right]
$$

video-to-text 同理：

$$
\mathcal{L}_{v2t}
=
\mathcal{L}_{t2v}(S^\top)
$$

最终：

$$
\mathcal{L}_{\text{nce}}
=
\alpha \mathcal{L}_{t2v}
\;+\;
\beta \mathcal{L}_{v2t}
$$

对应代码里的：

- `t2v_weight`
- `v2t_weight`

---

## 8. 额外正则项

训练除了 NCE，还会加两个辅助项。

### 8.1 `mu` 蒸馏项

让 projector 输出的 `mu` 不要离原始 ImageBind embedding 太远：

$$
\mathcal{L}_{\text{distill}}
=
\left\|
\frac{\mu}{\|\mu\|}
-
\frac{x}{\|x\|}
\right\|_2^2
$$

文本和视频都会算，再求和。

### 8.2 方差正则项

把 `sigma` 拉向目标值 `target_sigma`：

$$
\mathcal{L}_{\text{var}}
=
\left\|
\exp\left(\frac{1}{2}\log \sigma^2\right)
-
\sigma_{\text{target}}
\right\|_2^2
$$

### 8.3 总损失

最终训练目标：

$$
\mathcal{L}
=
\mathcal{L}_{\text{nce}}
-
\text{(这里不是减号)}
$$

更准确写法：

$$
\mathcal{L}
=
\mathcal{L}_{\text{nce}}
\;+\;
\lambda_d \mathcal{L}_{\text{distill}}
\;+\;
\lambda_v \mathcal{L}_{\text{var}}
$$

其中：

- `distill_weight = λ_d`
- `var_reg_weight = λ_v`

---

## 9. 训练时学到的参数

当前方法训练的参数包括：

1. 文本 projector 参数
2. 视频 projector 参数
3. `order_gates`
4. 双线性 kernel 矩阵 `A`
5. 可选的 `mu_weight`

其中 `A` 是最关键的新东西，因为它决定：

- 哪些 Chebyshev 阶是重要的
- 哪些阶之间需要交互

---

## 10. 推理 / 评测过程

评测时不再做 Monte Carlo。

对于测试集：

1. 文本 embedding 过 `text_proj` 得到 `(mu_t, logvar_t)`
2. 视频 embedding 过 `video_proj` 得到 `(mu_v, logvar_v)`
3. 两边都变成系数张量：
   $$
   C_t,\; C_v
   $$
4. 用训练好的双线性 kernel 直接计算：
   $$
   S_{ij} = \sum_d (c_i^{(d)})^\top A c_j^{(d)} + \lambda \langle \mu_i,\mu_j\rangle
   $$
5. 对相似度矩阵做排序，得到 R@1 / R@5 / R@10

所以这套方法的推理路径是：

- **确定性**
- **不需要 Monte Carlo**
- **主要由矩阵乘法构成**

---

## 11. 为什么这版比旧版好

之前失败的方法主要有两个问题：

### 11.1 裸 `coeff dot`

$$
\operatorname{vec}(C_i)^\top \operatorname{vec}(C_j)
$$

问题：

- 没有阶间交互
- 过于僵硬
- 不能表达“第 1 阶和第 3 阶组合起来才重要”

### 11.2 Gaussian overlap

用 PDF overlap 直接当 retrieval similarity。

问题：

- 它比较的是概率密度重叠
- 不一定对应语义相似度
- 在高维下很容易被方差项主导

### 11.3 双线性 kernel 的改进

双线性 kernel 的好处是：

1. 仍然保留 Chebyshev 系数表示
2. 但不强迫它们用“普通点积”比较
3. 让模型自己学习“什么样的系数组合对应更好的检索相似度”

所以它本质上是：

**保留 Gaussian -> Chebyshev 的表示能力，同时把 similarity 学出来。**

---

## 12. 数学上的整体视角

可以把整个方法看成两部分：

### 第一部分：表示

$$
x \mapsto (\mu,\sigma) \mapsto C
$$

这一步做的是：

- 把 embedding 映射成一个对角 Gaussian
- 再把 Gaussian 的每个维度编码成 Chebyshev 系数

### 第二部分：核

$$
(C_i, C_j) \mapsto s(i,j)
$$

这一步做的是：

- 在系数空间上定义一个适合 retrieval 的可学习 kernel

最终方法不是：

- “把 Gaussian 直接当 embedding”

而是：

- “把 Gaussian 编码成 Chebyshev 系数，再在这个函数表示空间上学相似度”

---

## 13. 当前代码入口

### 训练

```bash
python train_cheb_projector_v2.py \
  --emb_dir /mnt/data/pes/ImageBind/msrvtt_train_embeddings \
  --save_dir ./sweep_runs/run_gaussian_cheb_bilinear \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-4 \
  --temperature 0.07 \
  --loss_mode asymmetric \
  --v2t_weight 2.5 \
  --t2v_weight 1.0 \
  --mu_on_sphere \
  --cheb_order 3 \
  --num_nodes 16 \
  --kernel_init identity \
  --kernel_use_mu_residual \
  --init_mu_weight 1.0 \
  --infer_vid_ids \
  --caps_per_video 20 \
  --save_name best_gaussian_cheb_bilinear.pth
```

### 评测

```bash
python measure_cheb_v2.py \
  --emb_dir /mnt/data/pes/ImageBind/msrvtt_results \
  --cheb_ckpt ./sweep_runs/run_gaussian_cheb_bilinear/best_gaussian_cheb_bilinear.pth \
  --save ./sweep_runs/run_gaussian_cheb_bilinear/metrics.json
```

---

## 14. 一句话总结

当前保留下来的方法可以总结成：

$$
\text{ImageBind embedding}
\rightarrow
\text{Gaussian }(\mu,\sigma)
\rightarrow
\text{Chebyshev coefficients}
\rightarrow
\text{learned bilinear similarity}
$$

它的关键思想不是“只用 Chebyshev 做表示”，而是：

**用 Chebyshev 表示 Gaussian，用双线性 kernel 学习这类表示最适合 retrieval 的相似度。**

