# Toy Example: PCME Gaussian -> Chebyshev -> Bilinear Similarity

这份文档给一个**非常具体、可以直接放到 PPT** 的 toy example，演示当前方法是怎么从：

1. `mu` 和 `sigma`
2. 变成 Chebyshev polynomial coefficients
3. 再变成最终 similarity

为了简单，我们只假设：

- embedding 维度 `D = 3`
- Chebyshev 阶数 `order = 2`
  - 所以每一维有 `K = 3` 个系数：`T0, T1, T2`
- 节点数 `num_nodes = 4`
- 区间 `[u_min, u_max] = [-3, 3]`

说明：

- 真实训练时常用更大的维度和更多节点
- 这里的数字是为了**演示流程**，不是为了追求最优效果

---

## 1. 先给一组 text / video 的 Gaussian embedding

我们假设：

### Text

```text
mu_t    = [ 0.2, -0.5,  0.8]
sigma_t = [ 0.3,  0.6,  0.4]
```

### Positive Video

```text
mu_v_pos    = [ 0.1, -0.4,  0.7]
sigma_v_pos = [ 0.25, 0.5,  0.35]
```

### Negative Video

```text
mu_v_neg    = [-0.9,  0.6, -0.2]
sigma_v_neg = [ 0.7,  0.3,  0.8]
```

直觉上：

- `video_pos` 和 `text` 更接近
- `video_neg` 和 `text` 更不接近

---

## 2. Chebyshev nodes

我们在 `[-1, 1]` 上取 4 个第一类 Chebyshev 节点：

```text
x_nodes = [ 0.923880,  0.382683, -0.382683, -0.923880 ]
```

再映射到 `[-3, 3]`：

```text
u_nodes = [ 2.771639,  1.148050, -1.148050, -2.771639 ]
```

---

## 3. 用 recurrence definition 生成 Chebyshev basis

当前代码里采用的是递推定义：

```text
T0(x) = 1
T1(x) = x
T2(x) = 2*x*T1(x) - T0(x)
```

所以在这 4 个节点上：

```text
T0 = [ 1.000000,  1.000000,  1.000000,  1.000000 ]
T1 = [ 0.923880,  0.382683, -0.382683, -0.923880 ]
T2 = [ 0.707107, -0.707107, -0.707107,  0.707107 ]
```

这一步就是代码里 `basis_table` 的来源。

---

## 4. 对每一维 Gaussian 做 Chebyshev 拟合

对每一个维度 `d`，都有一个 1D Gaussian：

```text
g_d(u) = exp(-(u-mu_d)^2 / (2*sigma_d^2)) / (sigma_d * sqrt(2*pi))
```

然后在 `u_nodes` 上采样，再投影到 `T0/T1/T2` 上，得到每一维的 3 个系数。

---

## 5. Text 的系数长什么样

### Text dim 0: `mu=0.2, sigma=0.3`

在 4 个节点上采样得到：

```text
pdf = [0.000000, 0.009020, 0.000055, 0.000000]
```

投影后得到系数：

```text
c_t[0] = [ 0.002269,  0.001715, -0.003208 ]
```

### Text dim 1: `mu=-0.5, sigma=0.6`

```text
pdf = [0.000000, 0.015292, 0.371056, 0.000513]
c_t[1] = [ 0.096715, -0.068309, -0.136413 ]
```

### Text dim 2: `mu=0.8, sigma=0.4`

```text
pdf = [0.000005, 0.683036, 0.000007, 0.000000]
c_t[2] = [ 0.170762,  0.130694, -0.241490 ]
```

所以 text 的 Chebyshev coefficient tensor 是：

```text
C_t =
[
  [ 0.002269,  0.001715, -0.003208 ],
  [ 0.096715, -0.068309, -0.136413 ],
  [ 0.170762,  0.130694, -0.241490 ]
]
```

形状是：

```text
C_t in R^(3 x 3)
```

也就是：

- 3 个 embedding 维度
- 每个维度 3 个 Chebyshev 系数

---

## 6. Positive video 的系数

### Positive Video dim 0

```text
c_v_pos[0] = [ 0.000062,  0.000045, -0.000088 ]
```

### Positive Video dim 1

```text
c_v_pos[1] = [ 0.066794, -0.048594, -0.094454 ]
```

### Positive Video dim 2

```text
c_v_pos[2] = [ 0.125582,  0.096116, -0.177600 ]
```

所以：

```text
C_v_pos =
[
  [ 0.000062,  0.000045, -0.000088 ],
  [ 0.066794, -0.048594, -0.094454 ],
  [ 0.125582,  0.096116, -0.177600 ]
]
```

---

## 7. Negative video 的系数

```text
C_v_neg =
[
  [ 0.139775, -0.108282, -0.186375 ],
  [ 0.062667,  0.047963, -0.088624 ],
  [ 0.092753, -0.025291, -0.128806 ]
]
```

你会发现：

- positive pair 的各维系数形状和 text 更像
- negative pair 的系数方向和幅值变化更大

---

## 8. 不是直接 dot，而是 bilinear kernel

当前方法最关键的地方就是：

**不是**

```text
sim = vec(C_t) dot vec(C_v)
```

而是：

```text
sim = sum_d  c_t[d]^T A c_v[d]  +  mu_weight * <mu_t_norm, mu_v_norm>
```

这里我们用一个 toy 的 bilinear matrix：

```text
A =
[
  [1.0, 0.2, 0.0],
  [0.2, 1.0, 0.1],
  [0.0, 0.1, 0.8]
]
```

以及：

```text
mu_weight = 0.7
```

---

## 9. Positive pair 的 similarity 是怎么来的

### 9.1 每一维先算一个 bilinear score

```text
dim 0 raw score = c_t[0]^T A c_v_pos[0] = 0.000000
dim 1 raw score = c_t[1]^T A c_v_pos[1] = 0.019543
dim 2 raw score = c_t[2]^T A c_v_pos[2] = 0.070241
```

把三维加起来，得到系数分支的 raw score：

```text
coeff_raw_sum = 0.089784
```

### 9.2 再做 kernel normalization

当前代码会用 `A` 诱导的范数做归一化，所以最终系数分支相似度不是 raw score，而是：

```text
coeff_kernel = 0.999649
```

### 9.3 再加上 mu residual

先把 `mu` 做 L2 normalize：

```text
mu_t_norm     = [ 0.207390, -0.518476,  0.829561 ]
mu_v_pos_norm = [ 0.123091, -0.492366,  0.861640 ]
```

然后：

```text
mu_dot = <mu_t_norm, mu_v_pos_norm> = 0.995591
```

乘上权重：

```text
mu_residual = 0.7 * 0.995591 = 0.696914
```

### 9.4 最终 positive similarity

```text
final_sim_pos = coeff_kernel + mu_residual
              = 0.999649 + 0.696914
              = 1.696563
```

---

## 10. Negative pair 的 similarity 是怎么来的

### 10.1 系数分支

对 negative pair，最终算出来：

```text
coeff_kernel_neg = 0.477797
```

### 10.2 mu residual

```text
mu_dot_neg = -0.603317
mu_residual_neg = 0.7 * (-0.603317) = -0.422322
```

### 10.3 最终 negative similarity

```text
final_sim_neg = coeff_kernel_neg + mu_residual_neg
              = 0.477797 - 0.422322
              = 0.055475
```

---

## 11. 最后对比一下

### Positive pair

```text
final_sim_pos = 1.696563
```

### Negative pair

```text
final_sim_neg = 0.055475
```

所以：

```text
positive similarity  >>  negative similarity
```

这就是这套方法最终想达到的效果：

- 相近的 text/video pair，得到更大的相似度
- 不相近的 pair，得到更小的相似度

---

## 12. PPT 里可以怎么讲

你可以直接用下面这 4 句话做一页总结：

1. **PCME projector** 先把每个 embedding 变成对角 Gaussian：`(mu, sigma)`
2. 对 Gaussian 的每个维度 PDF，在固定节点上做 **Chebyshev polynomial projection**
3. 得到一个 `D x K` 的 coefficient tensor，而不是普通 embedding
4. 最后用 **learned bilinear kernel** 而不是普通 dot product 来算 similarity

---

## 13. 一页图的建议结构

如果你做 PPT，这一页可以排成：

### 左边：输入

```text
mu_t    = [0.2, -0.5, 0.8]
sigma_t = [0.3, 0.6, 0.4]

mu_v    = [0.1, -0.4, 0.7]
sigma_v = [0.25, 0.5, 0.35]
```

### 中间：Chebyshev 展开

```text
each dim Gaussian -> [c0, c1, c2]

C_t =
[
  [ 0.002269,  0.001715, -0.003208 ],
  [ 0.096715, -0.068309, -0.136413 ],
  [ 0.170762,  0.130694, -0.241490 ]
]
```

### 右边：相似度

```text
sim = sum_d c_t[d]^T A c_v[d] + 0.7 * <mu_t, mu_v>

positive pair: 1.696563
negative pair: 0.055475
```

---

## 14. 一句话总结

这个 toy example 想表达的是：

**我们不是直接比较 Gaussian，也不是直接比较 Chebyshev 系数，而是先把 Gaussian 编码成 Chebyshev coefficient tensor，再在这个 tensor 上学习一个更适合 retrieval 的 bilinear similarity。**

