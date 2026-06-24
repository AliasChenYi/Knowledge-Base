# 03：Softmax / LogSoftmax 详解

## 1. Softmax 算什么

Softmax 把一组实数变成概率分布：

```text
y_i = exp(x_i) / sum_j exp(x_j)
```

输出满足：

```text
y_i >= 0
sum_i y_i = 1
```

常见位置：

- 分类模型最后一层。
- attention 权重。
- beam search / sampling。

## 2. 为什么要稳定 softmax

直接算：

```text
exp(x_i)
```

如果 `x_i` 很大，会溢出。

稳定写法：

```text
m = max_j x_j
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

数学上等价，因为分子分母同时乘了 `exp(-m)`。

## 3. 计算结构

对每一行：

```text
1. row_max = max(x)
2. shifted = x - row_max
3. numerator = exp(shifted)
4. denominator = sum(numerator)
5. y = numerator / denominator
```

所以 softmax 是：

```text
reduction(max) + elementwise(exp) + reduction(sum) + elementwise(div)
```

## 4. 朴素多 kernel 实现的问题

如果每一步一个 kernel：

```text
kernel 1: max
kernel 2: subtract + exp
kernel 3: sum
kernel 4: divide
```

问题：

- 多次 kernel launch。
- shifted/numerator 等中间 tensor 写回 HBM。
- 下一步又从 HBM 读出。

因此 softmax 很适合 fusion。

## 5. Fused Softmax 思路

常见设计：

```text
一个 block / Triton program 处理一行。
```

在一个 kernel 内完成：

- load row。
- max。
- exp。
- sum。
- divide。
- store。

中间值尽量留在寄存器或片上存储。

## 6. Masked Softmax

Attention 里经常有 mask。

做法：

```text
被 mask 的位置在 softmax 前设为 -inf。
```

因为：

```text
exp(-inf) = 0
```

这样它不会贡献概率。

### Causal Mask

自回归模型里，第 i 个 token 不能看未来 token：

```text
mask[j > i] = true
```

实现时要注意：

- mask 在 max 前应用。
- 全部被 mask 的行要小心 NaN。

## 7. LogSoftmax

LogSoftmax：

```text
log_softmax(x_i) = x_i - log(sum_j exp(x_j))
```

稳定写法：

```text
m = max(x)
log_softmax(x_i) = x_i - m - log(sum_j exp(x_j - m))
```

为什么有用：

- 交叉熵中常用。
- 避免先 softmax 再 log 的数值问题。

## 8. Shape 对实现的影响

输入通常：

```text
X[M, N]
```

沿 N 做 softmax。

### N 小

问题：

- 一个 block 处理一行可能线程利用率低。
- launch overhead 明显。

优化：

- 一个 block 处理多行。
- fusion 到前后算子。

### N 中等

一个 block/program 处理一行通常不错。

### N 很大

问题：

- 一行放不进一个 block/program。
- 寄存器压力大。
- 需要分块 softmax。

FlashAttention 的 online softmax 就是更复杂场景下的重要思想。

## 9. 数值和 dtype

建议：

- FP16/BF16 输入时，max/sum 用 FP32。
- exp 可以根据实现路径处理，但最终误差要测。
- denominator 不能为 0。
- 注意 NaN 传播。

测试时检查：

```python
y.sum(dim=-1)
```

应该接近 1。

## 10. Triton 实现关键点

```python
row = tl.load(..., mask=mask, other=-float("inf"))
row = row - tl.max(row, axis=0)
num = tl.exp(row)
den = tl.sum(num, axis=0)
out = num / den
```

关键：

- `other=-inf`。
- `BLOCK_SIZE >= n_cols`。
- `BLOCK_SIZE` 常取 next power of 2。
- store 也要 mask。

## 11. 测试计划

必须测：

- N = 1。
- N = 2, 3。
- N = 127, 128, 129。
- N = 1024, 4096。
- 全 0。
- 大正数。
- 大负数。
- mask。
- causal mask。
- FP16/BF16。

检查：

- 和 PyTorch reference allclose。
- 每行 sum 接近 1。
- 没有 NaN/Inf。

## 12. Benchmark 计划

shape：

```text
M = 1024, 4096, 16384
N = 128, 256, 512, 1024, 2048, 4096, 8192
```

比较：

- PyTorch eager。
- torch.compile。
- Triton fused softmax。
- custom CUDA。

记录：

- latency。
- effective bandwidth。
- 是否有 mask。
- dtype。

## 13. 常见坑

1. 忘记减 max。
2. mask 在 max 后才应用。
3. 越界位置填 0，影响 max。
4. store 忘记 mask。
5. 行太长还用单 block。
6. 对 FP16 误差标准过严。
7. 没检查每行概率和。

## 14. 推荐资料

- Triton Fused Softmax Tutorial。
- PyTorch Softmax 文档。
- FlashAttention 论文里的 online softmax。
- 本资料包：[Triton fused softmax](../code_labs/triton/fused_softmax.py)。

