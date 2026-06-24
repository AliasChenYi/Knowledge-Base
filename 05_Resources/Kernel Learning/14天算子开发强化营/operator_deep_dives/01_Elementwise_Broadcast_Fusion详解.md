# 01：Elementwise / Broadcast / Fusion 详解

## 1. 这类算子是什么

Elementwise 算子逐元素计算。最简单例子：

```text
y[i] = x[i] + 1
z[i] = a[i] + b[i]
y[i] = relu(x[i])
```

它们的共同点是：

```text
每个输出元素通常只依赖一个或少数几个输入元素。
```

这类算子看起来简单，但在真实模型里数量巨大。很多模型端到端慢，不是因为某个 elementwise 算子本身复杂，而是因为：

- 小 kernel 太多。
- 中间 tensor 太多。
- 每个 kernel 都读写 HBM。
- launch overhead 累积。

所以 elementwise 的关键词是：

```text
memory bandwidth, launch overhead, fusion
```

## 2. 常见 elementwise 算子

| 算子 | 公式 |
|---|---|
| add | `y = a + b` |
| mul | `y = a * b` |
| relu | `y = max(x, 0)` |
| sigmoid | `y = 1 / (1 + exp(-x))` |
| tanh | `y = tanh(x)` |
| gelu | `y = 0.5 * x * (1 + erf(x / sqrt(2)))` |
| silu / swish | `y = x * sigmoid(x)` |
| bias add | `y[m,n] = x[m,n] + bias[n]` |
| residual add | `y = x + residual` |

## 3. Broadcast 是什么

Broadcast 允许不同 shape 的 tensor 按规则参与计算。

例子：

```text
x:    [M, N]
bias: [N]
y[m, n] = x[m, n] + bias[n]
```

这里 `bias` 沿 M 维广播。

另一个例子：

```text
x:     [B, H, S, D]
scale: [D]
y[b,h,s,d] = x[b,h,s,d] * scale[d]
```

Broadcast 的难点不是数学，而是：

- 如何计算每个输入的 offset。
- broadcast 维度 stride 是否为 0。
- 是否支持 non-contiguous。
- 是否会破坏连续访问。

## 4. 性能直觉

Elementwise 通常是 memory-bound。

以 FP32 add 为例：

```text
y = a + b
```

每个元素：

- 读 a：4 bytes。
- 读 b：4 bytes。
- 写 y：4 bytes。
- 计算：1 次加法。

算术强度：

```text
1 FLOP / 12 Bytes
```

非常低，所以优化重点不是“加法更快”，而是：

- 读写更连续。
- 一次 kernel 做更多事。
- 减少中间结果。
- 减少 launch。

## 5. Fusion 为什么重要

假设模型里有：

```text
y = gelu(x + bias)
```

如果拆成多个 op：

```text
kernel 1: tmp = x + bias
kernel 2: y = gelu(tmp)
```

中间的 `tmp` 要写回 HBM，再读出来。

融合后：

```text
kernel 1: y = gelu(x + bias)
```

收益：

- 少一次 kernel launch。
- 少一次中间 tensor 写。
- 少一次中间 tensor 读。

这就是 elementwise fusion 的核心。

## 6. 朴素 CUDA 并行映射

一维 flatten：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < numel) {
    y[idx] = f(x[idx]);
}
```

这个写法适合 contiguous tensor。

如果要支持 arbitrary stride，需要把 `idx` 反解成多维 index：

```text
idx -> i0, i1, i2...
offset = i0 * stride0 + i1 * stride1 + ...
```

这样更通用，但更慢、更复杂。

## 7. Triton 并行映射

Triton 通常让一个 program 处理一块元素：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = f(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

这适合快速写 fused elementwise。

## 8. 支持 broadcast 的思路

以 `x[M,N] + bias[N]` 为例：

flatten 下标：

```text
idx = m * N + n
m = idx / N
n = idx % N
```

输入 offset：

```text
x_offset = m * N + n
bias_offset = n
```

如果 bias 是 `[1, N]`，它在 M 维 stride 可以看作 0。

## 9. 数值问题

Elementwise 也可能有数值坑：

### exp / sigmoid

```text
exp(x)
```

如果 x 很大，可能溢出。

sigmoid 稳定实现会对正负分支做处理，避免 `exp(-x)` 溢出。

### gelu

GELU 有 exact 和 approximate：

```text
exact: erf 版本
approx: tanh 多项式近似
```

测试时必须确认 reference 用哪个。

### dtype cast

FP16 输入时，有些函数内部可能用 FP32 计算再 cast 回 FP16。你的实现要和 reference 对齐。

## 10. 优化动作

常见优化：

1. **fusion**：合并多个 elementwise。
2. **vectorized load/store**：例如 `float4`、`half2`。
3. **contiguous fast path**：连续 tensor 走快路径。
4. **broadcast specialization**：常见 broadcast pattern 写专门 kernel。
5. **避免不必要 cast**。
6. **减少中间 tensor allocation**。

## 11. 测试计划

必须测：

| 类型 | 示例 |
|---|---|
| contiguous | `torch.randn(M, N)` |
| broadcast | `x[M,N] + bias[N]` |
| scalar | `x + 1.0` |
| non-contiguous | `x[:, ::2]` |
| dtype | FP32/FP16/BF16 |
| 极值 | 大正数、大负数、0、NaN/Inf |
| shape | 很小、非 2 的幂、大 tensor |

如果暂不支持 non-contiguous，要明确报错或 `.contiguous()` fallback。

## 12. Benchmark 计划

建议 shape：

```text
N = 1K, 1M, 16M, 100M
[M,N] = [1024,1024], [4096,4096]
```

比较：

- PyTorch eager。
- torch.compile。
- Triton fused kernel。
- CUDA custom kernel。

记录：

- latency。
- effective bandwidth。
- 是否包含中间 tensor。

## 13. 常见坑

1. 只测一个 shape。
2. 只测 contiguous。
3. 没处理 broadcast offset。
4. GELU exact/approx reference 不一致。
5. benchmark 没同步。
6. fusion 后寄存器压力变大，反而变慢。
7. 小 tensor 上 launch overhead 主导，优化内存访问没用。

## 14. 推荐资料

- CUDA Programming Guide：thread hierarchy。
- CUDA Best Practices：memory coalescing。
- Triton vector add tutorial。
- PyTorch Performance Tuning Guide：fusion。
- 本资料包：[Triton vector add](../code_labs/triton/vector_add.py)。

