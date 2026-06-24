# 05：GEMM / Batched GEMM / GEMV 详解

## 1. GEMM 是什么

GEMM 是 General Matrix Multiplication：

```text
C = alpha * A * B + beta * C
```

最常见简化：

```text
C[M, N] = A[M, K] @ B[K, N]
```

每个输出元素：

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

GEMM 是深度学习最重要的算子之一。

## 2. GEMM 为什么重要

很多层都能转成 GEMM 或 GEMM-like：

- Linear。
- MLP。
- Attention 的 QK 和 PV。
- 1x1 convolution。
- im2col convolution。
- MoE expert MLP。
- 量化 matmul。

只要 GEMM 慢，模型大概率慢。

## 3. GEMM 的性能特点

GEMM 的算术强度高。

原因：

```text
A 的一个元素会被多个 N 方向输出复用。
B 的一个元素会被多个 M 方向输出复用。
```

如果 tile 设计好，A/B 从 HBM 搬到片上后可以被反复使用。

这和 elementwise 完全不同。

## 4. 朴素 GEMM

伪代码：

```cpp
for m in M:
  for n in N:
    acc = 0
    for k in K:
      acc += A[m,k] * B[k,n]
    C[m,n] = acc
```

问题：

- A/B 反复从 global memory 读。
- 没有显式复用。
- 可能不能使用 Tensor Core。

## 5. Tiled GEMM

核心思想：

```text
一个 block 负责 C 的一个 tile。
把 A tile 和 B tile 搬到 shared memory。
block 内线程重复使用这些 tile。
沿 K 维循环累加。
```

例如：

```text
C tile: 128 x 128
A tile: 128 x 32
B tile: 32 x 128
```

每轮处理 K 的 32 个元素，累加到 C tile。

## 6. 分层 tiling

高性能 GEMM 通常有多层：

### CTA / Threadblock tile

一个 block 负责的 C 区域。

### Warp tile

block 内多个 warp 分工。

### MMA instruction tile

Tensor Core/MMA 指令处理的小矩阵块。

### Thread tile

每个线程负责 accumulator 的一小部分。

## 7. Tensor Core / MMA

Tensor Core 是矩阵乘专用硬件。

你先记住：

```text
想让 GEMM 很快，通常要走 Tensor Core/MMA 路径。
```

关键条件：

- dtype 支持。
- shape/alignment 合适。
- layout 合适。
- 使用 cuBLAS/cuBLASLt/CUTLASS/Triton/WMMA。

## 8. Accumulator dtype

常见：

```text
FP16/BF16 input -> FP32 accumulator
INT8 input -> INT32 accumulator
FP8 input -> FP16/FP32 accumulator，视硬件和库而定
```

原因：

- 累加 K 次，误差会放大。
- 低精度 accumulator 容易损失精度。

## 9. GEMV

GEMV 是矩阵乘向量：

```text
y[N] = W[N, K] @ x[K]
```

它可以看成：

```text
M = 1 的 GEMM
```

但性能特征不同：

- 数据复用少。
- 更偏 memory-bound。
- LLM decode 阶段常见。
- weight-only quantized GEMV 很重要。

不要把大 GEMM 策略直接套到 GEMV。

## 10. Batched GEMM

Batched GEMM：

```text
for b in batch:
  C[b] = A[b] @ B[b]
```

常见场景：

- 多 head attention。
- 小矩阵批量计算。

难点：

- 单个 GEMM 可能很小。
- launch overhead 明显。
- 需要 grouped/batched 策略。

## 11. Grouped GEMM

Grouped GEMM 处理多个不同 shape 的 GEMM。

常见场景：

- MoE：不同 expert 有不同 token 数。
- 动态 batch。

难点：

- 负载不均衡。
- 调度复杂。
- 每个 GEMM shape 不同。

## 12. Epilogue Fusion

GEMM 后常接：

```text
bias
activation
scale
residual add
cast
quantize/dequantize
```

如果拆开：

```text
GEMM 写 C 到 HBM
下一个 kernel 读 C 做 bias/activation
```

如果融合：

```text
accumulator 在写回前直接做后处理
```

收益：

- 少 HBM 读写。
- 少 launch。

## 13. Layout

GEMM layout 很关键：

- row-major。
- column-major。
- transposed A/B。
- Tensor Core 需要特定 alignment。

如果 layout 不合适，可能出现：

- 访存不连续。
- 需要额外 transpose。
- 调库性能差。

## 14. 测试计划

shape：

```text
512 x 512 x 512
1024 x 1024 x 1024
4096 x 4096 x 4096
M=1, N=4096, K=4096
M=128, N=4096, K=11008
batched: B=32, M=N=K=128
```

dtype：

- FP32。
- TF32。
- FP16。
- BF16。
- INT8。

检查：

- max_abs。
- max_rel。
- 是否 NaN/Inf。
- accumulator 语义。

## 15. Benchmark 指标

GEMM 常用 TFLOPS：

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / seconds / 1e12
```

GEMV 则可能更关注 effective bandwidth。

## 16. 常见坑

1. 忽略 layout。
2. 以为手写 GEMM 很容易超过 cuBLAS。
3. 小 GEMM 用大 GEMM 思路。
4. 没用 Tensor Core。
5. accumulator dtype 错。
6. epilogue 拆开导致额外 HBM。
7. 只测正方形大矩阵。

## 17. 推荐资料

- NVIDIA Matrix Multiplication Background。
- CUTLASS Efficient GEMM。
- Triton Matmul Tutorial。
- CuTe GEMM Tutorial。
- 本资料包 Day 11/12。

