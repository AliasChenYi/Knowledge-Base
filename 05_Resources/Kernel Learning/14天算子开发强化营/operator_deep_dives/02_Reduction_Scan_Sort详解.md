# 02：Reduction / Scan / Sort 详解

## 1. Reduction 是什么

Reduction 把很多元素合成一个或少数几个元素。

常见形式：

```text
sum(x)
max(x)
min(x)
mean(x)
variance(x)
argmax(x)
```

深度学习里大量算子依赖 reduction：

- softmax：max 和 sum。
- layernorm：mean 和 variance。
- RMSNorm：mean square。
- batchnorm：batch 统计量。
- attention：softmax。
- topk/sort：比较和选择。

## 2. Reduction 为什么比 elementwise 难

Elementwise：

```text
y[i] = f(x[i])
```

每个输出互相独立。

Reduction：

```text
y = sum_i x[i]
```

多个线程的局部结果要合并，所以会出现：

- 同步。
- shared memory。
- warp-level primitive。
- 原子操作。
- 多阶段 kernel。
- 浮点误差。

## 3. Identity value

不同 reduction 有不同初始值：

| reduction | identity |
|---|---|
| sum | 0 |
| product | 1 |
| max | `-inf` |
| min | `+inf` |
| logical and | true |
| logical or | false |

这是新人很容易错的点。比如 max 初始化成 0，遇到全负数就错。

## 4. Row-wise Reduction

输入：

```text
X[M, N]
```

输出：

```text
Y[M]
```

常见设计：

```text
一个 block / Triton program 处理一行。
```

每个线程处理若干列：

```cpp
for (int col = threadIdx.x; col < N; col += blockDim.x) {
    local += X[row, col];
}
```

然后 block 内归约。

## 5. Block 内树形归约

典型流程：

```text
每个线程先算 local
local 写入 shared memory
stride = block_size / 2
前半线程合并后半线程
stride 不断减半
thread 0 写出结果
```

伪代码：

```cpp
smem[tid] = local;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] += smem[tid + stride];
    }
    __syncthreads();
}
```

教学上清楚，但工业实现会进一步优化：

- warp shuffle。
- unroll。
- 多元素每线程。
- 避免不必要同步。

## 6. Warp-level Reduction

在一个 warp 内，可以用 shuffle 指令交换数据，减少 shared memory 使用。

直觉：

```text
同一个 warp 内线程可以直接互相读寄存器里的值。
```

好处：

- 少 shared memory。
- 少同步。
- 更快。

但你初学先把 shared memory 版本写清楚，再学 shuffle。

## 7. 多 block Reduction

如果一行非常长，一个 block 处理不完，可能需要：

```text
kernel 1: 每个 block 处理一段，输出 partial sums
kernel 2: 对 partial sums 再 reduction
```

或者使用更复杂的 persistent/block-level 策略。

这会引入：

- 中间结果。
- 多次 kernel launch。
- 跨 block 合并。

## 8. Scan 是什么

Scan 又叫 prefix sum。

输入：

```text
x = [1, 2, 3, 4]
```

inclusive scan：

```text
[1, 3, 6, 10]
```

exclusive scan：

```text
[0, 1, 3, 6]
```

Scan 比 reduction 更复杂，因为它要保留每个位置的前缀结果。

应用：

- compaction。
- sparse 操作。
- stream compaction。
- radix sort。
- 某些分段处理。

## 9. Sort / TopK

Sort/TopK 属于更复杂的选择类算子。

常见场景：

- beam search。
- sampling top-k/top-p。
- detection 后处理。
- sparse expert routing。

优化难点：

- 比较和交换多。
- 分支复杂。
- 访存不规则。
- 小 K 和大 K 策略完全不同。

初学阶段建议：

```text
先学 reduction max，再学 topk。
```

## 10. 数值问题

### 浮点求和顺序

并行 reduction 改变加法顺序，所以和 CPU/PyTorch reference 不一定 bitwise 一致。

建议：

- FP32 accumulate。
- 合理 rtol/atol。
- 报告 max_abs/mean_abs。

### variance

方差有不同公式：

```text
E[x^2] - E[x]^2
mean((x - mean)^2)
```

数值稳定性不同。要和 reference 对齐。

## 11. 优化动作

1. 每线程处理多个元素。
2. 使用 shared memory 做 block reduction。
3. warp shuffle 优化 warp 内归约。
4. 减少同步次数。
5. 对小 N 和大 N 分别写策略。
6. FP16 输入、FP32 accumulate。
7. 避免 bank conflict。
8. 对常见 N 做 specialization。

## 12. 测试计划

必须测：

- N = 1, 2, 3。
- N = 31, 32, 33。
- N = 127, 128, 129。
- N = 1024, 4096, 8192。
- 全负数 max。
- 全 0。
- 大小混合值。
- FP16/BF16 误差。

## 13. Benchmark 计划

记录：

| M | N | dtype | baseline | my_impl | speedup | bandwidth |
|---|---:|---|---:|---:|---:|---:|

比较：

- PyTorch `sum/max`。
- Triton row-wise reduction。
- CUDA custom kernel。

## 14. 常见坑

1. max 初始化成 0。
2. 忘记 `__syncthreads()`。
3. shared memory 越界。
4. 非 2 的幂长度处理错。
5. FP16 误差过大。
6. 只测正数随机输入。
7. block size 太小或太大。

## 15. 推荐资料

- 本地 PDF：[NVIDIA Optimizing Parallel Reduction](../papers/02_NVIDIA_Optimizing_Parallel_Reduction.pdf)
- CUDA Best Practices：shared memory。
- Triton tutorials：reduction/softmax/layernorm。
- 本资料包：[CUDA reduction](../code_labs/cuda/reduction.cu)。

