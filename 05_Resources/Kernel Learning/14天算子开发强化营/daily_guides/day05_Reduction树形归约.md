# Day 05：Reduction 树形归约

![Reduction 树形归约](../assets/day05_reduction_tree.svg)

## 今天的核心结论

Reduction 是从一堆元素得到一个或少数几个结果：

```text
sum, max, min, mean, variance
```

很多深度学习算子看起来高级，底层其实都有 reduction：

- softmax：每行 max 和 sum。
- layernorm：每行 mean 和 variance。
- rmsnorm：每行 mean square。
- attention：softmax 需要 max/sum。

今天你学 reduction，是为了明天真正写 softmax/norm。

## 今天你要完成什么

最低 2 小时版：

1. 读 [NVIDIA Reduction 中文讲解](../paper_notes/02_NVIDIA_Reduction_中文讲解.md)。
2. 看懂树形归约。
3. 写 row-wise sum 或 row-wise max 的伪代码。

完整版 4-6 小时：

1. 实现 row-wise sum。
2. 实现 row-wise max。
3. 测不同 N。
4. 比较 block size 和每线程多元素处理。

## 必读资料与读法

1. 本地 PDF：[NVIDIA Optimizing Parallel Reduction](../papers/02_NVIDIA_Optimizing_Parallel_Reduction.pdf)
2. 中文讲解：[NVIDIA Reduction 中文讲解](../paper_notes/02_NVIDIA_Reduction_中文讲解.md)
3. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

读法：

- 先读中文讲解。
- PDF 看前几个 kernel 演进。
- 每个版本只问：它修掉了什么性能问题？

## 概念讲解：为什么 reduction 难并行

Elementwise 很好并行：

```text
y[i] = f(x[i])
```

每个输出只依赖一个输入，线程之间不用交流。

Reduction 不一样：

```text
y = sum(x[0], x[1], ..., x[n-1])
```

最终结果依赖所有输入。线程各算一部分后，还必须合并结果。

难点：

- 线程之间要同步。
- 部分结果要放 shared memory 或寄存器。
- 有些线程后期会闲下来。
- 访问模式可能产生 bank conflict。

## 树形归约直觉

8 个数求和：

```text
第 0 层：x0 x1 x2 x3 x4 x5 x6 x7
第 1 层：x0+x1, x2+x3, x4+x5, x6+x7
第 2 层：(x0+x1)+(x2+x3), (x4+x5)+(x6+x7)
第 3 层：最终 sum
```

并行 reduction 的关键就是把这个树映射到线程和 shared memory。

## row-wise reduction 设计

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
一个 block 处理一行。
block 内多个 thread 处理这一行的不同列。
每个 thread 先在寄存器里累加局部结果。
再把局部结果放到 shared memory 归约。
最后 thread 0 写出。
```

## 代码骨架：row-wise sum

```cpp
template<int BLOCK_SIZE>
__global__ void row_sum_kernel(
    const float* x,
    float* y,
    int M,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local = 0.0f;
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        local += x[row * N + col];
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[tid] = local;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = smem[0];
    }
}
```

注意：

- 这个版本用于教学，不一定是最高性能。
- 当 `BLOCK_SIZE` 不是 2 的幂时要更小心。
- 后续可以用 warp shuffle 优化。

## 代码骨架：row-wise max

```cpp
float local = -INFINITY;
for (int col = tid; col < N; col += BLOCK_SIZE) {
    local = fmaxf(local, x[row * N + col]);
}
```

max 的常见坑：

```text
不能初始化成 0，因为输入可能全是负数。
```

## 实验 shape

| M | N | 目的 |
|---:|---:|---|
| 1024 | 32 | 小行长度，测试 launch/并行粒度 |
| 1024 | 128 | 常见小 hidden |
| 1024 | 768 | BERT 类 hidden |
| 1024 | 1024 | 标准 2 的幂 |
| 1024 | 4096 | LLM hidden |
| 1024 | 8192 | 大 hidden |

## 实验表

| M | N | BLOCK_SIZE | row_sum ms | row_max ms | 是否正确 | 观察 |
|---:|---:|---:|---:|---:|---|---|

## 常见错误排查

### sum 误差比 PyTorch 大

可能原因：

- FP16 直接累加。
- 归约顺序不同，浮点误差略有差异。
- 容忍度设置过严。

### max 对负数错

可能原因：

- 初始化成 0。

### 程序偶发错

可能原因：

- 少了 `__syncthreads()`。
- shared memory 越界。

## 自测题

1. reduction 为什么需要同步？
2. 一个 block 处理一行有什么好处？
3. `for (col = tid; col < N; col += BLOCK_SIZE)` 是什么模式？
4. max 初始化为什么不能随便用 0？
5. softmax 为什么依赖 reduction？

参考答案：

1. 因为多个线程的局部结果要合并。
2. 行内归约自然，输出一行一个结果。
3. 每个线程跨步处理多个元素。
4. 如果输入全负，0 会变成错误最大值。
5. softmax 需要每行 max 和 exp 后的 sum。

## 今天的验收标准

你能说清楚：

```text
Reduction 的基本套路是每个线程先做局部累加，再在 block 内做树形合并。softmax 和 layernorm 不是凭空来的，它们的核心都包含 row-wise reduction。
```

