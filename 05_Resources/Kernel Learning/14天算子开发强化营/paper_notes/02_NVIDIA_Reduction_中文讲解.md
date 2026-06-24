# NVIDIA Optimizing Parallel Reduction 中文讲解

本地 PDF：[02_NVIDIA_Optimizing_Parallel_Reduction.pdf](../papers/02_NVIDIA_Optimizing_Parallel_Reduction.pdf)  
推荐阅读日：Day 5  
原始来源：[NVIDIA CUDA Reduction PDF](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

## 1. 它解决什么问题

Reduction 是算子开发里最基础也最容易低估的模式。sum、max、mean、variance、softmax、layernorm、rmsnorm、batchnorm、attention 里都能看到它。

Reduction 的本质是：

```text
很多元素 -> 一个或少数几个结果
```

难点在于：并行程序喜欢“每个线程独立干活”，但 reduction 要把很多线程的结果合并起来，这就引入同步、共享内存访问、线程分歧和多阶段归并。

## 2. 关键优化思路

### 2.1 树形归约

不要让一个线程从头加到尾，而是让多个线程先各自处理一部分，再两两合并。

直觉：

```text
8 个数求和：
第 1 轮：x0+x1, x2+x3, x4+x5, x6+x7
第 2 轮：前两个结果相加，后两个结果相加
第 3 轮：得到最终结果
```

### 2.2 避免分支发散

如果 warp 内不同线程走不同分支，硬件会串行执行不同路径。早期 reduction 写法很容易让线程按奇偶分支执行，导致效率差。

优化方向：

- 让活跃线程连续。
- 让索引模式更规整。
- 尽量减少 warp 内分歧。

### 2.3 减少 shared memory bank conflict

shared memory 很快，但不是“随便访问都快”。多个线程如果同时打到同一个 bank，会产生冲突。

优化方向：

- 调整访问 stride。
- 使用连续线程访问连续地址。
- 在 warp 内使用 shuffle 指令减少 shared memory 往返。

### 2.4 多元素每线程

每个线程不一定只处理一个元素。一个线程可以先连续读取多个元素，在寄存器里累加，再参与 block 内归约。

好处：

- 减少 block 数量。
- 提高每个线程的工作量。
- 更好利用内存带宽。

## 3. 对 softmax 和 layernorm 的意义

Row-wise softmax 通常需要：

1. 对一行求 max。
2. 对一行求 exp 后的 sum。
3. 每个元素除以 sum。

LayerNorm 通常需要：

1. 对一行求 mean。
2. 对一行求 variance。
3. 每个元素归一化。

这些都离不开 reduction。所以 Day 5 把 reduction 学明白，Day 6 的 softmax/layernorm 才不会只是在背公式。

## 4. 14 天里怎么读

建议读法：

1. 先看最朴素的 reduction kernel。
2. 每出现一个优化版本，就问：它减少了什么开销？
3. 把优化点写成表格：分支、bank conflict、同步、访存、循环展开。
4. 不要求完全背代码，但要能复现一个 block 内 sum/max。

## 5. 入职后怎么用

当你优化 row-wise 算子时，可以先检查：

- 一行长度是多少？
- 一个 block 处理一行还是多行？
- 每个线程处理几个元素？
- 是否需要跨 block 合并？
- warp 内用 shuffle 还是 shared memory？
- FP16 输入是否用 FP32 累加？

这几个问题能直接把“我不会优化”变成“我知道从哪里下手”。

