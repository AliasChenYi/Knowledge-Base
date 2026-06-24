# Triton 论文中文讲解

本地 PDF：[03_Triton_MAPL2019.pdf](../papers/03_Triton_MAPL2019.pdf)  
推荐阅读日：Day 8-10  
原始来源：[arXiv:1909.12082](https://arxiv.org/abs/1909.12082)

## 1. Triton 想解决什么问题

CUDA 很强，但写高性能 CUDA kernel 对新人很不友好。你要同时管：

- 线程层级
- shared memory
- warp 行为
- 同步
- 寄存器
- load/store 合并
- 边界 mask

Triton 的思路是：让程序员以“块”为单位描述计算，让编译器处理很多底层细节。

换句话说，CUDA 更像“你告诉每个线程做什么”；Triton 更像“你告诉每个 program 处理哪块数据”。

## 2. Triton 的核心抽象

### 2.1 Program

Triton kernel 由很多 program 实例组成。一个 program 通常处理一个 tile 或一行数据。

例如 softmax：

```text
program_id(0) = 当前处理第几行
offsets = row_start + tl.arange(0, BLOCK_SIZE)
```

### 2.2 Block Tensor

Triton 里的 `tl.arange`、`tl.load`、`tl.store` 操作的不是一个标量，而是一小块向量化的数据。

这让你可以自然地写：

```python
x = tl.load(ptr + offsets, mask=offsets < n_cols)
m = tl.max(x, axis=0)
```

这段代码像 NumPy，但会被编译成 GPU kernel。

### 2.3 Mask

真实 tensor 的长度常常不是 block size 的整数倍。Triton 用 mask 处理边界：

```python
mask = offsets < n_cols
tl.load(ptr + offsets, mask=mask, other=-float("inf"))
```

这个模式非常重要，softmax、layernorm、matmul 都会用到。

## 3. Triton 适合什么

适合：

- fused elementwise
- row-wise reduction
- softmax
- layernorm/rmsnorm
- 中小规模 matmul
- 快速验证一个 kernel 思路

不一定适合：

- 极限性能 GEMM，尤其已经有成熟 vendor library/CUTLASS。
- 需要非常底层硬件指令控制的场景。
- 很复杂的跨 block 同步。

## 4. 对新人最重要的启发

Triton 是很好的“入职缓冲层”：

- 它让你较快写出自定义 kernel。
- 它逼你理解 tile、mask、stride、program id。
- 它比 CUDA 更容易和 PyTorch 实验结合。

但你仍然要理解 CUDA/硬件，否则性能问题来了会不知道怎么解释。

## 5. 14 天里怎么读

Day 8-10 这样读：

1. Day 8：只理解 program model，不追求论文细节。
2. Day 9：对应官方 fused softmax 教程，看 `tl.max`、`tl.sum`、mask。
3. Day 10：对应 matmul 教程，看 BLOCK_M/N/K 和 accumulator。

读论文时重点关注：

- Triton 为什么选择 tile-level programming。
- 它如何把高层 block 操作降到 GPU。
- 它的目标不是替代所有 CUDA，而是降低高性能 kernel 的表达成本。

