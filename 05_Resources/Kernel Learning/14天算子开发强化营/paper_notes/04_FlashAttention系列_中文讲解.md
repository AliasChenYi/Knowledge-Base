# FlashAttention 系列中文讲解

本地 PDF：

- [04_FlashAttention_NeurIPS2022.pdf](../papers/04_FlashAttention_NeurIPS2022.pdf)
- [05_FlashAttention2.pdf](../papers/05_FlashAttention2.pdf)
- [06_FlashAttention3.pdf](../papers/06_FlashAttention3.pdf)

推荐阅读日：Day 13  
原始来源：

- [FlashAttention NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- [FlashAttention-2 arXiv](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3 arXiv](https://arxiv.org/abs/2407.08608)

## 1. 普通 Attention 为什么慢

Self-attention 的核心公式是：

```text
O = softmax(QK^T / sqrt(d)) V
```

看起来只是矩阵乘 + softmax + 矩阵乘。但问题在于 `QK^T` 会生成一个 `N x N` 的注意力矩阵。序列长度 N 一大，这个中间矩阵非常大。

普通实现的代价不是只在计算上，还在 HBM 读写上：

1. 读 Q/K，写出注意力分数 S。
2. 读 S，做 softmax，写出 P。
3. 读 P/V，写出 O。

中间矩阵 S/P 太大，写回和读回 HBM 非常贵。

## 2. FlashAttention 的核心思想

FlashAttention 的关键词是 **IO-aware**。

它不是近似 attention，而是 exact attention。它改变的是计算顺序和数据搬运方式：

- 把 Q/K/V 切成块。
- 每次只把一小块搬进片上 SRAM/shared memory。
- 在片上计算局部 attention。
- 用 online softmax 合并不同 block 的结果。
- 避免把完整的 `N x N` attention matrix 写回 HBM。

所以它快，不是因为少算了很多乘加，而是因为少搬了很多数据。

## 3. Online Softmax 为什么重要

普通 softmax 要先知道整行最大值，再算 exp，再算 sum。

分块之后，你一次只能看到一部分元素。online softmax 的作用是：当新块到来时，用新的 max 和 sum 修正旧结果。

直觉：

```text
旧块有 old_max、old_sum、old_output
新块有 block_max、block_sum、block_output
合并时用 new_max = max(old_max, block_max)
然后把旧结果和新结果都按 new_max 重新缩放
```

这就是 FlashAttention 能边处理 block 边保持精确 softmax 的原因。

## 4. FlashAttention-2 改进了什么

FlashAttention-1 的主要贡献是减少 HBM IO。FlashAttention-2 进一步关注：

- 更好的并行划分。
- 减少非矩阵乘部分的开销。
- 改善 warp/block 之间的工作分配。
- 让更多时间花在高吞吐的 matmul 上。

如果 FlashAttention-1 是“少搬数据”，FlashAttention-2 就是“搬得少的同时，让计算单元更忙”。

## 5. FlashAttention-3 改进了什么

FlashAttention-3 面向更新的 NVIDIA Hopper 架构，重点是：

- 利用异步机制把数据搬运和计算重叠。
- 利用 Hopper 的新矩阵乘能力。
- 面向 FP8 等低精度路径提高吞吐。

对新人来说，FlashAttention-3 不要求马上完全看懂，但它能让你看到资深算子优化的方向：不是只写一个 kernel，而是围绕硬件新能力重排整个流水线。

## 6. 14 天里怎么读

Day 13 只需要达到这个程度：

1. 能写出普通 attention 的数据流。
2. 能解释为什么 `N x N` 中间矩阵是问题。
3. 能说明 FlashAttention 用 tiling + online softmax 避免中间矩阵写回。
4. 能看懂 forward-only 的简化伪代码。

不要一上来啃 backward、dropout、causal mask、variable length、paged KV cache。先抓住 IO-aware 这根主线。

## 7. 入职后怎么用

你可以把 FlashAttention 当成算子优化“范式案例”：

- 先从性能模型出发，不从代码技巧出发。
- 找出真正的瓶颈是 IO。
- 改变计算组织方式，而不是只调参数。
- 用硬件层次结构解释优化收益。

这套思路可以迁移到 RMSNorm、fused MLP、MoE、KV cache、量化 GEMM 等很多任务上。

