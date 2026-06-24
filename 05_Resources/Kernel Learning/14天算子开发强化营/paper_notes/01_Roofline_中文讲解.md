# Roofline Model 中文讲解

本地 PDF：[01_Roofline_EECS-2008-134.pdf](../papers/01_Roofline_EECS-2008-134.pdf)  
推荐阅读日：Day 1-2  
原始来源：[Berkeley 技术报告页面](https://digicoll.lib.berkeley.edu/record/136692)

## 1. 这篇论文解决什么问题

刚开始做算子优化时，最容易犯的错是“看到慢就乱改”。Roofline 的价值是给你一个非常朴素但很强的判断框架：

> 一个 kernel 的性能上限，通常被两个东西夹住：计算峰值和内存带宽。

如果一个算子每读取 1 byte 数据只能做很少计算，那它很可能是 **memory-bound**；你拼命优化算术指令没有意义，应该减少读写、改善访问连续性、提高 cache/片上存储复用。

如果一个算子每读取 1 byte 数据能做很多计算，并且已经把数据复用得很好，那它可能是 **compute-bound**；这时要看 Tensor Core/MMA、SIMD、指令流水、occupancy、寄存器压力。

## 2. 核心概念

### 2.1 算术强度

算术强度是：

```text
Arithmetic Intensity = 计算量 FLOPs / 数据搬运量 Bytes
```

例子：

- Elementwise `y = x + 1`：每个元素读一次写一次，计算很少，算术强度低。
- Softmax：每行要 max、exp、sum、除法，计算比 elementwise 多，但仍然大量依赖 HBM 读写。
- 大矩阵乘 GEMM：A/B 的 tile 可以被复用很多次，算术强度高。

### 2.2 两条屋顶线

Roofline 图里有两段：

- 斜线：由内存带宽决定，越往右算术强度越高，理论性能越高。
- 横线：由计算峰值决定，再怎么提高算术强度也不能超过硬件峰值。

这就是“屋顶”的来源。

## 3. 对算子开发的启发

### 3.1 优化前先分类

你拿到一个慢算子，先不要急着改 block size。先粗略估计：

- 读了多少输入？
- 写了多少输出？
- 中间 tensor 有没有写回 HBM？
- 做了多少 FLOPs？

如果计算量很小、读写很多，优先考虑：

- fusion
- 减少中间结果
- contiguous/coalesced access
- vectorized load/store
- 利用 shared memory/SRAM 做复用

如果计算量很大，优先考虑：

- 使用 Tensor Core/MMA
- 合理 tiling
- 减少寄存器 spill
- 提高 occupancy
- pipeline 搬运和计算

### 3.2 GEMM 为什么是优化核心

GEMM 的特别之处在于：同一个 A/B 元素可以参与很多次乘加。只要 tiling 设计得好，数据从 HBM 搬到片上后能被复用很多次，所以它有机会靠近计算峰值。

这和 softmax、layernorm 这类 row-wise reduction 不一样。后者数据复用少，经常更像带宽问题。

## 4. 14 天里怎么读

Day 1-2 不需要读完整篇。按这个顺序：

1. 先看 Roofline 图，理解横轴和纵轴。
2. 只记住三个词：算术强度、memory-bound、compute-bound。
3. 找三个例子自己归类：vector add、softmax、GEMM。
4. 写一段 200 字笔记：为什么减少 HBM 读写经常比少算几次加法更重要。

## 5. 入职后怎么用

当你向导师或同事汇报性能问题时，可以这样说：

```text
我先估算了这个算子的算术强度。它每个元素大约只做几次运算，但至少读写多次全局内存，所以目前更像 memory-bound。我准备先看 global memory throughput、load/store efficiency 和中间 tensor 写回，再决定是否 fusion 或改 layout。
```

这比“我感觉它慢”专业太多。

