# 补充论文导读：编译器、量化与 LLM Serving

这份导读对应新下载的 8 篇资料：

- [TVM](../papers/07_TVM_OSDI2018.pdf)
- [Ansor](../papers/08_Ansor_OSDI2020.pdf)
- [MLIR](../papers/09_MLIR_2020.pdf)
- [Halide](../papers/10_Halide_PLDI2013.pdf)
- [PagedAttention/vLLM](../papers/11_PagedAttention_vLLM.pdf)
- [SmoothQuant](../papers/12_SmoothQuant.pdf)
- [AWQ](../papers/13_AWQ_MLSys2024.pdf)
- [LLM.int8](../papers/14_LLM_int8_NeurIPS2022.pdf)

## 1. Halide：为什么要分开“算什么”和“怎么调度”

Halide 的核心思想是 algorithm/schedule 分离。

对于算子开发，这个思想非常重要：

```text
数学公式描述“算什么”；
tiling、vectorize、parallel、unroll 描述“怎么在硬件上算”。
```

你写 CUDA/Triton 时其实也在做同样的事，只是没有显式分层。

读这篇时只抓：

- algorithm/schedule 分离。
- locality。
- parallelism。
- recomputation trade-off。

## 2. TVM：深度学习编译器的整体架构

TVM 解决的问题是：

```text
如何把深度学习模型从高层图优化到不同硬件上的高性能代码。
```

它包含两层：

- graph-level optimization：图融合、layout 选择等。
- operator-level optimization：schedule、tiling、vectorize、unroll。

对算子工程师的启发：

```text
手写 kernel 和编译器生成 kernel 不是对立的。手写经验会帮助你理解编译器 schedule；编译器思想会帮助你更系统地设计优化空间。
```

## 3. Ansor：自动生成高性能 tensor program

Ansor 关注自动调优：

- 自动构造搜索空间。
- 用 cost model 预测候选 schedule。
- 通过真实测量不断改进。

你要理解：

```text
很多 tile/block 参数不是靠背，而是靠搜索和测量。
```

这对入职后做 shape-specific 优化很重要。

## 4. MLIR：多层 IR 和 lowering

MLIR 的核心是多层 IR：

```text
高层 tensor/op 语义
-> loop / affine / linalg
-> vector / gpu
-> LLVM / target code
```

你第一次读不用懂所有 dialect，只要知道：

- dialect 是不同抽象层。
- lowering 是从高层逐步降到低层。
- pattern rewrite 是编译器优化的重要机制。

## 5. PagedAttention：LLM Serving 的内存管理问题

PagedAttention 不是单纯 kernel 优化，它解决 serving 阶段 KV cache 管理问题。

普通 LLM serving 的痛点：

- 每个 request 长度不同。
- KV cache 很大。
- 内存碎片和浪费严重。

PagedAttention 的直觉：

```text
像操作系统分页管理内存一样管理 KV cache。
```

对算子工程师的启发：

```text
端到端性能不只来自某个 kernel，runtime memory management 也可能是核心瓶颈。
```

## 6. LLM.int8：大模型 INT8 的 outlier 问题

LLM.int8 关注大模型中 activation outlier。

普通 INT8 对 LLM 直接量化可能掉精度，因为少数异常大的激活值很重要。

核心启发：

```text
量化不是只把 float 映射成 int。你必须理解数据分布和 outlier。
```

## 7. SmoothQuant：把量化难题从 activation 转移到 weight

SmoothQuant 的目标是 W8A8。

它的直觉：

```text
activation outlier 难量化，就通过平滑变换把难度迁移到 weight 上。
```

对 kernel 的启发：

- scale 可能进入 matmul 前后处理。
- dequant/scale 最好融合。
- 量化策略会影响算子数据流。

## 8. AWQ：Activation-aware Weight Quantization

AWQ 关注 INT4 weight-only。

核心直觉：

```text
不是所有权重同等重要；根据 activation 识别重要权重并保护它们。
```

对算子工程师的启发：

- INT4 需要 pack/unpack。
- group size 影响精度和性能。
- scale 读取和 dequant 可能成为新瓶颈。

## 9. 推荐阅读顺序

如果你关注编译器：

```text
Halide -> TVM -> Ansor -> MLIR
```

如果你关注 LLM 推理：

```text
FlashAttention -> PagedAttention -> LLM.int8 -> SmoothQuant -> AWQ
```

如果你现在刚入门：

```text
先别急着完整啃这些论文。每篇先读摘要、图 1、方法大图和结论，再回到代码实验。
```

