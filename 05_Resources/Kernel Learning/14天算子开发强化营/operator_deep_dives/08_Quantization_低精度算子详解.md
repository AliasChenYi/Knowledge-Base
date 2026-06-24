# 08：Quantization / 低精度算子详解

## 1. 量化是什么

量化把高精度数值映射到低精度表示。

例如：

```text
FP16/FP32 -> INT8
FP16/FP32 -> INT4
FP16/BF16 -> FP8
```

目标：

- 减少显存占用。
- 减少 HBM 带宽。
- 使用低精度硬件指令。
- 提高吞吐。

## 2. 基本公式

对称量化：

```text
x_float ≈ scale * x_int
```

非对称量化：

```text
x_float ≈ scale * (x_int - zero_point)
```

反量化：

```text
x_dequant = scale * (x_q - zero_point)
```

## 3. Scale 粒度

| 粒度 | 解释 | 特点 |
|---|---|---|
| per-tensor | 整个 tensor 一个 scale | 简单，精度可能差 |
| per-channel | 每个 channel 一个 scale | 常见，精度较好 |
| per-group | 每组元素一个 scale | INT4 LLM 常见 |
| per-token | 每个 token/row 一个 scale | activation 量化常见 |

scale 粒度越细，精度通常越好，但额外开销越大。

## 4. Weight-only Quantization

只量化权重：

```text
W_q: INT8/INT4
X: FP16/BF16
```

计算时：

```text
Y = X @ dequant(W_q)
```

实际 kernel 应尽量：

```text
边读 W_q，边 dequant，边 matmul。
```

不要真的把完整 W dequant 成 FP16 写回 HBM。

## 5. W8A8

W8A8：

```text
weight INT8
activation INT8
```

潜在性能更高，但激活量化更难，因为 activation 分布随输入变化。

SmoothQuant 解决的是：

```text
activation outlier 导致 W8A8 难量化。
```

## 6. INT4

INT4 常用于 LLM 权重压缩。

特点：

- 一个 byte 存两个 4-bit 值。
- 需要 pack/unpack。
- group scale 很重要。
- unpack 和 scale 读取可能吃掉性能收益。

INT4 kernel 常见瓶颈：

- 权重解包。
- scale 读取。
- dequant 计算。
- 低精度指令利用率。

## 7. FP8

FP8 是低精度浮点，常见格式：

- E4M3。
- E5M2。

需要关注：

- 动态范围。
- scale。
- accumulator dtype。
- 硬件是否支持。
- 框架是否支持。

FlashAttention-3、TensorRT-LLM 等资料会涉及 FP8。

## 8. Quantized GEMM/GEMV

INT8 GEMM 常见：

```text
A_int8 @ B_int8 -> int32 accumulator -> scale -> output
```

Weight-only GEMV 常见：

```text
x_fp16 @ w_int4/int8 -> fp16/fp32 output
```

Decode 阶段 batch 小，GEMV 很重要。

## 9. Pack / Unpack

INT4 需要 pack：

```text
byte = low4bits | (high4bits << 4)
```

unpack：

```text
low = byte & 0xF
high = byte >> 4
```

注意：

- signed / unsigned 表示。
- zero point。
- nibble 顺序。
- alignment。

## 10. Dequant Fusion

错误思路：

```text
先把整个 W_q dequant 成 W_fp16
再 GEMM
```

这样会失去显存优势。

更好思路：

```text
在 matmul/GEMV kernel 内融合 dequant。
```

也就是：

```text
load packed weight -> unpack -> apply scale -> multiply accumulate
```

## 11. 精度评估

量化不能只看单算子误差。

要看：

- 单层输出误差。
- 多层累积误差。
- 端到端模型指标。
- perplexity / accuracy / BLEU 等任务指标。

学习阶段先做单算子：

```text
max_abs
mean_abs
max_rel
cosine similarity
```

## 12. 测试计划

必须测：

- per-tensor scale。
- per-channel scale。
- per-group scale。
- group size = 32/64/128。
- INT8。
- INT4。
- 极值。
- outlier。
- small batch decode shape。

## 13. Benchmark 计划

记录：

- compression ratio。
- latency。
- memory bandwidth。
- dequant 开销。
- scale 读取开销。
- accuracy/误差。

比较：

- FP16 baseline。
- INT8 weight-only。
- INT4 weight-only。
- vendor quantized kernel。

## 14. 常见坑

1. 只看压缩率，不看 dequant 开销。
2. scale 粒度太细，读取 scale 变瓶颈。
3. pack 格式不适合连续读取。
4. INT4 unpack 实现低效。
5. accumulator dtype 错。
6. 单算子误差小但模型掉点。
7. 目标硬件没有高效低精度指令。

## 15. 推荐资料

- SmoothQuant。
- AWQ。
- LLM.int8。
- TensorRT-LLM Quantization。
- 本资料包：[量化算子入门](../supplements/量化算子入门.md)。

