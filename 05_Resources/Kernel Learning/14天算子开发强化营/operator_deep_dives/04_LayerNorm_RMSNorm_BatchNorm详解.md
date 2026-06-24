# 04：LayerNorm / RMSNorm / BatchNorm 详解

## 1. Norm 类算子解决什么问题

Normalization 的目标是让激活分布更稳定，帮助训练或推理。

常见 norm：

- BatchNorm。
- LayerNorm。
- RMSNorm。
- GroupNorm。

算子开发里最常遇到的是：

- LayerNorm。
- RMSNorm。

尤其 LLM 中 RMSNorm 非常常见。

## 2. LayerNorm 数学语义

对最后一维 N 做归一化：

```text
mean = sum_i x_i / N
var = sum_i (x_i - mean)^2 / N
y_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
```

输入：

```text
x: [M, N]
gamma: [N]
beta: [N]
```

输出：

```text
y: [M, N]
```

## 3. RMSNorm 数学语义

RMSNorm 不减 mean：

```text
rms = sqrt(sum_i x_i^2 / N + eps)
y_i = x_i / rms * weight_i
```

它比 LayerNorm 少一个 mean reduction，计算略简单。

LLM 中常见：

- LLaMA 系列使用 RMSNorm。
- 很多 decoder-only Transformer 都有 RMSNorm 变体。

## 4. BatchNorm 与 LayerNorm 的区别

BatchNorm 通常沿 batch 统计：

```text
对每个 channel，在 batch/spatial 上统计 mean/var
```

LayerNorm 通常对单个样本的 hidden 维统计：

```text
对每一行 hidden vector 统计 mean/var
```

工程差异：

- BatchNorm 训练/推理语义不同，有 running mean/var。
- LayerNorm/RMSNorm 训练/推理通常更一致。
- BatchNorm 在 CNN 中常见，LayerNorm/RMSNorm 在 Transformer 中常见。

## 5. 计算结构

LayerNorm：

```text
reduction(sum) -> mean
reduction(sum square) -> variance
elementwise normalize
elementwise scale/bias
```

RMSNorm：

```text
reduction(sum square) -> rms
elementwise normalize
elementwise scale
```

所以它们和 softmax 一样，核心是 row-wise reduction。

## 6. 数值精度

强烈建议：

```text
FP16/BF16 输入，统计量用 FP32。
```

原因：

- hidden size 可能很大。
- 半精度累加误差明显。
- variance 对误差敏感。

输出可以 cast 回输入 dtype。

## 7. Variance 的两种写法

写法 1：

```text
var = mean((x - mean)^2)
```

写法 2：

```text
var = mean(x^2) - mean(x)^2
```

写法 2 计算可能更省，但数值稳定性要注意。

务必和 reference 语义对齐。

## 8. 并行映射

常见输入：

```text
X[M, N]
```

设计：

```text
一个 block / Triton program 处理一行。
```

每行：

1. 读入 N 个元素。
2. 求 mean/rms。
3. 求 variance。
4. normalize。
5. 乘 weight，加 bias。
6. 写回。

## 9. 大 hidden size 怎么办

如果 N 很大：

- 一个 block 装不下整行。
- 寄存器压力变大。
- shared memory 压力变大。

可能策略：

- 多 block 处理一行，做 partial reduction。
- 分块读写。
- 对常见 hidden size specialization。
- 使用 vendor/library 或编译器生成。

## 10. Backward 简介

训练需要 backward。LayerNorm backward 比 forward 更复杂，因为梯度也涉及 reduction。

初学阶段建议：

```text
先把 forward 写扎实，再看 backward。
```

如果岗位偏训练算子，后续必须补 backward。

## 11. 测试计划

测试 shape：

```text
[1, 128]
[32, 768]
[64, 1024]
[64, 4096]
[16, 8192]
```

测试值：

- random。
- zeros。
- 全相同值。
- 大正数/大负数。
- FP32/FP16/BF16。

检查：

- 和 PyTorch reference。
- max_abs_error。
- mean_abs_error。
- 是否 NaN/Inf。

## 12. Benchmark 计划

比较：

- PyTorch LayerNorm/RMSNorm reference。
- Triton implementation。
- custom CUDA。
- vendor fused norm，如果有。

记录：

- M/N。
- dtype。
- 是否含 weight/bias。
- latency。
- effective bandwidth。

## 13. 常见坑

1. eps 加错位置。
2. FP16 直接算 mean/var。
3. variance 和 reference 定义不同。
4. weight/bias shape 错。
5. 对全相同输入 variance 为 0 时处理不稳。
6. cast 顺序不同导致误差。
7. non-contiguous 输入没处理。

## 14. 推荐资料

- PyTorch LayerNorm 文档。
- Triton LayerNorm tutorial。
- LLaMA/RMSNorm 相关实现。
- 本资料包：[Triton RMSNorm](../code_labs/triton/rmsnorm.py)。

