# Kernel Note Template

## Context
- 模型 / 算子：
- 硬件平台：
- Precision（精度）：
- Framework / Runtime：
- Input shape：

## Bottleneck
描述观测到的瓶颈现象，并附上 profiler 证据。

## Optimization Strategy
- Data layout
- Compute pattern
- Memory movement

## Evaluation
- Baseline：
- Optimized：
- Speedup：
- Numerical impact（数值影响）：
- Benchmark command：

## Failure Modes
- 可能退化的输入规模：
- 可能引入的数值或兼容性风险：

## Reusability
该优化模式还可以复用在哪些场景？
