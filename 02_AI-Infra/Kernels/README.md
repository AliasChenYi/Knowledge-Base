# Kernels

## Scope
记录算子优化、数据类型选择、访存策略与向量化实践。

## Suggested Topics
- FP16/BF16 计算与累加策略
- Vectorization 与 data layout 转换
- Shared memory 与 cache-aware kernel 设计
- 硬件指令利用与吞吐优化

## Core Files
- `kernel-note-template.md`：记录单个 kernel 优化过程。
- `profiling-checklist.md`：Nsight Systems / Nsight Compute 分析清单。

## Minimal Note Structure
- 瓶颈现象（Bottleneck）
- 优化思路（Optimization idea）
- Benchmark 方法
- 结果与注意事项（Caveats）

## Review Questions
- baseline 是否包含 warmup 和稳定区间？
- profiler 证据是否支持当前优化假设？
- speedup 是否有数值误差、输入规模或硬件限制？
