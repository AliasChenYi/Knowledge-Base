# Kernel Profiling Checklist

## Repro Setup
- 固定输入 shape、dtype、batch size 和 seed。
- 记录 GPU 型号、driver、CUDA、runtime、commit。
- 先跑 warmup，再采集稳定区间。
- 保留 baseline 命令和 optimized 命令。

## First Pass
- 正确性：输出误差是否在可接受范围内。
- 延迟：均值、P50、P90、P99。
- 吞吐：elements/s、tokens/s 或 TFLOPS。
- 显存：峰值显存、workspace、临时 buffer。

## Nsight Systems
- GPU 是否有空洞。
- CPU launch overhead 是否明显。
- H2D / D2H copy 是否阻塞计算。
- 多 stream 是否真的并行。
- 通信和计算是否重叠。

## Nsight Compute
- Achieved occupancy。
- Memory throughput。
- L2 hit rate。
- Shared memory bank conflict。
- Warp stall reasons。
- Tensor Core utilization。
- Register pressure。

## Optimization Hypotheses
- Data layout 是否导致非合并访存。
- Tile size 是否匹配 cache / shared memory / register 约束。
- 是否有多余同步或分支发散。
- 是否能融合相邻算子减少 memory round trip。
- 是否需要 vectorized load / store。

## Reporting Format
```md
- Baseline:
- Variant:
- Input:
- Metric:
- Speedup:
- Numerical diff:
- Caveat:
```
