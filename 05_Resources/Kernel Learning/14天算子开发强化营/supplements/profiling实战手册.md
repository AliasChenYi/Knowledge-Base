# Profiling 实战手册

## 1. 先记住一句话

不要凭感觉优化。一个成熟的算子优化流程应该是：

```text
先有 correctness -> 再有 benchmark -> 再有 profiler 证据 -> 最后才改代码
```

如果你还没有 reference 和 benchmark，就先别急着开 profiler。否则看到一堆指标也不知道该信哪个。

## 2. 三种性能问题

### 2.1 Kernel 慢

单个 kernel 执行时间长。

常见原因：

- global memory 访问不连续。
- HBM 带宽打满。
- shared memory bank conflict。
- 寄存器太多导致 occupancy 低或 spill。
- 没用上 Tensor Core。
- 分支发散。

### 2.2 Kernel 很多

每个 kernel 不慢，但 kernel 数量太多。

常见原因：

- elementwise 没有 fusion。
- softmax/norm 拆成多个小 op。
- 框架 eager 执行产生很多 launch。

优化方向：

- fusion。
- torch.compile / Inductor。
- Triton custom fused kernel。

### 2.3 End-to-end 慢

单 kernel 看起来不错，但模型端到端慢。

常见原因：

- host-device copy。
- layout conversion。
- 同步点太多。
- 通信等待。
- data loader 或 CPU 侧瓶颈。
- runtime 调度开销。

这时要用 timeline 工具，而不是只盯着单个 kernel。

## 3. 工具怎么选

| 问题 | 工具 |
|---|---|
| 单个 CUDA kernel 指标 | Nsight Compute |
| 端到端 timeline | Nsight Systems |
| PyTorch 侧 op 分布 | PyTorch Profiler |
| Triton kernel 快速计时 | `triton.testing.do_bench` |
| CPU hotspot | perf / VTune |
| AMD 平台 | rocprof / rocprofiler |
| 厂商 NPU | 厂商 profiler |

## 4. Nsight Systems 看什么

Nsight Systems 更适合回答：

```text
程序时间花在哪里？
CPU 和 GPU 是否并行？
kernel launch 是否太多？
有没有意外同步？
有没有 H2D/D2H copy？
通信和计算是否重叠？
```

你重点看：

- CUDA kernel timeline。
- CPU thread timeline。
- memcpy。
- synchronization。
- NCCL/通信。
- kernel 间空洞。

典型现象：

```text
很多很短的小 kernel 连成一串
```

说明 launch overhead 和中间 tensor 可能是问题。

## 5. Nsight Compute 看什么

Nsight Compute 更适合回答：

```text
这个 kernel 为什么慢？
```

常见指标维度：

- Memory throughput。
- L2 hit rate。
- DRAM throughput。
- SM utilization。
- Tensor Core utilization。
- Occupancy。
- Registers per thread。
- Shared memory usage。
- Warp stall reasons。

你不需要一开始看懂所有指标。先按这个顺序：

1. 看 kernel 时间。
2. 看 memory throughput 是否接近硬件上限。
3. 看 compute utilization 是否高。
4. 看是否有明显 uncoalesced access。
5. 看 register 和 occupancy。
6. 看 stall reason。

## 6. Profiling 前检查清单

- [ ] correctness 已通过。
- [ ] benchmark 可复现。
- [ ] 记录了硬件和软件版本。
- [ ] shape/dtype/layout 已记录。
- [ ] baseline 明确。
- [ ] 已做 warmup。
- [ ] 没有把首次 JIT 编译计入结果。
- [ ] 没有把 host-device copy 混进 kernel 时间，除非你明确要测端到端。

## 7. 常见结论怎么写

### Memory-bound

```text
该 kernel 算术强度较低，profiling 中 DRAM throughput 较高而 compute utilization 较低，说明主要瓶颈在 HBM 读写。下一步优先考虑 fusion、减少中间 tensor、改善 coalescing 或提高片上复用。
```

### Compute-bound

```text
该 kernel 的 compute utilization 较高，memory throughput 未接近上限，瓶颈更偏计算侧。下一步检查是否使用 Tensor Core/MMA、tile shape 是否合理、指令流水是否充分。
```

### Launch-bound

```text
单个 kernel 时间很短，但 timeline 中存在大量小 kernel，端到端时间受 launch overhead 和中间 tensor 调度影响。下一步考虑 fusion 或编译器路径。
```

## 8. 新人不要被指标淹没

你只要先抓三问：

1. 时间花在哪个 kernel 或哪个阶段？
2. 它更像 memory-bound、compute-bound，还是 launch-bound？
3. 有什么证据支持这个判断？

能回答这三个问题，就已经比“我感觉慢”强很多。

