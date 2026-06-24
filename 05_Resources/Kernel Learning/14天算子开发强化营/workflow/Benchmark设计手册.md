# Benchmark 设计手册

## 1. Benchmark 的目标

Benchmark 不是为了证明“我很快”，而是为了回答：

```text
在什么硬件、什么 shape、什么 dtype、什么 baseline 下，我的实现是否更好？
```

## 2. 一个可信 benchmark 必须包含

- 硬件信息。
- 软件版本。
- shape。
- dtype。
- layout。
- baseline。
- warmup。
- 多次重复。
- 同步。
- correctness。
- 统计口径。

## 3. 不可信 benchmark 的典型样子

```text
我跑了一次，快了 3 倍。
```

问题：

- baseline 不明。
- shape 不明。
- dtype 不明。
- 是否同步不明。
- 是否正确不明。
- 是否包含数据拷贝不明。

## 4. 统计指标

建议至少记录：

- median。
- p20/p80 或 p50/p90。
- min 仅供参考。
- 标准差可选。

不要只报最小值。最小值容易过于乐观。

## 5. Shape sweep

只测一个 shape 很危险。

示例：

```text
softmax:
rows = 1024, 4096
cols = 128, 256, 512, 1024, 2048, 4096
```

GEMM：

```text
(M,N,K) = (512,512,512)
(1024,1024,1024)
(4096,4096,4096)
(1,4096,4096)
(128,4096,11008)
```

## 6. Baseline 怎么选

优先级：

1. 线上旧实现。
2. vendor library。
3. PyTorch/框架内置实现。
4. 编译器生成实现。
5. naive reference。

不同 baseline 意义不同。和 naive 比快不代表生产有价值。

## 7. Latency vs Throughput

Latency：

```text
单次请求耗时。
```

Throughput：

```text
单位时间处理多少数据。
```

LLM 推理里还常看：

- tokens/s。
- time to first token。
- inter-token latency。

## 8. 计算有效带宽

适合 memory-bound 算子：

```text
effective_bandwidth = bytes_moved / time
```

例如 FP32 vector add：

```text
bytes = N * 3 * 4
```

## 9. 计算 TFLOPS

适合 GEMM：

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / seconds / 1e12
```

注意：

- 这是理论运算量。
- 如果有 sparsity、量化、特殊指令，要明确口径。

## 10. Benchmark 报告模板

```text
硬件：
软件：
算子：
支持范围：
baseline：
统计方式：

结果表：

结论：
1. 哪些 shape 有收益？
2. 哪些 shape 没收益？
3. 为什么？
4. 下一步是什么？
```

