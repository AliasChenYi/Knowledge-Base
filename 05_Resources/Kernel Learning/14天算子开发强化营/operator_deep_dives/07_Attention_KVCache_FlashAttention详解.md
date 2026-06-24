# 07：Attention / KV Cache / FlashAttention 详解

## 1. 普通 Attention

公式：

```text
O = softmax(QK^T / sqrt(d)) V
```

shape：

```text
Q: [B, H, S_q, D]
K: [B, H, S_k, D]
V: [B, H, S_k, D]
O: [B, H, S_q, D]
```

其中：

- B：batch。
- H：head 数。
- S：sequence length。
- D：head dimension。

## 2. Attention 的计算步骤

```text
1. scores = Q @ K^T
2. scores = scores / sqrt(D)
3. apply mask
4. probs = softmax(scores)
5. O = probs @ V
```

普通实现会产生：

```text
scores: [B, H, S_q, S_k]
probs:  [B, H, S_q, S_k]
```

当 S 很大时，中间矩阵非常大。

## 3. MHA / MQA / GQA

### MHA

Multi-Head Attention：

```text
Q/K/V 都有 H 个 head。
```

### MQA

Multi-Query Attention：

```text
Q 有多个 head，K/V 共享较少 head，甚至 1 个。
```

优点：

- KV cache 更小。
- decode 更省内存带宽。

### GQA

Grouped-Query Attention：

```text
多个 Q head 共享一组 K/V head。
```

介于 MHA 和 MQA 之间。

## 4. Prefill vs Decode

LLM 推理分两阶段：

### Prefill

输入 prompt，处理整段序列。

特点：

- S_q 和 S_k 都较大。
- attention 类似大矩阵计算。
- FlashAttention 很重要。

### Decode

每次生成一个新 token。

特点：

- S_q 通常是 1。
- K/V cache 很长。
- 更像 GEMV / memory bandwidth 问题。
- KV cache 读取是瓶颈。

## 5. KV Cache

自回归推理时，历史 token 的 K/V 可以缓存。

每步 decode：

```text
新 token 产生 q, k, v
把 k/v 追加到 cache
用 q attend 到所有历史 K/V
```

KV cache 大小：

```text
layers * batch * heads_kv * seq_len * head_dim * 2(K,V) * bytes
```

这会非常大。

## 6. FlashAttention

FlashAttention 的核心：

```text
不把完整 scores/probs 写回 HBM。
```

它通过：

- Q/K/V tiling。
- online softmax。
- 片上 SRAM 复用。

减少 HBM IO。

## 7. Online Softmax

分块处理时无法一次看到整行 scores。

维护：

```text
m: 当前最大值
l: 当前 exp sum
o: 当前输出累计
```

新 block 到来时更新：

```text
m_new = max(m_old, block_max)
旧结果按 exp(m_old - m_new) 缩放
新结果按 exp(block_score - m_new) 加入
```

这样保持 exact softmax。

## 8. Mask

Attention 常见 mask：

- padding mask。
- causal mask。
- sliding window mask。
- block sparse mask。

mask 必须在 softmax 前应用，通常把 masked position 设成 `-inf`。

## 9. PagedAttention

PagedAttention 关注 serving 中 KV cache 管理。

问题：

- 不同 request 长度不同。
- KV cache 频繁增长。
- 内存碎片和浪费。

思想：

```text
像操作系统分页一样管理 KV cache。
```

这说明 LLM 性能不只是 kernel，还包括 runtime memory management。

## 10. Attention 优化点

1. FlashAttention 减少中间 HBM。
2. MQA/GQA 减少 KV cache。
3. KV cache layout 优化。
4. paged KV cache。
5. causal mask 高效处理。
6. FP8/INT8 KV cache。
7. prefill/decode 分别优化。
8. attention 和 rotary embedding、bias、mask fusion。

## 11. 测试计划

shape：

```text
B = 1, 4, 16
H = 8, 16, 32
S = 128, 512, 2048, 4096, 8192
D = 64, 128
```

测试：

- causal / non-causal。
- MHA / MQA / GQA。
- FP16/BF16。
- extreme mask。
- short sequence / long sequence。

## 12. Benchmark 计划

区分：

- prefill latency。
- decode latency。
- tokens/s。
- memory usage。
- batch size。
- sequence length。

不要把 prefill 和 decode 混在一起报一个数字。

## 13. 常见坑

1. 误以为 FlashAttention 是近似。
2. 忽略 `N x N` 中间矩阵。
3. causal mask 位置错。
4. online softmax 缩放公式错。
5. prefill/decode 混淆。
6. KV cache layout 不合理。
7. 只测短序列。

## 14. 推荐资料

- FlashAttention 2022。
- FlashAttention-2。
- FlashAttention-3。
- PagedAttention/vLLM。
- TensorRT-LLM docs。
- 本资料包 FlashAttention 中文讲解。

