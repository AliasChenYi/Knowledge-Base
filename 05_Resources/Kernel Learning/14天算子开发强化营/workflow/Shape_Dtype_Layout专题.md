# Shape-Dtype-Layout 专题

## 1. 为什么这三个词很重要

算子任务里，性能和正确性很大程度取决于：

```text
shape, dtype, layout
```

同一个 softmax：

- `[32, 128]` 和 `[4096, 4096]` 是完全不同的问题。
- FP32 和 FP16 性能/误差不同。
- contiguous 和 non-contiguous 访问模式不同。

## 2. Shape：不要只写“支持任意 shape”

真实项目里要把 shape 写清楚：

```text
输入 x: [batch, hidden]
batch: 1-4096
hidden: 128/256/768/1024/4096/8192
```

shape 会影响：

- 一个 block 处理多少数据。
- 是否需要跨 block reduction。
- launch overhead 占比。
- tile size。
- 寄存器和 shared memory。

## 3. Dynamic Shape

动态 shape 意味着运行时 shape 变化。

风险：

- 为某个 shape 调好的 tile 不一定适合其他 shape。
- 编译器/JIT 可能为不同 shape 重新编译。
- 测试组合大幅增加。

建议：

```text
先收集线上或 benchmark 代表 shape，不要盲目追求完全通用。
```

## 4. Dtype

### FP32

适合 reference 和累加，性能/显存成本高。

### FP16/BF16

常用于训练/推理。注意：

- 输入可以低精度。
- reduction/accumulator 常用 FP32。
- 输出再 cast 回低精度。

### INT8/INT4/FP8

需要关注：

- scale。
- zero point。
- pack/unpack。
- 硬件指令支持。
- 端到端精度。

## 5. Layout

layout 是数据在内存里的排列方式。

常见：

- row-major。
- column-major。
- NCHW。
- NHWC。
- blocked layout。

layout 会影响：

- 相邻线程是否访问连续地址。
- 是否需要 transpose/layout conversion。
- 是否能直接调用 vendor library。

## 6. Stride

PyTorch tensor 有 stride。

例子：

```python
x = torch.randn(4, 8)
y = x[:, ::2]
print(y.shape)
print(y.stride())
print(y.is_contiguous())
```

`y` 不是 contiguous。它的逻辑 shape 可能简单，但内存访问跳着走。

## 7. 支持 non-contiguous 的三种策略

### 策略 1：不支持，直接报错

适合学习或内部低风险版本。

```text
优点：kernel 简单。
缺点：调用方必须保证 contiguous。
```

### 策略 2：进入 kernel 前 contiguous

```python
if not x.is_contiguous():
    x = x.contiguous()
```

优点：

- 简单。
- kernel 仍然高效。

缺点：

- 可能产生额外 copy。
- 端到端性能未必好。

### 策略 3：kernel 支持 stride

优点：

- 语义更完整。

缺点：

- 代码复杂。
- 访问可能不连续。
- 性能可能下降。

## 8. 支持范围写法

不成熟写法：

```text
支持 RMSNorm。
```

成熟写法：

```text
当前实现支持：
- input shape: [M, N]
- input dtype: FP16/FP32
- weight shape: [N]
- layout: contiguous row-major
- N <= 8192

当前不支持：
- arbitrary stride
- backward
- FP8/INT8
- N > 8192
```

## 9. 测试矩阵

| 维度 | 测试值 |
|---|---|
| batch | 1, 2, 32, 1024 |
| hidden | 127, 128, 256, 768, 1024, 4096 |
| dtype | FP32, FP16, BF16 |
| layout | contiguous, transposed, sliced |
| values | random, zeros, large, negative |

## 10. 一句话总结

```text
shape 决定并行粒度，dtype 决定数值和硬件路径，layout/stride 决定访问模式。
```

