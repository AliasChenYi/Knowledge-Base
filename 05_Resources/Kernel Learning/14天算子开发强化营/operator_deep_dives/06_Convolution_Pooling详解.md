# 06：Convolution / Pooling 详解

## 1. Convolution 是什么

2D convolution 常见于 CNN。

输入：

```text
X[N, C_in, H, W]
```

权重：

```text
W[C_out, C_in, R, S]
```

输出：

```text
Y[N, C_out, P, Q]
```

其中：

- N：batch。
- C：channel。
- H/W：输入高宽。
- R/S：卷积核高宽。
- P/Q：输出高宽。

## 2. 直接卷积

伪代码：

```text
for n in N:
  for co in C_out:
    for p in P:
      for q in Q:
        acc = 0
        for ci in C_in:
          for r in R:
            for s in S:
              acc += X[n,ci,p+r,q+s] * W[co,ci,r,s]
        Y[n,co,p,q] = acc
```

计算密集，但访问模式复杂。

## 3. im2col + GEMM

把输入展开成矩阵：

```text
X_col: [N*P*Q, C_in*R*S]
W_col: [C_in*R*S, C_out]
Y_col = X_col @ W_col
```

优点：

- 可以直接用 GEMM。

缺点：

- `im2col` 会产生巨大中间矩阵。
- 额外内存读写。

## 4. Implicit GEMM

Implicit GEMM 不显式生成 im2col 矩阵，而是在 kernel 内按 GEMM 方式组织计算。

优点：

- 利用 GEMM/Tensor Core 思路。
- 避免完整 im2col 写回 HBM。

CUTLASS 和 cuDNN 中都有类似思想。

## 5. Winograd / FFT

某些卷积可用特殊算法减少乘法：

- Winograd：常用于小 kernel，如 3x3。
- FFT：大卷积可能使用。

但这些算法复杂，数值误差和适用范围都要注意。初学先理解 direct、im2col、implicit GEMM。

## 6. Layout：NCHW vs NHWC

NCHW：

```text
[N, C, H, W]
```

NHWC：

```text
[N, H, W, C]
```

layout 影响：

- channel 是否连续。
- Tensor Core 使用。
- vectorized load/store。
- cache locality。

很多现代 GPU 上 NHWC 对 Tensor Core convolution 更友好，但具体要看库和硬件。

## 7. Stride / Padding / Dilation / Groups

卷积参数：

- stride：滑动步长。
- padding：边界补 0。
- dilation：卷积核空洞。
- groups：分组卷积。

这些都会改变索引计算和访问模式。

Depthwise convolution 是 groups = C_in 的特殊情况，计算量较低，常常更 memory-bound。

## 8. Pooling

Pooling 常见：

- max pooling。
- average pooling。

Max pooling：

```text
y = max(window)
```

Average pooling：

```text
y = mean(window)
```

它们本质上是局部 reduction。

优化关注：

- window 大小。
- overlapping。
- memory access。
- index 保存，maxpool backward 需要 argmax。

## 9. 卷积优化点

1. 选择合适 layout。
2. 使用 cuDNN/vendor library。
3. implicit GEMM。
4. Tensor Core。
5. fusion：conv + bias + activation。
6. 对 depthwise/group conv 用专门策略。
7. 避免显式 im2col 大中间矩阵。

## 10. 测试计划

必须测：

- kernel 1x1。
- kernel 3x3。
- stride 1/2。
- padding。
- dilation。
- groups。
- small batch / large batch。
- NCHW / NHWC。
- FP32/FP16/BF16。

## 11. Benchmark 计划

比较：

- PyTorch conv。
- cuDNN。
- 自己实现 direct conv。
- im2col + GEMM。

记录：

- latency。
- TFLOPS。
- effective bandwidth。
- layout。
- algorithm。

## 12. 常见坑

1. padding 边界错。
2. output shape 算错。
3. NCHW/NHWC 混淆。
4. im2col 中间矩阵太大。
5. depthwise conv 用普通 conv 策略导致低效。
6. backward 语义忽略。

## 13. 推荐资料

- NVIDIA Convolutional Layers Performance Guide。
- cuDNN Developer Guide。
- CUTLASS implicit GEMM convolution。
- oneDNN memory formats。

