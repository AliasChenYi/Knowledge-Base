# Day 07：PyTorch Custom Op 接入

![PyTorch Custom Op 接入](../assets/day07_pytorch_custom_op.svg)

## 今天的核心结论

写 kernel 只是第一步。真实工作里，你还要让框架能调用它、测试它、选择正确后端、处理 dtype/device/shape。

今天的目标是建立这条链路：

```text
Python API -> op schema -> dispatcher -> CPU reference / CUDA backend -> tests -> benchmark
```

## 今天你要完成什么

最低 2 小时版：

1. 读 PyTorch custom op 官方教程。
2. 理解 schema、CPU fallback、CUDA backend 的关系。
3. 跑通官方最小例子或写出项目结构。

完整版 4-6 小时：

1. 把 Day 6 的 RMSNorm 或 softmax 包成 PyTorch custom op。
2. 写 CPU reference。
3. 写 CUDA/Triton backend。
4. 写 correctness tests。

## 必读资料与读法

1. [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)  
   可靠性：A。今天主读。
2. [Registering a Dispatched Operator in C++](https://docs.pytorch.org/tutorials/advanced/dispatcher.html)  
   可靠性：A。理解 dispatcher。
3. [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)  
   可靠性：A。今天只浏览。

## 概念讲解：什么是 op schema

schema 描述一个算子的接口：

```text
rms_norm(Tensor x, Tensor weight, float eps) -> Tensor
```

它回答：

- 输入有哪些？
- 输出是什么？
- 参数类型是什么？
- 框架如何找到这个 op？

schema 不负责性能，它负责语义和注册。

## 概念讲解：CPU fallback 为什么重要

CPU fallback 的意义：

1. 提供 reference，方便测试。
2. CUDA 不可用时仍能跑。
3. 帮助你区分“语义错”还是“GPU kernel 错”。

初学阶段，CPU fallback 可以直接调用 PyTorch reference：

```python
def rms_norm_ref(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps)).to(x.dtype) * weight
```

## 推荐项目结构

```text
my_rmsnorm_op/
  README.md
  setup.py 或 CMakeLists.txt
  my_rmsnorm.cpp
  my_rmsnorm_kernel.cu
  my_rmsnorm.py
  tests/
    test_correctness.py
    test_opcheck.py
  benchmarks/
    bench_rmsnorm.py
```

## Python 调用层设计

建议给用户一个干净的 Python API：

```python
def rms_norm(x, weight, eps=1e-6):
    if not x.is_cuda:
        return rms_norm_ref(x, weight, eps)
    return torch.ops.my_ops.rms_norm(x, weight, eps)
```

好处：

- 用户不必直接记 `torch.ops...`。
- 你可以在 Python 层做 dtype/device 检查。
- 未来可加 fallback。

## 测试清单

必须测：

| 类型 | 例子 |
|---|---|
| shape | `[1,128]`, `[32,768]`, `[8,4096]` |
| dtype | fp32, fp16, bf16 |
| device | cpu, cuda |
| contiguous | `x.contiguous()` |
| non-contiguous | `x[:, ::2]` 或 transpose |
| 极值 | 全 0、大正数、大负数 |
| 随机 | 固定 seed 多组 shape |

如果你的 kernel 只支持 contiguous，要明确写：

```text
当前版本只支持 contiguous input；非 contiguous 输入会先 contiguous 或直接报错。
```

不要假装支持。

## benchmark 应该怎么写

至少比较：

- PyTorch reference。
- 你的 custom op。
- 如果有 Triton 版本，也一起比较。

注意：

- warmup。
- synchronize。
- median。
- 多 shape。
- 记录硬件和软件版本。

## 常见错误排查

### Python 找不到 op

可能原因：

- 扩展没有编译。
- 没有 import 触发动态库加载。
- namespace 写错。
- schema 名字和调用名字不一致。

### CUDA 报错位置很奇怪

可能原因：

- CUDA 异步执行，错误延迟暴露。

调试时加：

```python
torch.cuda.synchronize()
```

或设置：

```text
CUDA_LAUNCH_BLOCKING=1
```

### opcheck 不通过

可能原因：

- schema 不准确。
- dtype/device 行为不一致。
- mutation/aliasing 标注有问题。
- fake/meta 行为缺失。

## 自测题

1. schema 解决什么问题？
2. CPU fallback 有什么价值？
3. dispatcher 是干什么的？
4. custom op 为什么要测 non-contiguous？
5. benchmark custom op 时为什么要和 PyTorch reference 比？

参考答案：

1. 描述 op 接口和类型。
2. 提供 reference、可测试、可 fallback。
3. 根据 device/dtype/backend 选择实现。
4. 框架 tensor 可能不是连续布局。
5. 需要 baseline，判断性能和正确性。

## 今天的验收标准

你今天合格，如果你能说清楚：

```text
一个生产可用算子不只是 kernel，还包括 schema、后端注册、reference、测试、benchmark 和清晰的支持范围。
```

