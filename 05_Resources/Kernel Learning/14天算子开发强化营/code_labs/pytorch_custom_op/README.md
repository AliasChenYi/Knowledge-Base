# PyTorch Custom Op 实验骨架

这个目录先给出最小工程思路。真实 custom op 的编译配置会随 PyTorch/CUDA/公司内部工具链变化，建议你先把链路理解清楚，再照官方教程落地。

官方教程：[PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

## 目标

把一个算子封装成：

```python
torch.ops.my_ops.rms_norm(x, weight, eps)
```

最小闭环：

```text
schema -> CPU reference -> CUDA/Triton backend -> Python wrapper -> tests -> benchmark
```

## 推荐目录

```text
my_rmsnorm_op/
  README.md
  setup.py
  my_rmsnorm.cpp
  my_rmsnorm_kernel.cu
  my_rmsnorm.py
  tests/
    test_correctness.py
  benchmarks/
    bench_rmsnorm.py
```

## Python wrapper 示例

```python
import torch


def rmsnorm_ref(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps)
    return (y * weight).to(x.dtype)


def rms_norm(x, weight, eps=1e-6):
    if not x.is_cuda:
        return rmsnorm_ref(x, weight, eps)
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return torch.ops.my_ops.rms_norm(x, weight, eps)
```

## C++ schema 思路

示意，不保证直接可编译：

```cpp
TORCH_LIBRARY(my_ops, m) {
  m.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("rms_norm", rms_norm_cpu);
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("rms_norm", rms_norm_cuda);
}
```

## 测试重点

```text
shape: [1,128], [32,768], [64,4096]
dtype: fp32, fp16, bf16
layout: contiguous, non-contiguous
values: random, zeros, large values
device: cpu, cuda
```

## 你要写进 README 的支持范围

示例：

```text
当前版本支持：
- CUDA input
- contiguous [M, N]
- dtype: fp16/fp32
- weight shape: [N]

当前版本不支持：
- arbitrary stride
- backward
- dynamic rank
```

把“不支持”写清楚，是成熟工程习惯。

