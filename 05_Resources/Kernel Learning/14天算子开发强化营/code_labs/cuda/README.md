# CUDA 实验

## 编译示例

如果你的机器有 CUDA Toolkit：

```powershell
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add.exe
nvcc -O3 -arch=sm_80 transpose.cu -o transpose.exe
nvcc -O3 -arch=sm_80 reduction.cu -o reduction.exe
```

`sm_80` 只是示例，实际要按你的 GPU 架构调整。

常见架构参考：

```text
Turing: sm_75
Ampere A100: sm_80
Ampere RTX 30: sm_86
Ada RTX 40: sm_89
Hopper H100: sm_90
```

## 运行建议

每次运行记录：

```text
GPU:
CUDA:
Driver:
Command:
Output:
```

## 学习顺序

1. `vector_add.cu`：线程映射、边界、计时。
2. `transpose.cu`：coalescing、shared memory、bank conflict。
3. `reduction.cu`：block 内归约、row-wise sum/max。

