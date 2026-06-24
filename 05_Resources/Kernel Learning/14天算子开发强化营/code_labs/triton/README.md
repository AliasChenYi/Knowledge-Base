# Triton 实验

## 运行前检查

```powershell
python -c "import torch, triton; print(torch.__version__); print(triton.__version__); print(torch.cuda.get_device_name(0))"
```

## 运行示例

```powershell
python vector_add.py
python fused_softmax.py
python rmsnorm.py
```

## 学习目标

1. `vector_add.py`：理解 `program_id`、`tl.arange`、mask。
2. `fused_softmax.py`：理解 row-wise reduction 和融合。
3. `rmsnorm.py`：理解 norm 类算子的 FP32 accumulate。

## 重要提醒

Triton 代码看起来像 Python/NumPy，但它不是普通 Python 循环。`@triton.jit` 下的函数会被编译成设备 kernel。

