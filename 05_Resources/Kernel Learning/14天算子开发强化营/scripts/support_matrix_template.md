# 算子支持矩阵模板

算子名称：

负责人：

日期：

## 1. 支持范围

| 项 | 当前支持 | 备注 |
|---|---|---|
| device | CUDA / CPU / 其他 | |
| dtype | FP32 / FP16 / BF16 / INT8 | |
| shape rank | 2D / 3D / 4D | |
| dynamic shape | 是 / 否 | |
| contiguous | 是 / 否 | |
| non-contiguous | 是 / 否 | |
| backward | 是 / 否 | |
| deterministic | 是 / 否 | |

## 2. Shape 范围

| 维度 | 支持范围 | 常测值 |
|---|---|---|
| M / batch | | |
| N / hidden | | |
| K | | |

## 3. 测试覆盖

| 测试类型 | 是否覆盖 | 文件/命令 |
|---|---|---|
| random shape | | |
| boundary shape | | |
| dtype | | |
| extreme values | | |
| non-contiguous | | |
| benchmark | | |
| profiling | | |

## 4. 已知限制

1.
2.
3.

## 5. 下一步

1.
2.
3.

