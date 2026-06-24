# 代码实验包

这个目录是 14 天计划的动手部分。目标不是给你一套工业级高性能库，而是给你一套“能跑、能改、能测、能解释”的最小实验骨架。

建议顺序：

1. [cuda/vector_add.cu](cuda/vector_add.cu)
2. [cuda/transpose.cu](cuda/transpose.cu)
3. [cuda/reduction.cu](cuda/reduction.cu)
4. [triton/vector_add.py](triton/vector_add.py)
5. [triton/fused_softmax.py](triton/fused_softmax.py)
6. [triton/rmsnorm.py](triton/rmsnorm.py)
7. [pytorch_custom_op/README.md](pytorch_custom_op/README.md)

每个实验都要做三件事：

- correctness：和 CPU/PyTorch reference 对齐。
- benchmark：warmup、多次重复、同步计时。
- explanation：写清楚瓶颈和优化点。

## 推荐记录方式

每跑一个实验，把结果写到：

- [../templates/实验记录模板.md](../templates/实验记录模板.md)
- [../templates/benchmark_summary.csv](../templates/benchmark_summary.csv)

## 注意

这些代码是学习骨架，不承诺覆盖所有 dtype、layout、stride、边界和硬件架构。真实公司项目里，支持范围必须写清楚。

