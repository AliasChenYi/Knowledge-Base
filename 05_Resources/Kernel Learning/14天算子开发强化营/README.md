# 14 天算子开发强化营目录

建议先按这个顺序打开：

1. [14天详细学习计划.md](14天详细学习计划.md)：总路线图。
2. [daily_guides/README.md](daily_guides/README.md)：逐日超详细讲义入口。
3. [入职前知识地图.md](入职前知识地图.md)：从基础到热点算子的能力树。
4. [每日学习时间安排.md](每日学习时间安排.md)：2 小时版和 5 小时版学习节奏。
5. [环境准备与工具链.md](环境准备与工具链.md)：工具链、版本记录和调试建议。
6. [术语速查卡_扩展版.md](术语速查卡_扩展版.md)：遇到术语随时查。

## 核心学习材料

- 总计划：[14天详细学习计划.md](14天详细学习计划.md)
- 逐日超详细讲义：[daily_guides](daily_guides)
- 每日配图：[assets](assets)
- 已下载论文：[papers](papers)
- 论文中文讲解：[paper_notes](paper_notes)
- 可复用模板：[templates](templates)
- 代码实验包：[code_labs](code_labs)
- 排错手册：[debug_notes](debug_notes)
- 打卡与验收清单：[checklists](checklists)
- 面试/自测题库：[interview_bank](interview_bank)
- 补充专题：[supplements](supplements)
- 真实工作流：[workflow](workflow)
- 脚本模板：[scripts](scripts)
- 随身速查卡：[reference_cards](reference_cards)
- 算子专题详解：[operator_deep_dives](operator_deep_dives)
- 算子学习资料扩展库：[learning_resources](learning_resources)

## 已下载论文

1. [Roofline Model](papers/01_Roofline_EECS-2008-134.pdf)
2. [NVIDIA Optimizing Parallel Reduction](papers/02_NVIDIA_Optimizing_Parallel_Reduction.pdf)
3. [Triton MAPL 2019](papers/03_Triton_MAPL2019.pdf)
4. [FlashAttention NeurIPS 2022](papers/04_FlashAttention_NeurIPS2022.pdf)
5. [FlashAttention-2](papers/05_FlashAttention2.pdf)
6. [FlashAttention-3](papers/06_FlashAttention3.pdf)

## 中文讲解

1. [Roofline 中文讲解](paper_notes/01_Roofline_中文讲解.md)
2. [NVIDIA Reduction 中文讲解](paper_notes/02_NVIDIA_Reduction_中文讲解.md)
3. [Triton 中文讲解](paper_notes/03_Triton_中文讲解.md)
4. [FlashAttention 系列中文讲解](paper_notes/04_FlashAttention系列_中文讲解.md)

## 逐日讲义

1. [Day 01：硬件地图与内存层次](daily_guides/day01_硬件地图与内存层次.md)
2. [Day 02：Roofline 与性能瓶颈判断](daily_guides/day02_Roofline与性能瓶颈判断.md)
3. [Day 03：Vector Add 与正确计时](daily_guides/day03_VectorAdd与正确计时.md)
4. [Day 04：Transpose 与访存合并](daily_guides/day04_Transpose与访存合并.md)
5. [Day 05：Reduction 树形归约](daily_guides/day05_Reduction树形归约.md)
6. [Day 06：Softmax、LayerNorm 与数值稳定](daily_guides/day06_Softmax_LayerNorm与数值稳定.md)
7. [Day 07：PyTorch Custom Op 接入](daily_guides/day07_PyTorch_CustomOp接入.md)
8. [Day 08：Triton 编程模型入门](daily_guides/day08_Triton编程模型入门.md)
9. [Day 09：Triton Fused Softmax](daily_guides/day09_Triton_FusedSoftmax.md)
10. [Day 10：Triton Matmul 与 LayerNorm](daily_guides/day10_Triton_Matmul与LayerNorm.md)
11. [Day 11：GEMM、Tensor Core 与分块](daily_guides/day11_GEMM_TensorCore与分块.md)
12. [Day 12：CUTLASS 与 CuTe 入门](daily_guides/day12_CUTLASS与CuTe入门.md)
13. [Day 13：Attention 与 FlashAttention](daily_guides/day13_Attention与FlashAttention.md)
14. [Day 14：整理入职交付包](daily_guides/day14_整理入职交付包.md)

## 模板

- [实验记录模板](templates/实验记录模板.md)
- [性能分析报告模板](templates/性能分析报告模板.md)
- [论文阅读模板](templates/论文阅读模板.md)
- [benchmark_summary.csv](templates/benchmark_summary.csv)

## 代码实验包

- [代码实验包总入口](code_labs/README.md)
- [CUDA 实验说明](code_labs/cuda/README.md)
- [CUDA vector add](code_labs/cuda/vector_add.cu)
- [CUDA transpose](code_labs/cuda/transpose.cu)
- [CUDA reduction](code_labs/cuda/reduction.cu)
- [Triton 实验说明](code_labs/triton/README.md)
- [Triton vector add](code_labs/triton/vector_add.py)
- [Triton fused softmax](code_labs/triton/fused_softmax.py)
- [Triton RMSNorm](code_labs/triton/rmsnorm.py)
- [PyTorch custom op 实验骨架](code_labs/pytorch_custom_op/README.md)

## 排错与讲解

- [算子开发排错手册](debug_notes/算子开发排错手册.md)
- [代码实验逐行讲解](debug_notes/代码实验逐行讲解.md)

## 打卡、验收与面试

- [14 天每日打卡表](checklists/14天每日打卡表.md)
- [算子验收清单](checklists/算子验收清单.md)
- [错题本模板](checklists/错题本模板.md)
- [算子开发面试题：入门到进阶](interview_bank/算子开发面试题_入门到进阶.md)

## 补充专题

- [补充专题总入口](supplements/README.md)
- [Profiling 实战手册](supplements/profiling实战手册.md)
- [数值精度与误差分析](supplements/数值精度与误差分析.md)
- [量化算子入门](supplements/量化算子入门.md)
- [多后端与芯片平台迁移地图](supplements/多后端与芯片平台迁移地图.md)
- [通信与分布式算子入门](supplements/通信与分布式算子入门.md)
- [入职 30-60-90 天行动计划](supplements/入职30_60_90天行动计划.md)
- [四个进阶小项目任务书](supplements/进阶小项目任务书.md)

## 真实工作流

- [真实工作流总入口](workflow/README.md)
- [算子任务拆解手册](workflow/算子任务拆解手册.md)
- [Shape-Dtype-Layout 专题](workflow/Shape_Dtype_Layout专题.md)
- [Benchmark 设计手册](workflow/Benchmark设计手册.md)
- [代码评审自查清单](workflow/代码评审自查清单.md)
- [周报与汇报模板](workflow/周报与汇报模板.md)
- [入职沟通话术](workflow/入职沟通话术.md)

## 脚本模板

- [脚本模板总入口](scripts/README.md)
- [环境报告脚本](scripts/env_report.py)
- [Benchmark 工具](scripts/benchmark_utils.py)
- [Correctness 工具](scripts/correctness_utils.py)
- [Shape sweep 模板](scripts/shape_sweep_template.py)
- [算子支持矩阵模板](scripts/support_matrix_template.md)

## 随身速查卡

- [速查卡总入口](reference_cards/README.md)
- [英中术语对照表](reference_cards/英中术语对照表.md)
- [常用命令速查卡](reference_cards/常用命令速查卡.md)
- [学习自评表](reference_cards/学习自评表.md)
- [7 天复习计划](reference_cards/7天复习计划.md)
- [算子优化决策树](reference_cards/算子优化决策树.md)

## 算子学习资料扩展库

- [学习资料扩展库总入口](learning_resources/README.md)
- [算子学习资料库：扩展版](learning_resources/算子学习资料库_扩展版.md)
- [按算子类型阅读路线](learning_resources/按算子类型阅读路线.md)
- [官方文档与工程博客索引](learning_resources/官方文档与工程博客索引.md)
- [论文阅读路线：从入门到资深](learning_resources/论文阅读路线_从入门到资深.md)
- [中文资料使用建议](learning_resources/中文资料使用建议.md)

## 算子专题详解

- [算子专题详解总入口](operator_deep_dives/README.md)
- [00：算子学习方法论](operator_deep_dives/00_算子学习方法论.md)
- [01：Elementwise / Broadcast / Fusion 详解](operator_deep_dives/01_Elementwise_Broadcast_Fusion详解.md)
- [02：Reduction / Scan / Sort 详解](operator_deep_dives/02_Reduction_Scan_Sort详解.md)
- [03：Softmax / LogSoftmax 详解](operator_deep_dives/03_Softmax_LogSoftmax详解.md)
- [04：LayerNorm / RMSNorm / BatchNorm 详解](operator_deep_dives/04_LayerNorm_RMSNorm_BatchNorm详解.md)
- [05：GEMM / Batched GEMM / GEMV 详解](operator_deep_dives/05_GEMM_BatchedGEMM_GEMV详解.md)
- [06：Convolution / Pooling 详解](operator_deep_dives/06_Convolution_Pooling详解.md)
- [07：Attention / KV Cache / FlashAttention 详解](operator_deep_dives/07_Attention_KVCache_FlashAttention详解.md)
- [08：Quantization / 低精度算子详解](operator_deep_dives/08_Quantization_低精度算子详解.md)
- [09：编译器视角看算子优化](operator_deep_dives/09_编译器视角看算子优化.md)
