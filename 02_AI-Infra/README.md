# 02 AI Infra

## Scope
本目录沉淀你在底层优化与部署方向的核心工程经验。

## Subdirectories
- `Kernels/`：算子级优化与硬件感知实现。
- `Compilers/`：MLIR、TVM、Triton 与编译优化逻辑。
- `Inference/`：服务架构、量化策略与性能调优。

## Engineering Note Contract
- `Context`：模型、硬件、框架、输入规模、版本。
- `Bottleneck`：可观测证据，例如 profiler、日志、指标或复现实验。
- `Action`：具体优化动作，不只写结论。
- `Result`：至少包含 baseline、variant、metric、trade-off。
- `Reuse`：说明该经验适合迁移到哪些场景，不适合哪些场景。

## Writing Standard
- 先描述瓶颈，再解释优化策略。
- 记录测量方法与前后对比数据。
- 总结可迁移经验，避免一次性笔记。

## Cross-Linking
- 论文来源链接到 `01_Papers/`。
- 实验计划链接到 `03_Projects-Idea/Research-Pipeline/`。
- 可复用命令或部署步骤下沉到 `06_Tech-Snippets/`。
