# Compilers

## Scope
记录模型 lowering 与编译优化相关的理解与实践。

## Suggested Topics
- IR 设计与变换策略
- Pattern rewrite 与 fusion pass
- Auto-tuning 与 scheduling 策略
- Backend codegen 约束与权衡

## Core Files
- `compiler-note-template.md`：记录 compiler pass、lowering 或 codegen 观察。
- `pass-debugging-checklist.md`：定位 compile error、wrong result 和性能退化。

## Minimal Note Structure
- 输入图模式或算子模式
- Compiler pass 行为与影响
- 生成 kernel 的关键特征
- 后续优化机会（Opportunities）

## Review Questions
- pass 前后的 IR invariant 是否保持？
- 性能变化来自 pattern rewrite、fusion、layout 还是 backend codegen？
- 是否有最小 regression test？
