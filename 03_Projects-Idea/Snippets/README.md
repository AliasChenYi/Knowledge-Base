# Snippets

## Scope
存放可复用的 C++/CUDA/Python 工具片段。

## Design Rules
- 保持片段最小可运行、语义完整。
- 提供最小使用示例与边界条件说明。
- 明确平台、依赖与版本假设。

## Suggested Files
- `snippet-template.md`
- `cpp-utils.md`
- `cuda-utils.md`
- `python-utils.md`

## Promotion Rule
- 只保存至少复用过两次，或预计会在当前项目中反复使用的片段。
- 从实验脚本提取 snippet 时，去掉项目私有路径和一次性参数。
- 对涉及 GPU、网络、文件系统的片段，补充失败场景和验证命令。
