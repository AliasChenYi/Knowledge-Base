# Knowledge-Base
Research notes, AI infra optimization insights, engineering playbooks, and reusable project ideas.

## Structure
- `00_Overview.md`：仓库导览、研究兴趣与长期目标。
- `01_Papers/`：按主题分类的论文阅读笔记。
- `02_AI-Infra/`：Kernel、Compiler、Inference 的工程沉淀。
- `03_Projects-Idea/`：Idea 验证流水线与可复用代码片段。
- `04_Learning-Log/`：周记、会议记录与阶段复盘。
- `05_Resources/`：高质量 Blog、Repo、Course、Tool 链接。
- `06_Tech-Snippets/`：命令行与操作类速查笔记。

## Quick Entry Points
- 读论文：从 `01_Papers/paper-template.md` 开始。
- 做性能优化：从 `02_AI-Infra/Kernels/profiling-checklist.md` 或 `02_AI-Infra/Inference/serving-checklist.md` 开始。
- 做 idea 验证：从 `03_Projects-Idea/Research-Pipeline/experiment-log-index.md` 开始。
- 配服务器：从 `06_Tech-Snippets/linux-server.md` 和 `06_Tech-Snippets/gpu-server-checklist.md` 开始。

## Suggested Workflow
1. 每读完一篇论文，在 `01_Papers/` 写结构化笔记。
2. 将可落地的方法沉淀到 `02_AI-Infra/` 的工程笔记中。
3. 把想法放入 `03_Projects-Idea/Research-Pipeline/` 做最小验证。
4. 每周在 `04_Learning-Log/` 记录进展、阻塞与下一步计划。
5. 持续维护 `05_Resources/`，只保留高价值可复用资源。
6. 将常用命令、部署过程、排障结论写入 `06_Tech-Snippets/`。

## Knowledge Loop
1. `Input`：论文、博客、源码、实验现象、会议讨论。
2. `Distill`：用模板记录问题、方法、证据、限制与可复用点。
3. `Connect`：把论文洞察链接到工程笔记、实验计划和代码片段。
4. `Act`：每条重要结论落到下一步实验、工具脚本或项目决策。
5. `Review`：每周清理重复内容，每月更新方向判断和资源优先级。

## Writing Rules
- 每篇笔记都要回答三个问题：解决什么问题、证据是什么、下一步做什么。
- 优先沉淀可复用经验，少保存一次性聊天记录或原始流水账。
- 外部链接必须补充一行价值说明，避免知识库变成书签夹。
- 服务地址、API key、订阅链接、私钥路径等敏感信息只写占位符，不写真实值。

## Weekly Maintenance Checklist
- 合并重复笔记，补充双向链接。
- 给未完成条目标记 `Next Action` 和负责人/时间。
- 删除已经失效的资源或补充失效原因。
- 将实验结论迁移到对应的论文、工程或项目目录。
