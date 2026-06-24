# Inference

## Scope
面向生产场景的推理笔记：部署架构、延迟、吞吐与成本。

## Suggested Topics
- Quantization 与 calibration 的 trade-off
- KV cache 管理策略
- Batching 与 scheduling 策略
- Serving 可观测性与 SLO 设计

## Core Files
- `inference-note-template.md`：记录单次推理服务优化或问题排查。
- `serving-checklist.md`：部署前检查、运行指标、调参项和事故记录。

## Minimal Note Structure
- 场景背景（Serving context）
- 优化动作（Optimization action）
- 指标影响（Metric impact）
- 风险与回滚方案（Rollback plan）

## Review Questions
- 当前瓶颈是 prefill、decode、调度、KV cache 还是网络/队列？
- 优化动作对 TTFT、TPOT、吞吐和显存分别有什么影响？
- 是否有明确 rollback plan 和 replay traffic？
