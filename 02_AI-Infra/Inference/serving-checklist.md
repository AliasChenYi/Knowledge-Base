# LLM Serving Checklist

## Before Deployment
- 确认模型权重、tokenizer、runtime 版本和 commit。
- 记录 GPU 型号、driver、CUDA、NCCL、Python 环境。
- 明确 SLO：TTFT、TPOT、吞吐、错误率、显存上限。
- 准备最小回放请求集，覆盖短输入、长输入、多轮和异常输入。
- 准备回滚方案：旧镜像、旧配置、旧模型路径。

## Runtime Metrics
- `request_rate`：请求到达速率。
- `queue_time`：排队等待时间。
- `time_to_first_token`：首 token 延迟。
- `time_per_output_token`：生成阶段延迟。
- `tokens_per_second`：吞吐。
- `gpu_memory_used`：显存占用。
- `kv_cache_usage`：KV cache 使用率。
- `error_rate`：超时、取消、OOM、模型错误。

## Common Tuning Knobs
- `max_num_batched_tokens`：影响吞吐和延迟，需要结合长短请求比例调。
- `max_num_seqs`：影响并发度和 KV cache 压力。
- `gpu_memory_utilization`：过高容易 OOM，过低浪费吞吐。
- Quantization：降低显存和带宽压力，但要验证质量和 kernel 支持。
- Tensor parallel / pipeline parallel：优先用实测判断通信开销是否值得。

## Debug Flow
1. 先确认错误类型：超时、OOM、空响应、质量下降还是吞吐下降。
2. 固定模型和输入，只改一个 serving 参数。
3. 用相同 replay traffic 跑 baseline 和 variant。
4. 保存配置、日志、指标截图和 commit。
5. 把结论迁移到对应的 inference note。

## Incident Notes
- 事故时间：
- 影响范围：
- 触发条件：
- 临时恢复动作：
- 根因：
- 永久修复：
- 后续验证：
