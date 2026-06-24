# Compiler Pass Debugging Checklist

## Repro
- 记录 framework、compiler、backend、commit 和 flags。
- 保存最小模型或 IR 片段。
- 固定输入 shape、dtype 和随机种子。
- 明确 expected output 和 observed output。

## Dump Artifacts
- Before pass IR：
- After pass IR：
- Generated code：
- Runtime log：
- Benchmark command：

## Bisect Strategy
1. 先区分是 compile error、wrong result 还是 performance regression。
2. 关闭可疑 pass，确认问题是否消失。
3. 对比 pass 前后的 IR invariants。
4. 缩小到单个 op pattern 或 rewrite rule。
5. 添加最小 regression test。

## Invariants To Check
- Shape 和 dtype 没有被错误改写。
- Memory aliasing 没有破坏读写顺序。
- Layout transform 有成对的 producer / consumer 更新。
- Fusion 后没有引入非法 side effect。
- Backend codegen 约束仍然满足。

## Notes
- 性能退化不一定来自最后一个 pass，可能来自早期 canonicalization 改变了 pattern。
- Debug 时优先保留可复现命令，而不是只记录现象。
