# GPU Server Checklist

## Basic Checks
```bash
nvidia-smi
nvidia-smi topo -m
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Runtime State
- GPU 利用率是否持续接近 0。
- 显存是否被旧进程占用。
- 温度、功耗、显存 ECC 是否异常。
- 多卡任务的进程是否分布在预期 GPU 上。

## Process Mapping
```bash
nvidia-smi
ps -fp <PID>
```

如果需要释放显存，先确认进程用途和 owner，再处理：

```bash
kill <PID>
```

## Environment Variables
```bash
echo "$CUDA_VISIBLE_DEVICES"
echo "$LD_LIBRARY_PATH"
echo "$PATH"
```

## Common Issues
- `CUDA out of memory`：检查 batch size、KV cache、旧进程和显存碎片。
- `CUDA driver version is insufficient`：driver 与 runtime 版本不匹配。
- `invalid device ordinal`：`CUDA_VISIBLE_DEVICES` 和代码中的 device id 不一致。
- 多卡通信卡住：检查 NCCL 环境变量、网络、拓扑和防火墙。

## Experiment Record
- GPU：
- Driver / CUDA：
- Framework：
- Command：
- Peak memory：
- Runtime：
- Notes：
