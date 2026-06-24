"""Small benchmark helpers for PyTorch/Triton experiments.

This is a learning template. For production, follow your team's benchmark
harness and reporting rules.
"""

import statistics
import time


def _sync_if_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def time_ms(fn, warmup=20, repeat=100):
    """Return a dict with median/mean/min/max latency in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync_if_cuda()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        _sync_if_cuda()
        end = time.perf_counter()
        samples.append((end - start) * 1000.0)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeat": repeat,
    }


def effective_bandwidth_gbps(bytes_moved, ms):
    return bytes_moved / (ms * 1e-3) / 1e9


def gemm_tflops(m, n, k, ms):
    flops = 2.0 * m * n * k
    return flops / (ms * 1e-3) / 1e12


def print_result(name, result):
    fields = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items())
    print(f"{name}: {fields}")

