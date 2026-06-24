"""Shape sweep template.

Replace `reference_fn` and `candidate_fn` with your operator implementations.
"""

import torch

from benchmark_utils import time_ms
from correctness_utils import assert_close_with_report


def reference_fn(x):
    return torch.softmax(x, dim=-1)


def candidate_fn(x):
    # Replace this with your Triton/CUDA/custom op.
    return torch.softmax(x, dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    shapes = [
        (1024, 128),
        (1024, 512),
        (1024, 1024),
        (4096, 1024),
        (4096, 4096),
    ]

    print("shape,dtype,ref_ms,candidate_ms,speedup,max_abs,max_rel")
    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=dtype)
        ref = reference_fn(x)
        out = candidate_fn(x)
        report = assert_close_with_report(out, ref, rtol=1e-5, atol=1e-6)

        ref_ms = time_ms(lambda: reference_fn(x), warmup=10, repeat=50)["median_ms"]
        cand_ms = time_ms(lambda: candidate_fn(x), warmup=10, repeat=50)["median_ms"]
        speedup = ref_ms / cand_ms if cand_ms > 0 else float("inf")
        print(
            f"{shape},{dtype},{ref_ms:.6f},{cand_ms:.6f},{speedup:.4f},"
            f"{report['max_abs']:.6e},{report['max_rel']:.6e}"
        )


if __name__ == "__main__":
    main()

