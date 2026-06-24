import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    y = x * inv_rms * w

    tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


def rmsnorm_ref(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps)
    return (y * weight).to(x.dtype)


def triton_rmsnorm(x, weight, eps=1e-6):
    assert x.is_cuda and weight.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    assert weight.is_contiguous()
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](
        x, weight, y, n_cols, eps, BLOCK_SIZE=block_size
    )
    return y


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    shapes = [(32, 768), (64, 1024), (64, 4096), (16, 8192)]

    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=dtype)
        weight = torch.randn(shape[-1], device=device, dtype=dtype)
        ref = rmsnorm_ref(x, weight)
        out = triton_rmsnorm(x, weight)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

        triton_ms = triton.testing.do_bench(lambda: triton_rmsnorm(x, weight))
        ref_ms = triton.testing.do_bench(lambda: rmsnorm_ref(x, weight))
        print(
            f"shape={shape} ref_ms={ref_ms:.4f} "
            f"triton_ms={triton_ms:.4f} speedup={ref_ms / triton_ms:.2f}"
        )


if __name__ == "__main__":
    main()

