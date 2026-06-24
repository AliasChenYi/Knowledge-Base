import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


def triton_softmax(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=block_size)
    return y


def main():
    torch.manual_seed(0)
    device = "cuda"
    shapes = [(1024, 128), (1024, 512), (1024, 1024), (4096, 1024)]

    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        out = triton_softmax(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

        triton_ms = triton.testing.do_bench(lambda: triton_softmax(x))
        torch_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        print(
            f"shape={shape} torch_ms={torch_ms:.4f} "
            f"triton_ms={triton_ms:.4f} speedup={torch_ms / triton_ms:.2f}"
        )


if __name__ == "__main__":
    main()

