import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x, y, block_size=1024):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"
    n = 1 << 24
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)

    ref = x + y
    out = triton_add(x, y)
    torch.testing.assert_close(out, ref)

    for block_size in [256, 512, 1024, 2048]:
        ms = triton.testing.do_bench(
            lambda: triton_add(x, y, block_size=block_size)
        )
        bytes_moved = n * 3 * x.element_size()
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(
            f"N={n} BLOCK_SIZE={block_size} ms={ms:.4f} "
            f"effective_GB/s={gbps:.2f}"
        )


if __name__ == "__main__":
    main()

