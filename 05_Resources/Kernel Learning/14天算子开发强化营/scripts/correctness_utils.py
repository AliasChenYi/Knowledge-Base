"""Correctness helpers for tensor operator experiments."""


def tensor_error_report(out, ref):
    """Return max/mean absolute and relative error for PyTorch tensors."""
    import torch

    out_f = out.float()
    ref_f = ref.float()
    diff = (out_f - ref_f).abs()
    denom = ref_f.abs().clamp_min(1e-12)
    rel = diff / denom
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
        "has_nan_out": torch.isnan(out_f).any().item(),
        "has_inf_out": torch.isinf(out_f).any().item(),
        "has_nan_ref": torch.isnan(ref_f).any().item(),
        "has_inf_ref": torch.isinf(ref_f).any().item(),
    }


def assert_close_with_report(out, ref, rtol=1e-5, atol=1e-6, name="output"):
    import torch

    report = tensor_error_report(out, ref)
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    if not ok:
        print(f"{name} mismatch")
        for key, value in report.items():
            print(f"  {key}: {value}")
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    return report

