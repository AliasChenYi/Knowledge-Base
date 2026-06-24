"""Print a small environment report for benchmark records.

Usage:
    python env_report.py
"""

import platform
import sys


def try_import(name):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None


def main():
    print("## Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")

    torch = try_import("torch")
    if torch is None:
        print("PyTorch: not installed")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA: {getattr(torch.version, 'cuda', None)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  capability: {props.major}.{props.minor}")
                print(f"  memory GB: {props.total_memory / 1024**3:.2f}")

    triton = try_import("triton")
    if triton is None:
        print("Triton: not installed")
    else:
        print(f"Triton: {triton.__version__}")


if __name__ == "__main__":
    main()

