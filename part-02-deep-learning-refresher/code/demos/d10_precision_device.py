"""Demo 10 - number formats (precision) and choosing where the numbers are computed (device).

What it does: compares float32, bfloat16 and float16 (range, smallest step, how pi is rounded), picks the
best available device (cuda, mps or cpu) and times matrix multiplications of three sizes on it.
What it needs: torch.
What to expect: bfloat16 keeps float32's range with fewer digits (pi becomes 3.140625); the timing lines show
that a graphics chip gains more and more as the matrices grow.
Run it: uv run python -m demos.d10_precision_device
"""
import math
import time

import torch

from dl_refresher.report import run_and_save, save_results


def pick_device() -> str:
    """cuda if an Nvidia graphics chip is present, mps for the graphics chip in some laptops, otherwise cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    """Wait until the device has finished its queued work, so timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_matmul(device: str, size: int, repeats: int = 10) -> float:
    """Average seconds for one (size x size) matrix multiplication on the device."""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    a @ b                                                # warm-up run, not timed
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    sync(device)
    return (time.perf_counter() - start) / repeats


def main():
    print("number formats")
    print(f"{'format':<10} {'bits':>5} {'largest':>12} {'smallest step near 1':>22} {'pi stored as':>14}")
    formats = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        info = torch.finfo(dtype)
        pi_stored = torch.tensor(math.pi, dtype=dtype).item()
        formats[str(dtype)] = {"bits": info.bits, "max": info.max, "eps": info.eps, "pi": pi_stored}
        print(f"{str(dtype).replace('torch.', ''):<10} {info.bits:>5} {info.max:>12.3e} {info.eps:>22.3e} {pi_stored:>14.7f}")
    print("1.001 stored in bfloat16 :", torch.tensor(1.001, dtype=torch.bfloat16).item(), "(the extra 0.001 is rounded away)")
    print("1000 + 1 in float16      :", (torch.tensor(1000.0, dtype=torch.float16) + torch.tensor(1.0, dtype=torch.float16)).item())
    print("70000 in float16         :", torch.tensor(70000.0).to(torch.float16).item(), "(too large for float16)")
    print()
    print("mixed precision on the cpu (autocast runs the matrix multiplications in bfloat16)")
    try:
        layer = torch.nn.Linear(2, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(torch.randn(4, 2))
        print("output dtype inside autocast:", str(out.dtype).replace("torch.", ""))
    except Exception as error:                           # older builds may not offer cpu autocast
        print("cpu autocast not available here:", type(error).__name__)
    print()
    device = pick_device()
    print(f"best available device: {device}")
    timings = {}
    for size in (256, 1024, 4096):
        cpu_seconds = time_matmul("cpu", size)
        line = f"{size} x {size} matrix multiplication: cpu {cpu_seconds * 1000:8.2f} ms"
        timings[str(size)] = {"cpu_ms": cpu_seconds * 1000}
        if device != "cpu":
            fast_seconds = time_matmul(device, size)
            timings[str(size)]["device_ms"] = fast_seconds * 1000
            line += f"   {device} {fast_seconds * 1000:8.2f} ms   ({cpu_seconds / fast_seconds:.1f}x)"
        print(line)
    save_results("d10_precision_device", {"formats": formats, "device": device, "timings": timings})


if __name__ == "__main__":
    run_and_save("d10_precision_device", main)
