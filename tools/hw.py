#!/usr/bin/env python3
"""Measure the facts of THIS GPU that decide what is fast. Run once per machine.

Writes hardware/<device>.json and prints a summary. Datasheet numbers are wrong for
cut-down parts (H20 & co.), so everything here is measured.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from common import HARDWARE, slug, write_json  # noqa: E402


def best_ms(fn, iters=10, rounds=5) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def run_snippet(code: str, env=None, timeout=600) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory(prefix="krsi-hw-") as tmp:
        f = Path(tmp) / "snippet.py"  # @triton.jit needs a real source file
        f.write_text(code)
        return subprocess.run([sys.executable, "-I", str(f)], text=True, capture_output=True, timeout=timeout,
                              env=dict(os.environ, TRITON_CACHE_DIR=str(Path(tmp) / "cache"), **(env or {})))


TRITON_PROBE = r"""
import torch, triton, triton.language as tl
@triton.jit
def k(x, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x + o, tl.load(x + o, mask=o < n) * 2, mask=o < n)
x = torch.ones(4096, device="cuda")
k[(4,)](x, 4096, B=1024); torch.cuda.synchronize(); assert float(x.sum()) == 8192.0
def t(fn, it=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000
y = torch.empty(1024, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    k[(1,)](y, 1024, B=1024)
print(t(lambda: k[(1,)](y, 1024, B=1024)), t(lambda: y.add_(1)), t(g.replay))
"""


def ncu_probe() -> str:
    ncu = shutil.which("ncu") or ("/usr/local/cuda/bin/ncu" if Path("/usr/local/cuda/bin/ncu").is_file() else None)
    if not ncu:
        return "unavailable: ncu not found"
    code = "import torch\nx = torch.randn(1 << 20, device='cuda')\ny = x * 2\ntorch.cuda.synchronize()\n"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.py"
        f.write_text(code)
        try:
            p = subprocess.run([ncu, "--set", "basic", "-c", "1", sys.executable, "-I", str(f)],
                               text=True, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            return f"unavailable: {ncu} timed out"
    out = p.stdout + p.stderr
    if "ERR_NVGPUCTRPERM" in out:
        return (f"unavailable: {ncu} has no counter permission (ERR_NVGPUCTRPERM). Host fix: driver option "
                "NVreg_RestrictProfilingToAdminUsers=0, or run the container with --cap-add SYS_ADMIN")
    if p.returncode != 0 or "No kernels were profiled" in out:
        return f"unavailable: {out[-300:]}"
    return f"available: {ncu}"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    cap = (major, minor)
    small = props.total_memory < (12 << 30)
    n = 4096 if small else 8192

    def gemm_tflops(dtype, iters=10):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        t = 2 * n ** 3 / best_ms(lambda: a @ b, iters=iters) / 1e9
        del a, b
        return round(t, 1)

    # Dense GEMM peaks per dtype as cuBLAS reaches them on this card (the achievable ceiling).
    tflops = {"fp16": gemm_tflops(torch.float16)}
    if cap >= (8, 0):
        tflops["bf16"] = gemm_tflops(torch.bfloat16)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        tflops["fp32"] = gemm_tflops(torch.float32, iters=3)
        if cap >= (8, 0):
            torch.backends.cuda.matmul.allow_tf32 = True
            tflops["tf32"] = gemm_tflops(torch.float32, iters=3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    if cap >= (8, 9):
        try:
            a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
            one = torch.ones((), device="cuda")
            tflops["fp8"] = round(2 * n ** 3 / best_ms(
                lambda: torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)) / 1e9, 1)
            del a, b
        except Exception:  # noqa: BLE001 - fp8 GEMM API differs across torch versions
            pass
    torch.cuda.empty_cache()
    numel = min(512 << 20, props.total_memory // 16)        # <= 1 GiB per fp16 buffer, <= 1/8 of memory
    src = torch.randn(numel, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    copy = 4 * numel / best_ms(lambda: dst.copy_(src)) / 1e6
    read = 2 * numel / best_ms(lambda: src.sum(dtype=torch.float32)) / 1e6
    write = 2 * numel / best_ms(lambda: dst.fill_(1.0)) / 1e6
    del src, dst
    torch.cuda.empty_cache()

    half = "bf16" if "bf16" in tflops else "fp16"
    bw = max(copy, read)
    features = {"bf16_tensor_cores": cap >= (8, 0), "tf32": cap >= (8, 0), "fp8": cap >= (8, 9),
                "hopper_wgmma_tma": cap >= (9, 0)}
    info = {"device": props.name, "sm": f"sm_{major}{minor}", "sms": props.multi_processor_count,
            "l2_mb": round(getattr(props, "L2_cache_size", 0) / 2 ** 20, 1),
            "memory_gb": round(props.total_memory / 2 ** 30, 1), "tflops": tflops,
            "bw_copy_gbs": round(copy), "bw_read_gbs": round(read), "bw_write_gbs": round(write),
            "ridge_flop_per_byte": {k: round(v * 1e12 / (bw * 1e9), 1) for k, v in tflops.items()},
            "default_half_dtype": half, "features": features}
    try:
        import triton  # noqa: PLC0415

        p = triton.runtime.driver.active.utils.get_device_properties(0)
        info.update(smem_per_block_kb=p["max_shared_mem"] // 1024, regs_per_sm=p["max_num_regs"],
                    sm_clock_mhz=p["sm_clock_rate"] // 1000, triton=triton.__version__)
    except Exception as e:  # noqa: BLE001
        info["triton"] = f"import failed: {e}"
    probe = run_snippet(TRITON_PROBE)
    if probe.returncode == 0:
        tri, torch_op, graph = (float(v) for v in probe.stdout.split())
        info.update(launch_us_triton=round(tri, 1), launch_us_torch=round(torch_op, 1),
                    launch_us_graph_replay=round(graph, 1))
    else:
        err = probe.stderr
        hint = (" Triton cannot build its launcher because Python.h is missing: install python3-dev, or export "
                "CPATH=<dir with Python.h>:<its parent include dir>." if "Python.h" in err else "")
        info["triton_error"] = err[-800:] + hint
    info["ncu"] = ncu_probe()
    info["torch"] = torch.__version__
    write_json(HARDWARE / f"{slug(props.name)}.json", info)
    for k, v in info.items():
        print(f"{k:>24}: {v}")
    if "triton_error" in info:
        print("\nWARNING: Triton kernels cannot run here (see triton_error). Fix it if the hint says how; "
              "otherwise optimize at the torch/library level only (GEMM merging, layouts, cached work, "
              "attention backends, CUDA graphs) — bench.py and swap.py work without Triton.")


if __name__ == "__main__":
    main()
