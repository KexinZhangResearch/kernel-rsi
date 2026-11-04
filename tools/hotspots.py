#!/usr/bin/env python3
"""Where does the model spend GPU time, and where would fusion pay?

  hotspots.py --entry model_entry.py                 # module table
  hotspots.py --entry model_entry.py --module 0.mlp  # op sequence inside one module

Module table columns:
  ms/call     inclusive GPU time per call (kernels launched inside the module)
  kernels     kernel launches per call
  eager MB    HBM bytes the eager ops read+write in total (views excluded)
  bound MB    bytes that cross the module boundary (inputs + outputs + params): what a
              perfectly fused implementation must still move
  fuse x      eager MB / bound MB — how much traffic fusion could remove
  ceil us     max(flops / peak, bound bytes / bandwidth) on this GPU — the floor to aim at
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils.flop_counter import flop_registry  # noqa: E402

from common import compute_dtype, hw_info, peaks, load_entry, nbytes, run_model, tensors  # noqa: E402

ROOT = "<root>"


class OpTracer(TorchDispatchMode):
    """Records every aten op with its bytes, attributed to the module stack active at that moment."""

    def __init__(self, stack: list):
        super().__init__()
        self.stack, self.ops = stack, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func.__name__.startswith("_record_function"):  # our own module markers
            return out
        is_view = getattr(func, "is_view", False) or func.__name__.split(".")[0] in {
            "view", "_unsafe_view", "t", "transpose", "permute", "expand", "slice", "select", "unsqueeze",
            "squeeze", "as_strided", "detach", "alias", "split", "chunk", "unbind", "split_with_sizes"}
        b = 0 if is_view else nbytes(args) + nbytes(kwargs or {}) + nbytes(out)
        shapes = [list(t.shape) for t in tensors(args)][:3]
        fl = flop_registry.get(func.overloadpacket)
        try:
            flops = float(fl(*args, **(kwargs or {}), out_val=out)) if fl else 0.0
        except Exception:  # noqa: BLE001 - formula signatures differ across torch versions
            flops = 0.0
        self.ops.append({"op": func.__name__, "bytes": b, "flops": flops, "stack": tuple(self.stack), "shapes": shapes,
                         "out": [list(t.shape) for t in tensors(out)][:1], "view": is_view})
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--module", help="show the op sequence inside this module (qualname)")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    entry = load_entry(Path(a.entry))
    model, inputs = entry.build()
    names = {m: (q or ROOT) for q, m in model.named_modules()}
    stack: list[str] = []
    boundary: dict[str, int] = {}
    calls: dict[str, int] = defaultdict(int)
    rf: dict[int, list] = defaultdict(list)
    params = {m: sum(p.numel() * p.element_size() for p in m.parameters()) for m in names}

    pending: list[int] = []

    def pre(m, args, kwargs):
        stack.append(names[m])
        r = record_function(f"krsi::{names[m]}")
        r.__enter__()
        rf[id(m)].append(r)
        pending.append(nbytes(args) + nbytes(kwargs) + params[m])

    def post(m, args, kwargs, out):
        rf[id(m)].pop().__exit__(None, None, None)
        stack.pop()
        boundary.setdefault(names[m], pending.pop() + nbytes(out))  # bytes per call (first call)
        calls[names[m]] += 1

    hooks = [m.register_forward_pre_hook(pre, with_kwargs=True) for m in names]
    hooks += [m.register_forward_hook(post, with_kwargs=True) for m in names]

    with torch.no_grad():
        for _ in range(2):
            run_model(entry, model, inputs)
        torch.cuda.synchronize()
        calls.clear()
        boundary.clear()
        tracer = OpTracer(stack)
        with tracer:
            run_model(entry, model, inputs)
        n_calls = dict(calls)
        steps = 3
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                run_model(entry, model, inputs)
            torch.cuda.synchronize()
    for h in hooks:
        h.remove()


    is_cuda = lambda e: str(getattr(e, "device_type", "")).endswith("CUDA")  # noqa: E731
    dev = lambda e: float(getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0))  # noqa: E731
    total_us = sum(float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0))
                   for e in prof.key_averages() if is_cuda(e) and not e.key.startswith("krsi::")) / steps
    mod_us, launches = {}, defaultdict(int)
    for e in prof.key_averages():
        if e.key.startswith("krsi::") and not is_cuda(e):
            mod_us[e.key[6:]] = dev(e) / steps
    for evt in prof.events():
        if evt.name.startswith("krsi::") and not is_cuda(evt):
            todo = list(evt.cpu_children)
            while todo:
                c = todo.pop()
                if "LaunchKernel" in c.name or "cuLaunch" in c.name:
                    launches[evt.name[6:]] += 1
                todo.extend(c.cpu_children)

    eager, flops_by_mod = defaultdict(int), defaultdict(float)
    for op in tracer.ops:
        for m in set(op["stack"]):
            eager[m] += op["bytes"]
            flops_by_mod[m] += op["flops"]

    hw = hw_info()
    peak_f, peak_b = peaks(hw, compute_dtype(model))

    if a.module:
        ops = [o for o in tracer.ops if a.module in o["stack"] and not o["view"]]
        n = max(n_calls.get(a.module, 1), 1)
        print(f"{a.module}: {len(ops) // n} non-view aten ops per call, "
              f"{eager[a.module] / n / 1e6:.1f} MB eager traffic vs {boundary.get(a.module, 0) / 1e6:.1f} MB boundary")
        ops = ops[: max(1, len(ops) // n)]
        for o in ops[:120]:
            inner = o["stack"][o["stack"].index(a.module) + 1:] if a.module in o["stack"] else ()
            print(f"  {o['op']:<36} {o['bytes'] / 1e6:8.2f} MB  in {o['shapes']} -> {o['out']}  "
                  f"{'.'.join(s.split('.')[-1] for s in inner) or ''}")
        return

    print(f"device {torch.cuda.get_device_name(0)}  model GPU time {total_us / 1000:.3f} ms/step  "
          f"{len([o for o in tracer.ops if not o['view']])} aten ops  "
          + (f"(ceilings for {str(compute_dtype(model)).replace('torch.', '')}: {peak_f / 1e12:.0f} TFLOPS, "
             f"{peak_b / 1e9:.0f} GB/s, ridge {peak_f / peak_b:.0f} FLOP/B)" if hw and peak_b else
             "(run hw.py for ceilings)"))
    print(f"{'module':<28} {'class':<22} {'share':>6} {'ms/call':>8} {'kernels':>7} {'eager MB':>9} "
          f"{'bound MB':>9} {'fuse x':>7} {'ceil us':>8} {'now us':>8}")
    rows = []
    for q, us in mod_us.items():
        n = max(n_calls.get(q, 1), 1)
        b = boundary.get(q, 0)
        f = flops_by_mod.get(q, 0.0) / n
        ceil = max(f / peak_f if peak_f else 0, b / peak_b if peak_b else 0) * 1e6
        rows.append((us / total_us if total_us else 0, q, n, us, b, f, ceil))
    cls = {names[m]: type(m).__name__ for m in names}
    for share, q, n, us, b, f, ceil in sorted(rows, reverse=True)[: a.top]:
        print(f"{q[:28]:<28} {cls.get(q, '?')[:22]:<22} {share:6.1%} {us / n / 1000:8.3f} "
              f"{launches.get(q, 0) / steps / n:7.1f} {eager[q] / n / 1e6:9.2f} {b / 1e6:9.2f} "
              f"{(eager[q] / n / b) if b else 0:7.1f} {ceil:8.1f} {us / n:8.1f}")
    print("\nPick a region with a large share and a large gap between 'now us' and 'ceil us'. A parent module "
          "lets you fuse across its children (e.g. residual add + next norm). Then:\n"
          "  region.py --entry <model_entry.py> --module <qualname> --out <work dir>")


if __name__ == "__main__":
    main()
