#!/usr/bin/env python3
"""Where does the model spend GPU time, and where would fusion pay?

  hotspots.py --entry model_entry.py                 # module table
  hotspots.py --entry model_entry.py --module 0.mlp  # op sequence inside one module

With MODE = "train" in model_entry.py the profiled unit is a training step: forward + loss + backward.
Backward time is attributed to modules through backward hooks; "fwd us" / "bwd us" are shown separately
and eager traffic, flops, boundary and ceiling cover forward + backward.

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
from torch.utils._pytree import tree_map  # noqa: E402
from torch.utils.flop_counter import flop_registry  # noqa: E402

from common import (effective_dtype, entry_autocast, effective_dtype, entry_autocast, compute_dtype, hw_info, is_train, load_entry, nbytes, peaks, run_model,  # noqa: E402
                    set_train_mode, tensors, train_loss)

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
    train = is_train(entry)
    if train:
        set_train_mode(model)
        # Backward hooks of a module only complete when one of its inputs needs a gradient; make the
        # model's floating inputs leaves so every module (incl. the root) gets its backward attributed.
        inputs = tree_map(lambda t: t.detach().requires_grad_(True)
                          if isinstance(t, torch.Tensor) and t.is_floating_point() else t, inputs)
    else:
        model.eval()
    names = {m: (q or ROOT) for q, m in model.named_modules()}
    stack: list[str] = []
    bound_in: dict[str, int] = {}
    bound_out: dict[str, int] = {}
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
        bound_in.setdefault(names[m], pending.pop())   # bytes per call (first call)
        bound_out.setdefault(names[m], nbytes(out))
        calls[names[m]] += 1

    def bpre(m, grad_output):
        stack.append(names[m])
        r = record_function(f"krsi_bwd::{names[m]}")
        r.__enter__()
        rf[-id(m)].append(r)

    def bpost(m, grad_input, grad_output):
        if rf[-id(m)]:
            rf[-id(m)].pop().__exit__(None, None, None)
        if names[m] in stack:
            del stack[len(stack) - 1 - stack[::-1].index(names[m])]

    hooks = [m.register_forward_pre_hook(pre, with_kwargs=True) for m in names]
    hooks += [m.register_forward_hook(post, with_kwargs=True) for m in names]
    bwd_hooked = False
    if train:
        try:
            hooks += [m.register_full_backward_pre_hook(bpre) for m in names]
            hooks += [m.register_full_backward_hook(bpost) for m in names]
            bwd_hooked = True
        except Exception as e:  # noqa: BLE001
            print(f"note: backward hooks unavailable ({e}); backward time is reported for the whole model only")

    def one_step():
        if train:
            train_loss(entry, run_model(entry, model, inputs), inputs).backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                run_model(entry, model, inputs)

    try:
        for _ in range(2):
            one_step()
    except RuntimeError as e:
        if not bwd_hooked:
            raise
        # e.g. in-place ops on module outputs are incompatible with full backward hooks
        print(f"note: per-module backward attribution failed ({str(e)[:120]}); whole-model backward only")
        for h in hooks[2 * len(names):]:
            h.remove()
        hooks = hooks[: 2 * len(names)]
        bwd_hooked = False
        stack.clear()
        one_step()
    torch.cuda.synchronize()
    calls.clear()
    bound_in.clear()
    bound_out.clear()
    tracer = OpTracer(stack)
    with tracer:
        one_step()
    n_calls = dict(calls)
    steps = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            one_step()
        torch.cuda.synchronize()
    for h in hooks:
        h.remove()
    # Bytes a perfectly fused implementation must move: forward reads inputs+weights, writes outputs;
    # backward reads them again plus grad_output, writes grad_input and weight grads.
    boundary = {q: (3 * bound_in[q] + 2 * bound_out[q] if train else bound_in[q] + bound_out[q]) for q in bound_in}

    is_cuda = lambda e: str(getattr(e, "device_type", "")).endswith("CUDA")  # noqa: E731
    dev = lambda e: float(getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0))  # noqa: E731
    total_us = sum(float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0))
                   for e in prof.key_averages() if is_cuda(e) and not e.key.startswith(("krsi::", "krsi_bwd::"))) / steps
    mod_us, bwd_us, launches = {}, defaultdict(float), defaultdict(int)
    for e in prof.key_averages():
        if is_cuda(e):
            continue
        if e.key.startswith("krsi::"):
            mod_us[e.key[6:]] = dev(e) / steps
        elif e.key.startswith("krsi_bwd::"):
            bwd_us[e.key[10:]] = dev(e) / steps
    for evt in prof.events():
        if (evt.name.startswith("krsi::") or evt.name.startswith("krsi_bwd::")) and not is_cuda(evt):
            todo = list(evt.cpu_children)
            while todo:
                c = todo.pop()
                if "LaunchKernel" in c.name or "cuLaunch" in c.name:
                    launches[evt.name.split("::", 1)[1]] += 1
                todo.extend(c.cpu_children)

    eager, flops_by_mod = defaultdict(int), defaultdict(float)
    for op in tracer.ops:
        for m in set(op["stack"]):
            eager[m] += op["bytes"]
            flops_by_mod[m] += op["flops"]

    hw = hw_info()
    peak_f, peak_b = peaks(hw, effective_dtype(model, entry_autocast(entry)))

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

    print(f"device {torch.cuda.get_device_name(0)}  {'TRAINING step (fwd+loss+bwd)' if train else 'inference'} "
          f"GPU time {total_us / 1000:.3f} ms/step  "
          f"{len([o for o in tracer.ops if not o['view']])} aten ops  "
          + (f"(ceilings for {str(effective_dtype(model, entry_autocast(entry))).replace('torch.', '')}: {peak_f / 1e12:.0f} TFLOPS, "
             f"{peak_b / 1e9:.0f} GB/s, ridge {peak_f / peak_b:.0f} FLOP/B)" if hw and peak_b else
             "(run hw.py for ceilings)"))
    tcol = f"{'fwd us':>8} {'bwd us':>8} " if train else ""
    print(f"{'module':<28} {'class':<22} {'share':>6} {'ms/call':>8} {tcol}{'kernels':>7} {'eager MB':>9} "
          f"{'bound MB':>9} {'fuse x':>7} {'ceil us':>8} {'now us':>8}")
    rows = []
    for q, us in mod_us.items():
        n = max(n_calls.get(q, 1), 1)
        tot = us + bwd_us.get(q, 0.0)
        b = boundary.get(q, 0)
        f = flops_by_mod.get(q, 0.0) / n
        ceil = max(f / peak_f if peak_f else 0, b / peak_b if peak_b else 0) * 1e6
        rows.append((tot / total_us if total_us else 0, q, n, tot, us, bwd_us.get(q, 0.0), b, ceil))
    cls = {names[m]: type(m).__name__ for m in names}
    for share, q, n, tot, fu, bu, b, ceil in sorted(rows, reverse=True)[: a.top]:
        tvals = f"{fu / n:8.1f} {bu / n:8.1f} " if train else ""
        print(f"{q[:28]:<28} {cls.get(q, '?')[:22]:<22} {share:6.1%} {tot / n / 1000:8.3f} {tvals}"
              f"{launches.get(q, 0) / steps / n:7.1f} {eager[q] / n / 1e6:9.2f} {b / 1e6:9.2f} "
              f"{(eager[q] / n / b) if b else 0:7.1f} {ceil:8.1f} {tot / n:8.1f}")
    if train and not bwd_hooked:
        print("(bwd us not attributed per module; see the whole-model total above)")
    print("\nPick a region with a large share and a large gap between 'now us' and 'ceil us'. A parent module "
          "lets you fuse across its children (e.g. residual add + next norm). Then:\n"
          "  region.py --entry <model_entry.py> --module <qualname> --out <work dir>")


if __name__ == "__main__":
    main()
