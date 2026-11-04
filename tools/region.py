#!/usr/bin/env python3
"""Freeze one region (any module, including a parent block or the root) of a real model.

  region.py --entry model_entry.py --module 0 --out work/dit_block0

Writes into --out:
  module.pt, io.pt   the module with its real weights, and the real inputs/outputs of its first call
  region.json        metadata: shapes, flops, boundary bytes, ceiling on this GPU, sibling instances
  ORIGINAL.md        the code to preserve: forward() of the region and of its user-defined children
  impl.py            your replacement (created once; starts as a call to the original forward)
  NOTES.md           your working log: hypotheses, attempts, results
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402

from common import compute_dtype, hw_info, peaks, load_entry, nbytes, run_model, tensors, write_json  # noqa: E402
from hotspots import ROOT, OpTracer  # noqa: E402

IMPL_TEMPLATE = '''"""Replacement for {cls} ({qual}). Contract: compute the same function as the original forward.

Called as forward(mod, ...) where `mod` is the original module instance: use its parameters
(see ORIGINAL.md). Anything goes inside — Triton kernels, torch/cuBLAS calls, a reordered or
different algorithm, fusing across the children — as long as bench.py passes. Weight
preprocessing (packing, casting) may be cached on `mod` on first call.
"""
import torch

try:  # Triton is optional: on GPUs it does not support, optimize with torch/library calls only
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


def forward(mod, {params}):
    return type(mod).forward(mod, {call})  # start from the original, then replace it piece by piece
'''


def _sources(mod: torch.nn.Module) -> str:
    seen, parts = set(), []
    for name, m in mod.named_modules():
        cls = type(m)
        if cls in seen or cls.__module__.startswith("torch."):
            continue
        seen.add(cls)
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            src = f"(source unavailable for {cls.__qualname__})"
        parts.append(f"### {cls.__qualname__}  ({name or 'region root'})\n```python\n{src}```")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--module", required=True, help=f"qualname from hotspots.py ('{ROOT}' for the whole model)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    entry_path = Path(a.entry).resolve()
    entry = load_entry(entry_path)
    model, inputs = entry.build()
    mods = {(q or ROOT): m for q, m in model.named_modules()}
    if a.module not in mods:
        near = [q for q in mods if a.module in q][:10]
        raise SystemExit(f"no module '{a.module}'. Similar: {near}")
    target = mods[a.module]
    seen = {}

    def hook(m, args, kwargs, out):
        if "args" not in seen:
            seen.update(args=tree_map(lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t, args),
                        kwargs=tree_map(lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t, kwargs),
                        out=tree_map(lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t, out))

    h = target.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        run_model(entry, model, inputs)
    h.remove()
    if "args" not in seen:
        raise SystemExit(f"{a.module} was not called during the forward pass")

    with torch.no_grad():
        tracer = OpTracer([a.module])
        with tracer:
            target(*seen["args"], **seen["kwargs"])
    flops = sum(o["flops"] for o in tracer.ops)
    eager_bytes = sum(o["bytes"] for o in tracer.ops)
    params = sum(p.numel() * p.element_size() for p in target.parameters())
    bound = nbytes(seen["args"]) + nbytes(seen["kwargs"]) + nbytes(seen["out"]) + params
    hw = hw_info()
    peak_f, peak_b = peaks(hw, compute_dtype(target))
    ceil_us = max(flops / peak_f if peak_f else 0, bound / peak_b if peak_b else 0) * 1e6

    shapes = lambda x: [(list(t.shape), str(t.dtype).replace("torch.", "")) for t in tensors(x)]  # noqa: E731
    siblings = [q for q, m in mods.items() if type(m) is type(target) and q != a.module
                and [p.shape for p in m.parameters()] == [p.shape for p in target.parameters()]]
    sig = inspect.signature(type(target).forward)
    names = [p for p in sig.parameters if p != "self"][: len(seen["args"])]
    names += [f"arg{i}" for i in range(len(names), len(seen["args"]))]

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(target, out / "module.pt")
    torch.save({k: tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, v) for k, v in seen.items()},
               out / "io.pt")
    meta = {"entry": str(entry_path), "module": a.module, "class": type(target).__qualname__,
            "class_module": type(target).__module__, "arg_names": names, "kwarg_names": list(seen["kwargs"]),
            "inputs": shapes(seen["args"]) + shapes(seen["kwargs"]), "outputs": shapes(seen["out"]),
            "params_mb": round(params / 1e6, 2), "flops": flops, "boundary_mb": round(bound / 1e6, 2),
            "eager_traffic_mb": round(eager_bytes / 1e6, 2),
            "eager_ops": len([o for o in tracer.ops if not o["view"]]),
            "ceiling_us": round(ceil_us, 1), "same_class_instances": siblings, "hardware": hw.get("device")}
    write_json(out / "region.json", meta)
    (out / "ORIGINAL.md").write_text(
        f"# Region {a.module}: {type(target).__qualname__}\n\n"
        f"inputs: {meta['inputs']}\noutputs: {meta['outputs']}\n"
        f"flops {flops / 1e9:.3f} GFLOP, boundary {meta['boundary_mb']} MB, eager traffic "
        f"{meta['eager_traffic_mb']} MB in {meta['eager_ops']} ops, ceiling {meta['ceiling_us']} us on "
        f"{hw.get('device', 'unknown GPU (run hw.py)')}\n\n## Module\n```\n{target}\n```\n\n## Code\n{_sources(target)}\n")
    if not (out / "impl.py").exists():
        params_sig = ", ".join(names + [f"{k}=None" for k in seen["kwargs"]])
        call = ", ".join(names + [f"{k}={k}" for k in seen["kwargs"]])
        (out / "impl.py").write_text(IMPL_TEMPLATE.format(cls=type(target).__qualname__, qual=a.module,
                                                          params=params_sig, call=call))
    if not (out / "NOTES.md").exists():
        (out / "NOTES.md").write_text(f"# {a.module} ({type(target).__qualname__}) — working log\n\n"
                                      "Ceiling and eager traffic: see region.json. One entry per attempt:\n"
                                      "hypothesis -> change -> bench result (results/NNN.json) -> conclusion.\n")
    print(f"captured {a.module} ({type(target).__qualname__}) -> {out}")
    print(f"  {flops / 1e9:.3f} GFLOP, boundary {meta['boundary_mb']} MB, eager {meta['eager_traffic_mb']} MB "
          f"in {meta['eager_ops']} ops, ceiling {meta['ceiling_us']} us")
    if siblings:
        print(f"  {len(siblings)} other instances with the same class and shapes: {siblings[:6]} "
              f"(swap.py --all-instances applies your impl to all of them)")
    print(f"next: read {out}/ORIGINAL.md, edit {out}/impl.py, run bench.py {out}")


if __name__ == "__main__":
    main()
