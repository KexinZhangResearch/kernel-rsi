"""Shared helpers: model entry loading, honest GPU timing, numerical comparison."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten, tree_map

SKILL = Path(__file__).resolve().parent.parent
HARDWARE = SKILL / "hardware"


# ---------------------------------------------------------------- files

def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def hw_info() -> dict:
    path = HARDWARE / f"{slug(torch.cuda.get_device_name(torch.cuda.current_device()))}.json"
    return read_json(path) if path.is_file() else {}


def compute_dtype(x) -> torch.dtype:
    """The floating dtype carrying most of the bytes in x (tensors, modules, or nested containers)."""
    if isinstance(x, torch.nn.Module):
        x = list(x.parameters()) + list(x.buffers())
    sizes: dict = {}
    for t in tensors(x):
        if t.is_floating_point():
            sizes[t.dtype] = sizes.get(t.dtype, 0) + t.numel()
    return max(sizes, key=sizes.get) if sizes else torch.float32


def peaks(hw: dict, dtype: torch.dtype) -> tuple[float, float]:
    """(FLOP/s, bytes/s) ceilings for work in `dtype` on this GPU; zeros if hw.py has not run."""
    tf = hw.get("tflops") or {}
    key = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dtype, "fp16")
    if key == "fp32" and torch.backends.cuda.matmul.allow_tf32 and "tf32" in tf:
        key = "tf32"
    flops = (tf.get(key) or tf.get("fp16") or 0) * 1e12
    bw = max(hw.get("bw_copy_gbs") or 0, hw.get("bw_read_gbs") or 0) * 1e9
    return flops, bw


# ---------------------------------------------------------------- model entry

def load_py(path: Path, name: str | None = None):
    """Import a file under its own name so classes defined in it can be pickled and reloaded."""
    path = Path(path).resolve()
    name = name or path.stem
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_entry(path: Path):
    mod = load_py(path)
    if not hasattr(mod, "build"):
        raise SystemExit(f"{path} must define build() -> (model, inputs)")
    return mod


def run_model(entry, model, inputs):
    if hasattr(entry, "run"):
        return entry.run(model, inputs)
    return model(**inputs) if isinstance(inputs, dict) else model(*inputs)


def tensors(x) -> list[torch.Tensor]:
    return [t for t in tree_flatten(x)[0] if isinstance(t, torch.Tensor)]


def to_float32(x):
    return tree_map(lambda t: t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t, x)


def randomize_like(x, seed: int):
    """Fresh data with each tensor's own mean/std. Integer tensors and 0/1-style masks are kept."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def one(t):
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() < 2:
            return t
        if torch.unique(t.flatten()[:65536]).numel() <= 2:  # mask-like: keep semantics
            return t
        f = t.float()
        r = torch.randn(t.shape, generator=g, device=t.device) * f.std() + f.mean()
        return r.to(t.dtype)

    return tree_map(one, x)


# ---------------------------------------------------------------- timing

def l2_bytes() -> int:
    return int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 50 << 20) or (50 << 20))


def nbytes(x) -> int:
    return sum(t.numel() * t.element_size() for t in tensors(x))


def rotation_count(working_set: int, cap: int = 16) -> int:
    """Enough independent copies that each call reads cold (not L2-resident) data."""
    if working_set <= 0:
        return 1
    want = math.ceil(3 * l2_bytes() / working_set)
    free = torch.cuda.mem_get_info()[0]
    return max(1, min(cap, want, int(0.4 * free // working_set)))


def time_calls(fns: list, warmup: int = 5, reps: int = 20) -> dict:
    """Median GPU time per call of fns (rotated). Tries a CUDA graph first, which removes host
    launch overhead; falls back to eager events. Also reports host-inclusive time per call."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize()

    def eager_ms(n):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(n):
            start.record()
            for f in fns:
                f()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / len(fns))
        return sorted(samples)[len(samples) // 2]

    host = eager_ms(reps)
    out = {"host_ms": host, "copies": len(fns)}
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for f in fns:
                f()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for f in fns:
                f()
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(reps):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / len(fns))
        samples.sort()
        out.update(device_ms=samples[len(samples) // 2], mode="cuda_graph",
                   spread=(samples[-1] - samples[0]) / max(samples[len(samples) // 2], 1e-9))
        del graph
    except Exception as e:  # noqa: BLE001 - capture is best effort
        torch.cuda.synchronize()
        out.update(device_ms=host, mode="eager", note=f"graph capture failed: {str(e)[:200]}")
    return out


# ---------------------------------------------------------------- numerics

def _locate(bad: torch.Tensor) -> list[str]:
    """Which indices along each dim contain failing elements — points at the bug's shape."""
    out = []
    for d in range(bad.dim()):
        other = [i for i in range(bad.dim()) if i != d]
        along = bad.any(dim=tuple(other)) if other else bad
        idx = along.nonzero().flatten()
        if idx.numel() == 0:
            continue
        n = bad.shape[d]
        desc = f"dim{d} (size {n}): {idx.numel()} bad"
        if idx.numel() < n:
            desc += f", first {idx[0].item()}, last {idx[-1].item()}"
            if idx.numel() <= 8:
                desc += f", at {idx.tolist()}"
        out.append(desc)
    return out


def compare(ref32, orig, cand, *, slack: float = 2.0) -> dict:
    """Judge cand against a float32 run of the original.

    Pass if no element of cand is further from fp32 than `slack` x the original's worst
    element error, and cand's relative L2 error is within `slack` x the original's:
    "same model logic" at the numerical-noise level of the original, whatever the
    dtype or number of ops.
    """
    r_list, o_list, c_list = tensors(ref32), tensors(orig), tensors(cand)
    if len(r_list) != len(c_list):
        return {"pass": False, "reason": f"output count {len(c_list)} != {len(r_list)}"}
    report, ok = [], True
    for i, (r, o, c) in enumerate(zip(r_list, o_list, c_list)):
        item = {"output": i, "shape": list(r.shape), "dtype": str(c.dtype)}
        if tuple(c.shape) != tuple(r.shape):
            item.update(pass_=False, reason=f"shape {list(c.shape)} != {list(r.shape)}")
            report.append(item)
            ok = False
            continue
        if not r.is_floating_point():
            same = torch.equal(c.to(r.dtype), r)
            item.update(pass_=same, exact=True)
            ok &= same
            report.append(item)
            continue
        r, o, c = r.float(), o.float(), c.float()
        scale = r.abs().max().item() or 1.0
        err_o, err_c = (o - r).abs(), (c - r).abs()
        nonfinite = int((~torch.isfinite(c)).sum())
        # Noise level = the original's worst error over the whole tensor (its rounding is as
        # arbitrary as ours element by element; only the overall level is meaningful).
        budget = slack * err_o.max() + 1e-3 * scale + 1e-6
        bad = (err_c > budget) | ~torch.isfinite(c)
        rel = lambda e: (e.norm() / (r.norm() + 1e-12)).item()  # noqa: E731
        item.update(max_err=err_c.max().item(), orig_max_err=err_o.max().item(),
                    rel_l2=rel(err_c), orig_rel_l2=rel(err_o), nonfinite=nonfinite,
                    bad_fraction=bad.float().mean().item())
        passed = not bad.any().item() and item["rel_l2"] <= slack * item["orig_rel_l2"] + 1e-5
        item["pass_"] = passed
        if not passed:
            worst = int(err_c.flatten().nan_to_num(float("inf")).argmax())
            pos, rem = [], worst
            for size in reversed(r.shape):  # unravel without torch.unravel_index (torch >= 2.2 only)
                pos.append(rem % size)
                rem //= size
            item["worst"] = {"index": list(reversed(pos)), "ref": r.flatten()[worst].item(),
                             "got": c.flatten()[worst].item()}
            item["where"] = _locate(bad)
        ok &= passed
        report.append(item)
    for item in report:
        item["pass"] = item.pop("pass_")
    return {"pass": ok, "outputs": report}


def format_compare(res: dict) -> str:
    lines = [f"correctness: {'PASS' if res['pass'] else 'FAIL'}" + (f" ({res['reason']})" if res.get("reason") else "")]
    for o in res.get("outputs", []):
        if o.get("exact"):
            lines.append(f"  out{o['output']} {o['shape']}: exact match {'yes' if o['pass'] else 'NO'}")
            continue
        if "max_err" not in o:
            lines.append(f"  out{o['output']}: {o.get('reason')}")
            continue
        lines.append(f"  out{o['output']} {o['shape']} {o['dtype']}: max_err {o['max_err']:.3e} "
                     f"(original {o['orig_max_err']:.3e}), rel_l2 {o['rel_l2']:.2e} (original {o['orig_rel_l2']:.2e})"
                     + ("" if o["pass"] else f"  <-- FAIL, {o['bad_fraction']:.2%} of elements, nonfinite {o['nonfinite']}"))
        if not o["pass"]:
            w = o["worst"]
            lines.append(f"     worst at {w['index']}: ref {w['ref']:.5g} got {w['got']:.5g}")
            lines += [f"     {d}" for d in o["where"]]
    return "\n".join(lines)
