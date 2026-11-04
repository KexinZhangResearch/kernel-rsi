"""Shared helpers: model entry loading, honest GPU timing, numerical comparison."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch._dynamo
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


def effective_dtype(x, autocast: dict | None = None) -> torch.dtype:
    """Dtype the heavy math runs in: the autocast dtype if AMP is on, else the dominant parameter dtype."""
    return getattr(torch, autocast["dtype"]) if autocast else compute_dtype(x)


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


def entry_autocast(entry) -> dict | None:
    """model_entry.py may set AUTOCAST = "bfloat16" / "float16" (AMP). Do not call torch.autocast
    inside run(): the tools must be able to run the fp32 reference without it."""
    dt = getattr(entry, "AUTOCAST", None)
    return {"dtype": dt} if dt else None


def run_model(entry, model, inputs, amp: bool = True):
    """Run the model as the user does; amp=False for the fp32 reference."""
    with autocast_ctx(entry_autocast(entry) if amp else None):
        if hasattr(entry, "run"):
            return entry.run(model, inputs)
        return model(**inputs) if isinstance(inputs, dict) else model(*inputs)


def is_train(entry) -> bool:
    """model_entry.py sets MODE = "train" to optimize forward + backward."""
    return getattr(entry, "MODE", "infer") == "train"


def set_train_mode(module: torch.nn.Module) -> list[str]:
    """train() semantics, but dropout off: random masks cannot be checked for equivalence.
    Returns the names of dropout modules that were switched off (to report)."""
    module.train()
    off = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0:
            m.eval()
            off.append(name)
    return off


def train_loss(entry, out, inputs):
    """entry.loss(out, inputs) if defined, else a scalar output, else mean of squares of all outputs."""
    if hasattr(entry, "loss"):
        return entry.loss(out, inputs)
    ts = [t for t in tensors(out) if t.is_floating_point() and t.requires_grad]
    if len(ts) == 1 and ts[0].dim() == 0:
        return ts[0]
    return sum(t.float().pow(2).mean() for t in ts)


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


class pure_fp32:
    """Reference runs in true fp32: TF32 off for matmul AND cuDNN (cuDNN defaults to TF32 for conv)."""

    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved


def autocast_state() -> dict | None:
    """The CUDA autocast state active right now (None if off). Stored with a captured region so that
    bench.py replays the region exactly as the model runs it (AMP training/inference)."""
    try:
        on = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
    except TypeError:  # torch < 2.4
        on = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype()
    return {"dtype": str(dtype).replace("torch.", "")} if on else None


def autocast_ctx(state: dict | None):
    if not state:
        return _Null()
    return torch.autocast("cuda", dtype=getattr(torch, state["dtype"]))


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def versions() -> dict:
    out = {"device": torch.cuda.get_device_name(torch.cuda.current_device()), "torch": torch.__version__,
           "cuda": torch.version.cuda}
    try:
        import triton  # noqa: PLC0415

        out["triton"] = triton.__version__
    except ImportError:
        out["triton"] = None
    return out


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


def _host_sync(fns) -> str | None:
    """Run fns once with CUDA sync debugging set to error; return a description of the first sync."""
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        for f in fns[:1]:
            f()
        return None
    except RuntimeError as e:
        return str(e).splitlines()[0][:120]
    finally:
        torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()


def time_calls(fns: list, warmup: int = 5, reps: int = 20, graph: bool = True) -> dict:
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
    if not graph:
        out.update(device_ms=host, mode="eager")
        return out
    # A capture that fails half-way (e.g. on a host sync such as .item()) corrupts the process's CUDA
    # graph/RNG state for good (later randn() raises). So detect host syncs first, without capturing.
    sync = _host_sync(fns)
    if sync:
        out.update(device_ms=host, mode="eager", note=f"not graph-capturable: host sync ({sync})")
        return out
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


def compare(ref32, orig, cand, *, slack: float = 2.0, row_slack: float = 3.0) -> dict:
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
        # Per-row relative error vs the original's error on the SAME row: catches bugs confined to
        # small-magnitude slices (one head, padding tokens) that a tensor-wide budget dilutes.
        if r.dim() >= 2 and r.shape[-1] >= 64 and r.numel() // r.shape[-1] >= 2:  # short rows: too noisy
            rn = r.reshape(-1, r.shape[-1]).norm(dim=1)
            live = rn > 1e-6 * (rn.max() + 1e-30)
            if live.any():
                row_c = ((c - r).reshape(-1, r.shape[-1]).norm(dim=1) / (rn + 1e-30))[live]
                row_o = ((o - r).reshape(-1, r.shape[-1]).norm(dim=1) / (rn + 1e-30))[live]
                floor = row_o.median() + 1e-5
                # Row errors of a correct impl have a heavy tail (measured up to ~2.1x through an
                # attention backward); real localized bugs measured 6-23x. Hence a separate 3x.
                bad_rows = row_c > row_slack * torch.maximum(row_o, floor)
                item["bad_rows"] = int(bad_rows.sum())
                if item["bad_rows"]:
                    passed = False
                    w = int((row_c / torch.maximum(row_o, floor)).argmax())
                    item["worst_row"] = {"row": int(live.nonzero().flatten()[w]), "rel": row_c[w].item(),
                                         "orig_rel": row_o[w].item()}
        item["pass_"] = passed
        if not passed and bad.any():
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


def format_compare(res: dict, names: list[str] | None = None, compact: bool = False) -> str:
    """compact: one summary line plus only the failing tensors (for long gradient lists)."""
    outs = res.get("outputs", [])
    label = lambda o: names[o["output"]] if names and o["output"] < len(names) else f"out{o['output']}"  # noqa: E731
    lines = [f"correctness: {'PASS' if res['pass'] else 'FAIL'}" + (f" ({res['reason']})" if res.get("reason") else "")]
    if compact and outs:
        worst = max((o for o in outs if "rel_l2" in o), default=None,
                    key=lambda o: o["rel_l2"] / max(o["orig_rel_l2"], 1e-12))
        lines[0] += f" — {sum(o['pass'] for o in outs)}/{len(outs)} tensors pass"
        # A ratio vs an original error that is itself ~0 (e.g. a scalar loss that happened to round
        # exactly) is meaningless; rank by ratio only where the original error is measurable.
        meas = [o for o in outs if "rel_l2" in o and o["orig_rel_l2"] > 1e-5]
        worst = max(meas, default=worst, key=lambda o: o["rel_l2"] / o["orig_rel_l2"])
        if worst:
            lines[0] += (f"; worst rel_l2 vs original {worst['rel_l2'] / max(worst['orig_rel_l2'], 1e-12):.2f}x "
                         f"({label(worst)})")
        outs = [o for o in outs if not o["pass"]]
    for o in outs:
        if o.get("exact"):
            lines.append(f"  {label(o)} {o['shape']}: exact match {'yes' if o['pass'] else 'NO'}")
            continue
        if "max_err" not in o:
            lines.append(f"  {label(o)}: {o.get('reason')}")
            continue
        lines.append(f"  {label(o)} {o['shape']} {o['dtype']}: max_err {o['max_err']:.3e} "
                     f"(original {o['orig_max_err']:.3e}), rel_l2 {o['rel_l2']:.2e} (original {o['orig_rel_l2']:.2e})"
                     + ("" if o["pass"] else f"  <-- FAIL, {o['bad_fraction']:.2%} of elements, nonfinite {o['nonfinite']}"))
        if not o["pass"] and "worst" in o:
            w = o["worst"]
            lines.append(f"     worst at {w['index']}: ref {w['ref']:.5g} got {w['got']:.5g}")
            lines += [f"     {d}" for d in o["where"]]
        if o.get("bad_rows"):
            w = o["worst_row"]
            lines.append(f"     {o['bad_rows']} rows (flattened over all but the last dim) exceed the original's "
                         f"error on the same row; worst row {w['row']}: rel {w['rel']:.2e} vs original {w['orig_rel']:.2e}")
    return "\n".join(lines)


def time_ab(variants: dict, rounds: int = 5, reps: int = 10) -> dict:
    """Time several variants (name -> list of rotated callables) in interleaved rounds so that clock,
    thermal and neighbour noise hit all of them alike. Returns per-variant timing (as time_calls) plus
    `noise` = relative spread of that variant's round medians."""
    # All variants must be timed the same way: if any cannot be captured in a CUDA graph (host syncs,
    # .item(), dynamic shapes), every variant is timed eagerly, or the comparison is meaningless.
    probe = {name: time_calls(fns, warmup=3, reps=3) for name, fns in variants.items()}
    graph = all(t["mode"] == "cuda_graph" for t in probe.values())
    forced = None if graph else [n for n, t in probe.items() if t["mode"] != "cuda_graph"]
    per = {name: [] for name in variants}
    hosts = {name: [] for name in variants}
    first = {}
    for r in range(rounds):
        order = list(variants) if r % 2 == 0 else list(reversed(variants))
        for name in order:
            t = time_calls(variants[name], warmup=3 if r else 5, reps=reps, graph=graph)
            if forced:
                t["note"] = f"all variants timed eagerly: {forced} could not be graph-captured ({probe[forced[0]].get('note', '')[:120]})"
            per[name].append(t["device_ms"])
            hosts[name].append(t["host_ms"])
            first.setdefault(name, t)
    out = {}
    for name, xs in per.items():
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        hs = sorted(hosts[name])
        hmed = hs[len(hs) // 2]
        # Host time depends on CPU load from other processes; report its spread so a busy machine
        # is visible instead of looking like a slow implementation.
        out[name] = {**first[name], "device_ms": med, "rounds": xs, "noise": (xs[-1] - xs[0]) / max(med, 1e-12),
                     "host_ms": hmed, "host_rounds": hs, "host_noise": (hs[-1] - hs[0]) / max(hmed, 1e-12)}
    return out


def significance(base: dict, cand: dict) -> str:
    """'faster' / 'slower' only when the round ranges do not overlap; else 'within noise'."""
    if max(cand["rounds"]) < min(base["rounds"]):
        return "faster"
    if min(cand["rounds"]) > max(base["rounds"]):
        return "slower"
    return "within noise"


def selftest_autograd(fn, ref, inputs: list, slack: float = 2.0) -> dict:
    """Check fn against ref (output + gradient of every floating input) with the same criterion as
    bench.py, on fresh random upstream gradients; time fwd+bwd of both. Used by library/ self-tests."""
    def run(f, xs, dtype=None, cot=None):
        xs = [x.detach().to(dtype or x.dtype).requires_grad_(x.is_floating_point()) for x in xs]
        out = f(*xs)
        cot = torch.randn_like(out.float()).mul_(1e-2) if cot is None else cot
        grads = torch.autograd.grad(out, [x for x in xs if x.requires_grad], cot.to(out.dtype))
        return [out.detach()] + [g.detach() for g in grads], cot

    with pure_fp32():
        ref32, cot = run(ref, inputs, torch.float32)
    orig, _ = run(ref, inputs, cot=cot)
    cand, _ = run(fn, inputs, cot=cot)
    res = compare(ref32, orig, cand, slack=slack)
    leaves = [x.detach().requires_grad_(x.is_floating_point()) for x in inputs]
    g = cot.to(inputs[0].dtype)

    def step(f):
        return lambda: torch.autograd.grad(f(*leaves), [x for x in leaves if x.requires_grad], g)
    variants = {"reference": [step(ref)], "kernel": [step(fn)]}
    compiled = {}
    # Under torch.compile(fullgraph=True) the kernel must trace without graph breaks and stay correct;
    # it is timed against the compiled reference, which is the bar it has to beat in a compiled model.
    try:
        torch._dynamo.reset()
        cfn, cref = torch.compile(fn, fullgraph=True), torch.compile(ref)
        cand_c, _ = run(cfn, inputs, cot=cot)
        rc = compare(ref32, orig, cand_c, slack=slack)
        compiled = {"pass": rc["pass"]}
        variants.update(compiled_reference=[step(cref)], compiled_kernel=[step(cfn)])
    except Exception as e:  # noqa: BLE001
        compiled = {"pass": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    t = time_ab(variants, rounds=3)
    us = {k: round(v["device_ms"] * 1000, 1) for k, v in t.items()}
    if "compiled_kernel" in t:
        compiled.update(kernel_us=us["compiled_kernel"], reference_us=us["compiled_reference"],
                        speedup=round(us["compiled_reference"] / us["compiled_kernel"], 2),
                        significance=significance(t["compiled_reference"], t["compiled_kernel"]))
    return {"pass": res["pass"] and compiled.get("pass", False), "checks": res, **versions(),
            "reference_us": us["reference"], "kernel_us": us["kernel"],
            "speedup": round(us["reference"] / us["kernel"], 2),
            "significance": significance(t["reference"], t["kernel"]), "compiled": compiled}


def sweep(make, configs: list[dict], check=None, reps: int = 10) -> list[dict]:
    """Time make(**cfg) (a zero-arg callable) for each config; optional check(callable) -> bool rejects
    wrong configs. Returns rows sorted by time. For tile/warp/stage searches in scratch scripts:

        rows = sweep(lambda BM, BN, w: (lambda: kernel[grid](..., BM=BM, BN=BN, num_warps=w)),
                     [dict(BM=64, BN=32, w=4), dict(BM=128, BN=64, w=8)])
    """
    rows = []
    for cfg in configs:
        try:
            fn = make(**cfg)
            if check is not None and not check(fn):
                rows.append({**cfg, "us": None, "error": "check failed"})
                continue
            rows.append({**cfg, "us": round(time_calls([fn], reps=reps)["device_ms"] * 1000, 2)})
        except Exception as e:  # noqa: BLE001 - e.g. out of shared memory for this tile
            rows.append({**cfg, "us": None, "error": f"{type(e).__name__}: {str(e)[:80]}"})
    rows.sort(key=lambda r: (r["us"] is None, r["us"] or 0))
    for r in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    return rows
