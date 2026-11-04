#!/usr/bin/env python3
"""Check and time impl.py of a captured region against the original module.

  bench.py work/dit_block0 [--compile] [--slack 2] [--impl other_impl.py]

Correctness: on the real captured inputs and 3 fresh random draws (same per-tensor stats; masks
and integer tensors kept), the impl's error vs a float32 run of the original must be within
`slack` x the original's own low-precision error. On failure it prints where the bad elements are
(index ranges per dimension), which usually points straight at the bug.

Training regions (captured with MODE = "train"): the same check also covers the gradients of every
input that needs one and of every parameter, given the real upstream gradient (and random ones).
Between rounds the weights are re-randomized, so an impl that caches weight-derived tensors without
noticing updates fails. Timing is forward + backward, and peak extra memory is reported because
training trades memory (saved activations) for time.

Timing: GPU time per call with CUDA graphs (no host launch overhead) and inputs + weights rotated
through enough copies that every call reads cold data, same for original and impl. host_ms is
the launch-inclusive eager cost — what an un-graphed model pays.

Every run is saved as results/NNN.json plus a frozen copy of the impl (results/NNN_impl.py).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from torch.func import functional_call  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402

from common import (effective_dtype, entry_autocast, autocast_ctx, compare, compute_dtype, format_compare, hw_info, l2_bytes, load_py,  # noqa: E402
                    nbytes, peaks, pure_fp32,
                    randomize_like, read_json, rotation_count, set_train_mode, significance, tensors, time_ab,
                    to_float32, versions, write_json)


def load_region(d: Path):
    meta = read_json(d / "region.json")
    load_py(Path(meta["entry"]))  # makes the module's classes importable for torch.load
    mod = torch.load(d / "module.pt", weights_only=False).cuda()
    io = torch.load(d / "io.pt", weights_only=False)
    cuda = lambda x: tree_map(lambda t: t.cuda() if isinstance(t, torch.Tensor) else t, x)  # noqa: E731
    return meta, mod, cuda(io["args"]), cuda(io["kwargs"]), cuda(io.get("grad_out"))


def next_result(d: Path) -> int:
    (d / "results").mkdir(exist_ok=True)
    nums = [int(p.stem) for p in (d / "results").glob("[0-9][0-9][0-9].json")]
    return max(nums, default=0) + 1


def leafify(args, kwargs, grad_idx):
    """Copies of the inputs where the ones that need gradients are fresh leaf tensors."""
    counter = iter(range(10 ** 9))

    def one(t):
        if not isinstance(t, torch.Tensor):
            return t
        return t.detach().clone().requires_grad_(True) if next(counter) in grad_idx else t
    return tree_map(one, (args, kwargs))


def fwd_bwd(call, m, xa, xk, cots):
    """Outputs and gradients (inputs that require grad, then parameters) of one training step."""
    params = [p for p in m.parameters() if p.requires_grad]
    leaves = [t for t in tensors((xa, xk)) if t.requires_grad] + params
    outs = tensors(call(m, xa, xk))
    pairs = []
    for i, (o, c) in enumerate(zip(outs, cots)):
        if c is None:
            continue
        if not o.requires_grad:
            raise RuntimeError(f"output {i} does not require grad: the impl is not differentiable here "
                               "(wrap Triton kernels in a torch.autograd.Function with a backward)")
        pairs.append((o, c.to(o.dtype)))
    grads = torch.autograd.grad([o for o, _ in pairs], leaves, [c for _, c in pairs], allow_unused=True)
    grads = [torch.zeros_like(x) if g is None else g for g, x in zip(grads, leaves)]
    return [o.detach() for o in outs] + [g.detach() for g in grads]


def peak_extra_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--impl", help="default: <region>/impl.py")
    ap.add_argument("--compile", action="store_true", help="also time torch.compile of the original")
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--note", default="", help="one line saved with the result: what this attempt tries")
    a = ap.parse_args()
    d = Path(a.region).resolve()
    impl_path = Path(a.impl).resolve() if a.impl else d / "impl.py"
    meta, mod, args, kwargs, grad_out = load_region(d)
    train = meta.get("mode") == "train"
    grad_idx = set(meta.get("grad_inputs", []))
    sha = hashlib.sha1(impl_path.read_bytes()).hexdigest()[:10]
    impl = load_py(impl_path, name=f"krsi_impl_{sha}")
    set_train_mode(mod) if train else mod.eval()
    ac = meta.get("autocast")  # replay the region exactly as the model runs it (AMP)
    if ac:
        print(f"note: replaying under torch.autocast({ac['dtype']}) as captured; the reference is pure fp32")

    def fp32_copy():
        # Built per round and freed after use: a resident fp32 copy doubles the memory of large regions.
        m32 = copy.deepcopy(mod).float()
        set_train_mode(m32) if train else m32.eval()
        return m32

    in_names = [n for n in meta["arg_names"] + meta["kwarg_names"]]
    names = [f"out{i}" for i in range(len(meta["outputs"]))]
    if train:
        names += [f"grad {in_names[i] if i < len(in_names) else f'in{i}'}" for i in sorted(grad_idx)]
        names += [f"grad {n}" for n, p in mod.named_parameters() if p.requires_grad]

    result = {"region": str(d), "module": meta["module"], "mode": meta.get("mode", "infer"), "impl_sha": sha,
              "note": a.note, **versions(), "slack": a.slack, "checks": []}
    if meta.get("dropout_disabled"):
        print(f"note: dropout disabled in {meta['dropout_disabled']} for checking (random masks are not comparable)")
    def call_ref(m, x, y):
        return m(*x, **y)

    def call_orig(m, x, y):
        with autocast_ctx(ac):
            return m(*x, **y)

    def call_impl(m, x, y):
        with autocast_ctx(ac):
            return impl.forward(m, *x, **y)

    class ImplModule(torch.nn.Module):
        """impl.forward as a module, so torch.compile + functional_call can rotate weights without recompiles."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *x, **y):
            return impl.forward(self.inner, *x, **y)

    impl_module = ImplModule(mod)
    compiled_impl = torch.compile(lambda p, x, y: functional_call(impl_module, p, x, y), dynamic=False) \
        if a.compile else None

    def call_compiled_impl(m, x, y):
        params = {"inner." + k: v for k, v in (dict(m.named_parameters()) | dict(m.named_buffers())).items()}
        with autocast_ctx(ac):
            return compiled_impl(params, x, y)
    ok = True
    for seed in [None, 1, 2, 3]:
        xa, xk = (args, kwargs) if seed is None else (randomize_like(args, seed), randomize_like(kwargs, seed))
        if train and seed is not None:
            with torch.no_grad():  # new weights: catches impls that cache weight-derived tensors
                for p in mod.parameters():
                    p.copy_(randomize_like(p.detach(), 100 + seed))
        before = [t.clone() for t in tensors((xa, xk))]
        label = "captured" if seed is None else f"random seed {seed}"
        # Module state (buffers such as BatchNorm running stats, and the parameters themselves) is part
        # of the semantics: the impl must leave it exactly as the original does.
        state0 = {k: v.clone() for k, v in mod.state_dict().items()}
        try:
            if train:
                cots = grad_out if seed is None else randomize_like(grad_out, 50 + seed)
                mod32 = fp32_copy()
                with pure_fp32():
                    ref32 = fwd_bwd(call_ref, mod32, *to_float32(leafify(xa, xk, grad_idx)), to_float32(cots))
                del mod32
                orig = fwd_bwd(call_orig, mod, *leafify(xa, xk, grad_idx), cots)
                state_orig = {k: v.clone() for k, v in mod.state_dict().items()}
                mod.load_state_dict(state0)
                cand = fwd_bwd(call_impl, mod, *leafify(xa, xk, grad_idx), cots)
            else:
                with torch.no_grad():
                    mod32 = fp32_copy()
                    with pure_fp32():
                        ref32 = call_ref(mod32, to_float32(xa), to_float32(xk))
                    del mod32
                    orig = call_orig(mod, xa, xk)
                    state_orig = {k: v.clone() for k, v in mod.state_dict().items()}
                    mod.load_state_dict(state0)
                    cand = call_impl(mod, xa, xk)
            torch.cuda.synchronize()
            state_diff = [k for k, v in mod.state_dict().items()
                          if not torch.allclose(v.float(), state_orig[k].float(), rtol=1e-2, atol=1e-5)]
            mod.load_state_dict(state_orig)
            if seed is None:  # same inputs twice: flag run-to-run nondeterminism (e.g. float atomics)
                again = (fwd_bwd(call_impl, mod, *leafify(xa, xk, grad_idx), cots) if train
                         else torch.no_grad()(lambda: call_impl(mod, xa, xk))())
                mod.load_state_dict(state_orig)
                nondet = [i for i, (u, v) in enumerate(zip(tensors(cand), tensors(again))) if not torch.equal(u, v)]
                result["nondeterministic_outputs"] = nondet
                if nondet:
                    print(f"note: impl is not bitwise deterministic run-to-run (tensors {nondet}); fine for "
                          "inference, report it for training reproducibility")
        except Exception as e:  # noqa: BLE001 - report, don't crash
            print(f"[{label}] impl raised: {type(e).__name__}: {e}")
            result["checks"].append({"inputs": label, "error": f"{type(e).__name__}: {e}"})
            ok = False
            break
        mutated = [j for j, (b, c) in enumerate(zip(before, tensors((xa, xk)))) if not torch.equal(b, c)]
        res = compare(ref32, orig, cand, slack=a.slack)
        if mutated:
            res["pass"] = False
            res["reason"] = f"impl modified its input tensors {mutated}"
        if state_diff:
            res["pass"] = False
            res["reason"] = (res.get("reason", "") + "; " if res.get("reason") else "") + \
                f"module state differs from the original's after the call: {state_diff[:5]}"
        res["inputs"] = label
        result["checks"].append(res)
        print(f"[{label}] " + format_compare(res, names, compact=train))
        if not res["pass"]:
            ok = False
            break
    if ok and a.compile:
        # The impl inside a compiled model: must compile, should not graph-break, must stay correct
        # (Inductor re-fuses and re-rounds everything around the impl's own kernels).
        xa, xk = randomize_like(args, 7), randomize_like(kwargs, 7)
        try:
            torch._dynamo.reset()
            with autocast_ctx(ac), (torch.enable_grad() if train else torch.no_grad()):
                ex = torch._dynamo.explain(lambda *x, **y: impl.forward(mod, *x, **y))(*xa, **xk)
            breaks = {"graphs": ex.graph_count, "graph_breaks": ex.graph_break_count,
                      "reasons": [str(r.reason).splitlines()[0][:160] for r in ex.break_reasons[:5]]}
            torch._dynamo.reset()
            mod32 = fp32_copy()
            if train:
                cots = randomize_like(grad_out, 57)
                with pure_fp32():
                    ref32 = fwd_bwd(call_ref, mod32, *to_float32(leafify(xa, xk, grad_idx)), to_float32(cots))
                orig = fwd_bwd(call_orig, mod, *leafify(xa, xk, grad_idx), cots)
                cand = fwd_bwd(call_compiled_impl, mod, *leafify(xa, xk, grad_idx), cots)
            else:
                with torch.no_grad():
                    with pure_fp32():
                        ref32 = call_ref(mod32, to_float32(xa), to_float32(xk))
                    orig = call_orig(mod, xa, xk)
                    cand = call_compiled_impl(mod, xa, xk)
            del mod32
            res = compare(ref32, orig, cand, slack=a.slack)
        except Exception as e:  # noqa: BLE001
            breaks, res = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}, {"pass": False}
        result["compiled_impl"] = {**breaks, "check": res}
        print(f"[torch.compile(impl)] " + (format_compare(res, names, compact=train) if "outputs" in res
                                            else f"FAILED to compile: {breaks.get('error')}"))
        if "graph_breaks" in breaks:
            print(f"  {breaks['graphs']} graph(s), {breaks['graph_breaks']} graph break(s)"
                  + ("".join(f"\n    break: {r}" for r in breaks["reasons"]) if breaks["graph_breaks"] else ""))
        ok &= res["pass"]
    result["correct"] = ok

    if ok and not a.no_time:
        params = sum(p.numel() * p.element_size() for p in mod.parameters())
        out_bytes = nbytes(tensors(orig)[: len(meta["outputs"])])
        # Weights larger than ~2x L2 are evicted by the call itself: copying them only costs memory.
        share_weights = params > 2 * l2_bytes()
        act = (2 if train else 1) * (nbytes(args) + nbytes(kwargs)) + out_bytes
        k = rotation_count(act + (0 if share_weights else (2 if train else 1) * params))
        copies = [(mod, args, kwargs)] + [
            (mod if share_weights else copy.deepcopy(mod), randomize_like(args, 10 + j), randomize_like(kwargs, 10 + j))
            for j in range(k - 1)]
        variants = {"original": call_orig, "impl": call_impl}
        if a.compile:
            compiled = torch.compile(lambda p, x, y: functional_call(mod, p, x, y), dynamic=False)
            pd = {id(m): dict(m.named_parameters()) | dict(m.named_buffers()) for m, _, _ in copies}
            def call_compiled(m, x, y):
                with autocast_ctx(ac):
                    return compiled(pd[id(m)], x, y)
            variants["compiled_original"] = call_compiled
            variants["compiled_impl"] = call_compiled_impl
        memory, fns = {}, {}
        for name, call in variants.items():
            if train:
                steps = []
                for j, (m, x, y) in enumerate(copies):
                    lx, ly = leafify(x, y, grad_idx)
                    cots = grad_out if j == 0 else randomize_like(grad_out, 70 + j)
                    steps.append(lambda c=call, m=m, x=lx, y=ly, g=cots: fwd_bwd(c, m, x, y, g))
                fns[name] = steps
                memory[name] = round(peak_extra_mb(steps[0]), 1)
            else:
                fns[name] = [lambda c=call, m=m, x=x, y=y: torch.no_grad()(lambda: c(m, x, y))() for m, x, y in copies]
                memory[name] = round(peak_extra_mb(fns[name][0]), 1)
        timing = time_ab(fns)
        del copies
        result["timing"], result["peak_extra_mb"] = timing, memory
        # Ceiling for THIS GPU (the region may have been captured on another one).
        peak_f, peak_b = peaks(hw_info(), effective_dtype(mod, ac))
        ceil = round(max(meta["flops"] / peak_f if peak_f else 0,
                         meta["boundary_mb"] * 1e6 / peak_b if peak_b else 0) * 1e6, 1)
        result["ceiling_us"] = ceil
        t_impl = timing["impl"]
        print(f"\ntiming {'forward + backward' if train else 'forward'} (L2-cold x{t_impl['copies']}, {t_impl['mode']}):")
        for name, t in timing.items():
            frac = f"  {ceil / (t['device_ms'] * 1000):5.1%} of ceiling" if ceil else ""
            print(f"  {name:<18} device {t['device_ms'] * 1000:9.1f} us ±{t['noise'] / 2:4.1%}   "
                  f"host {t['host_ms'] * 1000:9.1f} us ±{t['host_noise'] / 2:4.0%}   peak +{memory[name]:8.1f} MB{frac}")
        base = timing["original"]["device_ms"]
        best_base = min(t["device_ms"] for n, t in timing.items() if n not in ("impl", "compiled_impl"))
        if "compiled_impl" in timing:
            result["compiled_speedup_vs_compiled_original"] = (timing["compiled_original"]["device_ms"]
                                                               / timing["compiled_impl"]["device_ms"])
        result["speedup"] = base / t_impl["device_ms"]
        result["speedup_vs_best_baseline"] = best_base / t_impl["device_ms"]
        result["significance"] = {n: significance(t, t_impl) for n, t in timing.items() if n not in ("impl", "compiled_impl")}
        print(f"  speedup {result['speedup']:.2f}x vs eager"
              + (f", {result['speedup_vs_best_baseline']:.2f}x vs best baseline" if a.compile else "")
              + f"; ceiling {ceil} us ({hw_info().get('device', 'run hw.py')})")
        if "compiled_impl" in timing:
            print(f"  in a compiled model: compiled impl {result['compiled_speedup_vs_compiled_original']:.2f}x vs "
                  f"compiled original ({significance(timing['compiled_original'], timing['compiled_impl'])})")
        print("  vs " + ", ".join(f"{n}: {v}" for n, v in result["significance"].items())
              + "  (5 interleaved rounds; 'within noise' = round ranges overlap)")
        if t_impl["mode"] != "cuda_graph":
            print(f"  note: {t_impl.get('note')}")

    n = next_result(d)
    path = d / "results" / f"{n:03d}.json"
    write_json(path, result)
    shutil.copy2(impl_path, d / "results" / f"{n:03d}_impl.py")
    print(f"\nsaved {path} — log it in NOTES.md; cite this path in LESSONS.md")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
