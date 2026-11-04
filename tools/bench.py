#!/usr/bin/env python3
"""Check and time impl.py of a captured region against the original module.

  bench.py work/dit_block0 [--compile] [--slack 2] [--impl other_impl.py]

Correctness: on the real captured inputs and 3 fresh random draws (same per-tensor stats; masks
and integer tensors kept), the impl's error vs a float32 run of the original must be within
`slack` x the original's own low-precision error. On failure it prints where the bad elements are
(index ranges per dimension), which usually points straight at the bug.

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

from common import (compare, compute_dtype, format_compare, hw_info, load_py, peaks, nbytes, randomize_like, read_json,  # noqa: E402
                    rotation_count, time_calls, to_float32, write_json)


def load_region(d: Path):
    meta = read_json(d / "region.json")
    load_py(Path(meta["entry"]))  # makes the module's classes importable for torch.load
    mod = torch.load(d / "module.pt", weights_only=False).cuda().eval()
    io = torch.load(d / "io.pt", weights_only=False)
    cuda = lambda x: tree_map(lambda t: t.cuda() if isinstance(t, torch.Tensor) else t, x)  # noqa: E731
    return meta, mod, cuda(io["args"]), cuda(io["kwargs"])


def next_result(d: Path) -> int:
    (d / "results").mkdir(exist_ok=True)
    nums = [int(p.stem) for p in (d / "results").glob("[0-9][0-9][0-9].json")]
    return max(nums, default=0) + 1


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
    meta, mod, args, kwargs = load_region(d)
    sha = hashlib.sha1(impl_path.read_bytes()).hexdigest()[:10]
    impl = load_py(impl_path, name=f"krsi_impl_{sha}")
    mod32 = copy.deepcopy(mod).float()

    result = {"region": str(d), "module": meta["module"], "impl_sha": sha, "note": a.note,
              "device": torch.cuda.get_device_name(0), "slack": a.slack, "checks": []}
    ok = True
    with torch.no_grad():
        for i, seed in enumerate([None, 1, 2, 3]):
            xa, xk = (args, kwargs) if seed is None else (randomize_like(args, seed), randomize_like(kwargs, seed))
            before = [t.clone() for t in (list(xa) + list(xk.values())) if isinstance(t, torch.Tensor)]
            ref32 = mod32(*to_float32(xa), **to_float32(xk))
            orig = mod(*xa, **xk)
            try:
                cand = impl.forward(mod, *xa, **xk)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 - report, don't crash
                print(f"impl raised on {'captured inputs' if seed is None else f'seed {seed}'}: "
                      f"{type(e).__name__}: {e}")
                result["checks"].append({"inputs": seed or "captured", "error": f"{type(e).__name__}: {e}"})
                ok = False
                break
            after = [t for t in (list(xa) + list(xk.values())) if isinstance(t, torch.Tensor)]
            mutated = [j for j, (b, c) in enumerate(zip(before, after)) if not torch.equal(b, c)]
            res = compare(ref32, orig, cand, slack=a.slack)
            if mutated:
                res["pass"] = False
                res["reason"] = f"impl modified its input tensors {mutated}"
            res["inputs"] = "captured" if seed is None else f"random seed {seed}"
            result["checks"].append(res)
            print(f"[{res['inputs']}] " + format_compare(res))
            if not res["pass"]:
                ok = False
                break
    result["correct"] = ok

    if ok and not a.no_time:
        params = sum(p.numel() * p.element_size() for p in mod.parameters())
        k = rotation_count(nbytes(args) + nbytes(kwargs) + params + nbytes(orig))
        copies = [(mod, args, kwargs)] + [
            (copy.deepcopy(mod), randomize_like(args, 10 + j), randomize_like(kwargs, 10 + j)) for j in range(k - 1)]
        with torch.no_grad():
            t_orig = time_calls([lambda m=m, x=x, y=y: m(*x, **y) for m, x, y in copies])
            t_impl = time_calls([lambda m=m, x=x, y=y: impl.forward(m, *x, **y) for m, x, y in copies])
            timing = {"original": t_orig, "impl": t_impl}
            if a.compile:
                fn = torch.compile(lambda p, x, y: functional_call(mod, p, x, y), dynamic=False)
                ps = [dict(m.named_parameters()) | dict(m.named_buffers()) for m, _, _ in copies]
                timing["compiled_original"] = time_calls(
                    [lambda p=p, x=x, y=y: fn(p, x, y) for p, (_, x, y) in zip(ps, copies)])
        del copies
        result["timing"] = timing
        # Ceiling for THIS GPU (the region may have been captured on another one).
        peak_f, peak_b = peaks(hw_info(), compute_dtype(mod))
        ceil = round(max(meta["flops"] / peak_f if peak_f else 0,
                         meta["boundary_mb"] * 1e6 / peak_b if peak_b else 0) * 1e6, 1)
        result["ceiling_us"] = ceil
        print(f"\ntiming (L2-cold x{t_orig['copies']}, {t_impl['mode']}):")
        for name, t in timing.items():
            frac = f"  {ceil / (t['device_ms'] * 1000):5.1%} of ceiling" if ceil else ""
            print(f"  {name:<18} device {t['device_ms'] * 1000:9.1f} us   host {t['host_ms'] * 1000:9.1f} us{frac}")
        base = timing["original"]["device_ms"]
        best_base = min(t["device_ms"] for n, t in timing.items() if n != "impl")
        result["speedup"] = base / t_impl["device_ms"]
        result["speedup_vs_best_baseline"] = best_base / t_impl["device_ms"]
        print(f"  speedup {result['speedup']:.2f}x vs eager"
              + (f", {result['speedup_vs_best_baseline']:.2f}x vs best baseline" if a.compile else "")
              + f"; ceiling {ceil} us ({hw_info().get('device', 'run hw.py')})")
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
