#!/usr/bin/env python3
"""Put one or more impls back into the real model; check the WHOLE model and time it end to end.

  swap.py --entry model_entry.py --impl work/dit_block0 [--impl work/other] [--all-instances] [--compile]

This is the final word: a region can pass bench.py and still break the model (e.g. it relied on an
input property only the captured activations had). Passing criterion is the same as bench.py:
model outputs vs a float32 run of the original model, within `slack` x the original's own error,
on the build() inputs and on one fresh random draw.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from common import (compare, format_compare, load_entry, load_py, randomize_like, read_json, run_model,  # noqa: E402
                    time_calls, to_float32, write_json)
from hotspots import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--impl", action="append", required=True, help="region dir with impl.py (repeatable)")
    ap.add_argument("--all-instances", action="store_true",
                    help="apply each impl to every module of the same class and parameter shapes")
    ap.add_argument("--compile", action="store_true", help="also time torch.compile(original model)")
    ap.add_argument("--slack", type=float, default=2.0)
    a = ap.parse_args()

    entry = load_entry(Path(a.entry))
    model, inputs = entry.build()
    mods = {(q or ROOT): m for q, m in model.named_modules()}
    patches = []
    for d in map(lambda p: Path(p).resolve(), a.impl):
        meta = read_json(d / "region.json")
        sha = hashlib.sha1((d / "impl.py").read_bytes()).hexdigest()[:10]
        impl = load_py(d / "impl.py", name=f"krsi_swap_{sha}")
        targets = [meta["module"]] + (meta.get("same_class_instances", []) if a.all_instances else [])
        for q in targets:
            if q not in mods:
                raise SystemExit(f"{d}: module {q} not found in this model")
            patches.append((q, mods[q], impl, str(d), sha))

    def install():
        for _, m, impl, _, _ in patches:
            m.forward = types.MethodType(lambda self, *x, _i=impl, **k: _i.forward(self, *x, **k), m)

    def uninstall():
        for _, m, _, _, _ in patches:
            m.__dict__.pop("forward", None)

    result = {"entry": str(Path(a.entry).resolve()), "patched": [(q, d, s) for q, _, _, d, s in patches],
              "device": torch.cuda.get_device_name(0), "checks": []}
    ok = True
    with torch.no_grad():
        model32 = copy.deepcopy(model).float()
        for label, x in (("build() inputs", inputs), ("random seed 1", randomize_like(inputs, 1))):
            ref32 = run_model(entry, model32, to_float32(x))
            orig = run_model(entry, model, x)
            install()
            try:
                cand = run_model(entry, model, x)
            finally:
                uninstall()
            res = compare(ref32, orig, cand, slack=a.slack)
            res["inputs"] = label
            result["checks"].append(res)
            print(f"[{label}] " + format_compare(res))
            ok &= res["pass"]
        del model32
        torch.cuda.empty_cache()
        if ok:
            timing = {"original": time_calls([lambda: run_model(entry, model, inputs)])}
            install()
            timing["patched"] = time_calls([lambda: run_model(entry, model, inputs)])
            uninstall()
            if a.compile:
                step = torch.compile(lambda x: run_model(entry, model, x))  # whole step, any run()
                timing["compiled_original"] = time_calls([lambda: step(inputs)])
            result["timing"] = timing
            print(f"\nwhole model ({len(patches)} module(s) patched):")
            for name, t in timing.items():
                print(f"  {name:<18} device {t['device_ms'] * 1000:9.1f} us ({t['mode']})   "
                      f"eager host-inclusive {t['host_ms'] * 1000:9.1f} us")
            for key in ("device_ms", "host_ms"):
                result[f"speedup_{key}"] = timing["original"][key] / timing["patched"][key]
            print(f"  speedup: {result['speedup_device_ms']:.3f}x device, {result['speedup_host_ms']:.3f}x eager")
    result["correct"] = ok
    out = Path(a.impl[0]).resolve() / "results"
    out.mkdir(exist_ok=True)
    n = 1 + max([int(p.stem[5:]) for p in out.glob("swap_[0-9][0-9][0-9].json")], default=0)
    write_json(out / f"swap_{n:03d}.json", result)
    print(f"\nsaved {out / f'swap_{n:03d}.json'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
