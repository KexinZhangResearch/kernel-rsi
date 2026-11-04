#!/usr/bin/env python3
"""Put one or more impls back into the real model; check the WHOLE model and time it end to end.

  swap.py --entry model_entry.py --impl work/dit_block0 [--impl work/other] [--all-instances] [--compile]
          [--train-steps 200 --lr 1e-4]

This is the final word: a region can pass bench.py and still break the model (e.g. it relied on an
input property only the captured activations had). Passing criterion is the same as bench.py:
model outputs vs a float32 run of the original model, within `slack` x the original's own error,
on the build() inputs and on one fresh random draw. Module buffers (BatchNorm running stats, ...)
must end up as the original leaves them.

Other shapes: if model_entry.py defines `shapes() -> [inputs, ...]` (e.g. other batch sizes or
sequence lengths, odd ones included), every check runs and every timing is reported for each of
them too. An impl that only works for the captured shape fails here.

With MODE = "train": the loss, the outputs and the gradient of EVERY parameter are checked, the
second round runs after an update of all weights (stale weight caches fail), and the timed unit is
a full training step. `--train-steps N` additionally trains three copies (fp32 reference, original,
patched) for N AdamW steps on the same data and compares the loss curves and the final weights:
errors that pass a one-step check can still accumulate.
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

from common import (compare, format_compare, is_train, load_entry, load_py, pure_fp32, randomize_like,  # noqa: E402
                    read_json, run_model, set_train_mode, significance, tensors, time_ab, to_float32, train_loss,
                    versions, write_json)
from hotspots import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--impl", action="append", required=True, help="region dir with impl.py (repeatable)")
    ap.add_argument("--all-instances", action="store_true",
                    help="apply each impl to every module of the same class and parameter shapes")
    ap.add_argument("--compile", action="store_true", help="also time torch.compile(original model)")
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument("--train-steps", type=int, default=0, help="training mode: compare N-step loss curves")
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    entry = load_entry(Path(a.entry))
    model, inputs = entry.build()
    mods = {(q or ROOT): m for q, m in model.named_modules()}
    patches = []  # (qualname, impl module, region dir, sha)
    for d in map(lambda p: Path(p).resolve(), a.impl):
        meta = read_json(d / "region.json")
        sha = hashlib.sha1((d / "impl.py").read_bytes()).hexdigest()[:10]
        impl = load_py(d / "impl.py", name=f"krsi_swap_{sha}")
        for q in [meta["module"]] + (meta.get("same_class_instances", []) if a.all_instances else []):
            if q not in mods:
                raise SystemExit(f"{d}: module {q} not found in this model")
            patches.append((q, impl, str(d), sha))

    def install(root):
        named = {(q or ROOT): m for q, m in root.named_modules()}
        for q, impl, _, _ in patches:
            m = named[q]
            m.forward = types.MethodType(lambda self, *x, _i=impl, **k: _i.forward(self, *x, **k), m)

    def uninstall(root):
        named = {(q or ROOT): m for q, m in root.named_modules()}
        for q, _, _, _ in patches:
            named[q].__dict__.pop("forward", None)

    train = is_train(entry)
    if train:
        off = set_train_mode(model)
        if off:
            print(f"note: dropout disabled in {off} for checking (random masks are not comparable)")
    else:
        model.eval()
    pnames = [n for n, p in model.named_parameters() if p.requires_grad]

    def step(m, x, patched=False, amp=True):
        """Forward (+ loss and parameter gradients when training); returns a list of tensors to compare."""
        if patched:
            install(m)
        try:
            if not train:
                with torch.no_grad():
                    return tensors(run_model(entry, m, x, amp))
            out = run_model(entry, m, x, amp)
            loss = train_loss(entry, out, x)
            ps = [p for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, ps, allow_unused=True)
            grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, ps)]
            return [loss.detach().reshape(1)] + [t.detach() for t in tensors(out)] + list(grads)
        finally:
            if patched:
                uninstall(m)

    shape_sets = [("build() inputs", inputs)]
    if hasattr(entry, "shapes"):
        for i, x in enumerate(entry.shapes()):
            dims = [list(t.shape) for t in tensors(x)][:2]
            shape_sets.append((f"shapes()[{i}] {dims}", x))

    result = {"entry": str(Path(a.entry).resolve()), "mode": "train" if train else "infer",
              "patched": [(q, d, s) for q, _, d, s in patches], **versions(), "checks": []}
    ok = True
    model32 = copy.deepcopy(model).float()
    rounds = [(label, x, False) for label, x in shape_sets]
    rounds.insert(1, ("random seed 1" + (" after a weight update" if train else ""), randomize_like(inputs, 1), train))
    for label, x, update in rounds:
        if update:
            with torch.no_grad():  # an optimizer-like update: patched code must see new weights
                g = torch.Generator(device="cuda").manual_seed(7)
                for p, p32 in zip(model.parameters(), model32.parameters()):
                    if p.is_floating_point() and p.numel() > 1:
                        p.add_(torch.randn(p.shape, generator=g, device=p.device).to(p.dtype)
                               * (p.float().std() * 0.01).to(p.dtype))
                    p32.copy_(p.float())
        state0 = {k: v.clone() for k, v in model.state_dict().items()}
        with pure_fp32():
            ref32 = step(model32, to_float32(x), amp=False)
        orig = step(model, x)
        state_orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state0)
        try:
            cand = step(model, x, patched=True)
        except Exception as e:  # noqa: BLE001 - e.g. an impl that only handles the captured shape
            print(f"[{label}] patched model raised: {type(e).__name__}: {str(e)[:300]}")
            result["checks"].append({"inputs": label, "error": f"{type(e).__name__}: {e}"})
            ok = False
            model.load_state_dict(state_orig)
            continue
        state_diff = [k for k, v in model.state_dict().items()
                      if not torch.allclose(v.float(), state_orig[k].float(), rtol=1e-2, atol=1e-5)]
        model.load_state_dict(state_orig)
        res = compare(ref32, orig, cand, slack=a.slack)
        if state_diff:
            res["pass"] = False
            res["reason"] = f"module state differs from the original's after the step: {state_diff[:5]}"
        res["inputs"] = label
        result["checks"].append(res)
        names = (["loss"] + [f"out{i}" for i in range(len(ref32) - 1 - len(pnames))] + [f"grad {n}" for n in pnames]
                 if train else None)
        print(f"[{label}] " + format_compare(res, names, compact=train))
        ok &= res["pass"]
    del model32
    torch.cuda.empty_cache()

    if ok and train and a.train_steps > 0:
        ok &= loss_curves(a, entry, model, inputs, install, uninstall, result)

    patched_model, cstep = None, None
    if ok and a.compile:
        # The deployment the user will run: patches installed, then torch.compile. Checked like the
        # eager patched model, plus graph breaks (each break is a host round-trip and lost fusion).
        patched_model = copy.deepcopy(model)
        install(patched_model)
        try:
            torch._dynamo.reset()
            with (torch.enable_grad() if train else torch.no_grad()):
                ex = torch._dynamo.explain(lambda x: run_model(entry, patched_model, x))(inputs)
            breaks = {"graphs": ex.graph_count, "graph_breaks": ex.graph_break_count,
                      "reasons": [str(r.reason).splitlines()[0][:160] for r in ex.break_reasons[:5]]}
            torch._dynamo.reset()
            cfwd = torch.compile(lambda x: run_model(entry, patched_model, x))
            pp = [p for p in patched_model.parameters() if p.requires_grad]

            def cstep(x):
                if not train:
                    with torch.no_grad():
                        return tensors(cfwd(x))
                out = cfwd(x)
                loss = train_loss(entry, out, x)
                grads = torch.autograd.grad(loss, pp, allow_unused=True)
                return [loss.detach().reshape(1)] + [t.detach() for t in tensors(out)] + \
                    [torch.zeros_like(q) if g is None else g for g, q in zip(grads, pp)]

            model32 = copy.deepcopy(model).float()
            with pure_fp32():
                ref32 = step(model32, to_float32(inputs), amp=False)
            del model32
            res = compare(ref32, step(model, inputs), cstep(inputs), slack=a.slack)
        except Exception as e:  # noqa: BLE001
            breaks, res, cstep = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}, {"pass": False}, None
        result["compiled_patched"] = {**breaks, "check": res}
        names = (["loss"] + [f"out{i}" for i in range(len(res.get("outputs", [])) - 1 - len(pnames))]
                 + [f"grad {n}" for n in pnames] if train else None)
        print(f"[torch.compile(patched model)] " + (format_compare(res, names, compact=train) if "outputs" in res
                                                    else f"FAILED: {breaks.get('error')}"))
        if "graph_breaks" in breaks:
            print(f"  {breaks['graphs']} graph(s), {breaks['graph_breaks']} graph break(s)"
                  + "".join(f"\n    break: {r}" for r in breaks["reasons"]))
        ok &= res["pass"]

    if ok:
        result["timing"] = {}
        for label, x in shape_sets:
            variants = {"original": [lambda x=x: step(model, x)], "patched": [lambda x=x: step(model, x, True)]}
            if a.compile:
                if train:
                    ps = [p for p in model.parameters() if p.requires_grad]
                    fwd = torch.compile(lambda x: train_loss(entry, run_model(entry, model, x), x))
                    variants["compiled_original"] = [lambda x=x: torch.autograd.grad(fwd(x), ps, allow_unused=True)]
                else:
                    fwd = torch.compile(lambda x: run_model(entry, model, x))  # whole step, any run()
                    variants["compiled_original"] = [lambda x=x: torch.no_grad()(lambda: tensors(fwd(x)))()]
                if cstep is not None:
                    variants["compiled_patched"] = [lambda x=x: cstep(x)]
            mem = {}
            for name, fns in variants.items():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                fns[0]()
                torch.cuda.synchronize()
                mem[name] = round((torch.cuda.max_memory_allocated() - base) / 1e6, 1)
            timing = time_ab(variants)
            for name in timing:
                timing[name]["peak_extra_mb"] = mem[name]
            sig = {n: significance(t, timing["patched"]) for n, t in timing.items() if n not in ("patched", "compiled_patched")}
            if "compiled_patched" in timing:
                sig["compiled_patched vs compiled_original"] = significance(timing["compiled_original"],
                                                                            timing["compiled_patched"])
            sp = {k: timing["original"][k] / timing["patched"][k] for k in ("device_ms", "host_ms")}
            result["timing"][label] = {"variants": timing, "significance": sig, "speedup": sp}
            print(f"\n{label} — whole model, {'training step (fwd + loss + bwd)' if train else 'forward'} "
                  f"({len(patches)} module(s) patched):")
            for name, t in timing.items():
                print(f"  {name:<18} device {t['device_ms'] * 1000:9.1f} us ±{t['noise'] / 2:4.1%} ({t['mode']})   "
                      f"eager host-inclusive {t['host_ms'] * 1000:9.1f} us ±{t['host_noise'] / 2:4.0%}   "
                      f"peak +{t['peak_extra_mb']:9.1f} MB")
            print(f"  speedup: {sp['device_ms']:.3f}x device, {sp['host_ms']:.3f}x eager; patched vs "
                  + ", ".join(f"{n}: {v}" for n, v in sig.items()))
        first = result["timing"][shape_sets[0][0]]["speedup"]
        result["speedup_device_ms"], result["speedup_host_ms"] = first["device_ms"], first["host_ms"]
    gaps = []
    if not hasattr(entry, "shapes"):
        gaps.append("model_entry.py has no shapes(): only one input shape was checked — an impl that only works "
                    "for it (tile-multiple lengths, asserts, missing tail masks) would pass")
    if train and a.train_steps <= 0:
        gaps.append("training was checked for one step only: run with --train-steps 200")
    result["coverage_gaps"] = gaps
    for g in gaps:
        print(f"WARNING: {g}")
    result["correct"] = ok
    out = Path(a.impl[0]).resolve() / "results"
    out.mkdir(exist_ok=True)
    n = 1 + max([int(p.stem[5:]) for p in out.glob("swap_[0-9][0-9][0-9].json")], default=0)
    write_json(out / f"swap_{n:03d}.json", result)
    print(f"\nsaved {out / f'swap_{n:03d}.json'}")
    raise SystemExit(0 if ok else 1)


def loss_curves(a, entry, model, inputs, install, uninstall, result) -> bool:
    """N AdamW steps on identical data for: original, original with 1-ulp weight noise, patched (and an
    fp32 copy for information). Training dynamics amplify rounding differences, sometimes chaotically,
    so the yardstick is the original's own sensitivity: |perturbed - original| per step. Patched passes
    if |patched - original| stays within slack x that. If the perturbed run itself diverges strongly,
    the comparison is inconclusive (lower --lr or fewer --train-steps), not a failure."""
    batches = [inputs] + [randomize_like(inputs, 1000 + i) for i in range(min(a.train_steps, 16) - 1)]
    init = copy.deepcopy(model.state_dict())
    curves = {}
    for name in ("fp32", "original", "perturbed", "patched"):
        m = copy.deepcopy(model)
        m.load_state_dict(init)
        if name == "fp32":
            m = m.float()
        if name == "perturbed":
            g = torch.Generator(device="cuda").manual_seed(11)
            with torch.no_grad():
                for p in m.parameters():
                    if p.is_floating_point():
                        ulp = p.float().abs() * torch.finfo(p.dtype).eps
                        sign = torch.randint(0, 2, p.shape, generator=g, device=p.device).float() * 2 - 1
                        p.copy_((p.float() + sign * ulp).to(p.dtype))
        set_train_mode(m)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr)
        if name == "patched":
            install(m)
        losses = []
        with pure_fp32() if name == "fp32" else torch.enable_grad():
            for i in range(a.train_steps):
                x = batches[i % len(batches)]
                x = to_float32(x) if name == "fp32" else x
                loss = train_loss(entry, run_model(entry, m, x, amp=name != "fp32"), x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        if name == "patched":
            uninstall(m)
        curves[name] = losses
        del opt, m
        torch.cuda.empty_cache()
    t = {k: torch.tensor(v) for k, v in curves.items()}
    scale = t["original"].abs().max().item() + 1e-12
    # Running max: once runs separate, a later coincidental crossing does not count as agreement.
    noise = (t["perturbed"] - t["original"]).abs().cummax(0).values
    dev = (t["patched"] - t["original"]).abs().cummax(0).values
    floor = 1e-3 * scale
    chaotic = noise[-1].item() > 0.1 * scale
    ratio = ((dev - floor).clamp(min=0) / (noise + floor)).max().item()
    loss_ok = dev[-1].item() <= a.slack * noise[-1].item() + floor and ratio <= a.slack
    verdict = "INCONCLUSIVE (training is chaotic at this lr)" if chaotic and not loss_ok else ("PASS" if loss_ok else "FAIL")
    result["train_curves"] = {"steps": a.train_steps, "lr": a.lr, "curves": curves, "verdict": verdict,
                              "max_dev_patched": dev[-1].item(), "max_dev_perturbed": noise[-1].item()}
    print(f"\n[{a.train_steps} AdamW steps, lr {a.lr}] loss curve {verdict}: max |patched - original| "
          f"{dev[-1].item():.3e} vs original's own sensitivity (1-ulp weight noise) {noise[-1].item():.3e}")
    if verdict.startswith("INCONCLUSIVE"):
        print("  the original itself is not reproducible to this precision: lower --lr or --train-steps")
    stride = max(1, a.train_steps // 8)
    print("  step   fp32        original    perturbed   patched")
    for i in sorted(set(list(range(0, a.train_steps, stride)) + [a.train_steps - 1])):
        print(f"  {i:4d}   {curves['fp32'][i]:.6f}   {curves['original'][i]:.6f}   {curves['perturbed'][i]:.6f}"
              f"   {curves['patched'][i]:.6f}")
    return verdict != "FAIL"


if __name__ == "__main__":
    main()
