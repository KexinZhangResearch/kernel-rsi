#!/usr/bin/env python3
"""Re-verify the skill's knowledge on THIS machine, without an LLM.

  replay.py                 # run every library/*.py selftest()
  replay.py --regions DIR   # also re-bench every captured region under DIR (impl.py vs original)

Run it after a new GPU, torch/triton/driver upgrade, or before trusting LESSONS.md on a new setup.
Results go to library/replay/<gpu>.json; lessons can cite them as evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SKILL, slug, versions, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", help="directory containing captured regions (each with region.json + impl.py)")
    a = ap.parse_args()
    info = versions()
    out = {"versions": info, "library": {}, "regions": {}}
    for f in sorted(p for p in (SKILL / "library").glob("*.py") if not p.name.startswith("_")):
        try:
            spec = importlib.util.spec_from_file_location(f"krsi_lib_{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            r = mod.selftest()
            out["library"][f.stem] = {k: v for k, v in r.items() if k != "checks"}
            c = r.get("compiled", {})
            ctxt = (f" | compile(fullgraph): {'ok' if c.get('pass') else 'FAIL'}"
                    + (f" {c['kernel_us']:.1f} us vs compiled ref {c['reference_us']:.1f} us {c['speedup']:.2f}x "
                       f"({c['significance']})" if "kernel_us" in c else f" {c.get('error', '')}"))
            print(f"{f.stem:<18} {'PASS' if r['pass'] else 'FAIL'}  eager: {r['kernel_us']:.1f} us vs "
                  f"{r['reference_us']:.1f} us {r['speedup']:.2f}x ({r['significance']}){ctxt}")
        except Exception as e:  # noqa: BLE001 - one broken component must not hide the others
            out["library"][f.stem] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            print(f"{f.stem:<22} ERROR {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=1)
    if a.regions:
        for d in sorted(p.parent for p in Path(a.regions).glob("*/region.json")):
            p = subprocess.run([sys.executable, str(SKILL / "tools" / "bench.py"), str(d), "--note", "replay"],
                               text=True, capture_output=True)
            line = next((l for l in p.stdout.splitlines() if l.strip().startswith("speedup")), p.stdout[-200:])
            out["regions"][str(d)] = {"pass": p.returncode == 0, "summary": line.strip()}
            print(f"{d.name:<22} {'PASS' if p.returncode == 0 else 'FAIL'}  {line.strip()}")
    path = SKILL / "library" / "replay" / f"{slug(info['device'])}.json"
    write_json(path, out)
    bad = [k for k, v in {**out["library"], **out["regions"]}.items() if not v["pass"]]
    print(f"\nsaved {path}" + (f"\nFAILED: {bad} — fix them or correct the lessons that rely on them" if bad else ""))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
