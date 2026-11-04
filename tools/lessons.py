#!/usr/bin/env python3
"""Check LESSONS.md: every lesson cites evidence that exists and matches it, and the file stays small.

Rules
- `seed` is accepted as evidence only inside the "## Seed" section (lessons shipped with the skill).
- Every other lesson starts with a tag `[<GPU>, <dtype>...]` and cites files that exist (absolute or
  relative to the skill directory).
- A cited result JSON (from bench.py / swap.py / replay.py) must come from the GPU named in the tag.
- Warnings (not errors): the cited result was measured with a different torch/triton than installed
  now — the lesson may be stale; re-run `tools/replay.py` or re-bench before relying on it.
"""

import json
import re
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "LESSONS.md"
MAX_LINES = 150
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731


def installed() -> dict:
    out = {}
    for name in ("torch", "triton"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out


def cited_meta(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    v = d.get("versions", d) if isinstance(d, dict) else {}
    return {k: v.get(k) for k in ("device", "torch", "triton") if v.get(k)}


def main() -> None:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PATH
    base = PATH.parent  # relative evidence paths are relative to the skill directory
    lines = path.read_text().splitlines()
    problems, warnings, in_seed = [], [], False
    now = installed()
    for n, line in enumerate(lines, 1):
        if line.startswith("## "):
            in_seed = line[3:].strip().lower().startswith("seed")
        if not line.lstrip().startswith("- "):
            continue
        m = re.search(r"\[evidence:\s*([^\]]+)\]", line)
        if not m:
            problems.append(f"line {n}: lesson without [evidence: ...]")
            continue
        tag = re.match(r"\s*-\s*\[([^\]]+)\]", line)
        gpu = tag.group(1).split(",")[0].strip() if tag and not tag.group(1).startswith("evidence") else None
        if not in_seed and not gpu:
            problems.append(f"line {n}: start the lesson with a tag like [H100 SXM, bf16]")
        for ref in [r.strip() for r in re.split(r"[,;]\s*", m.group(1)) if r.strip()]:
            if ref == "seed":
                if not in_seed:
                    problems.append(f"line {n}: 'seed' evidence is only allowed in the Seed section; cite a result file")
                continue
            cited = Path(ref) if ref.startswith("/") else base / ref
            if not cited.exists():
                problems.append(f"line {n}: cited file does not exist: {ref}")
                continue
            meta = cited_meta(cited) if cited.suffix == ".json" else {}
            if gpu and meta.get("device") and norm(gpu) not in norm(meta["device"]):
                problems.append(f"line {n}: tag says '{gpu}' but {ref} was measured on '{meta['device']}'")
            stale = {k: (meta[k], now[k]) for k in ("torch", "triton") if meta.get(k) and now.get(k)
                     and meta[k].split("+")[0] != now[k].split("+")[0]}
            if stale:
                warnings.append(f"line {n}: {ref} measured with " +
                                ", ".join(f"{k} {a} (now {b})" for k, (a, b) in stale.items()) +
                                " — re-verify before relying on it")
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines > {MAX_LINES}: condense (merge, delete superseded lessons)")
    for w in warnings:
        print("warning:", w)
    print("\n".join(problems) if problems else f"LESSONS.md ok ({len(lines)} lines)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
