#!/usr/bin/env python3
"""Check LESSONS.md: every lesson cites evidence that exists, and the file stays small.

`seed` is accepted as evidence only inside the "## Seed" section (lessons shipped with the skill,
measured during its development). Every other lesson must cite files that exist, absolute or
relative to the skill directory.
"""

import re
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "LESSONS.md"
MAX_LINES = 150


def main() -> None:
    lines = PATH.read_text().splitlines()
    problems, in_seed = [], False
    for n, line in enumerate(lines, 1):
        if line.startswith("## "):
            in_seed = line[3:].strip().lower().startswith("seed")
        if not line.lstrip().startswith("- "):
            continue
        m = re.search(r"\[evidence:\s*([^\]]+)\]", line)
        if not m:
            problems.append(f"line {n}: lesson without [evidence: ...]")
            continue
        refs = [r.strip() for r in re.split(r"[,;]\s*", m.group(1)) if r.strip()]
        for ref in refs:
            if ref == "seed":
                if not in_seed:
                    problems.append(f"line {n}: 'seed' evidence is only allowed in the Seed section; cite a result file")
                continue
            path = Path(ref) if ref.startswith("/") else PATH.parent / ref
            if not path.exists():
                problems.append(f"line {n}: cited file does not exist: {ref}")
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines > {MAX_LINES}: condense (merge, delete superseded lessons)")
    print("\n".join(problems) if problems else f"LESSONS.md ok ({len(lines)} lines)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
