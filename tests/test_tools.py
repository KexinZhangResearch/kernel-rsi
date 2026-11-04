"""Tests for the skill's own tools: each check must pass correct code AND catch a planted bug.

Run: python -m pytest tests -q        (needs an NVIDIA GPU; ~3-6 minutes)
Every threshold in tools/common.py was set from measurements; these tests pin that behaviour.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from common import compare, significance, time_ab  # noqa: E402


def tool(name, *args, check=None):
    p = subprocess.run([sys.executable, str(TOOLS / name), *map(str, args)], text=True, capture_output=True)
    if check is not None:
        assert (p.returncode == 0) == check, p.stdout[-3000:] + p.stderr[-3000:]
    return p


# ---------------------------------------------------------------- compare()

def _noisy(r, rel, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return r + rel * r.abs().mean() * torch.randn(r.shape, generator=g, device="cuda")


def test_compare_accepts_same_noise_level():
    r = torch.randn(64, 1024, device="cuda")
    assert compare([r], [_noisy(r, 1e-2, 1)], [_noisy(r, 1e-2, 2)])["pass"]


def test_compare_catches_localized_bug_by_rows():
    r = torch.randn(64, 1024, device="cuda")
    c = _noisy(r, 1e-2, 2)
    c[5] *= 1.05  # one row (e.g. one token / head) off by 5%: hardly moves the tensor-wide numbers
    res = compare([r], [_noisy(r, 1e-2, 1)], [c])
    assert not res["pass"] and res["outputs"][0]["bad_rows"] == 1 and res["outputs"][0]["worst_row"]["row"] == 5


def test_compare_catches_nonfinite_and_shape():
    r = torch.randn(8, 128, device="cuda")
    c = r.clone()
    c[0, 0] = float("nan")
    assert not compare([r], [r], [c])["pass"]
    assert not compare([r], [r], [r[:4]])["pass"]


# ---------------------------------------------------------------- timing

def test_failed_capture_does_not_poison_the_process():
    x = torch.randn(1 << 16, device="cuda")
    from common import time_calls  # noqa: PLC0415

    assert time_calls([lambda: (x * 2).sum().item()], reps=2)["mode"] == "eager"
    torch.randn(8, device="cuda")   # raised "Offset increment outside graph capture" before the fix


def test_time_ab_uses_one_mode_for_all_variants():
    x = torch.randn(1 << 20, device="cuda")
    res = time_ab({"graphable": [lambda: x * 2], "syncs": [lambda: (x * 2).sum().item()]}, rounds=2, reps=3)
    assert res["graphable"]["mode"] == res["syncs"]["mode"] == "eager"
    assert "could not be graph-captured" in res["graphable"]["note"] and "host sync" in res["graphable"]["note"]


def test_significance():
    assert significance({"rounds": [10, 11]}, {"rounds": [5, 6]}) == "faster"
    assert significance({"rounds": [10, 11]}, {"rounds": [10.5, 12]}) == "within noise"


# ---------------------------------------------------------------- end to end on a tiny model

ENTRY = """
import torch
from torch import nn
{extra}
class Block(nn.Module):
    def __init__(s, d=256):
        super().__init__(); s.norm = nn.LayerNorm(d); s.fc1 = nn.Linear(d, 4 * d); s.fc2 = nn.Linear(4 * d, d)
        {bn_init}
    def forward(s, x):
        h = s.fc2(torch.nn.functional.gelu(s.fc1(s.norm(x))))
        {bn_use}
        return x + h
def build():
    torch.manual_seed(0)
    return Block().cuda(){cast}, (torch.randn(4, 128, 256, device="cuda"){cast},)
def shapes():
    return [(torch.randn(3, 77, 256, device="cuda"){cast},)]
"""


def make_entry(tmp_path, name, extra="", cast=".to(torch.bfloat16)", bn=False):
    src = ENTRY.format(extra=extra, cast=cast, bn_init="s.bn = nn.BatchNorm1d(256)" if bn else "",
                       bn_use="h = s.bn(h.transpose(1, 2)).transpose(1, 2)" if bn else "")
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(src))
    return f


def capture(tmp_path, entry, impl_src):
    d = tmp_path / "region"
    tool("region.py", "--entry", entry, "--module", "<root>", "--out", d, check=True)
    (d / "impl.py").write_text(textwrap.dedent(impl_src))
    return d


GOOD = """
import torch, torch.nn.functional as F
def forward(mod, x):
    h = F.layer_norm(x, (x.shape[-1],), mod.norm.weight, mod.norm.bias, mod.norm.eps)
    return torch.addmm(x.reshape(-1, x.shape[-1]), F.gelu(F.linear(h, mod.fc1.weight, mod.fc1.bias)).reshape(-1, mod.fc2.in_features),
                       mod.fc2.weight.t(), beta=1).view_as(x) + mod.fc2.bias
"""


def test_inference_good_passes_and_bugs_fail(tmp_path):
    entry = make_entry(tmp_path, "infer")
    d = capture(tmp_path, entry, GOOD)
    tool("bench.py", d, check=True)
    # wrong formula on a few tokens -> fail
    (d / "bug.py").write_text(GOOD.replace("    return torch.addmm", "    out = torch.addmm").rstrip()
                              + "\n    out[:, -3:] *= 1.05\n    return out\n")
    tool("bench.py", d, "--impl", d / "bug.py", "--no-time", check=False)
    # shape-specialised impl passes on the captured shape but swap with shapes() catches it
    (d / "impl.py").write_text(GOOD.replace("def forward(mod, x):", "def forward(mod, x):\n    assert x.shape[1] == 128"))
    tool("bench.py", d, "--no-time", check=True)
    tool("swap.py", "--entry", entry, "--impl", d, check=False)


def test_training_gradients_and_state(tmp_path):
    entry = make_entry(tmp_path, "train", extra='MODE = "train"')
    d = capture(tmp_path, entry, GOOD)
    tool("bench.py", d, "--no-time", check=True)
    # a forward that is right but detaches a weight: its gradient silently becomes zero
    (d / "impl.py").write_text(GOOD.replace("mod.fc1.weight, mod.fc1.bias", "mod.fc1.weight.detach(), mod.fc1.bias"))
    tool("bench.py", d, "--no-time", check=False)


def test_batchnorm_running_stats_must_match(tmp_path):
    entry = make_entry(tmp_path, "bn", extra='MODE = "train"', cast="", bn=True)
    d = capture(tmp_path, entry, """
import torch, torch.nn.functional as F
def forward(mod, x):  # batch statistics right, running statistics never updated
    h = mod.fc2(F.gelu(mod.fc1(mod.norm(x))))
    h = F.batch_norm(h.transpose(1, 2), None, None, mod.bn.weight, mod.bn.bias, True, 0.1, mod.bn.eps).transpose(1, 2)
    return x + h
""")
    p = tool("bench.py", d, "--no-time", check=False)
    assert "module state differs" in p.stdout


FAITHFUL_AMP = """
import torch, torch.nn.functional as F
def forward(mod, x):
    h = F.layer_norm(x, (x.shape[-1],), mod.norm.weight, mod.norm.bias, mod.norm.eps)
    return x + F.linear(F.gelu(F.linear(h, mod.fc1.weight, mod.fc1.bias)), mod.fc2.weight, mod.fc2.bias)
"""


def test_autocast_is_recorded_and_replayed(tmp_path):
    entry = make_entry(tmp_path, "amp", extra='MODE = "train"\nAUTOCAST = "bfloat16"', cast="")
    d = capture(tmp_path, entry, FAITHFUL_AMP)
    meta = json.loads((d / "region.json").read_text())
    assert meta["autocast"] == {"dtype": "bfloat16"}
    p = tool("bench.py", d, "--no-time", check=True)
    assert "replaying under torch.autocast(bfloat16)" in p.stdout
    # Found by this test: folding the fp32 residual into addmm under autocast casts it to bf16
    # (the original adds it in fp32). Fine for a bf16 model, a real precision bug under AMP.
    (d / "fold.py").write_text(GOOD)
    tool("bench.py", d, "--impl", d / "fold.py", "--no-time", check=False)


# ---------------------------------------------------------------- lessons

def test_lessons_checker(tmp_path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"device": "NVIDIA H20-3e", "torch": torch.__version__}))
    ok = tmp_path / "ok.md"
    ok.write_text(f"## Seed\n- seeded [evidence: seed]\n## Mine\n- [H20-3e, bf16] fine [evidence: {res}]\n")
    assert tool("lessons.py", ok).returncode == 0
    for bad in (f"- [A100, bf16] wrong gpu [evidence: {res}]", f"- untagged [evidence: {res}]",
                "- [H20-3e] seed outside seed section [evidence: seed]", "- [H20-3e] missing [evidence: /nope.json]"):
        f = tmp_path / "bad.md"
        f.write_text(f"## Mine\n{bad}\n")
        assert tool("lessons.py", f).returncode == 1, bad


# ---------------------------------------------------------------- torch.compile integration

def test_library_kernels_compile_fullgraph_and_stay_correct():
    sys.path.insert(0, str(ROOT / "library"))
    import rmsnorm  # noqa: PLC0415

    r = rmsnorm.selftest()
    assert r["pass"] and r["compiled"]["pass"], r.get("compiled")


CACHED = GOOD.replace("    h = F.layer_norm(x, (x.shape[-1],), mod.norm.weight, mod.norm.bias, mod.norm.eps)",
                      "    h = F.layer_norm(x, (x.shape[-1],), mod.norm.weight, mod.norm.bias, mod.norm.eps)\n"
                      "    torch._dynamo.graph_break()   # stands in for Python-side state dynamo cannot trace")


def test_bench_compile_reports_graph_breaks(tmp_path):
    entry = make_entry(tmp_path, "cc", extra='MODE = "train"')
    d = capture(tmp_path, entry, GOOD)
    tool("bench.py", d, "--compile", "--no-time", check=True)
    clean = sorted((d / "results").glob("[0-9][0-9][0-9].json"))[-1]
    assert json.loads(clean.read_text())["compiled_impl"]["graph_breaks"] == 0
    (d / "impl.py").write_text(CACHED)
    p = tool("bench.py", d, "--compile", "--no-time", check=True)   # still correct, but breaks reported
    res = json.loads(sorted((d / "results").glob("[0-9][0-9][0-9].json"))[-1].read_text())
    assert res["compiled_impl"]["graph_breaks"] >= 1 and "graph break" in p.stdout
