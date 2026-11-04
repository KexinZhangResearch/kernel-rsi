# kernel-rsi

**A Claude Code skill that makes PyTorch models faster on *your* GPU — inference and training (forward + backward) — and gets better at it each time it is used.**

It finds hot regions of a model, which can be a single op or a whole block. It replaces them with
fused Triton/torch code that computes the same function, checks equivalence on the real model, and
records what it learned in `LESSONS.md` for the next task. It works with any model architecture on
NVIDIA GPUs from Volta to Blackwell, datacenter or consumer. It measures the card it runs on and asks the
model to reason from those numbers instead of following a fixed list of tricks.

> 中文简介：一个 Claude Code skill。它在本机 GPU 上加速任意 PyTorch 模型：找到热点区域（单个算子或整个 block，可跨算子 fuse），
> 用 Triton/torch 重写，并在真实模型上验证"逻辑等价"。每次任务的实测经验写入 `LESSONS.md`，下次任务先读，这就是它的自我改进（RSI）方式。
> 不固化优化方法，只给原则、工具和硬约束，保留大模型的创造性。支持在 Claude Code 中接入 DeepSeek 等模型使用。

## How it works

```
 hw.py ──► hardware/<gpu>.json         measured peaks, bandwidth, L2, launch cost, ncu availability
   │
 hotspots.py ──► where time goes        per module: GPU time, launches, eager vs. boundary traffic, ceiling
   │
 region.py ──► work/<region>/           real weights + activations, ORIGINAL.md, impl.py to edit
   │
 bench.py ◄──► impl.py  (iterate)       correctness vs fp32 + L2-cold CUDA-graph timing, every run saved
   │
 swap.py ──► whole-model check          equivalence and end-to-end time on the real model
   │
 LESSONS.md ◄── written by the agent    measured, cited lessons; read first next time
```

There are only three hard rules: outputs stay equivalent, every number is measured, and the user's
code is left alone. The rest is up to the model: where to cut regions, what to fuse, which algorithm,
layout or library to use, and whether to recompute or store.

## Install

Clone the repository into a skills directory. The folder name must stay `kernel-rsi`.

```bash
# for all your projects
git clone https://github.com/KexinZhangResearch/kernel-rsi.git ~/.claude/skills/kernel-rsi
# or only for one project
git clone https://github.com/KexinZhangResearch/kernel-rsi.git <project>/.claude/skills/kernel-rsi
```

Requirements: an NVIDIA GPU and Python with a CUDA build of `torch` (≥ 2.1 recommended). `triton` is
needed for custom kernels. Recent Triton releases support compute capability ≥ 8.0 (Ampere and newer);
on older cards (V100, T4, RTX 20xx) the skill still works, but it optimizes with torch/cuBLAS/cuDNN, layouts
and CUDA graphs instead of writing kernels.

Measure each GPU model once. Every card gets its own `hardware/<gpu>.json`:

```bash
python ~/.claude/skills/kernel-rsi/tools/hw.py
```

- **"Python.h is missing"**: Triton cannot build its launcher. Install `python3-dev` (or your distro's
  equivalent), or point `CPATH` at a directory that contains `Python.h`.
- **"ncu has no counter permission"**: common in containers; the skill works without it. To enable
  counters, set the host driver option `NVreg_RestrictProfilingToAdminUsers=0`, or start the container
  with `--cap-add SYS_ADMIN`.

### Using it with DeepSeek (or another Anthropic-compatible endpoint)

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN=<your DeepSeek key>      # leave ANTHROPIC_API_KEY unset
claude --model opus                                  # DeepSeek maps this to its strongest model
```

## Use

In Claude Code:

> Use the kernel-rsi skill to speed up my model. The entry point is `model_entry.py`.

`model_entry.py` tells the tools how to build and call your model (the agent writes one if it is missing):

```python
import torch
from my_project import MyModel

def build():                       # -> (model on cuda, inputs as a tuple of args or a dict of kwargs)
    model = MyModel().cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 1024, 1152, device="cuda", dtype=torch.bfloat16)
    return model, (x,)

def run(model, inputs):            # optional; default is model(*inputs) / model(**inputs)
    return model(*inputs)

def shapes():                      # recommended: other inputs the model sees (batch sizes, odd lengths)
    return [(torch.randn(1, 300, 1152, device="cuda", dtype=torch.bfloat16),)]

MODE = "train"                     # optional: optimize the training step (forward + backward)
AUTOCAST = "bfloat16"              # optional: the model runs under AMP (do not call autocast in run())
def loss(out, inputs):             # optional; default: scalar output, else mean of squares
    return out.float().pow(2).mean()
```

With `MODE = "train"`, every tool works on a full training step. The hotspot table splits each
module's time into forward and backward. `region.py` captures the real upstream gradient. `bench.py`
and `swap.py` check the gradient of every input and parameter and time forward + backward. They also
report peak memory, and they re-randomize the weights between checks so stale weight caches fail.
Dropout is switched off while checking, because random masks cannot be compared.

The tools can also be run by hand:

| Tool | Purpose |
|---|---|
| `tools/hw.py` | Measure this GPU (writes `hardware/<gpu>.json`) |
| `tools/hotspots.py --entry E [--module Q]` | Rank modules by GPU time and fusion headroom; list the ops inside one module |
| `tools/region.py --entry E --module Q --out DIR` | Freeze a region with its real weights and inputs; create `impl.py` |
| `tools/bench.py DIR [--compile]` | Check and time `impl.py` against the original (and `torch.compile`) |
| `tools/swap.py --entry E --impl DIR [--all-instances] [--compile]` | Patch the model; check the whole model; time it end to end |
| `tools/report.py WORK_DIR` | Summarize every attempt (REPORT.md) and export `krsi_patch.py` with `apply(model)` for your code |
| `tools/replay.py [--regions DIR]` | Re-verify all library kernels (and archived regions) on this machine, no LLM needed |
| `tools/lessons.py` | Check that lessons cite existing results from the GPU they name; warn about stale torch/triton versions |

## What "equivalent" and "faster" mean

- **Equivalent.** Outputs (and in training mode the loss and all gradients) are compared against a float32 run of the original. No element may be
  further off than 2× the original low-precision model's worst error, and the relative L2 error must be
  within 2× the original's. This is checked on the captured activations and on fresh random inputs,
  first for the region (`bench.py`) and then for the whole model (`swap.py`). On a failure, `bench.py`
  reports which index ranges of which dimension are wrong.
- **Robust.** A per-row check against the original's error on the same row catches bugs confined to
  one head or a few tokens, which a tensor-wide budget dilutes. Module buffers such as BatchNorm running
  stats must match. `shapes()` in `model_entry.py` adds other input shapes, and `swap.py --train-steps N`
  compares N-step loss curves against the original's own sensitivity to 1-ulp weight noise, so chaotic
  runs are reported as inconclusive rather than failed. The reference runs in true fp32, with TF32 off
  for both matmul and cuDNN. Run-to-run nondeterminism is flagged.
- **Faster.** GPU time per call, measured with CUDA graphs. Inputs and weights rotate through enough
  copies that every call reads cold memory, and the original, `torch.compile` and the replacement are
  measured the same way, in 5 interleaved rounds. A speedup whose round ranges overlap is reported as
  "within noise". Host-inclusive time is reported as well. Every run is saved as
  `results/NNN.json` together with a frozen copy of the impl.

## Self-improvement

`LESSONS.md` is the skill's memory, and the agent maintains it. After each task it adds what it
measured, using the form *context → tried → result → takeaway*, and every lesson cites a result file.
It rewrites lessons that newer measurements contradict, merges duplicates, and keeps the file under
about 150 lines. `tools/lessons.py` rejects lessons whose evidence is missing.

The shipped `LESSONS.md` starts with a **Seed** section. Those lessons were measured on an H20-3e while
the skill was being built; their raw evidence is not included, so the agent treats them as hints to
re-measure. Lessons from your own tasks go below that section and must cite result files. If you use the
skill on other hardware or models, consider contributing your lessons and `hardware/*.json` back.

## Results so far

All on an NVIDIA H20-3e (sm_90), bf16, measured with the tools above:

| Region | eager | torch.compile | kernel-rsi | whole model vs. eager |
|---|---|---|---|---|
| DiT block (adaLN, attention with head_dim 72, MLP) | 874 µs | 815 µs | 716 µs | 1.21× |
| T5 encoder layer (RMSNorm, relative-position-bias attention, GEGLU) | 1336 µs | 893 µs | 592 µs | 2.19× |
| Pairformer triangle multiplication (N=384, C=128) | 2482 µs | 1321 µs | 485 µs | 4.88× |
| T5 encoder layer, **training step** (fwd + bwd) | 3796 µs | 5037 µs | 2023 µs eager, 1960 µs compiled (0 graph breaks) | 1.85× eager, 2.58× vs compiled original, −37% peak memory |

These replacements were produced by Claude Opus following this skill; the code and result files are
not part of this release. Runs driven by other models have not been benchmarked yet.

## Tests

`python -m pytest tests -q` takes about a minute on a GPU. Each check in the tools is tested both ways:
correct code must pass, and a planted bug must fail. The planted bugs are a localized error, a detached
weight, BatchNorm stats that never update, a shape-specialized impl, mixed-mode timing, and an AMP
residual-fold precision loss.

## Repository layout

```
SKILL.md         instructions the agent loads (workflow, principles, rules)
LESSONS.md       the skill's memory (checked by tools/lessons.py)
library/         verified reusable kernels, each with selftest(); replay/ = per-GPU verification results
tools/           hw, hotspots, region, bench, swap, report, replay, lessons (+ common helpers)
tests/           pytest suite for the tools
hardware/        measured GPU profiles
```

## Limitations

- NVIDIA only (CUDA graphs, cuBLAS peaks, ncu). Tested on an H20-3e; on other generations, the code paths
  for missing bf16, tf32 or fp8 and for missing Triton were checked by simulation, not on real cards.
- Multi-GPU: the tools measure and run on the current device only. Use `CUDA_VISIBLE_DEVICES` to choose it.
- Caching weight transforms inside an impl is valid only for inference (frozen weights).
- With `--compile`, `bench.py` and `swap.py` also check and time the impl inside a compiled model and
  report graph breaks. Library kernels are `torch.library.triton_op`s under compile (fullgraph-safe) and
  plain autograd Functions in eager, where the registered op costs extra host time.
- Training checks run on a single device. AMP autocast is supported. Gradient scalers, gradient
  accumulation and distributed training (FSDP/DDP) sit outside the region, so they are not checked.
  Apply `krsi_patch.apply(model)` before `torch.compile` or FSDP wrapping.
- The self-improvement works at the level of knowledge: lessons are checked for evidence, but there is
  no automatic A/B test that a lesson helps. To check, rerun an old region from scratch and compare it
  with its earlier result file.

## License

MIT — see [LICENSE](LICENSE).
