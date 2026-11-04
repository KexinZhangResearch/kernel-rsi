---
name: kernel-rsi
description: Speed up inference or training (forward + backward) of a PyTorch model of any architecture on the local NVIDIA GPU (Volta through Blackwell, datacenter or consumer) by replacing hot regions (one op or a whole block, fused across ops) with Triton/torch code that computes the same function and the same gradients, verified on the real model; learns across tasks through LESSONS.md and a library of verified kernels. Use for "加速模型 / 优化算子 / 写 Triton kernel / fuse / 适配硬件 / 训练加速" and similar requests.
---

# kernel-rsi

Tools live in `<this skill>/tools/`; run them with the Python that has torch (+ triton if the GPU
supports it), absolute paths.

## The only hard rules

1. **Same model logic.** Anything may change inside a region, but the model's outputs — and, for
   training, the loss and every parameter's gradient — must stay equivalent: `swap.py` on the real
   model must pass. Never loosen `--slack`; a lower-precision path (fp8, approximations) needs the
   user's explicit consent.
2. **Measured, not estimated.** Every number you report or write down comes from a `bench.py` /
   `swap.py` result file. A speedup marked "within noise" is not a speedup.
3. Do not edit the user's model code or the captured files (`module.pt`, `io.pt`, `region.json`).

Everything else — region boundaries, what to fuse, algorithm, layout, precision of intermediates,
Triton vs cuBLAS/cuDNN vs torch, recompute vs store — is your decision. There is no list of allowed
methods; reason from this GPU's numbers and the computation.

## Setup: model_entry.py

```python
def build():                 # -> (model on cuda, inputs: tuple of args or dict of kwargs)
def run(model, inputs):      # optional; default model(*inputs) / model(**inputs). No autocast in here.
def shapes():                # strongly recommended: other inputs the model really sees (batch sizes,
                             # odd sequence lengths) — checked and timed by swap.py
MODE = "train"               # if the user trains: every tool then works on forward + backward
AUTOCAST = "bfloat16"        # if the model runs under AMP (fp32 master weights + autocast)
def loss(out, inputs):       # optional training loss; default: scalar output, else mean of squares
```

Use the user's real weights (a checkpoint) whenever they exist: with random init, attention is near
uniform and gates are small, which hid a real masking bug in testing.

## Workflow

0. `tools/hw.py` if `hardware/<this gpu>.json` is missing; `tools/replay.py` on a new machine or after
   an upgrade. Read `LESSONS.md` — trust a lesson only for the GPU/dtype in its tag.
1. `tools/hotspots.py --entry model_entry.py [--module <q>]` — modules by GPU time (forward and
   backward split in training mode), launches, eager vs boundary traffic ("fuse x"), ceiling.
2. Choose the **largest region whose gap to its ceiling you can explain**. A parent block lets you
   fuse across children. Keep near-peak library ops (big GEMMs) and fuse around them.
3. `tools/region.py --entry ... --module <q> --out <work>/<name> [--call N]` — freezes the region with
   real weights, activations, upstream gradients and autocast state. Read `ORIGINAL.md`; check
   `library/` for verified kernels you can import or adapt (or ignore); edit `impl.py`.
4. Iterate: hypothesis in `NOTES.md` → change `impl.py` → `tools/bench.py <dir> --compile --note "..."`
   → record what the result taught you.
5. `tools/swap.py --entry ... --impl <dir> [--all-instances] --compile` — whole model, every `shapes()`
   input. Training: finish with `--train-steps 200` ("INCONCLUSIVE" = the run is chaotic at that lr:
   lower `--lr`). Fix every `WARNING` it prints about unchecked coverage.
6. `tools/report.py <work>` — REPORT.md and `krsi_patch.py` (`apply(model)` for the user's code).
   Update `LESSONS.md`; add reusable kernels to `library/` with a `selftest()` and run `tools/replay.py`.

## How to think

- **Find the binding constraint first**: flops / peak (for the dtype the math runs in) vs bytes /
  bandwidth vs launches × launch cost — all measured in `hardware/<gpu>.json`. Ridge points differ
  ~10× between GPUs; derive decisions from this card, not from what kernels did on an H100.
- **The ceiling is a lower bound**, assuming one pass over memory and peak square-GEMM throughput.
  Skinny GEMMs (small M) reach far less (`tflops.skinny_*`); a region at 70% of its ceiling may be done.
- **Use what this architecture has**: `features` (bf16 from sm_80, fp8 from sm_89, wgmma/TMA from sm_90),
  `smem_per_block_kb`, SM count. If `triton_error` is set, work with torch/cuBLAS/cuDNN, layouts, cached
  work and CUDA graphs instead of kernels.
- **Fusion is about bytes and launches**; also question the structure: algebraic reordering, merged
  GEMMs sharing an input, layouts the next op reads contiguously, statistics kept instead of recomputed.
- **Precision follows the original**: keep its rounding points; under AMP, autocast also casts inputs of
  fused GEMM epilogues (an fp32 residual folded into `addmm` becomes bf16 — that failed the check).
- **Training**: the backward is usually 2/3 of the step. An autograd.Function chooses what to save vs
  recompute (watch peak MB), fuses backward ops, and writes gradients in the layout the next backward
  wants. Library kernels can differ in backward precision even when their forward matches — the
  gradient check decides. Never cache weight-derived tensors across steps or detach them.
- **torch.compile**: users often compile, so `bench.py --compile` / `swap.py --compile` also run your
  impl inside a compiled model, check it, count graph breaks and time it against the compiled original
  (the bar that matters there). Python-side state (dict caches on modules, `.item()`, data-dependent
  Python branches) breaks the graph; guard eager-only shortcuts with `torch.compiler.is_compiling()`.
  For Triton kernels follow `library/` (`_compat.py`): a `torch.library.triton_op` + `register_autograd`
  under compile, a plain autograd.Function in eager — the registered op costs extra host time in eager.
- **Falsify, don't confirm**: if a change did not help, find out why before the next idea.

## When stuck

- **Wrong results**: bench.py prints the worst element, which index ranges are bad per dimension, and
  rows whose error exceeds the original's on the same row (one head / a few tokens → indexing or tail
  mask). Bisect: swap one piece back to the original op per scratch variant (`bench.py --impl v.py
  --no-time`); test a custom backward with `torch.autograd.gradcheck` in float64 on tiny shapes.
- **Slow**: time pieces separately; `common.sweep()` for tile/warp/stage searches; ncu if the hardware
  file says it works. If bench.py says all variants were timed eagerly, something in a variant cannot
  be graph-captured (host sync, `.item()`): remove it or accept eager-only numbers.
- **Out of ideas**: change the region boundary (step 2).

## LESSONS.md — how this skill improves itself

You are its author. After each task add what is reusable and measured, one bullet each:
`[GPU, dtype(, train/AMP)] context → tried → measured result → takeaway [evidence: <result json>]`.
Rewrite or delete lessons that new measurements contradict, merge duplicates, stay under ~150 lines.
`tools/lessons.py` checks that evidence exists and comes from the tagged GPU, and warns when it was
measured with other torch/triton versions.
