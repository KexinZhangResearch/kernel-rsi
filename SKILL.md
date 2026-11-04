---
name: kernel-rsi
description: Speed up a PyTorch model of any architecture on the local NVIDIA GPU (Volta through Blackwell, datacenter or consumer) by replacing hot regions (one op or a whole block, fused across ops) with Triton/torch code that computes the same function, verified on the real model; learns across tasks through LESSONS.md. Use for "加速模型 / 优化算子 / 写 Triton kernel / fuse / 适配硬件" and similar requests.
---

# kernel-rsi

Tools live in `<this skill>/tools/`; run them with the Python that has torch (+ triton if the GPU
supports it), absolute paths.

## The only hard rules

1. **Same model logic.** A replacement may change anything inside a region, but the model's outputs
   must stay equivalent: `swap.py` on the real model must pass. Never loosen `--slack` silently —
   if you need a lower-precision path (fp8, approximations), ask the user first.
2. **Measured, not estimated.** Every speedup you report or write down comes from a `bench.py` /
   `swap.py` result file. Correctness first: a fast wrong kernel is worth nothing.
3. Do not edit the user's model code or the captured files (`module.pt`, `io.pt`, `region.json`).

Everything else — where to cut regions, which ops to fuse, algorithm, data layout, precision of
intermediates, Triton vs cuBLAS vs torch, recompute vs store — is your decision. There is no list of
allowed methods; think from the hardware and the computation.

## Workflow

0. Run `tools/hw.py` if `hardware/<this gpu>.json` is missing (each GPU model gets its own file). It
   measures GEMM peaks per dtype, bandwidth, L2, shared memory, launch overhead, architecture
   features, and whether Triton and ncu work. Then read `LESSONS.md`: lessons carry the GPU they were
   measured on — trust them only for the same GPU class, and re-measure before reusing one elsewhere.
1. `tools/hotspots.py --entry model_entry.py` — modules ranked by GPU time, with kernel launches,
   eager HBM traffic vs boundary traffic ("fuse x") and the ceiling on this GPU.
   `--module <q>` lists the ops inside one module. (`model_entry.py` = `build() -> (model, inputs)`,
   optional `run(model, inputs)`; write it for the user's model if it does not exist.)
2. Choose a region. Prefer the **largest region whose gap to its ceiling you can explain**: a parent
   block lets you fuse across children (residual add + next norm + modulation, GEMM epilogues,
   attention prologues). Ops that are already near their ceiling (big GEMMs at ~90% of peak) are
   better kept as library calls with their neighbours fused around them.
3. `tools/region.py --entry ... --module <q> --out <work>/<name>` — freezes the region with real
   weights and activations; read `ORIGINAL.md`, then edit `impl.py`.
4. Iterate: write the hypothesis in `NOTES.md` → change `impl.py` → `tools/bench.py <dir> --compile
   --note "<what this tries>"` → record the result and what it taught you. Compare against the
   ceiling and against torch.compile, not only against eager.
5. `tools/swap.py --entry ... --impl <dir> [--all-instances] --compile` — must pass on the whole model;
   report the end-to-end number. Repeat 2–5 for the next region while the gains are worth it.
6. Update `LESSONS.md` (below). Give the user: regions changed, model-level speedup, result paths.

## How to think (principles, not recipes)

- **Find the binding constraint first.** For a region: flops / peak flops (for the region's dtype)
  vs bytes / bandwidth vs kernel launches × launch cost — all measured in `hardware/<gpu>.json`.
  Whichever is largest is what to attack. The balance differs a lot between GPUs (ridge points range
  from ~10 to several hundred FLOP/B), so a trick tuned for one card can be wrong on another: derive
  decisions from this card's numbers, not from what published kernels did on an H100.
- **Use what this architecture has, and only that.** `features` in the hardware file: no bf16 tensor
  cores before sm_80 (use fp16/fp32 there and keep the model's dtype), tf32 from sm_80, fp8 from sm_89,
  wgmma/TMA from sm_90. Size tiles to `smem_per_block_kb` and the SM count (grid ≥ a few × SMs). If
  `triton_error` is set, Triton cannot run here: optimize with torch/cuBLAS/cuDNN, layouts, cached
  work and CUDA graphs instead of writing kernels.
- **Fusion is about bytes and launches.** Every intermediate tensor written to HBM and read back costs
  2× its size; every op costs a launch. "fuse x" from hotspots.py tells you how much is removable.
- **Question the region's structure**, not just its kernels: reorder ops algebraically (apply a scale
  before or after a GEMM, fold a bias, merge GEMMs that share an input, precompute weight-only
  transforms once), change layouts so the next op reads contiguously, keep statistics instead of
  recomputing, split work so all SMs are busy.
- **Use the measurement to falsify, not to confirm.** If a change did not help, find out why before
  trying the next idea — otherwise you are guessing.

## When stuck

- **Wrong results:** bench.py prints the worst element and which index ranges are bad per dimension
  (e.g. "only the last columns" → masking/tail bug; "one batch" → stride/offset bug; everything →
  formula/dtype). Bisect: return an intermediate from `impl.forward` in a scratch copy and compare it
  with the same intermediate computed by the original code. Check rounding order (where the original
  casts to its low-precision dtype) and fp16 overflow (range ±65504) if the model runs in fp16.
- **Slow:** compute the region's ceiling and each kernel's own ceiling; time pieces separately in a
  scratch script (`tools/common.py:time_calls`); try one variable at a time (tile, warps, grid size,
  layout). If ncu is available (hardware file says so), profile the kernel instead of guessing.
- **Out of ideas:** go back to step 2 — a different region boundary often removes the bottleneck
  you could not beat inside the old one.

## LESSONS.md — how this skill improves itself

`LESSONS.md` is the skill's memory; you are its author. At the end of every task:
- Add what is **reusable and measured**: hardware behaviour, what worked or failed for a computation
  pattern, and how you got unstuck. One bullet each, starting with the GPU it was measured on:
  `[GPU, dtype] context → what was tried → measured result → takeaway [evidence: <results json path>]`.
  Put lessons that hold across GPUs (process, numerics) without a GPU tag only after you saw them hold
  on more than one.
- **Fix, don't append:** if a new measurement contradicts a lesson, rewrite or delete that lesson and
  say why. Merge duplicates. Keep the file under ~150 lines; condense when it grows.
- Never write a lesson you did not measure. `tools/lessons.py` checks citations and size.
