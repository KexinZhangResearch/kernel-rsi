# LESSONS

Read before every task; rewrite after every task (see SKILL.md). One bullet per lesson:
context → tried → measured → takeaway [evidence: path to a bench.py/swap.py result or hardware file].

## Seed — measured on ONE GPU (NVIDIA H20-3e, sm_90, bf16) while building the skill; evidence not shipped
Starting knowledge, not ground truth. Numbers and tile sizes are H20-specific; the *patterns* (e.g. a
non-contiguous attention bias forcing a slow path, fusing around small-K GEMMs) may transfer but must be
re-measured on your GPU. Correct or delete a seed lesson when your measurements disagree; add new
lessons in the section below, never here.

### Hardware: NVIDIA H20-3e (sm_90, 78 SMs, 60 MB L2) — see hardware/nvidia_h20_3e.json
- Peaks measured: bf16 GEMM 139 TFLOPS, fp8 279, copy 2.25 TB/s, read 2.6, write 3.4 → ridge ~53 FLOP/B. cuBLAS reaches 92–118 TFLOPS on 2048×1152×(1152…4608) GEMMs, so blocks with real GEMMs are compute-bound here; the headroom is in what surrounds the GEMMs. [evidence: seed, hardware/nvidia_h20_3e.json]
- Triton's Python launcher costs ~20 µs per call (torch op ~5 µs, graph replay ~10 µs). A tiny Triton kernel is launch-bound in an eager model; fewer, larger kernels win twice. Report both bench.py device time and host time. [evidence: seed, hardware/nvidia_h20_3e.json]
- ncu has no counter permission in this container (ERR_NVGPUCTRPERM); reason from flops/bytes/launch counts and timed experiments instead. [evidence: seed, hardware/nvidia_h20_3e.json]

### Attention
- SDPA with an additive bias that is NOT contiguous (e.g. bias from `.permute()`) silently falls back to the fp32 math path: 612 µs vs 94 µs (cuDNN) / 124 µs (default) with a contiguous bias, B4 H16 S512 D64. Always `.contiguous()` the bias; if it depends only on shapes + weights (T5 relative position), compute it once and cache it. [evidence: seed]
- head_dim not a power of two (72): torch flash 203 µs (48 TFLOPS), cuDNN 167 µs. A Triton flash kernel computing the head dim as two dots 64 + 16 (8 zero-padded) and reading q/k/v straight from the qkv GEMM output: 136 µs. Best tile BM64 BN32 4 warps 3 stages; bigger BN / 8 warps were slower (two accumulators → register pressure). [evidence: seed]
- cuDNN SDPA backend beat flash on this GPU in both shapes tried; try `sdpa_kernel([CUDNN_ATTENTION, ...])` before writing a kernel. [evidence: seed]

### Fusion across modules
- Best region = a parent block, not a leaf: in a DiT block, residual + gate + next LayerNorm + modulation (spanning the parent and a child module) became one row kernel; 3 such kernels + better attention took the block from 874 to 716 µs (torch.compile 815). [evidence: seed]
- Cheap structural wins before any kernel: concatenate GEMMs that share an input (q|k|v, wi0|wi1), fold the residual add into the GEMM (`torch.addmm(residual, a, W.t())`) — only when the residual is already in the GEMM's dtype: under AMP autocast casts an fp32 residual to bf16 and the check fails (3× the original's error, tests/test_tools.py) —, cache weight-only transforms (inference only). T5 layer: 67 → 8 kernels, 1336 → 592 µs, 1.51× over torch.compile. [evidence: seed]
- GEMMs with a small contraction dim (K ≤ ~128, e.g. pair-representation projections [N², 128]) are output-bandwidth-bound, not compute-bound: merging them got slower (191 vs 165 µs). Instead fuse norm → dot(s) in registers → gates/mask → store in the layout the NEXT op wants, in one Triton kernel. Pairformer TriMul: 2482 → 485 µs (torch.compile 1321) with 3 launches. [evidence: seed]
- An fp32 upcast in the model (`einsum(a.float(), b.float())`) on values that are already bf16 can run as a bf16 tensor-core matmul with fp32 accumulation: same products, same final rounding. TriMul einsum 1096 → 124 µs; errors identical to the original's. [evidence: seed]
- LayerNorm over short rows (128 elements) in eager runs at ~230 GB/s on H20; process a [64 rows, C] tile per program and fuse it into its consumer. [evidence: seed]
- Keep the original's rounding points (e.g. T5 RMSNorm casts to bf16 before `* weight`; DiT residual is a bf16 op) and do statistics in fp32: errors then match or beat the original. [evidence: seed]

### Process
- Time the pieces of a region separately (scratch script with common.time_calls) before deciding what to write: in all three tasks one component dominated the gap (attention ×2, an fp32 einsum ×1), and it was never the one the module name suggested. [evidence: seed]
- Always compare against torch.compile too (`--compile`): it already fuses elementwise chains, so the real gain is what it cannot do (layouts, algorithm, attention, GEMM merging). [evidence: seed]

### Training (forward + backward), T5 encoder layer, bf16
- The backward was 2/3 of the step and the hot spot moved: the relative-position bias alone cost 745 µs fwd+bwd (embedding backward = contended scatter-add over S² indices). Sum dBias along diagonals into a per-distance table [H, 2S-1], then into the 32 buckets: no scatter. [evidence: seed]
- SDPA's efficient-attention backward failed the gradient check (q/k/bias grads ~7× the original's error) although its forward and the math-path backward were fine; bisected by swapping one piece back at a time. A Triton flash fwd+bwd that keeps an fp32 copy of O for delta = rowsum(dO·O) matched the math path's precision at 417 vs 2097 µs — reusable as `library/relpos_attention.py`. [evidence: seed, library/replay/nvidia_h20_3e.json]
- Triton fwd+bwd for RMSNorm/GEGLU (rstd saved per row, deterministic dw partials per program) + merged GEMMs + in-kernel bias: training step 3.80 → 2.02 ms, peak memory −37%; torch.compile of the same step was slower than eager (5.04 ms). Kernels: `library/rmsnorm.py`, `library/geglu.py`. [evidence: seed, library/replay/nvidia_h20_3e.json]
- Caching weight-derived tensors is a training bug twice over: stale after the optimizer step, and detached from the graph (the weight's gradient silently becomes zero). Recompute per call; the tools re-randomize weights between check rounds to catch it. [evidence: seed]

### torch.compile integration (H20-3e, torch 2.6, T5 training step)
- torch.compile of the *original* T5 training step was slower than eager (5.05 vs 3.85 ms), while the same impl compiled ran 1.96 ms: compile fuses elementwise work but not attention-with-bias or the bias-gradient scatter. Report both eager and compiled numbers. [evidence: seed]
- A dict cache on a module (`mod.__dict__.setdefault(...)`) caused a graph break; computing the same value inline under `torch.compiler.is_compiling()` removed it (1 graph, 0 breaks). [evidence: seed]
- Kernels as `torch.library.triton_op` + `register_autograd` compile with fullgraph=True, but in eager the registered path cost +2.8 ms host time per training step versus a plain autograd.Function launching the same kernels (device time equal). Dispatch per call: op when compiling, Function in eager. [evidence: seed, library/replay/nvidia_h20_3e.json]
- A CUDA-graph capture that fails on a host sync (`.item()`) poisons the process (later `randn` raises); the tools now detect host syncs with `torch.cuda.set_sync_debug_mode("error")` before capturing. [evidence: seed]

## Lessons from your tasks — start each with [GPU, dtype]; cite result files
<!-- Example format (delete when you add the first real lesson):
  e.g. "- [H20-3e, bf16, train] context → tried → measured → takeaway [evidence: <abs path>/results/004.json]"
-->
