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
- Cheap structural wins before any kernel: concatenate GEMMs that share an input (q|k|v, wi0|wi1), fold the residual add into the GEMM (`torch.addmm(residual, a, W.t())`), cache weight-only transforms. T5 layer: 67 → 8 kernels, 1336 → 592 µs, 1.51× over torch.compile. [evidence: seed]
- GEMMs with a small contraction dim (K ≤ ~128, e.g. pair-representation projections [N², 128]) are output-bandwidth-bound, not compute-bound: merging them got slower (191 vs 165 µs). Instead fuse norm → dot(s) in registers → gates/mask → store in the layout the NEXT op wants, in one Triton kernel. Pairformer TriMul: 2482 → 485 µs (torch.compile 1321) with 3 launches. [evidence: seed]
- An fp32 upcast in the model (`einsum(a.float(), b.float())`) on values that are already bf16 can run as a bf16 tensor-core matmul with fp32 accumulation: same products, same final rounding. TriMul einsum 1096 → 124 µs; errors identical to the original's. [evidence: seed]
- LayerNorm over short rows (128 elements) in eager runs at ~230 GB/s on H20; process a [64 rows, C] tile per program and fuse it into its consumer. [evidence: seed]
- Keep the original's rounding points (e.g. T5 RMSNorm casts to bf16 before `* weight`; DiT residual is a bf16 op) and do statistics in fp32: errors then match or beat the original. [evidence: seed]

### Process
- Time the pieces of a region separately (scratch script with common.time_calls) before deciding what to write: in all three tasks one component dominated the gap (attention ×2, an fp32 einsum ×1), and it was never the one the module name suggested. [evidence: seed]
- Always compare against torch.compile too (`--compile`): it already fuses elementwise chains, so the real gain is what it cannot do (layouts, algorithm, attention, GEMM merging). [evidence: seed]

## Lessons from your tasks — start each with [GPU, dtype]; cite result files
