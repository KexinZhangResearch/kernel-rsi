"""RMSNorm forward + backward (T5 / LLaMA style), Triton. Use rmsnorm(x2d, weight, eps) (differentiable, torch.compile-friendly).

y = weight * (x * rsqrt(mean(x^2) + eps)).to(x.dtype)   -- rounding as in HF T5LayerNorm / LlamaRMSNorm
Forward saves an fp32 rstd per row; backward computes dx in one pass and dw as deterministic
per-program partial sums (no atomics). Rows must fit one block (N <= ~16K).
Verified: NVIDIA H20-3e, torch 2.6, triton 3.2 (see replay output).
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import HAS_TRITON_OP, raw as _raw, triton_op, wrap  # noqa: E402

_NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def _rms_fwd(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    tl.store(rstd_ptr + row, r)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    tl.store(y_ptr + row * N + cols, w * (x * r).to(w.dtype), mask=m)


@triton.jit
def _rms_bwd(g_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dwp_ptr, rows, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    dw = tl.zeros([BLOCK_N], tl.float32)
    for row in range(pid, rows, tl.num_programs(0)):
        off = row.to(tl.int64) * N + cols
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + off, mask=m, other=0.0)
        r = tl.load(rstd_ptr + row)
        n = (x * r).to(w.dtype)
        dw += g.to(tl.float32) * n.to(tl.float32)
        dn = (g * w).to(tl.float32)                       # bf16 product, as autograd computes it
        dx = r * dn - x * (r * r * r) * (tl.sum(dn * x, 0) / N)
        tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dwp_ptr + pid * N + cols, dw, mask=m)


def _fwd(x2d, w, eps, launch=wrap):
    rows, N = x2d.shape
    y = torch.empty_like(x2d)
    rstd = torch.empty(rows, device=x2d.device, dtype=torch.float32)
    launch(_rms_fwd)[(rows,)](x2d, w, y, rstd, N, eps, BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return y, rstd


def _bwd(g, x2d, w, rstd, launch=wrap):
    rows, N = x2d.shape
    progs = min(rows, 4 * _NUM_SMS)
    dx = torch.empty_like(x2d)
    dwp = torch.empty(progs, N, device=x2d.device, dtype=torch.float32)
    launch(_rms_bwd)[(progs,)](g, x2d, w, rstd, dx, dwp, rows, N, BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return dx, dwp.sum(0).to(w.dtype)


class RMSNorm(torch.autograd.Function):
    """Fallback for torch without triton_op (eager-only friendly)."""

    @staticmethod
    def forward(ctx, x2d, w, eps):
        y, rstd = _fwd(x2d, w, eps, launch=_raw)
        ctx.save_for_backward(x2d, w, rstd)
        return y

    @staticmethod
    def backward(ctx, g):
        x2d, w, rstd = ctx.saved_tensors
        return (*_bwd(g.contiguous(), x2d, w, rstd, launch=_raw), None)


if HAS_TRITON_OP:
    @triton_op("krsi::rmsnorm_fwd", mutates_args=())
    def _fwd_op(x2d: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return _fwd(x2d, w, eps)

    @triton_op("krsi::rmsnorm_bwd", mutates_args=())
    def _bwd_op(g: torch.Tensor, x2d: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _bwd(g, x2d, w, rstd)

    def _setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1], output[1])

    def _backward(ctx, gy, _grstd):
        x2d, w, rstd = ctx.saved_tensors
        dx, dw = _bwd_op(gy.contiguous(), x2d, w, rstd)
        return dx, dw, None

    _fwd_op.register_autograd(_backward, setup_context=_setup)


def rmsnorm(x2d, w, eps):
    """y = w * (x * rsqrt(mean(x^2) + eps)).to(x.dtype), rows of a 2D contiguous tensor."""
    if HAS_TRITON_OP and torch.compiler.is_compiling():
        return _fwd_op(x2d, w, eps)[0]
    return RMSNorm.apply(x2d, w, eps)


def reference(x2d, w, eps):
    var = x2d.float().pow(2).mean(-1, keepdim=True)
    return w * (x2d.float() * torch.rsqrt(var + eps)).to(x2d.dtype)


def selftest():
    from common import selftest_autograd  # noqa: PLC0415

    x = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(1024, device="cuda")).to(torch.bfloat16)
    return selftest_autograd(lambda x, w: rmsnorm(x, w, 1e-6), lambda x, w: reference(x, w, 1e-6), [x, w])


if __name__ == "__main__":
    print(selftest())
