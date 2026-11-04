"""GEGLU forward + backward on a merged [a | b] GEMM output, Triton. Use geglu(u) (differentiable, torch.compile-friendly).

u: [rows, 2F] (the output of x @ cat([W_gate, W_up]).T); returns gelu_tanh(u[:, :F]) * u[:, F:], with
the same bf16 rounding points as eager. Backward writes d[a | b] in one pass, so the merged GEMM's
backward reads one gradient tensor.
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


@triton.jit
def _gelu_parts(a):
    z = 0.7978845608028654 * (a + 0.044715 * a * a * a)
    t = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)                # tanh(z)
    return t, 0.5 * a * (1.0 + t)


@triton.jit
def _geglu_fwd(u_ptr, out_ptr, n, Fd, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    row, col = i // Fd, i % Fd
    a = tl.load(u_ptr + row * 2 * Fd + col, mask=m, other=0.0).to(tl.float32)
    b = tl.load(u_ptr + row * 2 * Fd + Fd + col, mask=m, other=0.0)
    _, gel = _gelu_parts(a)
    tl.store(out_ptr + i, gel.to(b.dtype) * b, mask=m)


@triton.jit
def _geglu_bwd(g_ptr, u_ptr, du_ptr, n, Fd, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    row, col = i // Fd, i % Fd
    a = tl.load(u_ptr + row * 2 * Fd + col, mask=m, other=0.0).to(tl.float32)
    b = tl.load(u_ptr + row * 2 * Fd + Fd + col, mask=m, other=0.0)
    g = tl.load(g_ptr + i, mask=m, other=0.0)
    t, gel = _gelu_parts(a)
    db = g * gel.to(g.dtype)                               # bf16, like autograd's mul backward
    dgel = (g * b).to(tl.float32)
    dz = 0.7978845608028654 * (1.0 + 3 * 0.044715 * a * a)
    da = dgel * (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * dz)
    tl.store(du_ptr + row * 2 * Fd + col, da.to(g.dtype), mask=m)
    tl.store(du_ptr + row * 2 * Fd + Fd + col, db, mask=m)


def _fwd(u, launch=wrap):
    rows, F2 = u.shape
    Fd = F2 // 2
    out = torch.empty(rows, Fd, device=u.device, dtype=u.dtype)
    n = out.numel()
    launch(_geglu_fwd)[(triton.cdiv(n, 2048),)](u, out, n, Fd, BLOCK=2048, num_warps=8)
    return out


def _bwd(g, u, launch=wrap):
    du = torch.empty_like(u)
    n = g.numel()
    launch(_geglu_bwd)[(triton.cdiv(n, 2048),)](g, u, du, n, u.shape[1] // 2, BLOCK=2048, num_warps=8)
    return du


class GEGLU(torch.autograd.Function):
    """Fallback for torch without triton_op."""

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return _fwd(u, launch=_raw)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return _bwd(g.contiguous(), u, launch=_raw)


if HAS_TRITON_OP:
    @triton_op("krsi::geglu_fwd", mutates_args=())
    def _fwd_op(u: torch.Tensor) -> torch.Tensor:
        return _fwd(u)

    @triton_op("krsi::geglu_bwd", mutates_args=())
    def _bwd_op(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return _bwd(g, u)

    def _setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def _backward(ctx, g):
        (u,) = ctx.saved_tensors
        return _bwd_op(g.contiguous(), u)

    _fwd_op.register_autograd(_backward, setup_context=_setup)


def geglu(u):
    """gelu_tanh(u[:, :F]) * u[:, F:] for a contiguous [rows, 2F] tensor."""
    if HAS_TRITON_OP and torch.compiler.is_compiling():
        return _fwd_op(u)
    return GEGLU.apply(u)


def reference(u):
    a, b = u.chunk(2, dim=-1)
    return torch.nn.functional.gelu(a, approximate="tanh") * b


def selftest():
    from common import selftest_autograd  # noqa: PLC0415

    u = torch.randn(2048, 2 * 2816, device="cuda", dtype=torch.bfloat16)
    return selftest_autograd(geglu, reference, [u])


if __name__ == "__main__":
    print(selftest())
