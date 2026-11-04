"""Flash attention forward + backward with an additive bias that depends only on (j - i), Triton.

relpos_attention(qkv, pd, B, S, H, D, scale) -> [B*S, H*D]   (differentiable, torch.compile-friendly)
  qkv: [B*S, 3*H*D] (the fused q|k|v GEMM output, read in place: no permute copies)
  pd:  [H, 2S-1] fp32 per-distance bias, bias[h, i, j] = pd[h, j - i + S - 1] (T5 relative position,
       ALiBi, any Toeplitz bias); its gradient is returned via diagonal sums of dS (no scatter).
Backward keeps an fp32 copy of O for delta = rowsum(dO * O): with bf16 O the q/k gradients failed the
gradient check in PyTorch's efficient-attention backward (see LESSONS.md). No causal mask, D in {32,64,128}.
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
import torch.nn.functional as F  # noqa: E402

LOG2E = 1.4426950408889634


@triton.jit
def _fwd(qkv, pd, out, out32, lse, S, H, scale, stride_tok, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    rm = pid_m * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    d = tl.arange(0, D)
    tok0 = b * S
    mrow = rm < S
    q = tl.load(qkv + (tok0 + rm)[:, None].to(tl.int64) * stride_tok + h * D + d[None, :], mask=mrow[:, None], other=0.0)
    kb, vb = qkv + H * D + h * D, qkv + 2 * H * D + h * D
    pdh = pd + h * (2 * S - 1)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start in range(0, S, BN):
        cn = start + rn
        mcol = cn < S
        k = tl.load(kb + (tok0 + cn)[None, :].to(tl.int64) * stride_tok + d[:, None], mask=mcol[None, :], other=0.0)
        bias = tl.load(pdh + (cn[None, :] - rm[:, None] + S - 1), mask=mrow[:, None] & mcol[None, :], other=0.0)
        s = (tl.dot(q, k) * scale + bias) * 1.4426950408889634
        s = tl.where(mcol[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vb + (tok0 + cn)[:, None].to(tl.int64) * stride_tok + d[None, :], mask=mcol[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    off = (tok0 + rm)[:, None].to(tl.int64) * (H * D) + h * D + d[None, :]
    tl.store(out + off, o.to(out.dtype.element_ty), mask=mrow[:, None])
    tl.store(out32 + off, o, mask=mrow[:, None])
    tl.store(lse + bh * S + rm, m_i + tl.log2(l_i), mask=mrow)


@triton.jit
def _bwd_dkdv(qkv, pd, dout, lse, delta, dqkv, dbias, S, H, scale, stride_tok,
              D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    cn = pid_n * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    tok0 = b * S
    mcol = cn < S
    koff = (tok0 + cn)[:, None].to(tl.int64) * stride_tok + h * D + d[None, :]
    k = tl.load(qkv + H * D + koff, mask=mcol[:, None], other=0.0)
    v = tl.load(qkv + 2 * H * D + koff, mask=mcol[:, None], other=0.0)
    pdh = pd + h * (2 * S - 1)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for start in range(0, S, BM):
        rm = start + tl.arange(0, BM)
        mrow = rm < S
        valid = mrow[:, None] & mcol[None, :]
        q = tl.load(qkv + (tok0 + rm)[:, None].to(tl.int64) * stride_tok + h * D + d[None, :], mask=mrow[:, None], other=0.0)
        do = tl.load(dout + (tok0 + rm)[:, None].to(tl.int64) * (H * D) + h * D + d[None, :], mask=mrow[:, None], other=0.0)
        l = tl.load(lse + bh * S + rm, mask=mrow, other=0.0)
        dl = tl.load(delta + bh * S + rm, mask=mrow, other=0.0)
        bias = tl.load(pdh + (cn[None, :] - rm[:, None] + S - 1), mask=valid, other=0.0)
        s = (tl.dot(q, tl.trans(k)) * scale + bias) * 1.4426950408889634
        p = tl.where(valid, tl.exp2(s - l[:, None]), 0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dl[:, None])
        tl.store(dbias + (bh.to(tl.int64) * S + rm[:, None]) * S + cn[None, :], ds, mask=valid)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(dqkv + H * D + koff, (dk * scale).to(dqkv.dtype.element_ty), mask=mcol[:, None])
    tl.store(dqkv + 2 * H * D + koff, dv.to(dqkv.dtype.element_ty), mask=mcol[:, None])


@triton.jit
def _bwd_dq(qkv, pd, dout, lse, delta, dqkv, S, H, scale, stride_tok, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    rm = pid_m * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    d = tl.arange(0, D)
    tok0 = b * S
    mrow = rm < S
    qoff = (tok0 + rm)[:, None].to(tl.int64) * stride_tok + h * D + d[None, :]
    q = tl.load(qkv + qoff, mask=mrow[:, None], other=0.0)
    do = tl.load(dout + (tok0 + rm)[:, None].to(tl.int64) * (H * D) + h * D + d[None, :], mask=mrow[:, None], other=0.0)
    l = tl.load(lse + bh * S + rm, mask=mrow, other=0.0)
    dl = tl.load(delta + bh * S + rm, mask=mrow, other=0.0)
    pdh = pd + h * (2 * S - 1)
    dq = tl.zeros([BM, D], tl.float32)
    for start in range(0, S, BN):
        cn = start + rn
        mcol = cn < S
        valid = mrow[:, None] & mcol[None, :]
        koff = (tok0 + cn)[:, None].to(tl.int64) * stride_tok + h * D + d[None, :]
        k = tl.load(qkv + H * D + koff, mask=mcol[:, None], other=0.0)
        v = tl.load(qkv + 2 * H * D + koff, mask=mcol[:, None], other=0.0)
        bias = tl.load(pdh + (cn[None, :] - rm[:, None] + S - 1), mask=valid, other=0.0)
        s = (tl.dot(q, tl.trans(k)) * scale + bias) * 1.4426950408889634
        p = tl.where(valid, tl.exp2(s - l[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dl[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(dqkv + qoff, (dq * scale).to(dqkv.dtype.element_ty), mask=mrow[:, None])


CFG = dict(BM=64, BN=64, num_warps=4, num_stages=2)


def _forward(qkv, pd, B, S, H, D, scale, launch=wrap):
    out = torch.empty(B * S, H * D, device=qkv.device, dtype=qkv.dtype)
    out32 = torch.empty(B * S, H * D, device=qkv.device, dtype=torch.float32)
    lse = torch.empty(B * H, S, device=qkv.device, dtype=torch.float32)
    c = CFG
    launch(_fwd)[(triton.cdiv(S, c["BM"]), B * H)](qkv, pd, out, out32, lse, S, H, scale, qkv.stride(0), D=D,
                                                 BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"],
                                                 num_stages=c["num_stages"])
    return out, out32, lse


def _backward_impl(dout, qkv, pd, out32, lse, B, S, H, D, scale, launch=wrap):
    delta = (dout.float() * out32).view(B, S, H, D).sum(-1).permute(0, 2, 1).contiguous()   # [B, H, S]
    dqkv = torch.empty_like(qkv)
    dbias = torch.empty(B, H, S, S, device=qkv.device, dtype=torch.float32)
    c = CFG
    launch(_bwd_dkdv)[(triton.cdiv(S, c["BN"]), B * H)](qkv, pd, dout, lse, delta, dqkv, dbias, S, H, scale,
                                                      qkv.stride(0), D=D, BM=c["BM"], BN=c["BN"],
                                                      num_warps=c["num_warps"], num_stages=c["num_stages"])
    launch(_bwd_dq)[(triton.cdiv(S, c["BM"]), B * H)](qkv, pd, dout, lse, delta, dqkv, S, H, scale, qkv.stride(0),
                                                    D=D, BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"],
                                                    num_stages=c["num_stages"])
    dsum = dbias.sum(0)                                                    # [H, S, S]
    y = F.pad(dsum.flip(-1), (0, S)).reshape(H, 2 * S * S)[:, : S * (2 * S - 1)]
    dpd = y.reshape(H, S, 2 * S - 1).sum(1).flip(-1)                       # [H, 2S-1]
    return dqkv, dpd


class RelPosAttention(torch.autograd.Function):
    """Fallback for torch without triton_op."""

    @staticmethod
    def forward(ctx, qkv, pd, B, S, H, D, scale):
        out, out32, lse = _forward(qkv, pd, B, S, H, D, scale, launch=_raw)
        ctx.save_for_backward(qkv, pd, out32, lse)
        ctx.dims = (B, S, H, D, scale)
        return out

    @staticmethod
    def backward(ctx, dout):
        return (*_backward_impl(dout.contiguous(), *ctx.saved_tensors, *ctx.dims, launch=_raw), None, None, None, None, None)


if HAS_TRITON_OP:
    @triton_op("krsi::relpos_attn_fwd", mutates_args=())
    def _fwd_op(qkv: torch.Tensor, pd: torch.Tensor, B: int, S: int, H: int, D: int,
                scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _forward(qkv, pd, B, S, H, D, scale)

    @triton_op("krsi::relpos_attn_bwd", mutates_args=())
    def _bwd_op(dout: torch.Tensor, qkv: torch.Tensor, pd: torch.Tensor, out32: torch.Tensor, lse: torch.Tensor,
                B: int, S: int, H: int, D: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        return _backward_impl(dout, qkv, pd, out32, lse, B, S, H, D, scale)

    def _setup(ctx, inputs, output):
        qkv, pd, B, S, H, D, scale = inputs
        ctx.save_for_backward(qkv, pd, output[1], output[2])
        ctx.dims = (B, S, H, D, scale)

    def _backward(ctx, dout, _d32, _dlse):
        dqkv, dpd = _bwd_op(dout.contiguous(), *ctx.saved_tensors, *ctx.dims)
        return dqkv, dpd, None, None, None, None, None

    _fwd_op.register_autograd(_backward, setup_context=_setup)


def relpos_attention(qkv, pd, B, S, H, D, scale):
    if HAS_TRITON_OP and torch.compiler.is_compiling():
        return _fwd_op(qkv, pd, B, S, H, D, scale)[0]
    return RelPosAttention.apply(qkv, pd, B, S, H, D, scale)


def reference(qkv, pd, B, S, H, D, scale):
    q, k, v = qkv.view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
    pos = torch.arange(S, device=qkv.device)
    bias = pd[:, pos[None, :] - pos[:, None] + S - 1][None].to(qkv.dtype)
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

    with sdpa_kernel(SDPBackend.MATH):
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    return o.transpose(1, 2).reshape(B * S, H * D)


def selftest():
    from common import selftest_autograd  # noqa: PLC0415

    B, S, H, D = 4, 512, 16, 64
    qkv = (0.6 * torch.randn(B * S, 3 * H * D, device="cuda")).to(torch.bfloat16)
    pd = torch.randn(H, 2 * S - 1, device="cuda")
    return selftest_autograd(lambda q, p: relpos_attention(q, p, B, S, H, D, 1.0),
                             lambda q, p: reference(q, p, B, S, H, D, 1.0), [qkv, pd])


if __name__ == "__main__":
    print(selftest())
