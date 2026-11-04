"""Compile-friendly registration for library kernels.

Two paths over the same Triton kernels, chosen per call by the public function:
- under torch.compile (torch >= 2.6): a `torch.library.triton_op` with `register_autograd` — a real
  operator for the compiler (no graph break, fullgraph=True and torch.export work);
- in eager: a plain torch.autograd.Function launching the kernels directly. The registered op is
  much slower on the host in eager (dispatcher + wrap_triton higher-order op: +2.8 ms per T5 training
  step measured), so eager callers never pay for it.
"""
import torch

HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")

if HAS_TRITON_OP:
    triton_op = torch.library.triton_op
    wrap = torch.library.wrap_triton
else:  # pragma: no cover - old torch
    triton_op = None

    def wrap(kernel):
        return kernel


def raw(kernel):
    """Eager path: launch the Triton kernel directly. wrap_triton in eager goes through a higher-order op
    and costs milliseconds of host time per training step (measured), so it is only used when compiling."""
    return kernel
