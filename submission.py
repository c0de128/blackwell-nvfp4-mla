"""
NVFP4 Gated Dual GEMM Submission for Blackwell Hackathon
Challenge #3: C = silu(A @ B1) * (A @ B2)
"""

import torch
import torch.nn.functional as F

from task import input_t, output_t


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format for cuBLAS."""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def custom_kernel(data: input_t) -> output_t:
    """
    Block-scale fp4 dual gemm with silu activation.
    Computes: C = silu(A @ B1.T) * (A @ B2.T)
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data

    # Get dimensions
    m, n, l = c.shape

    # Process each batch
    for i in range(l):
        # Convert scale factors to blocked format (like reference)
        scale_a = to_blocked(sfa[:, :, i])
        scale_b1 = to_blocked(sfb1[:, :, i])
        scale_b2 = to_blocked(sfb2[:, :, i])

        # (m, k) @ (n, k).T -> (m, n)
        out1 = torch._scaled_mm(
            a[:, :, i],
            b1[:, :, i].transpose(0, 1),
            scale_a,
            scale_b1,
            bias=None,
            out_dtype=torch.float32,
        )

        out2 = torch._scaled_mm(
            a[:, :, i],
            b2[:, :, i].transpose(0, 1),
            scale_a,
            scale_b2,
            bias=None,
            out_dtype=torch.float32,
        )

        # Apply SiLU and gate, store as FP16
        c[:, :, i] = (F.silu(out1) * out2).to(torch.float16)

    return c
