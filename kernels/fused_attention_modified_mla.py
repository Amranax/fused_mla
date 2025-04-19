import torch
import torch.nn.functional as F

import triton
import triton.language as tl

import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, K, V, # Pointers to K and V tensors
    k_base_ptr, v_base_ptr, # Base pointers for the current batch/head
    stride_kn, stride_kk, stride_vk, stride_vn, # Strides for K and V
    start_m, qk_scale,
    offs_m: tl.constexpr, # Offsets for the M dimension (query sequence)
    N_CTX: tl.constexpr, HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr
    ):
    # Causal = True (1st Stage)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    # Causal = True (2nd Stage) - Diagonal block
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M) # Should already be true
    # Causal = False
    else: # STAGE == 3
        lo, hi = 0, N_CTX

    # Ranges for Kopfsets within the block
    offs_k_dim = tl.arange(0, triton.next_power_of_2(HEAD_DIM_QK))
    offs_v_dim = tl.arange(0, triton.next_power_of_2(HEAD_DIM_V))

    # loop over k, v and update accumulator
    for start_n in range(lo, tl.minimum(hi, N_CTX), BLOCK_N):
        # -- compute qk ----

        # K Pointers and Mask
        offs_n_block = start_n + tl.arange(0, BLOCK_N) # offsets along N dimension for the current block
        k_ptrs = k_base_ptr + offs_k_dim[:, None] * stride_kk + offs_n_block[None, :] * stride_kn
        k_mask = (offs_k_dim[:, None] < HEAD_DIM_QK) & (offs_n_block[None, :] < N_CTX)
        # Load K, masking out of bounds elements (necessary if N_CTX is not multiple of BLOCK_N)
        # Use other=0.0 or NaN. 0.0 is safer for dot product.
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK
        qk = tl.dot(q, k) # shape: (BLOCK_M, BLOCK_N)
        qk = qk * qk_scale

        # Apply causal mask if needed
        if STAGE == 2: # Only apply causal mask in the diagonal block stage
            offs_m_block = offs_m # Already passed, represents the M-dim indices for this Q block
            mask = offs_m_block[:, None] >= offs_n_block[None, :]
            # Mask out future positions - Add large negative number where mask is False
            qk = tl.where(mask, qk, -float('inf')) # Use float('inf') for softmax masking


        # Compute Softmax Stats
        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # Current block max along N dim
        qk = qk - m_ij[:, None] # Stabilize exp
        p = tl.math.exp2(qk) # Numerator part
        l_ij = tl.sum(p, 1) # Current block sum along N dim

        # Update Accumulator Stats
        alpha = tl.math.exp2(m_i - m_ij) # Scale factor for previous stats
        l_i = l_i * alpha + l_ij # Update total sum (denominator)
        m_i = m_ij # Update max

        # Update Accumulator Value
        acc = acc * alpha[:, None] # Scale previous accumulated values

        # V Pointers and Mask
        v_ptrs = v_base_ptr + offs_n_block[:, None] * stride_vk + offs_v_dim[None, :] * stride_vn
        v_mask = (offs_n_block[:, None] < N_CTX) & (offs_v_dim[None, :] < HEAD_DIM_V)
        # Load V, masking out of bounds elements
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Compute PV
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc) # acc += Pij * Vj

        # K and V pointers are recalculated each iteration, no tl.advance needed

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        # Basic configs, add more as needed for tuning
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
        # Add configs with more stages/warps if desired
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['N_CTX', 'HEAD_DIM_QK', 'HEAD_DIM_V', 'causal'] # Add causal to key
)
@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,  # Pointers to tensors
    stride_qb, stride_qh, stride_qm, stride_qk,  # Q strides
    stride_kb, stride_kh, stride_kn, stride_kk,  # K strides
    stride_vb, stride_vh, stride_vk, stride_vn,  # V strides
    stride_ob, stride_oh, stride_om, stride_on,  # Out strides
    B, H, N_CTX,  # Batch, Heads, Sequence Length
    HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    causal: tl.constexpr # Pass causal flag directly
    ):

    start_m = tl.program_id(0) # Index along N_CTX for Q
    off_hb = tl.program_id(1)  # Combined Batch and Head index
    off_b = off_hb // H
    off_h = off_hb % H

    # Calculate base pointers for K, V, O for the current batch/head
    # Note: Q base pointer is calculated dynamically before loading Q
    k_base_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_base_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_base_ptr = Out + off_b * stride_ob + off_h * stride_oh
    m_base_ptr = M + off_hb * N_CTX # M is (B*H, N_CTX)

    # Initialize Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # Row offsets for Q/O block
    offs_q_k = tl.arange(0, triton.next_power_of_2(HEAD_DIM_QK))              # Col offsets for Q block (QK head dim)
    offs_o_v = tl.arange(0, triton.next_power_of_2(HEAD_DIM_V))              # Col offsets for O block (V head dim)

    # Initialize pointers to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) # Denom starts at 0, safe due to exp2(m_i - m_ij) later? Let's keep 1.0 as before for safety with the scaling logic.
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 # Original had 1.0 - corresponds to exp2(-inf - (-inf)) = exp2(0) = 1? Safer.

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    # Load scales
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # Load Q
    q_offset = off_b * stride_qb + off_h * stride_qh # Base offset for batch/head
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_q_k[None, :] * stride_qk
    q_mask = (offs_m[:, None] < N_CTX) & (offs_q_k[None, :] < HEAD_DIM_QK)
    # Load Q for the current block, masking out rows beyond N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Determine computation stage based on causal flag
    # STAGE=1: causal=False - process all KV in one go
    # STAGE=3: causal=True - process KV in two stages (off-band and on-band)
    inner_stage = 3 if not causal else 1 # Stage for _attn_fwd_inner loop limits

    # Stage 1: Process off-band blocks if causal, or all blocks if not causal
    if not causal: # equivalent to original STAGE=1 -> inner_stage=3
       acc, l_i, m_i = _attn_fwd_inner(
           acc, l_i, m_i, q, K, V, k_base_ptr, v_base_ptr,
           stride_kn, stride_kk, stride_vk, stride_vn,
           start_m, qk_scale, offs_m,
           N_CTX, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_M, BLOCK_N,
           STAGE=3 # Process all N_CTX
       )
    else: # causal=True, original STAGE=3
       # Stage 1: Off-band computation (before diagonal)
       acc, l_i, m_i = _attn_fwd_inner(
           acc, l_i, m_i, q, K, V, k_base_ptr, v_base_ptr,
           stride_kn, stride_kk, stride_vk, stride_vn,
           start_m, qk_scale, offs_m,
           N_CTX, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_M, BLOCK_N,
           STAGE=1 # Process N < start_m * BLOCK_M
       )
       # Stage 2: On-band computation (diagonal block with causal mask)
       acc, l_i, m_i = _attn_fwd_inner(
           acc, l_i, m_i, q, K, V, k_base_ptr, v_base_ptr,
           stride_kn, stride_kk, stride_vk, stride_vn,
           start_m, qk_scale, offs_m,
           N_CTX, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_M, BLOCK_N,
           STAGE=2 # Process N within the diagonal block M, apply causal mask
       )

    # --- Epilogue ---
    # Store Max statistics (required for backward pass if implemented, useful for debugging)
    m_ptrs = m_base_ptr + offs_m
    m_mask = offs_m < N_CTX
    tl.store(m_ptrs, m_i, mask=m_mask) # Store max for each row

    # Normalize accumulator
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i) # Avoid division by zero
    acc_scale = (1.0 / l_i_safe) # Compute scale factor
    acc = acc * acc_scale[:, None] # Apply scale

    # Write output block
    o_ptrs = o_base_ptr + offs_m[:, None] * stride_om + offs_o_v[None, :] * stride_on
    o_mask = (offs_m[:, None] < N_CTX) & (offs_o_v[None, :] < HEAD_DIM_V)
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_mask)


# --- Python Wrapper ---

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # Input shape: (B, H, N_CTX, HEAD_DIM) - Make sure inputs are contiguous!
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        B, H, N_CTX, HEAD_DIM_K = q.shape
        HEAD_DIM_V = v.shape[3]

        # Removed padding logic - Kernel now handles arbitrary head dims via masking
        # assert HEAD_DIM_V >= 16, f"HEAD_DIM_V={HEAD_DIM_V} must be >= 16" # Min head dim check might still be relevant depending on kernel block sizes
        # assert HEAD_DIM_K >= 16, f"HEAD_DIM_K={HEAD_DIM_K} must be >= 16"

        o = torch.empty_like(v) # Output has same shape as V

        # M tensor for storing softmax max values (optional, but kept from original)
        M = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

        # Grid calculation
        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), B * H, 1)

        # print(f"[DEBUG] q.shape: {q.shape}, q.stride(): {q.stride()}")
        # print(f"[DEBUG] k.shape: {k.shape}, k.stride(): {k.stride()}")
        # print(f"[DEBUG] v.shape: {v.shape}, v.stride(): {v.stride()}")
        # print(f"[DEBUG] o.shape: {o.shape}, o.stride(): {o.stride()}")
        # print(f"[DEBUG] N_CTX: {N_CTX}, HEAD_DIM_K: {HEAD_DIM_K}, HEAD_DIM_V: {HEAD_DIM_V}, B: {B}, H: {H}")
        # print(f"[DEBUG] causal: {causal}, sm_scale: {sm_scale}")

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            # Strides: (Batch, Head, SeqLen, HeadDim)
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N_CTX,
            HEAD_DIM_QK=HEAD_DIM_K,
            HEAD_DIM_V=HEAD_DIM_V,
            causal=causal # Pass causal flag to kernel
            # BLOCK_M, BLOCK_N are handled by autotuner
        )

        # Removed unpadding logic

        # Context saving for backward pass (if needed later)
        # ctx.save_for_backward(q, k, v, o, M)
        # ctx.sm_scale = sm_scale
        # ctx.causal = causal
        # ctx.HEAD_DIM_K = HEAD_DIM_K
        # ctx.HEAD_DIM_V = HEAD_DIM_V

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not supported in this modified kernel")

attention = _attention.apply

# Example Usage (make sure inputs are on CUDA and contiguous)
# B, H, N_CTX, HDIM_K, HDIM_V = 2, 4, 100, 50, 60 # Example non-power-of-2 dims
# q = torch.randn(B, H, N_CTX, HDIM_K, device=DEVICE, dtype=torch.float16).contiguous()
# k = torch.randn(B, H, N_CTX, HDIM_K, device=DEVICE, dtype=torch.float16).contiguous()
# v = torch.randn(B, H, N_CTX, HDIM_V, device=DEVICE, dtype=torch.float16).contiguous()
# sm_scale = 1.0 / math.sqrt(HDIM_K)
# causal = True
#
# output = attention(q, k, v, causal, sm_scale)
# print("Output shape:", output.shape)
# print("Output dtype:", output.dtype)