"""
Flash Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
This original implementation was taken from the Triton-Lange Repo

The code was changed to suit my projects needs
- Removed Backward pass, AMD support, TMA support
- Added support for different head sizes


Credits: OpenAI kernel team
https://github.com/triton-lang/triton


"""
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    k_base, v_base,     #
                    start_n_offset, qk_scale,  #
                    stride_kk, stride_kn, stride_vk, stride_vn,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM_QK: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    
    for start_n in range(start_n_offset, min(start_n_offset + BLOCK_M, N_CTX), BLOCK_N):
        start_n_aligned = tl.multiple_of(start_n, BLOCK_N)

        # K load
        k_row = tl.arange(0, HEAD_DIM_QK)
        k_col = offs_n + start_n_aligned
        k_mask = k_col[None, :] < N_CTX
        k_ptrs = k_base + k_row[:, None] * stride_kk + k_col[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk
        qk = tl.dot(q, k)
        qk *= qk_scale

        if STAGE == 2:
            causal_mask = offs_m[:, None] >= k_col[None, :]
            qk = tl.where(causal_mask, qk, -1e6)

        # logsumexp-style
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # V load
        v_row = k_col
        v_col = tl.arange(0, acc.shape[1])
        v_mask = v_row[:, None] < N_CTX
        v_ptrs = v_base + v_row[:, None] * stride_vk + v_col[None, :] * stride_vn
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # acc update
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # m_i update
        m_i = m_ij

    return acc, l_i, m_i



# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [64, 128]\
#     for BN in [32, 64]\
#     for s in [3, 4, 7]\
#     for w in [4, 8]\
# ]
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM_QK", "HEAD_DIM_V"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qb, stride_qh, stride_qm, stride_qk,  #
              stride_kb, stride_kh, stride_kn, stride_kk,  #
              stride_vb, stride_vh, stride_vk, stride_vn,  #
              stride_ob, stride_oh, stride_om, stride_on,  #
              B, H, N_CTX,  #
              HEAD_DIM_QK: tl.constexpr,  #
              HEAD_DIM_V: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):

    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H

    # Base offsets
    q_offset = Q + off_b * stride_qb + off_h * stride_qh
    k_offset = K + off_b * stride_kb + off_h * stride_kh
    v_offset = V + off_b * stride_vb + off_h * stride_vh
    o_offset = Out + off_b * stride_ob + off_h * stride_oh

    # index helpers
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)

    # Load q
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX), other=0.0)

    # Init accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # STAGE 1 = causal off-band
    # STAGE 2 = causal on-band
    # STAGE 3 = full attention
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_offset, v_offset,
                                        0, qk_scale,
                                        stride_kk, stride_kn, stride_vk, stride_vn,
                                        BLOCK_M, HEAD_DIM_QK, BLOCK_N,
                                        3 if STAGE == 3 else 1, offs_m, offs_n, N_CTX)

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_offset, v_offset,
                                        start_m * BLOCK_M, qk_scale,
                                        stride_kk, stride_kn, stride_vk, stride_vn,
                                        BLOCK_M, HEAD_DIM_QK, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX)

    # Finalize output
    m_i += tl.log2(l_i)
    acc /= l_i[:, None]

    m_ptrs = M + off_hb * N_CTX + offs_m
    tl.store(m_ptrs, m_i)

    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))

def pad_to_pow2(x):
    head_dim = x.shape[-1]
    if head_dim not in {16, 32, 64, 128, 256}:
        next_pow2 = 2 ** math.ceil(math.log2(head_dim))
        pad_len = next_pow2 - head_dim
        padding = (0, pad_len)  # Only pad the last dimension
        x = F.pad(x, padding, value=0).contiguous()
    return x

def unpad_from_pow2(x, orig_head_dim):
    return x[..., :orig_head_dim].contiguous() # Not necessary to make contigous 

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):

        # Input shape (B, N_CTX, H, HEAD_DIM)
        B = q.shape[0]
        H = q.shape[1]     # Actual number of heads
        N_CTX = q.shape[2] # Actual sequence length

        # Pad if needed
        HEAD_DIM_V = v.shape[3] # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_K = q.shape[3]

        orig_head_dim_v = HEAD_DIM_V
        if HEAD_DIM_V not in {16, 32, 64, 128, 256}:
            v = pad_to_pow2(v)
            HEAD_DIM_V = v.shape[3]

        orig_head_dim_k = HEAD_DIM_K
        if HEAD_DIM_K not in {16, 32, 64, 128, 256}:
            k = pad_to_pow2(k)
            q = pad_to_pow2(q)
            HEAD_DIM_K = q.shape[3]

        assert HEAD_DIM_V in {16, 32, 64, 128, 256}, f"HEAD_DIM_V={HEAD_DIM_V} is invalid"
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}, f"HEAD_DIM_V={HEAD_DIM_K} is invalid"

        stage = 3 if causal else 1

        o = torch.empty_like(v)
        M = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

        # Grid dim 0 depends on N_CTX (sequence length along M dim)
        # Grid dim 1 depends on B * H (batch * heads)
        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), B * H, 1)

        # print(f"[DEBUG] q.shape: {q.shape}, q.stride(): {q.stride()}, q.is_contiguous(): {q.is_contiguous()}")
        # print(f"[DEBUG] k.shape: {k.shape}, k.stride(): {k.stride()}, k.is_contiguous(): {k.is_contiguous()}")
        # print(f"[DEBUG] v.shape: {v.shape}, v.stride(): {v.stride()}, v.is_contiguous(): {v.is_contiguous()}")
        # print(f"[DEBUG] o.shape: {o.shape}, o.stride(): {o.stride()}, o.is_contiguous(): {o.is_contiguous()}")
        # print(f"[DEBUG] Correct N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM_K}, B: {B}, Correct H: {H}")
        # print(f"[DEBUG] causal: {causal}, sm_scale: {sm_scale}, stage: {stage}")

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o, 
          # Stride arguments - REORDERED to match kernel expectations (B, H, M/N, K/V)
          #        (B=0,     H=1,         N_CTX=2,      HDIM=3)

          #   stride_qz,   stride_qh,   stride_qm,   stride_qk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # Strides for Q
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # Strides for K
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  # Strides for V
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  # Strides for O
            B, H,
            N_CTX=N_CTX,
            HEAD_DIM_QK=HEAD_DIM_K,
            HEAD_DIM_V=HEAD_DIM_V,
            STAGE=stage,
        )
        # Truncate q,k,v if they were changed
        if orig_head_dim_v != HEAD_DIM_V:
            v = unpad_from_pow2(v, orig_head_dim_v)
            o = unpad_from_pow2(o, orig_head_dim_v)
        if orig_head_dim_k != HEAD_DIM_K:
            q = unpad_from_pow2(v, orig_head_dim_k)
            k = unpad_from_pow2(v, orig_head_dim_k)

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not supported")

attention = _attention.apply