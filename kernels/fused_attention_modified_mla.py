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
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM_QK: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # Causal = True (1st Stage)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    # Causal = True (1st Stage)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # Causal = False
    else:
        lo, hi = 0, N_CTX
        
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo,  min(hi, N_CTX), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k)

        qk = qk * qk_scale # Scale

        if STAGE == 2:
            # Apply masking for diagonal part
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)

        # Remove local max to prevent overflow
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        # Numerator    
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        # Scaling factor based on old and new max
        alpha = tl.math.exp2(m_i - m_ij)
        
        # Update running denominator sum
        l_i = l_i * alpha + l_ij
        
        # Scale previous acc
        acc = acc * alpha[:, None]

        # Multiply with V
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        
        # Update
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

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
    # Ensure both the Head Dimension for both Q, K and V is less than N Block
    tl.static_assert(BLOCK_N <= HEAD_DIM_QK)
    tl.static_assert(BLOCK_N <= HEAD_DIM_V)

    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)

    off_b = off_hb // H
    off_h = off_hb % H

    qvk_offset = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM_QK, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM_QK, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM_QK, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hb * N_CTX + offs_m

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

def pad_to_pow2(x):
    head_dim = x.shape[-1]
    if head_dim not in {16, 32, 64, 128, 256}:
        next_pow2 = 2 ** math.ceil(math.log2(head_dim))
        pad_len = next_pow2 - head_dim
        padding = (0, pad_len)  # Only pad the last dimension
        x = F.pad(x, padding, value=0).contiguous()
    return x
    
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

        if HEAD_DIM_V not in {16, 32, 64, 128, 256}:
            v = pad_to_pow2(v)
            HEAD_DIM_V = v.shape[3]
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

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not supported")

attention = _attention.apply