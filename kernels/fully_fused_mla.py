import torch
import triton
import triton.language as tl

@triton.jit
def _mla_attn_core_fwd_inner(
    acc, l_i, m_i,
    q_nope_blk, q_pe_blk,
    kv_base_ptr, pe_base_ptr,
    stride_kv_seq, stride_kv_rank,
    stride_pe_seq, stride_pe_rank,
    start_m, qk_scale,
    N_CTX_Q: tl.constexpr,
    N_CTX_TOTAL: tl.constexpr,
    KV_RANK: tl.constexpr,
    QK_ROPE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    if STAGE == 1:
        lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = lo + BLOCK_M
    else:
        lo = 0
        hi = N_CTX_TOTAL

    offs_m      = tl.arange(0, BLOCK_M)
    offs_k_rank = tl.arange(0, triton.next_power_of_2(KV_RANK))
    offs_pe_r   = tl.arange(0, triton.next_power_of_2(QK_ROPE_DIM))

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        kv_ptrs = kv_base_ptr + offs_n[:, None] * stride_kv_seq \
                  + offs_k_rank[None, :] * stride_kv_rank
        kv_msk  = (offs_n[:, None] < N_CTX_TOTAL) & (offs_k_rank[None, :] < KV_RANK)
        kv_blk  = tl.load(kv_ptrs, mask=kv_msk, other=0.0)

        pe_ptrs = pe_base_ptr + offs_n[:, None] * stride_pe_seq \
                  + offs_pe_r[None, :] * stride_pe_rank
        pe_msk  = (offs_n[:, None] < N_CTX_TOTAL) & (offs_pe_r[None, :] < QK_ROPE_DIM)
        pe_blk  = tl.load(pe_ptrs, mask=pe_msk, other=0.0)

        qk_nope = tl.dot(q_nope_blk, tl.trans(kv_blk))
        qk_pe   = tl.dot(q_pe_blk,   tl.trans(pe_blk))
        scores  = (qk_nope + qk_pe) * qk_scale

        if STAGE == 2:
            q_abs = start_m * BLOCK_M + offs_m
            k_abs = offs_n
            mask  = q_abs[:, None] >= k_abs[None, :]
            scores = tl.where(mask, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        scores = scores - m_ij[:, None]
        p_ij   = tl.exp2(scores)
        l_ij   = tl.sum(p_ij, 1)

        alpha  = tl.exp2(m_i - m_ij)
        l_i    = l_i * alpha + l_ij
        m_i    = m_ij
        acc    = acc * alpha[:, None] + tl.dot(p_ij.to(kv_blk.dtype), kv_blk)

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_stages=1, num_warps=4),
    ],
    key=[
        "N_CTX_Q", "N_CTX_TOTAL",
        "KV_RANK", "QK_NOPE_DIM_PROJ", "QK_ROPE_DIM",
        "causal",
    ],
)
@triton.jit
def _mla_attn_core_fwd(
    Q_NOPE, Q_PE, KV_CACHE, PE_CACHE, W_KV, OUT,
    s_qnp_b, s_qnp_h, s_qnp_n, s_qnp_nd,
    s_qpe_b, s_qpe_h, s_qpe_n, s_qpe_rd,
    s_kv_b,  s_kv_n,  s_kv_r,
    s_pe_b,  s_pe_n,  s_pe_rd,
    s_wkv_t, s_wkv_r, s_wv_k, # Strides for W_KV tensor
    s_o_b,   s_o_h,   s_o_n,   s_o_vd,
    B, H, N_CTX, N_CTX_TOTAL,
    SM_SCALE,
    KV_RANK:            tl.constexpr,
    QK_NOPE_DIM_PROJ:   tl.constexpr,
    QK_ROPE_DIM:        tl.constexpr,
    V_HEAD_DIM:         tl.constexpr,
    BLOCK_M:            tl.constexpr,
    BLOCK_N:            tl.constexpr,
    causal:             tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    b       = off_bh // H
    h       = off_bh %  H

    q_nope_ptr = Q_NOPE   + b * s_qnp_b + h * s_qnp_h + start_m * BLOCK_M * s_qnp_n
    q_pe_ptr   = Q_PE     + b * s_qpe_b + h * s_qpe_h + start_m * BLOCK_M * s_qpe_n
    kv_ptr     = KV_CACHE + b * s_kv_b
    pe_ptr     = PE_CACHE + b * s_pe_b
    o_ptr      = OUT      + b * s_o_b   + h * s_o_h   + start_m * BLOCK_M * s_o_n
    # Calculate pointer to the start of the weights for the current head h
    # s_wkv_t is the stride (in elements) between the start of consecutive heads in W_KV
    wv_ptr     = W_KV     + h * s_wkv_t

    offs_m      = tl.arange(0, BLOCK_M)
    offs_c1     = tl.arange(0, triton.next_power_of_2(QK_NOPE_DIM_PROJ))
    offs_r      = tl.arange(0, triton.next_power_of_2(QK_ROPE_DIM))

    qnope_msk = (offs_m[:, None] < N_CTX) & (offs_c1[None, :] < QK_NOPE_DIM_PROJ)
    qpe_msk   = (offs_m[:, None] < N_CTX) & (offs_r [None, :] < QK_ROPE_DIM)

    q_nope_blk = tl.load(
        q_nope_ptr + offs_m[:, None] * s_qnp_n + offs_c1[None, :] * s_qnp_nd,
        mask=qnope_msk, other=0.0,
    )

    q_pe_blk   = tl.load(
        q_pe_ptr + offs_m[:, None] * s_qpe_n + offs_r[None, :] * s_qpe_rd,
        mask=qpe_msk, other=0.0,
    )

    acc = tl.zeros([BLOCK_M, KV_RANK], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

    stage1 = 3 if not causal else 1
    stage2 = 0 if not causal else 2

    if stage1:
        acc, l_i, m_i = _mla_attn_core_fwd_inner(
            acc, l_i, m_i,
            q_nope_blk, q_pe_blk,
            kv_ptr, pe_ptr,
            s_kv_n, s_kv_r, s_pe_n, s_pe_rd,
            start_m, SM_SCALE,
            N_CTX, N_CTX_TOTAL,
            KV_RANK, QK_ROPE_DIM,
            BLOCK_M, BLOCK_N,
            STAGE=stage1,
        )

    if stage2:
        acc, l_i, m_i = _mla_attn_core_fwd_inner(
            acc, l_i, m_i,
            q_nope_blk, q_pe_blk,
            kv_ptr, pe_ptr,
            s_kv_n, s_kv_r, s_pe_n, s_pe_rd,
            start_m, SM_SCALE,
            N_CTX, N_CTX_TOTAL,
            KV_RANK, QK_ROPE_DIM,
            BLOCK_M, BLOCK_N,
            STAGE=stage2,
        )

    acc = (acc / l_i[:, None]).to(tl.float32)

    offs_v  = tl.arange(0, triton.next_power_of_2(V_HEAD_DIM))
    offs_k  = tl.arange(0, triton.next_power_of_2(KV_RANK))
    wv_msk  = (offs_v[:, None] < V_HEAD_DIM) & (offs_k[None, :] < KV_RANK)

    # Load W_V for the current head h using wv_ptr and strides s_wkv_r, s_wv_k
    # s_wkv_r: stride for V dimension (rows within head's block)
    # s_wv_k: stride for K dimension (columns)
    w_v_h = tl.load(
        wv_ptr + offs_v[:, None] * s_wkv_r + offs_k[None, :] * s_wv_k,
        mask=wv_msk, other=0.0,
    ).to(acc.dtype)

    out_blk = tl.dot(acc, tl.trans(w_v_h))

    offs_v_ = tl.arange(0, triton.next_power_of_2(V_HEAD_DIM))
    out_msk = (offs_m[:, None] < N_CTX) & (offs_v_[None, :] < V_HEAD_DIM)

    tl.store(
        o_ptr + offs_m[:, None] * s_o_n + offs_v_[None, :] * s_o_vd,
        out_blk.to(Q_NOPE.dtype.element_ty), # Use element_ty for dtype access
        mask=out_msk,
    )


class _MLAttentionCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nope, q_pe, kv_cache, pe_cache, w_kv,
                causal: bool, sm_scale: float):

        q_nope   = q_nope.contiguous()
        q_pe     = q_pe.contiguous()
        kv_cache = kv_cache.contiguous()
        pe_cache = pe_cache.contiguous()
        w_kv     = w_kv.contiguous() # Assumed shape (H*Vdim, LRank)

        B, H, N, NoDim      = q_nope.shape
        _, _, _, RDim       = q_pe.shape
        _, N_tot, LRank     = kv_cache.shape
        # w_kv is 2D, get its shape
        TotalWeightRows, LRank_w = w_kv.shape

        # Basic dimension checks
        assert LRank == LRank_w, f"Oh no! KV_RANK mismatch: kv_cache has {LRank}, w_kv has {LRank_w}"
        assert TotalWeightRows % H == 0, f"Hmm, total rows in w_kv ({TotalWeightRows}) isn't divisible by H ({H})"

        # Calculate Vdim based on the 2D shape
        Vdim = TotalWeightRows // H

        out = torch.empty(
            (B, H, N, Vdim),
            dtype=q_nope.dtype,
            device=q_nope.device
        )

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_M"]),
            B * H,
        )

        # Calculate strides for the kernel based on the 2D w_kv tensor
        # Assumes w_kv is contiguous, layout (H*Vdim, LRank)
        # Stride0 = LRank, Stride1 = 1
        stride_wkv_0 = w_kv.stride(0) # stride between rows (Vdim dimension within a head)
        stride_wkv_1 = w_kv.stride(1) # stride between columns (LRank dimension)

        # s_wkv_t: element stride between the start of weights for consecutive heads
        #          = number of rows per head * stride between rows
        s_wkv_t = Vdim * stride_wkv_0

        # s_wkv_r: element stride between rows within a head's V-block
        #          = stride between rows in the original tensor
        s_wkv_r = stride_wkv_0

        # s_wv_k: element stride between columns (K-dimension)
        #         = stride between columns in the original tensor
        s_wv_k  = stride_wkv_1


        _mla_attn_core_fwd[grid](
            q_nope, q_pe, kv_cache, pe_cache, w_kv, out,

            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2), q_nope.stride(3),
            q_pe.stride(0),   q_pe.stride(1),   q_pe.stride(2),   q_pe.stride(3),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            pe_cache.stride(0), pe_cache.stride(1), pe_cache.stride(2),
            # Pass the calculated strides for the 2D w_kv tensor
            s_wkv_t, s_wkv_r, s_wv_k,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),

            B, H, N, N_tot,
            sm_scale,

            KV_RANK=LRank,
            QK_NOPE_DIM_PROJ=NoDim,
            QK_ROPE_DIM=RDim,
            V_HEAD_DIM=Vdim, # Use the calculated Vdim
            causal=causal,
        )

        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("Oops, the backward pass isn't ready for this fused MLA kernel yet.")


mla_attention_core = _MLAttentionCore.apply