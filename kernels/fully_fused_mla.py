import torch

import triton
import triton.language as tl

# ---------------------------------------------------------------------
#  Low‑level Triton kernel (forward only)
# ---------------------------------------------------------------------
@triton.jit
def _mla_attn_core_fwd_inner(
    acc, l_i, m_i,
    q_nope_blk, q_pe_blk,          # (BLOCK_M , C1)   and (BLOCK_M , R)
    kv_base_ptr, pe_base_ptr,      # already batch‑offset pointers
    stride_kv_seq, stride_kv_rank, # strides for KV cache   (S_total , K )
    stride_pe_seq, stride_pe_rank, # strides for PE cache   (S_total , R )
    start_m, qk_scale, start_pos,  # scalars
    N_CTX_Q: tl.constexpr,
    N_CTX_TOTAL: tl.constexpr,
    KV_RANK: tl.constexpr,
    QK_ROPE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,           # 1 = prefix, 2 = diagonal, 3 = full
):
    """Inner loop streams blocks of K / V from the caches and updates
    the running softmax statistics plus the accumulator `acc`.
    """

    # -----------------------------------------------------------------
    # 1. Work out the K/V range this stage is allowed to read
    # -----------------------------------------------------------------
    if STAGE == 1:          # causal‑prefix
        lo = 0
        hi = start_pos + start_m * BLOCK_M
    elif STAGE == 2:        # diagonal (current M block)
        lo = start_pos + start_m * BLOCK_M
        hi = lo + BLOCK_M
    else:                   # non‑causal
        lo = 0
        hi = N_CTX_TOTAL

    # -----------------------------------------------------------------
    # 2. Constant offset tensors
    # -----------------------------------------------------------------
    offs_m      = tl.arange(0, BLOCK_M)                                   # query row indices
    offs_k_rank = tl.arange(0, tl.next_power_of_2(KV_RANK))               #   … KV rank
    offs_pe_r   = tl.arange(0, tl.next_power_of_2(QK_ROPE_DIM))           #   … PE rank

    # -----------------------------------------------------------------
    # 3. Iterate over key / value blocks
    # -----------------------------------------------------------------
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # ---- Load KV --------------------------------------------------
        kv_ptrs = kv_base_ptr + offs_n[:, None] * stride_kv_seq \
                            + offs_k_rank[None, :] * stride_kv_rank
        kv_msk  = (offs_n[:, None] < N_CTX_TOTAL) & (offs_k_rank[None, :] < KV_RANK)
        kv_blk  = tl.load(kv_ptrs, mask=kv_msk, other=0.0)                # (BLOCK_N , K)

        # ---- Load PE --------------------------------------------------
        pe_ptrs = pe_base_ptr + offs_n[:, None] * stride_pe_seq \
                            + offs_pe_r[None, :] * stride_pe_rank
        pe_msk  = (offs_n[:, None] < N_CTX_TOTAL) & (offs_pe_r[None, :] < QK_ROPE_DIM)
        pe_blk  = tl.load(pe_ptrs, mask=pe_msk, other=0.0)                # (BLOCK_N , R)

        # ---- Dot‑products (NOPE + RoPE) -------------------------------
        qk_nope = tl.dot(q_nope_blk, tl.trans(kv_blk))                    # (M , N)
        qk_pe   = tl.dot(q_pe_blk,   tl.trans(pe_blk))                    # (M , N)
        scores  = (qk_nope + qk_pe) * qk_scale

        # ---- Causal masking (diagonal stage only) ---------------------
        if STAGE == 2:
            q_abs = start_pos + start_m * BLOCK_M + offs_m                # (M)
            k_abs = offs_n                                               # (N)
            mask  = q_abs[:, None] >= k_abs[None, :]
            scores = tl.where(mask, scores, -float("inf"))

        # ---- Online soft‑max -----------------------------------------
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
    # Tensors
    Q_NOPE, Q_PE, KV_CACHE, PE_CACHE, W_KV, OUT,

    # Strides
    s_qnp_b, s_qnp_h, s_qnp_n, s_qnp_nd,
    s_qpe_b, s_qpe_h, s_qpe_n, s_qpe_rd,
    s_kv_b,  s_kv_n,  s_kv_r,
    s_pe_b,  s_pe_n,  s_pe_rd,
    s_wkv_t,  s_wkv_r,
    s_o_b,   s_o_h,   s_o_n,   s_o_vd,

    # Sizes
    B, H, N_CTX, N_CTX_TOTAL,
    SM_SCALE,

    # Consts
    KV_RANK:            tl.constexpr,
    QK_NOPE_DIM_PROJ:   tl.constexpr,
    QK_ROPE_DIM:        tl.constexpr,
    V_HEAD_DIM:         tl.constexpr,
    BLOCK_M:            tl.constexpr,
    BLOCK_N:            tl.constexpr,
    causal:             tl.constexpr,
):
    
    start_m  = tl.program_id(0)                       
    off_bh   = tl.program_id(1) 
    b        = off_bh // H # Which Batch are we in
    h        = off_bh %  H # Which head are we in

    # Ptr arithmetic :)
    q_nope_ptr = Q_NOPE  + b * s_qnp_b + h * s_qnp_h + start_m * BLOCK_M * s_qnp_n
    q_pe_ptr   = Q_PE    + b * s_qpe_b + h * s_qpe_h + start_m * BLOCK_M * s_qpe_n
    kv_ptr     = KV_CACHE + b * s_kv_b
    pe_ptr     = PE_CACHE + b * s_pe_b
    o_ptr      = OUT      + b * s_o_b  + h * s_o_h  + start_m * BLOCK_M * s_o_n
    wv_ptr     = W_KV      + h * s_wkv_t

    # ------------------------------------------------------------------
    # 2. Load this M‑block of Q
    # ------------------------------------------------------------------
    offs_m      = tl.arange(0, BLOCK_M)
    offs_c1     = tl.arange(0, tl.next_power_of_2(QK_NOPE_DIM_PROJ))
    offs_r      = tl.arange(0, tl.next_power_of_2(QK_ROPE_DIM))

    qnope_msk = (offs_m[:, None] < N_CTX) & (offs_c1[None, :] < QK_NOPE_DIM_PROJ)
    qpe_msk   = (offs_m[:, None] < N_CTX) & (offs_r [None, :] < QK_ROPE_DIM)

    q_nope_blk = tl.load(
        q_nope_ptr + offs_m[:, None] * s_qnp_n + offs_c1[None, :] * s_qnp_nd,
        mask=qnope_msk, other=0.0,
    )                                                  # (M , C1)

    q_pe_blk   = tl.load(
        q_pe_ptr + offs_m[:, None] * s_qpe_n + offs_r[None, :] * s_qpe_rd,
        mask=qpe_msk, other=0.0,
    )                                                  # (M , R)

    # ------------------------------------------------------------------
    # 3. Initialise running soft‑max statistics
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_M, KV_RANK], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.ones([BLOCK_M],               dtype=tl.float32)

    # ------------------------------------------------------------------
    # 4. Decide which inner stages are executed
    # ------------------------------------------------------------------
    stage1 = 3 if not causal else 1
    stage2 = 0 if not causal else 2

    if stage1:
        acc, l_i, m_i = _mla_attn_core_fwd_inner(
            acc, l_i, m_i,
            q_nope_blk, q_pe_blk,
            kv_ptr, pe_ptr,
            s_kv_n, s_kv_r, s_pe_n, s_pe_rd,
            start_m, SM_SCALE, START_POS,
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
            start_m, SM_SCALE, START_POS,
            N_CTX, N_CTX_TOTAL,
            KV_RANK, QK_ROPE_DIM,
            BLOCK_M, BLOCK_N,
            STAGE=stage2,
        )

    # ------------------------------------------------------------------
    # 5. Soft‑max normalisation and projection with W_V
    # ------------------------------------------------------------------
    acc = (acc / l_i[:, None]).to(tl.float32)           # (M , K)

    # ---- load W_V[h]  -----------------------------------------------
    offs_v  = tl.arange(0, tl.next_power_of_2(V_HEAD_DIM))
    offs_k  = tl.arange(0, tl.next_power_of_2(KV_RANK))
    wv_msk  = (offs_v[:, None] < V_HEAD_DIM) & (offs_k[None, :] < KV_RANK)
    w_v_h   = tl.load(
        wv_ptr + offs_v[:, None] * s_wkv_r + offs_k[None, :] * s_wv_k,
        mask=wv_msk, other=0.0,
    ).to(acc.dtype)                                     # (V , K)

    out_blk = tl.dot(acc, tl.trans(w_v_h))              # (M , V)

    # ---- store -------------------------------------------------------
    offs_v_ = tl.arange(0, tl.next_power_of_2(V_HEAD_DIM))
    out_msk = (offs_m[:, None] < N_CTX) & (offs_v_[None, :] < V_HEAD_DIM)

    tl.store(
        o_ptr + offs_m[:, None] * s_o_n + offs_v_[None, :] * s_o_vd,
        out_blk.to(Q_NOPE.dtype),
        mask=out_msk,
    )


# ---------------------------------------------------------------------
#  Torch wrapper – forward only
# ---------------------------------------------------------------------
class _MLAttentionCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nope, q_pe, kv_cache, pe_cache, w_kv,
                causal: bool, sm_scale: float):

        # Ensure contigousy
        q_nope   = q_nope.contiguous()   # (B , H , N_ctx , NoDim)
        q_pe     = q_pe.contiguous()     # (B , H , N_ctx , RDim )

        kv_cache = kv_cache.contiguous() # (B , N_total , LRank)
        pe_cache = pe_cache.contiguous() # (B , N_total , RDim)

        w_kv      = w_kv.contiguous()      # (H*(NoDim + Vdim), LRank )

        B, H, N, NoDim        = q_nope.shape
        _, _, _, RDim         = q_pe.shape
        _, N_tot, LRank        = kv_cache.shape
        temp, _           = w_kv.shape
        
        Vdim = int((temp / H) - NoDim)

        print(f"[DEBUG] q_nope.shape:   {q_nope.shape}, is_contiguous: {q_nope.is_contiguous()}")
        print(f"[DEBUG] q_pe.shape:     {q_pe.shape}, is_contiguous: {q_pe.is_contiguous()}")
        print(f"[DEBUG] kv_cache.shape: {kv_cache.shape}, is_contiguous: {kv_cache.is_contiguous()}")
        print(f"[DEBUG] pe_cache.shape: {pe_cache.shape}, is_contiguous: {pe_cache.is_contiguous()}")
        print(f"[DEBUG] w_kv.shape:     {w_kv.shape}, is_contiguous: {w_kv.is_contiguous()}")

        print(f"[DEBUG] Batch size (B): {B}")
        print(f"[DEBUG] Num heads (H):  {H}")
        print(f"[DEBUG] Context len (N): {N}")
        print(f"[DEBUG] NoDim: {NoDim}, RDim: {RDim}")
        print(f"[DEBUG] N_total: {N_tot}, LRank: {LRank}")
        print(f"[DEBUG] LRank: {LRank}, temp: {temp}")
        print(f"[DEBUG] Vdim: {Vdim}")

        out = torch.empty(
            (B, H, N, Vdim),
            dtype=q_nope.dtype,
        )

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_M"]),
            B * H,
        )

        _mla_attn_core_fwd[grid](
            # ---- tensors ----
            q_nope, q_pe, kv_cache, pe_cache, w_kv, out,

            # Strides
            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2), q_nope.stride(3),
            q_pe.stride(0),    q_pe.stride(1),    q_pe.stride(2),   q_pe.stride(3),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            pe_cache.stride(0), pe_cache.stride(1), pe_cache.stride(2),
            w_kv.stride(0),         w_kv.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),

            # Sizes
            B, H, N, N_tot,
            sm_scale,

            # Consts
            KV_RANK=LRank,
            QK_NOPE_DIM_PROJ=NoDim,
            QK_ROPE_DIM=RDim,
            V_HEAD_DIM=Vdim,
            causal=causal,
        )

        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("Backward pass is not implemented for the fused MLA kernel.")


mla_attention_core = _MLAttentionCore.apply