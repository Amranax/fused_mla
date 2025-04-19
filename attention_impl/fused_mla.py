
import torch

from .mla_base import MLABase
from .General_Layers import *

from kernels.fully_fused_mla import mla_attention_core

class FusedMLA(MLABase):
    """
    Drop‑in replacement of the original `FusedMLA` but now wired to the
    fixed Triton kernel above.  All public behaviour is preserved.
    """

    def __init__(self, args):
        super().__init__(args)

        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.kv_lora_rank,
            ),
            persistent=False,
        )

        self.register_buffer(
            "pe_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.qk_rope_head_dim,
            ),
            persistent=False,
        )

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: torch.Tensor | None):

        B, S_q, _ = x.size()
        end_pos   = start_pos + S_q

        # ---- (1) build query / key / value ---------------------------
        # (B, N, H, NoDim), (B, N, H, RDim)
        q_nope, q_pe         = self.get_query(x, freqs_cis) 

        # (B, S, H, K), (B, S, 1, RDim)
        kv,     k_pe         = self.get_kv   (x, freqs_cis)

        # ---- (2) update caches --------------------------------------
        self.kv_cache[:B, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:B, start_pos:end_pos] = k_pe.squeeze(2)

        # ---- (3) prepare kernel inputs ------------------------------
        #   (B , N , H , …) -> (B , H , N , …)
        q_nope_proj = q_nope.permute(0, 2, 1, 3).contiguous()
        q_pe_proj   = q_pe  .permute(0, 2, 1, 3).contiguous()

        kv_cache_slice = self.kv_cache[:B, :end_pos]
        pe_cache_slice = self.pe_cache[:B, :end_pos]

        # de‑quantise if necessary 
        if self.wkv_b.scale is None:
            wkv_b = self.wkv_b.weight
        else:
            wkv_b = weight_dequant(                   # noqa: F405
                self.wkv_b.weight,
                self.wkv_b.scale,
                block_size,
            )

        # ---- (4) call fused kernel ----------------------------------
        o = mla_attention_core(
            q_nope_proj,           # (B , H , S_q , NoDim)
            q_pe_proj,             # (B , H , S_q , RDim )
            kv_cache_slice,        # (B , S_tot , K)
            pe_cache_slice,        # (B , S_tot , RDim)
            wkv_b,                 # (LRank, H*(NoDim + Vdim))
            mask is None,          # causal flag
            self.softmax_scale,
            start_pos,
        )                           # -> (B , H , S_q , V)

        # ---- (5) merge heads & output projection --------------------
        o = o.transpose(1, 2).reshape(B, S_q, self.n_local_heads * self.v_head_dim)
        return self.wo(o)           # final linear