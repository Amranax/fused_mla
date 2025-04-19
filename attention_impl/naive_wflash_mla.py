import torch

from .mla_base import MLABase
from kernels.fused_attention_modified_mla import attention as flash_attention

class FlashMLA(MLABase):
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q_nope, q_pe = self.get_query(x, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv, k_pe = self.get_kv(x, freqs_cis)
        kv = self.wkv_b(self.kv_norm(kv)).view(bsz, seqlen, self.n_local_heads, -1)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v

        cached_k = self.k_cache[:bsz, :end_pos]
        cached_v = self.v_cache[:bsz, :end_pos]
        
        # Reshape for flash_attention kernel
        q = q.permute(0, 2, 1, 3).contiguous()
        k = cached_k.permute(0, 2, 1, 3).contiguous()
        v = cached_v .permute(0, 2, 1, 3).contiguous()

        x = flash_attention(
            q, 
            k, 
            v, 
            True, 
            self.softmax_scale
        )

        x = x.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.n_heads * self.v_head_dim)

        return self.wo(x)
