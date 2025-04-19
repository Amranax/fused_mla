
import torch

from .mla_base import MLABase
from .General_Layers import *

class AbsorbMLA(MLABase):
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q_nope, q_pe = self.get_query(x, freqs_cis)
        kv, k_pe = self.get_kv(x, freqs_cis)

        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        wkv_b = self.wkv_b.weight
        if self.wkv_b.scale is not None:
            wkv_b = weight_dequant(wkv_b, self.wkv_b.scale, block_size)
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)

        kv_proj = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", kv_proj, wkv_b[:, -self.v_head_dim:])
        x = x.reshape(bsz, seqlen, self.n_heads * self.v_head_dim)
        return self.wo(x)
