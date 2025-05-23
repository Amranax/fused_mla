import torch

from .mla_base import MLABase

class NaiveMLA(MLABase):
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

        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)

        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        x = x.reshape(bsz, seqlen, self.n_heads * self.v_head_dim)
        return self.wo(x)