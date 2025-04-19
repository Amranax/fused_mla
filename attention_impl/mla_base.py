import torch
from torch import nn
from .Shared_Args import Args
from .General_Layers import *
import math

class MLABase(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn

        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim, dtype=self.dtype)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=self.dtype)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, dtype=self.dtype)

        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, dtype=self.dtype)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), dtype=self.dtype)
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim, dtype=self.dtype)

        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale ** 2

    def get_query(self, x, freqs_cis):
        q = self.wq(x) if self.q_lora_rank == 0 else self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(x.size(0), x.size(1), self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        return q_nope, apply_rotary_emb(q_pe, freqs_cis)

    def get_kv(self, x, freqs_cis):
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        return kv, k_pe
