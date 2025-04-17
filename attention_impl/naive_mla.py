
# Torch Imports
import torch
from torch import nn

# Transformer Imports
import logging

# Utils
from typing import List, Optional, Tuple, Union
import math

# Local Imports
from .Shared_Args import Args
from .General_Layers import *

from kernels.flash_attention_v2_tri_ref import _attention as _flashattention


triton_attention = _flashattention.apply

logger = logging.getLogger(__name__)

class MLA(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.attn_impl = args.attn_impl

        if Args.dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float8_e4m3fn

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
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.attn_impl == "naive" or self.attn_impl == "naive+flash":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # print(f"Processing x input of shape {x.size()}")
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        if self.attn_impl == "naive" or self.attn_impl == "naive+flash":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # print(f"K Shape {k.shape} V Shape {v.shape}")

            if self.attn_impl == "naive":
                scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        if self.attn_impl != "naive+flash": 
            if mask is not None:
                scores += mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        if self.attn_impl == "naive":
            # Fix: Cast to float32 for einsum operation
            scores_float = scores.float()
            v_cache_float = self.v_cache[:bsz, :end_pos].float()
            # print(f"Scores Shape {k.shape} V Cache Shape {v_cache_float.shape}")
            
            x = torch.einsum("bsht,bthd->bshd", scores_float, v_cache_float)
            
            # Cast back to original dtype
            x = x.to(q.dtype)
        elif self.attn_impl == "naive+flash":
            # Using flash attention with the cached keys and values
            cached_k = self.k_cache[:bsz, :end_pos]
            cached_v = self.v_cache[:bsz, :end_pos]
            
            # Apply flash attention
            x = triton_attention(
                q,                  # Query [bsz, seqlen, n_heads, head_dim]
                cached_k,           # Key [bsz, end_pos, n_heads, head_dim]
                cached_v,           # Value [bsz, end_pos, n_heads, head_dim]
                False,
                self.softmax_scale  # Scale factor
            )
        else:
            # Fix: Cast to float32 for einsum operation
            scores_float = scores.float()
            kv_cache_float = self.kv_cache[:bsz, :end_pos].float()
            wkv_b_float = wkv_b[:, -self.v_head_dim:].float()
            
            x = torch.einsum("bsht,btc->bshc", scores_float, kv_cache_float)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b_float)
            
            # Cast back to original dtype
            x = x.to(q.dtype)
            
        x = self.wo(x.flatten(2))
        return x