"""
attention_impl/naive_mla.py - Fixed implementation with properly aligned flash attention
"""

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

from kernels.flash_attention_v2_tri_ref import attention as flash_attention

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

        if args.dtype == "bf16":
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
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Process query embedding
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
            
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # Process key-value embedding
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Handle different attention implementations
        if self.attn_impl == "naive" or self.attn_impl == "naive+flash":
            # Combine query components
            q = torch.cat([q_nope, q_pe], dim=-1)
            
            # Process key-value
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # Update KV cache
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            
            # Get cached keys and values for attention
            cached_k = self.k_cache[:bsz, :end_pos]
            cached_v = self.v_cache[:bsz, :end_pos]
            
            if self.attn_impl == "naive":
                # Compute attention scores and apply them
                scores = torch.einsum("bshd,bthd->bsht", q, cached_k) * self.softmax_scale
                
                if mask is not None:
                    scores += mask.unsqueeze(1)
                
                scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
                x = torch.einsum("bsht,bthd->bshd", scores, cached_v)
                
            else:  # naive+flash
                # Use flash attention implementation
                # Ensure tensors are contiguous for optimal performance
                q_cont = q.contiguous()
                k_cont = cached_k.contiguous()
                v_cont = cached_v.contiguous()
                
                x = flash_attention(
                    q_cont,                # Query [bsz, seqlen, n_heads, head_dim]
                    k_cont,                # Key [bsz, end_pos, n_heads, head_dim]
                    v_cont,                # Value [bsz, end_pos, n_heads, head_dim]
                    True,                  # causal=True for autoregressive attention
                    self.softmax_scale     # Same scale factor as naive implementation
                )
        else:  # absorb implementation
            wkv_b = self.wkv_b.weight
            if self.wkv_b.scale is not None:
                wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
                
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            
            if mask is not None:
                scores += mask.unsqueeze(1)
                
            scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
            
            kv_cache = self.kv_cache[:bsz, :end_pos]
            wkv_b_v = wkv_b[:, -self.v_head_dim:]
            
            x = torch.einsum("bsht,btc->bshc", scores, kv_cache)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b_v)
            
        # Final projection
        x = x.reshape(bsz, seqlen, self.n_heads * self.v_head_dim)
        x = self.wo(x)
        return x