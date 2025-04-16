import torch
import triton
import math
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _attn_fwd_inner(
   Q, O, L, M,
   k_ptr, v_ptr,
   K_T_offsets, V_offsets,
   block_index_QO,
   scale,
   stride_k_N, stride_v_N,
   BLOCK_SIZE_QO: tl.constexpr,
   BLOCK_SIZE_KV: tl.constexpr,
   causal: tl.constexpr,
   offsets_QO_N: tl.constexpr,
   offsets_KV_N: tl.constexpr,
   N: tl.constexpr, 
   Dh: tl.constexpr
):
   # Q: (BLOCK_SIZE_QO, Dh)
   # O: (BLOCK_SIZE_QO, Dh)
   # L: (BLOCK_SIZE_QO)
   # M: (BLOCK_SIZE_QO)

   if causal:
      lo = block_index_QO * BLOCK_SIZE_QO
      hi = (block_index_QO + 1) * BLOCK_SIZE_QO
   else:
      lo,hi = 0, block_index_QO*BLOCK_SIZE_QO

   # Compiler may optimize based on this
   lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
   hi = tl.multiple_of(hi, BLOCK_SIZE_QO)
   
   K_T_offsets += lo * stride_k_N
   V_offsets += lo * stride_v_N
   offsets_KV_N += lo

   for start_KV in range(lo, hi, BLOCK_SIZE_KV):
      start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

      mask_KV_N = offsets_KV_N < N 
      K_T = tl.load(k_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)
      S = tl.dot(Q, K_T) * scale

      if causal:
         causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
         S += tl.where(causal_mask, 0, -1.0e6)
      
      M_new = tl.maximum(M, tl.max(S, axis=1)) 
      S -= M_new[:, None]

      P = tl.exp2(S)
      L_new = tl.sum(P, axis=1)
      alpha = tl.exp2(M - M_new)
      L = L * alpha + L_new

      V = tl.load(v_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
      O *= alpha[:, None]
      O = tl.dot(P, V, acc=O)

      M = M_new
      K_T_offsets += BLOCK_SIZE_KV * stride_k_N
      V_offsets += BLOCK_SIZE_KV * stride_v_N
      offsets_KV_N += BLOCK_SIZE_KV
   
   return O, L, M
      
@triton.autotune(
   [
      triton.Config(
         {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
         num_stages=num_stages, num_warps=num_warps,
      )
      for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
      for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
      for num_stages in [3]#, 5, 7]
      for num_warps in [4]#, 8, 16]
   ],
   key=["Dh"]
)

@triton.jit
def attn_fwd(
   q_ptr, k_ptr, v_ptr, o_ptr,
   LSE_ptr, scale,
   stride_q_B, stride_q_H, stride_q_N, stride_q_Dh,
   stride_k_B, stride_k_H, stride_k_N, stride_k_Dh,
   stride_v_B, stride_v_H, stride_v_N, stride_v_Dh,
   stride_o_B, stride_o_H, stride_o_N, stride_o_Dh,
   stride_LSE_B, stride_LSE_H, stride_LSE_N,
   B, 
   N: tl.constexpr,
   H: tl.constexpr,
   Dh: tl.constexpr,
   BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
   tl.static_assert(BLOCK_SIZE_KV <= Dh)

   rln2: tl.constexpr = 1.4426950408889634 # 1 / math.log(2)
   """
   e^x = 2^(x*rln2)
   2^(log_2(e)) = e
   e^x = (2^(log_2(e)))^x = 2^(x * log_2(e))
   Fundamental Prop log_2(e) = 1/log_e(2)
   therefore e^x = 2^(x * 1/log_e(2))
   AKA e^x = 2^(x * rln2)
   """
   scale *= rln2


   index_BH = tl.program_id(axis=1)
   index_B = index_BH // H
   index_H = index_BH % H

   q_ptr += index_B * stride_q_B + index_H * stride_q_H
   k_ptr += index_B * stride_k_B + index_H * stride_k_H
   v_ptr += index_B * stride_v_B + index_H * stride_v_H
   o_ptr += index_B * stride_o_B + index_H * stride_o_H

   block_index_QO = tl.program_id(axis=0)
   offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
   offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
   offsets_Dh = tl.arange(0, Dh)

   Q_offsets = offsets_QO_N[:, None] * stride_q_N + offsets_Dh[None, :] * stride_q_Dh   # Shape (BLOCK_SIZE_QO, Dh)

   K_T_offsets = offsets_Dh[:, None] * stride_k_Dh + offsets_KV_N[None, :] * stride_k_N # Shape (Dh, BLOCK_SIZE_KV)

   V_offsets = offsets_KV_N[:, None] * stride_v_N + offsets_Dh[None, :] * stride_v_Dh

   mask_QO_N = offsets_QO_N < N
   Q = tl.load(q_ptr+Q_offsets, mask=mask_QO_N[:,None], other=0.00)

   M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
   L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32) # e^0 = 1
   O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

   # Values below diagonal
   O, L, M = _attn_fwd_inner(
      Q, O, L, M,
      k_ptr, v_ptr,
      K_T_offsets, V_offsets,
      block_index_QO,
      scale,
      stride_k_N, stride_v_N,
      BLOCK_SIZE_QO, BLOCK_SIZE_KV,
      False,
      offsets_QO_N, offsets_KV_N,
      N, Dh
   )

   # Values above diagonal
   O, L, M = _attn_fwd_inner(
      Q, O, L, M,
      k_ptr, v_ptr,
      K_T_offsets, V_offsets,
      block_index_QO,
      scale,
      stride_k_N, stride_v_N,
      BLOCK_SIZE_QO, BLOCK_SIZE_KV,
      True,
      offsets_QO_N, offsets_KV_N,
      N, Dh
   )

   O = O / L[:, None]

   LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO)
   """
   softmax(x_i) = exp(x_i - m_i) / l_i
                = exp(x_i - m_i) / exp(log(l_i))
                = exp(x_i - m_i - log(l_i))
   """

   LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
   LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
   tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

   O_offsets = offsets_QO_N[:, None] * stride_o_N + offsets_Dh[None, :] * stride_o_Dh
   tl.store(o_ptr + O_offsets, O, mask=mask_QO_N[:,None])

class _flashattention(torch.autograd.Function):
   @staticmethod
   def forward(ctx, q, k, v, scale):
      assert q.shape == k.shape == v.shape
      assert q.shape[-1] <= 128, \
            f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
         # the kernel acutally isnt limited but large will overwhelm SRAM 
      assert q.device == k.device and q.device == v.device
      assert q.dtype == k.dtype == v.dtype == torch.float32

      B, H, N, Dh = q.shape

      O = torch.empty_like(q)
      LSE = torch.empty((B, H, N))

      grid = lambda args: (
         triton.cdiv(N, args["BLOCK_SIZE_QO"]),
         B * H,
      )

      attn_fwd[grid](
         q, k, v, O, LSE,
         scale,
         q.stride(0),     q.stride(1),   q.stride(2),  q.stride(3),
         k.stride(0),     k.stride(1),   k.stride(2),   k.stride(3),
         v.stride(0),     v.stride(1),   v.stride(2),   v.stride(3),
         O.stride(0),     O.stride(1),   O.stride(2),   O.stride(3),
         LSE.stride(0), LSE.stride(1), LSE.stride(2),
         B, N, H, Dh,
      )

      ctx.save_for_backward(q, k, v, O, LSE)
      ctx.grid = grid
      ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
      ctx.scale = scale

      return O
   
triton_attention = _flashattention.apply