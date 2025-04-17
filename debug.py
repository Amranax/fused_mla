#!/usr/bin/env python3
"""
Debug script for investigating differences between MLA implementations
"""

import torch
import time
import numpy as np
import os

# Local imports
from attention_impl.naive_mla import MLA
from attention_impl.Shared_Args import Args
from attention_impl.General_Layers import precompute_freqs_cis, EmbeddingLayer


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def debug_implementations(args, seq_len=128, batch_size=1):
    """Compare different MLA implementations with detailed debugging"""
    print(f"\nDebugging with batch_size={batch_size}, seq_len={seq_len}")
    
    # Setup environment
    torch.set_default_device('cuda')
    set_seed(42)
    
    # Create embedding layer
    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).cuda()
    
    # Generate test data
    x = torch.randint(0, args.vocab_size, (batch_size, seq_len), device='cuda')
    x_emb = embedding_layer(x)
    freqs_cis = precompute_freqs_cis(args)[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf"), device='cuda').triu_(1)
    start_pos = 0
    
    # Initialize test configurations
    test_configs = {
        'naive': Args(attn_impl="naive", dtype=args.dtype, max_seq_len=seq_len * 4),
        'naive+flash': Args(attn_impl="naive+flash", dtype=args.dtype, max_seq_len=seq_len * 4),
    }
    
    # Initialize models with the same weights
    set_seed(42)
    models = {}
    base_model = MLA(test_configs['naive']).cuda().eval()
    
    # Print the model configuration
    print("Model Configuration:")
    print(f"  dim = {args.dim}")
    print(f"  n_heads = {args.n_heads}")
    print(f"  qk_nope_head_dim = {args.qk_nope_head_dim}")
    print(f"  qk_rope_head_dim = {args.qk_rope_head_dim}")
    print(f"  v_head_dim = {args.v_head_dim}")
    print(f"  dtype = {args.dtype}")
    
    for impl in test_configs:
        model = MLA(test_configs[impl]).cuda().eval()
        model.load_state_dict(base_model.state_dict())
        models[impl] = model

    def hook_fn(name):
        """Create a hook function that captures tensor values"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                print(f"{name} output shape: {[o.shape for o in output]}")
                for i, o in enumerate(output):
                    print(f"{name}[{i}] output stats: min={o.min().item():.4f}, max={o.max().item():.4f}, mean={o.mean().item():.4f}")
            else:
                print(f"{name} output shape: {output.shape}")
                print(f"{name} output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
        return hook

    # Add hooks to capture intermediate values
    hooks = []
    for impl_name, model in models.items():
        # Add hooks to key components
        hooks.append(model.wq.register_forward_hook(hook_fn(f"{impl_name}.wq")))
        hooks.append(model.wkv_a.register_forward_hook(hook_fn(f"{impl_name}.wkv_a")))
        hooks.append(model.wkv_b.register_forward_hook(hook_fn(f"{impl_name}.wkv_b")))
        hooks.append(model.wo.register_forward_hook(hook_fn(f"{impl_name}.wo")))
    
    # Run forward pass for each implementation
    outputs = {}
    for impl_name, model in models.items():
        print(f"\nRunning {impl_name.upper()} implementation...")
        with torch.no_grad():
            outputs[impl_name] = model(x_emb, start_pos, freqs_cis, mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compare outputs
    print("\nOutput Comparison:")
    for impl_name, output in outputs.items():
        print(f"{impl_name.upper()} output shape: {output.shape}")
        print(f"{impl_name.upper()} output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
    
    naive_output = outputs['naive']
    flash_output = outputs['naive+flash']
    
    # Detailed comparison
    diff = naive_output - flash_output
    abs_diff = torch.abs(diff)
    
    print("\nDifference Analysis:")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"% elements with diff > 0.005: {(abs_diff > 0.005).float().mean().item() * 100:.2f}%")
    
    # Find indices of largest differences
    flat_diff = abs_diff.flatten()
    largest_indices = torch.topk(flat_diff, 5).indices
    
    print("\nLargest differences at indices:")
    for idx in largest_indices:
        # Convert flat index back to multi-dim
        multi_idx = np.unravel_index(idx.item(), naive_output.shape)
        print(f"  Index {multi_idx}: naive={naive_output[multi_idx].item():.6f}, flash={flash_output[multi_idx].item():.6f}, diff={abs_diff[multi_idx].item():.6f}")
    
    return outputs



if __name__ == "__main__":
    # Use a small test case
    model_args = Args(dtype="bf16")
    outputs = debug_implementations(model_args, seq_len=128, batch_size=1)