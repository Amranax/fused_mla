#!/usr/bin/env python3
"""
Benchmark script for comparing Multi-Latent Attention implementations
"""

import torch
import matplotlib.pyplot as plt
import time
import argparse
import math
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


def memory_benchmark(model, input_tensor, start_pos, freqs_cis, mask):
    """Measure GPU memory used by the model"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if hasattr(model, 'k_cache'):
        model.k_cache.zero_()
        model.v_cache.zero_()
    if hasattr(model, 'kv_cache'):
        model.kv_cache.zero_()
        model.pe_cache.zero_()

    with torch.no_grad():
        val = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return max_memory, val



def latency_benchmark(model, input_tensor, start_pos, freqs_cis, mask, warmup=10, repeats=50):
    """Measure model latency"""

    if hasattr(model, 'k_cache'):
        model.k_cache.zero_()
        model.v_cache.zero_()
    if hasattr(model, 'kv_cache'):
        model.kv_cache.zero_()
        model.pe_cache.zero_()

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(repeats):
        with torch.no_grad():
            _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)

    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / repeats



def incremental_token_benchmark(models, embedding_layer, args, batch_size=1, 
                               base_seq_len=64, max_new_tokens=128, 
                               warmup=3, repeats=10):
    """
    Benchmark token-by-token generation, simulating text generation.
    
    Args:
        models: Dictionary mapping implementation names to model objects
        embedding_layer: Embedding layer to convert token IDs to embeddings
        args: Model configuration args
        batch_size: Batch size for generation
        base_seq_len: Initial sequence length (prompt)
        max_new_tokens: Number of tokens to generate
        warmup: Number of warmup runs
        repeats: Number of benchmark repeats
    
    Returns:
        Dictionary with latency results
    """
    results = {name: [] for name in models.keys()}
    
    # Create initial prompt
    set_seed(42)
    prompt = torch.randint(0, args.vocab_size, (batch_size, base_seq_len), device='cuda')
    prompt_emb = embedding_layer(prompt)
    
    # Precompute freqs_cis for the maximum possible sequence length
    total_seq_len = base_seq_len + max_new_tokens
    freqs_cis = precompute_freqs_cis(args)
    
    print(f"\nRunning incremental token benchmark:")
    print(f"  Base sequence length: {base_seq_len}")
    print(f"  Generating {max_new_tokens} new tokens")
    
    # Measure generation time for each implementation
    for impl_name, model in models.items():
        print(f"\nBenchmarking {impl_name.upper()} implementation...")
        
        # Reset cache state for fair comparison
        if hasattr(model, 'k_cache'):
            model.k_cache.zero_()
            model.v_cache.zero_()
        if hasattr(model, 'kv_cache'):
            model.kv_cache.zero_()
            model.pe_cache.zero_()
            
        # Process the initial prompt
        x_emb = prompt_emb.clone()
        curr_seq_len = base_seq_len
        
        # First process the full prompt with proper masking
        full_mask = torch.full((curr_seq_len, curr_seq_len), float("-inf"), device='cuda').triu_(1)
        for _ in range(warmup):
            _ = model(x_emb, start_pos=0, freqs_cis=freqs_cis[:curr_seq_len], mask=full_mask)
        
        # Now generate tokens one by one and measure time
        for i in range(max_new_tokens):
            curr_seq_len = base_seq_len + i
            next_pos = curr_seq_len
            
            # Create dummy "next token" embedding
            next_token_emb = torch.rand((batch_size, 1, args.dim), 
                                        device='cuda', dtype=prompt_emb.dtype)
            
            # Time the forward pass for the single new token
            torch.cuda.synchronize()
            token_times = []
            
            for _ in range(repeats):
                start_time = time.time()
                # Only process the new token - for incremental generation, masking is handled in MLA
                with torch.no_grad():  # Prevent gradient tracking during benchmark
                    _ = model(next_token_emb, 
                            start_pos=next_pos, 
                            freqs_cis=freqs_cis[next_pos:next_pos+1], 
                            mask=None)  # Pass None for mask in incremental generation
                torch.cuda.synchronize()
                token_times.append(time.time() - start_time)
            
            # Record the median time
            token_time = np.median(token_times) * 1000  # Convert to ms
            results[impl_name].append(token_time)
            
            # Every 16 tokens, print progress
            if (i + 1) % 16 == 0 or i == 0 or i == max_new_tokens - 1:
                print(f"  Token {i+1}/{max_new_tokens}: {token_time:.3f} ms")
    
    return results


def compare_tensors_with_stats(tensor_a, tensor_b, name_a, name_b, tolerance=1e-2):
    """
    Compare tensors with detailed statistics for debugging
    
    Args:
        tensor_a, tensor_b: Tensors to compare
        name_a, name_b: Names for the tensors in output
        tolerance: Absolute tolerance for comparison
        
    Returns:
        is_close: Boolean indicating if tensors are close within tolerance
    """
    if tensor_a.shape != tensor_b.shape:
        print(f"Shape mismatch: {name_a}:{tensor_a.shape} vs {name_b}:{tensor_b.shape}")
        return False
    
    # Basic stats
    a_min, a_max, a_mean = tensor_a.min().item(), tensor_a.max().item(), tensor_a.mean().item()
    b_min, b_max, b_mean = tensor_b.min().item(), tensor_b.max().item(), tensor_b.mean().item()   
    
    # Compute differences
    abs_diff = torch.abs(tensor_a - tensor_b)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    num_diff = (abs_diff > tolerance).sum().item()
    pct_diff = 100 * num_diff / tensor_a.numel()
    
    
    # Find worst differences for debugging
    if num_diff > 0:
        print(f"{name_a} stats: min={a_min:.6f}, max={a_max:.6f}, mean={a_mean:.6f}")
        print(f"{name_b} stats: min={b_min:.6f}, max={b_max:.6f}, mean={b_mean:.6f}")
        print(f"Difference stats: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        print(f"Elements above tolerance: {num_diff}/{tensor_a.numel()} ({pct_diff:.4f}%)")
        # Get indices of top 5 differences or all if less than 5
        num_to_show = min(5, num_diff)
        top_diffs, top_indices = torch.topk(abs_diff.flatten(), num_to_show)
        
        print("Top differences:")
        for i, (diff, idx) in enumerate(zip(top_diffs, top_indices)):
            # Convert flat index to multi-dimensional index
            multi_idx = np.unravel_index(idx.item(), tensor_a.shape)
            a_val = tensor_a[multi_idx].item()
            b_val = tensor_b[multi_idx].item()
            print(f"  #{i+1}: idx={multi_idx}, {name_a}={a_val:.6f}, {name_b}={b_val:.6f}, diff={diff.item():.6f}")
    
    return num_diff == 0


def run_benchmarks(args, seq_lengths, batch_sizes, val_tolerance=1.5e-2):
    """Run comprehensive benchmarks across different sequence lengths and batch sizes"""
    results = {
        'naive': {'latency': {}, 'memory': {}},
        'absorb': {'latency': {}, 'memory': {}},
        'naive+flash': {'latency': {}, 'memory': {}},
    }

    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).cuda()

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
            set_seed(42)  # ensure same inputs and weights

            x = torch.randint(0, args.vocab_size, (batch_size, seq_len), device='cuda')
            x_emb = embedding_layer(x)
            freqs_cis = precompute_freqs_cis(args)[:seq_len]
            mask = torch.full((seq_len, seq_len), float("-inf"), device='cuda').triu_(1)
            start_pos = 0

            # Initialize test configurations
            test_configs = {
                'naive': Args(attn_impl="naive", dtype=args.dtype, max_seq_len=seq_len + 128),
                'absorb': Args(attn_impl="absorb", dtype=args.dtype, max_seq_len=seq_len + 128),
                'naive+flash': Args(attn_impl="naive+flash", dtype=args.dtype, max_seq_len=seq_len + 128),
            }

            # Initialize all models with the same weights
            ret_vals = {}
            for impl in test_configs:
                # Initialize model with fresh random seed
                set_seed(42)
                
                # Create model for memory test
                model = MLA(test_configs[impl]).cuda().eval()
                memory, val = memory_benchmark(model, x_emb, start_pos, freqs_cis, mask)
                
                # Reset
                del model
                torch.cuda.empty_cache()

                # Create model for latency test
                model = MLA(test_configs[impl]).cuda().eval()
                latency = latency_benchmark(model, x_emb, start_pos, freqs_cis, mask)
                # Reset
                del model
                torch.cuda.empty_cache()
                
                # Store results
                ret_vals[impl] = val
                results[impl]['latency'].setdefault(batch_size, {})[seq_len] = latency
                results[impl]['memory'].setdefault(batch_size, {})[seq_len] = memory
                
                print(f"  {impl.upper()}: Latency = {latency*1000:.3f} ms, Memory = {memory:.2f} GB")
                

            # Compare naive vs absorb
            naive_vs_absorb = compare_tensors_with_stats(
                ret_vals['naive'], ret_vals['absorb'], 'naive', 'absorb', val_tolerance
            )
            
            # Compare naive vs naive+flash
            naive_vs_flash = compare_tensors_with_stats(
                ret_vals['naive'], ret_vals['naive+flash'], 'naive', 'naive+flash', val_tolerance
            )
            
            if not naive_vs_absorb:
                print("\nWARNING: Significant differences between NAIVE and ABSORB implementations")
            
            if not naive_vs_flash:
                print("\nWARNING: Significant differences between NAIVE and NAIVE+FLASH implementations")

    return results


def run_incremental_benchmark(args):
    """Run the incremental token generation benchmark"""
    # Create embedding layer
    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).cuda()
    
    # Define maximum sequence length for this benchmark
    base_seq_len = 1024
    max_new_tokens = 1024
    max_seq_len = base_seq_len + max_new_tokens
    
    # Initialize test configurations
    test_configs = {
        'naive': Args(attn_impl="naive", dtype=args.dtype, max_seq_len=max_seq_len * 2),
        'absorb': Args(attn_impl="absorb", dtype=args.dtype, max_seq_len=max_seq_len * 2),
        'naive+flash': Args(attn_impl="naive+flash", dtype=args.dtype, max_seq_len=max_seq_len * 2),
    }
    
    # Initialize all models with the same weights
    models = {}
    set_seed(42)
    base_model = MLA(test_configs['naive']).cuda().eval()
    for impl in test_configs:
        model = MLA(test_configs[impl]).cuda().eval()
        model.load_state_dict(base_model.state_dict())
        models[impl] = model
    
    # Run incremental benchmark
    results = incremental_token_benchmark(
        models, 
        embedding_layer, 
        args, 
        batch_size=1,
        base_seq_len=base_seq_len, 
        max_new_tokens=max_new_tokens
    )
    
    return results


def plot_results(results, batch_sizes, seq_lengths, output_prefix="benchmark"):
    plt.figure(figsize=(20, 10))

    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(2, len(batch_sizes), i + 1)
        for name,val in results.items():
            latency = [results[name]['latency'][batch_size][seq_len] * 1000 for seq_len in seq_lengths]
            plt.plot(seq_lengths, latency, 'o-', label=name)
        plt.title(f'Latency (Batch Size = {batch_size})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid()

    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(2, len(batch_sizes), len(batch_sizes) + i + 1)
        for name,val in results.items():
            memory = [results[name]['memory'][batch_size][seq_len] for seq_len in seq_lengths]
            plt.plot(seq_lengths, memory, 'o-', label=name)
        plt.title(f'Memory Usage (Batch Size = {batch_size})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_results.png')
    print(f"Results saved to {output_prefix}_results.png")


def plot_incremental_results(results, output_prefix="benchmark"):
    """Plot the results from the incremental token benchmark"""
    plt.figure(figsize=(12, 6))
    
    # Plot incremental token generation time
    token_indices = np.arange(1, len(next(iter(results.values()))) + 1)
    
    for name, latencies in results.items():
        plt.plot(token_indices, latencies, 'o-', label=name)
    
    plt.title('Incremental Token Generation Latency')
    plt.xlabel('Token Position')
    plt.ylabel('Time per Token (ms)')
    plt.legend()
    plt.grid()
    
    # Add a moving average line for each implementation
    window_size = min(15, len(token_indices) // 4)
    if window_size > 1:
        for name, latencies in results.items():
            ma = np.convolve(latencies, np.ones(window_size)/window_size, mode='valid')
            ma_indices = token_indices[window_size-1:]
            plt.plot(ma_indices, ma, '--', linewidth=1.5, 
                     color=plt.gca().lines[-len(results.items())].get_color(),
                     alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_incremental_results.png')
    print(f"Incremental benchmark results saved to {output_prefix}_incremental_results.png")


def save_results_to_file(results, batch_sizes, seq_lengths, output_prefix="benchmark"):
    with open(f'{output_prefix}_results.csv', 'w') as f:
        f.write('implementation,batch_size,seq_length,latency_ms,memory_gb\n')
        for impl in ['naive', 'absorb', 'naive+flash']:
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    latency = results[impl]['latency'][batch_size][seq_len] * 1000
                    memory = results[impl]['memory'][batch_size][seq_len]
                    f.write(f'{impl},{batch_size},{seq_len},{latency:.3f},{memory:.3f}\n')
    print(f"Results saved to {output_prefix}_results.csv")


def save_incremental_results_to_file(results, output_prefix="benchmark"):
    """Save incremental token benchmark results to a CSV file"""
    with open(f'{output_prefix}_incremental_results.csv', 'w') as f:
        # Write header with implementation names
        f.write('token_position,' + ','.join(results.keys()) + '\n')
        
        # Write data rows
        max_tokens = len(next(iter(results.values())))
        for i in range(max_tokens):
            row = [str(i + 1)]  # Token position (1-indexed)
            for impl in results:
                row.append(f"{results[impl][i]:.3f}")
            f.write(','.join(row) + '\n')
    
    print(f"Incremental benchmark results saved to {output_prefix}_incremental_results.csv")


def main():
    parser = argparse.ArgumentParser(description='Benchmark MLA implementations')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp8'], help='Data type for benchmark')
    parser.add_argument('--min-seq-len', type=int, default=128, help='Minimum sequence length to benchmark')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length to benchmark')
    parser.add_argument('--seq-len-step', type=int, default=256, help='Step size for sequence length')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4], help='Batch sizes to benchmark')
    parser.add_argument('--output', type=str, default='', help='Output file prefix')
    parser.add_argument('--incremental-only', action='store_true', help='Run only the incremental token benchmark')
    parser.add_argument('--standard-only', action='store_true', help='Run only the standard benchmark')
    parser.add_argument('--tolerance', type=float, default=1.5e-2, help='Tolerance for tensor comparison')
    args = parser.parse_args()

    torch.set_default_device('cuda')
    if args.dtype == 'bf16':
        torch.set_default_dtype(torch.bfloat16)

    set_seed(42)
    seq_lengths = list(range(args.min_seq_len, args.max_seq_len + 1, args.seq_len_step))
    model_args = Args(dtype=args.dtype)

    os.makedirs('benchmark_results', exist_ok=True)

    if args.output == '':
        base_output_name = f"benchmark_results/mla_benchmark_{model_args.qk_nope_head_dim+model_args.qk_rope_head_dim}qkdim_{model_args.v_head_dim}vdim_{args.batch_sizes}"
    else:
        base_output_name = f'benchmark_results/{args.output}'

    # Run standard benchmarks (batch/sequence length comparisons)
    if not args.incremental_only:
        print("\n==== Running Standard Benchmarks ====")
        results = run_benchmarks(model_args, seq_lengths, args.batch_sizes, args.tolerance)
        plot_results(results, args.batch_sizes, seq_lengths, base_output_name)
        save_results_to_file(results, args.batch_sizes, seq_lengths, base_output_name)

    # Run incremental token benchmark
    if not args.standard_only:
        print("\n==== Running Incremental Token Benchmark ====")
        incremental_results = run_incremental_benchmark(model_args)
        plot_incremental_results(incremental_results, base_output_name)
        save_incremental_results_to_file(incremental_results, base_output_name)


if __name__ == "__main__":
    main()