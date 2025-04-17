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

    val = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return max_memory, val


def latency_benchmark(model, input_tensor, start_pos, freqs_cis, mask, warmup=10, repeats=50):
    """Measure model latency"""
    for _ in range(warmup):
        _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(repeats):
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
        
        # Warmup with full prompt
        mask = torch.full((curr_seq_len, curr_seq_len), float("-inf"), device='cuda').triu_(1)
        for _ in range(warmup):
            _ = model(x_emb, start_pos=0, freqs_cis=freqs_cis[:curr_seq_len], mask=mask)
        
        # Now generate tokens one by one and measure time
        for i in range(max_new_tokens):
            curr_seq_len = base_seq_len + i
            
            # Create dummy "next token" embedding
            next_token_emb = torch.rand((batch_size, 1, args.dim), 
                                        device='cuda', dtype=prompt_emb.dtype)
            
            # Create causal mask for the current sequence length
            mask = torch.full((curr_seq_len + 1, curr_seq_len + 1), 
                              float("-inf"), device='cuda').triu_(1)
            
            # Time the forward pass for the single new token
            torch.cuda.synchronize()
            token_times = []
            
            for _ in range(repeats):
                start_time = time.time()
                # Only process the new token, starting from position curr_seq_len
                with torch.no_grad():  # Prevent gradient tracking during benchmark
                    _ = model(next_token_emb, 
                            start_pos=curr_seq_len, 
                            freqs_cis=freqs_cis[curr_seq_len:curr_seq_len+1], 
                            mask=mask)
                torch.cuda.synchronize()
                token_times.append(time.time() - start_time)
            
            # Record the median time
            token_time = np.median(token_times) * 1000  # Convert to ms
            results[impl_name].append(token_time)
            
            # Every 16 tokens, print progress
            if (i + 1) % 16 == 0 or i == 0 or i == max_new_tokens - 1:
                print(f"  Token {i+1}/{max_new_tokens}: {token_time:.3f} ms")
    
    return results


def run_benchmarks(args, seq_lengths, batch_sizes):
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
                'naive': Args(attn_impl="naive", dtype=args.dtype, max_seq_len=seq_len * 4),
                'absorb': Args(attn_impl="absorb", dtype=args.dtype, max_seq_len=seq_len * 4),
                'naive+flash': Args(attn_impl="naive+flash", dtype=args.dtype, max_seq_len=seq_len * 4),
            }

            # Initialize all models with the same weights
            models = {}
            set_seed(42)
            base_model = MLA(test_configs['naive']).cuda().eval()
            for impl in test_configs:
                model = MLA(test_configs[impl]).cuda().eval()
                model.load_state_dict(base_model.state_dict())
                models[impl] = model

            ret_vals = {}
            for impl_name, model in models.items():
                memory, val = memory_benchmark(model, x_emb, start_pos, freqs_cis, mask)
                ret_vals[impl_name] = val

                latency = latency_benchmark(model, x_emb, start_pos, freqs_cis, mask)

                results[impl_name]['latency'].setdefault(batch_size, {})[seq_len] = latency
                results[impl_name]['memory'].setdefault(batch_size, {})[seq_len] = memory

                print(f"  {impl_name.upper()}: Latency = {latency*1000:.3f} ms, Memory = {memory:.2f} GB")

            # Convert to tensors with no gradients for comparison
            naive_val = torch.tensor(ret_vals['naive'], requires_grad=False)
            flash_val = torch.tensor(ret_vals['naive+flash'], requires_grad=False)
            absorb_val = torch.tensor(ret_vals['absorb'], requires_grad=False)
            
            torch.testing.assert_close(naive_val, absorb_val, rtol=0, atol=5e-3)
            torch.testing.assert_close(naive_val, flash_val, rtol=0, atol=5e-3)

    return results


def run_incremental_benchmark(args):
    """Run the incremental token generation benchmark"""
    # Create embedding layer
    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).cuda()
    
    # Define maximum sequence length for this benchmark
    base_seq_len = 64
    max_new_tokens = 128
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
    args = parser.parse_args()

    torch.set_default_device('cuda')
    if args.dtype == 'bf16':
        torch.set_default_dtype(torch.bfloat16)

    set_seed(42)
    seq_lengths = list(range(args.min_seq_len, args.max_seq_len + 1, args.seq_len_step))
    model_args = Args(dtype=args.dtype)

    import os
    os.makedirs('benchmark_results', exist_ok=True)

    if args.output == '':
        base_output_name = f"benchmark_results/mla_benchmark_{model_args.qk_nope_head_dim+model_args.qk_rope_head_dim}qkdim_{model_args.v_head_dim}vdim_{args.batch_sizes}"
    else:
        base_output_name = 'benchmark_results/mla_benchmark'

    # Run standard benchmarks (batch/sequence length comparisons)
    if not args.incremental_only:
        print("\n==== Running Standard Benchmarks ====")
        results = run_benchmarks(model_args, seq_lengths, args.batch_sizes)
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