#!/usr/bin/env python3
"""
Benchmark script for comparing Multi-Latent Attention implementations
"""

import torch
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from attention_impl.naive_mla import MLA
from attention_impl.Shared_Args import Args
from attention_impl.General_Layers import precompute_freqs_cis, EmbeddingLayer

def memory_benchmark(model, input_tensor, start_pos, freqs_cis, mask):
    """Measure GPU memory used by the model"""
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run forward pass
    _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    torch.cuda.synchronize()
    
    # Get memory stats
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return max_memory

def latency_benchmark(model, input_tensor, start_pos, freqs_cis, mask, warmup=10, repeats=50):
    """Measure model latency"""
    # Warmup
    for _ in range(warmup):
        _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        _ = model(input_tensor, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / repeats
    return avg_time

def run_benchmarks(args, seq_lengths, batch_sizes):
    """Run comprehensive benchmarks across different sequence lengths and batch sizes"""
    results = {
        'naive': {'latency': {}, 'memory': {}},
        'absorb': {'latency': {}, 'memory': {}}
    }
    
    # Initialize embedding layer
    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).cuda()
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
            
            # Create input tensors
            x = torch.randint(0, args.vocab_size, (batch_size, seq_len), device='cuda')
            x_emb = embedding_layer(x)
            
            # Create freqs_cis and mask
            freqs_cis = precompute_freqs_cis(args)
            freqs_cis = freqs_cis[:seq_len]
            mask = torch.full((seq_len, seq_len), float("-inf"), device='cuda').triu_(1)
            
            # Set start position
            start_pos = 0
            
            # Test configurations
            test_configs = {
                'naive': Args(attn_impl="naive", dtype=args.dtype, max_seq_len=seq_len*4),
                'absorb': Args(attn_impl="absorb", dtype=args.dtype, max_seq_len=seq_len*4)
            }
            
            for impl_name, impl_args in test_configs.items():
                # Create model
                model = MLA(impl_args).cuda().eval()
                
                # Benchmark memory
                memory = memory_benchmark(model, x_emb, start_pos, freqs_cis, mask)
                
                # Benchmark latency
                latency = latency_benchmark(model, x_emb, start_pos, freqs_cis, mask)
                
                # Store results
                if batch_size not in results[impl_name]['latency']:
                    results[impl_name]['latency'][batch_size] = {}
                    results[impl_name]['memory'][batch_size] = {}
                
                results[impl_name]['latency'][batch_size][seq_len] = latency
                results[impl_name]['memory'][batch_size][seq_len] = memory
                
                print(f"  {impl_name.upper()}: Latency = {latency*1000:.3f} ms, Memory = {memory:.2f} GB")
    
    return results

def plot_results(results, batch_sizes, seq_lengths, output_prefix="benchmark"):
    """Plot benchmark results"""
    # Create figure with subplots
    plt.figure(figsize=(20, 10))
    
    # Plot latency
    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(2, len(batch_sizes), i + 1)
        
        naive_latency = [results['naive']['latency'][batch_size][seq_len] * 1000 for seq_len in seq_lengths]
        absorb_latency = [results['absorb']['latency'][batch_size][seq_len] * 1000 for seq_len in seq_lengths]
        
        plt.plot(seq_lengths, naive_latency, 'o-', label='Naive')
        plt.plot(seq_lengths, absorb_latency, 'o-', label='Absorb')
        plt.title(f'Latency (Batch Size = {batch_size})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid()
    
    # Plot memory
    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(2, len(batch_sizes), len(batch_sizes) + i + 1)
        
        naive_memory = [results['naive']['memory'][batch_size][seq_len] for seq_len in seq_lengths]
        absorb_memory = [results['absorb']['memory'][batch_size][seq_len] for seq_len in seq_lengths]
        
        plt.plot(seq_lengths, naive_memory, 'o-', label='Naive')
        plt.plot(seq_lengths, absorb_memory, 'o-', label='Absorb')
        plt.title(f'Memory Usage (Batch Size = {batch_size})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_results.png')
    print(f"Results saved to {output_prefix}_results.png")

def save_results_to_file(results, batch_sizes, seq_lengths, output_prefix="benchmark"):
    """Save benchmark results to a CSV file"""
    with open(f'{output_prefix}_results.csv', 'w') as f:
        f.write('implementation,batch_size,seq_length,latency_ms,memory_gb\n')
        
        for impl in ['naive', 'absorb']:
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    latency = results[impl]['latency'][batch_size][seq_len] * 1000  # Convert to ms
                    memory = results[impl]['memory'][batch_size][seq_len]
                    f.write(f'{impl},{batch_size},{seq_len},{latency:.3f},{memory:.3f}\n')
    
    print(f"Results saved to {output_prefix}_results.csv")

def main():
    parser = argparse.ArgumentParser(description='Benchmark MLA implementations')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp8'], 
                        help='Data type for benchmark')
    parser.add_argument('--min-seq-len', type=int, default=128, 
                        help='Minimum sequence length to benchmark')
    parser.add_argument('--max-seq-len', type=int, default=4096, 
                        help='Maximum sequence length to benchmark')
    parser.add_argument('--seq-len-step', type=int, default=256, 
                        help='Step size for sequence length')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4], 
                        help='Batch sizes to benchmark')
    parser.add_argument('--output', type=str, default='mla_benchmark', 
                        help='Output file prefix')
    args = parser.parse_args()
    
    # Set CUDA device
    torch.set_default_device('cuda')
    if args.dtype == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Sequence lengths to benchmark
    seq_lengths = list(range(args.min_seq_len, args.max_seq_len + 1, args.seq_len_step))
    
    # Default args for models
    model_args = Args(dtype=args.dtype)
    
    # Run benchmarks
    results = run_benchmarks(model_args, seq_lengths, args.batch_sizes)
    
    # Plot and save results
    plot_results(results, args.batch_sizes, seq_lengths, args.output)
    save_results_to_file(results, args.batch_sizes, seq_lengths, args.output)

if __name__ == "__main__":
    main()