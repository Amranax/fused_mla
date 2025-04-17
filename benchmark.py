#!/usr/bin/env python3
"""
Benchmark script for comparing Multi-Latent Attention implementations
"""

import torch
import matplotlib.pyplot as plt
import time
import argparse
import math

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
                val = val.mean().item() if isinstance(val, torch.Tensor) and val.numel() > 1 else val.item() if isinstance(val, torch.Tensor) else val
                ret_vals[impl_name] = val

                latency = latency_benchmark(model, x_emb, start_pos, freqs_cis, mask)

                results[impl_name]['latency'].setdefault(batch_size, {})[seq_len] = latency
                results[impl_name]['memory'].setdefault(batch_size, {})[seq_len] = memory

                print(f"  {impl_name.upper()}: Latency = {latency*1000:.3f} ms, Memory = {memory:.2f} GB")

            torch.testing.assert_close(torch.tensor(ret_vals['naive']), torch.tensor(ret_vals['naive+flash']), rtol=0, atol=5e-3)
            torch.testing.assert_close(torch.tensor(ret_vals['naive']), torch.tensor(ret_vals['absorb']), rtol=0, atol=5e-3)

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


def main():
    parser = argparse.ArgumentParser(description='Benchmark MLA implementations')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp8'], help='Data type for benchmark')
    parser.add_argument('--min-seq-len', type=int, default=128, help='Minimum sequence length to benchmark')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length to benchmark')
    parser.add_argument('--seq-len-step', type=int, default=256, help='Step size for sequence length')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4], help='Batch sizes to benchmark')
    parser.add_argument('--output', type=str, default='', help='Output file prefix')
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
        output_name = f"benchmark_results/mla_benchmark_{model_args.qk_nope_head_dim+model_args.qk_rope_head_dim}qkdim_{model_args.v_head_dim}vdim_{args.batch_sizes}"
    else:
        output_name = 'benchmark_results/mla_benchmark'

    results = run_benchmarks(model_args, seq_lengths, args.batch_sizes)

    plot_results(results, args.batch_sizes, seq_lengths, output_name)
    save_results_to_file(results, args.batch_sizes, seq_lengths, output_name)


if __name__ == "__main__":
    main()
