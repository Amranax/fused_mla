#!/usr/bin/env python3
"""
MLA (Multi-Latent Attention) Benchmark

This script benchmarks MLA implementations with various configurations,
measuring performance metrics like latency, throughput, and memory usage.
"""

import torch
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Local imports
from attention_impl.Shared_Args import Args
from attention_impl.General_Layers import precompute_freqs_cis, EmbeddingLayer
from attention_impl.mla_factory import get_mla


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def benchmark_mla(
    args: Args,
    batch_size: int,
    seq_len: int,
    n_warmup: int = 5,
    n_trials: int = 20,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark MLA with specified configuration"""
    
    model = get_mla(args.attn_impl, args).to(device).eval()
    embedding_layer = EmbeddingLayer(args.vocab_size, args.dim).to(device)

    x = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    x_emb = embedding_layer(x)

    freqs_cis = precompute_freqs_cis(args)[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)
    start_pos = 0

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x_emb, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model(x_emb, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = torch.tensor(times).std().item()
    tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)

    output_mean = out.mean().item()
    output_std = out.std().item()

    return {
        "attn_impl": args.attn_impl,
        "dtype": args.dtype,
        "n_heads": args.n_heads,
        "dim": args.dim,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "v_head_dim": args.v_head_dim,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "latency_ms": avg_time,
        "latency_min_ms": min_time,
        "latency_max_ms": max_time,
        "latency_std_ms": std_time,
        "throughput_tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_mem,
        "output_mean": output_mean,
        "output_std": output_std,
    }


def sweep_mla_configs(
    dtype: str = "bf16",
    attn_impls: List[str] = ["naive", "absorb", "naive+flash"],
    batch_sizes: List[int] = [1, 4, 8],
    seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
    n_heads_list: List[int] = [8, 16],
    head_dims: List[Tuple[int, int, int]] = [(64, 0, 64), (64, 64, 128)],
    n_warmup: int = 5,
    n_trials: int = 20,
    device: str = "cuda",
    output_dir: str = "mla_benchmark_results"
) -> pd.DataFrame:
    """Run benchmarks across different MLA configurations"""

    os.makedirs(output_dir, exist_ok=True)
    results = []
    total_configs = len(attn_impls) * len(batch_sizes) * len(seq_lengths) * len(n_heads_list) * len(head_dims)
    config_count = 0

    print(f"Running {total_configs} MLA configurations...")

    for attn_impl in attn_impls:
        for n_heads in n_heads_list:
            for qk_rope_dim, qk_nope_dim, v_dim in head_dims:
                for batch_size in batch_sizes:
                    for seq_len in seq_lengths:
                        config_count += 1
                        print(f"[{config_count}/{total_configs}] Testing config: attn_impl={attn_impl}, "
                              f"n_heads={n_heads}, head_dims=({qk_rope_dim},{qk_nope_dim},{v_dim}), "
                              f"batch_size={batch_size}, seq_len={seq_len}")

                        head_dim = qk_rope_dim + qk_nope_dim
                        dim = n_heads * head_dim

                        args = Args(
                            attn_impl=attn_impl,
                            dtype=dtype,
                            n_heads=n_heads,
                            dim=dim,
                            qk_rope_head_dim=qk_rope_dim,
                            qk_nope_head_dim=qk_nope_dim,
                            v_head_dim=v_dim,
                            max_seq_len=seq_len * 2
                        )

                        try:
                            result = benchmark_mla(
                                args=args,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                n_warmup=n_warmup,
                                n_trials=n_trials,
                                device=device
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"Error benchmarking configuration: {e}")
                            results.append({
                                "attn_impl": attn_impl,
                                "dtype": dtype,
                                "n_heads": n_heads,
                                "dim": dim,
                                "qk_rope_head_dim": qk_rope_dim,
                                "qk_nope_head_dim": qk_nope_dim,
                                "v_head_dim": v_dim,
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "error": str(e)
                            })

                        df = pd.DataFrame(results)
                        df.to_csv(f"{output_dir}/mla_benchmark_partial.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/mla_benchmark_results.csv", index=False)
    print(f"Benchmark complete. Results saved to {output_dir}/mla_benchmark_results.csv")
    return df

def plot_latency_comparison(df, output_dir):
    """Plot latency comparison across different implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by head dimensions
    head_dim_configs = df.drop_duplicates(
        subset=['qk_rope_head_dim', 'qk_nope_head_dim', 'v_head_dim']
    )[['qk_rope_head_dim', 'qk_nope_head_dim', 'v_head_dim']].values.tolist()
    
    for qk_rope_dim, qk_nope_dim, v_dim in head_dim_configs:
        # Filter dataframe for this head config
        head_df = df[
            (df['qk_rope_head_dim'] == qk_rope_dim) & 
            (df['qk_nope_head_dim'] == qk_nope_dim) & 
            (df['v_head_dim'] == v_dim)
        ]
        
        # Plot for each n_heads
        for n_heads in head_df['n_heads'].unique():
            # Filter for this n_heads
            heads_df = head_df[head_df['n_heads'] == n_heads]
            
            # Plot for each batch size
            for batch_size in heads_df['batch_size'].unique():
                # Filter for this batch size
                batch_df = heads_df[heads_df['batch_size'] == batch_size]
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                
                for impl in batch_df['attn_impl'].unique():
                    # Get data for this implementation
                    impl_df = batch_df[batch_df['attn_impl'] == impl]
                    
                    # Sort by sequence length and plot
                    impl_df = impl_df.sort_values('seq_len')
                    plt.plot(
                        impl_df['seq_len'], 
                        impl_df['latency_ms'], 
                        marker='o', 
                        label=f"{impl}"
                    )
                
                plt.title(f"MLA Latency Comparison\nheads={n_heads}, dims=({qk_rope_dim},{qk_nope_dim},{v_dim}), batch={batch_size}")
                plt.xlabel("Sequence Length")
                plt.ylabel("Latency (ms)")
                plt.xscale('log', base=2)
                plt.yscale('log')
                plt.grid(True, which="both", ls="--", alpha=0.3)
                plt.legend()
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/latency_h{n_heads}_qkr{qk_rope_dim}_qkn{qk_nope_dim}_v{v_dim}_b{batch_size}.png"
                )
                plt.close()


def plot_throughput_comparison(df, output_dir):
    """Plot throughput comparison across different implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by sequence length
    for seq_len in sorted(df['seq_len'].unique()):
        # Filter for this sequence length
        seq_df = df[df['seq_len'] == seq_len]
        
        # Group by head dimensions
        head_dim_configs = seq_df.drop_duplicates(
            subset=['qk_rope_head_dim', 'qk_nope_head_dim', 'v_head_dim']
        )[['qk_rope_head_dim', 'qk_nope_head_dim', 'v_head_dim']].values.tolist()
        
        for qk_rope_dim, qk_nope_dim, v_dim in head_dim_configs:
            # Filter for this head config
            head_df = seq_df[
                (seq_df['qk_rope_head_dim'] == qk_rope_dim) & 
                (seq_df['qk_nope_head_dim'] == qk_nope_dim) & 
                (seq_df['v_head_dim'] == v_dim)
            ]
            
            # Plot for each n_heads
            for n_heads in head_df['n_heads'].unique():
                # Filter for this n_heads
                heads_df = head_df[head_df['n_heads'] == n_heads]
                
                # Create the plot
                plt.figure(figsize=(12, 6))
                
                # Set up data for grouped bar chart
                impls = sorted(heads_df['attn_impl'].unique())
                batch_sizes = sorted(heads_df['batch_size'].unique())
                
                # Set up the positions for grouped bars
                bar_width = 0.8 / len(impls)
                r = np.arange(len(batch_sizes))
                
                # Plot each implementation as a group
                for i, impl in enumerate(impls):
                    throughputs = []
                    
                    for batch_size in batch_sizes:
                        # Get data for this implementation and batch size
                        data = heads_df[(heads_df['attn_impl'] == impl) & 
                                        (heads_df['batch_size'] == batch_size)]
                        
                        if len(data) > 0:
                            throughputs.append(data['throughput_tokens_per_sec'].values[0] / 1000)  # K tokens/sec
                        else:
                            throughputs.append(0)
                    
                    # Plot bars
                    plt.bar(
                        r + i * bar_width - 0.4 + bar_width/2, 
                        throughputs, 
                        width=bar_width, 
                        label=impl
                    )
                
                # Customize plot
                plt.title(f"MLA Throughput Comparison\nheads={n_heads}, dims=({qk_rope_dim},{qk_nope_dim},{v_dim}), seq_len={seq_len}")
                plt.xlabel("Batch Size")
                plt.ylabel("Throughput (K tokens/sec)")
                plt.xticks(r, [str(b) for b in batch_sizes])
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/throughput_s{seq_len}_h{n_heads}_qkr{qk_rope_dim}_qkn{qk_nope_dim}_v{v_dim}.png"
                )
                plt.close()


def plot_memory_comparison(df, output_dir):
    """Plot memory usage comparison across different implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by sequence length
    for seq_len in sorted(df['seq_len'].unique()):
        # Filter for this sequence length
        seq_df = df[df['seq_len'] == seq_len]
        
        # Group by batch size
        for batch_size in sorted(seq_df['batch_size'].unique()):
            # Filter for this batch size
            batch_df = seq_df[seq_df['batch_size'] == batch_size]
            
            # Create plot
            plt.figure(figsize=(14, 8))
            
            # Prepare data for horizontal bar chart
            configs = []
            memory_usage = []
            labels = []
            
            # Group by implementation, n_heads, and head dimensions
            for impl in sorted(batch_df['attn_impl'].unique()):
                for n_heads in sorted(batch_df['n_heads'].unique()):
                    # Filter for this implementation and n_heads
                    impl_df = batch_df[(batch_df['attn_impl'] == impl) & 
                                       (batch_df['n_heads'] == n_heads)]
                    
                    # Group by head dimensions
                    for _, row in impl_df.iterrows():
                        if 'error' in row and not pd.isna(row['error']):
                            continue
                            
                        qk_rope_dim = row['qk_rope_head_dim']
                        qk_nope_dim = row['qk_nope_head_dim']
                        v_dim = row['v_head_dim']
                        
                        # Create label and add data
                        label = f"{impl}, h={n_heads}, dims=({qk_rope_dim},{qk_nope_dim},{v_dim})"
                        labels.append(label)
                        memory_usage.append(row['peak_memory_mb'])
                        
                        # Store config for coloring
                        configs.append(impl)
            
            # Sort by memory usage
            idx = np.argsort(memory_usage)
            sorted_labels = [labels[i] for i in idx]
            sorted_memory = [memory_usage[i] for i in idx]
            sorted_configs = [configs[i] for i in idx]
            
            # Create colors by implementation
            unique_impls = list(set(sorted_configs))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_impls)))
            color_map = {impl: colors[i] for i, impl in enumerate(unique_impls)}
            bar_colors = [color_map[impl] for impl in sorted_configs]
            
            # Plot horizontal bars
            bars = plt.barh(sorted_labels, sorted_memory, color=bar_colors)
            
            # Add memory values as text
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width + 10, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{width:.1f} MB", 
                    va='center'
                )
            
            # Customize plot
            plt.title(f"MLA Memory Usage Comparison (seq_len={seq_len}, batch_size={batch_size})")
            plt.xlabel("Peak Memory Usage (MB)")
            plt.grid(axis='x', alpha=0.3)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=color_map[impl], lw=4, label=impl)
                for impl in unique_impls
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/memory_s{seq_len}_b{batch_size}.png")
            plt.close()


def generate_summary_report(df, output_file):
    """Generate a summary report in Markdown format"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Multi-Latent Attention (MLA) Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares different MLA configurations across various dimensions:\n")
        f.write("- Different attention implementations (naive, absorb, naive+flash)\n")
        f.write("- Various head dimensions (rope, nope, value)\n")
        f.write("- Different numbers of heads\n")
        f.write("- Various batch sizes and sequence lengths\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("- Latency (ms)\n")
        f.write("- Throughput (tokens/second)\n")
        f.write("- Peak memory usage (MB)\n\n")
        
        # Filter out rows with errors
        if 'error' in df.columns:
            clean_df = df[df['error'].isna()]
        else:
            clean_df = df
        
        # Fastest implementation by sequence length
        f.write("## Fastest Implementation by Sequence Length\n\n")
        f.write("| Seq Length | Best Implementation | Head Config | Batch | Latency (ms) | Throughput (K tok/s) |\n")
        f.write("|------------|---------------------|-------------|-------|--------------|----------------------|\n")
        
        for seq_len in sorted(clean_df['seq_len'].unique()):
            seq_df = clean_df[clean_df['seq_len'] == seq_len]
            
            if len(seq_df) == 0:
                continue
                
            fastest = seq_df.loc[seq_df['latency_ms'].idxmin()]
            head_config = f"({fastest['qk_rope_head_dim']},{fastest['qk_nope_head_dim']},{fastest['v_head_dim']})"
            
            f.write(f"| {seq_len:,} | {fastest['attn_impl']} | {head_config} | {fastest['batch_size']} | ")
            f.write(f"{fastest['latency_ms']:.2f} | {fastest['throughput_tokens_per_sec']/1000:.2f} |\n")
        
        f.write("\n")
        
        # Most memory-efficient implementation
        f.write("## Most Memory-Efficient Implementation by Sequence Length\n\n")
        f.write("| Seq Length | Best Implementation | Head Config | Batch | Memory (MB) |\n")
        f.write("|------------|---------------------|-------------|-------|------------|\n")
        
        for seq_len in sorted(clean_df['seq_len'].unique()):
            seq_df = clean_df[clean_df['seq_len'] == seq_len]
            
            if len(seq_df) == 0:
                continue
                
            most_efficient = seq_df.loc[seq_df['peak_memory_mb'].idxmin()]
            head_config = f"({most_efficient['qk_rope_head_dim']},{most_efficient['qk_nope_head_dim']},{most_efficient['v_head_dim']})"
            
            f.write(f"| {seq_len:,} | {most_efficient['attn_impl']} | {head_config} | ")
            f.write(f"{most_efficient['batch_size']} | {most_efficient['peak_memory_mb']:.2f} |\n")
        
        f.write("\n")
        
        # Implementation comparison
        f.write("## Implementation Comparison\n\n")
        f.write("| Implementation | Avg Latency (ms) | Avg Memory (MB) | Best For |\n")
        f.write("|----------------|------------------|-----------------|----------|\n")
        
        impl_summary = clean_df.groupby('attn_impl').agg({
            'latency_ms': 'mean',
            'peak_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        }).reset_index()
        
        # Determine best use case for each implementation
        best_for = {
            'naive': "Baseline implementation",
            'absorb': "Memory efficiency",
            'naive+flash': "Low latency"
        }
        
        # Verify with data
        fastest_impl = impl_summary.loc[impl_summary['latency_ms'].idxmin()]['attn_impl']
        most_memory_efficient = impl_summary.loc[impl_summary['peak_memory_mb'].idxmin()]['attn_impl']
        
        if fastest_impl != 'naive+flash':
            best_for['naive+flash'] = f"Previously optimized for speed, but {fastest_impl} is faster in this benchmark"
            best_for[fastest_impl] = "Low latency"
            
        if most_memory_efficient != 'absorb':
            best_for['absorb'] = f"Previously optimized for memory, but {most_memory_efficient} uses less in this benchmark"
            best_for[most_memory_efficient] = "Memory efficiency"
        
        for _, row in impl_summary.iterrows():
            impl = row['attn_impl']
            f.write(f"| {impl} | {row['latency_ms']:.2f} | {row['peak_memory_mb']:.2f} | {best_for.get(impl, 'N/A')} |\n")
        
        f.write("\n")
        
        # Head dimension analysis
        f.write("## Head Dimension Configuration Analysis\n\n")
        f.write("| Head Config | Avg Latency (ms) | Avg Memory (MB) | Avg Throughput (K tok/s) |\n")
        f.write("|-------------|------------------|-----------------|-------------------------|\n")
        
        head_summary = clean_df.groupby(['qk_rope_head_dim', 'qk_nope_head_dim', 'v_head_dim']).agg({
            'latency_ms': 'mean',
            'peak_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        }).reset_index()
        
        for _, row in head_summary.iterrows():
            head_config = f"({row['qk_rope_head_dim']},{row['qk_nope_head_dim']},{row['v_head_dim']})"
            f.write(f"| {head_config} | {row['latency_ms']:.2f} | {row['peak_memory_mb']:.2f} | ")
            f.write(f"{row['throughput_tokens_per_sec']/1000:.2f} |\n")
        
        f.write("\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Find overall best implementation
        best_latency = impl_summary.loc[impl_summary['latency_ms'].idxmin()]['attn_impl']
        best_memory = impl_summary.loc[impl_summary['peak_memory_mb'].idxmin()]['attn_impl']
        best_throughput = impl_summary.loc[impl_summary['throughput_tokens_per_sec'].idxmax()]['attn_impl']
        
        # Find best head configuration
        best_head_latency_idx = head_summary['latency_ms'].idxmin()
        best_head_config_latency = f"({head_summary.iloc[best_head_latency_idx]['qk_rope_head_dim']}, " \
                                  f"{head_summary.iloc[best_head_latency_idx]['qk_nope_head_dim']}, " \
                                  f"{head_summary.iloc[best_head_latency_idx]['v_head_dim']})"
        
        best_head_memory_idx = head_summary['peak_memory_mb'].idxmin()
        best_head_config_memory = f"({head_summary.iloc[best_head_memory_idx]['qk_rope_head_dim']}, " \
                                 f"{head_summary.iloc[best_head_memory_idx]['qk_nope_head_dim']}, " \
                                 f"{head_summary.iloc[best_head_memory_idx]['v_head_dim']})"
        
        f.write(f"- **Fastest implementation**: `{best_latency}`\n")
        f.write(f"- **Most memory-efficient implementation**: `{best_memory}`\n")
        f.write(f"- **Highest throughput implementation**: `{best_throughput}`\n")
        f.write(f"- **Best head configuration for speed**: {best_head_config_latency}\n")
        f.write(f"- **Best head configuration for memory**: {best_head_config_memory}\n\n")
        
        # Add recommendations
        f.write("### Recommendations\n\n")
        
        if best_latency == best_throughput:
            f.write(f"- For **high performance** applications: Use `{best_latency}` implementation ")
            f.write(f"with head configuration {best_head_config_latency}\n")
        else:
            f.write(f"- For **low latency** applications: Use `{best_latency}` implementation ")
            f.write(f"with head configuration {best_head_config_latency}\n")
            f.write(f"- For **high throughput** applications: Use `{best_throughput}` implementation\n")
            
        f.write(f"- For **memory-constrained** environments: Use `{best_memory}` implementation ")
        f.write(f"with head configuration {best_head_config_memory}\n")
        
        # Analysis insights
        f.write("\n### Key Insights\n\n")
        
        # Calculate speedup factor of best vs worst implementation
        impl_latency_max = impl_summary['latency_ms'].max()
        impl_latency_min = impl_summary['latency_ms'].min()
        speedup = impl_latency_max / impl_latency_min if impl_latency_min > 0 else 0
        
        f.write(f"- The fastest implementation (`{best_latency}`) is {speedup:.2f}x faster than the slowest one.\n")
        
        # Check if RoPE+NoPE config is consistently better
        rope_nope_df = clean_df[clean_df['qk_nope_head_dim'] > 0]
        rope_only_df = clean_df[clean_df['qk_nope_head_dim'] == 0]
        
        if len(rope_nope_df) > 0 and len(rope_only_df) > 0:
            rope_nope_latency = rope_nope_df['latency_ms'].mean()
            rope_only_latency = rope_only_df['latency_ms'].mean()
            
            if rope_nope_latency < rope_only_latency:
                f.write("- Configurations using both RoPE and NoPE attention (mixed head dimensions) perform better, with ")
                f.write(f"{rope_only_latency/rope_nope_latency:.2f}x lower latency on average.\n")
            else:
                f.write("- Configurations using only RoPE attention (without NoPE) perform better, with ")
                f.write(f"{rope_nope_latency/rope_only_latency:.2f}x lower latency on average.\n")
        
        # Analyze sequence length scaling
        f.write("- As sequence length increases, the performance gap between implementations ")
        
        # Get data for longest and shortest sequence lengths
        min_seq = min(clean_df['seq_len'].unique())
        max_seq = max(clean_df['seq_len'].unique())
        
        min_seq_df = clean_df[clean_df['seq_len'] == min_seq]
        max_seq_df = clean_df[clean_df['seq_len'] == max_seq]
        
        # Calculate performance gaps at min and max sequence lengths
        if len(min_seq_df) > 0 and len(max_seq_df) > 0:
            min_seq_gap = min_seq_df['latency_ms'].max() / min_seq_df['latency_ms'].min() if min_seq_df['latency_ms'].min() > 0 else 0
            max_seq_gap = max_seq_df['latency_ms'].max() / max_seq_df['latency_ms'].min() if max_seq_df['latency_ms'].min() > 0 else 0
            
            if max_seq_gap > min_seq_gap:
                f.write(f"widens. At sequence length {min_seq}, the fastest implementation is {min_seq_gap:.2f}x faster than the slowest, ")
                f.write(f"while at sequence length {max_seq}, the gap increases to {max_seq_gap:.2f}x.\n")
            else:
                f.write(f"narrows. At sequence length {min_seq}, the fastest implementation is {min_seq_gap:.2f}x faster than the slowest, ")
                f.write(f"while at sequence length {max_seq}, the gap decreases to {max_seq_gap:.2f}x.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MLA implementations with different configurations")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp8"], default="bf16",
                       help="Data type to use for benchmarking")
    parser.add_argument("--implementations", type=str, nargs="+",
                        default=["naive", "absorb", "naive+flash"],
                        help="MLA implementations to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4],
                       help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[2**i for i in range(8, 12)],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--heads", type=int, nargs="+", default=[16],
                       help="Number of heads to benchmark")
    parser.add_argument("--rope-dims", type=int, nargs="+", default=[128],
                       help="RoPE head dimensions to benchmark")
    parser.add_argument("--nope-dims", type=int, nargs="+", default=[64],
                       help="NoPE head dimensions to benchmark")
    parser.add_argument("--v-dims", type=int, nargs="+", default=[128],
                       help="Value head dimensions to benchmark")
    parser.add_argument("--trials", type=int, default=20,
                       help="Number of trials for each benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")

    args = parser.parse_args()

    torch.set_default_device('cuda')
    if args.dtype == 'bf16':
        torch.set_default_dtype(torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        assert False & "WARNING: CUDA not available, benchmarks will run on CPU"

    head_dims = [(r, n, v) for r in args.rope_dims for n in args.nope_dims for v in args.v_dims]

    print(f"Running benchmarks with {args.dtype} precision...")
    df = sweep_mla_configs(
        dtype=args.dtype,
        attn_impls=args.implementations,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        n_heads_list=args.heads,
        head_dims=head_dims,
        n_warmup=args.warmup,
        n_trials=args.trials,
        device=device,
        output_dir=args.output_dir
    )

    # Generate plots
    print("Generating visualization plots...")
    plot_latency_comparison(df, args.output_dir)
    plot_throughput_comparison(df, args.output_dir)
    plot_memory_comparison(df, args.output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df, f"{args.output_dir}/mla_benchmark_report.md")

    print(f"Benchmark completed. Results and visualizations saved to '{args.output_dir}' directory.")
