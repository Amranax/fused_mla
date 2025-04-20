# fused_mla
Fused Multi-Latent-Attention-Kernel
# Fused Multi-Latent Attention Kernel (FusedMLA)

**Project Members:** 
- Abdulmajeed Amran (Student ID: 43344936)

## Overview

This repository implements optimized versions of Multi-Latent Attention (MLA), a novel attention mechanism used in models like DeepSeek V2/V3. Our project focuses on optimizing MLA implementations to reduce memory usage and computational overhead.

## Repository Structure

```
├── attention_impl/          # MLA implementations
│   ├── General_Layers.py    # Common building blocks
│   ├── Shared_Args.py       # Configuration dataclass
│   ├── absorb_mla.py        # Memory-efficient implementation
│   ├── mla_base.py          # Base MLA implementation
│   ├── mla_factory.py       # Factory for selecting implementation
│   ├── naive_mla.py         # Standard implementation
│   ├── naive_wflash_mla.py  # Flash attention implementation
│   └── fused_mla.py         # Fused kernel implementation
├── benchmarks/              # Benchmark scripts
│   ├── benchmark.py         # Performance benchmark
│   └── correctness_benchmark.py  # Correctness verification
├── kernels/                 # CUDA/Triton kernels
│   ├── deep_seek_kernels.py        # Core quantization kernels
│   ├── fused_attention_modified_mla.py  # Modified flash attention
│   └── fully_fused_mla.py   # Custom MLA kernel
└── README.md                # This file
```

## Installation and Setup

### Requirements

- Python 3.10 or higher
- CUDA-compatible GPU with CUDA 12.0+
- PyTorch 2.6.0 or higher

### Setup Environment

1. Clone this repository
2. Create and activate a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Compiling the Code

The codebase primarily uses PyTorch and Triton, which are JIT-compiled at runtime. No manual compilation is required. The Triton kernels are compiled automatically when first executed.

## Running the Code

### Basic Usage

To run a simple test of the MLA implementation:

```python
from attention_impl.Shared_Args import Args
from attention_impl.mla_factory import get_mla
import torch

# Set up model parameters
args = Args(
    max_batch_size=4,
    max_seq_len=1024,
    n_heads=16,
    dim=2048,
    dtype="bf16",
    attn_impl="absorb"  # Choose from: "naive", "absorb", "naive+flash"
)

# Create model
model = get_mla(args.attn_impl, args).cuda()

# Create sample inputs
batch_size, seq_len = 2, 256
x = torch.randn(batch_size, seq_len, args.dim, device='cuda', dtype=torch.bfloat16)
start_pos = 0
freqs_cis = model.precompute_freqs_cis(args)[:seq_len]
mask = torch.full((seq_len, seq_len), float("-inf"), device='cuda').triu_(1)

# Run forward pass
output = model(x, start_pos, freqs_cis, mask)
```

### Running Benchmarks

#### Performance Benchmark

This benchmark tests performance across different implementations, sequence lengths, and batch sizes:

```bash
python -m benchmarks.benchmark --dtype bf16 --implementations naive absorb "naive+flash" \
    --batch-sizes 1 4 --seq-lengths 256 512 1024 --output-dir benchmark_results
```

Options:
- `--dtype`: Data type (bf16, fp8)
- `--implementations`: MLA implementations to benchmark
- `--batch-sizes`: Batch sizes to test
- `--seq-lengths`: Sequence lengths to test
- `--heads`: Number of attention heads
- `--rope-dims`: RoPE head dimensions to test
- `--nope-dims`: Non-RoPE head dimensions to test
- `--v-dims`: Value head dimensions to test
- `--trials`: Number of benchmark trials
- `--warmup`: Number of warmup iterations
- `--output-dir`: Directory to save results

#### Correctness Benchmark

This benchmark verifies implementation correctness while comparing performance:

```bash
python -m benchmarks.correctness_benchmark --dtype bf16 --min-seq-len 256 --max-seq-len 1024 \
    --seq-len-step 256 --batch-sizes 1 4
```

Options:
- `--dtype`: Data type (bf16, fp8)
- `--min-seq-len`: Minimum sequence length to test
- `--max-seq-len`: Maximum sequence length to test
- `--seq-len-step`: Step size for sequence length
- `--batch-sizes`: Batch sizes to test
- `--output`: Output file prefix
- `--tolerance`: Tolerance for tensor comparison (default: 1.5e-2)

### Example Benchmark Run

To run a complete benchmark suite testing all implementations:

```bash
# Create results directory
mkdir -p benchmark_results

# Run performance benchmark
python -m benchmarks.benchmark --dtype bf16 --implementations naive absorb "naive+flash" \
    --batch-sizes 1 4 8 --seq-lengths 256 512 1024 2048 --output-dir benchmark_results/perf

# Run correctness benchmark
python -m benchmarks.correctness_benchmark --dtype bf16 --min-seq-len 256 --max-seq-len 2048 \
    --seq-len-step 256 --batch-sizes 1 4 --output correctness
```

## Understanding the Results

The benchmarks generate several outputs:

### Performance Benchmark Results

- **CSV files**: Detailed metrics for each implementation/configuration
- **Latency plots**: Compare latency across implementations
- **Throughput plots**: Compare tokens/second across implementations
- **Memory plots**: Compare GPU memory usage across implementations
- **Summary report**: Markdown report with findings and recommendations

### Correctness Benchmark Results

- **PNG visualization**: Compares latency and memory usage between implementations
- **CSV data**: Raw benchmark results
- **Console output**: Numerical differences between implementations

## Implementation Details

We've implemented several versions of MLA:

1. **Naive MLA**: Standard implementation with separate K/V caching
2. **Absorb MLA**: Memory-efficient implementation with fused KV caching
3. **Naive+Flash MLA**: Implementation leveraging flash attention for speed
4. **Fused MLA**: Custom Triton kernel implementation (experimental)

Our optimizations focus on:
- Reducing memory bandwidth requirements
- Efficient caching strategies
- Kernel fusions

## Understanding Implementation Differences

- **Naive** implementation uses separate caches for keys and values
- **Absorb** implementation uses a latent-space cache to reduce memory usage
- **Naive+Flash** implementation uses the flash attention algorithm for computation
- **Fused** implementation uses custom Triton kernels to avoid redundant computations

## Troubleshooting

- If you encounter CUDA out-of-memory errors, reduce batch size or sequence length
- For compilation errors with Triton, ensure you have compatible CUDA drivers
- If benchmarks show large differences between implementations, increase the tolerance parameter
