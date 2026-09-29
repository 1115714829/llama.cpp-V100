# V100 Q4 support: 2-way tensor parallel + Q4_K_M weights + Q4_0 KV cache

Development notes for extending the V100 (SM70) optimizations to Q4 quantization
types and verifying 2-GPU tensor parallelism. Working configuration:

- 2 x Tesla V100-SXM2-16GB, CUDA 12.8, NCCL 2.26.2
- Model: Qwen3.8-27B Q4_K_M (`Qwen3.8-27B-Q4_K_M.gguf`)
- KV cache: Q4_0 for both K and V (`-ctk q4_0 -ctv q4_0`)
- Parallelism: pure 2-way tensor parallel (`--split-mode tensor --tensor-split 1,1`,
  flat one-shot push all-reduce for 2 ranks; the 2+2 / 3+3 clique reductions are
  not used)

## Scope of the Q4 kernel work

| Optimization | Status today | Q4 target |
|---|---|---|
| `q8-skinny` (weight repack, small-M GEMM, fused SwiGLU, multi-weight) | Q8_0 only | full Q4_0 twin |
| `fattn-sm70-grouped` (decode / verify attention) | KV = F16/Q8_0 | + Q4_0 KV |
| `fattn-sm70-d256` (prefill attention) | KV = F16/Q8_0 | + Q4_0 KV |
| rms_norm registers, silu fusion, device top-k, 32-row output split, I32 range mask, push all-reduce | quant-agnostic | none |

## P0 baseline (this commit)

Build: `Release`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_CUDA_FA=ON`,
`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_NCCL=ON`.

Server launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-server \
  -m Qwen3.8-27B-Q4_K_M.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048
```

Measured on 2026-09-29 (before any Q4 kernel work; all GEMMs on the generic
mmq/mmvq/cuBLAS paths, attention on the generic fattn-vec/dequant paths):

| Case | Prefill | Decode (M=1) | Peak VRAM per GPU |
|---|---:|---:|---:|
| synthetic 28001-token prompt, 1 output token | 1641.6 tok/s | - | 9.3 GB |
| synthetic 3001-token prompt, 256 output tokens | 1398.9 tok/s | 57.0 tok/s | 9.3 GB |
| short essay prompt, 256 output tokens | 98.8 tok/s | 57.3 tok/s | 9.3 GB |

Smoke test (`llama-cli --single-turn`, 4096 ctx): correct output, exit 0.

Follow-up phases: P1 2-way TP verification, P5.1 Q4_0 KV quality evaluation,
P3 sm70 attention Q4_0 KV, P2 q4_0-skinny, P4 Q4_K weight path decision,
P6 tests and benchmarks.
