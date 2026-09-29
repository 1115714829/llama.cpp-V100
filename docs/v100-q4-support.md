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

## P1 2-way tensor parallel (this commit)

Topology: 2 x V100-SXM2-16GB on NV2 NVLink (P2P available), PCIe gen3 x8.

All-reduce dispatch per call: push (small F32, <= 512 KiB, flat kernel for 2
ranks) in front, then NCCL (Linux default) or the internal 2-GPU copy-engine
pipeline (`GGML_CUDA_ALLREDUCE=internal`). The 2+2 / 3+3 clique reductions
apply to 4/6 ranks only; 2 ranks always use the flat kernel.

Verified:
- `test-cuda-allreduce`: ALL_OK on NCCL and internal, including the
  CUDA-graph chain cases (20-320 KB chained calls 3-10 us).
- `test-backend-ops -o MUL_MAT,RMS_NORM`: pass on all backends.
- Greedy decode deterministic across repeated runs.
- Tensor split 1,1 vs 1,3 perplexity (3 chunks of a 20 KB corpus, 2048 ctx):
  1.7380/2.0739/2.3896 vs 1.7395/2.0755/2.3890. The delta (<= 0.002) comes
  from the BF16 wire rounding of large reductions at different split
  boundaries, not from split errors.

Fixed in the internal (2-GPU) all-reduce path:
- `exact` collectives were reduced over the BF16 wire and lost bitwise
  equality (observed: exact test got -0.116211 vs -0.11619). The flag now
  disables the BF16 round-trip, so exact payloads reduce in F32.
- A hard `GGML_ASSERT` on 16-byte-multiple nbytes aborted instead of falling
  back. The kernels handle tails, so the check was dropped; unaligned nbytes
  now pass through the pipeline like any other size.

Known pre-existing issue (not introduced here): the full `test-backend-ops`
run aborts in `fattn-mma-f16.cuh` `cudaFuncSetAttribute` for the hsk=320
FLASH_ATTN_EXT case (dynamic shared memory above the V100 48 KB limit).

Launch template:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-server \
  -m Qwen3.8-27B-Q4_K_M.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048
```

## P5.1 Q4_0 KV cache quality (this commit)

Perplexity on a 300 KB repo-docs corpus (2-way TP, Q4_K_M weights, -fa on):

| KV cache | PPL ctx=2048 | PPL ctx=8192 |
|---|---:|---:|
| f16 / f16 | 2.7734 | - |
| q8_0 / q8_0 | 2.7734 | 2.6017 |
| q4_0 / q8_0 | 2.7765 | 2.6024 |
| q4_0 / q4_0 | 2.7792 | 2.6050 |

Needle test (magic word buried at 50% of the context, greedy decode):

| Context | q8_0/q8_0 | q4_0/q8_0 | q4_0/q4_0 |
|---|---|---|---|
| ~35K tokens | correct | correct | correct |
| ~145K tokens | correct | - | correct |

Conclusion: K=q4_0 V=q4_0 is usable as-is. The PPL delta (<= 0.006 at 2K,
<= 0.003 at 8K) is inside the measurement noise and the needle tests pass at
both lengths. No need for the mixed K=q4_0 / V=q8_0 fallback on quality
grounds; the sm70 kernel work should still cover the mixed pair (it is the
same code path with different launch templates).

Performance note: on the generic attention paths Q4_0 KV prefill is slower
than Q8_0 KV (35K needle: 1080 vs 1300 tok/s; 145K needle: 585 vs 1019
tok/s) because K/V are dequantized to F16 staging. This is the gap the P3
sm70 Q4_0 KV kernels close.

## P3 sm70 attention kernels with Q4_0 KV (this commit)

- `fattn-sm70-grouped` (decode / speculative verify, n_q 2..16): Q4_0 added
  next to F16/Q8_0. The register prefetch and staging already work per type;
  `flash_attn_sm70_grouped_dequant_kv` grows a q4_0 branch and the launch
  table covers all 9 K/V type pairs for both the 8- and 16-token variants.
- `fattn-sm70-d256` (prefill, q >= 17): `sm70_d256_kv_type_ok` accepts q4_0
  and a q4_0 twin of the range-mask partial mirror kernel converts only the
  rows the mask bounds allow. The full-mirror fallback path is unchanged
  (generic to_fp16_nc). The partial/full choice is now per K/V tensor, which
  also fixes a latent bug where a mixed Q8_0 K with a non-F16 V ran the q8_0
  row kernel over V.

Q4_0 block layout gotcha (cost a debug round): the 32 values of a q4_0 block
are not nibble-interleaved. Low nibbles of qs[0..15] hold values 0..15 and
high nibbles hold values 16..31 (see the vec_dot_q4_0_q8_0 pairing in
ggml/src/ggml-cpu/quants.c).

Verified (2-way TP, Q4_K_M weights):
- perplexity 2048: 2.7796 (baseline 2.7792); 8192: 2.6048 (baseline 2.6050)
- needle tests 35K and 145K tokens: both correct
- prefill with q4_0 KV now matches q8_0 KV: 35K 1307 vs 1309 tok/s,
  145K 1017 vs 1020 tok/s (before this commit the generic path gave
  1080 and 585 tok/s)
- DFlash2 speculative decode with q4_0 KV works through the grouped verify
  kernel (96-token run, sensible output)
- test-backend-ops FLASH_ATTN_EXT hsk=256 and MUL_MAT: pass; test-cuda-allreduce: ALL_OK
