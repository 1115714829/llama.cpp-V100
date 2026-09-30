# llama.cpp-v100

[简体中文](README.md) | **English**

A modified version of llama.cpp for the NVIDIA V100 (SM70). Current version **1.0.6**; see [CHANGELOG.md](CHANGELOG.md) for the changes in each release.

---

## What's new in 1.0.6

1.0.6 improves prefill speed on 2, 3, 5, 6 GPUs and decode speed with the Q4 configuration; on 4 GPUs with Q8 the speed, VRAM and output are all the same as 1.0.5.

- **GDN chunked prefill supports any number of heads per GPU**: previously only 4 GPUs (12 GDN heads per GPU) used the chunked algorithm, and other GPU counts fell back to per-token recursion. Now 2 to 6 GPUs all use the chunked path. 6-GPU 200K synthetic prefill 1710 -> 1801 tok/s (+5.4%), 6-GPU real 16K 2568 -> 2774 tok/s (+8%), 2 GPUs with Q4 +8-10%, 3 GPUs with Q8 +6%. Perplexity unchanged (6 GPUs 1.6558 -> 1.6555, 3 GPUs 1.6554 -> 1.6555).
- **Small-batch matrix multiplication kernel for Q4_K weights**: speculative verification (8 tokens per round) no longer goes through the generic path. The Q4_K weights are rearranged in place at load time, with no extra VRAM. Time per speculative round with the Q4 configuration drops by 5-9% (e.g. 2 GPUs at 262144: 39.1 -> 35.8 ms, 4 GPUs at 524288: 28.4 -> 26.7 ms). The output of the Q4 configuration changes slightly, and perplexity is unchanged (2 GPUs 1.7321 -> 1.7320).
- **Baseline**: measured alternately with 1.0.0 in the same session, 4-GPU 200K prefill 1641.0 vs 1641.6 tok/s, time per speculative round unchanged; concurrent speed is on par with 1.0.4.

---

## Acknowledgments

**Special thanks to [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).** Its engineering work on the V100 provided many references for this project, and some of the SM70 compute kernels are ported from it.

Thanks also to:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)**: the base framework of this project.
- **[vLLM](https://github.com/vllm-project/vllm)**: its designs for speculative decoding and other areas provided many references for this project.

---

## Introduction

Based on llama.cpp (master `08618ff8e` as of 2026-09-26, already includes upstream v0.5.0). Some SM70 compute kernels are ported from other open-source projects; see the file headers and the `licenses/` directory for provenance and license.

The test data below all uses Qwen3.8-27B (Q8_0) with DFlash2 speculative decoding.

---

## Optimization directions

Systematic optimization around the hardware characteristics of the V100 (SM70), from compute kernels to the runtime:

- **Multi-GPU parallelism**: optimized scheduling and kernel launching for tensor parallelism to reduce the time GPUs wait for the host; all-reduce between GPUs is done hierarchically following the NVLink topology; supports split optimization for the case where the number of attention heads is not divisible by the number of GPUs (e.g. 6 GPUs).
- **Long context**: keeps prefill and decode speed stable at very long contexts, and reduces the VRAM used by compute buffers.
- **Speculative decoding**: a complete DFlash2 speculative decoding pipeline, with optimized sampling and per-round overhead for higher decode speed.
- **Compute kernels**: attention, matrix multiplication and other compute kernels for SM70, adapted to the Q8_0 and Q4 formats, combined with operator fusion.
- **Platform adaptation**: adapted to the NVLink interconnect and the NUMA memory architecture of the IBM AC922.
- **Concurrent requests**: with multiple requests, decode gets priority and the time taken by prompt processing is limited; the decode batch, speculative decoding and VRAM reservation for multiple requests are all optimized by the number of requests.
- **Multimodal**: the vision module (mmproj) can run on GPU to process images, PDFs and videos.

---

## Test data

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---:|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1638 tok/s | 278.9 tok/s | 28.6 ms | 13.3 GB |
| real code 209228 tokens (1024 output tokens) | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| real code 131184 tokens (1024 output tokens) | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| real code 16500 tokens (1024 output tokens) | 2547 tok/s | 119.5 tok/s | 22.0 ms | 13.3 GB |

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2552 tok/s | 116.5 tok/s | 0.22 | 22.0 ms |
| thinking mode | 128K | 1963 tok/s | 97.6 tok/s | 0.22 | 25.9 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2620 tok/s | 105.4 tok/s | 0.25 | 26.0 ms |
| non-thinking mode | 128K | 1971 tok/s | 89.1 tok/s | 0.24 | 29.8 ms |

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1801 tok/s, decode 277.1 tok/s |
| 8192 | not enough VRAM, fails to start |

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1708 tok/s | 293.1 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 209227 tokens (1024 output tokens) | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 16494 tokens (1024 output tokens) | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16500 tokens (1024 output tokens) | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 363 s | 1154 tok/s | 233.1 tok/s | 12.8 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Concurrent requests

Real code, thinking mode (T=0.6), 1024 output tokens per request, default `--prefill-pace 30`. Measured sequentially in the same server; "concurrent" means the requests are sent at the same time.
- TTFT is counted from when the request is sent; decode per request = the speed of that request from its first token to the end.
- Aggregate decode = total output across all requests divided by (earliest first token to the end of the last request).
- All done = from sending the requests to the end of the last request.

**4 GPUs, `-c 262144 -np 2` (131072 per request)**, startup VRAM 13.5 GB per GPU, peak 13.7 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 115.7 tok/s | 116 tok/s | 15.4 s |
| two concurrent requests of 16K each | 8.1 s / 22.7 s | 81.4 / 111.4 tok/s | 86 tok/s | 31.9 s |
| single request about 115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of about 115K tokens each | 58.0 s / 122.6 s | 70.2 / 81.4 tok/s | 27 tok/s | 135.2 s |

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each | 8.5 / 27.1 / 43.6 / 57.5 s | 64.0 / 69.9 / 81.9 / 124.4 tok/s | 72 tok/s | 65.7 s |

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4-12.8 GB per GPU, peak 12.7-13.0 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each | 8.0 s / 21.9 s | 85.4 / 116.1 tok/s | 90 tok/s | 30.7 s |
| two concurrent requests of about 115K tokens each | 55.3 s / 118.8 s | 66.0 / 90.5 tok/s | 27 tok/s | 130.1 s |

Compared with 1.0.3 (4 GPUs, same test):

| Scenario | 1.0.3 | 1.0.4 |
|---|---|---|
| `-np 2` two 16K requests: TTFT | 9.2 s / 16.7 s | 8.1 s / 22.7 s |
| `-np 2` two 16K requests: decode per request | 45.7 / 58.8 tok/s | 81.4 / 111.4 tok/s |
| `-np 2` two 16K requests: all done | 34.1 s | 31.9 s |
| `-np 2` two 115K requests: TTFT | 59.6 s / 133.0 s | 58.0 s / 122.6 s |
| `-np 2` two 115K requests: decode per request | 11.4 / 58.9 tok/s | 70.2 / 81.4 tok/s |
| `-np 2` two 115K requests: all done | 150.6 s | 135.2 s |
| `-np 2` startup VRAM per GPU | 14.0 GB | 13.5 GB |
| `-np 4` (64K per request) | fails at startup for lack of VRAM | runs |

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 -> 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2549 / 131.7 tok/s / 22.0 ms | 209715 / 128 s / 1641 / 277.8 tok/s | 13.3 GB |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 351 s / 1197 / 232.2 tok/s | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8: the Q4_K small-batch kernel currently only covers a single matrix multiplication; gate/up fusion and multi-weight fusion still go through the generic path.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

| GPUs | Config | Context (per request) | single 16K: TTFT / decode | two concurrent 16K: TTFT | Decode per request | Peak VRAM per GPU |
|---|---|---:|---|---|---|---:|
| 2 | Q4 | 131072 (65536) | 10.0 s / 67.2 tok/s | 12.5 s / 37.2 s | 44.6 / 64.7 tok/s | 14.4 GB |
| 3 | Q8 | 131072 (65536) | 7.3 s / 74.1 tok/s | 9.1 s / 26.4 s | 64.4 / 103.8 tok/s | 15.2 GB |
| 3 | Q4 | 524288 (262144) | 7.6 s / 82.0 tok/s | 9.5 s / 28.7 s | 57.5 / 62.8 tok/s | 14.7 GB |
| 4 | Q8 | 262144 (131072) | 6.5 s / 117.0 tok/s | 8.2 s / 23.9 s | 70.6 / 89.6 tok/s | 13.7 GB |
| 4 | Q4 | 524288 (262144) | 6.7 s / 78.9 tok/s | 8.5 s / 24.5 s | 69.4 / 87.9 tok/s | 12.0 GB |
| 5 | Q8 | 524288 (262144) | 6.7 s / 109.2 tok/s | 8.6 s / 22.8 s | 84.2 / 121.5 tok/s | 14.5 GB |
| 5 | Q4 | 524288 (262144) | 6.8 s / 76.5 tok/s | 8.4 s / 26.4 s | 61.4 / 98.2 tok/s | 10.6 GB |
| 6 | Q8 | 524288 (262144) | 6.4 s / 117.7 tok/s | 8.0 s / 23.1 s | 73.7 / 121.4 tok/s | 13.0 GB |
| 6 | Q4 | 524288 (262144) | 6.5 s / 81.9 tok/s | 8.0 s / 24.8 s | 66.1 / 78.7 tok/s | 9.7 GB |

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 182 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides) | 23104 | 19.7 s | 150 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 233.3 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.3 s and the PDF first 6 pages 3.2 s; peak VRAM 14.8 GB on GPU 0.
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/path/to/nccl/include -DNCCL_LIBRARY=/path/to/nccl/lib/libnccl.so.2
cmake --build build -j --target llama-server llama-quantize
```

Multi-GPU tensor parallelism requires NCCL. Use GCC 12 or newer.

Every push on GitHub automatically runs an sm_70 compile check; releases include prebuilt binaries for x86_64 Linux (CUDA 12, see the Releases page). For other platforms such as POWER9, build with the command above.

---

## Model download

The tests use the files below; the vision module is only needed for images and video. **Users in mainland China are recommended to download from ModelScope.**

| Purpose | File | ModelScope | Hugging Face |
|---|---|---|---|
| target model | `Qwen3.8-27B-Q8_0.gguf` (about 29 GB) | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| draft model (DFlash2) | `Qwen3.8-27B-DFlash2-BF16.gguf` (about 3.9 GB), convert to F16 after download | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
| target model (Q4 configuration) | `Qwen3.8-27B-UD-Q4_K_M.gguf` (about 16.5 GB) | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| draft model (when VRAM is tight) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (about 1.1 GB), use directly | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
| vision module (optional) | `mmproj-F16.gguf` (about 0.9 GB) | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

The target model can be used directly after download. The draft model is officially released only as BF16, Q8_0, and Q4_K_M GGUF; the tests use F16 (the V100 has no BF16 hardware support), which requires a one-time conversion with the `llama-quantize` built from this project.

First build as in the previous section, then one command downloads and converts:

```bash
scripts/v100-get-models.sh            # downloads from ModelScope to ./models by default
scripts/v100-get-models.sh -s hf      # use Hugging Face instead
```

The script downloads the target model and the draft model (resumable; already downloaded files are skipped automatically), then converts the draft model to `Qwen3.8-27B-DFlash2-F16.gguf`. `-d` sets the model directory, `-q` sets the path to `llama-quantize`, and `-h` shows the usage. The vision module is not handled by the script; download `mmproj-F16.gguf` separately from the repository in the table above when needed.

If you have already downloaded the files yourself, only the conversion step is needed:

```bash
./build/bin/llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-F16.gguf F16
```

---

## Launch parameters

Below are the launch commands corresponding to the test data. `numactl --membind=0,8` limits host memory to the two CPU nodes of this machine (the AC922 exposes GPU VRAM as NUMA nodes; without this limit, page cache may occupy VRAM); on other machines, use `numactl -H` to check the node numbers.

### 4 GPUs, 262144 context

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 GPUs, 262144 context

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 GPUs, 524288 context (YaRN)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### Concurrent requests

Change `-np` on the single-request commands above; the total context is divided among the requests:

```bash
# 4 GPUs, two requests, 131072 each
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 2 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --prefill-pace 30 \
  --host 127.0.0.1 --port 8080 --metrics

# 4 GPUs, four requests, 65536 each: replace -np 2 above with -np 4

# 6 GPUs, two requests, 262144 each (YaRN)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 2 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --prefill-pace 30 \
  --host 127.0.0.1 --port 8080 --metrics
```

`--prefill-pace 30` is the default; it is written out only for clarity and can be omitted.

### Multimodal: 6 GPUs, 524288 context (YaRN), vision module on GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --mmproj ./models/mmproj-F16.gguf \
  --video-ffmpeg-dir /path/to/ffmpeg/bin \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080 --metrics
```

### Multimodal: 4 GPUs, 262144 context, vision module on GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --mmproj ./models/mmproj-F16.gguf \
  --video-ffmpeg-dir /path/to/ffmpeg/bin \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080 --metrics
```

Multimodal requests use the OpenAI-compatible `/v1/chat/completions`: images via `image_url`, videos via `input_video` (or `video_url`); videos are sampled at 4 frames per second by default. For PDF, the client renders each page into an image and sends it (about 600 tokens per page at 80 dpi).

### Other GPU counts and Q4 configuration

Change these parts of the commands above:
- GPU count: `CUDA_VISIBLE_DEVICES` and `--tensor-split` (one 1 for each GPU);
- Q4 configuration: `-m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf`, `-ctk q4_0 -ctv q4_0`;
- context: pick from the "GPU counts and quantization" table; for 524288 add the three YaRN parameters;
- 2 GPUs with Q4 at 262144: `--model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`.

For example, 2 GPUs with Q4, 262144 context:

```bash
CUDA_VISIBLE_DEVICES=0,1 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 262144 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### Parameter description

| Parameter | Description |
|---|---|
| `-ngl 999` | all layers on GPU |
| `--split-mode tensor --tensor-split 1,...,1` | tensor parallelism, evenly split across GPUs |
| `-c` | context length |
| `-np` | number of concurrent requests; context per request = `-c` divided by the number of requests |
| `--prefill-pace` | with multiple requests, the maximum percentage of time prompt processing of other requests may take while a request is decoding (0-100, default 30; 0 or 100 means no throttling) |
| `-fa on` | Flash Attention (required by the SM70 attention kernels) |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache; `-ctk q4_0 -ctv q4_0` is 4-bit, using half the VRAM |
| `-b 2048 -ub 2048` | batch size; the test data uses 2048 unless noted otherwise |
| `--model-draft ...`, `--spec-type draft-dflash`, `--spec-draft-n-max 7` | DFlash2 speculative decoding, drafts 7 tokens per round |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` | use YaRN to extend the context to 524288 (the model's native length is 262144) |
| `--override-kv qwen35.context_length=int:524288` | allow a single-slot context beyond the model's training length |
| `--mmproj` | vision module |
| `--video-ffmpeg-dir` | directory containing `ffmpeg` and `ffprobe`, needed for video |
| `--no-mmproj-offload` | put the vision module on CPU (by default it is on the first GPU; weights about 0.9 GB, compute buffers up to about 1.2 GB) |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--chat-template-kwargs` | default sampling parameters and thinking mode switch (the values above are the model card's thinking mode parameters) |
| `--metrics` | Prometheus metrics (optional) |

Sampling: when `top_k ≤ 64` and only top-k, top-p, min-p, and temperature are used, the top-k candidates are selected on the device; when penalties, DRY, or other samplers are used, or `top_k` is larger, sampling over the full vocabulary is done on the host (see the non-thinking mode in the "sampling parameters recommended by the model card" table for the time per round).

---

## Next steps

- **Multi-GPU splitting**: Qwen3.8 has 4 attention KV heads; on 3 GPUs the attention heads are split 6/6/12 (one GPU has twice the attention computation of the other two), and on 5 or 6 GPUs 1-2 GPUs have no attention heads in each attention layer. Future work will let adjacent GPUs share KV heads so that the attention computation is balanced across the GPUs.
- **Q4 configuration**: gate/up fusion and multi-weight fusion kernels for Q4_K, and a small-batch kernel for 17-64 rows.
- **Long context**: prefill no longer keeping a full f16 copy of the K/V (saves VRAM), and long-context decode speed and prefill speed when appending.
- **Concurrent requests**: the total throughput of concurrent requests is still about the same as processing them one after another. Future work will improve the time per round when multiple requests decode at the same time, and prefix cache reuse between concurrent requests.

---

## Community

Scan the QR code below to add the author on WeChat and join the discussion group, where you can talk with other users about more ways to use this project.

<img src="media/v100-wechat.jpg" width="260" alt="WeChat QR code">

---

## Contributors

- **Repository author**: initiated the project, provided the hardware and test environment, set the goals and trade-offs, participated in testing.
- **Claude** (Anthropic, Claude Opus 5.5, via Claude Code): solution design, measurement and analysis, compilation, testing and stress testing on the server, code review, commits and documentation.
- **DeepSeek V4.1 Flash**: since the evening of 2026-09-26, took on most of the code implementation, as well as source research, design drafts, and diagnostic scripts, about 170 tasks.
- **Xiaomi MiMo v2.6-pro**: code research and the first implementation at the start of the project (2026-09-26), as well as later in-depth research and solution design, about 33 tasks.
- **[ATIVX928](https://github.com/ATIVX928)** (external contributor): SM70 attention kernel support for the q4_0 KV cache, 2-GPU internal all-reduce fix ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1), merged in 1.0.5).

Thanks to all the projects and participants above.

---

## License

Inherits the MIT license of llama.cpp (see `LICENSE`). Ported code keeps its original license; see the file headers for provenance and license.
