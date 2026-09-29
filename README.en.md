# llama.cpp-v100

[简体中文](README.md) | **English**

A modified version of llama.cpp for the NVIDIA V100 (SM70). Current version **1.0.4**; see [CHANGELOG.md](CHANGELOG.md) for the changes in each release.

---

## What's new in 1.0.4

1.0.4 supports concurrent requests (`-np 2`, `-np 4`); single-request speed is essentially the same as 1.0.3.

- **Decode is no longer held back by other requests**: added `--prefill-pace` (default 30). While a request is decoding, prompt processing of other requests takes at most 30% of the time. On 4 GPUs with two concurrent requests of about 115K tokens each: the request that starts decoding first goes from 11.4 to 70.2 tok/s, and the TTFT of the other goes from 133.0 s down to 122.6 s.
- **Concurrent requests decode faster together**: the decode batch limit grows with the number of requests (8 tokens per request, up to 64); the Q8_0 small-batch matrix multiplication kernel now supports 17-64 rows; with multiple requests the DFlash2 draft injection is fused into one pass and done entirely in VRAM.
- **Concurrent requests use less VRAM**: with multiple requests the compute buffers are also reserved only for the attention ranges actually used. On 4 GPUs with `-c 262144 -np 2`, startup VRAM per GPU drops from 14031 MiB to 13513 MiB; 4 GPUs with `-np 4` (64K per request) used to fail at startup for lack of VRAM and now runs; 6 GPUs with `-c 524288 -np 2` (256K per request) now runs.
- **Single request**: speed and VRAM are essentially the same as 1.0.3 (prefill of long prompts is about 0.2% slower, see the known issue in [CHANGELOG.md](CHANGELOG.md)). In long prompts, the scattered 17-64 token batches now use the new kernel, so the compute order differs and the output may differ slightly from 1.0.3, with unchanged perplexity.

---

## Acknowledgments

**Special thanks to [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).** Its engineering work on the V100 provided many references for this project, and some of the SM70 compute kernels are ported from it.

Thanks also to:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)**: the base framework of this project.
- **[vLLM](https://github.com/vllm-project/vllm)**: its designs for speculative decoding and other areas provided many references for this project.

---

## Introduction

Based on llama.cpp (master `08618ff8e` as of 2026-09-26, already includes upstream v0.5.0). Some SM70 compute kernels are ported from 1Cat-vLLM; see the file headers and the `licenses/` directory for provenance and license.

The test data below all uses Qwen3.8-27B (Q8_0) with DFlash2 speculative decoding.

---

## Optimization directions

Systematic optimization around the hardware characteristics of the V100 (SM70), from compute kernels to the runtime:

- **Multi-GPU parallelism**: optimized scheduling and kernel launching for tensor parallelism to reduce the time GPUs wait for the host; all-reduce between GPUs is done hierarchically following the NVLink topology; supports split optimization for the case where the number of attention heads is not divisible by the number of GPUs (e.g. 6 GPUs).
- **Long context**: keeps prefill and decode speed stable at very long contexts, and reduces the VRAM used by compute buffers.
- **Speculative decoding**: a complete DFlash2 speculative decoding pipeline, with optimized sampling and per-round overhead for higher decode speed.
- **Compute kernels**: some SM70 compute kernels are ported from 1Cat-vLLM and adapted to the Q8_0 format, combined with operator fusion.
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
- Unless noted otherwise, all data was measured on 1.0.4 (2026-09-29 to 30).
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

### Parameter description

| Parameter | Description |
|---|---|
| `-ngl 999` | all layers on GPU |
| `--split-mode tensor --tensor-split 1,...,1` | tensor parallelism, evenly split across GPUs |
| `-c` | context length |
| `-np` | number of concurrent requests; context per request = `-c` divided by the number of requests |
| `--prefill-pace` | with multiple requests, the maximum percentage of time prompt processing of other requests may take while a request is decoding (0-100, default 30; 0 or 100 means no throttling) |
| `-fa on` | Flash Attention (required by the SM70 attention kernels) |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache |
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

- **Multi-GPU splitting**: Qwen3.8 has 4 attention KV heads, so with 6-GPU tensor parallelism 2 GPUs have no attention heads in each attention layer, and prefill on 6 GPUs is about as fast as on 4 GPUs (the same at 16K input, about 3% faster at 200K input). Future work will improve splitting for such uneven cases and inter-GPU communication during prefill to raise prefill speed on 6 GPUs.
- **Concurrent requests**: 1.0.4 fixed the problem of decode being held back by prompt processing with multiple requests, but the total throughput of concurrent requests is still about the same as processing them one after another (prompt processing is queued, and requests share the GPU while decoding). Future work will improve the time per round when multiple requests decode at the same time, and prefix cache reuse between concurrent requests.
- **More quantization formats**: support Q4 quantization (Q4_K_M model weights and a q4_0 KV cache), so that configurations with less VRAM (for example 2 or 3 GPUs) can also run long contexts; and add a small-batch matrix multiplication kernel for Q4_K weights to raise decode speed with Q4.

---

## Community

Scan the QR code below to add the author on WeChat and join the discussion group, where you can talk with other users about more ways to use this project.

<img src="media/v100-wechat.jpg" width="260" alt="WeChat QR code">

---

## Contributors

- **Repository author**: initiated the project, provided the hardware and test environment, set the goals and trade-offs, participated in testing.
- **Claude** (Anthropic, Claude Opus 5.5, via Claude Code): solution design, measurement and analysis, compilation, testing and stress testing on the server, code review, commits and documentation.
- **DeepSeek V4.1 Flash**: since the evening of 2026-09-26, took on most of the code implementation, as well as source research, design drafts, and diagnostic scripts, about 140 tasks.
- **Xiaomi MiMo v2.6-pro**: code research and the first implementation at the start of the project (2026-09-26), about 26 tasks.

Thanks to all the projects and participants above.

---

## License

Inherits the MIT license of llama.cpp (see `LICENSE`). Ported code keeps its original license; see the file headers for provenance and license.
