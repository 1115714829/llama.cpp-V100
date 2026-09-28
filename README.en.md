# llama.cpp-v100

[简体中文](README.md) | **English**

A modified version of llama.cpp for the NVIDIA V100 (SM70). Current version **1.0.2**; see [CHANGELOG.md](CHANGELOG.md) for the changes in each release.

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

- **Multi-GPU parallelism**: optimized scheduling and kernel launching for tensor parallelism to reduce the time GPUs wait for the host; supports splits where the number of attention heads is not divisible by the number of GPUs (e.g. 6 GPUs).
- **Long context**: keeps prefill and decode speed stable at very long contexts, and reduces the VRAM used by compute buffers.
- **Speculative decoding**: a complete DFlash2 speculative decoding pipeline, with optimized sampling and per-round overhead for higher decode speed.
- **Compute kernels**: some SM70 compute kernels are ported from 1Cat-vLLM and adapted to the Q8_0 format, combined with operator fusion.
- **Platform adaptation**: adapted to the NVLink interconnect and the NUMA memory architecture of the IBM AC922.
- **Multimodal**: the vision module (mmproj) can run on GPU to process images, PDFs and videos.

---

## Test data

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "two concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- Speeds were measured on 1.0.0-1.0.2; peak VRAM was measured on 1.0.2.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Peak VRAM per GPU |
|---|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1652 tok/s | 281.5 tok/s | 13.3 GB |
| real code 209115 tokens (1024 output tokens) | 1646 tok/s | 28.3 ms per speculative round | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1997 tok/s | 310.4 tok/s | 13.3 GB |
| real code 131188 tokens (1024 output tokens) | 1983 tok/s | 97.6 tok/s | 13.3 GB |
| real code 16499 tokens (1024 output tokens) | 2570 tok/s | 123.5 tok/s | 13.3 GB |

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| thinking mode | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| non-thinking mode | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1799 tok/s, decode 278.0 tok/s |
| 8192 | not enough VRAM, fails to start |

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1720 tok/s | 272.9 tok/s | 29.3 ms | 10.2 GB |
| 2048 | real code 209233 tokens (1024 output tokens) | 1719 tok/s | 84.3 tok/s | 29.1 ms | 10.2 GB |
| 2048 | real code 16496 tokens (1024 output tokens) | 2592 tok/s | 135.3 tok/s | 22.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.6 tok/s | 29.2 ms | 11.4 GB |
| 4096 | real code 209229 tokens (1024 output tokens) | 1905 tok/s | 87.4 tok/s | 29.1 ms | 11.4 GB |
| 4096 | real code 16497 tokens (1024 output tokens) | 2731 tok/s | 115.7 tok/s | 22.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.3 tok/s | 29.2 ms | 13.8 GB |

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 360 s | 1165 tok/s | 220.2 tok/s | 12.7 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1309 tok/s | 218.1 tok/s | 13.9 GB |

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### 4 GPUs, two concurrent requests

`-c 262144 -np 2` (131072 per request), measured sequentially in the same server, peak VRAM 14.5-14.7 GB per GPU:

| Scenario | TTFT | Decode per request | Total decode |
|---|---|---:|---:|
| single request 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| two concurrent requests of 16K each | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| single request about 115K tokens | 55.8 s | 101.5 tok/s | 102 tok/s |
| two concurrent requests of about 115K tokens each | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

`-c 524288 -np 2` (262144 per request) does not fit in VRAM and fails to start.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | TTFT | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 136 tok/s |
| image (diagram) | 4069 | 5.6 s | 126 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 151 tok/s |
| 12-second video (slides) | 23104 | 19.5 s | 164 tok/s |
| 10-second video | 6387 | 7.3 s | 160 tok/s |
| 200K tokens of text + 1 image | 206958 | 122 s | 91 tok/s |

- Appending 8030 tokens to the same session after 208K tokens: TTFT 7.7 s (only the new part is processed).
- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 217.9 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: a one-image request takes 7.6 s and the PDF first 6 pages 5.5 s (both include answer generation); peak VRAM about 14.7 GB on GPU 0.
- With the vision module on CPU (`--no-mmproj-offload`): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

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
| `-np 1` | single request (one slot) |
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

- **Multi-GPU splitting**: Qwen3.8 has 4 attention KV heads, so with 6-GPU tensor parallelism 2 GPUs have no attention heads in each attention layer, and the time per speculative round on 6 GPUs is about 3% higher than on 4 GPUs. Future work will improve splitting and inter-GPU communication for such uneven cases to raise prefill and decode speed on 6 GPUs.
- **Concurrent requests**: optimization currently targets single requests; with multiple concurrent requests, prefill and decode affect each other. Future work will improve scheduling and throughput for concurrent requests.

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
