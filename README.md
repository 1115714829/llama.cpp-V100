# llama.cpp-v100

**简体中文** | [English](README.en.md)

面向 NVIDIA V100（SM70）的 llama.cpp 修改版。当前版本 **1.0.2**，各版本的更新内容见 [CHANGELOG.md](CHANGELOG.md)。

---

## 致谢

**特别感谢 [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。** 它在 V100 上的工程实践给了本项目很多参考，部分 SM70 计算内核移植自它。

同样感谢：

- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)**：本项目的基础框架。
- **[vLLM](https://github.com/vllm-project/vllm)**：投机解码等方面的设计给了本项目很多借鉴。

---

## 简介

基于 llama.cpp（2026-09-26 的 master `08618ff8e`，已包含上游 v0.5.0）修改。部分 SM70 计算内核移植自 1Cat-vLLM，出处与许可见各文件头和 `licenses/` 目录。

下面的测试数据都用 Qwen3.8-27B（Q8_0）配合 DFlash2 投机解码。

---

## 本项目优化的方向

围绕 V100（SM70）的硬件特性，从计算内核到运行时进行了系统性优化：

- **多卡并行**：优化张量并行的调度与内核发射，降低 GPU 等待主机的开销；支持注意力头数不能被卡数整除时的切分（如 6 卡）。
- **长上下文**：在超长上下文下保持预填充与吐字速度的稳定，并降低计算缓冲的显存占用。
- **投机解码**：实现 DFlash2 投机解码的完整链路，优化采样与每轮开销，提升吐字速度。
- **计算内核**：移植 1Cat-vLLM 的部分 SM70 计算内核并适配 Q8_0 量化格式，同时进行算子融合。
- **平台适配**：针对 IBM AC922 的 NVLink 互联与 NUMA 内存架构进行适配。
- **多模态**：支持视觉模块（mmproj）在 GPU 上运行，处理图片、PDF 与视频。

---

## 测试数据

测试环境：
- 服务器：IBM Power AC922（2 × POWER9），6 × Tesla V100-SXM2-16GB（NVLink），CUDA 12.4。4 卡测试用 0、1、3、4 号卡（两颗 CPU 各两张）。
- 模型：Qwen3.8-27B，Q8_0 GGUF；投机解码用 DFlash2 草稿模型（F16），每轮起草 7 个。
- 启动参数：见“启动参数”一节的对应配置。
- 除“两路并发”和“多模态”外，每个用例都在全新启动的服务上测；这两节的用例在同一个服务里依次测。
- 速度在 1.0.0～1.0.2 上测得，显存峰值为 1.0.2 实测。
- 真实内容的吐字速度随接受率变化，部分表格同时给出每轮投机耗时。

### 4 卡，上下文 262144

| 输入 | 预填充 | 吐字 | 每卡显存峰值 |
|---|---:|---:|---:|
| 合成 209715 token（输出 512） | 1652 tok/s | 281.5 tok/s | 13.3 GB |
| 真实代码 209115 token（输出 1024） | 1646 tok/s | 每轮投机 28.3 ms | 13.3 GB |
| 合成 131072 token（输出 512） | 1997 tok/s | 310.4 tok/s | 13.3 GB |
| 真实代码 131188 token（输出 1024） | 1983 tok/s | 97.6 tok/s | 13.3 GB |
| 真实代码 16499 token（输出 1024） | 2570 tok/s | 123.5 tok/s | 13.3 GB |

### 4 卡，模型卡推荐的采样参数

真实代码，输出 1024：

| 模式 | 输入 | 预填充 | 吐字 | 接受率 | 每轮投机耗时 |
|---|---|---:|---:|---:|---:|
| 思考模式（T=1.0、top_p 0.95、top_k 20） | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| 思考模式 | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| 非思考模式（T=0.7、top_p 0.8、top_k 20、presence_penalty 1.5） | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| 非思考模式 | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

### 4 卡，批大小（`-ub`），上下文 262144

| `-ub` | 结果 |
|---|---|
| 2048 | 正常运行，每卡显存峰值 13.3 GB |
| 4096 | 正常运行，每卡显存峰值 14.5 GB；合成 209715 token（输出 512）：预填充 1799 tok/s，吐字 278.0 tok/s |
| 8192 | 显存不够，启动失败 |

### 6 卡，上下文 262144

| `-ub` | 输入 | 预填充 | 吐字 | 每轮投机耗时 | 每卡显存峰值 |
|---|---|---:|---:|---:|---:|
| 2048 | 合成 209715 token（输出 512） | 1720 tok/s | 272.9 tok/s | 29.3 ms | 10.2 GB |
| 2048 | 真实代码 209233 token（输出 1024） | 1719 tok/s | 84.3 tok/s | 29.1 ms | 10.2 GB |
| 2048 | 真实代码 16496 token（输出 1024） | 2592 tok/s | 135.3 tok/s | 22.6 ms | 10.2 GB |
| 4096 | 合成 209715 token（输出 512） | 1912 tok/s | 273.6 tok/s | 29.2 ms | 11.4 GB |
| 4096 | 真实代码 209229 token（输出 1024） | 1905 tok/s | 87.4 tok/s | 29.1 ms | 11.4 GB |
| 4096 | 真实代码 16497 token（输出 1024） | 2731 tok/s | 115.7 tok/s | 22.6 ms | 11.4 GB |
| 8192 | 合成 209715 token（输出 512） | 1912 tok/s | 273.3 tok/s | 29.2 ms | 13.8 GB |

### 6 卡，上下文 524288（YaRN）

| `-ub` | 输入 | 首字 | 预填充 | 吐字 | 每卡显存峰值 |
|---|---|---:|---:|---:|---:|
| 2048 | 合成 419430 token（输出 512） | 360 s | 1165 tok/s | 220.2 tok/s | 12.7 GB |
| 4096 | 合成 419430 token（输出 512） | 320 s | 1309 tok/s | 218.1 tok/s | 13.9 GB |

`-ub 8192` 显存不够，启动失败。只测了速度和显存，没有测 262144 以上的输出质量。YaRN 设置对所有长度都生效。

### 4 卡，两路并发

`-c 262144 -np 2`（每路 131072），同一个服务里依次测，每卡显存峰值 14.5～14.7 GB：

| 场景 | 首字 | 每路吐字 | 合计吐字 |
|---|---|---:|---:|
| 单路 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| 两路各 16K 同时请求 | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| 单路约 11.5 万 token | 55.8 s | 101.5 tok/s | 102 tok/s |
| 两路各约 11.5 万 token 同时请求 | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

`-c 524288 -np 2`（每路 262144）显存不够，启动失败。

### 多模态（图片 / PDF / 视频）

6 卡、上下文 524288（YaRN）、视觉模块放 GPU、思考模式，同一个服务里依次测：

| 输入 | token 数 | 首字 | 吐字 |
|---|---:|---:|---:|
| 图片（论文配图） | 3078 | 5.0 s | 136 tok/s |
| 图片（示意图） | 4069 | 5.6 s | 126 tok/s |
| PDF 前 6 页（每页一张图） | 3590 | 3.5 s | 151 tok/s |
| 视频 12 秒（幻灯片） | 23104 | 19.5 s | 164 tok/s |
| 视频 10 秒 | 6387 | 7.3 s | 160 tok/s |
| 20 万 token 文本 + 1 张图 | 206958 | 122 s | 91 tok/s |

- 同一会话在 20.8 万 token 之后追加 8030 token：首字 7.7 s（只处理新增部分）。
- 加载视觉模块时，纯文字合成 419430 token（输出 512）：预填充 1154 tok/s，吐字 217.9 tok/s；每卡显存峰值 0 号卡 13.6 GB、其余 12.7 GB。
- 4 卡、上下文 262144、视觉模块放 GPU：一张图的请求 7.6 s，PDF 前 6 页 5.5 s（都含生成回答）；0 号卡显存峰值约 14.7 GB。
- 视觉模块放 CPU（`--no-mmproj-offload`）时：同样两张图片首字 150 s、232 s，12 秒视频 649 s。

---

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/path/to/nccl/include -DNCCL_LIBRARY=/path/to/nccl/lib/libnccl.so.2
cmake --build build -j --target llama-server llama-quantize
```

多卡张量并行需要 NCCL。编译器用 GCC 12 或更新的版本。

GitHub 上每次推送都会自动按 sm_70 编译检查；发布版本时附带 x86_64 Linux 的预编译程序（CUDA 12，见 Releases 页面）。POWER9 等其他平台按上面的命令自行编译。

---

## 模型下载

测试用的是下面这些文件，视觉模块只在处理图片、视频时需要。**国内用户建议从魔搭（ModelScope）下载。**

| 用途 | 文件 | 魔搭 ModelScope | Hugging Face |
|---|---|---|---|
| 目标模型 | `Qwen3.8-27B-Q8_0.gguf`（约 29 GB） | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| 草稿模型（DFlash2） | `Qwen3.8-27B-DFlash2-BF16.gguf`（约 3.9 GB），下载后转成 F16 | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
| 视觉模块（可选） | `mmproj-F16.gguf`（约 0.9 GB） | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

目标模型下载后直接用。草稿模型官方只发布了 BF16、Q8_0、Q4_K_M 三种 GGUF，测试用的是 F16（V100 没有 BF16 硬件支持），需要用本项目编译出的 `llama-quantize` 转换一次。

先按上一节编译，然后一条命令完成下载和转换：

```bash
scripts/v100-get-models.sh            # 默认从魔搭下载到 ./models
scripts/v100-get-models.sh -s hf      # 改用 Hugging Face
```

脚本会下载目标模型和草稿模型（支持断点续传，已下载的文件自动跳过），再把草稿模型转成 `Qwen3.8-27B-DFlash2-F16.gguf`。`-d` 指定模型目录，`-q` 指定 `llama-quantize` 的路径，`-h` 查看用法。视觉模块不在脚本里，需要时从上表的仓库单独下载 `mmproj-F16.gguf`。

如果已经自己下载好了，只需要转换这一步：

```bash
./build/bin/llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-F16.gguf F16
```

---

## 启动参数

以下是测试数据对应的启动命令。`numactl --membind=0,8` 把主机内存限定在本机的两个 CPU 节点（AC922 把 GPU 显存上线成 NUMA 节点，不限定时页缓存可能占用显存），其他机器用 `numactl -H` 查看节点编号。

### 4 卡，上下文 262144

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 卡，上下文 262144

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 卡，上下文 524288（YaRN）

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

### 多模态：6 卡，上下文 524288（YaRN），视觉模块放 GPU

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

### 多模态：4 卡，上下文 262144，视觉模块放 GPU

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

多模态请求走 OpenAI 兼容的 `/v1/chat/completions`：图片用 `image_url`，视频用 `input_video`（或 `video_url`），视频默认每秒取 4 帧。PDF 由客户端把每页渲染成图片后发送（80 dpi 时一页约 600 token）。

### 参数说明

| 参数 | 说明 |
|---|---|
| `-ngl 999` | 所有层放 GPU |
| `--split-mode tensor --tensor-split 1,…,1` | 张量并行，各卡均分 |
| `-c` | 上下文长度 |
| `-np 1` | 单并发 |
| `-fa on` | Flash Attention（SM70 注意力内核需要） |
| `-ctk q8_0 -ctv q8_0` | KV 缓存用 8 位 |
| `-b 2048 -ub 2048` | 批大小，测试数据除注明外都按 2048 |
| `--model-draft …`、`--spec-type draft-dflash`、`--spec-draft-n-max 7` | DFlash2 投机解码，每轮起草 7 个 |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` | 用 YaRN 把上下文扩到 524288（模型原生 262144） |
| `--override-kv qwen35.context_length=int:524288` | 允许单路上下文超过模型训练长度 |
| `--mmproj` | 视觉模块 |
| `--video-ffmpeg-dir` | `ffmpeg`、`ffprobe` 所在目录，处理视频时需要 |
| `--no-mmproj-offload` | 视觉模块放 CPU（默认放第一张 GPU，权重约 0.9 GB，计算缓冲最多约 1.2 GB） |
| `--temp`、`--top-p`、`--top-k`、`--min-p`、`--chat-template-kwargs` | 默认采样参数与思考模式开关（上面是模型卡的思考模式参数） |
| `--metrics` | Prometheus 指标（可选） |

采样：`top_k ≤ 64` 且只用 top-k、top-p、min-p、temperature 时，在设备端取 top-k 候选；使用 penalties、DRY 等采样器或 `top_k` 更大时，在主机上对全词表采样（每轮耗时见“模型卡推荐的采样参数”表中的非思考模式）。

---

## 下一步

- **多卡切分**：Qwen3.8 的注意力 KV 头数为 4，6 卡张量并行时每个注意力层有 2 张 GPU 未分配注意力头，6 卡的每轮投机耗时比 4 卡高约 3%。后续将优化此类非均分情况下的切分与卡间通信，提升 6 卡的预填充与吐字速度。
- **多路并发**：当前主要针对单路请求优化，多路并发时预填充与吐字相互影响。后续将优化多路并发的调度与吞吐。

---

## 交流群

扫描下方二维码添加微信，拉你进交流群，与群友讨论更多使用方法。

<img src="media/v100-wechat.jpg" width="260" alt="微信二维码">

---

## 贡献

- **仓库作者**：发起项目，提供硬件与测试环境，确定目标与取舍，参与测试。
- **Claude**（Anthropic，Claude Opus 5.5，通过 Claude Code）：方案设计，测量与分析，服务器上的编译、测试与压测，代码审核，提交与文档。
- **DeepSeek V4.1 Flash**：2026-09-26 晚起承担大部分代码实现，以及源码调研、设计初稿和诊断脚本，约 140 次任务。
- **小米 MiMo v2.6-pro**：项目初期（2026-09-26）的代码调研与第一版实现，约 26 次任务。

感谢以上所有项目和参与者。

---

## 许可

继承 llama.cpp 的 MIT 许可（见 `LICENSE`）。移植的代码保留原许可，出处与许可见各文件头。
