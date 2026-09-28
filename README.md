# llama.cpp-v100

面向 NVIDIA V100（SM70）的 llama.cpp 专项优化版本。当前版本 **1.0.1**。

---

## 致谢

**特别感谢 [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。** 它在 V100 上的工程实践给了本项目很多参考，部分 SM70 计算内核移植自它。

同样感谢：

- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)**：本项目的基础框架。
- **[vLLM](https://github.com/vllm-project/vllm)**：投机解码等方面的设计给了本项目很多借鉴。

---

## 简介

本项目基于 llama.cpp（2026-09-26 的 master `08618ff8e`，已包含上游 v0.5.0）改造，针对 V100 做专项优化。重点是 Qwen3.8-27B（Q8_0）配合 DFlash2 投机解码，在 4 卡张量并行下的长上下文预填充和吐字速度。

除部分 SM70 计算内核移植自 1Cat-vLLM 外，多卡运行时与调度、长上下文注意力、投机解码链路、大部分算子融合和平台适配都是本项目自研。

## 专项优化方向

围绕 V100 的硬件特点，从内核到运行时做了整体优化：

- **多卡并行（自研）**：重做 4 卡张量并行的调度与发射，让 GPU 少等待。
- **长上下文（自研）**：让预填充和吐字在超长上下文下保持稳定的速度。
- **投机解码（自研）**：打通 DFlash2 投机解码的完整链路，并优化采样和每轮开销，提升吐字速度。
- **计算内核**：部分关键算子移植自 1Cat-vLLM 并适配 Q8_0；大量算子融合为自研。
- **平台适配（自研）**：针对 IBM AC922 的多卡互联和内存特点做了适配。

---

## 性能

测试环境：
- 服务器：IBM Power AC922（2 × POWER9），4 × Tesla V100-SXM2-16GB（NVLink），CUDA 12.4。
- 模型：Qwen3.8-27B，Q8_0 GGUF；投机解码用 DFlash2 草稿模型，每轮起草 7 个。
- 参数：见下文“推荐启动参数”。
- 每个用例都在全新启动的服务上测。

| 输入 | 预填充 | 吐字 | 每卡显存峰值 |
|---|---:|---:|---:|
| 合成 209715 token（输出 512） | 1652 tok/s | 281.5 tok/s | 14.3 GB |
| 真实代码 209115 token（输出 1024） | 1646 tok/s | 每轮投机 28.3 ms | 14.3 GB |
| 合成 131072 token（输出 512） | 1997 tok/s | 310.4 tok/s | 14.3 GB |
| 真实代码 131188 token（输出 1024） | 1983 tok/s | 97.6 tok/s | 14.3 GB |
| 真实代码 16499 token（输出 1024） | 2570 tok/s | 123.5 tok/s | 14.3 GB |

真实内容的吐字速度随接受率波动较大，所以同时给出每轮投机耗时。

### 官方推荐的采样参数

按模型卡推荐的两套采样参数实测（真实代码，输出 1024）：

| 模式 | 输入 | 预填充 | 吐字 | 接受率 | 每轮投机耗时 |
|---|---|---:|---:|---:|---:|
| 思考模式（T=1.0、top_p 0.95、top_k 20） | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| 思考模式 | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| 非思考模式（T=0.7、top_p 0.8、top_k 20、presence_penalty 1.5） | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| 非思考模式 | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

- 思考模式走设备端采样的快速路径。每轮耗时与 T=0.6 时相同，吐字速度的差别只来自接受率。
- 非思考模式的 `presence_penalty` 会让采样退回全词表，每轮慢约 4.5 ms（17~21%）；这类内容接受率更高，所以吐字速度相近。

### 批大小（`-ub`）

4 卡、上下文 262144 时：

| `-ub` | 结果 |
|---|---|
| 2048 | 正常运行，每卡显存峰值 14.3 GB |
| 4096 | 显存不够，启动失败 |
| 8192 | 显存不够，启动失败 |

4 卡开满 256K 上下文时，2048 是上限；6 卡每卡放的权重更少，见下一节。

### 6 卡

从 1.0.1 起可以用 6 卡运行（`--tensor-split 1,1,1,1,1,1`，其余参数同上）。千问的注意力 KV 头只有 4 个，6 卡时每个注意力层有 2 张卡分不到头；这部分切分还没有优化，是下一步的方向。下面是目前的情况（上下文 262144，每个用例都在全新启动的服务上测）：

| `-ub` | 输入 | 预填充 | 吐字 | 每轮投机耗时 | 每卡显存峰值 |
|---|---|---:|---:|---:|---:|
| 2048 | 合成 209715 token（输出 512） | 1720 tok/s | 272.9 tok/s | 29.3 ms | 11.3 GB |
| 2048 | 真实代码 209233 token（输出 1024） | 1719 tok/s | 84.3 tok/s | 29.1 ms | 11.3 GB |
| 2048 | 真实代码 16496 token（输出 1024） | 2592 tok/s | 135.3 tok/s | 22.6 ms | 11.3 GB |
| 4096 | 合成 209715 token（输出 512） | 1912 tok/s | 273.6 tok/s | 29.2 ms | 13.5 GB |
| 4096 | 真实代码 209229 token（输出 1024） | 1905 tok/s | 87.4 tok/s | 29.1 ms | 13.5 GB |
| 4096 | 真实代码 16497 token（输出 1024） | 2731 tok/s | 115.7 tok/s | 22.6 ms | 13.5 GB |

- 与 4 卡相比：200K 预填充在 `-ub 2048` 时快约 4%，`-ub 4096` 时快约 16%；吐字每轮慢约 3%（卡多了通信增加，注意力层有 2 张卡空闲）。
- `-ub 6144` 也能启动（启动后每卡约 15.0 GB），`-ub 8192` 显存不够。

**单并发 512K 上下文**：6 卡时每卡显存有富余，可以开到 512K。千问原生支持 262,144 token，更长需要按模型卡的建议用 YaRN 扩展，同时放开服务端“单路上下文不超过训练长度”的限制：

```bash
-c 524288 -np 1 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288
```

实测合成 419430 token（512K 的 80%，输出 512）：首字 360 s，预填充 1165 tok/s，吐字 220.2 tok/s，每卡显存峰值 14.8 GB；`-ub 4096` 时显存不够。这里只测了速度和显存，没有评估 256K 以上的输出质量；YaRN 会作用于所有长度，不需要超过 256K 时不建议打开。

### 两路并发

每路 256K（`-c 524288 -np 2`）显存不够，启动失败。每路 128K（`-c 262144 -np 2`）可以运行，每卡显存峰值 14.5~14.7 GB：

| 场景 | 首字 | 每路吐字 | 合计吐字 |
|---|---|---:|---:|
| 单路 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| 两路各 16K 同时请求 | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| 单路约 11.5 万 token | 55.8 s | 101.5 tok/s | 102 tok/s |
| 两路各约 11.5 万 token 同时请求 | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

目前两路并发比单路还慢：
- 预填充基本排队，第二路要等第一路预填充完。
- 一路吐字时如果另一路正在预填充，吐字那路每轮都要等对方的预填充块算完，速度明显下降。

所以目前推荐单并发（`-np 1`）；并发优化列在“下一步”。

---

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/path/to/nccl/include -DNCCL_LIBRARY=/path/to/nccl/lib/libnccl.so.2
cmake --build build -j --target llama-server llama-quantize
```

多卡张量并行需要 NCCL。编译器用 GCC 12 或更新的版本。

GitHub 上每次推送都会自动按 sm_70 编译检查；发布版本时附带 x86_64 Linux 的预编译程序（CUDA 12，见 Releases 页面）。POWER9 等其他平台请按上面的命令自行编译。

---

## 模型下载

本项目实测用的是下面两个模型。**国内用户建议从魔搭（ModelScope）下载**，速度快、连接稳定。

| 用途 | 文件 | 魔搭 ModelScope | Hugging Face |
|---|---|---|---|
| 目标模型 | `Qwen3.8-27B-Q8_0.gguf`（约 29 GB） | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| 草稿模型（DFlash2） | `Qwen3.8-27B-DFlash2-BF16.gguf`（约 3.9 GB），下载后转成 F16 | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |

目标模型下载后直接用。草稿模型官方只发布了 BF16、Q8_0、Q4_K_M 三种 GGUF，本项目实测用的是 F16（V100 没有 BF16 硬件支持），需要用本项目编译出的 `llama-quantize` 转换一次。

先按上一节编译，然后一条命令完成下载和转换：

```bash
scripts/v100-get-models.sh            # 默认从魔搭下载到 ./models
scripts/v100-get-models.sh -s hf      # 改用 Hugging Face
```

脚本会下载这两个 GGUF（支持断点续传，已下载的文件自动跳过），再把草稿模型转成 `Qwen3.8-27B-DFlash2-F16.gguf`。`-d` 指定模型目录，`-q` 指定 `llama-quantize` 的路径，`-h` 查看用法。

如果已经自己下载好了，只需要转换这一步：

```bash
./build/bin/llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-F16.gguf F16
```

---

## 推荐启动参数

以下就是本项目测试所用的参数：

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

| 参数 | 说明 |
|---|---|
| `-ngl 999` | 所有层都放 GPU |
| `--split-mode tensor --tensor-split 1,1,1,1` | 4 卡张量并行，均分 |
| `-c 262144` | 上下文 256K，实测每卡显存峰值 14.3 GB；显存更紧时可以调小 |
| `-np 1` | 单并发。目前只针对单并发优化，多路并发还没有优化（见“下一步”） |
| `-fa on` | 开启 Flash Attention，SM70 注意力内核依赖它 |
| `-ctk q8_0 -ctv q8_0` | KV 缓存用 8 位 |
| `-b 2048 -ub 2048` | 批大小 2048，预填充按这个大小调优；开满 256K 上下文时，16 GB 的 V100 放不下更大的值 |
| `--model-draft ...`、`--spec-type draft-dflash`、`--spec-draft-n-max 7` | DFlash2 投机解码，每轮起草 7 个 |
| `--metrics` | 可选，开启 Prometheus 指标 |

**AC922 平台注意事项：**
- **限定 CPU 内存节点**：AC922 会把 GPU 显存上线成 NUMA 节点，页缓存可能落到显存上，导致 CUDA 显存不足。用 `numactl --membind=<CPU 节点>` 限定内存节点，本机是 0 和 8，用 `numactl -H` 查看。
- **选卡**：4 卡时两颗 CPU 各选两张，本机是 0,1,3,4。

**采样参数：** `top_k ≤ 64` 且只用 top-k、top-p、min-p、temperature 时，走设备端 top-k + 稀疏拒绝采样的快速路径。Qwen 推荐的 `top_k=20` 正好满足。使用 penalties、DRY 等采样器，或者 `top_k` 更大时，会退回全词表采样，吐字变慢。例如官方非思考模式的 `presence_penalty=1.5`，每轮慢约 20%。

---

## 下一步

- **多卡切分优化**：6 卡已能正常运行（1.0.1，现状见“性能 / 6 卡”）。千问模型的部分维度（如注意力头数）不能被 6 整除，目前按头切分时每个注意力层有 2 张卡空闲，吐字每轮比 4 卡慢约 3%。接下来优化这类不能均分时的切分效率，让多出来的卡真正带来提速。
- **并发优化**：目前针对单并发优化，多路并发时预填充和吐字会互相拖慢。后续优化多路并发的调度与吞吐。

---

## 贡献

- **仓库作者与 Claude 共同担任总调度与决策。**
  - **仓库作者**：
    - 提出项目，提供硬件与测试环境。
    - 定下目标和关键方向：先单卡后多卡；内核直接移植已验证的实现、运行时按 llama.cpp 架构自己设计；8 位权重；以 200K 上下文为标准线；调参带来的提速不计入成果。
    - 做关键取舍：哪些优化做、哪些不做。
    - 在测试中发现问题、指导方向，例如多卡利用率不均衡、长上下文吐字变慢。
    - 带来早先 V100 项目的经验和部分实现。
  - **Claude**（Anthropic，Claude Opus 5.5，通过 Claude Code）：
    - 技术方案与设计；
    - 测量与根因分析：nvprof、perf、自写探针；
    - 服务器上的全部编译、测试与压测；
    - 审核每一处改动；
    - 提交与文档。
- **DeepSeek V4.1 Flash**：2026-09-26 晚起承担大部分代码实现，约 124 次任务。
  - 实现举例：区间 mask、解码图跨请求保留、meta 后端的临时缓冲与计划淘汰、n_kv 粒度自适应。
  - 也做源码调研、设计初稿和诊断脚本。
  - 响应快，质量稳定。
- **小米 MiMo v2.6-pro**：项目初期（2026-09-26）的主力，约 26 次任务。
  - 前期代码调研。
  - 第一版实现：grouped 验证注意力、DFlash2 拒绝采样。
  - 词表分片 top-k 的设计与实现，DFlash2 草稿链路一致性审计。
  - 后期承担并行调研，例如解码图保留、设备端整轮投机。

感谢以上所有项目和参与者。

---

## 许可

继承 llama.cpp 的 MIT 许可（见 `LICENSE`）。移植的代码保留原许可，出处与许可见各文件头。
