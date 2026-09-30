# llama.cpp-v100

**简体中文** | [English](README.en.md)

面向 NVIDIA V100（SM70）的 llama.cpp 修改版。当前版本 **1.0.6**，各版本的更新内容见 [CHANGELOG.md](CHANGELOG.md)。

---

## 本版更新（1.0.6）

1.0.6 提升 2、3、5、6 卡的预填充速度和 Q4 配置的吐字速度；4 卡 Q8 的速度、显存与输出都与 1.0.5 相同。

- **GDN 分块预填充支持任意每卡头数**：此前只有 4 卡（每卡 12 个 GDN 头）走分块算法，其他卡数退回逐 token 递归。现在 2～6 卡都走分块。6 卡 200K 合成预填充 1710 → 1801 tok/s（+5.4%），6 卡真实 16K 2568 → 2774 tok/s（+8%），2 卡 Q4 +8～10%，3 卡 Q8 +6%。困惑度不变（6 卡 1.6558 → 1.6555，3 卡 1.6554 → 1.6555）。
- **Q4_K 权重的小批量矩阵乘内核**：投机验证（每轮 8 个 token）不再走通用路径。加载时把 Q4_K 权重原地重排，不额外占显存。Q4 配置每轮投机耗时降低 5～9%（例如 2 卡 262144：39.1 → 35.8 ms，4 卡 524288：28.4 → 26.7 ms）。Q4 配置的输出会有细微变化，困惑度不变（2 卡 1.7321 → 1.7320）。
- **底线**：与 1.0.0 同一时段交替对照，4 卡 200K 预填充 1641.0 对 1641.6 tok/s，每轮投机耗时持平；并发与 1.0.4 持平。

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

- **多卡并行**：优化张量并行的调度与内核发射，降低 GPU 等待主机的开销；卡间 all-reduce 按 NVLink 拓扑分层进行；支持对注意力头数不能被卡数整除时的切分优化（如 6 卡）。
- **长上下文**：在超长上下文下保持预填充与吐字速度的稳定，并降低计算缓冲的显存占用。
- **投机解码**：实现 DFlash2 投机解码的完整链路，优化采样与每轮开销，提升吐字速度。
- **计算内核**：移植 1Cat-vLLM 的部分 SM70 计算内核并适配 Q8_0 量化格式，同时进行算子融合。
- **平台适配**：针对 IBM AC922 的 NVLink 互联与 NUMA 内存架构进行适配。
- **多路并发**：多路请求时吐字优先、限制提示词处理占用的时间，多路的解码批次、投机解码与显存预留都按路数优化。
- **多模态**：支持视觉模块（mmproj）在 GPU 上运行，处理图片、PDF 与视频。

---

## 测试数据

测试环境：
- 服务器：IBM Power AC922（2 × POWER9），6 × Tesla V100-SXM2-16GB（NVLink），CUDA 12.4。4 卡测试用 0、1、3、4 号卡（两颗 CPU 各两张）。
- 模型：Qwen3.8-27B，Q8_0 GGUF；投机解码用 DFlash2 草稿模型（F16），每轮起草 7 个。
- 启动参数：见“启动参数”一节的对应配置。
- 除“多路并发”和“多模态”外，每个用例都在全新启动的服务上测；这两节的用例在同一个服务里依次测。
- “卡数与量化适配”的单路表为 1.0.6 实测，两路表为 1.0.5 实测（2026-09-30）；其余各节为 1.0.4 实测（2026-09-29～30）。4 卡 Q8 在 1.0.5、1.0.6 上输出与 1.0.4 逐字相同、速度不变；6 卡从 1.0.6 起预填充更快（见“本版更新”），6 卡各表的吐字与显存不变。
- 真实内容的吐字速度随接受率变化，部分表格同时给出每轮投机耗时。

### 4 卡，上下文 262144

| 输入 | 预填充 | 吐字 | 每轮投机耗时 | 每卡显存峰值 |
|---|---:|---:|---:|---:|
| 合成 209715 token（输出 512） | 1638 tok/s | 278.9 tok/s | 28.6 ms | 13.3 GB |
| 真实代码 209228 token（输出 1024） | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| 合成 131072 token（输出 512） | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| 真实代码 131184 token（输出 1024） | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| 真实代码 16500 token（输出 1024） | 2547 tok/s | 119.5 tok/s | 22.0 ms | 13.3 GB |

### 4 卡，模型卡推荐的采样参数

真实代码，输出 1024：

| 模式 | 输入 | 预填充 | 吐字 | 接受率 | 每轮投机耗时 |
|---|---|---:|---:|---:|---:|
| 思考模式（T=1.0、top_p 0.95、top_k 20） | 16K | 2552 tok/s | 116.5 tok/s | 0.22 | 22.0 ms |
| 思考模式 | 128K | 1963 tok/s | 97.6 tok/s | 0.22 | 25.9 ms |
| 非思考模式（T=0.7、top_p 0.8、top_k 20、presence_penalty 1.5） | 16K | 2620 tok/s | 105.4 tok/s | 0.25 | 26.0 ms |
| 非思考模式 | 128K | 1971 tok/s | 89.1 tok/s | 0.24 | 29.8 ms |

### 4 卡，批大小（`-ub`），上下文 262144

| `-ub` | 结果 |
|---|---|
| 2048 | 正常运行，每卡显存峰值 13.3 GB |
| 4096 | 正常运行，每卡显存峰值 14.5 GB；合成 209715 token（输出 512）：预填充 1801 tok/s，吐字 277.1 tok/s |
| 8192 | 显存不够，启动失败 |

### 6 卡，上下文 262144

| `-ub` | 输入 | 预填充 | 吐字 | 每轮投机耗时 | 每卡显存峰值 |
|---|---|---:|---:|---:|---:|
| 2048 | 合成 209715 token（输出 512） | 1708 tok/s | 293.1 tok/s | 27.2 ms | 10.2 GB |
| 2048 | 真实代码 209227 token（输出 1024） | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | 真实代码 16494 token（输出 1024） | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | 合成 209715 token（输出 512） | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | 真实代码 209236 token（输出 1024） | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | 真实代码 16500 token（输出 1024） | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | 合成 209715 token（输出 512） | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

### 6 卡，上下文 524288（YaRN）

| `-ub` | 输入 | 首字 | 预填充 | 吐字 | 每卡显存峰值 |
|---|---|---:|---:|---:|---:|
| 2048 | 合成 419430 token（输出 512） | 363 s | 1154 tok/s | 233.1 tok/s | 12.8 GB |
| 4096 | 合成 419430 token（输出 512） | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

`-ub 8192` 显存不够，启动失败。只测了速度和显存，没有测 262144 以上的输出质量。YaRN 设置对所有长度都生效。

### 多路并发

真实代码，思考模式（T=0.6），每路输出 1024，默认 `--prefill-pace 30`。同一个服务里依次测，“同时请求”指各路同时发出。
- 首字从发出请求算起；每路吐字 = 这一路从第一个字到结束的速度。
- 合计吐字 = 所有路的输出总数 ÷（最早的第一个字 → 最后一路结束）。
- 全部完成 = 从发出请求到最后一路结束。

**4 卡，`-c 262144 -np 2`（每路 131072）**，每卡启动显存 13.5 GB，峰值 13.7 GB：

| 场景 | 首字 | 每路吐字 | 合计吐字 | 全部完成 |
|---|---|---:|---:|---:|
| 单路 16K | 6.5 s | 115.7 tok/s | 116 tok/s | 15.4 s |
| 两路各 16K 同时请求 | 8.1 s / 22.7 s | 81.4 / 111.4 tok/s | 86 tok/s | 31.9 s |
| 单路约 11.5 万 token | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| 两路各约 11.5 万 token 同时请求 | 58.0 s / 122.6 s | 70.2 / 81.4 tok/s | 27 tok/s | 135.2 s |

**4 卡，`-c 262144 -np 4`（每路 65536）**，每卡启动显存 14.2 GB，峰值 14.3 GB：

| 场景 | 首字 | 每路吐字 | 合计吐字 | 全部完成 |
|---|---|---:|---:|---:|
| 单路 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| 四路各 16K 同时请求 | 8.5 / 27.1 / 43.6 / 57.5 s | 64.0 / 69.9 / 81.9 / 124.4 tok/s | 72 tok/s | 65.7 s |

**6 卡，`-c 524288 -np 2`（YaRN，每路 262144）**，每卡启动显存 12.4～12.8 GB，峰值 12.7～13.0 GB：

| 场景 | 首字 | 每路吐字 | 合计吐字 | 全部完成 |
|---|---|---:|---:|---:|
| 单路 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| 两路各 16K 同时请求 | 8.0 s / 21.9 s | 85.4 / 116.1 tok/s | 90 tok/s | 30.7 s |
| 两路各约 11.5 万 token 同时请求 | 55.3 s / 118.8 s | 66.0 / 90.5 tok/s | 27 tok/s | 130.1 s |

与 1.0.3 对比（4 卡，同一测试）：

| 场景 | 1.0.3 | 1.0.4 |
|---|---|---|
| `-np 2` 两路各 16K：首字 | 9.2 s / 16.7 s | 8.1 s / 22.7 s |
| `-np 2` 两路各 16K：每路吐字 | 45.7 / 58.8 tok/s | 81.4 / 111.4 tok/s |
| `-np 2` 两路各 16K：全部完成 | 34.1 s | 31.9 s |
| `-np 2` 两路各 11.5 万：首字 | 59.6 s / 133.0 s | 58.0 s / 122.6 s |
| `-np 2` 两路各 11.5 万：每路吐字 | 11.4 / 58.9 tok/s | 70.2 / 81.4 tok/s |
| `-np 2` 两路各 11.5 万：全部完成 | 150.6 s | 135.2 s |
| `-np 2` 每卡启动显存 | 14.0 GB | 13.5 GB |
| `-np 4`（每路 64K） | 启动即显存不足 | 可以运行 |

说明：
- 提示词处理是按顺序排队的，多路同时请求时后面几路的首字要等前面的处理完。所有请求全部完成的总时间与依次处理差不多；多路的好处是先开始吐字的请求不会被后来的长提示词卡住，后来的请求也不必等前一路全部吐完才开始。
- `--prefill-pace` 调大，后来请求的首字更快，但正在吐字的请求更慢；设为 0 或 100 等于不限流（与 1.0.3 相同）。16K 这类短提示词时，默认 30 会让第二路的首字晚几秒（上表 16.7 → 22.7 s）。
- 单路长上下文（例如一个 Agent 开满 256K）仍建议 `-np 1`，多路时每路的上下文是总上下文的几分之一。

### 卡数与量化适配

单路表为 1.0.6 实测，两路表为 1.0.5 实测。两套配置：
- **Q8**：主模型 Q8_0、KV 缓存 q8_0（`-ctk q8_0 -ctv q8_0`）；
- **Q4**：主模型 UD-Q4_K_M、KV 缓存 q4_0（`-ctk q4_0 -ctv q4_0`）。

草稿模型默认用 DFlash2 F16；2 卡 Q4 要开到 262144 时改用 DFlash2 Q4_K_M（F16 草稿时最高 131072）。卡号：2 卡 0、1，3 卡 0、1、2，4 卡 0、1、3、4，5 卡 0～4，6 卡 0～5。每种组合从 524288 起逐档往下试，下表是能启动的最高上下文；524288 需要 YaRN（见“启动参数”）。

**单路**（真实代码 16K 输入、输出 1024；合成输入为上下文的 80%，输出 512）：

| 卡数 | 配置 | 上下文 | 真实 16K：预填充 / 吐字 / 每轮投机 | 合成长输入：token / 首字 / 预填充 / 吐字 | 每卡显存峰值 |
|---|---|---:|---|---|---:|
| 2 | Q4（草稿 Q4_K_M） | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4（草稿 F16） | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2549 / 131.7 tok/s / 22.0 ms | 209715 / 128 s / 1641 / 277.8 tok/s | 13.3 GB |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 351 s / 1197 / 232.2 tok/s | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

- 2 卡请用 Q4 配置：Q8_0 主模型约 29 GB，2 张 16 GB 卡放不下。
- 3 卡 Q8 的 262144 显存不够（换 Q4_K_M 草稿也不够），最高 131072。
- Q4 配置的吐字仍比 Q8 慢：Q4_K 的小批量内核目前只覆盖单个矩阵乘，gate/up 融合与多权重融合还走通用路径。

**两路并发**（`-np 2`，每路上下文 = 上下文 ÷ 2；1 路 16K 与 2 路各 16K 同时请求，输出 1024）：

| 卡数 | 配置 | 上下文（每路） | 单路 16K：首字 / 吐字 | 两路各 16K 同时：首字 | 每路吐字 | 每卡显存峰值 |
|---|---|---:|---|---|---|---:|
| 2 | Q4 | 131072（65536） | 10.0 s / 67.2 tok/s | 12.5 s / 37.2 s | 44.6 / 64.7 tok/s | 14.4 GB |
| 3 | Q8 | 131072（65536） | 7.3 s / 74.1 tok/s | 9.1 s / 26.4 s | 64.4 / 103.8 tok/s | 15.2 GB |
| 3 | Q4 | 524288（262144） | 7.6 s / 82.0 tok/s | 9.5 s / 28.7 s | 57.5 / 62.8 tok/s | 14.7 GB |
| 4 | Q8 | 262144（131072） | 6.5 s / 117.0 tok/s | 8.2 s / 23.9 s | 70.6 / 89.6 tok/s | 13.7 GB |
| 4 | Q4 | 524288（262144） | 6.7 s / 78.9 tok/s | 8.5 s / 24.5 s | 69.4 / 87.9 tok/s | 12.0 GB |
| 5 | Q8 | 524288（262144） | 6.7 s / 109.2 tok/s | 8.6 s / 22.8 s | 84.2 / 121.5 tok/s | 14.5 GB |
| 5 | Q4 | 524288（262144） | 6.8 s / 76.5 tok/s | 8.4 s / 26.4 s | 61.4 / 98.2 tok/s | 10.6 GB |
| 6 | Q8 | 524288（262144） | 6.4 s / 117.7 tok/s | 8.0 s / 23.1 s | 73.7 / 121.4 tok/s | 13.0 GB |
| 6 | Q4 | 524288（262144） | 6.5 s / 81.9 tok/s | 8.0 s / 24.8 s | 66.1 / 78.7 tok/s | 9.7 GB |

真实内容的吐字随接受率波动较大（单次测量），比较引擎快慢请看单路表的“每轮投机”。

### 多模态（图片 / PDF / 视频）

6 卡、上下文 524288（YaRN）、视觉模块放 GPU、思考模式，同一个服务里依次测：

| 输入 | token 数 | 提示词处理 | 吐字 |
|---|---:|---:|---:|
| 图片（论文配图） | 3078 | 5.0 s | 182 tok/s |
| 图片（示意图） | 4069 | 5.7 s | 163 tok/s |
| PDF 前 6 页（每页一张图） | 3590 | 3.5 s | 190 tok/s |
| 视频 12 秒（幻灯片） | 23104 | 19.7 s | 150 tok/s |
| 视频 10 秒 | 6387 | 7.3 s | 131 tok/s |
| 20 万 token 文本 + 1 张图 | 206958 | 122.7 s | 87 tok/s |

- 加载视觉模块时，纯文字合成 419430 token（输出 512）：预填充 1154 tok/s，吐字 233.3 tok/s；每卡显存峰值 0 号卡 13.6 GB、其余 12.7 GB。
- 4 卡、上下文 262144、视觉模块放 GPU：图片提示词处理 4.3 s，PDF 前 6 页 3.2 s；0 号卡显存峰值 14.8 GB。
- 同一会话在 20.9 万 token 之后追加 8034 token：首字 7.7 s（只处理新增部分；1.0.3 实测）。
- 视觉模块放 CPU（`--no-mmproj-offload`，1.0.1 实测）时：同样两张图片首字 150 s、232 s，12 秒视频 649 s。

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
| 目标模型（Q4 配置） | `Qwen3.8-27B-UD-Q4_K_M.gguf`（约 16.5 GB） | [unsloth/Qwen3.8-27B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| 草稿模型（显存紧张时） | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（约 1.1 GB），直接用 | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
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

### 多路并发

在上面单路命令的基础上改 `-np`，总上下文按路数平分：

```bash
# 4 卡两路，每路 131072
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 2 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --prefill-pace 30 \
  --host 127.0.0.1 --port 8080 --metrics

# 4 卡四路，每路 65536：把上面的 -np 2 换成 -np 4

# 6 卡两路，每路 262144（YaRN）
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

`--prefill-pace 30` 是默认值，写出来只为说明，可以省略。

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

### 其他卡数与 Q4 配置

在上面命令的基础上改这几处：
- 卡数：`CUDA_VISIBLE_DEVICES` 与 `--tensor-split`（几张卡就写几个 1）；
- Q4 配置：`-m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf`，`-ctk q4_0 -ctv q4_0`；
- 上下文：按“卡数与量化适配”表选，524288 加上 YaRN 那三个参数；
- 2 卡 Q4 开 262144：`--model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`。

例如 2 卡 Q4、上下文 262144：

```bash
CUDA_VISIBLE_DEVICES=0,1 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 262144 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `-ngl 999` | 所有层放 GPU |
| `--split-mode tensor --tensor-split 1,…,1` | 张量并行，各卡均分 |
| `-c` | 上下文长度 |
| `-np` | 并发路数，每路上下文 = `-c` ÷ 路数 |
| `--prefill-pace` | 多路时，有请求在吐字，其他请求的提示词处理最多占用的时间百分比（0～100，默认 30；0 或 100 为不限流） |
| `-fa on` | Flash Attention（SM70 注意力内核需要） |
| `-ctk q8_0 -ctv q8_0` | KV 缓存用 8 位；`-ctk q4_0 -ctv q4_0` 为 4 位，显存减半 |
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

- **多卡切分**：Qwen3.8 的注意力 KV 头数为 4，3 卡时注意力头按 6/6/12 分配（一张卡的注意力计算量是另两张的两倍），5 卡、6 卡时每个注意力层有 1～2 张 GPU 未分配注意力头。后续将让相邻 GPU 共享 KV 头，使各卡的注意力计算量均衡。
- **Q4 配置**：Q4_K 的 gate/up 融合与多权重融合内核，以及 17～64 行的小批量内核。
- **长上下文**：预填充不再保存整份 f16 的 K/V 副本（省显存），长上下文吐字与追加预填充的速度。
- **多路并发**：多路的总吞吐仍与依次处理相当。后续将优化多路同时吐字时每轮的耗时，以及多路之间的前缀缓存复用。

---

## 交流群

扫描下方二维码添加微信，拉你进交流群，与群友讨论更多使用方法。

<img src="media/v100-wechat.jpg" width="260" alt="微信二维码">

---

## 贡献

- **仓库作者**：发起项目，提供硬件与测试环境，确定目标与取舍，参与测试。
- **Claude**（Anthropic，Claude Opus 5.5，通过 Claude Code）：方案设计，测量与分析，服务器上的编译、测试与压测，代码审核，提交与文档。
- **DeepSeek V4.1 Flash**：2026-09-26 晚起承担大部分代码实现，以及源码调研、设计初稿和诊断脚本，约 170 次任务。
- **小米 MiMo v2.6-pro**：项目初期（2026-09-26）的代码调研与第一版实现，以及之后的深度调研与方案设计，约 33 次任务。
- **[ATIVX928](https://github.com/ATIVX928)**（外部贡献者）：SM70 注意力内核支持 q4_0 KV 缓存、2 卡 internal all-reduce 修复（[PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1)，1.0.5 合入）。

感谢以上所有项目和参与者。

---

## 许可

继承 llama.cpp 的 MIT 许可（见 `LICENSE`）。移植的代码保留原许可，出处与许可见各文件头。
