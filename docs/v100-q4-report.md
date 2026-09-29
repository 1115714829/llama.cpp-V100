# V100 Q4 支持改造报告

分支 `v100-q4-support`，基于 1.0.3（`c30640202`）。目标：2 x V100-16GB 张量并行、
Q4_K_M 权重、K/V 均 Q4_0 缓存，并把 V100 专用优化扩展出 Q4 支持。

## 变更清单

| commit | 内容 |
|---|---|
| `c2dc083f5` | docs: 改造计划与 P0 基线（2-way TP、Q4_K_M、Q4_0 KV） |
| `be634fdd0` | ggml-cuda: 修复 2 卡 internal AllReduce 的 exact 与尾部处理 |
| `b0fc5c08f` | docs: Q4_0 KV 缓存质量评估 |
| `23cd84da8` | ggml-cuda: sm70 注意力内核接受 Q4_0 KV |
| `ea944928d` | ggml-cuda: skinny GEMM 流水线的 Q4_0 支持 |

### 代码变更说明

1. **2 卡 AllReduce（`allreduce.cu`、`ggml-cuda.cu`）**
   - `exact` 标志此前被 internal（copy-engine）路径忽略，大 F32 张量走 BF16 线路
     导致非位精确（实测 exact 用例 -0.116211 vs -0.11619）。现在转发到管线，
     exact 归约禁用 BF16 往返。
   - 删除 nbytes 16 字节倍数的硬 assert（内核本就处理尾部），改为放行，异常尺寸
     不再 abort。
   - 验证：`test-cuda-allreduce` 在 NCCL 与 internal 两种模式下均 ALL_OK，含
     CUDA graph 链式用例。2 rank 一律走平面 push/NCCL，不涉及 4/6 卡的分层归约。

2. **sm70 注意力 Q4_0 KV（`fattn-sm70-grouped.cu/.cuh`、`fattn-sm70-d256.cu`）**
   - `fattn-sm70-grouped`（解码/投机验证，n_q 2..16）：新增 Q4_0 反量化路径，
     8/16 token 两个变体各支持 9 种 K/V 类型组合。
   - `fattn-sm70-d256`（预填充，q >= 17）：`sm70_d256_kv_type_ok` 接受 Q4_0，
     range mask 部分镜像反量化新增 q4_0 行内核；全量镜像回退路径不变。
   - 部分/全量反量化改为按张量选择，顺带修复一个潜在 bug：K 为 Q8_0、V 为非 F16
     的混合类型时，V 会被错误地用 q8_0 行内核处理。
   - q4_0 块格式陷阱：32 个值按半区存放（qs[0..15] 低半字节为值 0..15，高半字节为
     值 16..31），不是字节内交错（见 `ggml/src/ggml-cpu/quants.c` 的
     vec_dot_q4_0_q8_0 配对）。

3. **skinny GEMM 流水线 Q4_0（`q8-skinny.cu`）**
   - 重排、小 M GEMM、融合 SwiGLU、多权重合批、to_f16 回退全部按码流格式模板化，
     新增 Q4_0 权重支持。QPN8 执行布局不变：一个 (group, lane) 码记录为 16 字节
     int8 或 8 字节 nibble 打包，解码出相同的 half2 权重对。
   - 多权重内核的窄行 dot 路径直接读 Q4_0 块。
   - 重排张量带按类型的 marker；融合入口拒绝混合类型。

## Q4_0 KV 质量（P5.1）

语料 300 KB，2-way TP，Q4_K_M 权重，`-fa on`：

| KV 缓存 | PPL ctx=2048 | PPL ctx=8192 |
|---|---:|---:|
| q8_0 / q8_0 | 2.7734 | 2.6017 |
| q4_0 / q8_0 | 2.7765 | 2.6024 |
| q4_0 / q4_0 | 2.7792 | 2.6050 |

针检索测试（词埋在上下文 50% 处，贪心解码）：约 35K 与约 145K token 两档，
q8_0 与 q4_0 全部正确召回。结论：K=q4_0 / V=q4_0 可直接使用，无需混合降级。

## sm70 Q4_0 KV 性能（P3）

Q4_0 KV 预填充此前走通用路径（K/V 反量化到 F16 staging），改造后走 sm70 内核：

| 上下文 | 改造前 q4_0 KV | 改造后 q4_0 KV | q8_0 KV 参考 |
|---|---:|---:|---:|
| ~35K 预填充 | 1080 tok/s | 1307 tok/s | 1309 tok/s |
| ~145K 预填充 | 585 tok/s | 1017 tok/s | 1020 tok/s |

## Q4_0 权重 skinny 评估（P2，重要结论）

2 x V100 实测（Q4_0 目标模型，Q8_0 KV）：

| 场景 | skinny 开 | 通用路径 |
|---|---:|---:|
| perplexity 2048 | 2.8333 | 2.8333 |
| 贪心解码 512 tok | 52.0 tok/s | 64.2 tok/s |
| DFlash2 投机解码 512 tok | 94.2 tok/s | 97.4 tok/s |
| 预填充 512 tok | 57.3 tok/s | 76.7 tok/s |

Q8_0 权重同口径中性（解码 42.8 vs 43.1，预填充 75.7 vs 76.0）。数值完全一致
（PPL 逐位吻合），差异纯属性能：M=1 解码在 16 GB 卡上是带宽瓶颈；M > 16 的回退
每次 MUL_MAT 都要把重排权重展开成 F16，形成预填充开销。

**结论**：Q4_0 权重目前在本硬件上走通用 mmvq/mmq 更快。支持代码保留（能力对齐），
如目标机器基准同样显示回退，把 `ggml_cuda_q8_skinny_can_repack` 收窄到仅 Q8_0
是一行改动。Q4_K 权重（Q4_K_M）不参与重排，建议保持通用路径（P4 决策）。

## 原功能回归（P6）

| 项目 | 结果 |
|---|---|
| test-cuda-allreduce（NCCL / internal，2 rank） | ALL_OK |
| test-backend-ops（MUL_MAT/RMS_NORM/ADD/MUL、FLASH_ATTN_EXT hsk=256） | 3/3 后端，1604/1604 x2 |
| test-spec-reject | 通过 |
| Q8_0 KV PPL 2048 / 8192 | 2.7734 / 2.6017，与改造前逐位一致 |
| Q8_0 KV 145K 针检索 | 正确，预填充 1021 tok/s |
| Q8_0 KV + DFlash2 投机解码 | 75.0 tok/s，输出正常 |
| 多模态（mmproj-F16 上 GPU，测试图） | 正确描述"白底红色背景上的白色方块" |
| Q8_0 权重 skinny 路径数值 | PPL 2.8339，与通用路径一致 |

已知问题（预存，非本次引入）：`test-backend-ops` 全量跑在 fattn-mma-f16 的
hsk=320 用例上因动态共享内存超 V100 48 KB 触发 `cudaFuncSetAttribute` 失败而中止。

## 16K / 128K 上下文基准（P6）

Q4_K_M 权重 + Q4_0 KV，2-way TP，`-c 147456 -b 2048 -ub 2048`，合成文本，
`cache_prompt: false`：

| 上下文 | 预填充 | 解码（普通） | 解码（DFlash2 投机，7 tok/轮） |
|---|---:|---:|---:|
| 16K（16001 tok） | 1755.6 tok/s | 52.4 tok/s | 171.8 tok/s |
| 128K（128001 tok） | 1250.6 tok/s | 31.3 tok/s | 107.0 tok/s |

147K 上下文占用约 10.7 GB/卡（含权重 7.7 GB）。

## 测试环境

- 服务器：2 x Tesla V100-SXM2-16GB（NV2 NVLink），x86_64，CUDA 12.8，NCCL 2.26.2
- 模型：Qwen3.8-27B Q4_K_M（`Qwen3.8-27B-Q4_K_M.gguf`），DFlash2 草稿
  （`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`），mmproj-F16
- 构建：`Release`，`GGML_CUDA=ON`，`CMAKE_CUDA_ARCHITECTURES=70`，
  `GGML_CUDA_FA=ON`，`GGML_CUDA_GRAPHS=ON`，`GGML_CUDA_NCCL=ON`

详细开发记录见 [v100-q4-support.md](v100-q4-support.md)。
