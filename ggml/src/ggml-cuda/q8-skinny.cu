// Q8_0 skinny GEMM for sm_70 (Volta).
//
// Adapted from 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (Apache-2.0). The QPN8
// execution layout is derived from dnv2003/v100-skinny (MIT) and its block-scale adaptation
// in haohervchb/sglang-V100. See LICENSE.v100-skinny in this directory.

#include "common.cuh"
#include "convert.cuh"
#include "q8-skinny.cuh"

#include <algorithm>
#include <vector>

// sentinel object whose address tags a repacked tensor
static const char q8_skinny_marker = 0;

static int q8_skinny_split_k(const int64_t k) {
    for (int split_k : {16, 8, 4}) {
        if ((k / 16) % split_k == 0) {
            return split_k;
        }
    }
    return 0;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---- 1Cat fp8_qpn8_sm70.cu:39-50 (lane/column mapping, unchanged) ----

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// ---- int8 decoder (replaces the 1Cat FP8 decoder) ----

// Byte i of q.x and q.y hold the two int8 values of the i-th half2. Xor flips the sign bit,
// so each byte becomes an offset-binary u in [0, 255]. Byte i of x and y are merged into
// (1024 + u0, 1024 + u1) with the f16 bits of 1024 (0x6400) or-ed in: 1024..1279 are exactly
// representable in f16, so subtracting 1152 gives the signed values without rounding.
__device__ __forceinline__ void s8x8_to_half2x4(uint2 q, half2 out[4]) {
    const unsigned x = q.x ^ 0x80808080u;
    const unsigned y = q.y ^ 0x80808080u;
    const unsigned bias_bits = 0x64806480u; // f16 1152, 1152
    const half2 bias = *reinterpret_cast<const half2 *>(&bias_bits);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // bytes (x_i, x_0, y_i, x_0): the two filler bytes are masked off before or-ing in 0x64
        const unsigned packed = (__byte_perm(x, y, 0x0400u + 0x0101u*i) & 0x00FF00FFu) | 0x64006400u;
        out[i] = __hsub2(*reinterpret_cast<const half2 *>(&packed), bias);
    }
}

// ---- 1Cat fp8_qpn8_sm70.cu:168-175 (mma.m8n8k4 macro, unchanged) ----

#define Q8_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                       \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// ---- repack: Q8_0 row-major -> codes + scales ----

// One CTA per tile. src points at the first tile of the range, codes at its codes
// destination and scales at the full scales array indexed by the global tile.
__global__ void q8_skinny_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ codes,
                                        half * __restrict__ scales, int n, int k, int tile0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    const int blocks_k = k >> 5;
    const int tile = tile0 + blockIdx.x;
    uint8_t * codes_tile = codes + (size_t) blockIdx.x * 32 * k;
    for (int i = threadIdx.x; i < 32 * blocks_k; i += blockDim.x) {
        const int row = i / blocks_k;
        const int kb  = i - row * blocks_k;
        const uint8_t * block = src + ((size_t) blockIdx.x * 32 + row) * blocks_k * 34 + (size_t) kb * 34;
        const half d = *reinterpret_cast<const half *>(block);
        const uint8_t * qs = block + 2;
        const int lane   = qpn8_lane_from_col(row & 31);
        const int group0 = kb * 2;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int physical_k = qpn8_physical_k(j);
            codes_tile[((size_t) (group0 + 0) * 32 + lane) * 16 + physical_k] = qs[j];
            codes_tile[((size_t) (group0 + 1) * 32 + lane) * 16 + physical_k] = qs[16 + j];
        }
        scales[(size_t) kb * n + tile * 32 + lane] = d;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(src, codes, scales, n, k, tile0);
#endif
}

// ---- dequantize: codes -> dense F16 [N][K], same layout as the Q8_0 to F16 conversion ----

// One CTA per tile of 32 rows and 8 groups (128 K). A code word decodes to one output row,
// so warps stage the decoded values in shared memory first. That keeps the stores to global
// memory coalesced along K instead of one 4-byte store per row.
__global__ void q8_skinny_to_f16_kernel(half * __restrict__ output, const uint8_t * __restrict__ codes,
                                        const half * __restrict__ scales, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    // 64 half2 per row plus 4 half2 of padding
    __shared__ __align__(8) half2 tile_smem[32][8 * 8 + 4];

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int tile   = blockIdx.y;
    const int groups = k >> 4;
    const int g0     = blockIdx.x * 8;
    const int group  = g0 + warp;

    // warp `warp` decodes group `g0 + warp` into the rows of the tile
    if (group < groups) {
        const uint4 packed =
            reinterpret_cast<const uint4 *>(codes)[((size_t) tile * groups + group) * 32 + lane];
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half scale   = __ldg(scales + (size_t) (group >> 1) * n + tile * 32 + lane);
        const half2 scale2 = __halves2half2(scale, scale);
        half2 * smem_row = tile_smem[qpn8_col_from_lane(lane)];
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            smem_row[warp * 8 + pair] = __hmul2(weights[pair], scale2);
        }
    }
    __syncthreads();

    // 4 rows per warp, 2 half2 (8 bytes) per lane and row
    const int groups_here = groups - g0 < 8 ? groups - g0 : 8;
    half * out_tile = output + (size_t) tile * 32 * k + g0 * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = warp * 4 + i;
        if (2 * lane + 1 < groups_here * 8) {
            const uint2 value = *reinterpret_cast<const uint2 *>(&tile_smem[row][2 * lane]);
            *reinterpret_cast<uint2 *>(out_tile + (size_t) row * k + 4 * lane) = value;
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, scales, n, k);
#endif
}

// ---- 1Cat fp8_qpn8_sm70.cu:207-432 (main kernel, adapted) ----

template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ scales,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_scale_group = -1;
    half loaded_scale = __float2half(0.0f);
    uint4 prefetched = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
        prefetched = __ldcs(code_ptr + (size_t) group_begin * 32);
    }

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed =
            PrefetchCodes ? prefetched
                          : __ldcs(code_ptr + (size_t) group * 32);
        uint4 next = make_uint4(0, 0, 0, 0);
        if constexpr (PrefetchCodes) {
            if (group + 1 < group_begin + groups_per_warp) {
                next = __ldcs(code_ptr + (size_t) (group + 1) * 32);
            }
        }
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weights[i] = __hmul2(weights[i], scale2);
        }

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
        }
        if constexpr (PrefetchCodes) {
            prefetched = next;
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    // fixed reduction order over the split-K warps keeps the result reproducible
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, scales, input, output, n, k, m);
#endif
}

template <int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q8_skinny_launch(const uint8_t * codes, const half * scales, const half * input,
                             float * output, int n, int k, int m, cudaStream_t stream) {
    q8_skinny_kernel<SplitK, NAcc, false, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, scales, input, output, n, k, m);
}

#define Q8_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                            \
    do {                                                                                    \
        switch (split_k) {                                                                  \
            case 16: q8_skinny_launch<16, NAcc, M1Only, RowTiles>(codes, scales, input,     \
                         output, n, k, m, stream); break;                                   \
            case 8:  q8_skinny_launch<8, NAcc, M1Only, RowTiles>(codes, scales, input,      \
                         output, n, k, m, stream); break;                                   \
            default: q8_skinny_launch<4, NAcc, M1Only, RowTiles>(codes, scales, input,      \
                         output, n, k, m, stream); break;                                   \
        }                                                                                   \
    } while (0)

static void q8_skinny_mul_mat_launch(const uint8_t * codes, const half * scales, const half * input,
                                     float * output, int n, int k, int m, int split_k,
                                     cudaStream_t stream) {
    const int n_acc = k >= 4096 ? 2 : 1;
    if (m == 1) {
        if (n_acc == 2) { Q8_SKINNY_LAUNCH(2, true, 1); } else { Q8_SKINNY_LAUNCH(1, true, 1); }
    } else if (m <= 8) {
        if (n_acc == 2) { Q8_SKINNY_LAUNCH(2, false, 1); } else { Q8_SKINNY_LAUNCH(1, false, 1); }
    } else {
        if (n_acc == 2) { Q8_SKINNY_LAUNCH(2, false, 2); } else { Q8_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q8_SKINNY_LAUNCH

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_q8_skinny_is_repacked(const ggml_tensor * t) {
    return t->extra == (const void *) &q8_skinny_marker;
}

bool ggml_cuda_q8_skinny_can_repack(const ggml_tensor * t) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(t);
    return false;
#else
    if (t->type != GGML_TYPE_Q8_0 || t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
        return false;
    }
    if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ggml_cuda_get_device())) {
        return false;
    }
    if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (!ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    // the small-M kernel needs a split-K that divides the group count
    return k % 32 == 0 && n % 32 == 0 && n >= 512 && q8_skinny_split_k(k) != 0;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    const int64_t ntiles = n / 32;
    const size_t tile_src_bytes = 34 * (size_t) k;
    const size_t tile_dst_bytes = 32 * (size_t) k;
    const size_t scales_bytes = (size_t) n * k / 16;

    constexpr size_t staging_max = 32ull * 1024 * 1024;
    size_t tiles_per_chunk = staging_max / tile_src_bytes;
    if (tiles_per_chunk < 1) {
        tiles_per_chunk = 1;
    }
    if (tiles_per_chunk > (size_t) ntiles) {
        tiles_per_chunk = (size_t) ntiles;
    }

    cudaStream_t stream = ctx.stream();
    uint8_t * data = (uint8_t *) t->data;

    void * staging = nullptr;
    void * scales_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, tiles_per_chunk * tile_src_bytes));
    CUDA_CHECK(cudaMalloc(&scales_tmp, scales_bytes));

    // Chunk [t0, t1) writes codes to [t0*32K, t1*32K), which ends before the source
    // [t1*34K, ...) of the remaining tiles, and its own source was copied to staging first.
    // The scales region [N*K, N*K*17/16) overlaps the sources of the last tiles, so it is
    // written only after all codes are done.
    for (int64_t t0 = 0; t0 < ntiles; t0 += (int64_t) tiles_per_chunk) {
        const int64_t t1 = std::min(t0 + (int64_t) tiles_per_chunk, ntiles);
        CUDA_CHECK(cudaMemcpyAsync(staging, data + t0 * tile_src_bytes, (t1 - t0) * tile_src_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        q8_skinny_repack_kernel<<<(unsigned) (t1 - t0), 256, 0, stream>>>(
            (const uint8_t *) staging, data + t0 * tile_dst_bytes, (half *) scales_tmp,
            (int) n, (int) k, (int) t0);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(data + n * k, scales_tmp, scales_bytes, cudaMemcpyDeviceToDevice, stream));

    // no pool: the staging must not stay resident after the one-time repack
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(scales_tmp));

    t->extra = (void *) &q8_skinny_marker;
}

void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    // Skipping the scan is safe: a weight that is missed stays on the regular path and only
    // loses the acceleration, the result is still correct. A new model gets a new context.
    if (ctx.q8_skinny_idle_scans >= 1024) {
        return;
    }

    // candidates: Q8_0 weights used as MUL_MAT src0; none in steady state
    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        ggml_tensor * src0 = node->src[0];
        if (!ggml_cuda_q8_skinny_can_repack(src0) || ggml_cuda_q8_skinny_is_repacked(src0)) {
            continue;
        }
        if (std::find(candidates.begin(), candidates.end(), src0) == candidates.end()) {
            candidates.push_back(src0);
        }
    }
    if (candidates.empty()) {
        ++ctx.q8_skinny_idle_scans;
        return;
    }
    ctx.q8_skinny_idle_scans = 0;

    // repacking cannot be recorded into a CUDA graph; the capture is checked only here so that
    // the steady state costs no CUDA API call
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    // Repacked data can only be read by the new kernel, so drop candidates that are read
    // anywhere else (non-MUL_MAT src, MUL_MAT src1) or viewed by another tensor.
    for (int i = 0; i < cgraph->n_nodes && !candidates.empty(); ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->view_src != nullptr) {
            candidates.erase(std::remove(candidates.begin(), candidates.end(), node->view_src),
                             candidates.end());
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            if (src->view_src != nullptr) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src->view_src),
                                 candidates.end());
            }
            if (node->op != GGML_OP_MUL_MAT || j != 0) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src),
                                 candidates.end());
            }
        }
    }

    for (ggml_tensor * t : candidates) {
        ggml_cuda_q8_skinny_repack_inplace(ctx, t);
    }
}

bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int split_k = q8_skinny_split_k(k);
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || split_k == 0 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 16) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    const uint8_t * data = (const uint8_t *) src0->data;
    q8_skinny_mul_mat_launch(data, (const half *) (data + n * k), input.get(), (float *) dst->data,
                             (int) n, (int) k, (int) m, split_k, ctx.stream());
    return true;
}

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int n = (int) src0->ne[1];
    const int k = (int) src0->ne[0];
    const uint8_t * data = (const uint8_t *) src0->data;
    const dim3 grid((unsigned) ((k / 16 + 7) / 8), (unsigned) (n / 32), 1);
    q8_skinny_to_f16_kernel<<<grid, 256, 0, stream>>>(
        dst, data, (const half *) (data + n * k), n, k);
    CUDA_CHECK(cudaGetLastError());
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
