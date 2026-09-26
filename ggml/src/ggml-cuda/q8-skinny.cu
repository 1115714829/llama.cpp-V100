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

// ---- multiple projections of one input in one launch ----

// One segment is a repacked weight (N is a multiple of 32) plus its output tile range.
// first_tile is the running sum of n / 32 over the previous segments, so a CTA finds its
// segment from blockIdx.x with a short scan.
struct q8_skinny_multi_seg {
    const uint8_t * codes;
    const half *    scales;
    int             n;
    float *         dst;
    int             first_tile;
};

// A narrow weight (N is not a multiple of 32, e.g. ssm_beta/ssm_alpha at 12) keeps the
// standard Q8_0 rows: K/32 blocks of 34 bytes per row, half d first, then 32 int8.
// One warp computes one row for all M tokens. Like the repacked tiles above, first_row
// is a running sum of n_rows.
struct q8_skinny_multi_dot {
    const uint8_t * rows;
    int             n_rows;
    float *         dst;
    int             first_row;
};

// Passed by value. The repacked tiles come first in the grid, the dot rows follow.
struct q8_skinny_multi_params {
    q8_skinny_multi_seg seg[4];
    q8_skinny_multi_dot dot[4];
    int n_seg;
    int n_dot;
    int n_main_tiles;
    int n_dot_rows;
    const half * input;
    int k;
    int m;
};

// Same execution as q8_skinny_kernel, but each CTA picks its segment from blockIdx.x and
// writes to that segment's dst with its own row length. The dot CTAs have no split-K.
template <int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_multi_kernel(const q8_skinny_multi_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny multi supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny multi M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if ((int) blockIdx.x >= p.n_main_tiles) {
        // One narrow row per CTA: all threads split K, so that each warp step reads exactly one
        // Q8_0 block (a broadcast scale and 32 contiguous quants), then the block reduces in a
        // fixed order. With one warp per row the K loop was latency bound and these few rows
        // outlasted all repacked tiles of the launch.
        const int row_all = (int) blockIdx.x - p.n_main_tiles;
        int seg = 0;
        while (seg + 1 < p.n_dot && row_all >= p.dot[seg + 1].first_row) {
            ++seg;
        }
        const int row = row_all - p.dot[seg].first_row;
        const block_q8_0 * wrow = reinterpret_cast<const block_q8_0 *>(p.dot[seg].rows) + (size_t) row * (p.k / QK8_0);

        float accum[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            accum[t] = 0.0f;
        }
#pragma unroll 2
        for (int i = threadIdx.x; i < p.k; i += blockDim.x) {
            const block_q8_0 * block = wrow + i / QK8_0;
            const float weight = __half2float(block->d) * (float) block->qs[i % QK8_0];
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (t < p.m) {
                    accum[t] = fmaf(weight, __half2float(p.input[(size_t) t * p.k + i]), accum[t]);
                }
            }
        }

        // the partials buffer holds at least 16 values per warp (SplitK * 256 >= warps * 16)
        float * red = &partials[0][0];
        const int n_warps = blockDim.x >> 5;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t < p.m) {
                float v = accum[t];
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_down_sync(0xffffffffU, v, offset);
                }
                if (lane == 0) {
                    red[warp * 16 + t] = v;
                }
            }
        }
        __syncthreads();
        if ((int) threadIdx.x < p.m) {
            float v = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                v += red[w * 16 + threadIdx.x];
            }
            p.dot[seg].dst[(size_t) threadIdx.x * p.dot[seg].n_rows + row] = v;
        }
        return;
    }

    int seg = 0;
    while (seg + 1 < p.n_seg && (int) blockIdx.x >= p.seg[seg + 1].first_tile) {
        ++seg;
    }
    const int n = p.seg[seg].n;
    const int k = p.k;
    const int m = p.m;
    const int tile = (int) blockIdx.x - p.seg[seg].first_tile;
    const uint8_t * codes = p.seg[seg].codes;
    const half * scales = p.seg[seg].scales;
    const half * input = p.input;
    float * output = p.seg[seg].dst;

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

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed = __ldcs(code_ptr + (size_t) group * 32);
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
    GGML_UNUSED(p);
#endif
}

// ---- 1Cat fp8_qpn8_sm70.cu:588-770 (gated pair kernel, adapted) ----

// gate and up are two separate tensors instead of one combined weight: projection p
// (0 = gate, 1 = up) reads its own codes/scales, both for the tile at blockIdx.x. The
// epilogue applies silu(gate) * up and writes float.
template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_gated_kernel(
    const uint8_t * __restrict__ gate_codes, const half * __restrict__ gate_scales,
    const uint8_t * __restrict__ up_codes, const half * __restrict__ up_scales,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny gated pair supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny gated M=1 specialization uses one row tile");
    __shared__ float partials[2][SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int projection = warp_in_block / SplitK;
    const int warp = warp_in_block - projection * SplitK;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint8_t * codes = projection ? up_codes : gate_codes;
    const half * scales = projection ? up_scales : gate_scales;
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
                    partials[projection][warp][quadpair * 8 + output_col] = accum[0][0][i];
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
                partials[projection][warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            gate += partials[0][k_warp][element];
            up += partials[1][k_warp][element];
        }
        const float silu = gate / (1.0f + __expf(-gate));
        if constexpr (M1Only) {
            output[tile * 32 + element] = silu * up;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = silu * up;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, gate_scales, up_codes, up_scales, input, output, n, k, m);
#endif
}

// Static split-K/accumulator-chain/prefetch table from 1Cat's sm_70 measurements
// (fp8.py:121-129). Shapes outside the table keep the generic rule.
struct q8_skinny_config {
    int  split_k;
    int  n_acc;
    bool prefetch;
};

static q8_skinny_config q8_skinny_config_for(const int64_t n, const int64_t k) {
    if (k == 4352 || k == 1536) {
        return {16, 1, false};
    }
    if (k == 5120) {
        if (n == 4352) {
            // 1Cat's unfused gate/up (5120, 8704) also uses one chain with prefetching;
            // here that weight is two separate (5120, 4352) tensors
            return {16, 1, true};
        }
        return {16, 2, false};
    }
    return {q8_skinny_split_k(k), k >= 4096 ? 2 : 1, false};
}

// The multi-weight kernel never prefetches, so the K=5120/N=4352 special case is not used.
static q8_skinny_config q8_skinny_multi_config_for(const int64_t k) {
    if (k == 4352 || k == 1536) {
        return {16, 1, false};
    }
    if (k == 5120) {
        return {16, 2, false};
    }
    return {q8_skinny_split_k(k), k >= 4096 ? 2 : 1, false};
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only, int RowTiles>
static void q8_skinny_launch(const uint8_t * codes, const half * scales, const half * input,
                             float * output, int n, int k, int m, cudaStream_t stream) {
    q8_skinny_kernel<SplitK, NAcc, PrefetchCodes, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, scales, input, output, n, k, m);
}

#define Q8_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                            \
    do {                                                                                    \
        switch (config.split_k) {                                                           \
            case 16: q8_skinny_launch<16, NAcc, false, M1Only, RowTiles>(codes, scales,     \
                         input, output, n, k, m, stream); break;                            \
            case 8:  q8_skinny_launch<8, NAcc, false, M1Only, RowTiles>(codes, scales,      \
                         input, output, n, k, m, stream); break;                            \
            default: q8_skinny_launch<4, NAcc, false, M1Only, RowTiles>(codes, scales,      \
                         input, output, n, k, m, stream); break;                            \
        }                                                                                   \
    } while (0)

static void q8_skinny_mul_mat_launch(const uint8_t * codes, const half * scales, const half * input,
                                     float * output, int n, int k, int m, cudaStream_t stream) {
    const q8_skinny_config config = q8_skinny_config_for(n, k);
    if (config.prefetch) {
        // only (K=5120, N=4352) selects prefetching, always at split 16 with one chain
        GGML_ASSERT(config.split_k == 16 && config.n_acc == 1);
        if (m == 1) {
            q8_skinny_launch<16, 1, true, true, 1>(codes, scales, input, output, n, k, m, stream);
        } else if (m <= 8) {
            q8_skinny_launch<16, 1, true, false, 1>(codes, scales, input, output, n, k, m, stream);
        } else {
            q8_skinny_launch<16, 1, true, false, 2>(codes, scales, input, output, n, k, m, stream);
        }
    } else if (config.n_acc == 2) {
        if (m == 1) { Q8_SKINNY_LAUNCH(2, true, 1); } else if (m <= 8) { Q8_SKINNY_LAUNCH(2, false, 1); }
        else { Q8_SKINNY_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q8_SKINNY_LAUNCH(1, true, 1); } else if (m <= 8) { Q8_SKINNY_LAUNCH(1, false, 1); }
        else { Q8_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q8_SKINNY_LAUNCH

template <bool M1Only, int RowTiles>
static void q8_skinny_gated_launch(const uint8_t * gate_codes, const half * gate_scales,
                                   const uint8_t * up_codes, const half * up_scales,
                                   const half * input, float * output, int n, int k, int m,
                                   cudaStream_t stream) {
    q8_skinny_gated_kernel<8, 2, true, M1Only, RowTiles><<<n / 32, 64 * 8, 0, stream>>>(
        gate_codes, gate_scales, up_codes, up_scales, input, output, n, k, m);
}

static void q8_skinny_gated_mul_mat_launch(const uint8_t * gate_codes, const half * gate_scales,
                                           const uint8_t * up_codes, const half * up_scales,
                                           const half * input, float * output, int n, int k, int m,
                                           cudaStream_t stream) {
    if (m == 1) {
        q8_skinny_gated_launch<true, 1>(gate_codes, gate_scales, up_codes, up_scales,
                                        input, output, n, k, m, stream);
    } else if (m <= 8) {
        q8_skinny_gated_launch<false, 1>(gate_codes, gate_scales, up_codes, up_scales,
                                         input, output, n, k, m, stream);
    } else {
        q8_skinny_gated_launch<false, 2>(gate_codes, gate_scales, up_codes, up_scales,
                                         input, output, n, k, m, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q8_skinny_multi_launch(const q8_skinny_multi_params & p, cudaStream_t stream) {
    // one CTA per narrow row after the repacked tiles
    const int n_tiles = p.n_main_tiles + p.n_dot_rows;
    q8_skinny_multi_kernel<SplitK, NAcc, M1Only, RowTiles><<<n_tiles, 32 * SplitK, 0, stream>>>(p);
}

#define Q8_SKINNY_MULTI_LAUNCH(NAcc, M1Only, RowTiles)                                      \
    do {                                                                                    \
        switch (config.split_k) {                                                           \
            case 16: q8_skinny_multi_launch<16, NAcc, M1Only, RowTiles>(p, stream); break;  \
            case 8:  q8_skinny_multi_launch<8, NAcc, M1Only, RowTiles>(p, stream); break;   \
            default: q8_skinny_multi_launch<4, NAcc, M1Only, RowTiles>(p, stream); break;   \
        }                                                                                   \
    } while (0)

static void q8_skinny_multi_mul_mat_launch(const q8_skinny_multi_params & p, cudaStream_t stream) {
    const q8_skinny_config config = q8_skinny_multi_config_for(p.k);
    GGML_ASSERT(!config.prefetch);
    if (config.n_acc == 2) {
        if (p.m == 1) { Q8_SKINNY_MULTI_LAUNCH(2, true, 1); } else if (p.m <= 8) { Q8_SKINNY_MULTI_LAUNCH(2, false, 1); }
        else { Q8_SKINNY_MULTI_LAUNCH(2, false, 2); }
    } else {
        if (p.m == 1) { Q8_SKINNY_MULTI_LAUNCH(1, true, 1); } else if (p.m <= 8) { Q8_SKINNY_MULTI_LAUNCH(1, false, 1); }
        else { Q8_SKINNY_MULTI_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q8_SKINNY_MULTI_LAUNCH

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
    // the small-M kernel needs a split-K that divides the group count; the N >= 32 bound only
    // keeps the tile grid non-empty. Small-N weights (wk/wv at 256, ...) usually run in a
    // group of projections that share the input, the multi-weight kernel below handles them.
    return k % 32 == 0 && n % 32 == 0 && n >= 32 && q8_skinny_split_k(k) != 0;
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
                             (int) n, (int) k, (int) m, ctx.stream());
    return true;
}

bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                       const ggml_tensor * up_w, const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    if (!ggml_cuda_q8_skinny_is_repacked(gate_w) || !ggml_cuda_q8_skinny_is_repacked(up_w)) {
        return false;
    }
    const int64_t k = gate_w->ne[0];
    const int64_t n = gate_w->ne[1];
    const int64_t m = src1->ne[1];
    if (!ggml_are_same_shape(gate_w, up_w) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            m < 1 || m > 16 ||
            k % 16 != 0 || (k / 16) % 8 != 0 || n % 32 != 0) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    const uint8_t * gate_data = (const uint8_t *) gate_w->data;
    const uint8_t * up_data   = (const uint8_t *) up_w->data;
    q8_skinny_gated_mul_mat_launch(gate_data, (const half *) (gate_data + n * k),
                                   up_data, (const half *) (up_data + n * k),
                                   input.get(), (float *) dst->data, (int) n, (int) k, (int) m,
                                   ctx.stream());
    return true;
}

bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                       int n_nodes, const ggml_tensor * src1) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 16 ||
            k % 32 != 0 || q8_skinny_split_k(k) == 0) {
        return false;
    }

    q8_skinny_multi_params p = {};
    p.k = (int) k;
    p.m = (int) m;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        ggml_tensor * dst = dsts[i];
        const int64_t n = w->ne[1];
        if (w->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
        if (ggml_cuda_q8_skinny_is_repacked(w)) {
            if (p.n_seg >= 4) {
                return false;
            }
            q8_skinny_multi_seg & seg = p.seg[p.n_seg++];
            seg.codes = (const uint8_t *) w->data;
            seg.scales = (const half *) (seg.codes + n * k);
            seg.n = (int) n;
            seg.dst = (float *) dst->data;
        } else {
            // narrow rows keep the row-major Q8_0 layout, the dot path reads them directly
            if (w->type != GGML_TYPE_Q8_0 || w->view_src != nullptr || w->op != GGML_OP_NONE || n % 32 == 0) {
                return false;
            }
            if (p.n_dot >= 4) {
                return false;
            }
            q8_skinny_multi_dot & dot = p.dot[p.n_dot++];
            dot.rows = (const uint8_t *) w->data;
            dot.n_rows = (int) n;
            dot.dst = (float *) dst->data;
        }
    }

    for (int s = 0; s < p.n_seg; ++s) {
        p.seg[s].first_tile = p.n_main_tiles;
        p.n_main_tiles += p.seg[s].n / 32;
    }
    for (int s = 0; s < p.n_dot; ++s) {
        p.dot[s].first_row = p.n_dot_rows;
        p.n_dot_rows += p.dot[s].n_rows;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());
    p.input = input.get();

    q8_skinny_multi_mul_mat_launch(p, ctx.stream());
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

bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                       const ggml_tensor * up_w, const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(gate_w);
    GGML_UNUSED(up_w);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                       int n_nodes, const ggml_tensor * src1) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0s);
    GGML_UNUSED(dsts);
    GGML_UNUSED(n_nodes);
    GGML_UNUSED(src1);
    return false;
}

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
