// Q4_K skinny GEMM for sm_70 (Volta).
//
// Design follows the QPN8 execution layout of q8-skinny.cu (itself adapted from 1Cat-vLLM
// fp8_qpn8_sm70.cu, Apache-2.0) and the int4 scale/bias folding of 1Cat's awq_qpn_m1_sm70.cu
// and ninfer-v100's q4 Volta kernels. The QPN8 layout is derived from dnv2003/v100-skinny
// (MIT) and its block-scale adaptation in haohervchb/sglang-V100. See LICENSE.v100-skinny in
// this directory and docs/design/v106-q4k-skinny.md for the layout and the numeric analysis.

#include "common.cuh"
#include "convert.cuh"
#include "q-skinny-common.cuh"
#include "q4k-skinny.cuh"

#include <algorithm>
#include <vector>

// sentinel object whose address tags a repacked Q4_K tensor (not const: it is stored in tensor->extra)
static char q4k_skinny_marker = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---- codec: per-type code decode plus scale/min parsing ----

template <ggml_type T> struct qskinny_codec;

// Q4_K: 8 code bytes per 16 K values and lane (16 nibbles, physical slot s in byte s/2, half
// s&1) and 16 meta bytes per 256 K values (the raw super-block header). decode() turns the two
// records of one 32-value sub-block plus the meta of its super-block into 16 half2 weights,
// with w = d*sc*q - dmin*m folded into one half FMA. Adding Q5_K/Q6_K means adding another
// specialization of this interface.
template <> struct qskinny_codec<GGML_TYPE_Q4_K> {
    using record_t = uint2;
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 16;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __device__ __forceinline__ uint8_t meta_byte(const uint4 meta, const int i) {
        // scales[12] starts at byte 4 of the super-block header
        const unsigned word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w;
        return (uint8_t) (word >> (8 * (i & 3)));
    }

    // get_scale_min_k4 from dequantize.cuh, applied to the raw meta bytes
    static __device__ __forceinline__ void get_scale_min(const uint4 meta, const int j,
                                                         uint8_t & sc, uint8_t & m) {
        if (j < 4) {
            sc = meta_byte(meta, j) & 63;
            m  = meta_byte(meta, j + 4) & 63;
        } else {
            sc = (meta_byte(meta, j + 4) & 0x0F) | ((meta_byte(meta, j - 4) >> 6) << 4);
            m  = (meta_byte(meta, j + 4) >> 4)   | ((meta_byte(meta, j)     >> 6) << 4);
        }
    }

    // One record holds 16 nibbles in physical order. The m8n8k4 B fragment pairs physical
    // slots (s, s+4), exactly like the Q8_0 decoder pairs bytes (i, i+4).
    static __device__ __forceinline__ void decode_record(const record_t rec, half2 out[8]) {
        // the second byte goes to result byte 2, where the mask below reads it
        const unsigned pairs[4] = {
            __byte_perm(rec.x, 0, 0x0200),  // bytes (0, 2)
            __byte_perm(rec.x, 0, 0x0301),  // bytes (1, 3)
            __byte_perm(rec.y, 0, 0x0200),  // bytes (4, 6)
            __byte_perm(rec.y, 0, 0x0301),  // bytes (5, 7)
        };
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned lo = (pairs[i] & 0x000F000Fu) | offset_bits;
            const unsigned hi = ((pairs[i] >> 4) & 0x000F000Fu) | offset_bits;
            out[2*i + 0] = __hsub2(*reinterpret_cast<const half2 *>(&lo), offset);
            out[2*i + 1] = __hsub2(*reinterpret_cast<const half2 *>(&hi), offset);
        }
    }

    static __device__ __forceinline__ void decode(const record_t records[2], const uint4 meta,
                                                  const int sub_block, half2 out[16]) {
        decode_record(records[0], out);
        decode_record(records[1], out + 8);

        uint8_t sc, m;
        get_scale_min(meta, sub_block, sc, m);
        const half2 dm = *reinterpret_cast<const half2 *>(&meta.x);
        const float d    = __half2float(__low2half(dm));
        const float dmin = __half2float(__high2half(dm));
        // f32 folding, rounded once to f16 (see 3.4 of the design note)
        const half2 scale2 = __float2half2_rn(d * (float) sc);
        const half2 bias2  = __float2half2_rn(-dmin * (float) m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hfma2(out[i], scale2, bias2);
        }
    }
};

// ---- repack: Q4_K row-major -> codes + meta ----

// One CTA per tile. src points at the first tile of the range, codes at its codes
// destination and meta at the full meta array indexed by the global tile.
__global__ void q4k_skinny_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ codes,
                                         uint8_t * __restrict__ meta, int n, int k, int tile0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    const int blocks_k = k >> 8;
    const int tile = tile0 + blockIdx.x;
    uint8_t * codes_tile = codes + (size_t) blockIdx.x * 32 * (k >> 1);
    for (int i = threadIdx.x; i < 32 * blocks_k; i += blockDim.x) {
        const int row = i / blocks_k;
        const int kb  = i - row * blocks_k;
        const uint8_t * block = src + ((size_t) blockIdx.x * 32 + row) * blocks_k * 144 + (size_t) kb * 144;
        const int lane = qpn8_lane_from_col(row & 31);

        // the raw 16-byte super-block header (dm + scales) is copied unchanged
        *reinterpret_cast<uint4 *>(meta + ((size_t) kb * n + (size_t) tile * 32 + lane) * 16) =
            *reinterpret_cast<const uint4 *>(block);

        // a 32-value sub-block uses 32 consecutive qs bytes: low nibbles for even sub-blocks,
        // high nibbles for odd ones; group g covers the first (g&1 == 0) or second 16 values
        const uint8_t * qs = block + 16;
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            const int qs_base  = 32 * (g >> 2) + ((g & 1) ? 16 : 0);
            const int qs_shift = (g & 2) ? 4 : 0;
            unsigned rec0 = 0;
            unsigned rec1 = 0;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int slot = qpn8_physical_k(j);
                const unsigned nib = (unsigned) (((qs[qs_base + j] >> qs_shift) & 0xF) << ((slot & 1) * 4));
                if ((slot >> 1) < 4) {
                    rec0 |= nib << (8 * (slot >> 1));
                } else {
                    rec1 |= nib << (8 * ((slot >> 1) - 4));
                }
            }
            *reinterpret_cast<uint2 *>(codes_tile + (((size_t) kb * 16 + g) * 32 + lane) * 8) =
                make_uint2(rec0, rec1);
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(src, codes, meta, n, k, tile0);
#endif
}

// ---- dequantize: codes + meta -> dense F16 [N][K], same layout as the Q4_K to F16 conversion ----

// One CTA per (tile of 32 rows, 256-K super-block). A warp decodes one 32-value sub-block per
// lane; a code record decodes to one output row, so the warps stage the decoded values in
// shared memory first. That keeps the stores to global memory coalesced along K instead of one
// 4-byte store per row.
template <ggml_type T>
__global__ void qskinny_to_f16_kernel(half * __restrict__ output, const uint8_t * __restrict__ codes,
                                      int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    // values_per_meta / 2 half2 per row plus 4 half2 of padding
    __shared__ __align__(16) half2 tile_smem[32][codec::values_per_meta / 2 + 4];
    __shared__ __align__(16) uint4 meta_smem[32];

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int tile   = blockIdx.y;
    const int kb     = blockIdx.x;
    const int groups = k >> 4;

    // one 16-byte meta per lane, for the row this lane holds the codes of
    if (warp == 0) {
        meta_smem[lane] = reinterpret_cast<const uint4 *>(codes + (size_t) n * k / 2)
                              [(size_t) kb * n + tile * 32 + lane];
    }
    __syncthreads();

    // sub-block `warp` of this super-block: records 2*warp and 2*warp+1
    const int group = kb * (codec::values_per_meta / codec::values_per_record) + 2 * warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes);
    const typename codec::record_t records[2] = {
        code_ptr[((size_t) tile * groups + group) * 32 + lane],
        code_ptr[((size_t) tile * groups + group + 1) * 32 + lane],
    };
    half2 weights[codec::values_per_sub_block / 2];
    codec::decode(records, meta_smem[lane], warp, weights);

    half2 * smem_row = tile_smem[qpn8_col_from_lane(lane)];
#pragma unroll
    for (int i = 0; i < codec::values_per_sub_block / 2; ++i) {
        smem_row[warp * (codec::values_per_sub_block / 2) + i] = weights[i];
    }
    __syncthreads();

    // 4 uint4 (four half2 each) per thread
    half * out_tile = output + (size_t) tile * 32 * k + (size_t) kb * codec::values_per_meta;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * 256;
        const int row = idx >> 5;
        const int col = (idx & 31) << 2;
        const uint4 value = *reinterpret_cast<const uint4 *>(&tile_smem[row][col]);
        *reinterpret_cast<uint4 *>(out_tile + (size_t) row * k + col * 2) = value;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, n, k);
#endif
}

// ---- 1Cat fp8_qpn8_sm70.cu:168-175 (mma.m8n8k4 macro, unchanged) ----

#define Q4K_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// ---- 1Cat fp8_qpn8_sm70.cu:207-432 (main kernel, adapted to the codec) ----

template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q4k_skinny_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    // one iteration per 32-value sub-block; SplitK divides the sub-block count, the meta is
    // reloaded whenever the 256-value super-block index changes
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    const uint4 * meta_ptr = reinterpret_cast<const uint4 *>(codes + (size_t) n * k / 2) +
                             tile * 32 + lane;

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
    int loaded_meta_kb = -1;
    uint4 meta = make_uint4(0, 0, 0, 0);

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            // one 16-byte meta per 256 values; shared by the 8 sub-blocks that follow
            meta = __ldcs(meta_ptr + (size_t) kb * n);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const typename codec::record_t records[2] = {
            __ldcs(code_ptr + (size_t) (group + 0) * 32),
            __ldcs(code_ptr + (size_t) (group + 1) * 32),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
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
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

// ---- gated pair: gate and up projected in one kernel (same warp split as q8_skinny_gated_kernel) ----

// gate and up are two separate weights: projection p (0 = gate, 1 = up) reads its own codes
// and meta for the tile at blockIdx.x and the epilogue applies silu(gate) * up. A single
// 8-row tile is used because at split-K 16 the two partials arrays would need 64 KiB of
// shared memory; m = 9..16 runs as two single-projection launches plus a SwiGLU pass instead.
// The two projections double the block to 64*SplitK threads, so __launch_bounds__ caps the
// register count and the split-16 launch (1024 threads) stays valid. Split-K and the
// accumulator chains match q4k_skinny_kernel, so both projections of an m <= 8 result are
// bit-identical to two separate q4k_skinny_kernel launches; only the trailing silu rounds
// differently from the regular GLU node (__expf here, expf there).
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false>
__global__ void __launch_bounds__(64 * SplitK) q4k_skinny_gated_kernel(
    const uint8_t * __restrict__ gate_codes, const uint8_t * __restrict__ up_codes,
    const half * __restrict__ input, float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    __shared__ float partials[2][SplitK][M1Only ? 32 : 256];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int projection = warp_in_block / SplitK;
    const int warp = warp_in_block - projection * SplitK;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const uint8_t * codes = projection ? up_codes : gate_codes;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    const uint4 * meta_ptr = reinterpret_cast<const uint4 *>(codes + (size_t) n * k / 2) +
                             tile * 32 + lane;

    float accum[NAcc][8];
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            accum[chain][i] = 0.0f;
        }
    }
    int loaded_meta_kb = -1;
    uint4 meta = make_uint4(0, 0, 0, 0);

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            meta = __ldcs(meta_ptr + (size_t) kb * n);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const typename codec::record_t records[2] = {
            __ldcs(code_ptr + (size_t) (group + 0) * 32),
            __ldcs(code_ptr + (size_t) (group + 1) * 32),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        uint4 input45 = make_uint4(0, 0, 0, 0);
        uint4 input67 = make_uint4(0, 0, 0, 0);
        if (row < m) {
            const half * input_row = input + (size_t) row * k;
            input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
            input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
            input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
        }

        const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
        const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
        const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
        const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
        Q4K_SKINNY_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
        Q4K_SKINNY_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
        Q4K_SKINNY_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
        Q4K_SKINNY_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
        Q4K_SKINNY_MMA_8N8K4(accum[0], a2[0], a2[1], b[8], b[9]);
        Q4K_SKINNY_MMA_8N8K4(accum[1 % NAcc], a2[2], a2[3], b[10], b[11]);
        Q4K_SKINNY_MMA_8N8K4(accum[2 % NAcc], a3[0], a3[1], b[12], b[13]);
        Q4K_SKINNY_MMA_8N8K4(accum[3 % NAcc], a3[2], a3[3], b[14], b[15]);
    }

#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            accum[0][i] += accum[chain][i];
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
                    partials[projection][warp][quadpair * 8 + output_col] = accum[0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int output_row =
                (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
            const int output_col =
                (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            partials[projection][warp][output_row * 32 + quadpair * 8 + output_col] =
                accum[0][i];
        }
    }
    __syncthreads();

    constexpr int kOutputElements = M1Only ? 32 : 256;
    for (int element = threadIdx.x; element < kOutputElements; element += blockDim.x) {
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
    GGML_UNUSED_VARS(gate_codes, up_codes, input, output, n, k, m);
#endif
}

// Split-K 16 for the long K weights, 8 for the smallest ones (k = 256 has only 8 sub-blocks
// per tile). can_repack guarantees k % 256 == 0, so one of the two always divides the
// sub-block count. The long K weights use two accumulator chains, as in the Q8_0 path. No
// prefetch variant yet.
struct q4k_skinny_config {
    int split_k;
    int n_acc;
};

static q4k_skinny_config q4k_skinny_config_for(const int64_t k) {
    return { k % 512 == 0 ? 16 : 8, k >= 4096 ? 2 : 1 };
}

template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_launch(const uint8_t * codes, const half * input, float * output,
                              int n, int k, int m, cudaStream_t stream) {
    q4k_skinny_kernel<T, SplitK, NAcc, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, input, output, n, k, m);
}

#define Q4K_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                               \
    do {                                                                                        \
        switch (config.split_k) {                                                               \
            case 16: q4k_skinny_launch<GGML_TYPE_Q4_K, 16, NAcc, M1Only, RowTiles>(codes,       \
                         input, output, n, k, m, stream); break;                                \
            default: q4k_skinny_launch<GGML_TYPE_Q4_K, 8, NAcc, M1Only, RowTiles>(codes,        \
                         input, output, n, k, m, stream); break;                                \
        }                                                                                       \
    } while (0)

static void q4k_skinny_mul_mat_launch(const uint8_t * codes, const half * input, float * output,
                                      int n, int k, int m, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_LAUNCH(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q4K_SKINNY_LAUNCH(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q4K_SKINNY_LAUNCH

template <int SplitK, int NAcc, bool M1Only>
static void q4k_skinny_gated_launch(const uint8_t * gate_codes, const uint8_t * up_codes,
                                    const half * input, float * output,
                                    int n, int k, int m, cudaStream_t stream) {
    q4k_skinny_gated_kernel<GGML_TYPE_Q4_K, SplitK, NAcc, M1Only>
        <<<n / 32, 32 * 2 * SplitK, 0, stream>>>(gate_codes, up_codes, input, output, n, k, m);
}

#define Q4K_SKINNY_GATED_LAUNCH(NAcc, M1Only)                                          \
    do {                                                                               \
        switch (config.split_k) {                                                      \
            case 16: q4k_skinny_gated_launch<16, NAcc, M1Only>(gate_codes, up_codes,   \
                         input, output, n, k, m, stream); break;                       \
            default: q4k_skinny_gated_launch<8, NAcc, M1Only>(gate_codes, up_codes,    \
                         input, output, n, k, m, stream); break;                       \
        }                                                                              \
    } while (0)

static void q4k_skinny_gated_mul_mat_launch(const uint8_t * gate_codes, const uint8_t * up_codes,
                                            const half * input, float * output,
                                            int n, int k, int m, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(2, true); } else { Q4K_SKINNY_GATED_LAUNCH(2, false); }
    } else {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(1, true); } else { Q4K_SKINNY_GATED_LAUNCH(1, false); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q4K_SKINNY_GATED_LAUNCH

// Same epilogue as q4k_skinny_gated_kernel, applied after both projections of an M > 8 run.
__global__ void q4k_skinny_swiglu_kernel(float * __restrict__ gate, const float * __restrict__ up,
                                         int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + __expf(-g)) * up[i];
    }
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_q4k_skinny_is_repacked(src0));
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 16) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    q4k_skinny_mul_mat_launch((const uint8_t *) src0->data, input.get(), (float *) dst->data,
                              (int) n, (int) k, (int) m, ctx.stream());
    return true;
}

bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                        const ggml_tensor * up_w, const ggml_tensor * src1,
                                        ggml_tensor * dst) {
    if (!ggml_cuda_q4k_skinny_is_repacked(gate_w) || !ggml_cuda_q4k_skinny_is_repacked(up_w)) {
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
            k % 256 != 0 || n % 32 != 0) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    const uint8_t * gate_data = (const uint8_t *) gate_w->data;
    const uint8_t * up_data   = (const uint8_t *) up_w->data;
    if (m <= 8) {
        q4k_skinny_gated_mul_mat_launch(gate_data, up_data, input.get(), (float *) dst->data,
                                        (int) n, (int) k, (int) m, ctx.stream());
        return true;
    }

    // M = 9..16: the fused kernel would need 64 KiB of split-K partials at split 16; run both
    // projections with the single-projection kernel and the SwiGLU epilogue instead
    float * gate = (float *) dst->data;
    ggml_cuda_pool_alloc<float> up(ctx.pool(), n * m);
    q4k_skinny_mul_mat_launch(gate_data, input.get(), gate, (int) n, (int) k, (int) m, ctx.stream());
    q4k_skinny_mul_mat_launch(up_data, input.get(), up.get(), (int) n, (int) k, (int) m, ctx.stream());

    const int count = (int) (n * m);
    q4k_skinny_swiglu_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(gate, up.get(), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- multiple projections of one input in one launch ----

// One segment is a repacked Q4_K weight plus its output tile range. first_tile is the
// running sum of n / 32 over the previous segments, so a CTA finds its segment from
// blockIdx.x with a short scan.
struct q4k_skinny_multi_seg {
    const uint8_t * codes;
    const uint8_t * meta;
    int             n;
    float *         dst;
    int             first_tile;
};

// Passed by value.
struct q4k_skinny_multi_params {
    q4k_skinny_multi_seg seg[4];
    int n_seg;
    int n_main_tiles;
    const half * input;
    int k;
    int m;
};

// Same execution as q4k_skinny_kernel, but each CTA picks its segment from blockIdx.x and
// writes to that segment's dst with its own row length. Every weight is a repacked Q4_K
// tensor, so there is no narrow-row dot path like in the Q8_0 multi kernel.
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q4k_skinny_multi_kernel(const q4k_skinny_multi_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny multi supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny multi M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    int seg = 0;
    while (seg + 1 < p.n_seg && (int) blockIdx.x >= p.seg[seg + 1].first_tile) {
        ++seg;
    }
    const int n = p.seg[seg].n;
    const int k = p.k;
    const int m = p.m;
    const int tile = (int) blockIdx.x - p.seg[seg].first_tile;
    const uint8_t * codes = p.seg[seg].codes;
    const half * input = p.input;
    float * output = p.seg[seg].dst;

    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    // one iteration per 32-value sub-block; SplitK divides the sub-block count, the meta is
    // reloaded whenever the 256-value super-block index changes
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    const uint4 * meta_ptr = reinterpret_cast<const uint4 *>(p.seg[seg].meta) + tile * 32 + lane;

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
    int loaded_meta_kb = -1;
    uint4 meta = make_uint4(0, 0, 0, 0);

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            // one 16-byte meta per 256 values; shared by the 8 sub-blocks that follow
            meta = __ldcs(meta_ptr + (size_t) kb * n);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const typename codec::record_t records[2] = {
            __ldcs(code_ptr + (size_t) (group + 0) * 32),
            __ldcs(code_ptr + (size_t) (group + 1) * 32),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
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

#undef Q4K_SKINNY_MMA_8N8K4

template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_multi_launch(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    // one CTA per 32-row tile of every segment
    q4k_skinny_multi_kernel<T, SplitK, NAcc, M1Only, RowTiles><<<p.n_main_tiles, 32 * SplitK, 0, stream>>>(p);
}

#define Q4K_SKINNY_MULTI_LAUNCH(NAcc, M1Only, RowTiles)                                             \
    do {                                                                                            \
        switch (config.split_k) {                                                                   \
            case 16: q4k_skinny_multi_launch<GGML_TYPE_Q4_K, 16, NAcc, M1Only, RowTiles>(p,         \
                         stream); break;                                                            \
            default: q4k_skinny_multi_launch<GGML_TYPE_Q4_K, 8, NAcc, M1Only, RowTiles>(p,          \
                         stream); break;                                                            \
        }                                                                                           \
    } while (0)

static void q4k_skinny_multi_mul_mat_launch(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    // same split-K and accumulator chains as q4k_skinny_mul_mat_launch, so every segment gets
    // the same result as a single-weight launch
    const q4k_skinny_config config = q4k_skinny_config_for(p.k);
    if (config.n_acc == 2) {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(2, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(2, false, 2); }
    } else {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(1, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q4K_SKINNY_MULTI_LAUNCH

bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                        int n_nodes, const ggml_tensor * src1) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 16 ||
            k % 256 != 0) {
        return false;
    }

    q4k_skinny_multi_params p = {};
    p.k = (int) k;
    p.m = (int) m;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        ggml_tensor * dst = dsts[i];
        const int64_t n = w->ne[1];
        if (!ggml_cuda_q4k_skinny_is_repacked(w) || w->view_src != nullptr || w->op != GGML_OP_NONE ||
                !ggml_is_contiguous(w) || w->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
        q4k_skinny_multi_seg & seg = p.seg[p.n_seg++];
        seg.codes = (const uint8_t *) w->data;
        seg.meta = seg.codes + n * k / 2;
        seg.n = (int) n;
        seg.dst = (float *) dst->data;
    }

    for (int s = 0; s < p.n_seg; ++s) {
        p.seg[s].first_tile = p.n_main_tiles;
        p.n_main_tiles += p.seg[s].n / 32;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());
    p.input = input.get();

    q4k_skinny_multi_mul_mat_launch(p, ctx.stream());
    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t) {
    return t->extra == (const void *) &q4k_skinny_marker;
}

bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(t);
    return false;
#else
    if (t->type != GGML_TYPE_Q4_K || t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
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
    // a whole number of 256-value super-blocks (k % 256 == 0) and whole 32-row tiles; k >= 256
    // keeps the split-K, the M=1 specialization and the to_f16 tile loop non-empty. Under
    // tensor split a device can hold an empty K slice, such a weight is never multiplied.
    return k >= 256 && k % 256 == 0 && n % 32 == 0 && n >= 32;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    const int64_t ntiles = n / 32;
    GGML_ASSERT(k >= 256 && ntiles >= 1);
    const size_t tile_src_bytes = 18 * (size_t) k;    // 32 rows of 9/16 bytes per element
    const size_t tile_dst_bytes = 16 * (size_t) k;    // 32 rows of 8 code bytes per 16 values
    const size_t meta_bytes = (size_t) n * k / 16;

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
    void * meta_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, tiles_per_chunk * tile_src_bytes));
    CUDA_CHECK(cudaMalloc(&meta_tmp, meta_bytes));

    // Chunk [t0, t1) writes codes to [t0*16K, t1*16K), which ends before the source
    // [t1*18K, ...) of the remaining tiles, and its own source was copied to staging first.
    // The meta region [N*K/2, N*K*9/16) overlaps the sources of the last tiles, so it is
    // written only after all codes are done.
    for (int64_t t0 = 0; t0 < ntiles; t0 += (int64_t) tiles_per_chunk) {
        const int64_t t1 = std::min(t0 + (int64_t) tiles_per_chunk, ntiles);
        CUDA_CHECK(cudaMemcpyAsync(staging, data + t0 * tile_src_bytes, (t1 - t0) * tile_src_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        q4k_skinny_repack_kernel<<<(unsigned) (t1 - t0), 256, 0, stream>>>(
            (const uint8_t *) staging, data + t0 * tile_dst_bytes, (uint8_t *) meta_tmp,
            (int) n, (int) k, (int) t0);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(data + (size_t) n * k / 2, meta_tmp, meta_bytes,
                               cudaMemcpyDeviceToDevice, stream));

    // no pool: the staging must not stay resident after the one-time repack
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(meta_tmp));

    t->extra = &q4k_skinny_marker;
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    // The idle counter is shared with the Q8_0 prepass, which controls the scan rate for both:
    // once it reaches 1024, neither prepass finds anything new, so the graph walk is skipped.
    // This prepass never modifies the counter so that the Q8_0 behaviour is unchanged.
    if (ctx.q8_skinny_idle_scans >= 1024) {
        return;
    }

    // candidates: Q4_K weights used as MUL_MAT src0; none in steady state
    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        ggml_tensor * src0 = node->src[0];
        if (!ggml_cuda_q4k_skinny_can_repack(src0) || ggml_cuda_q4k_skinny_is_repacked(src0)) {
            continue;
        }
        if (std::find(candidates.begin(), candidates.end(), src0) == candidates.end()) {
            candidates.push_back(src0);
        }
    }
    if (candidates.empty()) {
        return;
    }

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
        ggml_cuda_q4k_skinny_repack_inplace(ctx, t);
    }
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int n = (int) src0->ne[1];
    const int k = (int) src0->ne[0];
    const uint8_t * data = (const uint8_t *) src0->data;
    const dim3 grid((unsigned) (k / 256), (unsigned) (n / 32), 1);
    qskinny_to_f16_kernel<GGML_TYPE_Q4_K><<<grid, 256, 0, stream>>>(dst, data, n, k);
    CUDA_CHECK(cudaGetLastError());
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                        const ggml_tensor * up_w, const ggml_tensor * src1,
                                        ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(gate_w);
    GGML_UNUSED(up_w);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                        int n_nodes, const ggml_tensor * src1) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0s);
    GGML_UNUSED(dsts);
    GGML_UNUSED(n_nodes);
    GGML_UNUSED(src1);
    return false;
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
