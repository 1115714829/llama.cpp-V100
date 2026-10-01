// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// SM70 (Volta) D256 Split-D prefill attention, host side: dispatch predicate,
// scratch layout, K/V dequant (F16 read in place, Q8_0 or Q4_0 mirrored to f16),
// Q staging, mask pre-scan, kernel launch and O scatter. Device kernel
// provenance: fattn-sm70-d256-kernel.cuh.

// The HIP and MUSA backends glob ../ggml-cuda/*.cu into their own libraries and
// cannot compile CuTe/CUTLASS. This file is only used by the CUDA backend.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-sm70-d256.cuh"
#include "fattn-sm70-d256-kernel.cuh"

#include <climits>

// NOTE: no anonymous namespace in this file. The CuTe vendor headers
// (cute/atom/mma_traits_sm70.hpp) open their own anonymous namespaces; a second
// one in this TU makes cudafe's _GLOBAL__N__<hash> symbol mangling ambiguous.

// Natural-log scale -> exp2 domain, matching the kernel's softmax_scale_log2.
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634
#endif

constexpr int SM70_D256_BLOCK_M = 64;
constexpr int SM70_D256_D       = 256;

// Windowed KV mirror: when a range mask covers far fewer rows than the K/V
// view, the q8_0/q4_0 f16 mirror costs many hundreds of MB per device. The
// launcher then converts at most this many KV rows per window and runs the
// Partial kernel once per window. 131072 keeps the prefill speed of the unwindowed
// mirror while still capping it, and is a multiple of kBlockN (64).
#define SM70_D256_KV_WINDOW 131072
static_assert(SM70_D256_KV_WINDOW % SM70_D256_BLOCK_M == 0, "window is not a multiple of the q block");

// SplitKV2: when the CTAs of one KV window (or of the dense launch) do not
// fill the device (one CTA per SM, the kernel is register-bound), the KV range
// is split into this many segments and all run in one launch. The partial
// buffers are always sized for this many slices; one segment must hold at
// least SM70_D256_SPLIT_MIN_BLOCKS KV blocks or the range is left unsplit (the
// fixed merge cost is not worth it for short tails). The windowed path only
// ever sees a short last window: the first window is always
// SM70_D256_KV_WINDOW rows.
constexpr int SM70_D256_KV_SPLITS = 2;
constexpr int SM70_D256_SPLIT_MIN_BLOCKS = 32;

// All CC 7.0 devices (GV100) have 80 SMs. The split predicate needs the SM
// count at alloc_size time, which has no device id, so both the scratch layout
// and the launcher use this constant and can not diverge.
constexpr int SM70_D256_NSM = 80;

// Q f32 -> f16 staging. grid = (q_pad, batch*hkv, gqa), block = 128 threads
// (one float2 of the 256-wide row each). Qs layout, which is what the kernel
// reads: [batch][heads_q][q_pad][D] f16. Pad rows stay at the zero memset done
// by the launcher and are never scattered back.
__global__ void sm70_d256_stage_q_kernel(
        const float2 * __restrict__ src, half2 * __restrict__ dst,
        const int q_len, const int hkv, const int gqa, const int q_pad,
        const int64_t src_row, const int64_t src_head, const int64_t src_batch) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;
    const int c  = blockIdx.z;
    if (r >= q_len) {
        return;
    }
    const int j = bj % hkv;
    const int b = bj / hkv;
    const int head_q = j * gqa + c;
    const int heads_q = hkv * gqa;
    const float2 v = src[threadIdx.x
        + (int64_t) r * src_row
        + (int64_t) head_q * src_head
        + (int64_t) b * src_batch];
    dst[threadIdx.x
        + (int64_t) b * heads_q * q_pad * (SM70_D256_D/2)
        + (int64_t) head_q * q_pad * (SM70_D256_D/2)
        + (int64_t) r * (SM70_D256_D/2)] = __float22half2_rn(v);
}

// O staging [batch][row][heads_q][D] f32 -> dst, the llama.cpp FA output
// (D, heads_q, q_len, batch) f32 with its real nb strides. grid = (q_len,
// batch*heads_q), block = 128 threads (one float2 each). Only real rows are
// written, so the Q pad rows never reach dst. Normalize applies the windowed
// row normalization here instead of the removed finalize pass:
// out = sum > 0 ? acc * (1/sum) : 0, the same expression and order the
// finalize kernel used. acc_sum holds one sum per staging row.
template <bool Normalize = false>
__global__ void sm70_d256_scatter_kernel(
        const float2 * __restrict__ src, float2 * __restrict__ dst,
        const float * __restrict__ acc_sum,
        const int heads_q, const int q_pad,
        const int64_t dst_row, const int64_t dst_head, const int64_t dst_batch) {
    const int r  = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / heads_q;
    const int head_q = bh % heads_q;
    float2 v = src[threadIdx.x
        + (int64_t) b * q_pad * heads_q * (SM70_D256_D/2)
        + (int64_t) r * heads_q * (SM70_D256_D/2)
        + (int64_t) head_q * (SM70_D256_D/2)];
    if constexpr (Normalize) {
        const int64_t row = (int64_t) b * q_pad * heads_q
            + (int64_t) r * heads_q + head_q;
        const float sum = acc_sum[row];
        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        if (sum > 0.0f) {
            v.x *= inv_sum;
            v.y *= inv_sum;
        } else {
            v = make_float2(0.0f, 0.0f);
        }
    }
    dst[threadIdx.x
        + (int64_t) r * dst_row
        + (int64_t) head_q * dst_head
        + (int64_t) b * dst_batch] = v;
}

// Per q-block mask pre-scan. llama.cpp masks are causal plus tail padding: for
// one 64-row query block the mask is zero over the whole leading KV range and
// -inf over the trailing range, with only a few mixed KV blocks in between.
// One CTA scans one q block over one SM70_D256_MASK_SPLIT-column slice; the
// split CTAs of a q block combine with atomicMax into one int2 per (batch,
// q block):
//   .x = kv_len - first_nz, where first_nz is the first column with a non-zero
//        entry (0 when the whole q block is zero);
//   .y = last_fin + 1, where last_fin is the last column above -inf (0 when
//        the whole q block is all -inf).
// The dense kernel then starts its KV walk at the block that covers .y-1 and
// skips the explicit mask add for blocks entirely below first_nz.
static constexpr int SM70_D256_MASK_BLOCK_N = 32;
static constexpr int SM70_D256_MASK_SPLIT   = 8192;
static_assert(SM70_D256_MASK_BLOCK_N == FLASH_NAMESPACE::Sm70D256SplitDTraits::kBlockN,
              "mask bounds are in units of the kernel's KV block size");

__global__ void sm70_d256_mask_bounds_kernel(
        const __half * __restrict__ mask, int2 * __restrict__ bounds,
        const int q_len, const int kv_len,
        const int64_t mask_row_stride, const int64_t mask_batch_stride,
        const bool vec_ok) {
    const int q_block = blockIdx.x;
    const int batch   = blockIdx.y;
    const int split   = blockIdx.z;
    const int row_begin = q_block * SM70_D256_BLOCK_M;
    const int row_end = (row_begin + SM70_D256_BLOCK_M) < q_len
        ? (row_begin + SM70_D256_BLOCK_M) : q_len;
    const int col_begin = split * SM70_D256_MASK_SPLIT;
    const int col_end = (col_begin + SM70_D256_MASK_SPLIT) < kv_len
        ? (col_begin + SM70_D256_MASK_SPLIT) : kv_len;

    const __half * batch_mask = mask + batch * mask_batch_stride;
    int first_nz = kv_len;
    int last_fin = -1;
    for (int row = row_begin; row < row_end; ++row) {
        const __half * mask_row = batch_mask + (int64_t) row * mask_row_stride;
        // All 256 threads scan one row together: thread t covers columns
        // col_begin + 8*t + 2048*j, j = 0..3. The vec_ok alignment (checked on
        // the host) makes every 8-half group a 16-byte aligned uint4; the tail
        // of the last split falls back to scalars.
#pragma unroll
        for (int j = 0; j < SM70_D256_MASK_SPLIT / (8 * 256); ++j) {
            const int col = col_begin + 8 * threadIdx.x + j * (8 * 256);
            if (col >= col_end) {
                continue;
            }
            if (vec_ok && col + 8 <= col_end) {
                const uint4 raw = *reinterpret_cast<const uint4 *>(mask_row + col);
                const __half * h = reinterpret_cast<const __half *>(&raw);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = __half2float(h[i]);
                    if (v != 0.0f && col + i < first_nz) {
                        first_nz = col + i;
                    }
                    if (v > -INFINITY && col + i > last_fin) {
                        last_fin = col + i;
                    }
                }
            } else {
                const int col_last = col + 8 < col_end ? col + 8 : col_end;
                for (int c = col; c < col_last; ++c) {
                    const float v = __half2float(mask_row[c]);
                    if (v != 0.0f && c < first_nz) {
                        first_nz = c;
                    }
                    if (v > -INFINITY && c > last_fin) {
                        last_fin = c;
                    }
                }
            }
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        const int other_first = __shfl_down_sync(0xffffffffu, first_nz, off);
        const int other_last = __shfl_down_sync(0xffffffffu, last_fin, off);
        if (other_first < first_nz) {
            first_nz = other_first;
        }
        if (other_last > last_fin) {
            last_fin = other_last;
        }
    }
    __shared__ int smem_first[8];
    __shared__ int smem_last[8];
    if ((threadIdx.x & 31) == 0) {
        smem_first[threadIdx.x / 32] = first_nz;
        smem_last[threadIdx.x / 32] = last_fin;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) {
            if (smem_first[w] < first_nz) {
                first_nz = smem_first[w];
            }
            if (smem_last[w] > last_fin) {
                last_fin = smem_last[w];
            }
        }
        int2 & b = bounds[(int64_t) batch * gridDim.x + q_block];
        // atomicMax keeps the widest bound: the smallest first_nz and the
        // largest last_fin reported by the split CTAs.
        if (first_nz < kv_len) {
            atomicMax(&b.x, kv_len - first_nz);
        }
        if (last_fin >= 0) {
            atomicMax(&b.y, last_fin + 1);
        }
    }
}

// Per q-block bounds of a range mask, the same int2 the dense pre-scan produces (see above).
// One CTA per (q block, batch), one thread per row (blockDim.x == SM70_D256_BLOCK_M).
__global__ void sm70_d256_range_bounds_kernel(
        const int2 * __restrict__ range, int2 * __restrict__ bounds,
        int * __restrict__ kv_limit,
        const int q_len, const int kv_len,
        const int64_t range_row_stride, const int64_t range_batch_stride) {
    const int q_block = blockIdx.x;
    const int batch   = blockIdx.y;
    const int row     = q_block * SM70_D256_BLOCK_M + threadIdx.x;

    // Neutral element for rows outside q_len (Q pad rows).
    int first_nz = kv_len;
    int last_fin = -1;
    if (row < q_len) {
        const int2 r = range[(int64_t) batch * range_batch_stride + (int64_t) row * range_row_stride];
        const int lo = r.x < 0 ? 0 : (r.x > kv_len ? kv_len : r.x);
        const int hi = r.y < 0 ? 0 : (r.y > kv_len ? kv_len : r.y);
        if (hi > lo) {
            first_nz = lo > 0 ? 0 : (hi < kv_len ? hi : kv_len);
            last_fin = hi - 1;
        } else {
            // Empty row: every column is -inf, so column 0 is the first non-zero one.
            first_nz = 0;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        const int other_first = __shfl_down_sync(0xffffffffu, first_nz, off);
        const int other_last = __shfl_down_sync(0xffffffffu, last_fin, off);
        if (other_first < first_nz) {
            first_nz = other_first;
        }
        if (other_last > last_fin) {
            last_fin = other_last;
        }
    }
    __shared__ int smem_first[SM70_D256_BLOCK_M / 32];
    __shared__ int smem_last[SM70_D256_BLOCK_M / 32];
    if ((threadIdx.x & 31) == 0) {
        smem_first[threadIdx.x / 32] = first_nz;
        smem_last[threadIdx.x / 32] = last_fin;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < SM70_D256_BLOCK_M / 32; ++w) {
            if (smem_first[w] < first_nz) {
                first_nz = smem_first[w];
            }
            if (smem_last[w] > last_fin) {
                last_fin = smem_last[w];
            }
        }
        // Every (batch, q block) is written exactly once, no memset needed.
        bounds[(int64_t) batch * gridDim.x + q_block] = make_int2(kv_len - first_nz, last_fin + 1);

        // Widest KV row bound across all (batch, q block): the dense kernel only reads whole
        // kBlockN blocks, so round the last used row up. Empty rows contribute 0.
        const int rows = ((last_fin + 1 + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N) * SM70_D256_MASK_BLOCK_N;
        atomicMax(kv_limit, rows < kv_len ? rows : kv_len);
    }
}

// Scratch layout, carved from the graph extra region after dst->data:
//   [f16 K mirror][f16 V mirror]   (ggml_cuda_flash_attn_ext_get_f16_extra_data)
//   PAD(., 128)
//   [Qs f16: q_pad rows per (batch, head)][Os f32: same shape]
//   PAD(., 128)
//   [mask bounds int2: one per (batch, q block)]
//   PAD(., 128)
//   [kv_limit int: upper bound of the KV rows the bounds allow]
// windowed or split dense only:
//   [p_out f32: SM70_D256_KV_SPLITS slices of Qs shape]
//   [p_max f32: SM70_D256_KV_SPLITS slices of one f32 per row]
//   [p_sum f32: same]
//   [acc_max f32: one per row][acc_sum f32: one per row]
// alloc_size and the launcher both go through sm70_d256_get_scratch() so the
// offsets and the need_f16 predicates can not diverge. The partial buffers are
// always sized for SM70_D256_KV_SPLITS slices: alloc_size has no device id to
// evaluate the launcher's SM-count predicate with, and over-allocating cannot
// under-write.
struct sm70_d256_scratch {
    size_t total;            // full buffer size: nnbytes(dst) + extra
    size_t qs_offset;        // relative to dst->data + ggml_nbytes(dst)
    size_t os_offset;
    size_t bounds_offset;
    size_t kv_limit_offset;
    size_t p_out_offset;     // partial only
    size_t acc_max_offset;   // partial only
    size_t acc_sum_offset;   // partial only
    size_t p_max_offset;     // partial only
    size_t p_sum_offset;     // partial only
    size_t n_q;              // Qs/Os elements per buffer
    size_t rows;             // n_q / D: rows of the per-row partial buffers
    int q_pad;
    int n_q_blocks;          // q_pad / SM70_D256_BLOCK_M
    bool windowed;
    bool partial;            // launcher takes the Partial kernel (windowed or split dense)
    bool need_f16_k;
    bool need_f16_v;
    bool v_is_k_view;
    ggml_cuda_flash_attn_ext_f16_extra_data f16_extra;
};

static sm70_d256_scratch sm70_d256_get_scratch(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool v_is_k_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const bool need_f16_k = K->type != GGML_TYPE_F16;
    // V is a view of an f16 K: the K data itself serves as V.
    const bool need_f16_v = !(v_is_k_view && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16;
    // Fixed-size windowed mirror, only for a quantized K under a range mask
    // that spans more rows than one window.
    const bool windowed = mask != nullptr && mask->type == GGML_TYPE_I32
        && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0)
        && K->ne[1] > SM70_D256_KV_WINDOW;

    // The dense path (no fixed-size mirror) takes the same KV split when one
    // launch would not fill the device. kv_len follows the launcher: a range
    // mask spans the whole K/V view, an explicit mask only its own columns.
    const int q_pad = (int) GGML_PAD(Q->ne[1], SM70_D256_BLOCK_M);
    const int64_t kv_len = mask == nullptr ? 0
        : (mask->type == GGML_TYPE_I32 ? K->ne[1] : mask->ne[0]);
    const int64_t launch_ctas =
        (int64_t) (q_pad / SM70_D256_BLOCK_M) * Q->ne[3] * Q->ne[2];
    const int64_t kv_blocks =
        (kv_len + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N;
    const bool split_dense = !windowed
        && launch_ctas < 4 * SM70_D256_NSM
        && kv_blocks >= 2 * SM70_D256_SPLIT_MIN_BLOCKS;
    const bool partial = windowed || split_dense;

    sm70_d256_scratch s = {};
    s.windowed = windowed;
    s.partial = partial;
    s.need_f16_k = need_f16_k;
    s.need_f16_v = need_f16_v;
    s.v_is_k_view = v_is_k_view;

    const char * base = (const char *) dst->data + ggml_nbytes(dst);

    if (windowed) {
        // Same packing rule as ggml_cuda_flash_attn_ext_get_f16_extra_data
        // (PAD 128, V = K when V is a view of K), but each mirror holds
        // SM70_D256_KV_WINDOW rows per head instead of the full ne1.
        uintptr_t end = (uintptr_t) base;
        if (need_f16_k) {
            end = GGML_PAD(end, 128);
            s.f16_extra.K = end;
            end += (size_t) K->ne[0] * SM70_D256_KV_WINDOW * K->ne[2] * K->ne[3] * sizeof(half);
        }
        if (need_f16_v) {
            if (v_is_k_view) {
                s.f16_extra.V = s.f16_extra.K;
            } else {
                end = GGML_PAD(end, 128);
                s.f16_extra.V = end;
                end += (size_t) V->ne[0] * SM70_D256_KV_WINDOW * V->ne[2] * V->ne[3] * sizeof(half);
            }
        }
        s.f16_extra.end = end;
    } else {
        s.f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_k, need_f16_v);
    }

    const size_t f16_bytes = GGML_PAD((size_t) (s.f16_extra.end - (uintptr_t) base), 128);
    s.q_pad = q_pad;
    s.n_q = (size_t) Q->ne[2] * s.q_pad * SM70_D256_D * Q->ne[3];
    s.rows = s.n_q / SM70_D256_D;
    s.qs_offset = f16_bytes;
    s.os_offset = s.qs_offset + s.n_q * sizeof(half);
    s.n_q_blocks = s.q_pad / SM70_D256_BLOCK_M;
    s.bounds_offset = GGML_PAD(s.os_offset + s.n_q * sizeof(float), 128);
    s.kv_limit_offset = GGML_PAD(s.bounds_offset + (size_t) Q->ne[3] * s.n_q_blocks * sizeof(int2), 128);
    if (partial) {
        s.p_out_offset   = GGML_PAD(s.kv_limit_offset + sizeof(int), 128);
        s.p_max_offset   = GGML_PAD(s.p_out_offset + SM70_D256_KV_SPLITS * s.n_q * sizeof(float), 128);
        s.p_sum_offset   = GGML_PAD(s.p_max_offset + SM70_D256_KV_SPLITS * s.rows * sizeof(float), 128);
        s.acc_max_offset = GGML_PAD(s.p_sum_offset + SM70_D256_KV_SPLITS * s.rows * sizeof(float), 128);
        s.acc_sum_offset = GGML_PAD(s.acc_max_offset + s.rows * sizeof(float), 128);
        s.total = ggml_nbytes(dst) + GGML_PAD(s.acc_sum_offset + s.rows * sizeof(float), 128);
    } else {
        s.total = ggml_nbytes(dst) + s.kv_limit_offset + sizeof(int);
    }
    return s;
}

// K/V -> f16 mirror, the same to_fp16_nc conversion the stock launch_fattn
// uses for strided sources. The mirror is packed canonically as
// [batch][hkv][kv][D] (row stride D, head stride kv*D): to_fp16_nc writes
// output index ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0. The linear to_fp16() variant
// is deliberately not used: it preserves the source memory order, which for a
// packed permuted view is not the canonical one.
static void sm70_d256_dequant_kv_tensor(
        const ggml_tensor * t, half * dst, cudaStream_t stream) {
    const size_t ts = ggml_type_size(t->type);
    const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
    to_fp16(t->data, dst, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[1] / ts, t->nb[2] / ts, t->nb[3] / ts, stream);
}

// Block layout of the quantized K/V types the partial mirror supports. QK8_0 == QK4_0 == 32
// values per block, so one lane of the row kernel decodes 8 values from one block.
template <ggml_type type>
struct sm70_d256_dequant_block;

template <>
struct sm70_d256_dequant_block<GGML_TYPE_Q8_0> {
    static constexpr int kBlockBytes = sizeof(block_q8_0);

    // iq is the lane's 8-value group inside the block, d its scale.
    static __device__ __forceinline__ void decode(const char * block, const int iq, const float d, half2 * h) {
        const int8_t * qs = reinterpret_cast<const int8_t *>(block) + sizeof(__half);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            h[i] = __floats2half2_rn(d * qs[iq + 2*i], d * qs[iq + 2*i + 1]);
        }
    }
};

template <>
struct sm70_d256_dequant_block<GGML_TYPE_Q4_0> {
    static constexpr int kBlockBytes = sizeof(block_q4_0);

    static __device__ __forceinline__ __half2 pair(const uint32_t packed, const int shift, const float d) {
        return __floats2half2_rn(d * ((int) ((packed >> shift) & 0xf) - 8),
                                 d * ((int) ((packed >> (8 + shift)) & 0xf) - 8));
    }

    // q4_0 packs values 0..15 into the low nibbles of qs[0..15] and 16..31 into the high nibbles.
    static __device__ __forceinline__ void decode(const char * block, const int iq, const float d, half2 * h) {
        // The 18 B block makes qs only 2 B aligned, so assemble each u32 code word from two
        // u16 loads and shift the nibbles out.
        const char * codes = block + sizeof(__half) + (iq & 15);
        const uint32_t w0 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes)
                          | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 2) << 16;
        const uint32_t w1 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 4)
                          | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 6) << 16;
        const int shift = iq >= 16 ? 4 : 0;
        h[0] = pair(w0,      shift, d);
        h[1] = pair(w0 >> 16, shift, d);
        h[2] = pair(w1,      shift, d);
        h[3] = pair(w1 >> 16, shift, d);
    }
};

// q8_0/q4_0 -> f16 mirror of the rows [0, *kv_limit) only. With a range mask the dense kernel never reads
// a KV row at or past kv_limit, so a K/V view much wider than the attended range (a prompt ubatch that
// attends the whole cache) costs no conversion. Output layout as sm70_d256_dequant_kv:
// [ne3][ne2][win_rows][ne0] contiguous, i.e. ((i3*ne2 + i2)*win_rows + i1_local)*ne0 + i0, where i1_local
// is the row inside the window and i1 = row_lo + i1_local the source row. The full-tensor call passes
// row_lo = 0, win_rows = ne1. One warp per row, grid.y = ne2, grid.z = ne3, rows i1 >= *kv_limit are skipped.
template <ggml_type type>
static __global__ void sm70_d256_dequant_rows(
        const char * __restrict__ src, half * __restrict__ dst,
        const int ne0, const int ne2,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int * __restrict__ kv_limit,
        const int row_lo, const int win_rows) {
    static_assert(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0, "unsupported KV type");

    const int i1_local = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i1_local >= win_rows) {
        return;
    }
    const int i1 = row_lo + i1_local;
    if (i1 >= *kv_limit) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    // lane's block of the row and its 8-value group inside the block
    const int ib = lane / 4;
    const int iq = (lane % 4) * 8;

    const char * src_row = src + (int64_t) i3*nb3 + (int64_t) i2*nb2 + (int64_t) i1*nb1;
    half * dst_row = dst + (((int64_t) i3*ne2 + i2)*win_rows + i1_local)*ne0;

    const char * src_block = src_row + ib * sm70_d256_dequant_block<type>::kBlockBytes;
    const float d = __half2float(*reinterpret_cast<const __half *>(src_block));
    half2 h[4];
    sm70_d256_dequant_block<type>::decode(src_block, iq, d, h);
    uint4 out;
    memcpy(&out.x, &h[0], sizeof(uint32_t));
    memcpy(&out.y, &h[1], sizeof(uint32_t));
    memcpy(&out.z, &h[2], sizeof(uint32_t));
    memcpy(&out.w, &h[3], sizeof(uint32_t));
    *reinterpret_cast<uint4 *>(dst_row + ib*32 + iq) = out;
}

// One window [row_lo, row_lo + win_rows) of one q8_0/q4_0 K/V tensor into its
// fixed-size mirror; rows at or past *kv_limit are skipped on the device, so an
// empty window is a no-op.
static void sm70_d256_dequant_kv_window(
        const ggml_tensor * t, half * dst, const int * kv_limit,
        const int row_lo, const int win_rows, cudaStream_t stream) {
    GGML_ASSERT(t->ne[0] == SM70_D256_D);
    GGML_ASSERT(row_lo >= 0 && row_lo % 8 == 0 && win_rows > 0);
    const dim3 grid((unsigned) ((win_rows + 7) / 8), (unsigned) t->ne[2], (unsigned) t->ne[3]);
    if (t->type == GGML_TYPE_Q8_0) {
        sm70_d256_dequant_rows<GGML_TYPE_Q8_0><<<grid, 256, 0, stream>>>(
            (const char *) t->data, dst,
            (int) t->ne[0], (int) t->ne[2],
            t->nb[1], t->nb[2], t->nb[3], kv_limit, row_lo, win_rows);
    } else {
        sm70_d256_dequant_rows<GGML_TYPE_Q4_0><<<grid, 256, 0, stream>>>(
            (const char *) t->data, dst,
            (int) t->ne[0], (int) t->ne[2],
            t->nb[1], t->nb[2], t->nb[3], kv_limit, row_lo, win_rows);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Full mirror of one q8_0/q4_0 K/V tensor: rows [0, *kv_limit) only, bounds known host side.
static void sm70_d256_dequant_kv_partial(
        const ggml_tensor * t, half * dst, const int * kv_limit, cudaStream_t stream) {
    sm70_d256_dequant_kv_window(t, dst, kv_limit, 0, (int) t->ne[1], stream);
}

// F16 is read in place: contiguous rows plus 16 B aligned strides for the
// 128-bit global loads. The q8_0/q4_0 mirrors need block-aligned source strides.
static bool sm70_d256_kv_type_ok(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        return t->nb[0] == sizeof(half) && t->nb[1] % 16 == 0 &&
               t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    }
    if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        const size_t ts = ggml_type_size(t->type);
        return t->nb[0] == ts && t->nb[1] % ts == 0 &&
               t->nb[2] % ts == 0 && t->nb[3] % ts == 0;
    }
    return false;
}

bool ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA || !volta_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != SM70_D256_D || K->ne[0] != SM70_D256_D || V->ne[0] != SM70_D256_D) {
        return false;
    }
    // Rows below 17 stay on the grouped/vec kernels.
    if (Q->ne[1] < 17) {
        return false;
    }
    // Under tensor split a device can get 0 KV heads (fewer KV heads than devices); fall back to the generic path.
    if (K->ne[2] == 0 || Q->ne[2] == 0) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || K->ne[2] != V->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    // The kernel consumes the explicit mask (f16, or an I32 range); a missing mask
    // would silently drop llama.cpp's causal/padding semantics.
    const bool mask_range = mask && mask->type == GGML_TYPE_I32;
    if (!mask || (mask->type != GGML_TYPE_F16 && !mask_range) || mask->ne[2] != 1) {
        return false;
    }
    if (mask_range) {
        // [lo, hi) pairs read as int2
        if (mask->ne[0] != 2 || mask->nb[0] != sizeof(int32_t) || mask->nb[1] != sizeof(int2) || (uintptr_t) mask->data % sizeof(int2) != 0) {
            return false;
        }
    } else if (mask->nb[0] != sizeof(half)) {
        return false;
    }
    if (mask->ne[1] != Q->ne[1]) {
        return false;
    }
    if (mask->ne[3] != Q->ne[3] && mask->ne[3] != 1) {
        return false;
    }
    // kv_len is the real length; the K/V views must cover it. A range mask spans the whole K/V view.
    const int64_t kv_len = mask_range ? K->ne[1] : mask->ne[0];
    if (kv_len < Q->ne[1] || kv_len > K->ne[1] || kv_len > V->ne[1]) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) { // sparse KV hint
        return false;
    }
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    if (!(scale > 0.0f)) {
        return false;
    }
    // Q is staged through f32 rows read as float2 (sm70_d256_stage_q_kernel),
    // so the strides only need float2 granularity plus 8-byte data alignment.
    // llama.cpp permutes Q to (D, heads_q, tokens, batch): the token stride is
    // then D * heads_q * 4 bytes, not the canonical row stride.
    if (Q->nb[0] != sizeof(float) ||
        Q->nb[1] % sizeof(float2) != 0 || Q->nb[2] % sizeof(float2) != 0 ||
        Q->nb[3] % sizeof(float2) != 0 || (uintptr_t) Q->data % 8 != 0) {
        return false;
    }
    if (!sm70_d256_kv_type_ok(K) || !sm70_d256_kv_type_ok(V)) {
        return false;
    }
    return true;
}

// Raw launcher for the dense staged-f16 path: packed f16 Q (q_pad rows), f16
// K/V, f16 or I32-range mask, f32 output. q_len_real = kv_len - kv_offset;
// q_pad >= q_len_real and q_pad % 64 == 0. mask_scale = 1/scale (the mask is
// added to the raw QK domain, see the kernel). mask/mask_bounds may be null (no
// mask). With mask_is_range the mask strides are in int2 units, else in halfs.
// Partial selects the partial variant, which walks only the KV blocks
// [win_block_lo, win_block_hi) and writes raw max/sum + unnormalized O to the
// partial buffers. kv_splits > 1 (Partial only) splits that block range into
// kv_splits equal segments, one per grid.y slice; each segment writes its own
// partial slice. The public wrapper below uses the dense variant only when the
// launch is not split.
template <bool Partial>
static void sm70_d256_launch_dense(
        const void * q, const void * k, const void * v, void * out,
        const void * mask, const int2 * mask_bounds,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_outer_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_outer_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t mask_row_stride, int64_t mask_batch_stride, bool mask_is_range,
        int q_pad, int kv_len, int heads_q, int heads_kv, int batch, int kv_offset,
        float softmax_scale, float mask_scale,
        int win_block_lo, int win_block_hi, int kv_splits,
        float * partial_out, float * partial_max, float * partial_sum,
        cudaStream_t stream) {
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    using El = cutlass::half_t;

    GGML_ASSERT(q_pad % Traits::kBlockM == 0);
    GGML_ASSERT(kv_len - kv_offset >= 1 && kv_len - kv_offset <= q_pad);
    if constexpr (Partial) {
        GGML_ASSERT(win_block_lo % Traits::kBlockN == 0);
        GGML_ASSERT(kv_splits >= 1 && kv_splits <= SM70_D256_KV_SPLITS);
        GGML_ASSERT(partial_out != nullptr && partial_max != nullptr && partial_sum != nullptr);
    } else {
        GGML_ASSERT(kv_splits == 1);
    }

    auto kernel = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Traits, El, float, false, Partial>;
    CUDA_SET_SHARED_MEMORY_LIMIT((const void *) kernel, Traits::kSmemBytes);

    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / Traits::kBlockM, batch * kv_splits, heads_q);
    kernel<<<grid, block, Traits::kSmemBytes, stream>>>(
        (const El *) q, (const El *) k, (const El *) v, (float *) out,
        (const __half *) mask,
        (int) q_batch_stride, (int) q_row_stride, (int) q_head_stride,
        (int) k_outer_stride, (int) k_row_stride, (int) k_head_stride,
        (int) v_outer_stride, (int) v_row_stride, (int) v_head_stride,
        mask_row_stride, mask_batch_stride, mask_is_range, (const int2 *) mask_bounds,
        q_pad, kv_len, heads_q, heads_kv, kv_offset,
        softmax_scale * float(M_LOG2E), mask_scale,
        partial_out, partial_max, partial_sum,
        win_block_lo, win_block_hi, kv_splits);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_sm70_d256_launch_raw(
        const void * q, const void * k, const void * v, void * out,
        const void * mask, const int2 * mask_bounds,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_outer_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_outer_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t mask_row_stride, int64_t mask_batch_stride, bool mask_is_range,
        int q_pad, int kv_len, int heads_q, int heads_kv, int batch, int kv_offset,
        float softmax_scale, float mask_scale, cudaStream_t stream) {
    sm70_d256_launch_dense<false>(
        q, k, v, out, mask, mask_bounds,
        q_batch_stride, q_row_stride, q_head_stride,
        k_outer_stride, k_row_stride, k_head_stride,
        v_outer_stride, v_row_stride, v_head_stride,
        mask_row_stride, mask_batch_stride, mask_is_range,
        q_pad, kv_len, heads_q, heads_kv, batch, kv_offset,
        softmax_scale, mask_scale,
        /*win_block_lo*/ 0, /*win_block_hi*/ INT_MAX, /*kv_splits*/ 1,
        /*partial_out*/ nullptr, /*partial_max*/ nullptr, /*partial_sum*/ nullptr,
        stream);
}

// ---------------------------------------------------------------------------
// Meta assist: cross-device epochs, idle-rank workspace and idle-rank run
// ---------------------------------------------------------------------------

// Cross-device epoch flags. The owner bumps its own epoch after the data it produced (Q staging)
// or consumed (raw partials) is visible; the waiting rank spins on that epoch before it touches
// the data. __threadfence_system orders the data accesses (peer visible) against the flag update.
static __global__ void sm70_d256_assist_signal(uint32_t * epoch) {
    __threadfence_system();
    atomicAdd(epoch, 1);
}

// One thread spins until the epoch changes, then records it. `last` lives on the waiting device,
// `epoch` may live on a peer device; the volatile read keeps the poll in the spin loop. The
// fences order the data reads/writes around the flag access.
static __global__ void sm70_d256_assist_wait(volatile uint32_t * epoch, uint32_t * last) {
    if (threadIdx.x != 0) {
        return;
    }
    __threadfence_system();
    const uint32_t seen = *last;
    if (*epoch == seen) {
        while (*epoch == seen) {
        }
    }
    __threadfence_system();
    *last = *epoch;
}

static void sm70_d256_assist_signal_launch(uint32_t * epoch, cudaStream_t stream) {
    sm70_d256_assist_signal<<<1, 1, 0, stream>>>(epoch);
    CUDA_CHECK(cudaGetLastError());
}

static void sm70_d256_assist_wait_launch(volatile uint32_t * epoch, uint32_t * last, cudaStream_t stream) {
    sm70_d256_assist_wait<<<1, 1, 0, stream>>>(epoch, last);
    CUDA_CHECK(cudaGetLastError());
}

// Idle-rank workspace, one block per device reused by every layer. All offsets are relative to
// the buffer base, padded to 128 B:
//   [q8 K staging: one window][q8 V staging: one window]
//   [f16 K mirror: one window][f16 V mirror: one window]
//   [Qs f16 copy of the owner staging][mask bounds int2][kv_limit int]
//   [window O f32: q_pad*heads_q rows][window max][window sum]
struct sm70_d256_assist_workspace {
    size_t total;
    size_t stage_k;
    size_t stage_v;
    size_t mirror_k;
    size_t mirror_v;
    size_t qs;
    size_t bounds;
    size_t kv_limit;
    size_t win_out;
    size_t win_max;
    size_t win_sum;
    int64_t rows;      // q_pad * heads_q
    int q_pad;
    int q_blocks;      // q_pad / SM70_D256_BLOCK_M
};

static sm70_d256_assist_workspace sm70_d256_assist_get_workspace(int q_pad, int heads_q, int kv_type) {
    sm70_d256_assist_workspace w = {};
    const size_t row_bytes = ggml_row_size((enum ggml_type) kv_type, SM70_D256_D);
    const size_t staging = (size_t) SM70_D256_KV_WINDOW * row_bytes;
    const size_t mirror = (size_t) SM70_D256_KV_WINDOW * SM70_D256_D * sizeof(half);
    w.rows = (int64_t) q_pad * heads_q;
    w.q_pad = q_pad;
    w.q_blocks = q_pad / SM70_D256_BLOCK_M;

    size_t off = 0;
    off = GGML_PAD(off, 128); w.stage_k  = off; off += staging;
    off = GGML_PAD(off, 128); w.stage_v  = off; off += staging;
    off = GGML_PAD(off, 128); w.mirror_k = off; off += mirror;
    off = GGML_PAD(off, 128); w.mirror_v = off; off += mirror;
    off = GGML_PAD(off, 128); w.qs       = off; off += (size_t) w.rows * SM70_D256_D * sizeof(half);
    off = GGML_PAD(off, 128); w.bounds   = off; off += (size_t) w.q_blocks * sizeof(int2);
    off = GGML_PAD(off, 128); w.kv_limit = off; off += sizeof(int);
    off = GGML_PAD(off, 128); w.win_out  = off; off += (size_t) w.rows * SM70_D256_D * sizeof(float);
    off = GGML_PAD(off, 128); w.win_max  = off; off += (size_t) w.rows * sizeof(float);
    off = GGML_PAD(off, 128); w.win_sum  = off; off += (size_t) w.rows * sizeof(float);
    w.total = GGML_PAD(off, 128);
    return w;
}

size_t ggml_cuda_sm70_d256_assist_workspace_size(int q_pad, int heads_q) {
    // q8_0 gives the largest staging rows of the supported KV types, so size for it
    return sm70_d256_assist_get_workspace(q_pad, heads_q, GGML_TYPE_Q8_0).total;
}

// One window of the q8_0/q4_0 staging rows -> the local f16 mirror. `src` is the staging base,
// shifted back by the segment start so the kernel row indices stay global (as in the owner's
// windowed path); rows at or past *kv_limit are skipped on the device.
template <ggml_type type>
static void sm70_d256_dequant_staging_window(
        const char * src, half * dst, const int * kv_limit,
        const int row_lo, const int win_rows, const int64_t row_bytes, cudaStream_t stream) {
    const dim3 grid((unsigned) ((win_rows + 7) / 8), 1, 1);
    sm70_d256_dequant_rows<type><<<grid, 256, 0, stream>>>(
        src, dst, SM70_D256_D, 1, row_bytes, 0, 0, kv_limit, row_lo, win_rows);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_sm70_d256_assist_run(void * backend, const ggml_backend_meta_assist_rank * d) {
    if (backend == nullptr || d == nullptr || d->role != 1 || d->work == nullptr) {
        return false;
    }
    if (d->n_owners < 1 || d->n_owners > 2 || d->batch != 1 || d->heads_kv != 1 || d->mask == nullptr || !d->mask_is_range) {
        return false;
    }
    if (d->q_len < GGML_BACKEND_META_ASSIST_MIN_Q || d->kv_len < GGML_BACKEND_META_ASSIST_MIN_KV) {
        return false;
    }
    if (d->q_pad <= 0 || d->q_pad % SM70_D256_BLOCK_M != 0 || d->heads_q <= 0 || d->row0 < 0 || d->n_rows <= 0 ||
        d->row0 + d->n_rows > d->kv_len) {
        return false;
    }
    if (d->kv_type != GGML_TYPE_Q8_0 && d->kv_type != GGML_TYPE_Q4_0) {
        return false;
    }
    ggml_backend_t be = (ggml_backend_t) backend;
    if (!ggml_backend_is_cuda(be)) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) be->context;
    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    const sm70_d256_assist_workspace w = sm70_d256_assist_get_workspace(d->q_pad, d->heads_q, d->kv_type);
    GGML_ASSERT(d->work_bytes >= w.total);
    GGML_ASSERT((uintptr_t) d->work % 128 == 0);

    char * const work = (char *) d->work;
    char * const stage_k  = work + w.stage_k;
    char * const stage_v  = work + w.stage_v;
    half * const mirror_k = (half *) (work + w.mirror_k);
    half * const mirror_v = (half *) (work + w.mirror_v);
    half * const qs       = (half *) (work + w.qs);
    int2 * const bounds   = (int2 *) (work + w.bounds);
    int  * const kv_limit = (int *) (work + w.kv_limit);
    float * const win_out = (float *) (work + w.win_out);
    float * const win_max = (float *) (work + w.win_max);
    float * const win_sum = (float *) (work + w.win_sum);

    const int64_t row_bytes = ggml_row_size((enum ggml_type) d->kv_type, SM70_D256_D);
    const int64_t qs_bytes = (int64_t) d->heads_q * d->q_pad * SM70_D256_D * sizeof(half);
    const int n_win = (int) ((d->n_rows + SM70_D256_KV_WINDOW - 1) / SM70_D256_KV_WINDOW);
    const unsigned n_merge = (unsigned) ((w.rows
        + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
        / FLASH_NAMESPACE::kWindowMergeRowsPerCta);

    // Two owners are processed back to back on the one stream; the owner cards run in parallel
    // with each other, but a single idle device serves both, so the copies and kernels of the
    // second owner are simply enqueued after the first owner's epoch signalled.
    for (int j = 0; j < d->n_owners; ++j) {
        GGML_ASSERT(d->k_src[j] != nullptr && d->v_src[j] != nullptr && d->qs_src[j] != nullptr);
        GGML_ASSERT(d->p_out_dst[j] != nullptr && d->p_max_dst[j] != nullptr && d->p_sum_dst[j] != nullptr);
        GGML_ASSERT(d->epoch_kv[j] != nullptr && d->epoch_partial[j] != nullptr && d->epoch_free[j] != nullptr);
        GGML_ASSERT(d->last_kv[j] != nullptr && d->last_free[j] != nullptr);

        // The owner's Q staging and its K/V cache writes are complete once it bumps this epoch.
        sm70_d256_assist_wait_launch(d->epoch_kv[j], d->last_kv[j], stream);

        CUDA_CHECK(cudaMemcpyPeerAsync(qs, ctx->device, d->qs_src[j], d->owner_dev[j], (size_t) qs_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));
        {
            const dim3 grid(w.q_blocks, d->batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int2 *) d->mask, bounds, kv_limit,
                d->q_len, d->kv_len, d->mask_row_stride, d->mask_batch_stride);
            CUDA_CHECK(cudaGetLastError());
        }

        for (int wi = 0; wi < n_win; ++wi) {
            const int64_t row_lo = d->row0 + (int64_t) wi * SM70_D256_KV_WINDOW;
            const int64_t rows_left = d->row0 + d->n_rows - row_lo;
            const int win_rows = (int) (rows_left < SM70_D256_KV_WINDOW ? rows_left : SM70_D256_KV_WINDOW);

            CUDA_CHECK(cudaMemcpyPeerAsync(stage_k, ctx->device,
                (const char *) d->k_src[j] + row_lo * row_bytes, d->owner_dev[j],
                (size_t) win_rows * row_bytes, stream));
            CUDA_CHECK(cudaMemcpyPeerAsync(stage_v, ctx->device,
                (const char *) d->v_src[j] + row_lo * row_bytes, d->owner_dev[j],
                (size_t) win_rows * row_bytes, stream));

            if (d->kv_type == GGML_TYPE_Q8_0) {
                sm70_d256_dequant_staging_window<GGML_TYPE_Q8_0>(
                    stage_k - row_lo * row_bytes, mirror_k, kv_limit, (int) row_lo, win_rows, row_bytes, stream);
                sm70_d256_dequant_staging_window<GGML_TYPE_Q8_0>(
                    stage_v - row_lo * row_bytes, mirror_v, kv_limit, (int) row_lo, win_rows, row_bytes, stream);
            } else {
                sm70_d256_dequant_staging_window<GGML_TYPE_Q4_0>(
                    stage_k - row_lo * row_bytes, mirror_k, kv_limit, (int) row_lo, win_rows, row_bytes, stream);
                sm70_d256_dequant_staging_window<GGML_TYPE_Q4_0>(
                    stage_v - row_lo * row_bytes, mirror_v, kv_limit, (int) row_lo, win_rows, row_bytes, stream);
            }

            // The owner may still be reading its partial slots of the previous execution; wait
            // before the first write of this execution (the first merge below). On the very
            // first execution the free epoch is still the initial value, so this passes.
            if (wi == 0) {
                sm70_d256_assist_wait_launch(d->epoch_free[j], d->last_free[j], stream);
            }

            const half * const K_win = mirror_k - row_lo * SM70_D256_D;
            const half * const V_win = mirror_v - row_lo * SM70_D256_D;
            const int64_t kv_head_stride = (int64_t) win_rows * SM70_D256_D;
            const int64_t kv_outer_stride = (int64_t) d->heads_kv * win_rows * SM70_D256_D;

            sm70_d256_launch_dense<true>(
                qs, K_win, V_win, win_out, d->mask, bounds,
                /*q_batch_stride */ (int64_t) d->heads_q * d->q_pad * SM70_D256_D,
                /*q_row_stride   */ SM70_D256_D,
                /*q_head_stride  */ (int64_t) d->q_pad * SM70_D256_D,
                kv_outer_stride, SM70_D256_D, kv_head_stride,
                kv_outer_stride, SM70_D256_D, kv_head_stride,
                d->mask_row_stride, d->mask_batch_stride, true,
                d->q_pad, d->kv_len, d->heads_q, d->heads_kv, d->batch, d->kv_len - d->q_len,
                d->scale, 1.0f / d->scale,
                (int) (row_lo / SM70_D256_MASK_BLOCK_N),
                (int) ((row_lo + win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N),
                1,
                win_out, win_max, win_sum,
                stream);

            // Merge this window into the owner's slot as one more partial; window 0 initializes
            // the owner slot (first = true), later windows fold into it. No local accumulator and
            // no final copy: the merged partial is what the owner waits for.
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                d->p_out_dst[j], d->p_max_dst[j], d->p_sum_dst[j],
                win_out, win_max, win_sum,
                w.rows, wi == 0, d->scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        }

        sm70_d256_assist_signal_launch(d->epoch_partial[j], stream);
    }
    return true;
}

size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst) {
    return sm70_d256_get_scratch(dst).total;
}

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool mask_range = mask->type == GGML_TYPE_I32;

    const int q_len   = (int) Q->ne[1];
    const int kv_len  = mask_range ? (int) K->ne[1] : (int) mask->ne[0];
    const int heads_q = (int) Q->ne[2];
    const int hkv     = (int) K->ne[2];
    if (heads_q == 0 || hkv == 0) {
        return; // defensive: the dispatch predicate already rejects 0 heads
    }
    const int gqa     = heads_q / hkv;
    const int batch   = (int) Q->ne[3];
    const int q_pad   = (int) GGML_PAD(q_len, SM70_D256_BLOCK_M);
    const int kv_offset = kv_len - q_len;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const sm70_d256_scratch scratch = sm70_d256_get_scratch(dst);
    char * base = (char *) dst->data + ggml_nbytes(dst);
    half  * Qs = (half  *) (base + scratch.qs_offset);
    float * Os = (float *) (base + scratch.os_offset);
    int2  * mask_bounds = (int2 *) (base + scratch.bounds_offset);
    int   * kv_limit = (int *) (base + scratch.kv_limit_offset);

    cudaStream_t stream = ctx.stream();

    // Owner side of a meta assist: the assist rank covers the KV tail [n_rows, kv_len), this
    // launcher limits its window loop to [0, n_rows), signals the staged Q, then merges the
    // assist partial and signals the partial slot as free for the next execution.
    const ggml_backend_meta_assist_rank * const assist = ctx.assist;
    const bool assist_owner = assist != nullptr && assist->role == 0;
    if (assist_owner && !scratch.windowed) {
        // the meta backend only builds assist when the KV range needs the windowed path
        GGML_ABORT("%s: assist owner requires the windowed partial path", __func__);
    }

    if (scratch.windowed) {
        // Fixed-size mirror: convert and attend one KV window at a time and
        // merge each window's raw partials into the Os/acc_max/acc_sum
        // accumulator. The first window of an unsplit launch writes the
        // accumulator directly; otherwise the partial slices merge in.
        if (assist_owner) {
            GGML_ASSERT(assist->batch == 1 && assist->q_pad == q_pad && assist->q_len == q_len);
            GGML_ASSERT(assist->heads_q == heads_q && assist->kv_len == kv_len);
            GGML_ASSERT(assist->n_rows > 0 && assist->n_rows < K->ne[1]);
            GGML_ASSERT(assist->p_out_local != nullptr && assist->p_max_local != nullptr && assist->p_sum_local != nullptr);
            GGML_ASSERT(assist->epoch_kv_local != nullptr && assist->epoch_partial_local != nullptr);
            GGML_ASSERT(assist->last_partial_local != nullptr && assist->epoch_free_local != nullptr);
        }
        CUDA_CHECK(cudaMemsetAsync(Qs, 0, scratch.n_q * sizeof(half), stream));
        // the bounds kernel raises it to the rows its bounds allow
        CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));

        {
            const dim3 grid(q_pad, batch * hkv, gqa);
            sm70_d256_stage_q_kernel<<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Q->data, (half2 *) Qs,
                q_len, hkv, gqa, q_pad,
                Q->nb[1] / sizeof(float2), Q->nb[2] / sizeof(float2), Q->nb[3] / sizeof(float2));
            CUDA_CHECK(cudaGetLastError());
        }

        // The assist rank needs the staged Q; the K/V cache rows are written by earlier nodes
        // on this stream, so stream order already makes them visible to the copy engine.
        if (assist_owner) {
            sm70_d256_assist_signal_launch(assist->epoch_kv_local, stream);
        }

        {
            const dim3 grid(scratch.n_q_blocks, batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int2 *) mask->data, mask_bounds, kv_limit,
                q_len, kv_len,
                mask->nb[1] / sizeof(int2),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int2));
            CUDA_CHECK(cudaGetLastError());
        }

        const half * const K_mirror = (const half *) scratch.f16_extra.K;
        GGML_ASSERT(K_mirror != nullptr);
        float * const p_out   = (float *) (base + scratch.p_out_offset);
        float * const acc_max = (float *) (base + scratch.acc_max_offset);
        float * const acc_sum = (float *) (base + scratch.acc_sum_offset);
        float * const p_max   = (float *) (base + scratch.p_max_offset);
        float * const p_sum   = (float *) (base + scratch.p_sum_offset);

        // One window launches q_pad/64 * batch * heads_q CTAs and the kernel
        // is register-bound to 1 CTA/SM. SplitKV2 splits each window's KV
        // range in two when that does not fill the device (4 CTAs per SM is
        // the point where the tail wave stops dominating).
        const int m_blocks = q_pad / SM70_D256_BLOCK_M;
        const int window_ctas = m_blocks * batch * heads_q;
        const bool split_window = window_ctas < 4 * SM70_D256_NSM;

        // In assist owner mode only the head segment [0, n_rows) is computed here; the assist
        // rank owns [n_rows, kv_len).
        const int64_t n_kv_rows = assist_owner ? assist->n_rows : K->ne[1];
        const int n_win = (int) ((n_kv_rows + SM70_D256_KV_WINDOW - 1) / SM70_D256_KV_WINDOW);
        for (int w = 0; w < n_win; ++w) {
            const int64_t row_lo = (int64_t) w * SM70_D256_KV_WINDOW;
            const int64_t rows_left = n_kv_rows - row_lo;
            const int win_rows = (int) (rows_left < SM70_D256_KV_WINDOW ? rows_left : SM70_D256_KV_WINDOW);

            // Both segments run in one launch (grid.y = batch * kv_splits);
            // a segment with no visible KV block zeroes its max/sum slice and
            // returns. A short window stays unsplit: the per-window merge cost
            // is fixed, so splitting it is not worth the extra slice pass.
            const int win_blocks =
                (win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N;
            const int kv_splits = split_window
                && win_blocks >= 2 * SM70_D256_SPLIT_MIN_BLOCKS
                ? SM70_D256_KV_SPLITS : 1;

            // Convert this window; rows past *kv_limit are skipped on the device.
            sm70_d256_dequant_kv_window(K, (half *) K_mirror, kv_limit, (int) row_lo, win_rows, stream);
            if (scratch.need_f16_v && !scratch.v_is_k_view) {
                sm70_d256_dequant_kv_window(V, (half *) scratch.f16_extra.V, kv_limit, (int) row_lo, win_rows, stream);
            }

            // The kernel addresses absolute KV rows; the mirror holds this
            // window's rows, so the base shifts back by row_lo and the head
            // stride is this window's row count.
            const half * const K_win = K_mirror - row_lo * SM70_D256_D;
            const int64_t k_row_stride   = SM70_D256_D;
            const int64_t k_head_stride  = (int64_t) win_rows * SM70_D256_D;
            const int64_t k_outer_stride = (int64_t) K->ne[2] * win_rows * SM70_D256_D;

            const half * V_win;
            int64_t v_row_stride, v_head_stride, v_outer_stride;
            if (scratch.v_is_k_view) {
                V_win = K_win;
                v_row_stride   = k_row_stride;
                v_head_stride  = k_head_stride;
                v_outer_stride = k_outer_stride;
            } else if (V->type == GGML_TYPE_F16) {
                // read in place, absolute rows
                V_win = (const half *) V->data;
                v_row_stride   = V->nb[1] / sizeof(half);
                v_head_stride  = V->nb[2] / sizeof(half);
                v_outer_stride = V->nb[3] / sizeof(half);
            } else {
                V_win = (const half *) scratch.f16_extra.V - row_lo * SM70_D256_D;
                v_row_stride   = SM70_D256_D;
                v_head_stride  = (int64_t) win_rows * SM70_D256_D;
                v_outer_stride = (int64_t) V->ne[2] * win_rows * SM70_D256_D;
            }

            // An unsplit first window writes the accumulator directly; every
            // split segment and every later window writes a partial slice that
            // is merged in below.
            const bool direct_acc = w == 0 && kv_splits == 1;
            float * const win_out = direct_acc ? Os : p_out;
            float * const win_max = direct_acc ? acc_max : p_max;
            float * const win_sum = direct_acc ? acc_sum : p_sum;

            sm70_d256_launch_dense<true>(
                Qs, K_win, V_win, Os, mask->data, mask_bounds,
                /*q_batch_stride */ (int64_t) heads_q * q_pad * SM70_D256_D,
                /*q_row_stride   */ SM70_D256_D,
                /*q_head_stride  */ (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                /*mask_row_stride  */ mask->nb[1] / sizeof(int2),
                /*mask_batch_stride*/ mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int2),
                /*mask_is_range*/ true,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale,
                (int) (row_lo / SM70_D256_MASK_BLOCK_N),
                (int) ((row_lo + win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N),
                kv_splits,
                win_out, win_max, win_sum,
                stream);

            if (!direct_acc) {
                const unsigned n_merge = (unsigned) ((scratch.rows
                    + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                    / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
                const int64_t slice_out = (int64_t) scratch.rows * SM70_D256_D;
                // Slice 0 carries first = true for the first window: it must
                // initialize the accumulator even for rows whose slice 0 is
                // empty, so that slice 1 and later windows can merge into it.
                FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                    Os, acc_max, acc_sum, win_out, win_max, win_sum,
                    (int64_t) scratch.rows, w == 0, scale * float(M_LOG2E));
                if (kv_splits > 1) {
                    FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                        Os, acc_max, acc_sum,
                        win_out + slice_out, win_max + scratch.rows, win_sum + scratch.rows,
                        (int64_t) scratch.rows, false, scale * float(M_LOG2E));
                }
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (assist_owner) {
            // Wait for the assist partial [n_rows, kv_len) and fold it into the accumulator as
            // one more window; the merge kernel skips empty (-inf) rows and takes the assist
            // values for rows this segment never saw.
            sm70_d256_assist_wait_launch(assist->epoch_partial_local, assist->last_partial_local, stream);
            const unsigned n_merge = (unsigned) ((scratch.rows
                + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum,
                assist->p_out_local, assist->p_max_local, assist->p_sum_local,
                (int64_t) scratch.rows, false, scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // A range mask bounds the rows the dense kernel reads, so a quantized K/V mirror
        // only needs [0, *kv_limit) rows. Everything else keeps the full mirror.
        const bool k_partial = mask_range && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0);
        const bool v_partial = mask_range && !scratch.v_is_k_view &&
                               (V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0);

        // Zero the Q pad rows; their outputs are dropped by the scatter.
        CUDA_CHECK(cudaMemsetAsync(Qs, 0, scratch.n_q * sizeof(half), stream));

        if (k_partial || v_partial) {
            // the bounds kernel raises it to the rows its bounds allow
            CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));
        }
        if (scratch.need_f16_k && !k_partial) {
            GGML_ASSERT(scratch.f16_extra.K != 0);
            sm70_d256_dequant_kv_tensor(K, (half *) scratch.f16_extra.K, stream);
        }
        // A view of a quantized K shares the K mirror; only a separate tensor is
        // converted here.
        if (scratch.need_f16_v && !scratch.v_is_k_view && !v_partial) {
            GGML_ASSERT(scratch.f16_extra.V != 0);
            sm70_d256_dequant_kv_tensor(V, (half *) scratch.f16_extra.V, stream);
        }

        // K/V are read either in place (f16, strides from the tensor) or from the
        // packed f16 mirror [batch][hkv][kv][D].
        const half * K_h2;
        const half * V_h2;
        int64_t k_outer_stride, k_row_stride, k_head_stride;
        int64_t v_outer_stride, v_row_stride, v_head_stride;
        if (K->type == GGML_TYPE_F16) {
            K_h2 = (const half *) K->data;
            k_row_stride   = K->nb[1] / sizeof(half);
            k_head_stride  = K->nb[2] / sizeof(half);
            k_outer_stride = K->nb[3] / sizeof(half);
        } else {
            K_h2 = (const half *) scratch.f16_extra.K;
            k_row_stride   = K->ne[0];
            k_head_stride  = K->ne[1] * K->ne[0];
            k_outer_stride = K->ne[2] * K->ne[1] * K->ne[0];
        }
        if (scratch.v_is_k_view) {
            V_h2 = K_h2;
            v_row_stride   = k_row_stride;
            v_head_stride  = k_head_stride;
            v_outer_stride = k_outer_stride;
        } else if (V->type == GGML_TYPE_F16) {
            V_h2 = (const half *) V->data;
            v_row_stride   = V->nb[1] / sizeof(half);
            v_head_stride  = V->nb[2] / sizeof(half);
            v_outer_stride = V->nb[3] / sizeof(half);
        } else {
            V_h2 = (const half *) scratch.f16_extra.V;
            v_row_stride   = V->ne[0];
            v_head_stride  = V->ne[1] * V->ne[0];
            v_outer_stride = V->ne[2] * V->ne[1] * V->ne[0];
        }

        {
            const dim3 grid(q_pad, batch * hkv, gqa);
            sm70_d256_stage_q_kernel<<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Q->data, (half2 *) Qs,
                q_len, hkv, gqa, q_pad,
                Q->nb[1] / sizeof(float2), Q->nb[2] / sizeof(float2), Q->nb[3] / sizeof(float2));
            CUDA_CHECK(cudaGetLastError());
        }

        if (mask_range) {
            // One CTA per (q block, batch) computes the int2 bounds the dense pre-scan produces.
            const dim3 grid(scratch.n_q_blocks, batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int2 *) mask->data, mask_bounds, kv_limit,
                q_len, kv_len,
                mask->nb[1] / sizeof(int2),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int2));
            CUDA_CHECK(cudaGetLastError());

            if (k_partial) {
                // the bounds are known now: mirror only the rows they allow
                sm70_d256_dequant_kv_partial(K, (half *) scratch.f16_extra.K, kv_limit, stream);
            }
            if (v_partial) {
                sm70_d256_dequant_kv_partial(V, (half *) scratch.f16_extra.V, kv_limit, stream);
            }
        } else {
            const int n_splits = (kv_len + SM70_D256_MASK_SPLIT - 1) / SM70_D256_MASK_SPLIT;
            const dim3 grid(scratch.n_q_blocks, batch, n_splits);
            // The split CTAs accumulate into bounds with atomicMax; zero first on
            // the same stream.
            CUDA_CHECK(cudaMemsetAsync(mask_bounds, 0,
                (size_t) batch * scratch.n_q_blocks * sizeof(int2), stream));
            // 16-byte vector loads need every row and every batch base aligned:
            // nb[1] and nb[3] in bytes, mask->data, and the 8-half column steps.
            const bool vec_ok = mask->nb[1] % 16 == 0
                && (mask->ne[3] == 1 || mask->nb[3] % 16 == 0)
                && (uintptr_t) mask->data % 16 == 0;
            sm70_d256_mask_bounds_kernel<<<grid, 256, 0, stream>>>(
                (const __half *) mask->data, mask_bounds,
                q_len, kv_len,
                mask->nb[1] / sizeof(half),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(half),
                vec_ok);
            CUDA_CHECK(cudaGetLastError());
        }

        if (scratch.partial) {
            // Dense SplitKV2: the whole KV range is one window, split in two.
            // Uses the same partial buffers and merge as the windowed path;
            // slice 0 initializes the accumulator, slice 1 merges into it.
            float * const p_out   = (float *) (base + scratch.p_out_offset);
            float * const p_max   = (float *) (base + scratch.p_max_offset);
            float * const p_sum   = (float *) (base + scratch.p_sum_offset);
            float * const acc_max = (float *) (base + scratch.acc_max_offset);
            float * const acc_sum = (float *) (base + scratch.acc_sum_offset);

            sm70_d256_launch_dense<true>(
                Qs, K_h2, V_h2, Os, mask->data, mask_bounds,
                /*q_batch_stride */ (int64_t) heads_q * q_pad * SM70_D256_D,
                /*q_row_stride   */ SM70_D256_D,
                /*q_head_stride  */ (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                /*mask_row_stride  */ mask_range ? mask->nb[1] / sizeof(int2)   : mask->nb[1] / sizeof(half),
                /*mask_batch_stride*/ mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / (mask_range ? sizeof(int2) : sizeof(half)),
                mask_range,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale,
                /*win_block_lo*/ 0, /*win_block_hi*/ INT_MAX,
                SM70_D256_KV_SPLITS,
                p_out, p_max, p_sum,
                stream);

            const unsigned n_merge = (unsigned) ((scratch.rows
                + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
            const int64_t slice_out = (int64_t) scratch.rows * SM70_D256_D;
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum, p_out, p_max, p_sum,
                (int64_t) scratch.rows, true, scale * float(M_LOG2E));
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum,
                p_out + slice_out, p_max + scratch.rows, p_sum + scratch.rows,
                (int64_t) scratch.rows, false, scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_sm70_d256_launch_raw(
                Qs, K_h2, V_h2, Os, mask->data, mask_bounds,
                /*q_batch_stride */ (int64_t) heads_q * q_pad * SM70_D256_D,
                /*q_row_stride   */ SM70_D256_D,
                /*q_head_stride  */ (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                /*mask_row_stride  */ mask_range ? mask->nb[1] / sizeof(int2)   : mask->nb[1] / sizeof(half),
                /*mask_batch_stride*/ mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / (mask_range ? sizeof(int2) : sizeof(half)),
                mask_range,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale, stream);
        }
    }

    // dst is the FA output (D, heads_q, q_len, batch) f32; the strides below
    // are its token/head/batch strides in float2 units.
    {
        const dim3 grid(q_len, batch * heads_q);
        if (scratch.partial) {
            // The partial paths (windowed or split dense) normalize the
            // accumulator here, in place of the separate finalize pass.
            sm70_d256_scatter_kernel<true><<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Os, (float2 *) dst->data,
                (const float *) (base + scratch.acc_sum_offset),
                heads_q, q_pad,
                dst->nb[2] / sizeof(float2),
                dst->nb[1] / sizeof(float2),
                dst->nb[3] / sizeof(float2));
        } else {
            sm70_d256_scatter_kernel<false><<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Os, (float2 *) dst->data, nullptr,
                heads_q, q_pad,
                dst->nb[2] / sizeof(float2),
                dst->nb[1] / sizeof(float2),
                dst->nb[3] / sizeof(float2));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // The scatter has consumed the merged partial: the assist rank may now overwrite the partial
    // slots for the next execution.
    if (assist_owner) {
        sm70_d256_assist_signal_launch(assist->epoch_free_local, stream);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
