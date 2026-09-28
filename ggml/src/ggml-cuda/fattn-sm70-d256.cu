// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// SM70 (Volta) D256 Split-D prefill attention, host side: dispatch predicate,
// scratch layout, K/V dequant, Q staging, mask pre-scan, kernel launch and O
// scatter. Device kernel provenance: fattn-sm70-d256-kernel.cuh.

// The HIP and MUSA backends glob ../ggml-cuda/*.cu into their own libraries and
// cannot compile CuTe/CUTLASS. This file is only used by the CUDA backend.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-sm70-d256.cuh"
#include "fattn-sm70-d256-kernel.cuh"

// NOTE: no anonymous namespace in this file. The CuTe vendor headers
// (cute/atom/mma_traits_sm70.hpp) open their own anonymous namespaces; a second
// one in this TU makes cudafe's _GLOBAL__N__<hash> symbol mangling ambiguous.

// Natural-log scale -> exp2 domain, matching the kernel's softmax_scale_log2.
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634
#endif

constexpr int SM70_D256_BLOCK_M = 64;
constexpr int SM70_D256_D       = 256;

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
// written, so the Q pad rows never reach dst.
__global__ void sm70_d256_scatter_kernel(
        const float2 * __restrict__ src, float2 * __restrict__ dst,
        const int heads_q, const int q_pad,
        const int64_t dst_row, const int64_t dst_head, const int64_t dst_batch) {
    const int r  = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / heads_q;
    const int head_q = bh % heads_q;
    const float2 v = src[threadIdx.x
        + (int64_t) b * q_pad * heads_q * (SM70_D256_D/2)
        + (int64_t) r * heads_q * (SM70_D256_D/2)
        + (int64_t) head_q * (SM70_D256_D/2)];
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
// alloc_size and the launcher both go through sm70_d256_get_scratch() so the
// offsets and the need_f16 predicates can not diverge.
struct sm70_d256_scratch {
    size_t total;            // full buffer size: nnbytes(dst) + extra
    size_t qs_offset;        // relative to dst->data + ggml_nbytes(dst)
    size_t os_offset;
    size_t bounds_offset;
    size_t kv_limit_offset;
    size_t n_q;              // Qs/Os elements per buffer
    int q_pad;
    int n_q_blocks;          // q_pad / SM70_D256_BLOCK_M
    bool need_f16_k;
    bool need_f16_v;
    bool v_is_k_view;
    ggml_cuda_flash_attn_ext_f16_extra_data f16_extra;
};

static sm70_d256_scratch sm70_d256_get_scratch(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool v_is_k_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const bool need_f16_k = K->type != GGML_TYPE_F16;
    // V is a view of an f16 K: the K data itself serves as V.
    const bool need_f16_v = !(v_is_k_view && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16;

    sm70_d256_scratch s = {};
    s.need_f16_k = need_f16_k;
    s.need_f16_v = need_f16_v;
    s.v_is_k_view = v_is_k_view;
    s.f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_k, need_f16_v);

    const char * base = (const char *) dst->data + ggml_nbytes(dst);
    const size_t f16_bytes = GGML_PAD((size_t) (s.f16_extra.end - (uintptr_t) base), 128);
    s.q_pad = (int) GGML_PAD(Q->ne[1], SM70_D256_BLOCK_M);
    s.n_q = (size_t) Q->ne[2] * s.q_pad * SM70_D256_D * Q->ne[3];
    s.qs_offset = f16_bytes;
    s.os_offset = s.qs_offset + s.n_q * sizeof(half);
    s.n_q_blocks = s.q_pad / SM70_D256_BLOCK_M;
    s.bounds_offset = GGML_PAD(s.os_offset + s.n_q * sizeof(float), 128);
    s.kv_limit_offset = GGML_PAD(s.bounds_offset + (size_t) Q->ne[3] * s.n_q_blocks * sizeof(int2), 128);
    s.total = ggml_nbytes(dst) + s.kv_limit_offset + sizeof(int);
    return s;
}

// K/V -> f16 mirror, the same to_fp16_nc conversion the stock launch_fattn
// uses for strided sources. The mirror is packed canonically as
// [batch][hkv][kv][D] (row stride D, head stride kv*D): to_fp16_nc writes
// output index ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0. The linear to_fp16() variant
// is deliberately not used: it preserves the source memory order, which for a
// packed permuted view is not the canonical one.
static void sm70_d256_dequant_kv(
        const ggml_tensor * K, const ggml_tensor * V,
        const sm70_d256_scratch & s, cudaStream_t stream) {
    if (s.need_f16_k) {
        GGML_ASSERT(s.f16_extra.K != 0);
        const size_t ts = ggml_type_size(K->type);
        const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
        to_fp16(K->data, (half *) s.f16_extra.K, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                K->nb[1] / ts, K->nb[2] / ts, K->nb[3] / ts, stream);
    }
    // A view of a quantized K shares the K mirror; only a separate tensor is
    // converted here.
    if (!s.need_f16_v || s.v_is_k_view) {
        return;
    }
    GGML_ASSERT(s.f16_extra.V != 0);
    const size_t ts = ggml_type_size(V->type);
    const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
    to_fp16(V->data, (half *) s.f16_extra.V, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
            V->nb[1] / ts, V->nb[2] / ts, V->nb[3] / ts, stream);
}

// q8_0 -> f16 mirror of the rows [0, *kv_limit) only. With a range mask the dense kernel never reads
// a KV row at or past kv_limit, so a K/V view much wider than the attended range (a prompt ubatch that
// attends the whole cache) costs no conversion. Output layout as sm70_d256_dequant_kv:
// [ne3][ne2][ne1][ne0] contiguous, i.e. ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0.
// One warp per row, grid.y = ne2, grid.z = ne3, rows i1 >= *kv_limit are skipped.
static __global__ void sm70_d256_dequant_q8_0_rows(
        const char * __restrict__ src, half * __restrict__ dst,
        const int ne0, const int ne1, const int ne2,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int * __restrict__ kv_limit) {
    const int i1 = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i1 >= *kv_limit) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    // lane's q8_0 block of the row and its 8-value group inside the block
    const int ib = lane / 4;
    const int iq = (lane % 4) * 8;

    const block_q8_0 * src_row = (const block_q8_0 *) (src + (int64_t) i3*nb3 + (int64_t) i2*nb2 + (int64_t) i1*nb1);
    half * dst_row = dst + (((int64_t) i3*ne2 + i2)*ne1 + i1)*ne0;

    const float d = __half2float(src_row[ib].d);
    half2 h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = __floats2half2_rn(d * src_row[ib].qs[iq + 2*i], d * src_row[ib].qs[iq + 2*i + 1]);
    }
    uint4 out;
    memcpy(&out.x, &h[0], sizeof(uint32_t));
    memcpy(&out.y, &h[1], sizeof(uint32_t));
    memcpy(&out.z, &h[2], sizeof(uint32_t));
    memcpy(&out.w, &h[3], sizeof(uint32_t));
    *reinterpret_cast<uint4 *>(dst_row + ib*32 + iq) = out;
}

// F16 is read in place: contiguous rows plus 16 B aligned strides for the
// 128-bit global loads. The q8_0 mirror needs block-aligned source strides.
static bool sm70_d256_kv_type_ok(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        return t->nb[0] == sizeof(half) && t->nb[1] % 16 == 0 &&
               t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    }
    if (t->type == GGML_TYPE_Q8_0) {
        const size_t ts = ggml_type_size(GGML_TYPE_Q8_0);
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
void ggml_cuda_sm70_d256_launch_raw(
        const void * q, const void * k, const void * v, void * out,
        const void * mask, const int2 * mask_bounds,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_outer_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_outer_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t mask_row_stride, int64_t mask_batch_stride, bool mask_is_range,
        int q_pad, int kv_len, int heads_q, int heads_kv, int batch, int kv_offset,
        float softmax_scale, float mask_scale, cudaStream_t stream) {
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    using El = cutlass::half_t;

    GGML_ASSERT(q_pad % Traits::kBlockM == 0);
    GGML_ASSERT(kv_len - kv_offset >= 1 && kv_len - kv_offset <= q_pad);

    auto kernel = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Traits, El, float, false>;
    CUDA_SET_SHARED_MEMORY_LIMIT((const void *) kernel, Traits::kSmemBytes);

    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / Traits::kBlockM, batch, heads_q);
    kernel<<<grid, block, Traits::kSmemBytes, stream>>>(
        (const El *) q, (const El *) k, (const El *) v, (float *) out,
        (const __half *) mask,
        (int) q_batch_stride, (int) q_row_stride, (int) q_head_stride,
        (int) k_outer_stride, (int) k_row_stride, (int) k_head_stride,
        (int) v_outer_stride, (int) v_row_stride, (int) v_head_stride,
        mask_row_stride, mask_batch_stride, mask_is_range, (const int2 *) mask_bounds,
        q_pad, kv_len, heads_q, heads_kv, kv_offset,
        softmax_scale * float(M_LOG2E), mask_scale,
        /*partial_out*/ nullptr, /*partial_max*/ nullptr, /*partial_sum*/ nullptr);
    CUDA_CHECK(cudaGetLastError());
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

    // A range mask bounds the rows the dense kernel reads, so a q8_0 K/V mirror only needs
    // [0, *kv_limit) rows. Everything else keeps the full mirror.
    const bool range_q8 = mask_range && K->type == GGML_TYPE_Q8_0;

    // Zero the Q pad rows; their outputs are dropped by the scatter.
    CUDA_CHECK(cudaMemsetAsync(Qs, 0, scratch.n_q * sizeof(half), stream));

    if (range_q8) {
        // the bounds kernel raises it to the rows its bounds allow
        CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));
    } else {
        sm70_d256_dequant_kv(K, V, scratch, stream);
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

        if (range_q8) {
            // the bounds are known now: mirror only the rows they allow
            GGML_ASSERT(K->ne[0] == SM70_D256_D);
            const dim3 grid_k((unsigned) ((K->ne[1] + 7) / 8), (unsigned) K->ne[2], (unsigned) K->ne[3]);
            sm70_d256_dequant_q8_0_rows<<<grid_k, 256, 0, stream>>>(
                (const char *) K->data, (half *) scratch.f16_extra.K,
                (int) K->ne[0], (int) K->ne[1], (int) K->ne[2],
                K->nb[1], K->nb[2], K->nb[3], kv_limit);
            CUDA_CHECK(cudaGetLastError());
            if (!scratch.v_is_k_view && V->type != GGML_TYPE_F16) {
                GGML_ASSERT(V->ne[0] == SM70_D256_D);
                const dim3 grid_v((unsigned) ((V->ne[1] + 7) / 8), (unsigned) V->ne[2], (unsigned) V->ne[3]);
                sm70_d256_dequant_q8_0_rows<<<grid_v, 256, 0, stream>>>(
                    (const char *) V->data, (half *) scratch.f16_extra.V,
                    (int) V->ne[0], (int) V->ne[1], (int) V->ne[2],
                    V->nb[1], V->nb[2], V->nb[3], kv_limit);
                CUDA_CHECK(cudaGetLastError());
            }
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

    // dst is the FA output (D, heads_q, q_len, batch) f32; the strides below
    // are its token/head/batch strides in float2 units.
    {
        const dim3 grid(q_len, batch * heads_q);
        sm70_d256_scatter_kernel<<<grid, SM70_D256_D/2, 0, stream>>>(
            (const float2 *) Os, (float2 *) dst->data,
            heads_q, q_pad,
            dst->nb[2] / sizeof(float2),
            dst->nb[1] / sizeof(float2),
            dst->nb[3] / sizeof(float2));
        CUDA_CHECK(cudaGetLastError());
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
