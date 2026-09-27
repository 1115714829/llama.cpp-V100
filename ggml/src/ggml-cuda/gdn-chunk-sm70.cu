// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// SM70 (Volta) chunked prefill for GATED_DELTA_NET, host side: dispatch
// predicate, scratch layout, f32 -> f16 staging of q/k/v, chunk metadata and
// kernel launches. Device kernel provenance: gdn-chunk-sm70/gdn-chunk-*.cuh
// (TileLang 0.1.13 generated, FlashQLA sm70 path of 1Cat-vLLM).

// The HIP and MUSA backends glob ../ggml-cuda/*.cu into their own libraries and
// cannot compile CuTe/CUTLASS. This file is only used by the CUDA backend.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "gdn-chunk-sm70.cuh"

#include "gdn-chunk-sm70/gdn-chunk-cumsum.cuh"
#include "gdn-chunk-sm70/gdn-chunk-fwd.cuh"
#include "gdn-chunk-sm70/gdn-chunk-kkt.cuh"

using namespace gdn_chunk_sm70;

constexpr int GDN_CHUNK_SM70_H_K   = 4;
constexpr int GDN_CHUNK_SM70_H_V   = 12;
constexpr int GDN_CHUNK_SM70_D     = 128;
constexpr int GDN_CHUNK_SM70_CHUNK = 64;
constexpr int GDN_CHUNK_SM70_WARP  = 32;

// dynamic shared memory sizes baked into the TileLang kernels
constexpr int GDN_CHUNK_SM70_SMEM_CUMSUM = 6400;
constexpr int GDN_CHUNK_SM70_SMEM_KKT    = 16384;
constexpr int GDN_CHUNK_SM70_SMEM_FWD    = 91072;

// Single-sequence chunk metadata: cu_seqlens {0, n}, chunk_offsets {0, num_chunks},
// chunk_indices {0, i}.
__global__ void gdn_chunk_meta_kernel(
        int * __restrict__ cu_seqlens, int * __restrict__ chunk_offsets, int * __restrict__ chunk_indices,
        const int n_tokens, const int num_chunks) {
    if (threadIdx.x == 0) {
        cu_seqlens[0]    = 0;
        cu_seqlens[1]    = n_tokens;
        chunk_offsets[0] = 0;
        chunk_offsets[1] = num_chunks;
    }
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        chunk_indices[2*i + 0] = 0;
        chunk_indices[2*i + 1] = i;
    }
}

// One warp per 128-element row: all q heads, then all k heads, then all v heads
// of the chunked tokens. The q/k rows are l2-normalized in flight when l2norm
// is set, with the same formula as the recurrent kernel's prologue.
__global__ void gdn_chunk_prep_kernel(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        half_t * __restrict__ q16, half_t * __restrict__ k16, half_t * __restrict__ v16,
        const int64_t sq1, const int64_t sq2,
        const int64_t sk1, const int64_t sk2,
        const int64_t sv1, const int64_t sv2,
        const int n, const bool l2norm,
        const float eps_q, const float scale_q, const float eps_k, const float scale_k) {
    const int64_t n_q = (int64_t) n * GDN_CHUNK_SM70_H_K;
    const int64_t n_k = n_q + (int64_t) n * GDN_CHUNK_SM70_H_K;

    const int64_t warp_id = (int64_t) blockIdx.x * (blockDim.x / GDN_CHUNK_SM70_WARP) + threadIdx.x / GDN_CHUNK_SM70_WARP;
    if (warp_id >= n_k + (int64_t) n * GDN_CHUNK_SM70_H_V) {
        return;
    }
    const int lane = threadIdx.x % GDN_CHUNK_SM70_WARP;

    const float * src;
    half_t      * dst;
    float         eps   = 0.0f;
    float         scale = 0.0f;
    if (warp_id < n_q) {
        const int64_t t = warp_id / GDN_CHUNK_SM70_H_K;
        const int64_t h = warp_id % GDN_CHUNK_SM70_H_K;
        src   = q + t * sq2 + h * sq1;
        dst   = q16 + warp_id * GDN_CHUNK_SM70_D;
        eps   = eps_q;
        scale = scale_q;
    } else if (warp_id < n_k) {
        const int64_t row = warp_id - n_q;
        const int64_t t   = row / GDN_CHUNK_SM70_H_K;
        const int64_t h   = row % GDN_CHUNK_SM70_H_K;
        src   = k + t * sk2 + h * sk1;
        dst   = k16 + row * GDN_CHUNK_SM70_D;
        eps   = eps_k;
        scale = scale_k;
    } else {
        const int64_t row = warp_id - n_k;
        const int64_t t   = row / GDN_CHUNK_SM70_H_V;
        const int64_t h   = row % GDN_CHUNK_SM70_H_V;
        src = v + t * sv2 + h * sv1;
        dst = v16 + row * GDN_CHUNK_SM70_D;
    }

    float x[GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP];
#pragma unroll
    for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
        x[r] = src[r * GDN_CHUNK_SM70_WARP + lane];
    }
    if (l2norm && warp_id < n_k) {
        float ss = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
            ss += x[r] * x[r];
        }
        ss = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(ss);
        const float nrm = rsqrtf(ss / GDN_CHUNK_SM70_D + eps) * scale;
#pragma unroll
        for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
            x[r] *= nrm;
        }
    }
#pragma unroll
    for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
        dst[r * GDN_CHUNK_SM70_WARP + lane] = half_t(x[r]);
    }
}

// o16 [n][12][128] -> dst f32, both contiguous
__global__ void gdn_chunk_store_kernel(const half2 * __restrict__ src, float2 * __restrict__ dst, const int64_t n2) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        dst[i] = __half22float2(src[i]);
    }
}

int64_t ggml_cuda_gdn_chunk_sm70_n_tokens(int cc, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3, bool kda, int K) {
    if (cc != GGML_CUDA_CC_VOLTA || kda || S_v != GDN_CHUNK_SM70_D || H != GDN_CHUNK_SM70_H_V || H_k != GDN_CHUNK_SM70_H_K ||
            n_seqs != 1 || rq3 != 1) {
        return 0;
    }
    GGML_ASSERT(K >= 1 && K <= GDN_CHUNK_SM70_CHUNK);

    // the chunked kernels do not write snapshots, keep >= K tokens for the recurrent kernel
    int64_t tail = n_tokens % GDN_CHUNK_SM70_CHUNK;
    if (tail < K) {
        tail += GDN_CHUNK_SM70_CHUNK;
    }
    const int64_t n_c = n_tokens - tail;
    return n_c >= GDN_CHUNK_SM70_CHUNK ? n_c : 0;
}

void ggml_cuda_gdn_chunk_sm70(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & args) {
    const int64_t n = args.n_tokens;
    GGML_ASSERT(n > 0 && n % GDN_CHUNK_SM70_CHUNK == 0 && n <= INT32_MAX);

    const int num_chunks = (int) (n / GDN_CHUNK_SM70_CHUNK);
    cudaStream_t stream  = ctx.stream();

    ggml_cuda_pool & pool = ctx.pool();
    const size_t n_qk = (size_t) n * GDN_CHUNK_SM70_H_K * GDN_CHUNK_SM70_D;
    const size_t n_v  = (size_t) n * GDN_CHUNK_SM70_H_V * GDN_CHUNK_SM70_D;

    ggml_cuda_pool_alloc<half_t> q16(pool, n_qk);
    ggml_cuda_pool_alloc<half_t> k16(pool, n_qk);
    ggml_cuda_pool_alloc<half_t> v16(pool, n_v);
    ggml_cuda_pool_alloc<float>  gcum(pool, (size_t) n * GDN_CHUNK_SM70_H_V);
    ggml_cuda_pool_alloc<half_t> a(pool, (size_t) n * GDN_CHUNK_SM70_H_V * GDN_CHUNK_SM70_CHUNK);
    ggml_cuda_pool_alloc<half_t> o16(pool, n_v);
    ggml_cuda_pool_alloc<int>    meta(pool, 4 + 2 * (size_t) num_chunks);

    int * cu_seqlens    = meta.get();
    int * chunk_offsets = cu_seqlens + 2;
    int * chunk_indices = chunk_offsets + 2;

    gdn_chunk_meta_kernel<<<1, 128, 0, stream>>>(cu_seqlens, chunk_offsets, chunk_indices, (int) n, num_chunks);
    CUDA_CHECK(cudaGetLastError());

    const int64_t n_rows  = n * (GDN_CHUNK_SM70_H_K + GDN_CHUNK_SM70_H_K + GDN_CHUNK_SM70_H_V);
    const unsigned int n_warps = (unsigned int) ((n_rows + 3) / 4);
    gdn_chunk_prep_kernel<<<n_warps, 128, 0, stream>>>(
        args.q, args.k, args.v, q16.get(), k16.get(), v16.get(),
        args.sq1, args.sq2, args.sk1, args.sk2, args.sv1, args.sv2,
        (int) n, args.l2norm, args.eps_q, args.scale_q, args.eps_k, args.scale_k);
    CUDA_CHECK(cudaGetLastError());

    tilelang_chunk_local_cumsum_kernel_kernel<<<num_chunks, 128, GDN_CHUNK_SM70_SMEM_CUMSUM, stream>>>(
        chunk_indices, cu_seqlens, gcum.get(), args.g, num_chunks, (int) n, 1);
    CUDA_CHECK(cudaGetLastError());

    tilelang_kkt_solve_kernel_kernel<<<num_chunks * GDN_CHUNK_SM70_H_V, 128, GDN_CHUNK_SM70_SMEM_KKT, stream>>>(
        a.get(), args.beta, chunk_indices, cu_seqlens, k16.get(), num_chunks, (int) n, 1);
    CUDA_CHECK(cudaGetLastError());

    CUDA_SET_SHARED_MEMORY_LIMIT(tilelang_fused_chunk_gdr_fwd_kernel_kernel, GDN_CHUNK_SM70_SMEM_FWD);
    tilelang_fused_chunk_gdr_fwd_kernel_kernel<<<GDN_CHUNK_SM70_H_V * 4, 128, GDN_CHUNK_SM70_SMEM_FWD, stream>>>(
        a.get(), args.beta, chunk_offsets, cu_seqlens, gcum.get(), args.s0, args.h_out,
        k16.get(), o16.get(), q16.get(), v16.get(), 1, (int) n, 1);
    CUDA_CHECK(cudaGetLastError());

    const int64_t n2 = n * GDN_CHUNK_SM70_H_V * (GDN_CHUNK_SM70_D / 2);
    const unsigned int n_blocks = (unsigned int) ((n2 + 255) / 256);
    gdn_chunk_store_kernel<<<n_blocks, 256, 0, stream>>>(
        reinterpret_cast<const half2 *>(o16.get()), reinterpret_cast<float2 *>(args.dst), n2);
    CUDA_CHECK(cudaGetLastError());
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
