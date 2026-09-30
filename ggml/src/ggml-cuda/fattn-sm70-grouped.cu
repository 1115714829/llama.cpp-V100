// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// SM70 (Volta) grouped verify attention, host side: dispatch predicate, K/V
// type combination selection and kernel launch. K/V cache types: F16, Q8_0 or
// Q4_0. Device kernel: fattn-sm70-grouped.cuh.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-sm70-grouped.cuh"

bool ggml_cuda_flash_attn_ext_sm70_grouped_supported(const ggml_tensor * dst, int cc) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!volta_mma_available(cc)) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != 256) {
        return false;
    }
    for (const ggml_tensor * t : {K, V}) {
        if ((t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_Q4_0) || t->ne[0] != 256) {
            return false;
        }
        if (reinterpret_cast<uintptr_t>(t->data) % 16 != 0) {
            return false;
        }
        for (int i = 1; i < 4; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    // Only matching K/V types are instantiated below, so mixed caches take the generic path.
    if (K->type != V->type) {
        return false;
    }
    // Under tensor split a device can get 0 KV heads (fewer KV heads than devices); fall back to the generic path.
    if (K->ne[2] == 0 || Q->ne[2] == 0) {
        return false;
    }
    if (K->ne[2] != V->ne[2]) {
        return false;
    }
    // Only GQA 2, 4 and 6 are instantiated below.
    const int gqa = Q->ne[2] / K->ne[2]; // K->ne[2] > 0, checked above
    if (Q->ne[2] % K->ne[2] != 0 || (gqa != 2 && gqa != 4 && gqa != 6)) {
        return false;
    }
    // single-token decode stays on the vec kernel, which is faster there
    if (Q->ne[1] < 2 || Q->ne[1] > 16) {
        return false;
    }
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (sinks != nullptr) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16 && !(mask->type == GGML_TYPE_I32 && mask->ne[0] == 2 && mask->nb[0] == sizeof(int32_t))) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) { // n_kv_max hint
        return false;
    }

    return true;
}

template <int MAX_QUERY_TOKENS, int HEADS, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_sm70_grouped_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int n_q        = (int) Q->ne[1];
    const int n_heads    = (int) Q->ne[2];
    const int n_seq      = (int) Q->ne[3];
    const int n_kv       = (int) K->ne[1];
    const int n_kv_heads = (int) K->ne[2];

    constexpr int head_groups = GroupedVerifyTraits<MAX_QUERY_TOKENS, HEADS>::kHeadGroups;
    const int grid_x = n_kv_heads * head_groups;
    if (n_kv_heads == 0 || grid_x == 0) {
        return; // defensive: the dispatch predicate already rejects 0 heads
    }
    const int splits = std::max(1, std::min(80 / grid_x, (n_kv + 63) / 64));

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    const int64_t n_rows = (int64_t) n_q * n_heads * n_seq;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>  dst_partial(pool, n_rows * splits * kGroupedVerifyHeadDim);
    ggml_cuda_pool_alloc<float2> dst_meta(pool, n_rows * splits);

    const dim3 blocks_num(grid_x, splits, n_seq);
    const dim3 block_dim(kGroupedVerifyThreads, 1, 1);
    const size_t nbytes_shared = sizeof(GroupedVerifySmem);

    CUDA_SET_SHARED_MEMORY_LIMIT((flash_attn_ext_sm70_grouped<MAX_QUERY_TOKENS, HEADS, type_K, type_V>), nbytes_shared);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, nbytes_shared, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_sm70_grouped<MAX_QUERY_TOKENS, HEADS, type_K, type_V>, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        dst_partial.ptr, dst_meta.ptr,
        scale,
        n_q, n_kv, n_heads, splits,
        mask ? (int32_t) mask->ne[3] : (int32_t) 0,
        (int64_t) Q->nb[1], (int64_t) Q->nb[2], (int64_t) Q->nb[3],
        (int64_t) K->nb[1], (int64_t) K->nb[2], (int64_t) K->nb[3],
        (int64_t) V->nb[1], (int64_t) V->nb[2], (int64_t) V->nb[3],
        mask ? (int64_t) mask->nb[1] : (int64_t) 0,
        mask ? (int64_t) mask->nb[3] : (int64_t) 0,
        mask && mask->type == GGML_TYPE_I32 ? 1 : 0);

    const dim3 blocks_num_combine(n_q, n_heads, n_seq);
    const dim3 block_dim_combine(kGroupedVerifyHeadDim, 1, 1);
    const size_t nbytes_shared_combine = splits*sizeof(float2);

    const ggml_cuda_kernel_launch_params launch_params_combine(blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream);
    ggml_cuda_kernel_launch(flash_attn_combine_results<kGroupedVerifyHeadDim>, launch_params_combine,
        dst_partial.ptr, dst_meta.ptr, (float *) dst->data, splits);
}

// The predicate rejects mixed K/V types, so only the matching combinations exist.
template <int MAX_QUERY_TOKENS, int HEADS>
static void ggml_cuda_flash_attn_ext_sm70_grouped_select_kv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    if (K->type == GGML_TYPE_F16) {
        ggml_cuda_flash_attn_ext_sm70_grouped_launch<MAX_QUERY_TOKENS, HEADS, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
    } else if (K->type == GGML_TYPE_Q8_0) {
        ggml_cuda_flash_attn_ext_sm70_grouped_launch<MAX_QUERY_TOKENS, HEADS, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
    } else if (K->type == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_sm70_grouped_launch<MAX_QUERY_TOKENS, HEADS, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst);
    } else {
        GGML_ABORT("unsupported K/V type combination");
    }
}

// The predicate only lets GQA 2, 4 and 6 through.
template <int MAX_QUERY_TOKENS>
static void ggml_cuda_flash_attn_ext_sm70_grouped_select_heads(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    switch (gqa) {
        case 2:
            ggml_cuda_flash_attn_ext_sm70_grouped_select_kv<MAX_QUERY_TOKENS, 2>(ctx, dst);
            break;
        case 4:
            ggml_cuda_flash_attn_ext_sm70_grouped_select_kv<MAX_QUERY_TOKENS, 4>(ctx, dst);
            break;
        case 6:
            ggml_cuda_flash_attn_ext_sm70_grouped_select_kv<MAX_QUERY_TOKENS, 6>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported GQA ratio");
    }
}

void ggml_cuda_flash_attn_ext_sm70_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    GGML_ASSERT(ggml_cuda_flash_attn_ext_sm70_grouped_supported(dst, ggml_cuda_info().devices[ggml_cuda_get_device()].cc));

    if (Q->ne[1] <= 8) {
        ggml_cuda_flash_attn_ext_sm70_grouped_select_heads<8>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_sm70_grouped_select_heads<16>(ctx, dst);
    }
}
