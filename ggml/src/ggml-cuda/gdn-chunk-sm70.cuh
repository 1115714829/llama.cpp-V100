#pragma once

#include "common.cuh"

// SM70 (Volta) chunked prefill for GATED_DELTA_NET (staged f16, TileLang/FlashQLA
// kernels). Device kernels: gdn-chunk-sm70/gdn-chunk-*.cuh. The kernels are
// compiled by a dedicated sm_70-only object target, so the dispatch site guards
// its calls with GGML_CUDA_GDN_CHUNK_SM70_COMPILED.
#if defined(__CUDA_ARCH_LIST__)
#define GGML_CUDA_GDN_CHUNK_SM70_COMPILED ggml_cuda_has_arch(GGML_CUDA_CC_VOLTA)
#else
#define GGML_CUDA_GDN_CHUNK_SM70_COMPILED false
#endif

struct ggml_cuda_gdn_chunk_args {
    const float * q; const float * k;          // raw (prologue) or already l2-normalized
    int64_t sq1, sq2, sk1, sk2;                // strides in floats (head, token)
    bool    l2norm;                            // apply x * rsqrtf(ss/128 + eps) * scale in the prep kernel
    float   eps_q, scale_q, eps_k, scale_k;
    const float * v; int64_t sv1, sv2;         // strides in floats (head, token)
    const float * g; const float * beta;       // contiguous [T][H]
    const float * s0;                          // initial state [H][128][128]
    int64_t n_tokens;                          // tokens processed by the chunked path (multiple of 64)
    int64_t H;                                 // number of v heads
    int64_t H_k;                               // number of q/k heads
    float * dst;                               // attention output, f32 [n_tokens][H][128]
    float * h_out;                             // state after n_tokens tokens, [H][128][128]
};

// leading tokens for the chunked path (a multiple of 64, leaving >= K and >= 1 tokens for the recurrent kernel), 0 if not applicable
int64_t ggml_cuda_gdn_chunk_sm70_n_tokens(int cc, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3, bool kda, int K);
void    ggml_cuda_gdn_chunk_sm70(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & args);
