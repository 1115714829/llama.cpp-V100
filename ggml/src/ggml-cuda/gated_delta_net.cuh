#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// raw inputs of the q/k l2 normalization fused into the gated_delta_net kernel
// (see ggml_cuda_match_gdn_prologue); all strides are in elements
struct ggml_cuda_gdn_prologue {
    const float * q_raw; // rms_norm q src
    const float * k_raw; // rms_norm k src
    int64_t sq1, sq2, sq3;
    int64_t sk1, sk2, sk3;
    float   eps_q, scale_q;
    float   eps_k, scale_k;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// same op, but l2-normalizes the raw q/k in-kernel (see ggml_cuda_match_gdn_prologue)
void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gdn_prologue & prologue,
                                        const ggml_cuda_gated_delta_net_fused_cache * cache);
