#include "common.cuh"

// parameters for the fused GET_ROWS -> CONCAT -> CPY x K -> SSM_CONV -> SILU sequence of a
// linear attention layer (see ggml_cuda_try_ssm_conv_rollback_fusion); strides are in elements
struct ggml_cuda_ssm_conv_rollback_params {
    const float *   state;        // state cache rows, [3*C, n_rows]
    int64_t         state_stride; // stride between rows
    const int32_t * row;          // row index of the sequence being processed
    const float *   qkv;          // transposed qkv view, [n_tok, C, 1]
    int64_t         qkv_nb0;      // stride between tokens
    int64_t         qkv_nb1;      // stride between channels
    const float *   w;            // conv weights, [4, C]
    int64_t         w_nb1;        // stride between channels
    float *         dst;          // silu output, [C, n_tok, 1]
    int64_t         dst_nb1;      // stride between tokens
    int64_t         n_tok;
    int64_t         n_channels;
    int64_t         snap_off[16]; // first column of the snapshot in the convolution window
    float *         snap_dst[16]; // snapshot row in the state cache
    int             n_snap;
};

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_ssm_conv_rollback_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & params);
