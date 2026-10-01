// F32 pair GEMV: two narrow F32 weights that share one src1 (qwen35 linear
// attention ssm_alpha/ssm_beta) computed in one launch. Each CTA computes one output
// column of both weights and streams the input once per step through registers.
//
// The accumulation and the reduction copy the F32 path of mul_mat_vec_f (mmvf.cu) exactly:
// per-thread strided float2 partial sums with ggml_cuda_mad, then the two-stage
// warp_reduce_sum reduction. Results are bit-identical to running the two MUL_MAT nodes
// with ggml_cuda_mul_mat_vec_f.

#pragma once

#include "common.cuh"

// Runs the pair of MUL_MAT nodes (dst_a = w_a * src1, dst_b = w_b * src1) in one launch.
// Returns false for anything the kernel does not handle, the caller then runs the nodes
// on the regular path.
bool ggml_cuda_f32_pair_mul_mat(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * w_a, ggml_tensor * dst_a,
                                const ggml_tensor * w_b, ggml_tensor * dst_b,
                                const ggml_tensor * src1);
