// Q2_K/Q3_K/Q4_K/Q5_K/Q6_K skinny GEMM for sm_70 (Volta).
//
// Design follows the QPN8 execution layout of q8-skinny.cu (itself adapted from 1Cat-vLLM
// fp8_qpn8_sm70.cu, Apache-2.0) and the int4 scale/bias folding of 1Cat's awq_qpn_m1_sm70.cu
// and ninfer-v100's q4 Volta kernels. The QPN8 layout is derived from dnv2003/v100-skinny
// (MIT) and its block-scale adaptation in haohervchb/sglang-V100. See LICENSE.v100-skinny in
// this directory and docs/design/v106-q4k-skinny.md for the layout and the numeric analysis.

#pragma once

#include "common.cuh"

// Weights are repacked in place from the row-major K-quant layout (ne0 = K, ne1 = N) into:
//   codes: [N/32][K/16][32][record_bytes] uint8 at data
//   meta:  [K/256][N/32][32][meta_bytes]  uint8 at data + codes_bytes
// record_bytes is 4/6/8/10/12 and meta_bytes is 20/14/16/16/18 for Q2_K/Q3_K/Q4_K/Q5_K/Q6_K,
// and 16*record_bytes + meta_bytes equals the block size of the type. codes hold the raw
// quants in QPN8 order, one record per 16 K values and lane; meta holds the raw super-block
// header, copied unchanged. A type keeps its byte count. A repacked tensor is tagged through
// tensor->extra; reading it with ggml_backend_tensor_get() returns the repacked bytes, there
// is no conversion back to the original layout.
bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t);

// Repacks all repackable weights of the graph that are not read by another op, then marks
// them. Repacking all of them up front keeps it out of CUDA graph captures.
void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);

// Returns true if the small-M kernel handled the multiplication (M up to 64; M = 17..64 runs
// the two-phase M=32 kernel on split-16 shapes). If it returns false the caller must expand
// the repacked weights to dense F16 and use the regular path.
bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Small-M gate/up pair: one kernel computes both projections and silu(gate) * up into dst.
bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

// Small-M multiple projections of one input: one input conversion and one launch for up to
// four repacked weights that share src1.
bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

// Expands repacked weights to dense F16 [N][K], for M > 64.
void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
