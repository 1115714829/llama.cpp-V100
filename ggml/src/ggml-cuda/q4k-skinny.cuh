// Q4_K skinny GEMM for sm_70 (Volta).
//
// Design follows the QPN8 execution layout of q8-skinny.cu (itself adapted from 1Cat-vLLM
// fp8_qpn8_sm70.cu, Apache-2.0) and the int4 scale/bias folding of 1Cat's awq_qpn_m1_sm70.cu
// and ninfer-v100's q4 Volta kernels. The QPN8 layout is derived from dnv2003/v100-skinny
// (MIT) and its block-scale adaptation in haohervchb/sglang-V100. See LICENSE.v100-skinny in
// this directory and docs/design/v106-q4k-skinny.md for the layout and the numeric analysis.

#pragma once

#include "common.cuh"

// Weights are repacked in place from the row-major Q4_K layout (ne0 = K, ne1 = N) into:
//   codes: [N/32][K/16][32][ 8] uint8 at data,         N*K/2 bytes
//   meta:  [K/256][N/32][32][16] uint8 at data + N*K/2, N*K/16 bytes
// codes hold the 4-bit quants in QPN8 order, one 8-byte record per 16 K values and lane.
// meta holds the raw 16-byte super-block header (dm + 12 scale bytes), copied unchanged.
// The byte count is the same as Q4_K. A repacked tensor is tagged through tensor->extra;
// reading it with ggml_backend_tensor_get() returns the repacked bytes, there is no
// conversion back to the Q4_K layout.
bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t);

// Repacks all repackable weights of the graph that are not read by another op, then marks
// them. Repacking all of them up front keeps it out of CUDA graph captures.
void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);

// Returns true if the small-M kernel handled the multiplication. If it returns false the
// caller must expand the repacked weights to dense F16 and use the regular path.
bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Fused gate/up projection with a trailing silu(gate) * up, writing to the GLU node dst.
// Returns false for anything the kernel does not handle, the caller then runs the nodes
// on the regular path.
bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

// Runs 2 to 4 MUL_MAT nodes that share one src1 with a single input conversion and a single
// kernel launch. Every src0 must be a repacked Q4_K weight with the same K. Returns false if
// any node does not fit, the caller then runs them all on the regular path.
bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

// Expands repacked weights to dense F16 [N][K], for M > 16.
void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
