// QPN8 lane/column/physical-K permutations shared by the skinny GEMM kernels for sm_70 (Volta).
//
// Adapted from 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (Apache-2.0). The QPN8
// execution layout is derived from dnv2003/v100-skinny (MIT) and its block-scale adaptation
// in haohervchb/sglang-V100. See LICENSE.v100-skinny in this directory.

#pragma once

#include "common.cuh"

// ---- 1Cat fp8_qpn8_sm70.cu:39-50 (lane/column mapping, unchanged) ----

static __device__ __forceinline__ int qpn8_col_from_lane(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

static __device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

static __device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}
