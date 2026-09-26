#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// One-shot P2P push AllReduce for small FP32 tensors.
//
// Kernel ported from 1Cat-vLLM `csrc/custom_all_reduce.cuh`
// (`sm70_cross_device_reduce_1stage_push`, Apache-2.0), which adapts
// SGLang-V100's one-shot push collective.
struct ggml_cuda_ar_push;

// devices[i] is the CUDA device number of rank i.
// Returns nullptr when n is outside [2, 8], when virtual devices are in use,
// when any device pair cannot do P2P, or when allocation fails.
ggml_cuda_ar_push * ggml_cuda_ar_push_init(const int * devices, size_t n);

// Release all device buffers owned by the push context.
void ggml_cuda_ar_push_free(ggml_cuda_ar_push * ar);

// In-place AllReduce (sum) across tensors[0..n-1].
// exact selects the f32 push kernel; otherwise the payload travels as f16.
// Returns false for unsupported inputs without doing any work, so the caller
// can fall back to NCCL. When it returns true the kernel is enqueued on each
// backend's stream.
bool ggml_cuda_ar_push_allreduce(
    ggml_cuda_ar_push * ar,
    ggml_backend_t    * backends,
    ggml_tensor       ** tensors,
    bool                exact);

// Admission check for a single rank's tensor, shared with ggml_cuda_ar_push_allreduce.
bool ggml_cuda_ar_push_can(
    const ggml_cuda_ar_push * ar,
    const ggml_tensor       * t,
    bool                      exact);

// Per-rank variant for callers that enqueue each rank from its own thread.
// Enqueues rank's part of the reduction on backends[rank]; returns false for
// inputs that ggml_cuda_ar_push_can rejects.
bool ggml_cuda_ar_push_allreduce_rank(
    ggml_cuda_ar_push * ar,
    ggml_backend_t      backend,
    size_t              rank,
    ggml_tensor       * t,
    bool                exact);
