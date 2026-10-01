#pragma once

#include "common.cuh"

// SM70 (Volta) D256 Split-D prefill attention (staged f16, explicit mask).
// Device kernel: fattn-sm70-d256-kernel.cuh. The kernel is compiled by a
// dedicated sm_70-only object target, so the dispatch sites guard their calls
// with GGML_CUDA_SM70_D256_COMPILED.
#if defined(__CUDA_ARCH_LIST__)
#define GGML_CUDA_SM70_D256_COMPILED ggml_cuda_has_arch(GGML_CUDA_CC_VOLTA)
#else
#define GGML_CUDA_SM70_D256_COMPILED false
#endif

bool   ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst);
size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst);
void   ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Assist: the idle rank (role 1) pulls and attends the owner's KV tail [s, kv_end) on its own
// device, the owner rank (role 0) merges the result in its FA launcher. See ggml-backend.h.
size_t ggml_cuda_sm70_d256_assist_workspace_size(int q_pad, int heads_q);
bool   ggml_cuda_sm70_d256_assist_run(void * backend, const struct ggml_backend_meta_assist_rank * desc);
bool   ggml_cuda_sm70_d256_assist_scratch_layout(const struct ggml_tensor * dst, struct ggml_backend_meta_assist_scratch * layout);
