#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64
#define CUDA_CPY_BATCH_MAX  16

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// copy n same-layout tensors with one kernel launch, one copy per blockIdx.y
bool ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * srcs, ggml_tensor * const * dsts, int n);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
