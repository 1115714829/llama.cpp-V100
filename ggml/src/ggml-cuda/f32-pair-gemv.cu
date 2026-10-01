#include "f32-pair-gemv.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_f32_pair_mul_mat(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * w_a, ggml_tensor * dst_a,
                                const ggml_tensor * w_b, ggml_tensor * dst_b,
                                const ggml_tensor * src1) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(w_a);
    GGML_UNUSED(dst_a);
    GGML_UNUSED(w_b);
    GGML_UNUSED(dst_b);
    GGML_UNUSED(src1);
    return false;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// One CTA computes one output column of both weights for all M tokens of the shared src1.
// Each loop step reads the input once and feeds both accumulators, so the input vector is
// streamed from global memory once per column pair. The per-thread accumulation and the
// reduction copy the F32 path of mul_mat_vec_f (mmvf.cu) exactly so results stay
// bit-identical to the two separate MUL_MAT nodes. M is compile-time: a runtime token loop
// with accumulator arrays is not bit-exact with the mmvf codegen on sm_70.
template <int M, int warp_size = 32>
__global__ static void __launch_bounds__(256) f32_pair_gemv_kernel(
        const float * GGML_CUDA_RESTRICT w_a, const float * GGML_CUDA_RESTRICT w_b,
        const float * GGML_CUDA_RESTRICT x,
        float * GGML_CUDA_RESTRICT dst_a, float * GGML_CUDA_RESTRICT dst_b,
        const int k, const int n, const int sa, const int sb, const int sx, const int d) {
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float * buf_a = smem;
    float * buf_b = smem + warp_size;

    if (blockDim.x > warp_size && tid < warp_size) {
        buf_a[tid] = 0.0f;
        buf_b[tid] = 0.0f;
    }
    if (blockDim.x > warp_size) {
        __syncthreads();
    }

    const float2 * wa2 = (const float2 *) (w_a + (size_t) c * sa);
    const float2 * wb2 = (const float2 *) (w_b + (size_t) c * sb);

    float acc_a[M];
    float acc_b[M];
#pragma unroll
    for (int j = 0; j < M; ++j) {
        acc_a[j] = 0.0f;
        acc_b[j] = 0.0f;
    }

    ggml_cuda_pdl_sync();

    const int k2 = k / 2;
    for (int col2 = tid; col2 < k2; col2 += blockDim.x) {
        const float2 wv_a = wa2[col2];
        const float2 wv_b = wb2[col2];
#pragma unroll
        for (int j = 0; j < M; ++j) {
            const float2 yv = ((const float2 *) (x + (size_t) j * sx))[col2];
            ggml_cuda_mad(acc_a[j], wv_a.x, yv.x);
            ggml_cuda_mad(acc_a[j], wv_a.y, yv.y);
            ggml_cuda_mad(acc_b[j], wv_b.x, yv.x);
            ggml_cuda_mad(acc_b[j], wv_b.y, yv.y);
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int j = 0; j < M; ++j) {
        acc_a[j] = warp_reduce_sum<warp_size>(acc_a[j]);
        acc_b[j] = warp_reduce_sum<warp_size>(acc_b[j]);
        if (blockDim.x > warp_size) {
            buf_a[tid / warp_size] = acc_a[j];
            buf_b[tid / warp_size] = acc_b[j];
            __syncthreads();
            if (tid < warp_size) {
                acc_a[j] = buf_a[tid];
                acc_a[j] = warp_reduce_sum<warp_size>(acc_a[j]);
                acc_b[j] = buf_b[tid];
                acc_b[j] = warp_reduce_sum<warp_size>(acc_b[j]);
            }
            __syncthreads();
        }
    }

    if (tid < M) {
        dst_a[(size_t) tid * d + c] = acc_a[tid];
        dst_b[(size_t) tid * d + c] = acc_b[tid];
    }
}

template <int M>
static void f32_pair_gemv_launch(const ggml_cuda_kernel_launch_params & params,
        const float * w_a, const float * w_b, const float * x,
        float * dst_a, float * dst_b,
        int k, int n, int sa, int sb, int sx, int d) {
    ggml_cuda_kernel_launch(f32_pair_gemv_kernel<M>, params,
            w_a, w_b, x, dst_a, dst_b, k, n, sa, sb, sx, d);
}

bool ggml_cuda_f32_pair_mul_mat(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * w_a, ggml_tensor * dst_a,
                                const ggml_tensor * w_b, ggml_tensor * dst_b,
                                const ggml_tensor * src1) {
    if (src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t k = src1->ne[0];
    const int64_t m = src1->ne[1];
    if (m < 1 || m > 3 || (k & 1)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || (src1->nb[1] & 7) != 0 || (src1->nb[2] & 7) != 0 || (src1->nb[3] & 7) != 0) {
        return false;
    }

    const ggml_tensor * ws[2]  = { w_a, w_b };
    ggml_tensor *       dts[2] = { dst_a, dst_b };
    for (int t = 0; t < 2; ++t) {
        const ggml_tensor * w  = ws[t];
        const ggml_tensor * dt = dts[t];
        if (w->type != GGML_TYPE_F32 || dt->type != GGML_TYPE_F32) {
            return false;
        }
        if (w->ne[0] != k || w->ne[1] != w_a->ne[1] || w->ne[2] != 1 || w->ne[3] != 1) {
            return false;
        }
        // same alignment rules as ggml_cuda_should_use_mmvf, so the pair kernel only runs
        // where the regular path would run the mmvf F32 kernel
        if (w->nb[0] != sizeof(float) || (w->nb[1] & 7) != 0 || (w->nb[2] & 7) != 0 || (w->nb[3] & 7) != 0) {
            return false;
        }
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(w) != ggml_backend_buffer_get_alloc_size(w->buffer, w) && w->view_src;
        if (bad_padding_clear) {
            return false;
        }
        if (dt->ne[0] != w->ne[1] || dt->ne[1] != m || dt->ne[2] != 1 || dt->ne[3] != 1) {
            return false;
        }
        if (dt->nb[0] != sizeof(float)) {
            return false;
        }
    }

    const int n  = (int) w_a->ne[1];
    const int sa = (int) (w_a->nb[1] / sizeof(float));
    const int sb = (int) (w_b->nb[1] / sizeof(float));
    const int sx = (int) (src1->nb[1] / sizeof(float));
    const int d  = (int) (dst_a->nb[1] / sizeof(float));

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // same block size selection as launch_mul_mat_vec_f_cuda, so each thread sums the
    // same strided subset of k and the reduction order matches
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (k + 2*warp_size - 1) / (2*warp_size);
    const int64_t max_block_size = 256;
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (k + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const size_t nbytes_shared = 2 * warp_size * sizeof(float);
    const dim3 block_nums((unsigned) n, 1, 1);
    const dim3 block_dims((unsigned) block_size_best, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = { block_nums, block_dims, nbytes_shared, ctx.stream() };

    const float * wa_ptr = (const float *) w_a->data;
    const float * wb_ptr = (const float *) w_b->data;
    const float * x_ptr  = (const float *) src1->data;
    float *       da_ptr = (float *) dst_a->data;
    float *       db_ptr = (float *) dst_b->data;

    if (m == 1) {
        f32_pair_gemv_launch<1>(launch_params, wa_ptr, wb_ptr, x_ptr, da_ptr, db_ptr, (int) k, n, sa, sb, sx, d);
    } else if (m == 2) {
        f32_pair_gemv_launch<2>(launch_params, wa_ptr, wb_ptr, x_ptr, da_ptr, db_ptr, (int) k, n, sa, sb, sx, d);
    } else {
        f32_pair_gemv_launch<3>(launch_params, wa_ptr, wb_ptr, x_ptr, da_ptr, db_ptr, (int) k, n, sa, sb, sx, d);
    }

    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
