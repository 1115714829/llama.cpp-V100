// GDN chunked prefill (sm_70): inclusive prefix sum of g over the tokens of each
// chunk, per (chunk, head). Hand written for arbitrary head counts; replaces the
// TileLang generated kernel, whose shared-memory staging was specialized for Hv = 12.
// The g layout ([T][H] f32, chunk-major addresses) matches the other GDN kernels.
// ----------------------------------------------------------------------------
namespace gdn_chunk_sm70 {

// g_cumsum[t][h] = sum of g_raw[u][h] over the chunk tokens u <= t (inclusive).
// One warp per (chunk, head), two consecutive tokens per lane, f32 accumulation.
__global__ void __launch_bounds__(128, 1) gdn_chunk_cumsum_kernel(const int* __restrict__ chunk_indices, const int* __restrict__ cu_seqlens, float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int num_tokens, int real_batch_size, int H) {
  const int64_t row = (int64_t) blockIdx.x * 2;
  const int batch_idx = chunk_indices[row];
  const int chunk_idx = chunk_indices[row + 1];

  int seq_start_idx = 0;
  if (0 <= batch_idx && batch_idx <= real_batch_size) {
    seq_start_idx = cu_seqlens[batch_idx];
  }
  int seq_end_idx = 0;
  if (-1 <= batch_idx && batch_idx < real_batch_size) {
    seq_end_idx = cu_seqlens[batch_idx + 1];
  }

  const int lane = threadIdx.x & 31;
  const int64_t idx = (int64_t) seq_start_idx + (int64_t) chunk_idx * 64 + 2 * lane;

  for (int h = (int) (threadIdx.x >> 5); h < H; h += (int) (blockDim.x >> 5)) {
    const bool v0 = 0 <= idx && idx < num_tokens && idx < seq_end_idx;
    const bool v1 = 0 <= idx + 1 && idx + 1 < num_tokens && idx + 1 < seq_end_idx;
    const float a = v0 ? g_raw[idx * H + h] : 0.0f;
    const float b = v1 ? g_raw[(idx + 1) * H + h] : 0.0f;
    // inclusive scan of the per-lane pair sums, then place each token inside its pair:
    // token 2*lane gets the sum of all earlier pairs plus a, token 2*lane+1 adds b as well
    float p = a + b;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float n = __shfl_up_sync(0xffffffffu, p, off);
      if (lane >= off) {
        p += n;
      }
    }
    float excl = __shfl_up_sync(0xffffffffu, p, 1); // sum of all earlier pairs
    if (lane == 0) {
      excl = 0.0f;
    }
    const float s0 = excl + a;
    const float s1 = p;
    if (v0) {
      g_cumsum[idx * H + h] = s0;
    }
    if (v1) {
      g_cumsum[(idx + 1) * H + h] = s1;
    }
  }
}

} // namespace gdn_chunk_sm70
