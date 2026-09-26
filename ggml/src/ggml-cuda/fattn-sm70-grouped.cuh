// Adapted from 1Cat-vLLM flash-attention-v100: include/fused_mma.h (Volta WMMA wrapper)
// and kernel/flash_decode_paged.cu (grouped verify kernel), https://github.com/1CatAI/1Cat-vLLM

#pragma once

#include "common.cuh"

// Volta WMMA wrapper, raw PTX. Fragments follow the sm_70 register layout.
#if defined(VOLTA_MMA_AVAILABLE)
namespace ggml_sm70_wmma {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <>
struct fragment<matrix_a, 16, 16, 16, half, row_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, col_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, row_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};

template <>
struct fragment<accumulator, 16, 16, 16, float> {
  float x[8];
  static constexpr int num_elements = 8;
};

template <int M, int N, int K>
__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, M, N, K, float>& frag, float value) {
#pragma unroll
  for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, row_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, row_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.b.sync.aligned.row.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, col_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr, const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m16n16k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  } else {
    asm volatile(
        "wmma.store.d.sync.aligned.col.m16n16k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  }
}

#define VOLTA_WMMA_MMA_F32(M, N, K, ALAY, BLAY)                            \
  __device__ __forceinline__ void mma_sync(                                \
      fragment<accumulator, M, N, K, float>& d,                            \
      const fragment<matrix_a, M, N, K, half, ALAY##_major>& a,            \
      const fragment<matrix_b, M, N, K, half, BLAY##_major>& b,            \
      const fragment<accumulator, M, N, K, float>& c) {                    \
    asm volatile(                                                          \
        "wmma.mma.sync.aligned." #ALAY "." #BLAY ".m" #M "n" #N "k" #K     \
        ".f32.f32 "                                                        \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "                                      \
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "                                \
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "                              \
        "{%24,%25,%26,%27,%28,%29,%30,%31};"                               \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),          \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])           \
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(a.x[4]), \
          "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]), "r"(b.x[0]), "r"(b.x[1]), \
          "r"(b.x[2]), "r"(b.x[3]), "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), \
          "r"(b.x[7]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));             \
  }

VOLTA_WMMA_MMA_F32(16, 16, 16, row, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, row, row)

#undef VOLTA_WMMA_MMA_F32

}  // namespace ggml_sm70_wmma
#endif // VOLTA_MMA_AVAILABLE

// Packed GQA flash attention for decode and speculative verification with n_q <= 16.
// Each CTA owns 48 query rows for one KV head: MAX_QUERY_TOKENS tokens x kHeadsPerCta
// query heads, split over the KV context and merged with flash_attn_combine_results.
constexpr float kXQANegInf = -1.0e30f;

constexpr int kGroupedVerifyHeads        = 6;
constexpr int kGroupedVerifyHeadDim      = 256;
constexpr int kGroupedVerifyRows         = 48;
constexpr int kGroupedVerifyBlockN       = 32;
constexpr int kGroupedVerifyQStride      = 264; // half per row, 528 B, 16 B aligned
constexpr int kGroupedVerifyKVStride     = 264;
constexpr int kGroupedVerifyScoreStride  = 32;
constexpr int kGroupedVerifyProbStride   = 40;
constexpr int kGroupedVerifyKVQ8RowBytes = 272; // 8 q8_0 blocks of 34 B
constexpr int kGroupedVerifyThreads      = 512;
constexpr int kGroupedVerifyWarps        = kGroupedVerifyThreads / WARP_SIZE;
constexpr int kGroupedVerifyQKWarps      = (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles  = (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp = kGroupedVerifyOutputTiles / kGroupedVerifyWarps;

template <int MAX_QUERY_TOKENS>
struct GroupedVerifyTraits {
    static_assert(MAX_QUERY_TOKENS == 8 || MAX_QUERY_TOKENS == 16, "grouped verify supports 8 and 16 query tokens");
    static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
    static constexpr int kHeadGroups  = kGroupedVerifyHeads / kHeadsPerCta;
    static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows, "the CTA must keep 48 rows");
};

struct alignas(256) GroupedVerifySmem {
    union {
        struct {
            alignas(16) __half  q[kGroupedVerifyRows * kGroupedVerifyQStride];
            alignas(16) __half  kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
            alignas(16) float   scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
            alignas(16) __half  probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
            alignas(16) uint8_t kv_stage[kGroupedVerifyBlockN * kGroupedVerifyKVQ8RowBytes];
        } compute;
        alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
    } storage;
    alignas(16) float row_max[kGroupedVerifyRows];
    alignas(16) float row_sum[kGroupedVerifyRows];
    alignas(16) float row_scale[kGroupedVerifyRows];
};

static_assert(sizeof(GroupedVerifySmem) <= 64 * 1024, "grouped verify must fit Volta's 64 KiB opt-in budget");
static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0, "output tiles must divide evenly across warps");

#if defined(VOLTA_MMA_AVAILABLE)

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// One raw KV tile held in registers between the global load and the shared write,
// so the global latency of tile i+1 hides behind the compute of tile i.
template <ggml_type type_KV>
struct GroupedVerifyKVRegs {
    static constexpr int kRowBytes  = type_KV == GGML_TYPE_F16 ? kGroupedVerifyHeadDim * (int) sizeof(__half)
                                                               : kGroupedVerifyKVQ8RowBytes;
    static constexpr int kVecsPerRow  = kRowBytes / 16;
    static constexpr int kVecsPerTile = kGroupedVerifyBlockN * kVecsPerRow;
    static constexpr int kVecsPerThread = (kVecsPerTile + kGroupedVerifyThreads - 1) / kGroupedVerifyThreads;
    static_assert(kVecsPerThread <= 2, "the prefetch must stay small");
    uint4 vec[kVecsPerThread];
};

// Sign-extend two packed int8 and scale them by d, all in half precision.
__device__ __forceinline__ __half2 grouped_verify_q8_pair_half2(const uint32_t packed, const __half2 d2) {
    const __half2 q = __halves2half2(
        __short2half_rn((int8_t) (uint8_t)  packed),
        __short2half_rn((int8_t) (uint8_t) (packed >> 8)));
    return __hmul2(d2, q);
}

__device__ __forceinline__ uint32_t grouped_verify_half2_uint(const __half2 h) {
    uint32_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

// Issue the global loads for one 32 x 256 KV tile. Rows with kv_idx >= n_kv load as zero.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_prefetch_kv(
        GroupedVerifyKVRegs<type_KV> & regs,
        const char * __restrict__ KV, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int seq, const int kv_head, const int tile_start, const int n_kv) {
    static_assert(type_KV == GGML_TYPE_F16 || type_KV == GGML_TYPE_Q8_0, "unsupported KV type");
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerTile = GroupedVerifyKVRegs<type_KV>::kVecsPerTile;
    const int tid = threadIdx.x;
    const char * tile_base = KV + int64_t(tile_start)*nb11 + kv_head*nb12 + int64_t(seq)*nb13;
#pragma unroll
    for (int i = 0; i < GroupedVerifyKVRegs<type_KV>::kVecsPerThread; ++i) {
        const int idx = tid + i * kGroupedVerifyThreads;
        if (idx >= kVecsPerTile) {
            continue;
        }
        const int row     = idx / kVecsPerRow;
        const int vec_col = idx % kVecsPerRow;
        if (tile_start + row < n_kv) {
            const char * src = tile_base + row*nb11 + vec_col*16;
            regs.vec[i] = __ldg(reinterpret_cast<const uint4 *>(src));
        } else {
            regs.vec[i] = make_uint4(0, 0, 0, 0);
        }
    }
}

// fp16 tiles go straight into the half panel, q8_0 tiles into the raw staging buffer.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_store_kv(
        __half * shared_kv, uint8_t * kv_stage, const GroupedVerifyKVRegs<type_KV> & regs) {
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerTile = GroupedVerifyKVRegs<type_KV>::kVecsPerTile;
    constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < GroupedVerifyKVRegs<type_KV>::kVecsPerThread; ++i) {
        const int idx = tid + i * kGroupedVerifyThreads;
        if (idx >= kVecsPerTile) {
            continue;
        }
        if constexpr (type_KV == GGML_TYPE_F16) {
            const int row     = idx / kVecsPerRow;
            const int vec_col = idx % kVecsPerRow;
            reinterpret_cast<uint4 *>(shared_kv)[row * kSharedStrideVec + vec_col] = regs.vec[i];
        } else {
            reinterpret_cast<uint4 *>(kv_stage)[idx] = regs.vec[i];
        }
    }
}

// Dequantize a staged q8_0 tile into the half panel. No-op for fp16, which is stored there directly.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_dequant_kv(
        __half * shared_kv, const uint8_t * kv_stage) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        return;
    }
    constexpr int kColsPerItem = 8;
    constexpr int kGroupsPerRow = kGroupedVerifyHeadDim / kColsPerItem;
    constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;
    const int tid = threadIdx.x;
#pragma unroll
    for (int idx = tid; idx < kGroupedVerifyBlockN * kGroupsPerRow; idx += kGroupedVerifyThreads) {
        const int row = idx / kGroupsPerRow;
        const int c   = (idx % kGroupsPerRow) * kColsPerItem;
        const int blk = c / 32;
        const int base = row * kGroupedVerifyKVQ8RowBytes + blk * 34;
        const __half d = *reinterpret_cast<const __half *>(kv_stage + base);
        const __half2 d2 = __half2half2(d);
        // 8 columns = 16 B, and base + 2 + (c % 32) is even, so use u16 loads and one uint4 store.
        const uint16_t * packed = reinterpret_cast<const uint16_t *>(kv_stage + base + 2 + (c % 32));
        uint4 out;
        out.x = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[0], d2));
        out.y = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[1], d2));
        out.z = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[2], d2));
        out.w = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[3], d2));
        __half * out_ptr = reinterpret_cast<__half *>(shared_kv + row * kGroupedVerifyKVStride) + c;
        *reinterpret_cast<uint4 *>(out_ptr) = out;
    }
}

__device__ __forceinline__ void grouped_verify_qk(
        const __half * __restrict__ shared_q, const __half * __restrict__ shared_k,
        float * __restrict__ shared_scores, const float qk_scale) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id >= kGroupedVerifyQKWarps) {
        return;
    }
    const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
    const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_a, 16, 16, 16, half, ggml_sm70_wmma::row_major> q_fragment;
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_b, 16, 16, 16, half, ggml_sm70_wmma::col_major> k_fragment;
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float> score_fragment;
    ggml_sm70_wmma::fill_fragment(score_fragment, 0.0f);

#pragma unroll
    for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
        ggml_sm70_wmma::load_matrix_sync(
            q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset, kGroupedVerifyQStride);
        // K is row-major [N, D]. The same bytes represent K^T as a col-major
        // [D, N] matrix, which is the B operand needed by Q @ K^T.
        ggml_sm70_wmma::load_matrix_sync(
            k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset, kGroupedVerifyKVStride);
        ggml_sm70_wmma::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
    }
#pragma unroll
    for (int i = 0; i < score_fragment.num_elements; ++i) {
        score_fragment.x[i] *= qk_scale;
    }
    ggml_sm70_wmma::store_matrix_sync(
        shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
        score_fragment, kGroupedVerifyScoreStride, ggml_sm70_wmma::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
        ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float>& fragment,
        const float * __restrict__ row_scale, const int tile_row_start) {
    const int lane = threadIdx.x & 31;
    const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
    const float first_scale  = row_scale[tile_row_start + row];
    const float second_scale = row_scale[tile_row_start + row + 2];
    fragment.x[0] *= first_scale;
    fragment.x[1] *= first_scale;
    fragment.x[2] *= second_scale;
    fragment.x[3] *= second_scale;
    fragment.x[4] *= first_scale;
    fragment.x[5] *= first_scale;
    fragment.x[6] *= second_scale;
    fragment.x[7] *= second_scale;
}

#endif // VOLTA_MMA_AVAILABLE

template <int MAX_QUERY_TOKENS, ggml_type type_K, ggml_type type_V>
__launch_bounds__(kGroupedVerifyThreads, 1)
static __global__ void flash_attn_ext_sm70_grouped(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        float      * dst_partial_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const int32_t n_q,
        const int32_t n_kv,
        const int32_t n_heads,
        const int32_t n_splits,
        const int32_t ne33,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q            = Q_ptr;
    const char * GGML_CUDA_RESTRICT K            = K_ptr;
    const char * GGML_CUDA_RESTRICT V            = V_ptr;
    const char * GGML_CUDA_RESTRICT mask         = mask_ptr;
    float       * GGML_CUDA_RESTRICT dst_partial = dst_partial_ptr;
    float2      * GGML_CUDA_RESTRICT dst_meta    = dst_meta_ptr;

    using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
    constexpr int kHeadsPerCta = Traits::kHeadsPerCta;
    constexpr int kHeadGroups  = Traits::kHeadGroups;

    const int kv_head    = blockIdx.x / kHeadGroups;
    const int head_group = blockIdx.x % kHeadGroups;
    const int split_id   = blockIdx.y;
    const int seq        = blockIdx.z;

    const int total_tiles    = (n_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
    const int base_tiles     = total_tiles / n_splits;
    const int extra_tiles    = total_tiles % n_splits;
    const int split_tile_start = split_id * base_tiles + min(split_id, extra_tiles);
    const int split_tiles    = base_tiles + (split_id < extra_tiles ? 1 : 0);
    const int split_start    = split_tile_start * kGroupedVerifyBlockN;
    const int split_end      = min(n_kv, split_start + split_tiles * kGroupedVerifyBlockN);

    const int64_t seq_mask_off = mask ? nb33 * (seq % ne33) : 0;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char grouped_verify_smem_raw[];
    GroupedVerifySmem& smem = *reinterpret_cast<GroupedVerifySmem*>(grouped_verify_smem_raw);
    __half * shared_q      = smem.storage.compute.q;
    __half * shared_kv     = smem.storage.compute.kv;
    float  * shared_scores = smem.storage.compute.scores;
    __half * shared_probs  = smem.storage.compute.probs;
    uint8_t * kv_stage     = smem.storage.compute.kv_stage;

    // Q is F32 [256, n_q, n_heads, n_seq], one shared row per (token, head).
    for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim; idx += kGroupedVerifyThreads) {
        const int row        = idx / kGroupedVerifyHeadDim;
        const int d          = idx % kGroupedVerifyHeadDim;
        const int token_idx  = row / kHeadsPerCta;
        const int local_head = row % kHeadsPerCta;
        const int head = kv_head * kGroupedVerifyHeads + head_group * kHeadsPerCta + local_head;
        __half * dst = shared_q + row * kGroupedVerifyQStride + d;
        if (token_idx < n_q) {
            const float val = *(const float *) (Q + token_idx*nb01 + head*nb02 + int64_t(seq)*nb03 + d*sizeof(float));
            *dst = __float2half_rn(val);
        } else {
            *dst = __float2half_rn(0.0f);
        }
    }
    if (tid < kGroupedVerifyRows) {
        smem.row_max[tid]   = kXQANegInf;
        smem.row_sum[tid]   = 0.0f;
        smem.row_scale[tid] = 1.0f;
    }
    __syncthreads();

    ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float>
        output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        ggml_sm70_wmma::fill_fragment(output_fragments[fragment_idx], 0.0f);
    }

    // Tile i+1 is fetched into registers while tile i is computed.
    GroupedVerifyKVRegs<type_K> k_regs;
    GroupedVerifyKVRegs<type_V> v_regs;
    flash_attn_sm70_grouped_prefetch_kv<type_K>(k_regs, K, nb11, nb12, nb13, seq, kv_head, split_start, n_kv);

    for (int tile_start = split_start; tile_start < split_end; tile_start += kGroupedVerifyBlockN) {
        // K: q8_0 first fills the staging buffer, fp16 goes straight into the panel.
        if constexpr (type_K == GGML_TYPE_F16) {
            __syncthreads(); // the previous PV must be done reading the panel
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs);
            __syncthreads();
        } else {
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs);
            __syncthreads();
            flash_attn_sm70_grouped_dequant_kv<type_K>(shared_kv, kv_stage);
            __syncthreads();
        }

        // Hide the next global loads behind QK.
        flash_attn_sm70_grouped_prefetch_kv<type_V>(v_regs, V, nb21, nb22, nb23, seq, kv_head, tile_start, n_kv);
        if (tile_start + kGroupedVerifyBlockN < split_end) {
            flash_attn_sm70_grouped_prefetch_kv<type_K>(
                k_regs, K, nb11, nb12, nb13, seq, kv_head, tile_start + kGroupedVerifyBlockN, n_kv);
        }

        grouped_verify_qk(shared_q, shared_kv, shared_scores, scale);
        __syncthreads(); // scores are ready and QK is done with the panel

        // V: the panel is free once QK is done, the staging buffer once the K dequantize is done.
        flash_attn_sm70_grouped_store_kv<type_V>(shared_kv, kv_stage, v_regs);

#pragma unroll
        for (int row = warp_id; row < kGroupedVerifyRows; row += kGroupedVerifyWarps) {
            const int token_idx = row / kHeadsPerCta;
            const int kv_idx = tile_start + lane_id;
            bool  visible  = false;
            float mask_val = 0.0f;
            if (token_idx < n_q && kv_idx < n_kv) {
                if (mask) {
                    const half * mask_row = (const half *) (mask + seq_mask_off + nb31*token_idx);
                    mask_val = __half2float(mask_row[kv_idx]);
                    visible  = mask_val > -INFINITY;
                } else {
                    visible = true;
                }
            }
            const float score = visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id] + mask_val : kXQANegInf;
            const float tile_max_lane = warp_reduce_max(score);
            const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
            const float old_max = smem.row_max[row];
            const float new_max = fmaxf(old_max, tile_max);
            const float probability = visible ? __expf(fmaxf(score - new_max, -80.0f)) : 0.0f;
            const float tile_sum_lane = warp_reduce_sum(probability);
            const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
            const float exp_diff = tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
            shared_probs[row * kGroupedVerifyProbStride + lane_id] = __float2half_rn(probability);
            // Finish every lane's shared-state reads before lane 0 overwrites the
            // online maximum. Shuffle synchronization does not order memory.
            __syncwarp();
            if (lane_id == 0) {
                if (tile_sum > 0.0f) {
                    smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
                    smem.row_max[row] = new_max;
                }
                smem.row_scale[row] = exp_diff;
            }
        }
        __syncthreads();
        if constexpr (type_V == GGML_TYPE_Q8_0) {
            flash_attn_sm70_grouped_dequant_kv<type_V>(shared_kv, kv_stage);
        }
#pragma unroll
        for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
            const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
            const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
            grouped_verify_scale_output_fragment(output_fragments[fragment_idx], smem.row_scale, m_tile * 16);
        }
        __syncthreads(); // probs and the V panel are ready for PV

#pragma unroll
        for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
            const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
            const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
            const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
            ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_a, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                probability_fragment;
            ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_b, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                value_fragment;
            auto& pv_fragment = output_fragments[fragment_idx];
#pragma unroll
            for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
                ggml_sm70_wmma::load_matrix_sync(
                    probability_fragment,
                    shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
                    kGroupedVerifyProbStride);
                ggml_sm70_wmma::load_matrix_sync(
                    value_fragment,
                    shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,
                    kGroupedVerifyKVStride);
                ggml_sm70_wmma::mma_sync(pv_fragment, probability_fragment, value_fragment, pv_fragment);
            }
        }
        // No sync at the loop bottom: the next iteration starts by writing the staging
        // buffer, which PV does not read, and its own syncthreads() orders the panel.
    }

    // The compute buffers are dead. Reuse their storage for the dense FP32 output.
    __syncthreads();
    float* shared_output = smem.storage.output;
#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
        const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
        const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
        ggml_sm70_wmma::store_matrix_sync(
            shared_output + m_tile * 16 * kGroupedVerifyHeadDim + d_tile * 16,
            output_fragments[fragment_idx], kGroupedVerifyHeadDim, ggml_sm70_wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim; idx += kGroupedVerifyThreads) {
        const int row        = idx / kGroupedVerifyHeadDim;
        const int d          = idx % kGroupedVerifyHeadDim;
        const int token_idx  = row / kHeadsPerCta;
        const int local_head = row % kHeadsPerCta;
        if (token_idx >= n_q) {
            continue;
        }
        const int head = kv_head * kGroupedVerifyHeads + head_group * kHeadsPerCta + local_head;
        const int64_t j = (int64_t(seq) * n_q + token_idx) * n_heads + head;
        // Unnormalized numerator, normalized by flash_attn_combine_results.
        dst_partial[(j * n_splits + split_id) * kGroupedVerifyHeadDim + d] = shared_output[idx];
    }
    if (tid < kGroupedVerifyRows) {
        const int token_idx  = tid / kHeadsPerCta;
        const int local_head = tid % kHeadsPerCta;
        if (token_idx < n_q) {
            const int head = kv_head * kGroupedVerifyHeads + head_group * kHeadsPerCta + local_head;
            const int64_t j = (int64_t(seq) * n_q + token_idx) * n_heads + head;
            const float sum = smem.row_sum[tid];
            dst_meta[j * n_splits + split_id] = make_float2(sum > 0.0f ? smem.row_max[tid] : kXQANegInf, sum);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, dst_partial_ptr, dst_meta_ptr, scale,
        n_q, n_kv, n_heads, n_splits, ne33,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        nb31, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
}

bool ggml_cuda_flash_attn_ext_sm70_grouped_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_sm70_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
