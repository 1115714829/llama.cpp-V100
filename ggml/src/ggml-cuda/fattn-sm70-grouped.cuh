// Adapted from 1Cat-vLLM (Apache-2.0), https://github.com/1CatAI/1Cat-vLLM:
//   include/fused_mma.h (Volta WMMA wrapper)
//   csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu (grouped verify kernel)
// K/V cache types: F16, Q8_0 or Q4_0. Quantized tiles are staged in raw block
// form and dequantized into the shared memory panel.

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

constexpr int kGroupedVerifyHeadDim      = 256;
constexpr int kGroupedVerifyRows         = 48;
constexpr int kGroupedVerifyBlockN       = 64;
constexpr int kGroupedVerifyQStride      = 264; // half per row, 528 B, 16 B aligned
constexpr int kGroupedVerifyKVStride     = 264;
constexpr int kGroupedVerifyScoreStride  = 64;
constexpr int kGroupedVerifyProbStride   = 72; // BlockN + 8 half, keeps the WMMA A loads conflict free
constexpr int kGroupedVerifyKVQ8RowBytes = 272; // 8 q8_0 blocks of 34 B
constexpr int kGroupedVerifyKVQ4RowBytes = 144; // 8 q4_0 blocks of 18 B
constexpr int kGroupedVerifyThreads      = 512;
constexpr int kGroupedVerifyWarps        = kGroupedVerifyThreads / WARP_SIZE;
constexpr int kGroupedVerifyQKWarps      = (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles  = (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp = kGroupedVerifyOutputTiles / kGroupedVerifyWarps;
constexpr int kGroupedVerifyRowsPerWarp  = kGroupedVerifyRows / kGroupedVerifyWarps;
// Two adjacent KV columns per lane, so mask, scores and probabilities move in pairs.
constexpr int kGroupedVerifyColsPerLane  = kGroupedVerifyBlockN / WARP_SIZE;
// The quantized (q8_0/q4_0) staging buffer is split into one row block per warp. A warp only reads
// back its own rows, so the store and the dequantize need no block wide fence.
constexpr int kGroupedVerifyStageRowsPerWarp = kGroupedVerifyBlockN / kGroupedVerifyWarps;

static_assert(kGroupedVerifyBlockN % 16 == 0, "the KV tile must be a multiple of the WMMA N");
static_assert(kGroupedVerifyQKWarps <= kGroupedVerifyWarps, "QK tiles must fit into the warps");
static_assert(kGroupedVerifyColsPerLane == 2, "the softmax handles two adjacent columns per lane");
static_assert(kGroupedVerifyRows % kGroupedVerifyWarps == 0, "softmax rows must divide evenly");
static_assert(kGroupedVerifyWarps == kGroupedVerifyHeadDim / 16, "one warp per V output tile");
static_assert(kGroupedVerifyOutputTilesPerWarp == kGroupedVerifyRows / 16, "one accumulator per M tile");
static_assert(kGroupedVerifyBlockN % kGroupedVerifyWarps == 0, "staging rows must divide evenly");

template <int MAX_QUERY_TOKENS, int HEADS>
struct GroupedVerifyTraits {
    static_assert(MAX_QUERY_TOKENS == 8 || MAX_QUERY_TOKENS == 16, "grouped verify supports 8 and 16 query tokens");
    static_assert(HEADS == 2 || HEADS == 4 || HEADS == 6, "grouped verify supports GQA 2, 4 and 6");
    static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
    static constexpr int kHeadGroups  = (HEADS + kHeadsPerCta - 1) / kHeadsPerCta;
    static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows, "the CTA must keep 48 rows");
};

// Shared memory budget (Volta allows 96 KiB per block with opt-in):
//   q        48 * 264 * 2 = 25344 B
//   kv       64 * 264 * 2 = 33792 B  one K or V tile panel
//   scores   48 *  64 * 4 = 12288 B
//   probs    48 *  72 * 2 =  6912 B
//   stage    64 * 272     = 17408 B  raw q8_0/q4_0 tile (q4_0 rows use 144 B)
//   rows          3 * 48 * 4 =  576 B
//   total                  = 96512 B
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

static_assert(sizeof(GroupedVerifySmem) <= 96 * 1024, "grouped verify must fit Volta's 96 KiB opt-in budget");
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
// so the global latency of tile i+1 hides behind the compute of tile i. A q4_0 tile
// needs only 2 uint4 per thread, so two tiles fit the register budget, see
// kPrefetchTiles. Each warp owns a contiguous row block of the staging buffer, see
// kGroupedVerifyStageRowsPerWarp.
template <ggml_type type_KV>
struct GroupedVerifyKVRegs {
    static constexpr int kRowBytes  = type_KV == GGML_TYPE_F16  ? kGroupedVerifyHeadDim * (int) sizeof(__half)
                                     : type_KV == GGML_TYPE_Q4_0 ? kGroupedVerifyKVQ4RowBytes
                                                                 : kGroupedVerifyKVQ8RowBytes;
    static constexpr int kVecsPerRow  = kRowBytes / 16;
    static constexpr int kVecsPerWarp = kGroupedVerifyStageRowsPerWarp * kVecsPerRow;
    static constexpr int kVecsPerTile = (kVecsPerWarp + WARP_SIZE - 1) / WARP_SIZE;
    static constexpr int kPrefetchTiles = type_KV == GGML_TYPE_Q4_0 ? 2 : 1;
    static constexpr int kVecsPerThread = kPrefetchTiles * kVecsPerTile;
    static_assert(kVecsPerThread <= 4, "the prefetch must stay small");
    uint4 vec[kVecsPerThread];
};

// Sign-extend two packed int8 and scale them by d, all in half precision.
__device__ __forceinline__ __half2 grouped_verify_q8_pair_half2(const uint32_t packed, const __half2 d2) {
    const __half2 q = __halves2half2(
        __short2half_rn((int8_t) (uint8_t)  packed),
        __short2half_rn((int8_t) (uint8_t) (packed >> 8)));
    return __hmul2(d2, q);
}

// Unpack one q4_0 value pair (two code bytes in the low half of packed) and scale it by d,
// all in half precision. q4_0 packs values 0..15 into the low nibbles of qs[0..15] and
// 16..31 into the high nibbles.
__device__ __forceinline__ __half2 grouped_verify_q4_pair_half2(const uint32_t packed, const bool hi_half, const __half2 d2) {
    const int shift = hi_half ? 4 : 0;
    const __half2 q = __halves2half2(
        __short2half_rn((int) ((packed >> shift) & 0xf) - 8),
        __short2half_rn((int) ((packed >> (8 + shift)) & 0xf) - 8));
    return __hmul2(d2, q);
}

__device__ __forceinline__ uint32_t grouped_verify_half2_uint(const __half2 h) {
    uint32_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

// Issue the global loads for the staging rows of this warp into the register slot of
// one tile. Rows with kv_idx >= n_kv load as zero.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_prefetch_kv(
        GroupedVerifyKVRegs<type_KV> & regs,
        const char * __restrict__ KV, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int seq, const int kv_head, const int tile_start, const int n_kv, const int tile) {
    static_assert(type_KV == GGML_TYPE_F16 || type_KV == GGML_TYPE_Q8_0 || type_KV == GGML_TYPE_Q4_0,
                  "unsupported KV type");
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerWarp = GroupedVerifyKVRegs<type_KV>::kVecsPerWarp;
    constexpr int kVecsPerTile = GroupedVerifyKVRegs<type_KV>::kVecsPerTile;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
    const char * tile_base = KV + int64_t(tile_start)*nb11 + kv_head*nb12 + int64_t(seq)*nb13;
#pragma unroll
    for (int i = 0; i < kVecsPerTile; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        if (idx >= kVecsPerWarp) {
            continue;
        }
        const int row     = row_base + idx / kVecsPerRow;
        const int vec_col = idx % kVecsPerRow;
        if (tile_start + row < n_kv) {
            regs.vec[tile * kVecsPerTile + i] = __ldg(reinterpret_cast<const uint4 *>(tile_base + row*nb11 + vec_col*16));
        } else {
            regs.vec[tile * kVecsPerTile + i] = make_uint4(0, 0, 0, 0);
        }
    }
}

// fp16 tiles go straight into the half panel, quantized (q8_0/q4_0) tiles into the raw staging buffer.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_store_kv(
        __half * shared_kv, uint8_t * kv_stage, const GroupedVerifyKVRegs<type_KV> & regs, const int tile) {
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerWarp = GroupedVerifyKVRegs<type_KV>::kVecsPerWarp;
    constexpr int kVecsPerTile = GroupedVerifyKVRegs<type_KV>::kVecsPerTile;
    constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
#pragma unroll
    for (int i = 0; i < kVecsPerTile; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        if (idx >= kVecsPerWarp) {
            continue;
        }
        const int row     = row_base + idx / kVecsPerRow;
        const int vec_col = idx % kVecsPerRow;
        if constexpr (type_KV == GGML_TYPE_F16) {
            reinterpret_cast<uint4 *>(shared_kv)[row * kSharedStrideVec + vec_col] = regs.vec[tile * kVecsPerTile + i];
        } else {
            reinterpret_cast<uint4 *>(kv_stage)[row * kVecsPerRow + vec_col] = regs.vec[tile * kVecsPerTile + i];
        }
    }
}

// Dequantize a staged quantized tile into the half panel. No-op for fp16, which is stored there directly.
// Only the staging rows of this warp are read back, see kGroupedVerifyStageRowsPerWarp.
template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_dequant_kv(
        __half * shared_kv, const uint8_t * kv_stage) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        return;
    }
    constexpr int kColsPerItem = 8;
    constexpr int kGroupsPerRow = kGroupedVerifyHeadDim / kColsPerItem;
    constexpr int kItemsPerWarp = kGroupedVerifyStageRowsPerWarp * kGroupsPerRow;
    constexpr int kBlockBytes   = type_KV == GGML_TYPE_Q4_0 ? 18 : 34;
    constexpr int kRowBytes     = type_KV == GGML_TYPE_Q4_0 ? kGroupedVerifyKVQ4RowBytes
                                                            : kGroupedVerifyKVQ8RowBytes;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
#pragma unroll
    for (int idx = lane_id; idx < kItemsPerWarp; idx += WARP_SIZE) {
        const int row = row_base + idx / kGroupsPerRow;
        const int c   = (idx % kGroupsPerRow) * kColsPerItem;
        const int blk = c / 32;
        const int base = row * kRowBytes + blk * kBlockBytes;
        const __half d = *reinterpret_cast<const __half *>(kv_stage + base);
        const __half2 d2 = __half2half2(d);
        uint4 out;
        if constexpr (type_KV == GGML_TYPE_Q4_0) {
            // 8 columns = 8 values of one nibble half; codes are 8 consecutive bytes.
            // The 18 B block makes qs only 2 B aligned, so assemble each u32 code word
            // from two u16 loads and shift the nibbles out.
            const uint8_t * codes = kv_stage + base + 2 + (c & 15);
            const uint32_t w0 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes)
                              | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 2) << 16;
            const uint32_t w1 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 4)
                              | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 6) << 16;
            const bool hi_half = (c % 32) >= 16;
            out.x = grouped_verify_half2_uint(grouped_verify_q4_pair_half2(w0,      hi_half, d2));
            out.y = grouped_verify_half2_uint(grouped_verify_q4_pair_half2(w0 >> 16, hi_half, d2));
            out.z = grouped_verify_half2_uint(grouped_verify_q4_pair_half2(w1,      hi_half, d2));
            out.w = grouped_verify_half2_uint(grouped_verify_q4_pair_half2(w1 >> 16, hi_half, d2));
        } else {
            // 8 columns = 16 B, and base + 2 + (c % 32) is even, so use u16 loads and one uint4 store.
            const uint16_t * packed = reinterpret_cast<const uint16_t *>(kv_stage + base + 2 + (c % 32));
            out.x = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[0], d2));
            out.y = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[1], d2));
            out.z = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[2], d2));
            out.w = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[3], d2));
        }
        __half * out_ptr = reinterpret_cast<__half *>(shared_kv + row * kGroupedVerifyKVStride) + c;
        *reinterpret_cast<uint4 *>(out_ptr) = out;
    }
}

// One 16 x 16 QK tile per warp. BlockN = 64 gives 3 x 4 = 12 tiles for the first 12 warps,
// twice the tensor core parallelism per KV token of a BlockN = 32 tile.
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

// Additive mask values of the two adjacent columns handled by one lane.
struct GroupedVerifyMaskPair {
    float value[2];
    bool  visible[2];
};

// Load the mask pair of one row. The result feeds the softmax after QK, so the global
// latency of this load overlaps the tensor core work. mask == nullptr makes every
// column below n_kv visible. mask_is_range makes the row an [lo, hi) pair instead.
__device__ __forceinline__ GroupedVerifyMaskPair grouped_verify_load_mask_pair(
        const char * __restrict__ mask, const int64_t seq_mask_off, const int64_t nb31,
        const int token_idx, const int n_q, const int kv_idx, const int n_kv,
        const bool mask_is_range) {
    GroupedVerifyMaskPair pair;
    pair.value[0]   = 0.0f;
    pair.value[1]   = 0.0f;
    pair.visible[0] = false;
    pair.visible[1] = false;
    if (token_idx >= n_q || kv_idx >= n_kv) {
        return pair;
    }
    if (mask == nullptr) {
        pair.visible[0] = true;
        pair.visible[1] = kv_idx + 1 < n_kv;
        return pair;
    }
    if (mask_is_range) {
        const int32_t * r = reinterpret_cast<const int32_t *>(mask + seq_mask_off + nb31 * token_idx);
        const int lo = r[0];
        const int hi = r[1];
        pair.visible[0] = kv_idx >= lo && kv_idx < hi;
        pair.visible[1] = kv_idx + 1 < n_kv && kv_idx + 1 >= lo && kv_idx + 1 < hi;
        return pair;
    }
    const __half * mask_row = reinterpret_cast<const __half *>(mask + seq_mask_off + nb31 * token_idx);
    pair.value[0] = __half2float(mask_row[kv_idx]);
    pair.visible[0] = pair.value[0] > -INFINITY;
    if (kv_idx + 1 < n_kv) {
        pair.value[1] = __half2float(mask_row[kv_idx + 1]);
        pair.visible[1] = pair.value[1] > -INFINITY;
    }
    return pair;
}

// Online softmax for one BlockN tile. Warp w owns rows w, w+16, w+32 and each lane the
// two adjacent columns (2*lane, 2*lane+1). FP32 max and sum, half probabilities, exactly
// like the original per-row update; only the reduction order changed with the tile width.
__device__ __forceinline__ void grouped_verify_softmax_tile(
        const float * __restrict__ shared_scores, __half * __restrict__ shared_probs,
        float * __restrict__ row_max, float * __restrict__ row_sum, float * __restrict__ row_scale,
        const GroupedVerifyMaskPair * __restrict__ row_mask, const int warp_id, const int lane_id) {
    const int col = kGroupedVerifyColsPerLane * lane_id;
#pragma unroll
    for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
        const int row = warp_id + i * kGroupedVerifyWarps;
        const float2 score_pair = *reinterpret_cast<const float2 *>(
            shared_scores + row * kGroupedVerifyScoreStride + col);
        const float score0 = row_mask[i].visible[0] ? score_pair.x + row_mask[i].value[0] : kXQANegInf;
        const float score1 = row_mask[i].visible[1] ? score_pair.y + row_mask[i].value[1] : kXQANegInf;
        const float tile_max = __shfl_sync(0xffffffffu, warp_reduce_max(fmaxf(score0, score1)), 0);
        const float old_max = row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability0 = row_mask[i].visible[0] ? __expf(fmaxf(score0 - new_max, -80.0f)) : 0.0f;
        const float probability1 = row_mask[i].visible[1] ? __expf(fmaxf(score1 - new_max, -80.0f)) : 0.0f;
        const float tile_sum = __shfl_sync(0xffffffffu, warp_reduce_sum(probability0 + probability1), 0);
        const float exp_diff = tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        *reinterpret_cast<__half2 *>(shared_probs + row * kGroupedVerifyProbStride + col) =
            __floats2half2_rn(probability0, probability1);
        // Finish every lane's shared-state reads before lane 0 overwrites the
        // online maximum. Shuffle synchronization does not order memory.
        __syncwarp();
        if (lane_id == 0) {
            if (tile_sum > 0.0f) {
                row_sum[row] = row_sum[row] * exp_diff + tile_sum;
                row_max[row] = new_max;
            }
            row_scale[row] = exp_diff;
        }
    }
}

#endif // VOLTA_MMA_AVAILABLE

template <int MAX_QUERY_TOKENS, int HEADS, ggml_type type_K, ggml_type type_V>
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
        const int64_t nb31, const int64_t nb33,
        const int32_t mask_is_range) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q            = Q_ptr;
    const char * GGML_CUDA_RESTRICT K            = K_ptr;
    const char * GGML_CUDA_RESTRICT V            = V_ptr;
    const char * GGML_CUDA_RESTRICT mask         = mask_ptr;
    float       * GGML_CUDA_RESTRICT dst_partial = dst_partial_ptr;
    float2      * GGML_CUDA_RESTRICT dst_meta    = dst_meta_ptr;

    using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS, HEADS>;
    constexpr int kHeadsPerCta = Traits::kHeadsPerCta;
    constexpr int kHeadGroups  = Traits::kHeadGroups;

    const int kv_head    = blockIdx.x / kHeadGroups;
    const int head_group = blockIdx.x % kHeadGroups;
    const int split_id   = blockIdx.y;
    const int seq        = blockIdx.z;

    const int64_t seq_mask_off = mask ? nb33 * (seq % ne33) : 0;

    // With a range mask only the KV rows below the widest [lo, hi) of this sequence's query rows can
    // contribute, so the tiles are split over [0, kv_end) instead of the whole view: a verify batch that
    // views the whole cache keeps the parallelism of its real length.
    int kv_end = n_kv;
    if (mask_is_range) {
        int hi_max = 0;
        for (int t = 0; t < n_q; ++t) {
            const int32_t * r = reinterpret_cast<const int32_t *>(mask + seq_mask_off + nb31 * t);
            hi_max = max(hi_max, r[1]);
        }
        kv_end = min(n_kv, max(hi_max, 0));
    }

    const int total_tiles    = (kv_end + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
    const int base_tiles     = total_tiles / n_splits;
    const int extra_tiles    = total_tiles % n_splits;
    const int split_tile_start = split_id * base_tiles + min(split_id, extra_tiles);
    const int split_tiles    = base_tiles + (split_id < extra_tiles ? 1 : 0);
    const int split_start    = split_tile_start * kGroupedVerifyBlockN;
    const int split_end      = min(kv_end, split_start + split_tiles * kGroupedVerifyBlockN);

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
    // When the head count does not fill the CTA the padding slots of the last group
    // hold zero instead of reading past the Q heads; their output is skipped below.
    for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim; idx += kGroupedVerifyThreads) {
        const int row        = idx / kGroupedVerifyHeadDim;
        const int d          = idx % kGroupedVerifyHeadDim;
        const int token_idx  = row / kHeadsPerCta;
        const int local_head = row % kHeadsPerCta;
        const int head = kv_head * HEADS + head_group * kHeadsPerCta + local_head;
        __half * dst = shared_q + row * kGroupedVerifyQStride + d;
        if constexpr (HEADS % kHeadsPerCta != 0) {
            if (head_group * kHeadsPerCta + local_head >= HEADS) {
                *dst = __float2half_rn(0.0f);
                continue;
            }
        }
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

    // Tile i+1 is fetched into registers while tile i is computed. The staging rows of
    // this warp are private, so a block wide fence is only needed around the panel.
    // Two q4_0 tiles fit the register array: while tile t is computed, tile t+1 sits in
    // the other slot and the load of t+2 goes into the slot just consumed by t. The
    // other types keep one tile in flight.
    constexpr bool kPrefetchTwoTiles = type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0;
    constexpr int  kPrefetchAhead    = kPrefetchTwoTiles ? 2 : 1;
    GroupedVerifyKVRegs<type_K> k_regs;
    GroupedVerifyKVRegs<type_V> v_regs;
    if (split_start < split_end) {
        flash_attn_sm70_grouped_prefetch_kv<type_K>(
            k_regs, K, nb11, nb12, nb13, seq, kv_head, split_start, n_kv, 0);
        flash_attn_sm70_grouped_prefetch_kv<type_V>(
            v_regs, V, nb21, nb22, nb23, seq, kv_head, split_start, n_kv, 0);
        if constexpr (kPrefetchTwoTiles) {
            if (split_start + kGroupedVerifyBlockN < split_end) {
                flash_attn_sm70_grouped_prefetch_kv<type_K>(
                    k_regs, K, nb11, nb12, nb13, seq, kv_head, split_start + kGroupedVerifyBlockN, n_kv, 1);
                flash_attn_sm70_grouped_prefetch_kv<type_V>(
                    v_regs, V, nb21, nb22, nb23, seq, kv_head, split_start + kGroupedVerifyBlockN, n_kv, 1);
            }
        }
    }

    for (int tile_start = split_start; tile_start < split_end; tile_start += kGroupedVerifyBlockN) {
        const bool has_prefetch = tile_start + kPrefetchAhead * kGroupedVerifyBlockN < split_end;
        const int  tile_slot    = kPrefetchTwoTiles ? ((tile_start - split_start) / kGroupedVerifyBlockN) & 1 : 0;

        if constexpr (type_K == GGML_TYPE_F16) {
            __syncthreads(); // the previous P x V must be done reading the panel
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs, tile_slot);
            __syncthreads();
        } else {
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs, tile_slot);
            __syncthreads(); // the previous P x V must be done reading the panel
            flash_attn_sm70_grouped_dequant_kv<type_K>(shared_kv, kv_stage);
            __syncthreads();
        }

        if (has_prefetch) {
            flash_attn_sm70_grouped_prefetch_kv<type_K>(
                k_regs, K, nb11, nb12, nb13, seq, kv_head, tile_start + kPrefetchAhead * kGroupedVerifyBlockN, n_kv, tile_slot);
        }
        if constexpr (type_V != GGML_TYPE_F16) {
            // The staging buffer is free again once the K dequantize is done. For a
            // quantized V the raw tile waits there until the K panel dies after QK.
            // The next V tile starts loading right away so its latency hides behind QK.
            flash_attn_sm70_grouped_store_kv<type_V>(shared_kv, kv_stage, v_regs, tile_slot);
            if (has_prefetch) {
                flash_attn_sm70_grouped_prefetch_kv<type_V>(
                    v_regs, V, nb21, nb22, nb23, seq, kv_head, tile_start + kPrefetchAhead * kGroupedVerifyBlockN, n_kv, tile_slot);
            }
        }

        // The mask does not depend on the panel, so load it before QK to hide the
        // global latency behind the tensor core work.
        GroupedVerifyMaskPair row_mask[kGroupedVerifyRowsPerWarp];
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            const int row = warp_id + i * kGroupedVerifyWarps;
            row_mask[i] = grouped_verify_load_mask_pair(
                mask, seq_mask_off, nb31, row / kHeadsPerCta, n_q,
                tile_start + kGroupedVerifyColsPerLane * lane_id, n_kv,
                mask_is_range != 0);
        }

        grouped_verify_qk(shared_q, shared_kv, shared_scores, scale);
        __syncthreads(); // scores are ready and QK is done with the panel

        if constexpr (type_V == GGML_TYPE_F16) {
            // The panel is free now; fp16 V goes straight into it. The next V tile
            // starts loading right away so its latency hides behind softmax and P x V.
            flash_attn_sm70_grouped_store_kv<type_V>(shared_kv, kv_stage, v_regs, tile_slot);
            if (has_prefetch) {
                flash_attn_sm70_grouped_prefetch_kv<type_V>(
                    v_regs, V, nb21, nb22, nb23, seq, kv_head, tile_start + kPrefetchAhead * kGroupedVerifyBlockN, n_kv, tile_slot);
            }
        }

        grouped_verify_softmax_tile(
            shared_scores, shared_probs, smem.row_max, smem.row_sum, smem.row_scale, row_mask, warp_id, lane_id);
        if constexpr (type_V != GGML_TYPE_F16) {
            flash_attn_sm70_grouped_dequant_kv<type_V>(shared_kv, kv_stage);
        }
        // row_scale is written by every warp and read by every warp below, so the
        // rescale of the accumulators must wait for this barrier.
        __syncthreads(); // probs, row_scale and the V panel are ready for P x V

#pragma unroll
        for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
            grouped_verify_scale_output_fragment(output_fragments[fragment_idx], smem.row_scale, fragment_idx * 16);
        }

        // One warp per V column tile, all three M tiles share the same B fragment:
        // 4 B loads + 12 A loads per warp instead of 4 + 4 + 4 loads per M tile.
#pragma unroll
        for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
            ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_b, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                value_fragment;
            ggml_sm70_wmma::load_matrix_sync(
                value_fragment,
                shared_kv + k_offset * kGroupedVerifyKVStride + warp_id * 16,
                kGroupedVerifyKVStride);
#pragma unroll
            for (int m_tile = 0; m_tile < kGroupedVerifyOutputTilesPerWarp; ++m_tile) {
                ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_a, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                    probability_fragment;
                ggml_sm70_wmma::load_matrix_sync(
                    probability_fragment,
                    shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
                    kGroupedVerifyProbStride);
                ggml_sm70_wmma::mma_sync(
                    output_fragments[m_tile], probability_fragment, value_fragment, output_fragments[m_tile]);
            }
        }
    }

    // The compute buffers are dead. Reuse their storage for the dense FP32 output.
    __syncthreads();
    float* shared_output = smem.storage.output;
#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        ggml_sm70_wmma::store_matrix_sync(
            shared_output + fragment_idx * 16 * kGroupedVerifyHeadDim + warp_id * 16,
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
        if constexpr (HEADS % kHeadsPerCta != 0) {
            if (head_group * kHeadsPerCta + local_head >= HEADS) {
                continue;
            }
        }
        const int head = kv_head * HEADS + head_group * kHeadsPerCta + local_head;
        const int64_t j = (int64_t(seq) * n_q + token_idx) * n_heads + head;
        // Unnormalized numerator, normalized by flash_attn_combine_results.
        dst_partial[(j * n_splits + split_id) * kGroupedVerifyHeadDim + d] = shared_output[idx];
    }
    if (tid < kGroupedVerifyRows) {
        const int token_idx  = tid / kHeadsPerCta;
        const int local_head = tid % kHeadsPerCta;
        bool head_filled = true;
        if constexpr (HEADS % kHeadsPerCta != 0) {
            head_filled = head_group * kHeadsPerCta + local_head < HEADS;
        }
        if (token_idx < n_q && head_filled) {
            const int head = kv_head * HEADS + head_group * kHeadsPerCta + local_head;
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
