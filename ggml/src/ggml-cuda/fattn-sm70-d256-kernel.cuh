// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// fattn-sm70-d256-kernel.cuh
//
// SM70 (Volta) D256 FlashAttention "Split-D N32" kernel.
//
// ORIGIN: core adapted from the 1CatAI D256 Split-D kernel
//   repo : 1CatAI/1Cat-vLLM
//   path : csrc/attention/sm70_v37/tail.cu   (HEAD fcf59f8e9)
//   (c) 1CatAI, BSD-3-Clause compatible.
// The support layer (Mask, sm70_* helpers, gemm) is vendored from
//   zhinianqin/flash-attention-v100 @ c2eda5e (BSD-3-Clause);
//   see sm70-vendor/LICENSE-flash-attention.
// CuTe/CUTLASS: see sm70-vendor/ (LICENSE-cute-cutlass).
//
// This file contains ONLY the device-side kernel + traits + helpers. The
// upstream torch host wrapper (onecat_v37_dense_state_float_raw) is NOT
// included -- the llama.cpp launcher in fattn-sm70-d256.cu drives the kernel
// directly.
//
// Only the staged-f16 dense path is kept: no paged KV, no in-kernel q4/q8
// dequant, no per-warp register P. SplitKV3 remains behind the template flag;
// the llama.cpp launcher does not select it yet.
//
// The verified compute core is UNMODIFIED: SmemLayout (pitch-68 K / TT swizzled
// V), HMMA.884 QK+PV atoms, K/V double-buffered pipeline, online softmax with
// row_scale_exchange and the __launch_bounds__(256,1) smem budget
// (kSmemBytes==45568 -> 2 CTA/SM). Deviations from upstream, all required for
// llama.cpp semantics:
//  * the causal Mask is replaced by a range mask (col >= kv_len -> -inf): the
//    llama.cpp mask carries the causal/padding semantics and test masks are
//    arbitrary;
//  * one explicit mask add after the range mask: an f16 mask is added, an I32
//    range mask sets every column outside [lo, hi) to -inf;
//  * a mask pre-scan (sm70_d256_mask_bounds_kernel) gives, per (batch, q
//    block), the first non-zero and the last above -inf column: the explicit
//    mask add skips the all-zero prefix and the KV walk stops at the last
//    block that can contribute. Without mask_bounds every block in
//    [0, ceil(kv_len/kBlockN)) is visited (no causal bound);
//  * a 0/0 guard for fully masked rows in the dense output and the merge.
// ============================================================================
#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

#include "sm70-vendor/cute/tensor.hpp"
#include "sm70-vendor/cutlass/numeric_types.h"

#include "sm70-vendor/flash/namespace_config.h"
#include "sm70-vendor/flash/kernel_traits.h"
#include "sm70-vendor/flash/utils.h"
#include "sm70-vendor/flash/softmax.h"
#include "sm70-vendor/flash/mask.h"
#include "sm70-vendor/flash/philox.cuh"

namespace FLASH_NAMESPACE {

using namespace cute;

// DChunk: 64 = stock Path A Split-D (HMMA tiles over 64-wide K/V chunks).
template <int DChunk = 64>
struct Sm70D256SplitDTraitsT {
    using Element = cutlass::half_t;
    using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
    using PvMmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TT>;

    static constexpr int kHeadDim = 256;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 32;
    static constexpr int kDChunk = DChunk;
    static constexpr int kDChunks = kHeadDim / kDChunk;
    static constexpr int kOwnedDChunks = kDChunks / 2;
    static constexpr int kNThreads = 256;
    static constexpr int kMmaThreads = 32;
    static constexpr int kWarpsPerGroup = 2;
    static constexpr int kMmaGroups =
        kNThreads / (kWarpsPerGroup * kMmaThreads);
    static constexpr int kGroupRows = kBlockM / kMmaGroups;
    static constexpr int kQkWarpRows = kGroupRows / kWarpsPerGroup;
    static constexpr int kQkRowsPerThread = kQkWarpRows / 4;
    static constexpr int kOutputRowsPerThread = kGroupRows / 4;

    // Each warp in a pair owns eight distinct Q rows for QK, then the pair
    // shares the resulting P tile and each warp owns D/2 for PV. This keeps
    // the standard FA2 N32 online-softmax order without duplicating QK work.
    using QkTiledMma = TiledMMA<
        MmaAtom,
        Layout<Shape<_1, _4, _1>>,
        Tile<Int<kQkWarpRows>, Int<kBlockN / 4>, _4>>;
    using PvTiledMma = TiledMMA<
        PvMmaAtom,
        Layout<Shape<_1, _4, _1>>,
        Tile<Int<kGroupRows>, Int<kDChunk / 4>, _4>>;
    static_assert(decltype(size(QkTiledMma{}))::value == kMmaThreads);
    static_assert(decltype(size(PvTiledMma{}))::value == kMmaThreads);

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kDChunk>>{}));
    // Volta HMMA.884 assigns one warp to four 8-thread quadpairs and services
    // the 64-bit operand loads as two half-warps. Pitch 68 advances each row
    // by one bank pair; the extra 16-half phase every 16 rows folds row bit 4
    // into bank-pair bit 2 so every half-warp covers all 16 bank pairs once.
    using SmemLayoutK = Layout<
        Shape<Shape<Int<16>, Int<2>>, Int<kDChunk>>,
        Stride<Stride<Int<kDChunk + 4>, Int<16 * (kDChunk + 4) + 16>>,
               _1>>;
    // TT PV consumes V as KxD. Row bit 1 is folded into D-address bit 2;
    // the producer applies the inverse 64-bit-half swap before STS.128.
    using SmemLayoutV = Layout<
        Shape<Shape<_2, Int<kBlockN / 2>>,
              Shape<_32, Int<kDChunk / 32>>>,
        Stride<Stride<_32, Int<4 * kBlockN>>,
               Stride<_1, _64>>>;
    using SmemLayoutP = Layout<
        Shape<Int<kBlockM>, Int<kBlockN>>,
        Stride<Int<kBlockN>, _1>>;

    using SmemCopyAtom =
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomTransposed = SmemCopyAtom;

    static constexpr int kGmemElemsPerLoad = 8;
    static constexpr int kGmemThreadsPerRow = kDChunk / kGmemElemsPerLoad;
    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>,
              Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<SM70_LDG_GLOBAL_CG_128b, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{}));

    static constexpr int kGmemKElemsPerLoad = 4;
    static constexpr int kGmemKThreadsPerRow =
        kDChunk / kGmemKElemsPerLoad;
    using GmemKLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemKThreadsPerRow>,
              Int<kGmemKThreadsPerRow>>,
        Stride<Int<kGmemKThreadsPerRow>, _1>>;
    using GmemKTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint64_t>, Element>{},
        GmemKLayoutAtom{},
        Layout<Shape<_1, _4>>{}));

    static constexpr int kQElements = size(SmemLayoutQ{});
    static constexpr int kKVElements = size(SmemLayoutKV{});
    static_assert(size(SmemLayoutV{}) == kKVElements);
    static constexpr int kPElements = size(SmemLayoutP{});
    static constexpr int kExchangeRows = kMmaGroups * kGroupRows;
    static constexpr int kTensorSmemBytes =
        (kQElements + 2 * kKVElements + kPElements) * sizeof(Element);
    static constexpr int kExchangeBytes =
        2 * kExchangeRows * sizeof(float);
    static constexpr int kSmemBytes = kTensorSmemBytes + kExchangeBytes;
    static_assert(kSmemBytes <= 65536);
};

using Sm70D256SplitDTraits = Sm70D256SplitDTraitsT<64>;
static_assert(Sm70D256SplitDTraits::kSmemBytes == 45568);

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
__device__ __forceinline__ void copy_even_tile(
    TiledCopy tiled_copy, const SrcTensor &src, DstTensor &dst) {
    static_assert(decltype(rank(src))::value == 3);
    static_assert(decltype(rank(dst))::value == 3);
#pragma unroll
    for (int m = 0; m < size<1>(src); ++m) {
#pragma unroll
        for (int k = 0; k < size<2>(src); ++k) {
            cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
        }
    }
}

template <typename RegTensor, typename SmemTensor, typename CoordTensor>
__device__ __forceinline__ void store_v_fragment_128_swizzled(
    const RegTensor &source,
    SmemTensor &destination,
    const CoordTensor &coordinates) {
    static_assert(decltype(size<0>(source))::value == 8);
    static_assert(decltype(size<0>(destination))::value == 8);
    static_assert(decltype(size<1>(source))::value
                  == decltype(size<1>(destination))::value);
    static_assert(decltype(size<2>(source))::value
                  == decltype(size<2>(destination))::value);
#pragma unroll
    for (int k = 0; k < size<2>(source); ++k) {
#pragma unroll
        for (int m = 0; m < size<1>(source); ++m) {
            auto words = recast<uint32_t const>(source(_, m, k));
            const uint32_t address = static_cast<uint32_t>(
                __cvta_generic_to_shared(&destination(0, m, k)));
            const int row = get<0>(coordinates(0, m, k));
            if (row & 2) {
                asm volatile(
                    "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(address), "r"(words(2)), "r"(words(3)),
                       "r"(words(0)), "r"(words(1)));
            } else {
                asm volatile(
                    "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(address), "r"(words(0)), "r"(words(1)),
                       "r"(words(2)), "r"(words(3)));
            }
        }
    }
}

template <typename SmemTensor, typename TensorB>
__device__ __forceinline__ void load_v_fragment_tt(
    const SmemTensor &sV,
    TensorB &b_words,
    int phase,
    int lane) {
    auto *v = sV.data().get();
    const int d_lane = ((lane & 0x0c) << 1) | ((lane & 0x10) >> 2);
    const int k = phase * 4 + (lane & 0x03);
    const int offset = (d_lane ^ ((k & 0x02) << 1))
        | ((k & 0x01) << 5) | ((k & 0x1e) << 6);
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(v + offset));
    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
    asm volatile(
        "ld.shared.v2.u32 {%0, %1}, [%4];\n"
        "ld.shared.v2.u32 {%2, %3}, [%4+128];\n"
        : "=r"(word0), "=r"(word1), "=r"(word2), "=r"(word3)
        : "r"(address));
    b_words(0, 0) = word0;
    b_words(1, 0) = word1;
    b_words(0, 1) = word2;
    b_words(1, 1) = word3;
}

template <int kPhase, typename TensorO, typename TensorP,
          typename SmemTensor, typename TiledMma, typename TensorB,
          typename TensorBWords, typename TensorBNext,
          typename TensorBNextWords>
__device__ __forceinline__ void splitd_pv_gemm_tt_phase(
    TensorO &acc_o,
    const TensorP &tPrP,
    const SmemTensor &sV,
    TensorB &current_b,
    TensorBWords &current_b_words,
    TensorBNext &next_b,
    TensorBNextWords &next_b_words,
    TiledMma tiled_mma,
    int lane) {
    constexpr int kPhases = decltype(size<2>(tPrP))::value;
    static_assert(kPhase < kPhases);
    if constexpr (kPhase + 1 < kPhases) {
        load_v_fragment_tt(sV, next_b_words, kPhase + 1, lane);
    }
    cute::gemm(tiled_mma, tPrP(_, _, kPhase), current_b, acc_o);
    if constexpr (kPhase + 1 < kPhases) {
        splitd_pv_gemm_tt_phase<kPhase + 1>(
            acc_o, tPrP, sV, next_b, next_b_words,
            current_b, current_b_words, tiled_mma, lane);
    }
}

template <typename TensorO, typename TensorP, typename SmemTensor,
          typename TiledMma>
__device__ __forceinline__ void splitd_pv_gemm_tt(
    TensorO &acc_o,
    const TensorP &tPrP,
    const SmemTensor &sV,
    TiledMma tiled_mma,
    int lane) {
    using Element = typename Sm70D256SplitDTraitsT<64>::Element;
    using BLayout = Layout<Shape<_4, _2>, Stride<_1, _4>>;
    auto b0 = make_tensor<Element>(BLayout{});
    auto b1 = make_tensor<Element>(BLayout{});
    auto b0_words = recast<uint32_t>(b0);
    auto b1_words = recast<uint32_t>(b1);
    // PV K-steps along P's N (= BlockN) are 4-wide; count is BlockN/4.
    static_assert(decltype(size<2>(tPrP))::value
                  == Sm70D256SplitDTraitsT<64>::kBlockN / 4);
    static_assert(decltype(size(b0))::value == 8);
    static_assert(decltype(size<0>(b0_words))::value == 2);
    static_assert(decltype(size<1>(b0_words))::value == 2);
    load_v_fragment_tt(sV, b0_words, 0, lane);
    splitd_pv_gemm_tt_phase<0>(
        acc_o, tPrP, sV, b0, b0_words, b1, b1_words,
        tiled_mma, lane);
}

// kRows: number of rows in this warp's o accumulator (kGroupRows=16 for the
// stock pair-shared path). kPerWarp: when true the warp's o accumulator covers
// only its own kRows rows; when false it covers the whole kGroupRows group (no
// offset). kOChunks/kOElements are deduced from the o_storage array shape.
template <int kRows, bool kPerWarp, typename TensorScores, typename OLayout,
          int kOChunks, int kOElements>
__device__ __forceinline__ void splitd_n32_online_softmax(
    TensorScores &acc_s,
    float (&o_storage)[kOChunks][kOElements],
    OLayout o_layout,
    float (&row_max)[Sm70D256SplitDTraitsT<64>::kQkRowsPerThread],
    float (&row_sum)[Sm70D256SplitDTraitsT<64>::kQkRowsPerThread],
    float *row_scale_exchange,
    int mma_group,
    int n_warp,
    int lane,
    float softmax_scale_log2,
    bool first_tile) {
    using Traits = Sm70D256SplitDTraits;
    auto scores = make_tensor(
        acc_s.data(),
        FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(
        decltype(size<0>(scores))::value == Traits::kQkRowsPerThread);

    float work[Traits::kQkRowsPerThread];
    auto work_tensor = make_tensor(
        make_rmem_ptr(&work[0]), Shape<Int<Traits::kQkRowsPerThread>>{});
    if (first_tile) {
        FLASH_NAMESPACE::sm70_reduce_max<true>(scores, work_tensor);
    } else {
#pragma unroll
        for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
            work[slot] = row_max[slot];
        }
        FLASH_NAMESPACE::sm70_reduce_max<false>(scores, work_tensor);
    }

    float scores_max[Traits::kQkRowsPerThread];
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        const float next_max = work[slot];
        const float safe_max = next_max == -INFINITY ? 0.0f : next_max;
        work[slot] = first_tile
            ? 1.0f
            : exp2f((row_max[slot] - safe_max) * softmax_scale_log2);
        row_max[slot] = next_max;
        scores_max[slot] = safe_max;
        if (!first_tile) {
            row_sum[slot] *= work[slot];
        }
    }

    if ((lane & 0x0e) == 0) {
#pragma unroll
        for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
            const int row = FLASH_NAMESPACE::sm70_row_slot<
                Traits::kQkWarpRows>(slot, lane);
            row_scale_exchange[
                mma_group * Traits::kGroupRows
                + n_warp * Traits::kQkWarpRows + row] = work[slot];
        }
    }
    __syncthreads();

    if (!first_tile) {
        const int row_scale_base = mma_group * Traits::kGroupRows
            + (kPerWarp ? n_warp * kRows : 0);
#pragma unroll
        for (int d = 0; d < kOChunks; ++d) {
            auto acc_o = make_tensor(make_rmem_ptr(&o_storage[d][0]), o_layout);
            auto acc_o_rc = make_tensor(
                acc_o.data(),
                FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int row = 0; row < kRows / 4; ++row) {
                const int logical_row = FLASH_NAMESPACE::sm70_row_slot<
                    kRows>(row, lane);
                const float row_scale = row_scale_exchange[
                    row_scale_base + logical_row];
#pragma unroll
                for (int col = 0; col < size<1>(acc_o_rc); ++col) {
                    acc_o_rc(row, col) *= row_scale;
                }
            }
        }
    }

    auto scores_max_tensor = make_tensor(
        make_rmem_ptr(&scores_max[0]),
        Shape<Int<Traits::kQkRowsPerThread>>{});
    FLASH_NAMESPACE::sm70_scale_apply_exp2(
        scores, scores_max_tensor, softmax_scale_log2);

    FLASH_NAMESPACE::sm70_reduce_sum<true>(scores, work_tensor);
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        row_sum[slot] += work[slot];
    }
}

// Additive f16 mask applied before the online softmax. Row and column indices
// follow the same lane layout as Mask::apply_mask in mask.h. Entries outside
// [0, q_len_real) x [0, kv_len) read as zero: Q pad rows and the tail of the
// last KV tile. llama.cpp adds the mask to the scaled logits (QK*scale + mask)
// while this kernel folds scale into exp2, so the launcher passes
// mask_scale = 1/scale and the mask enters the raw-QK domain as mask/scale.
template <typename TensorScores>
__device__ __forceinline__ void splitd_add_explicit_mask(
    TensorScores &acc_s,
    const __half *__restrict__ mask,
    const int64_t mask_row_stride,
    const int64_t mask_batch_offset,
    const int q_len_real,
    const int kv_len,
    const int row_base,
    const int col_base,
    const float mask_scale) {
    auto scores = make_tensor(
        acc_s.data(),
        FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0, 0>(scores))::value == 2,
                  "unexpected SM70 QK row-inner layout");
    static_assert(decltype(size<0, 1>(scores))::value == 1,
                  "unexpected SM70 QK row-outer layout");
    static_assert(decltype(size<1, 0>(scores))::value == 2,
                  "unexpected SM70 QK col-inner layout");
    static_assert(decltype(size<1, 1>(scores))::value == 2,
                  "unexpected SM70 QK col-middle layout");
    static_assert(decltype(size<1, 2>(scores))::value == 1,
                  "unexpected SM70 QK col-outer layout");

    const int lane_id = threadIdx.x % 32;
    const int lane_row_base = (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
    const int lane_col_base =
        (((lane_id >> 1) & 0x1) << 1) |
        (((lane_id >> 2) & 0x1) << 3) |
        (((lane_id >> 3) & 0x1) << 4);
    const __half *mask_batch = mask + mask_batch_offset;
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(scores); ++mi) {
#pragma unroll
        for (int i = 0; i < size<0, 0>(scores); ++i) {
            const int row = FLASH_NAMESPACE::sm70_mask_row_idx<8>(
                lane_row_base, row_base, i, mi);
            const bool row_ok = row < q_len_real;
#pragma unroll
            for (int n = 0; n < size<1, 2>(scores); ++n) {
#pragma unroll
                for (int nj = 0; nj < size<1, 1>(scores); ++nj) {
#pragma unroll
                    for (int j = 0; j < size<1, 0>(scores); ++j) {
                        const int col = FLASH_NAMESPACE::sm70_mask_col_idx<32>(
                            lane_col_base, col_base, j, nj, n);
                        auto coord = make_coord(
                            make_coord(i, mi), make_coord(j, nj, n));
                        if (row_ok && col < kv_len) {
                            scores(coord) += mask_scale
                                * __half2float(mask_batch[
                                      (int64_t) row * mask_row_stride + col]);
                        }
                    }
                }
            }
        }
    }
}

// Range mask: the row's [lo, hi) KV window stays visible, every other column
// is -inf. Row and column traversal are identical to splitd_add_explicit_mask;
// the pair is read once per row (8-byte load) before the column loop.
template <typename TensorScores>
__device__ __forceinline__ void splitd_apply_range_mask(
    TensorScores &acc_s,
    const int2 *__restrict__ range,
    const int64_t range_row_stride,
    const int64_t range_batch_offset,
    const int q_len_real,
    const int kv_len,
    const int row_base,
    const int col_base) {
    auto scores = make_tensor(
        acc_s.data(),
        FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0, 0>(scores))::value == 2,
                  "unexpected SM70 QK row-inner layout");
    static_assert(decltype(size<0, 1>(scores))::value == 1,
                  "unexpected SM70 QK row-outer layout");
    static_assert(decltype(size<1, 0>(scores))::value == 2,
                  "unexpected SM70 QK col-inner layout");
    static_assert(decltype(size<1, 1>(scores))::value == 2,
                  "unexpected SM70 QK col-middle layout");
    static_assert(decltype(size<1, 2>(scores))::value == 1,
                  "unexpected SM70 QK col-outer layout");

    const int lane_id = threadIdx.x % 32;
    const int lane_row_base = (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
    const int lane_col_base =
        (((lane_id >> 1) & 0x1) << 1) |
        (((lane_id >> 2) & 0x1) << 3) |
        (((lane_id >> 3) & 0x1) << 4);
    const int2 *range_batch = range + range_batch_offset;
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(scores); ++mi) {
#pragma unroll
        for (int i = 0; i < size<0, 0>(scores); ++i) {
            const int row = FLASH_NAMESPACE::sm70_mask_row_idx<8>(
                lane_row_base, row_base, i, mi);
            const bool row_ok = row < q_len_real;
            int lo = 0;
            int hi = 0;
            if (row_ok) {
                const int2 r = range_batch[(int64_t) row * range_row_stride];
                lo = r.x < 0 ? 0 : (r.x > kv_len ? kv_len : r.x);
                hi = r.y < 0 ? 0 : (r.y > kv_len ? kv_len : r.y);
            }
#pragma unroll
            for (int n = 0; n < size<1, 2>(scores); ++n) {
#pragma unroll
                for (int nj = 0; nj < size<1, 1>(scores); ++nj) {
#pragma unroll
                    for (int j = 0; j < size<1, 0>(scores); ++j) {
                        const int col = FLASH_NAMESPACE::sm70_mask_col_idx<32>(
                            lane_col_base, col_base, j, nj, n);
                        auto coord = make_coord(
                            make_coord(i, mi), make_coord(j, nj, n));
                        if (row_ok && col < kv_len && (col < lo || col >= hi)) {
                            scores(coord) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

// ElementOut (default = Element): output element type. The llama.cpp launcher
// instantiates ElementOut=float so the attention output is written as f32
// directly (the f16 output staging was a per-layer rounding source). Q/K/V
// must stay f16 (HMMA operand constraint).
//
// SplitKV3: 3-way KV split for long-prefix prefill. Not selected by the
// llama.cpp launcher yet; the dense and merge code paths are kept.
//
// Partial: windowed mode for a fixed-size KV mirror. Like SplitKV3 the kernel
// writes only the per-row raw max/sum and the unnormalized numerator, but at
// split 0 with gridDim.y = batch and a plain [row][D] partial buffer
// (row = (batch*query_len + query_row)*heads_q + head_q). The caller launches
// one kernel per KV block window and merges the windows afterwards with
// sm70_d256_window_merge_kernel + sm70_d256_window_finalize_kernel. A row that
// sees no KV block in the window writes max -inf, sum 0 and O 0.
template <typename TraitsT, typename Element, typename ElementOut = Element,
          bool SplitKV3 = false, bool Partial = false>
__global__ __launch_bounds__(256, 1)
void sm70_d256_splitd_dense_kernel(
    const Element *__restrict__ q,
    const Element *__restrict__ k,
    const Element *__restrict__ v,
    ElementOut *__restrict__ out,
    const __half *__restrict__ mask,
    int q_batch_stride,
    int q_row_stride,
    int q_head_stride,
    int k_outer_stride,
    int k_row_stride,
    int k_head_stride,
    int v_outer_stride,
    int v_row_stride,
    int v_head_stride,
    int64_t mask_row_stride,
    int64_t mask_batch_stride,
    bool mask_is_range,
    const int2 *__restrict__ mask_bounds,
    int query_len,
    int kv_len,
    int heads_q,
    int heads_kv,
    int kv_offset,
    float softmax_scale_log2,
    float mask_scale,
    float *__restrict__ partial_out,
    float *__restrict__ partial_max,
    float *__restrict__ partial_sum,
    int win_block_lo,
    int win_block_hi) {
    static_assert(!(SplitKV3 && Partial), "SplitKV3 and Partial are mutually exclusive");
    using Traits = TraitsT;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kDChunk = Traits::kDChunk;

    const int tid = threadIdx.x;
    const int warp = tid / Traits::kMmaThreads;
    const int mma_group = warp / Traits::kWarpsPerGroup;
    const int lane = tid % Traits::kMmaThreads;
    const int m_block = blockIdx.x;
    const int split = SplitKV3 ? blockIdx.y % 3 : 0;
    const int batch = SplitKV3 ? blockIdx.y / 3 : blockIdx.y;
    const int head_q = blockIdx.z;
    const int head_kv = head_q / (heads_q / heads_kv);
    const int query_row_base = m_block * kBlockM;

    extern __shared__ __align__(128) Element smem[];
    Element *q_smem_ptr = smem;
    Element *kv_smem_ptr = q_smem_ptr + Traits::kQElements;
    auto sQ = make_tensor(
        make_smem_ptr(q_smem_ptr), typename Traits::SmemLayoutQ{});

    typename Traits::GmemTiledCopy gmem_copy;
    auto gmem_thread = gmem_copy.get_thread_slice(tid);
    typename Traits::GmemTiledCopy gmem_v_copy;
    auto gmem_v_thread = gmem_v_copy.get_thread_slice(tid);
    typename Traits::GmemKTiledCopy gmem_k_copy;
    auto gmem_k_thread = gmem_k_copy.get_thread_slice(tid);
    {
        const int64_t q_batch_offset = static_cast<int64_t>(batch)
            * q_batch_stride;
        auto mQ = make_tensor(
            make_gmem_ptr(q + q_batch_offset + head_q * q_head_stride),
            make_shape(query_len, Int<Traits::kHeadDim>{}),
            make_stride(q_row_stride, _1{}));
        auto gQ = local_tile(
            mQ,
            Shape<Int<kBlockM>, Int<Traits::kHeadDim>>{},
            make_coord(m_block, 0));
        auto tQgQ = gmem_thread.partition_S(gQ);
        auto tQsQ = gmem_thread.partition_D(sQ);
        copy_even_tile(gmem_copy, tQgQ, tQsQ);
    }
    __syncthreads();

    typename Traits::QkTiledMma qk_tiled_mma;
    auto qk_mma_thread = qk_tiled_mma.get_thread_slice(lane);
    typename Traits::PvTiledMma pv_tiled_mma;
    auto pv_mma_thread = pv_tiled_mma.get_thread_slice(lane);

    using OFragment = decltype(partition_fragment_C(
        pv_tiled_mma,
        Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{}));
    constexpr int kOElements = decltype(size(OFragment{}))::value;
    using OLayout = typename OFragment::layout_type;
    float o_storage[Traits::kOwnedDChunks][kOElements];
#pragma unroll
    for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
#pragma unroll
        for (int i = 0; i < kOElements; ++i) {
            o_storage[d][i] = 0.0f;
        }
    }

    float row_max[Traits::kQkRowsPerThread];
    float row_sum[Traits::kQkRowsPerThread];
#pragma unroll
    for (int row = 0; row < Traits::kQkRowsPerThread; ++row) {
        // SplitKV3: a row's causal window may not overlap this segment at all
        // (fully-masked row) - keep row_max finite so exp2 never sees
        // (-inf) - (-inf) = NaN. -1e30 behaves as -inf under exp2 (flushes to
        // zero) and the merge kernel's scale stays 0. The dense path keeps
        // -INFINITY (every row always sees causal block 0 there).
        row_max[row] = SplitKV3 ? -1.0e30f : -INFINITY;
        row_sum[row] = 0.0f;
    }

    // Without mask_bounds every KV block [0, ceil(kv_len/kBlockN)) is
    // processed: llama.cpp masks are arbitrary (tests) and carry the
    // causal/padding semantics, so the kernel must not apply a causal bound or
    // a causal mask of its own. mask_bounds (sm70_d256_mask_bounds_kernel)
    // describes the mask shape instead: first_nz = kv_len - b.x is the first
    // column with a non-zero entry and b.y - 1 the last column above -inf.
    // Blocks past b.y-1 cannot contribute, and the explicit mask add is a
    // no-op for blocks entirely below first_nz.
    const int visible_n_blocks = cute::ceil_div(kv_len, kBlockN);
    int mask_z_end = 0;
    int n_visited_blocks = visible_n_blocks;
    if constexpr (!SplitKV3) {
        if (mask_bounds != nullptr) {
            const int2 b = mask_bounds[(int64_t) batch * gridDim.x + blockIdx.x];
            mask_z_end = (kv_len - b.x) / kBlockN;
            n_visited_blocks = cute::ceil_div(b.y, kBlockN);
            if (n_visited_blocks > visible_n_blocks) {
                n_visited_blocks = visible_n_blocks;
            }
        }
    }
    int n_block_min = 0;
    int n_block_max = n_visited_blocks - 1;
    if constexpr (SplitKV3) {
        n_block_min = visible_n_blocks * split / 3;
        n_block_max = visible_n_blocks * (split + 1) / 3 - 1;
    }
    // Restrict the walk to the caller's KV block window [win_block_lo, win_block_hi).
    // The windowed path passes the window bounds; every other call passes 0 and
    // INT_MAX, for which both clamps are no-ops and the walk is bit-identical to
    // the unwindowed one. The K/V pointers already map the window's rows, so no
    // index below changes.
    n_block_min = n_block_min > win_block_lo ? n_block_min : win_block_lo;
    n_block_max = n_block_max < win_block_hi - 1 ? n_block_max : win_block_hi - 1;
    // SplitKV3 guard: an empty split segment (n_block_max < n_block_min) must
    // still load a valid first tile to keep the gmem addresses in range; use
    // n_block_min (always < visible_n_blocks) for that degenerate case.
    const int n_block_first = n_block_max >= n_block_min ? n_block_max : n_block_min;

    const int64_t k_batch_offset = static_cast<int64_t>(batch)
        * k_outer_stride;
    auto mK = make_tensor(
        make_gmem_ptr(k + k_batch_offset + head_kv * k_head_stride),
        make_shape(kv_len, Int<Traits::kHeadDim>{}),
        make_stride(k_row_stride, _1{}));
    auto sK = make_tensor(
        make_smem_ptr(kv_smem_ptr), typename Traits::SmemLayoutK{});
    auto tKsK = gmem_k_thread.partition_D(sK);
    auto tKrKNext = make_fragment_like(tKsK);
    {
        auto gKFirst = local_tile(
            mK,
            Shape<Int<kBlockN>, Int<kDChunk>>{},
            make_coord(n_block_first, 0));
        auto tKgKFirst = gmem_k_thread.partition_S(gKFirst);
        copy_even_tile(gmem_k_copy, tKgKFirst, tKsK);
    }
    __syncthreads();

    for (int n_block = n_block_max; n_block >= n_block_min; --n_block) {
        const int n_warp = warp & 1;
        const int group_row_base = mma_group * Traits::kGroupRows;
        const int qk_row_base = group_row_base
            + n_warp * Traits::kQkWarpRows;
        auto acc_s = partition_fragment_C(
            qk_tiled_mma,
            Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
        clear(acc_s);

#pragma unroll
        for (int d_chunk = 0; d_chunk < Traits::kDChunks; ++d_chunk) {
            if (d_chunk + 1 < Traits::kDChunks) {
                auto gKNext = local_tile(
                    mK,
                    Shape<Int<kBlockN>, Int<kDChunk>>{},
                    make_coord(n_block, d_chunk + 1));
                auto tKgKNext = gmem_k_thread.partition_S(gKNext);
                copy_even_tile(gmem_k_copy, tKgKNext, tKrKNext);
            }

            auto sQChunk = local_tile(
                sQ,
                Shape<Int<Traits::kQkWarpRows>, Int<kDChunk>>{},
                make_coord(
                    mma_group * Traits::kWarpsPerGroup + n_warp,
                    d_chunk));
            auto tSrQ = qk_mma_thread.partition_fragment_A(sQChunk);
            auto tSrK = qk_mma_thread.partition_fragment_B(sK);
            auto tOsQ = qk_mma_thread.partition_A(sQChunk);
            auto tOsK = qk_mma_thread.partition_B(sK);
            auto smem_copy_q = make_tiled_copy_A(
                typename Traits::SmemCopyAtom{}, qk_tiled_mma);
            auto smem_copy_k = make_tiled_copy_B(
                typename Traits::SmemCopyAtom{}, qk_tiled_mma);
            auto smem_thread_q = smem_copy_q.get_thread_slice(lane);
            auto smem_thread_k = smem_copy_k.get_thread_slice(lane);
            auto tSsQ = smem_thread_q.retile_S(tOsQ);
            auto tSsK = smem_thread_k.retile_S(tOsK);
            FLASH_NAMESPACE::gemm<false, false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, qk_tiled_mma,
                smem_copy_q, smem_copy_k, smem_thread_q, smem_thread_k);
            if (d_chunk + 1 < Traits::kDChunks) {
                __syncthreads();
                cute::copy(tKrKNext, tKsK);
                __syncthreads();
            }
        }

        const int64_t v_batch_offset = static_cast<int64_t>(batch)
            * v_outer_stride;
        auto mV = make_tensor(
            make_gmem_ptr(v + v_batch_offset + head_kv * v_head_stride),
            make_shape(kv_len, Int<Traits::kHeadDim>{}),
            make_stride(v_row_stride, _1{}));
        auto sV0 = make_tensor(
            make_smem_ptr(kv_smem_ptr),
            typename Traits::SmemLayoutV{});
        auto sV1 = make_tensor(
            make_smem_ptr(kv_smem_ptr + Traits::kKVElements),
            typename Traits::SmemLayoutV{});
        auto tVsV0 = gmem_v_thread.partition_D(sV0);
        auto tVsV1 = gmem_v_thread.partition_D(sV1);
        auto tVrV0 = make_fragment_like(tVsV0);
        auto tVrV1 = make_fragment_like(tVsV1);
        auto cV = make_identity_tensor(
            Shape<Int<kBlockN>, Int<kDChunk>>{});
        auto tVcV = gmem_v_thread.partition_S(cV);
        auto gV0 = local_tile(
            mV,
            Shape<Int<kBlockN>, Int<kDChunk>>{},
            make_coord(n_block, 0));
        auto gV2 = local_tile(
            mV,
            Shape<Int<kBlockN>, Int<kDChunk>>{},
            make_coord(n_block, Int<Traits::kOwnedDChunks>{}));
        auto tVgV0 = gmem_v_thread.partition_S(gV0);
        auto tVgV2 = gmem_v_thread.partition_S(gV2);
        copy_even_tile(gmem_v_copy, tVgV0, tVrV0);
        copy_even_tile(gmem_v_copy, tVgV2, tVrV1);

        // Mask the tail columns of the last KV tile (col >= kv_len) to -inf.
        // Causal_mask stays false: the explicit mask already carries the
        // causal/padding semantics and is added below.
        FLASH_NAMESPACE::Mask<false, false, false> range_mask(
            kv_len, kv_len - kv_offset, -1, 0, 0.0f);
        range_mask.template apply_mask<false, false>(
            acc_s,
            n_block * kBlockN,
            query_row_base + qk_row_base,
            0);
        // Skip the explicit mask add where the mask is identically zero.
        if (mask != nullptr && n_block >= mask_z_end) {
            if (mask_is_range) {
                splitd_apply_range_mask(
                    acc_s, reinterpret_cast<const int2 *>(mask), mask_row_stride,
                    static_cast<int64_t>(batch) * mask_batch_stride,
                    kv_len - kv_offset, kv_len,
                    query_row_base + qk_row_base, n_block * kBlockN);
            } else {
                splitd_add_explicit_mask(
                    acc_s, mask, mask_row_stride,
                    static_cast<int64_t>(batch) * mask_batch_stride,
                    kv_len - kv_offset, kv_len,
                    query_row_base + qk_row_base, n_block * kBlockN,
                    mask_scale);
            }
        }

        Element *p_smem_ptr =
            kv_smem_ptr + 2 * Traits::kKVElements;
        float *row_scale_exchange = reinterpret_cast<float *>(
            p_smem_ptr + Traits::kPElements);
        splitd_n32_online_softmax<Traits::kGroupRows, false>(
            acc_s, o_storage, OLayout{}, row_max, row_sum,
            row_scale_exchange, mma_group, n_warp, lane,
            softmax_scale_log2, n_block == n_block_max);

        store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
        store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);

        // The pair's 16-row P tile is staged into shared memory (each warp
        // writes its own 8 rows) and loaded back as the pair MMA's A
        // fragment. The barrier makes the P stores - and the V smem stores
        // above - visible before the PV gemms read them.
        auto sP = make_tensor(
            make_smem_ptr(p_smem_ptr), typename Traits::SmemLayoutP{});
        auto sPGroup = local_tile(
            sP,
            Shape<Int<Traits::kGroupRows>, Int<kBlockN>>{},
            make_coord(mma_group, 0));
        auto tPrP = pv_mma_thread.partition_fragment_A(sPGroup);
        {
            auto cS = make_identity_tensor(
                Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
            auto tScS = qk_mma_thread.partition_C(cS);
#pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                const int row = get<0>(tScS(i));
                const int col = get<1>(tScS(i));
                sP(qk_row_base + row, col) = Element(acc_s(i));
            }
            __syncthreads();
        }

        auto gV1 = local_tile(
            mV,
            Shape<Int<kBlockN>, Int<kDChunk>>{},
            make_coord(n_block, 1));
        auto gV3 = local_tile(
            mV,
            Shape<Int<kBlockN>, Int<kDChunk>>{},
            make_coord(n_block, Int<Traits::kOwnedDChunks + 1>{}));
        auto tVgV1 = gmem_v_thread.partition_S(gV1);
        auto tVgV3 = gmem_v_thread.partition_S(gV3);
        copy_even_tile(gmem_v_copy, tVgV1, tVrV0);
        copy_even_tile(gmem_v_copy, tVgV3, tVrV1);

        auto tOsP = pv_mma_thread.partition_A(sPGroup);
        auto smem_copy_p = make_tiled_copy_A(
            typename Traits::SmemCopyAtom{}, pv_tiled_mma);
        auto smem_thread_p = smem_copy_p.get_thread_slice(lane);
        auto tPsP = smem_thread_p.retile_S(tOsP);
        auto tPrPView = smem_thread_p.retile_D(tPrP);
#pragma unroll
        for (int k_tile = 0; k_tile < size<2>(tPrP); ++k_tile) {
            cute::copy(
                smem_copy_p,
                tPsP(_, _, k_tile),
                tPrPView(_, _, k_tile));
        }
#pragma unroll
        for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
            auto acc_o = make_tensor(
                make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
            auto sV = make_tensor(
                make_smem_ptr(
                    kv_smem_ptr + n_warp * Traits::kKVElements),
                typename Traits::SmemLayoutV{});
            splitd_pv_gemm_tt(
                acc_o, tPrP, sV, pv_tiled_mma, lane);
            __syncthreads();
            if (d_local + 1 < Traits::kOwnedDChunks) {
                store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
                store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);
                __syncthreads();
                if (n_block > n_block_min) {
                    auto gKNextBlock = local_tile(
                        mK,
                        Shape<Int<kBlockN>, Int<kDChunk>>{},
                        make_coord(n_block - 1, 0));
                    auto tKgKNextBlock =
                        gmem_k_thread.partition_S(gKNextBlock);
                    copy_even_tile(
                        gmem_k_copy, tKgKNextBlock, tKrKNext);
                }
            }
        }
        if (n_block > n_block_min) {
            cute::copy(tKrKNext, tKsK);
            __syncthreads();
        }
    }

    const int n_warp = warp & 1;
    const int group_row_base = mma_group * Traits::kGroupRows;
    Element *p_smem_ptr = kv_smem_ptr + 2 * Traits::kKVElements;
    float *row_sum_exchange = reinterpret_cast<float *>(
        p_smem_ptr + Traits::kPElements)
        + Traits::kExchangeRows;
    SumOp<float> sum_op;
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        row_sum[slot] = FLASH_NAMESPACE::sm70_row_allreduce_8(
            row_sum[slot], sum_op);
    }
    if constexpr (SplitKV3) {
        // 3-way partial output: unnormalized numerator + raw max/sum per row.
        // The merge kernel (below) combines the three segments with the
        // standard flash-decode scale formula. Layout: [split][row][D] where
        // row = (batch * query_len + query_row) * heads_q + head_q.
        const int64_t split_row_stride =
            static_cast<int64_t>(gridDim.y / 3) * query_len * heads_q;
        if ((lane & 0x0e) == 0) {
#pragma unroll
            for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
                const int row = FLASH_NAMESPACE::sm70_row_slot<
                    Traits::kQkWarpRows>(slot, lane);
                const int query_row = query_row_base
                    + group_row_base + n_warp * Traits::kQkWarpRows + row;
                const int64_t row_offset =
                    (static_cast<int64_t>(batch) * query_len + query_row)
                        * heads_q
                    + head_q;
                const int64_t partial_row =
                    split * split_row_stride + row_offset;
                partial_max[partial_row] = row_max[slot];
                partial_sum[partial_row] = row_sum[slot];
            }
        }

#pragma unroll
        for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
            auto acc_o = make_tensor(
                make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
            auto cO = make_identity_tensor(
                Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
            auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
            for (int i = 0; i < size(acc_o); ++i) {
                const int row = get<0>(tOcO(i));
                const int col = get<1>(tOcO(i));
                const int query_row = query_row_base + group_row_base + row;
                const int64_t row_offset =
                    (static_cast<int64_t>(batch) * query_len + query_row)
                        * heads_q
                    + head_q;
                const int64_t partial_row =
                    split * split_row_stride + row_offset;
                const int d =
                    (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk
                    + col;
                partial_out[partial_row * Traits::kHeadDim + d] = acc_o(i);
            }
        }
    } else if constexpr (Partial) {
        // Window partial output: same stores as SplitKV3 at split 0 (no
        // gridDim.y / 3) and a plain [row][D] slice per window. An empty
        // window (n_block_max < n_block_min) skips the KV loop, so the row
        // values below are the -inf/0/0 initialization.
        if ((lane & 0x0e) == 0) {
#pragma unroll
            for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
                const int row = FLASH_NAMESPACE::sm70_row_slot<
                    Traits::kQkWarpRows>(slot, lane);
                const int query_row = query_row_base
                    + group_row_base + n_warp * Traits::kQkWarpRows + row;
                const int64_t row_offset =
                    (static_cast<int64_t>(batch) * query_len + query_row)
                        * heads_q
                    + head_q;
                partial_max[row_offset] = row_max[slot];
                partial_sum[row_offset] = row_sum[slot];
            }
        }

#pragma unroll
        for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
            auto acc_o = make_tensor(
                make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
            auto cO = make_identity_tensor(
                Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
            auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
            for (int i = 0; i < size(acc_o); ++i) {
                const int row = get<0>(tOcO(i));
                const int col = get<1>(tOcO(i));
                const int query_row = query_row_base + group_row_base + row;
                const int64_t row_offset =
                    (static_cast<int64_t>(batch) * query_len + query_row)
                        * heads_q
                    + head_q;
                const int d =
                    (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk
                    + col;
                partial_out[row_offset * Traits::kHeadDim + d] = acc_o(i);
            }
        }
    } else {
        if ((lane & 0x0e) == 0) {
#pragma unroll
            for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
                const int row = FLASH_NAMESPACE::sm70_row_slot<
                    Traits::kQkWarpRows>(slot, lane);
                row_sum_exchange[
                    mma_group * Traits::kGroupRows
                    + n_warp * Traits::kQkWarpRows + row] = row_sum[slot];
            }
        }
        __syncthreads();

        const int64_t out_batch_offset =
            static_cast<int64_t>(batch) * query_len * heads_q * Traits::kHeadDim;
#pragma unroll
        for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
            auto acc_o = make_tensor(
                make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
            auto acc_o_rc = make_tensor(
                acc_o.data(),
                FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
                const int logical_row = FLASH_NAMESPACE::sm70_row_slot<
                    Traits::kGroupRows>(row, lane);
                const float sum = row_sum_exchange[
                    mma_group * Traits::kGroupRows + logical_row];
                // A fully masked row has sum 0; llama.cpp expects 0 there.
                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
                for (int col = 0; col < size<1>(acc_o_rc); ++col) {
                    acc_o_rc(row, col) *= inv_sum;
                }
            }

            auto cO = make_identity_tensor(
                Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
            auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
            for (int i = 0; i < size(acc_o); ++i) {
                const int row = get<0>(tOcO(i));
                const int col = get<1>(tOcO(i));
                const int query_row = query_row_base + group_row_base + row;
                const int64_t offset = out_batch_offset
                    + static_cast<int64_t>(query_row) * heads_q
                      * Traits::kHeadDim
                    + head_q * Traits::kHeadDim
                    + (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk
                      + col;
                out[offset] = ElementOut(acc_o(i));
            }
        }
    }
}

// SplitKV3 merge kernel (from the upstream sm70_flash_attn_d256_splitkv3
// patch): combines the three partial segments per output row. Writes into the
// f32 Os staging buffer (same [row][D] layout the dense path produces and the
// scatter kernel consumes).
__global__ __launch_bounds__(256, 1)
void sm70_d256_splitkv3_merge_kernel(
        const float *__restrict__ partial_out,
        const float *__restrict__ partial_max,
        const float *__restrict__ partial_sum,
        float *__restrict__ out,
        int64_t rows,
        float softmax_scale_log2) {
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x;
    __shared__ float merge[4];
    if (d == 0) {
        const float max0 = partial_max[row];
        const float max1 = partial_max[rows + row];
        const float max2 = partial_max[2 * rows + row];
        const float global_max = fmaxf(fmaxf(max2, max1), max0);
        const float scale2 = exp2f((max2 - global_max) * softmax_scale_log2);
        const float scale1 = exp2f((max1 - global_max) * softmax_scale_log2);
        const float scale0 = exp2f((max0 - global_max) * softmax_scale_log2);
        const float denominator =
            (partial_sum[2 * rows + row] * scale2
             + partial_sum[rows + row] * scale1)
            + partial_sum[row] * scale0;
        merge[0] = scale0;
        merge[1] = scale1;
        merge[2] = scale2;
        // Fully masked rows have denominator 0; llama.cpp expects 0 there.
        merge[3] = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    }
    __syncthreads();

    const int64_t element = row * Sm70D256SplitDTraitsT<64>::kHeadDim + d;
    const int64_t split_stride = rows * Sm70D256SplitDTraitsT<64>::kHeadDim;
    const float numerator =
        (partial_out[2 * split_stride + element] * merge[2]
         + partial_out[split_stride + element] * merge[1])
        + partial_out[element] * merge[0];
    out[element] = numerator * merge[3];
}

// Window merge: folds one window's partials p_* into the running accumulator
// acc_* (first = true copies the first window into the accumulator). Two-way
// online-softmax merge: m = max(ma, mp) and each side is scaled by
// exp2((max - m) * softmax_scale_log2), the same exponent base and factor as
// sm70_d256_splitkv3_merge_kernel. Both sides -inf (no KV row in either) keeps
// max -inf, sum 0, O 0 so later merges and the finalize stay empty. One CTA per
// row, 256 threads cover D=256; the caller launches it once per window with
// p_* pointing at the window's [row][D] partial slice.
__global__ __launch_bounds__(256, 1)
void sm70_d256_window_merge_kernel(
        float *__restrict__ acc_out,
        float *__restrict__ acc_max,
        float *__restrict__ acc_sum,
        const float *__restrict__ p_out,
        const float *__restrict__ p_max,
        const float *__restrict__ p_sum,
        int64_t rows,
        bool first,
        float softmax_scale_log2) {
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t element = row * Sm70D256SplitDTraitsT<64>::kHeadDim + d;
    if (first) {
        acc_max[row] = p_max[row];
        acc_sum[row] = p_sum[row];
        acc_out[element] = p_out[element];
        return;
    }

    __shared__ float merge[2];
    if (d == 0) {
        const float max_a = acc_max[row];
        const float max_p = p_max[row];
        const float m = fmaxf(max_a, max_p);
        if (m == -INFINITY) {
            acc_max[row] = -INFINITY;
            acc_sum[row] = 0.0f;
            merge[0] = 0.0f;
            merge[1] = 0.0f;
        } else {
            merge[0] = exp2f((max_a - m) * softmax_scale_log2);
            merge[1] = exp2f((max_p - m) * softmax_scale_log2);
            acc_max[row] = m;
            acc_sum[row] = acc_sum[row] * merge[0] + p_sum[row] * merge[1];
        }
    }
    __syncthreads();
    acc_out[element] = acc_out[element] * merge[0] + p_out[element] * merge[1];
}

// Window finalize: out = acc_out / acc_sum, with 0 for rows that saw no KV
// block. The output index is the same one the non-Partial dense kernel uses:
// ((batch*query_len + query_row)*heads_q + head_q)*kHeadDim + d, i.e. the
// [row][D] row-major layout of the partial buffers. One CTA per row, 256
// threads cover D=256.
__global__ __launch_bounds__(256, 1)
void sm70_d256_window_finalize_kernel(
        const float *__restrict__ acc_out,
        const float *__restrict__ acc_sum,
        float *__restrict__ out,
        int64_t rows) {
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float sum = acc_sum[row];
    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    const int64_t element = row * Sm70D256SplitDTraitsT<64>::kHeadDim + d;
    out[element] = acc_out[element] * inv_sum;
}

}  // namespace FLASH_NAMESPACE
