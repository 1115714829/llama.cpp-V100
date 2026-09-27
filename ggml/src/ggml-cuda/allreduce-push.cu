#include "allreduce-push.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "ggml-impl.h"

#include <climits>
#include <cstdint>

// One-shot push AllReduce.
//
// Every rank writes its own payload directly into the receive slot of every
// rank (including itself). Slots are pre-filled with an all-ones sentinel, so
// a poller can tell a filled slot from an empty one without any extra flag or
// barrier. After summing the slots and restoring the sentinel, each block
// flips its own epoch word; two epochs alternate so a block never touches the
// slots another in-flight call of the same stream is still using.
//
// Two payload formats share the same buffers and slot layout:
// ggml_cuda_ar_push_f32 sends the raw f32 words (exact), while
// ggml_cuda_ar_push_f32_via_f16 sends them as f16 to halve the P2P traffic
// (approximate, as in 1Cat's sm70 push AllReduce). The f16 kernel only uses
// the first half of each slot. The two kernels can be interleaved freely:
// every call restores the slots it touched to all-0xFF bytes, which both
// formats read as an empty slot.
//
// For 4 ranks that split into two cliques of two (e.g. two GPU pairs on
// different CPU sockets), ggml_cuda_ar_push_f32_via_f16_hier halves the
// traffic on the expensive cross-clique links: each rank first pushes its f16
// payload to itself and to its clique peer only (slots 0/1), both clique
// members reduce those two slots in the same order and round the partial sum
// to f16, and each rank sends that rounded partial sum to its counterpart in
// the other clique (slot 2). Both sides of the final sum are f16-rounded and
// the sum is always "clique 0 + clique 1", so all ranks get bitwise identical
// results. Protocol from 1Cat-vLLM `csrc/custom_all_reduce.cuh`
// (`sm70_tp8_hierarchical_reduce_push`, Apache-2.0), except that the
// cross-clique exchange carries f16 partial sums instead of f32.
//
// The grid size is fixed at GGML_CUDA_AR_PUSH_BLOCKS for every call: the epoch
// words are indexed by block, so a different grid would make blocks disagree
// about which slot set is current. For the same reason two epochs are only
// safe as long as calls on all ranks stay in lockstep, which the collective
// itself guarantees (a call cannot finish before every rank has pushed).

static constexpr int    GGML_CUDA_AR_PUSH_MAX_RANKS    = 8;
static constexpr int    GGML_CUDA_AR_PUSH_BLOCKS       = 80;
static constexpr int    GGML_CUDA_AR_PUSH_THREADS      = 128;
static constexpr int    GGML_CUDA_AR_PUSH_EPOCHS       = 2;
static constexpr size_t GGML_CUDA_AR_PUSH_MAX_BYTES    = 512 * 1024; // limit on the f32 input
static constexpr size_t GGML_CUDA_AR_PUSH_SIGNAL_BYTES =
    ((GGML_CUDA_AR_PUSH_BLOCKS * sizeof(uint32_t) + 127) / 128) * 128;

static constexpr uint16_t GGML_CUDA_AR_PUSH_SENTINEL_H  = 0xFFFFu;     // empty half word
static constexpr uint16_t GGML_CUDA_AR_PUSH_ESCAPE_H    = 0x7FFFu;     // quiet NaN, still distinguishable from an empty half
static constexpr uint32_t GGML_CUDA_AR_PUSH_SENTINEL    = 0xFFFFFFFFu; // empty f32 word, also the byte pattern of an empty slot
static constexpr uint32_t GGML_CUDA_AR_PUSH_ESCAPE      = 0x7FFFFFFFu; // quiet NaN, still distinguishable from an empty slot

struct ggml_cuda_ar_push {
    size_t n;
    int    devices[GGML_CUDA_AR_PUSH_MAX_RANKS];
    char * bufs[GGML_CUDA_AR_PUSH_MAX_RANKS];
    // 4 ranks in two cliques of two, with intra-clique links cheaper than
    // cross-clique links; see ggml_cuda_ar_push_f32_via_f16_hier.
    bool   hier;
    int    clique_slot[GGML_CUDA_AR_PUSH_MAX_RANKS]; // position inside the clique, 0 or 1
    int    clique_peer[GGML_CUDA_AR_PUSH_MAX_RANKS]; // other rank of the same clique
    int    pair[GGML_CUDA_AR_PUSH_MAX_RANKS];        // rank with the same clique_slot in the other clique
    int    clique_id[GGML_CUDA_AR_PUSH_MAX_RANKS];   // 0 for the clique of rank 0, 1 for the other
};

struct ggml_cuda_ar_push_ptrs {
    char * ptrs[GGML_CUDA_AR_PUSH_MAX_RANKS];
};

// One 16-byte pack, viewed either as raw words (sentinel check) or as payload.
union ggml_cuda_ar_push_pack {
    uint4  bits;
    float4 vals;
    __half halves[8];
};

static __device__ __forceinline__ uint4 ggml_cuda_ar_push_ld_volatile(const uint4 * p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

static __device__ __forceinline__ void ggml_cuda_ar_push_st_volatile(uint4 * p, uint4 v) {
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
                 :
                 : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(p)
                 : "memory");
}

// A half word equal to the sentinel is rewritten to a quiet NaN so it stays a
// NaN but can no longer be mistaken for an empty slot.
static __device__ __forceinline__ uint16_t ggml_cuda_ar_push_to_half(const float x) {
    const uint16_t h = __half_as_ushort(__float2half_rn(x));
    return h == GGML_CUDA_AR_PUSH_SENTINEL_H ? GGML_CUDA_AR_PUSH_ESCAPE_H : h;
}

// input == output (in-place), so no __restrict__ on either pointer.
template <int nranks>
static __global__ void __launch_bounds__(GGML_CUDA_AR_PUSH_THREADS, 1) ggml_cuda_ar_push_f32(
        const ggml_cuda_ar_push_ptrs bufs, const float * input, float * output, const int rank, const int n_packs) {
    char     * local  = bufs.ptrs[rank];
    uint32_t * epochs = reinterpret_cast<uint32_t *>(local);
    const uint32_t epoch = epochs[blockIdx.x];
    const size_t base = GGML_CUDA_AR_PUSH_SIGNAL_BYTES +
                        (size_t) epoch * nranks * GGML_CUDA_AR_PUSH_MAX_BYTES;
    const int stride = gridDim.x * blockDim.x;

    // Push: write my pack into slot [epoch][rank] of every rank.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        uint4 v = reinterpret_cast<const uint4 *>(input)[i];
        // A payload word equal to the sentinel is rewritten to a quiet NaN so
        // it stays a NaN but can no longer be mistaken for an empty slot.
        if (v.x == GGML_CUDA_AR_PUSH_SENTINEL) { v.x = GGML_CUDA_AR_PUSH_ESCAPE; }
        if (v.y == GGML_CUDA_AR_PUSH_SENTINEL) { v.y = GGML_CUDA_AR_PUSH_ESCAPE; }
        if (v.z == GGML_CUDA_AR_PUSH_SENTINEL) { v.z = GGML_CUDA_AR_PUSH_ESCAPE; }
        if (v.w == GGML_CUDA_AR_PUSH_SENTINEL) { v.w = GGML_CUDA_AR_PUSH_ESCAPE; }

        #pragma unroll
        for (int dst = 0; dst < nranks; ++dst) {
            uint4 * slot = reinterpret_cast<uint4 *>(
                bufs.ptrs[dst] + base + (size_t) rank * GGML_CUDA_AR_PUSH_MAX_BYTES);
            ggml_cuda_ar_push_st_volatile(slot + i, v);
        }
    }

    // Poll until every source filled its slot for this pack, then sum, store
    // and restore the sentinels. All ranks sum in the same source order so the
    // results are bitwise identical.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        ggml_cuda_ar_push_pack packs[nranks];
        bool ready = false;
        while (!ready) {
            ready = true;
            #pragma unroll
            for (int src = 0; src < nranks; ++src) {
                const uint4 * slot = reinterpret_cast<const uint4 *>(
                    local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
                packs[src].bits = ggml_cuda_ar_push_ld_volatile(slot + i);
                ready = ready &&
                        packs[src].bits.x != GGML_CUDA_AR_PUSH_SENTINEL &&
                        packs[src].bits.y != GGML_CUDA_AR_PUSH_SENTINEL &&
                        packs[src].bits.z != GGML_CUDA_AR_PUSH_SENTINEL &&
                        packs[src].bits.w != GGML_CUDA_AR_PUSH_SENTINEL;
            }
        }

        float4 sum = packs[0].vals;
        #pragma unroll
        for (int src = 1; src < nranks; ++src) {
            const float4 p = packs[src].vals;
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
            sum.w += p.w;
        }
        reinterpret_cast<float4 *>(output)[i] = sum;

        const uint4 empty = make_uint4(GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL,
                                       GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL);
        #pragma unroll
        for (int src = 0; src < nranks; ++src) {
            uint4 * slot = reinterpret_cast<uint4 *>(
                local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
            ggml_cuda_ar_push_st_volatile(slot + i, empty);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        epochs[blockIdx.x] = (epoch + 1) % GGML_CUDA_AR_PUSH_EPOCHS;
    }
}

// Same collective as ggml_cuda_ar_push_f32, but the payload travels as f16 on
// push and is accumulated in f32. Only the first half of each slot is used.
// input == output (in-place), so no __restrict__ on either pointer.
template <int nranks>
static __global__ void __launch_bounds__(GGML_CUDA_AR_PUSH_THREADS, 1) ggml_cuda_ar_push_f32_via_f16(
        const ggml_cuda_ar_push_ptrs bufs, const float * input, float * output, const int rank, const int n_packs) {
    char     * local  = bufs.ptrs[rank];
    uint32_t * epochs = reinterpret_cast<uint32_t *>(local);
    const uint32_t epoch = epochs[blockIdx.x];
    const size_t base = GGML_CUDA_AR_PUSH_SIGNAL_BYTES +
                        (size_t) epoch * nranks * GGML_CUDA_AR_PUSH_MAX_BYTES;
    const int stride = gridDim.x * blockDim.x;

    // Push: convert my pack to f16 and write it into slot [epoch][rank] of every rank.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        const float4 lo = reinterpret_cast<const float4 *>(input)[2 * i];
        const float4 hi = reinterpret_cast<const float4 *>(input)[2 * i + 1];
        const uint4 v = make_uint4(
            (uint32_t) ggml_cuda_ar_push_to_half(lo.x) | ((uint32_t) ggml_cuda_ar_push_to_half(lo.y) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(lo.z) | ((uint32_t) ggml_cuda_ar_push_to_half(lo.w) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(hi.x) | ((uint32_t) ggml_cuda_ar_push_to_half(hi.y) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(hi.z) | ((uint32_t) ggml_cuda_ar_push_to_half(hi.w) << 16));

        #pragma unroll
        for (int dst = 0; dst < nranks; ++dst) {
            uint4 * slot = reinterpret_cast<uint4 *>(
                bufs.ptrs[dst] + base + (size_t) rank * GGML_CUDA_AR_PUSH_MAX_BYTES);
            ggml_cuda_ar_push_st_volatile(slot + i, v);
        }
    }

    // Poll until every source filled its slot for this pack, then sum in f32,
    // store and restore the sentinels. All ranks sum in the same source order
    // so the results are bitwise identical.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        ggml_cuda_ar_push_pack packs[nranks];
        bool ready = false;
        while (!ready) {
            ready = true;
            #pragma unroll
            for (int src = 0; src < nranks; ++src) {
                const uint4 * slot = reinterpret_cast<const uint4 *>(
                    local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
                packs[src].bits = ggml_cuda_ar_push_ld_volatile(slot + i);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    ready = ready && __half_as_ushort(packs[src].halves[k]) != GGML_CUDA_AR_PUSH_SENTINEL_H;
                }
            }
        }

        float sum[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            sum[k] = __half2float(packs[0].halves[k]);
        }
        #pragma unroll
        for (int src = 1; src < nranks; ++src) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum[k] += __half2float(packs[src].halves[k]);
            }
        }
        reinterpret_cast<float4 *>(output)[2 * i]     = make_float4(sum[0], sum[1], sum[2], sum[3]);
        reinterpret_cast<float4 *>(output)[2 * i + 1] = make_float4(sum[4], sum[5], sum[6], sum[7]);

        const uint4 empty = make_uint4(GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL,
                                       GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL);
        #pragma unroll
        for (int src = 0; src < nranks; ++src) {
            uint4 * slot = reinterpret_cast<uint4 *>(
                local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
            ggml_cuda_ar_push_st_volatile(slot + i, empty);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        epochs[blockIdx.x] = (epoch + 1) % GGML_CUDA_AR_PUSH_EPOCHS;
    }
}

// Hierarchical variant of the f16 kernel for 4 ranks in two cliques of two.
// Each rank pushes its f16 payload to itself and to its clique peer into slot
// [clique_slot]; both clique members then poll slots 0 and 1, sum them in that
// order and round the result to f16 once. The rounded value is written to the
// pair rank's slot 2 and added to the partner's rounded value in clique order.
// input == output (in-place), so no __restrict__ on either pointer.
static __global__ void __launch_bounds__(GGML_CUDA_AR_PUSH_THREADS, 1) ggml_cuda_ar_push_f32_via_f16_hier(
        const ggml_cuda_ar_push_ptrs bufs, const float * input, float * output, const int rank, const int n_packs,
        const int clique_slot, const int clique_peer, const int pair, const int clique_id) {
    constexpr int nranks = 4;
    char     * local  = bufs.ptrs[rank];
    uint32_t * epochs = reinterpret_cast<uint32_t *>(local);
    const uint32_t epoch = epochs[blockIdx.x];
    const size_t base = GGML_CUDA_AR_PUSH_SIGNAL_BYTES +
                        (size_t) epoch * nranks * GGML_CUDA_AR_PUSH_MAX_BYTES;
    const int stride = gridDim.x * blockDim.x;

    // Push: convert my pack to f16 and write it into slot [clique_slot] of
    // myself and of my clique peer.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        const float4 lo = reinterpret_cast<const float4 *>(input)[2 * i];
        const float4 hi = reinterpret_cast<const float4 *>(input)[2 * i + 1];
        const uint4 v = make_uint4(
            (uint32_t) ggml_cuda_ar_push_to_half(lo.x) | ((uint32_t) ggml_cuda_ar_push_to_half(lo.y) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(lo.z) | ((uint32_t) ggml_cuda_ar_push_to_half(lo.w) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(hi.x) | ((uint32_t) ggml_cuda_ar_push_to_half(hi.y) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(hi.z) | ((uint32_t) ggml_cuda_ar_push_to_half(hi.w) << 16));

        uint4 * own = reinterpret_cast<uint4 *>(
            local + base + (size_t) clique_slot * GGML_CUDA_AR_PUSH_MAX_BYTES);
        uint4 * peer = reinterpret_cast<uint4 *>(
            bufs.ptrs[clique_peer] + base + (size_t) clique_slot * GGML_CUDA_AR_PUSH_MAX_BYTES);
        ggml_cuda_ar_push_st_volatile(own + i, v);
        ggml_cuda_ar_push_st_volatile(peer + i, v);
    }

    // Poll my slots 0 and 1, sum them in slot order and round the partial sum
    // to f16 once. Every rank uses the rounded value, so the f32 partial sums
    // are bitwise identical across the clique.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packs; i += stride) {
        ggml_cuda_ar_push_pack packs[2];
        bool ready = false;
        while (!ready) {
            ready = true;
            #pragma unroll
            for (int src = 0; src < 2; ++src) {
                const uint4 * slot = reinterpret_cast<const uint4 *>(
                    local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
                packs[src].bits = ggml_cuda_ar_push_ld_volatile(slot + i);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    ready = ready && __half_as_ushort(packs[src].halves[k]) != GGML_CUDA_AR_PUSH_SENTINEL_H;
                }
            }
        }

        float partial[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            partial[k] = __half2float(packs[0].halves[k]) + __half2float(packs[1].halves[k]);
        }
        ggml_cuda_ar_push_pack rounded;
        rounded.bits = make_uint4(
            (uint32_t) ggml_cuda_ar_push_to_half(partial[0]) | ((uint32_t) ggml_cuda_ar_push_to_half(partial[1]) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(partial[2]) | ((uint32_t) ggml_cuda_ar_push_to_half(partial[3]) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(partial[4]) | ((uint32_t) ggml_cuda_ar_push_to_half(partial[5]) << 16),
            (uint32_t) ggml_cuda_ar_push_to_half(partial[6]) | ((uint32_t) ggml_cuda_ar_push_to_half(partial[7]) << 16));

        // Cross-clique exchange: send my rounded partial sum to the pair rank
        // and poll its slot 2 for the partner's rounded partial sum.
        uint4 * cross = reinterpret_cast<uint4 *>(
            bufs.ptrs[pair] + base + 2 * GGML_CUDA_AR_PUSH_MAX_BYTES);
        ggml_cuda_ar_push_st_volatile(cross + i, rounded.bits);

        ggml_cuda_ar_push_pack other;
        bool other_ready = false;
        while (!other_ready) {
            const uint4 * slot = reinterpret_cast<const uint4 *>(
                local + base + 2 * GGML_CUDA_AR_PUSH_MAX_BYTES);
            other.bits = ggml_cuda_ar_push_ld_volatile(slot + i);
            other_ready = true;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                other_ready = other_ready && __half_as_ushort(other.halves[k]) != GGML_CUDA_AR_PUSH_SENTINEL_H;
            }
        }

        // Fixed clique 0 + clique 1 order, so the sum is bitwise identical
        // everywhere. Store the result and restore my three slots.
        float total[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float own = __half2float(rounded.halves[k]);
            const float oth = __half2float(other.halves[k]);
            total[k] = clique_id == 0 ? own + oth : oth + own;
        }
        reinterpret_cast<float4 *>(output)[2 * i]     = make_float4(total[0], total[1], total[2], total[3]);
        reinterpret_cast<float4 *>(output)[2 * i + 1] = make_float4(total[4], total[5], total[6], total[7]);

        const uint4 empty = make_uint4(GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL,
                                       GGML_CUDA_AR_PUSH_SENTINEL, GGML_CUDA_AR_PUSH_SENTINEL);
        #pragma unroll
        for (int src = 0; src < 3; ++src) {
            uint4 * slot = reinterpret_cast<uint4 *>(
                local + base + (size_t) src * GGML_CUDA_AR_PUSH_MAX_BYTES);
            ggml_cuda_ar_push_st_volatile(slot + i, empty);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        epochs[blockIdx.x] = (epoch + 1) % GGML_CUDA_AR_PUSH_EPOCHS;
    }
}

template <int nranks>
static void ggml_cuda_ar_push_launch(
        const ggml_cuda_ar_push * ar,
        const ggml_cuda_ar_push_ptrs & bufs, const float * input, float * output,
        const int rank, const int n_packs, const bool exact, cudaStream_t stream) {
    if (exact) {
        ggml_cuda_ar_push_f32<nranks><<<GGML_CUDA_AR_PUSH_BLOCKS, GGML_CUDA_AR_PUSH_THREADS, 0, stream>>>(
            bufs, input, output, rank, n_packs);
    } else if (nranks == 4 && ar->hier) {
        ggml_cuda_ar_push_f32_via_f16_hier<<<GGML_CUDA_AR_PUSH_BLOCKS, GGML_CUDA_AR_PUSH_THREADS, 0, stream>>>(
            bufs, input, output, rank, n_packs,
            ar->clique_slot[rank], ar->clique_peer[rank], ar->pair[rank], ar->clique_id[rank]);
    } else {
        ggml_cuda_ar_push_f32_via_f16<nranks><<<GGML_CUDA_AR_PUSH_BLOCKS, GGML_CUDA_AR_PUSH_THREADS, 0, stream>>>(
            bufs, input, output, rank, n_packs);
    }
}

ggml_cuda_ar_push * ggml_cuda_ar_push_init(const int * devices, size_t n) {
    if (n < 2 || n > GGML_CUDA_AR_PUSH_MAX_RANKS) {
        return nullptr;
    }

    // Virtual (MIG-style) device splits would give several ranks the same
    // physical GPU; a pure device-to-device collective cannot work there.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_DEBUG("%s: virtual devices in use; push AllReduce disabled\n", __func__);
        return nullptr;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]) != cudaSuccess || !can_access) {
                (void) cudaGetLastError();
                GGML_LOG_DEBUG("%s: devices %d and %d cannot do P2P; push AllReduce disabled\n",
                               __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }

    const size_t buf_bytes = GGML_CUDA_AR_PUSH_SIGNAL_BYTES +
                             (size_t) GGML_CUDA_AR_PUSH_EPOCHS * n * GGML_CUDA_AR_PUSH_MAX_BYTES;

    auto * ar = new ggml_cuda_ar_push{};
    ar->n = n;
    for (size_t i = 0; i < n; ++i) {
        ar->devices[i] = devices[i];
    }

    // Look for two cliques of two ranks: the intra-clique links must be the
    // cheapest the devices report and the cross-clique links must be more
    // expensive. cl0 holds the clique of rank 0, cl1 the other one, both
    // sorted by rank. Without such a split the flat kernel is used.
    int cl0[2] = { -1, -1 };
    int cl1[2] = { -1, -1 };
    if (n == 4) {
        int perf[4][4] = {};
        bool perf_ok = true;
        for (int i = 0; i < 4 && perf_ok; ++i) {
            for (int j = 0; j < 4 && perf_ok; ++j) {
                if (i == j) {
                    continue;
                }
                if (cudaDeviceGetP2PAttribute(&perf[i][j], cudaDevP2PAttrPerformanceRank, devices[i], devices[j]) != cudaSuccess) {
                    (void) cudaGetLastError();
                    perf_ok = false;
                }
            }
        }
        if (perf_ok) {
            int best = INT_MAX;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i != j && perf[i][j] < best) {
                        best = perf[i][j];
                    }
                }
            }
            for (int p = 1; p < 4 && cl0[0] < 0; ++p) {
                int g1[2] = { -1, -1 };
                int n1 = 0;
                for (int r = 1; r < 4; ++r) {
                    if (r != p) {
                        g1[n1++] = r;
                    }
                }
                bool cheap_inside = true;
                bool expensive_across = true;
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                        if (a == b) {
                            continue;
                        }
                        const bool inside = (a == 0 || a == p) == (b == 0 || b == p);
                        if (inside) {
                            cheap_inside = cheap_inside && perf[a][b] == best;
                        } else {
                            expensive_across = expensive_across && perf[a][b] > best;
                        }
                    }
                }
                if (!cheap_inside || !expensive_across) {
                    continue;
                }
                cl0[0] = 0;     cl0[1] = p;
                cl1[0] = g1[0]; cl1[1] = g1[1];
                ar->hier = true;
                ar->clique_slot[0]     = 0; ar->clique_peer[0]     = p;      ar->pair[0]     = g1[0]; ar->clique_id[0]     = 0;
                ar->clique_slot[p]     = 1; ar->clique_peer[p]     = 0;      ar->pair[p]     = g1[1]; ar->clique_id[p]     = 0;
                ar->clique_slot[g1[0]] = 0; ar->clique_peer[g1[0]] = g1[1];  ar->pair[g1[0]] = 0;     ar->clique_id[g1[0]] = 1;
                ar->clique_slot[g1[1]] = 1; ar->clique_peer[g1[1]] = g1[0];  ar->pair[g1[1]] = p;     ar->clique_id[g1[1]] = 1;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        ggml_cuda_set_device(devices[i]);

        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const cudaError_t rc = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
            } else if (rc != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: cudaDeviceEnablePeerAccess(%d -> %d) failed (%s)\n",
                               __func__, devices[i], devices[j], cudaGetErrorString(rc));
                ggml_cuda_ar_push_free(ar);
                return nullptr;
            }
        }

        if (cudaMalloc(reinterpret_cast<void **>(&ar->bufs[i]), buf_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc failed (%zu bytes) on device %d\n", __func__, buf_bytes, devices[i]);
            (void) cudaGetLastError();
            ggml_cuda_ar_push_free(ar);
            return nullptr;
        }
        if (cudaMemset(ar->bufs[i], 0, GGML_CUDA_AR_PUSH_SIGNAL_BYTES) != cudaSuccess ||
            cudaMemset(ar->bufs[i] + GGML_CUDA_AR_PUSH_SIGNAL_BYTES, 0xFF, buf_bytes - GGML_CUDA_AR_PUSH_SIGNAL_BYTES) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMemset failed on device %d\n", __func__, devices[i]);
            (void) cudaGetLastError();
            ggml_cuda_ar_push_free(ar);
            return nullptr;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        ggml_cuda_set_device(devices[i]);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaDeviceSynchronize failed on device %d\n", __func__, devices[i]);
            (void) cudaGetLastError();
            ggml_cuda_ar_push_free(ar);
            return nullptr;
        }
    }

    char hier_info[64] = "";
    if (ar->hier) {
        snprintf(hier_info, sizeof(hier_info), ", hierarchical 2+2 (cliques {%d,%d} {%d,%d})",
                 cl0[0], cl0[1], cl1[0], cl1[1]);
    }
    GGML_LOG_INFO("%s: P2P push AllReduce enabled for %zu GPUs, up to %zu KiB per call%s\n",
                  __func__, n, GGML_CUDA_AR_PUSH_MAX_BYTES >> 10, hier_info);

    return ar;
}

void ggml_cuda_ar_push_free(ggml_cuda_ar_push * ar) {
    if (ar == nullptr) {
        return;
    }
    for (size_t i = 0; i < ar->n; ++i) {
        if (ar->bufs[i] != nullptr) {
            ggml_cuda_set_device(ar->devices[i]);
            CUDA_CHECK(cudaFree(ar->bufs[i]));
        }
    }
    delete ar;
}

// Admission checks for one rank's tensor, shared by the all-rank and the per-rank
// entry points. On success n_packs is the pack count for the launch.
static bool ggml_cuda_ar_push_can_impl(const ggml_tensor * t, bool exact, int & n_packs) {
    if (t == nullptr ||
        t->type != GGML_TYPE_F32 ||
        !ggml_is_contiguously_allocated(t) ||
        ((uintptr_t) t->data & 0xF) != 0) {
        return false;
    }

    const int64_t ne = ggml_nelements(t);
    const size_t nbytes = (size_t) ne * sizeof(float);
    if (nbytes > GGML_CUDA_AR_PUSH_MAX_BYTES) {
        return false;
    }

    if (exact) {
        if (nbytes % 16 != 0) {
            return false;
        }
        n_packs = (int) (nbytes / 16);
    } else {
        if (ne % 8 != 0) {
            return false;
        }
        n_packs = (int) (ne / 8);
    }
    return true;
}

bool ggml_cuda_ar_push_can(const ggml_cuda_ar_push * ar, const ggml_tensor * t, bool exact) {
    if (ar == nullptr) {
        return false;
    }
    int n_packs;
    return ggml_cuda_ar_push_can_impl(t, exact, n_packs);
}

// Enqueue one rank's part of the reduction. No synchronization: the kernel
// spins on the peer ranks, so waiting for one rank here would deadlock the
// whole collective.
static void ggml_cuda_ar_push_launch_rank(
        ggml_cuda_ar_push * ar, ggml_backend_t backend, size_t rank, ggml_tensor * t, bool exact, int n_packs) {
    GGML_ASSERT(rank < ar->n);
    ggml_backend_cuda_context * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    GGML_ASSERT(cuda_ctx->device == ar->devices[rank]);
    ggml_cuda_set_device(cuda_ctx->device);

    if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        CUDA_CHECK(cudaMemsetAsync(t->data, 0, ggml_nbytes(t), cuda_ctx->stream()));
    }

    ggml_cuda_ar_push_ptrs bufs = {};
    for (size_t i = 0; i < ar->n; ++i) {
        bufs.ptrs[i] = ar->bufs[i];
    }

    const float * input  = static_cast<const float *>(t->data);
    float       * output = static_cast<float *>(t->data);

    switch (ar->n) {
        case 2: ggml_cuda_ar_push_launch<2>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 3: ggml_cuda_ar_push_launch<3>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 4: ggml_cuda_ar_push_launch<4>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 5: ggml_cuda_ar_push_launch<5>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 6: ggml_cuda_ar_push_launch<6>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 7: ggml_cuda_ar_push_launch<7>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        case 8: ggml_cuda_ar_push_launch<8>(ar, bufs, input, output, (int) rank, n_packs, exact, cuda_ctx->stream()); break;
        default: GGML_ABORT("unsupported number of ranks: %zu", ar->n);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_ar_push_allreduce_rank(
        ggml_cuda_ar_push * ar, ggml_backend_t backend, size_t rank, ggml_tensor * t, bool exact) {
    GGML_ASSERT(ar != nullptr);
    GGML_ASSERT(rank < ar->n);

    if (t != nullptr && ggml_nelements(t) == 0) {
        return true;
    }

    int n_packs = 0;
    if (!ggml_cuda_ar_push_can_impl(t, exact, n_packs)) {
        return false;
    }

    ggml_cuda_ar_push_launch_rank(ar, backend, rank, t, exact, n_packs);
    return true;
}

bool ggml_cuda_ar_push_allreduce(
        ggml_cuda_ar_push * ar,
        ggml_backend_t   * backends,
        ggml_tensor      ** tensors,
        bool               exact) {
    GGML_ASSERT(ar != nullptr);

    const size_t n = ar->n;

    if (tensors[0] == nullptr) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }

    // Check every rank before launching any of them, so an unsupported input
    // cannot leave the collective half-enqueued.
    int n_packs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (tensors[i] == nullptr ||
            ggml_nelements(tensors[i]) != ne ||
            !ggml_cuda_ar_push_can_impl(tensors[i], exact, n_packs)) {
            return false;
        }
    }

    // No synchronization in this loop: the kernel spins on the peer ranks, so
    // waiting for one rank here would deadlock the whole collective.
    for (size_t i = 0; i < n; ++i) {
        ggml_cuda_ar_push_launch_rank(ar, backends[i], i, tensors[i], exact, n_packs);
    }

    return true;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

ggml_cuda_ar_push * ggml_cuda_ar_push_init(const int * devices, size_t n) {
    GGML_UNUSED(devices);
    GGML_UNUSED(n);
    return nullptr;
}

void ggml_cuda_ar_push_free(ggml_cuda_ar_push * ar) {
    GGML_UNUSED(ar);
}

bool ggml_cuda_ar_push_allreduce(
        ggml_cuda_ar_push * ar,
        ggml_backend_t   * backends,
        ggml_tensor      ** tensors,
        bool               exact) {
    GGML_UNUSED(ar);
    GGML_UNUSED(backends);
    GGML_UNUSED(tensors);
    GGML_UNUSED(exact);
    return false;
}

bool ggml_cuda_ar_push_can(
        const ggml_cuda_ar_push * ar,
        const ggml_tensor       * t,
        bool                      exact) {
    GGML_UNUSED(ar);
    GGML_UNUSED(t);
    GGML_UNUSED(exact);
    return false;
}

bool ggml_cuda_ar_push_allreduce_rank(
        ggml_cuda_ar_push * ar,
        ggml_backend_t      backend,
        size_t              rank,
        ggml_tensor       * t,
        bool                exact) {
    GGML_UNUSED(ar);
    GGML_UNUSED(backend);
    GGML_UNUSED(rank);
    GGML_UNUSED(t);
    GGML_UNUSED(exact);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
