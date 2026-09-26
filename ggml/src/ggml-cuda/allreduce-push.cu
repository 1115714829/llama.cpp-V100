#include "allreduce-push.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "ggml-impl.h"

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

template <int nranks>
static void ggml_cuda_ar_push_launch(
        const ggml_cuda_ar_push_ptrs & bufs, const float * input, float * output,
        const int rank, const int n_packs, const bool exact, cudaStream_t stream) {
    if (exact) {
        ggml_cuda_ar_push_f32<nranks><<<GGML_CUDA_AR_PUSH_BLOCKS, GGML_CUDA_AR_PUSH_THREADS, 0, stream>>>(
            bufs, input, output, rank, n_packs);
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

    GGML_LOG_INFO("%s: P2P push AllReduce enabled for %zu GPUs, up to %zu KiB per call\n",
                  __func__, n, GGML_CUDA_AR_PUSH_MAX_BYTES >> 10);

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

    for (size_t i = 0; i < n; ++i) {
        if (tensors[i] == nullptr ||
            tensors[i]->type != GGML_TYPE_F32 ||
            ggml_nelements(tensors[i]) != ne ||
            !ggml_is_contiguously_allocated(tensors[i]) ||
            ((uintptr_t) tensors[i]->data & 0xF) != 0) {
            return false;
        }
    }

    const size_t nbytes = (size_t) ne * sizeof(float);
    if (nbytes > GGML_CUDA_AR_PUSH_MAX_BYTES) {
        return false;
    }

    int n_packs;
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

    ggml_cuda_ar_push_ptrs bufs = {};
    for (size_t i = 0; i < n; ++i) {
        bufs.ptrs[i] = ar->bufs[i];
    }

    // No synchronization in this loop: the kernel spins on the peer ranks, so
    // waiting for one rank here would deadlock the whole collective.
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_cuda_context * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == ar->devices[i]);
        ggml_cuda_set_device(cuda_ctx->device);

        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, nbytes, cuda_ctx->stream()));
        }

        const float * input  = static_cast<const float *>(tensors[i]->data);
        float       * output = static_cast<float *>(tensors[i]->data);

        switch (n) {
            case 2: ggml_cuda_ar_push_launch<2>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 3: ggml_cuda_ar_push_launch<3>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 4: ggml_cuda_ar_push_launch<4>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 5: ggml_cuda_ar_push_launch<5>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 6: ggml_cuda_ar_push_launch<6>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 7: ggml_cuda_ar_push_launch<7>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            case 8: ggml_cuda_ar_push_launch<8>(bufs, input, output, (int) i, n_packs, exact, cuda_ctx->stream()); break;
            default: GGML_ABORT("unsupported number of ranks: %zu", n);
        }
        CUDA_CHECK(cudaGetLastError());
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

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
