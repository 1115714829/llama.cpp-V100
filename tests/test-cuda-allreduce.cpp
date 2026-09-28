// Correctness and timing test for the CUDA AllReduce comm interface used by
// the meta backend for tensor parallelism.
//
// Only the public ggml-backend API is used, including the per-backend
// ggml_backend_comm_* proc addresses, so the same binary can run against
// different libggml-cuda builds (for example to time the NCCL fallback).
//
// Small aligned F32 tensors with a multiple of 8 elements are expected to take
// the P2P push path: the default proc transports f16 and accumulates in f32,
// so it must produce bitwise identical results on all ranks, but only to f16
// accuracy. The exact proc transports f32 instead. Larger tensors fall back to
// NCCL, which reduces large tensors in BF16 for the default proc, hence the
// looser relative tolerance there. The exact proc is optional: with an older
// library the exact part of the test is skipped.
//
// The chain timing records many per-rank push AllReduce calls into one CUDA
// graph per rank and replays the graph, which removes the host launch overhead
// and the rank-to-rank skew from the per-call time.
//
// With 6 devices whose topology splits into two cliques the default comm
// picks the hierarchical 3+3 kernel; to time the flat one against it, run
// this test built against the older library.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

struct ar_test {
    size_t n = 0;
    std::vector<ggml_backend_t> backends;
    void * comm = nullptr;
    ggml_backend_comm_free_t      comm_free      = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce_exact = nullptr; // optional

    // CUDA graph capture of the per-rank push AllReduce; all optional.
    ggml_backend_comm_allreduce_rank_t            comm_allreduce_rank = nullptr;
    ggml_backend_comm_allreduce_rank_capturable_t comm_allreduce_rank_capturable = nullptr;
    ggml_backend_capture_begin_t  capture_begin  = nullptr;
    ggml_backend_capture_end_t    capture_end    = nullptr;
    ggml_backend_capture_launch_t capture_launch = nullptr;
    ggml_backend_capture_free_t   capture_free   = nullptr;
};

// Per-case ggml contexts and their backend buffers, freed together.
struct ar_case {
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<ggml_tensor *>         tensors;

    ~ar_case() {
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
        for (ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
    }
};

static bool run_case(const ar_test & t, ggml_backend_comm_allreduce_tensor_t allreduce, bool exact,
                     int64_t ne, const std::vector<bool> & compute, uint32_t seed, const char * phase) {
    const size_t n = t.n;

    ar_case c;
    std::vector<std::vector<float>> host(n);

    for (size_t i = 0; i < n; ++i) {
        struct ggml_init_params params = {
            /* .mem_size   = */ 16 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: ggml_init failed (ne=%lld rank=%zu)\n", phase, (long long) ne, i);
            return false;
        }
        c.ctxs.push_back(ctx);

        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
        if (compute[i]) {
            tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
        }
        c.tensors.push_back(tensor);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, t.backends[i]);
        if (buf == nullptr) {
            fprintf(stderr, "%s: alloc failed (ne=%lld rank=%zu)\n", phase, (long long) ne, i);
            return false;
        }
        c.bufs.push_back(buf);

        std::mt19937 rng(seed + (uint32_t) i * 7919u);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        host[i].resize(ne);
        for (int64_t j = 0; j < ne; ++j) {
            host[i][j] = dist(rng);
        }
        ggml_backend_tensor_set(tensor, host[i].data(), 0, ne * sizeof(float));
    }

    // Double-precision reference sum of the compute ranks.
    std::vector<double> expected(ne, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (!compute[i]) {
            continue;
        }
        for (int64_t j = 0; j < ne; ++j) {
            expected[j] += (double) host[i][j];
        }
    }

    if (!allreduce(t.comm, c.tensors.data())) {
        fprintf(stderr, "%s: allreduce returned false (ne=%lld)\n", phase, (long long) ne);
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        ggml_backend_synchronize(t.backends[i]);
    }

    const size_t nbytes = (size_t) ne * sizeof(float);
    const bool f16_push = !exact && ne % 8 == 0 && nbytes <= 512 * 1024;

    std::vector<float> result(ne);
    std::vector<float> reference(ne);
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_tensor_get(c.tensors[i], result.data(), 0, nbytes);

        for (int64_t j = 0; j < ne; ++j) {
            const double diff = std::fabs((double) result[j] - expected[j]);
            // The exact proc must match the f64 reference up to f32 rounding.
            // The f16 push path converts its input (here in [-1, 1]) to f16,
            // which loses at most 4.9e-4 per element. The NCCL fallback uses
            // BF16 for large tensors, so it gets a relative tolerance.
            const bool good = exact ? diff < 1e-5 :
                (f16_push ? diff < 4e-3 :
                (diff < 1e-5 || diff / std::max(std::fabs(expected[j]), 1.0) < 2e-2));
            if (!good) {
                fprintf(stderr, "%s: ne=%lld rank=%zu elem=%lld got=%g want=%g\n",
                        phase, (long long) ne, i, (long long) j, (double) result[j], expected[j]);
                return false;
            }
        }

        if (i == 0) {
            reference = result;
        } else if (memcmp(result.data(), reference.data(), nbytes) != 0) {
            fprintf(stderr, "%s: ne=%lld rank=%zu result differs bitwise from rank 0\n",
                    phase, (long long) ne, i);
            return false;
        }
    }

    return true;
}

// The meta backend uses the exact collective to gather f32 token ids: the
// TOP_K candidates and the GET_ROWS rows. As f16, any id above 65504 turns
// into inf. Only rank 0 is nonzero here, so the result must be bitwise equal
// to its input. Sizes cover both the push path (512 KiB) and the fallback.
static bool run_case_exact_ids(const ar_test & t) {
    const size_t n = t.n;
    const int64_t sizes[] = { 131072, 250000 };

    for (int64_t ne : sizes) {
        ar_case c;
        std::vector<float> host(ne);
        std::vector<float> input(ne);

        for (size_t i = 0; i < n; ++i) {
            struct ggml_init_params params = {
                /* .mem_size   = */ 16 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            if (i == 0) {
                tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
            }
            c.ctxs.push_back(ctx);
            c.tensors.push_back(tensor);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, t.backends[i]);
            if (buf == nullptr) {
                fprintf(stderr, "exact ids: alloc failed (ne=%lld rank=%zu)\n", (long long) ne, i);
                return false;
            }
            c.bufs.push_back(buf);

            std::fill(host.begin(), host.end(), 0.0f);
            if (i == 0) {
                // Start above 65504 so every value would break as f16, and
                // stay inside the 0..250000 range the real ids live in.
                const int64_t base = ne == 131072 ? 100000 : 0;
                for (int64_t j = 0; j < ne; ++j) {
                    host[j] = (float) (base + j);
                }
                input = host;
            }
            ggml_backend_tensor_set(tensor, host.data(), 0, ne * sizeof(float));
        }

        std::vector<float> result(ne);

        if (!t.comm_allreduce_exact(t.comm, c.tensors.data())) {
            fprintf(stderr, "exact ids: allreduce returned false (ne=%lld)\n", (long long) ne);
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }

        for (size_t i = 0; i < n; ++i) {
            ggml_backend_tensor_get(c.tensors[i], result.data(), 0, ne * sizeof(float));
            if (memcmp(result.data(), input.data(), ne * sizeof(float)) != 0) {
                for (int64_t j = 0; j < ne; ++j) {
                    if (result[j] != input[j]) {
                        fprintf(stderr, "exact ids: ne=%lld rank=%zu elem=%lld got=%g want=%g\n",
                                (long long) ne, i, (long long) j, (double) result[j], (double) input[j]);
                        break;
                    }
                }
                return false;
            }
        }
    }

    return true;
}

static void run_timing(const ar_test & t, ggml_backend_comm_allreduce_tensor_t allreduce, const char * label) {
    const size_t n = t.n;
    const int64_t sizes[] = { 5120, 40960, 131072, 1310720, 10485760 };
    const int n_warmup = 50;
    const int n_iter   = 500;

    printf("timing %s (%d calls each)\n", label, n_iter);

    for (int64_t ne : sizes) {
        const size_t nbytes = (size_t) ne * sizeof(float);

        ar_case c;
        std::vector<float> zeros(ne, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            struct ggml_init_params params = {
                /* .mem_size   = */ 16 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, t.backends[i]);
            if (buf == nullptr) {
                c.ctxs.push_back(ctx);
                fprintf(stderr, "timing: alloc failed (ne=%lld rank=%zu)\n", (long long) ne, i);
                return;
            }
            c.ctxs.push_back(ctx);
            c.tensors.push_back(tensor);
            c.bufs.push_back(buf);
            ggml_backend_tensor_set(tensor, zeros.data(), 0, nbytes);
        }

        for (int it = 0; it < n_warmup; ++it) {
            allreduce(t.comm, c.tensors.data());
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < n_iter; ++it) {
            allreduce(t.comm, c.tensors.data());
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iter;
        printf("ne=%lld bytes=%lld us/call=%.2f\n", (long long) ne, (long long) nbytes, us);
    }
}

// Median per-call time of the default-proc allreduce over n_iter calls. All
// ranks are synchronized after every call, so a sample is one full round trip.
static double ar_median_us(const ar_test & t, void * comm,
                           ggml_backend_comm_allreduce_tensor_t allreduce, int64_t ne) {
    const size_t n = t.n;
    const size_t nbytes = (size_t) ne * sizeof(float);
    const int n_warmup = 20;
    const int n_iter   = 101;

    ar_case c;
    std::vector<float> zeros(ne, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        struct ggml_init_params params = {
            /* .mem_size   = */ 16 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
        tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, t.backends[i]);
        if (buf == nullptr) {
            c.ctxs.push_back(ctx);
            fprintf(stderr, "six: alloc failed (ne=%lld rank=%zu)\n", (long long) ne, i);
            return -1.0;
        }
        c.ctxs.push_back(ctx);
        c.tensors.push_back(tensor);
        c.bufs.push_back(buf);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, nbytes);
    }

    for (int it = 0; it < n_warmup; ++it) {
        allreduce(comm, c.tensors.data());
    }
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_synchronize(t.backends[i]);
    }

    std::vector<double> samples;
    samples.reserve(n_iter);
    for (int it = 0; it < n_iter; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        allreduce(comm, c.tensors.data());
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// On 6 devices whose topology splits into two cliques the default comm picks
// the hierarchical 3+3 kernel; the flat one is timed by running this test
// built against the older library. Shapes are the decode sizes of the
// tensor-parallel graph: [5120,8] and [5120,1] f32.
static bool run_six_rank_default(ar_test & t) {
    const int64_t sizes[] = { 5120 * 8, 5120 };
    const char * labels[] = { "5120x8", "5120x1" };

    bool ok = true;
    for (int s = 0; s < 2; ++s) {
        std::vector<bool> compute(t.n, true);
        const uint32_t seed = 0x6100u + (uint32_t) s;
        if (!run_case(t, t.comm_allreduce, false, sizes[s], compute, seed, "six-default")) {
            ok = false;
        }
    }

    for (int s = 0; s < 2; ++s) {
        const double us = ar_median_us(t, t.comm, t.comm_allreduce, sizes[s]);
        if (us < 0.0) {
            ok = false;
            continue;
        }
        printf("six-default %s ne=%lld median_us=%.2f\n", labels[s], (long long) sizes[s], us);
    }

    printf("six-default: %s\n", ok ? "OK" : "FAILED");
    return ok;
}

// Records n_per_graph per-rank push AllReduce calls of one size into one CUDA
// graph per rank, replays the graphs together and reports the per-call time.
// The push kernel keeps its state in device memory, so the replay needs no
// host-side bookkeeping. A skip is not a failure, a failed check is.
static bool run_chain_timing(const ar_test & t, bool exact) {
    if (t.comm_allreduce_rank == nullptr || t.comm_allreduce_rank_capturable == nullptr ||
        t.capture_begin == nullptr || t.capture_end == nullptr ||
        t.capture_launch == nullptr || t.capture_free == nullptr) {
        printf("chain: SKIP (no capture proc interface)\n");
        return true;
    }

    const size_t n = t.n;
    const int64_t sizes[] = { 5120, 20480, 40960, 81920 };
    const int n_per_graph = 128;
    const int n_warmup    = 5;
    const int n_replay    = 50;
    const int n_single    = 200;
    const int n_fill      = 20; // warmup allreduce calls before the capture

    printf("chain timing exact=%d (%d calls each)\n", (int) exact, n_replay * n_per_graph);

    bool ok = true;

    for (int64_t ne : sizes) {
        const size_t nbytes = (size_t) ne * sizeof(float);

        ar_case c;
        std::vector<std::vector<float>> host(n);
        for (size_t i = 0; i < n; ++i) {
            struct ggml_init_params params = {
                /* .mem_size   = */ 16 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, t.backends[i]);
            if (buf == nullptr) {
                c.ctxs.push_back(ctx);
                fprintf(stderr, "chain: alloc failed (ne=%lld rank=%zu)\n", (long long) ne, i);
                return false;
            }
            c.ctxs.push_back(ctx);
            c.tensors.push_back(tensor);
            c.bufs.push_back(buf);

            std::mt19937 rng(0xCAFEu + (uint32_t) ne + (uint32_t) i * 7919u);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            host[i].resize(ne);
            for (int64_t j = 0; j < ne; ++j) {
                host[i][j] = dist(rng);
            }
            ggml_backend_tensor_set(tensor, host[i].data(), 0, nbytes);
        }

        if (!t.comm_allreduce_rank_capturable(t.comm, c.tensors.data(), exact)) {
            printf("chain ne=%lld SKIP (not capturable)\n", (long long) ne);
            continue;
        }

        for (int it = 0; it < n_fill; ++it) {
            for (size_t i = 0; i < n; ++i) {
                t.comm_allreduce_rank(t.comm, i, c.tensors[i], exact);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }

        std::vector<void *> execs(n, nullptr);
        bool captured = true;
        for (size_t i = 0; i < n; ++i) {
            if (!t.capture_begin(t.backends[i])) {
                captured = false;
                break;
            }
            for (int k = 0; k < n_per_graph; ++k) {
                t.comm_allreduce_rank(t.comm, i, c.tensors[i], exact);
            }
            execs[i] = t.capture_end(t.backends[i]);
            if (execs[i] == nullptr) {
                captured = false;
                break;
            }
        }
        if (!captured) {
            for (void * exec : execs) {
                t.capture_free(exec);
            }
            printf("chain ne=%lld SKIP (capture failed)\n", (long long) ne);
            continue;
        }

        for (int rep = 0; rep < n_warmup; ++rep) {
            for (size_t i = 0; i < n; ++i) {
                t.capture_launch(t.backends[i], execs[i]);
            }
            for (size_t i = 0; i < n; ++i) {
                ggml_backend_synchronize(t.backends[i]);
            }
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < n_replay; ++rep) {
            for (size_t i = 0; i < n; ++i) {
                t.capture_launch(t.backends[i], execs[i]);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (n_replay * n_per_graph);
        printf("chain ne=%lld bytes=%lld exact=%d us/call=%.2f\n", (long long) ne, (long long) nbytes, (int) exact, us);

        for (void * exec : execs) {
            t.capture_free(exec);
        }

        // One-call graphs replayed with an idle gap each time: this number
        // includes the launch overhead, so it is only a reference point.
        std::vector<void *> singles(n, nullptr);
        bool single_captured = true;
        for (size_t i = 0; i < n; ++i) {
            if (!t.capture_begin(t.backends[i])) {
                single_captured = false;
                break;
            }
            t.comm_allreduce_rank(t.comm, i, c.tensors[i], exact);
            singles[i] = t.capture_end(t.backends[i]);
            if (singles[i] == nullptr) {
                single_captured = false;
                break;
            }
        }
        if (!single_captured) {
            for (void * exec : singles) {
                t.capture_free(exec);
            }
            printf("chain ne=%lld SKIP (single capture failed)\n", (long long) ne);
            continue;
        }

        double single_us = 0.0;
        for (int rep = 0; rep < n_single; ++rep) {
            const auto s0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < n; ++i) {
                t.capture_launch(t.backends[i], singles[i]);
            }
            for (size_t i = 0; i < n; ++i) {
                ggml_backend_synchronize(t.backends[i]);
            }
            const auto s1 = std::chrono::steady_clock::now();
            single_us += std::chrono::duration<double, std::micro>(s1 - s0).count();
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        printf("single ne=%lld exact=%d us/call=%.2f\n", (long long) ne, (int) exact, single_us / n_single);

        // The replayed tensors hold repeated sums, so reload the inputs and
        // replay one fresh one-call graph before comparing rank 0.
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_tensor_set(c.tensors[i], host[i].data(), 0, nbytes);
        }
        for (size_t i = 0; i < n; ++i) {
            t.capture_launch(t.backends[i], singles[i]);
        }
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_synchronize(t.backends[i]);
        }

        std::vector<float> result(ne);
        ggml_backend_tensor_get(c.tensors[0], result.data(), 0, nbytes);

        bool check = true;
        if (exact) {
            // The f32 kernel sums the ranks in rank order, so the reference does too.
            std::vector<float> ref(ne);
            for (int64_t j = 0; j < ne; ++j) {
                float sum = host[0][j];
                for (size_t i = 1; i < n; ++i) {
                    sum += host[i][j];
                }
                ref[j] = sum;
            }
            check = memcmp(result.data(), ref.data(), nbytes) == 0;
            if (!check) {
                for (int64_t j = 0; j < ne; ++j) {
                    if (memcmp(&result[j], &ref[j], sizeof(float)) != 0) {
                        fprintf(stderr, "chain: ne=%lld exact=1 elem=%lld got=%g want=%g\n",
                                (long long) ne, (long long) j, (double) result[j], (double) ref[j]);
                        break;
                    }
                }
            }
        } else {
            for (int64_t j = 0; j < ne; ++j) {
                double expected = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    expected += (double) host[i][j];
                }
                const double diff = std::fabs((double) result[j] - expected);
                if (diff / std::max(std::fabs(expected), 1.0) >= 1e-2) {
                    fprintf(stderr, "chain: ne=%lld exact=0 elem=%lld got=%g want=%g\n",
                            (long long) ne, (long long) j, (double) result[j], expected);
                    check = false;
                    break;
                }
            }
        }

        printf("chain ne=%lld check %s\n", (long long) ne, check ? "OK" : "FAILED");
        ok = ok && check;

        for (void * exec : singles) {
            t.capture_free(exec);
        }
    }

    return ok;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr) {
        printf("SKIP: CUDA backend not available\n");
        return 0;
    }

    ggml_backend_comm_init_t comm_init =
        (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    ggml_backend_comm_free_t comm_free =
        (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    ggml_backend_comm_allreduce_tensor_t comm_allreduce =
        (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    if (comm_init == nullptr || comm_free == nullptr || comm_allreduce == nullptr) {
        printf("SKIP: CUDA backend has no comm proc interface\n");
        return 0;
    }

    // Optional: older libraries have no exact variant, the exact part is then skipped.
    ggml_backend_comm_allreduce_tensor_t comm_allreduce_exact =
        (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor_exact");

    // Optional: CUDA graph capture of the per-rank push AllReduce, for the chain timing.
    ggml_backend_comm_allreduce_rank_t comm_allreduce_rank =
        (ggml_backend_comm_allreduce_rank_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_rank");
    ggml_backend_comm_allreduce_rank_capturable_t comm_allreduce_rank_capturable =
        (ggml_backend_comm_allreduce_rank_capturable_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_rank_capturable");
    ggml_backend_capture_begin_t capture_begin =
        (ggml_backend_capture_begin_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_begin");
    ggml_backend_capture_end_t capture_end =
        (ggml_backend_capture_end_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_end");
    ggml_backend_capture_launch_t capture_launch =
        (ggml_backend_capture_launch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_launch");
    ggml_backend_capture_free_t capture_free =
        (ggml_backend_capture_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_free");

    size_t n = ggml_backend_reg_dev_count(reg);
    if (n > 8) {
        n = 8;
    }
    if (n < 2) {
        printf("SKIP: need at least 2 CUDA devices, found %zu\n", n);
        return 0;
    }

    ar_test t;
    t.n = n;
    t.comm_free = comm_free;
    t.comm_allreduce = comm_allreduce;
    t.comm_allreduce_exact = comm_allreduce_exact;
    t.comm_allreduce_rank = comm_allreduce_rank;
    t.comm_allreduce_rank_capturable = comm_allreduce_rank_capturable;
    t.capture_begin = capture_begin;
    t.capture_end = capture_end;
    t.capture_launch = capture_launch;
    t.capture_free = capture_free;

    bool backends_ok = true;
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, i), nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "failed to init CUDA backend %zu\n", i);
            backends_ok = false;
            break;
        }
        t.backends.push_back(backend);
    }

    if (backends_ok) {
        t.comm = comm_init(t.backends.data(), t.backends.size());
        if (t.comm == nullptr) {
            printf("SKIP: CUDA comm init failed\n");
            backends_ok = false;
        }
    }

    int n_failed = 0;

    if (backends_ok) {
        printf("testing AllReduce on %zu CUDA devices%s\n", n,
               t.comm_allreduce_exact != nullptr ? ", with exact proc" : ", without exact proc");

        const int64_t sizes[] = { 4, 1280, 5120, 5124, 40960, 81920, 131072, 131076, 3, 5122, 2621440 };
        for (int64_t ne : sizes) {
            std::mt19937 rng(0xC0FFEEu + (uint32_t) ne);
            std::bernoulli_distribution keep(0.75);
            std::vector<bool> compute(n);
            for (size_t i = 0; i < n; ++i) {
                compute[i] = keep(rng);
            }
            if (!run_case(t, t.comm_allreduce, false, ne, compute, 0x5EEDu + (uint32_t) ne, "single")) {
                n_failed++;
            }
            if (t.comm_allreduce_exact != nullptr &&
                !run_case(t, t.comm_allreduce_exact, true, ne, compute, 0x5EEDu + (uint32_t) ne, "single-exact")) {
                n_failed++;
            }
        }
        printf("single sizes: %s\n", n_failed == 0 ? "OK" : "FAILED");
    }

    if (backends_ok && n_failed == 0) {
        // Alternate random sizes, COMPUTE flags and procs: the per-block
        // epochs and the shared slot buffers must stay correct across
        // back-to-back calls of both kernels.
        const int64_t stress_sizes[] = { 4, 1280, 5120, 40960, 81920, 131072 };
        std::mt19937 rng(1234567u);
        std::uniform_int_distribution<size_t> pick(0, sizeof(stress_sizes) / sizeof(stress_sizes[0]) - 1);
        std::bernoulli_distribution keep(0.75);
        std::bernoulli_distribution use_exact(0.5);
        bool ok = true;
        for (int iter = 0; iter < 300 && ok; ++iter) {
            const int64_t ne = stress_sizes[pick(rng)];
            std::vector<bool> compute(n);
            for (size_t i = 0; i < n; ++i) {
                compute[i] = keep(rng);
            }
            const bool exact = t.comm_allreduce_exact != nullptr && use_exact(rng);
            ggml_backend_comm_allreduce_tensor_t allreduce = exact ? t.comm_allreduce_exact : t.comm_allreduce;
            if (!run_case(t, allreduce, exact, ne, compute, 0xBEEFu + (uint32_t) iter, exact ? "stress-exact" : "stress")) {
                fprintf(stderr, "stress iteration %d failed (ne=%lld exact=%d)\n", iter, (long long) ne, (int) exact);
                ok = false;
            }
        }
        if (!ok) {
            n_failed++;
        }
        printf("stress: %s\n", ok ? "OK (300 calls)" : "FAILED");
    }

    if (backends_ok && n_failed == 0 && t.comm_allreduce_exact != nullptr) {
        const bool ok = run_case_exact_ids(t);
        if (!ok) {
            n_failed++;
        }
        printf("exact ids: %s\n", ok ? "OK" : "FAILED");
    }

    if (backends_ok && n_failed == 0) {
        run_timing(t, t.comm_allreduce, t.comm_allreduce_exact != nullptr ? "default proc" : "default proc (no exact proc)");
        if (t.comm_allreduce_exact != nullptr) {
            run_timing(t, t.comm_allreduce_exact, "exact proc");
        }
        if (n == 6 && !run_six_rank_default(t)) {
            n_failed++;
        }
        if (!run_chain_timing(t, false)) {
            n_failed++;
        }
        if (t.comm_allreduce_exact != nullptr && !run_chain_timing(t, true)) {
            n_failed++;
        }
    }

    if (t.comm != nullptr) {
        comm_free(t.comm);
    }
    for (ggml_backend_t backend : t.backends) {
        ggml_backend_free(backend);
    }

    if (backends_ok && n_failed == 0) {
        printf("ALL_OK\n");
        return 0;
    }

    return 1;
}
