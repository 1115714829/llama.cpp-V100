#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

// TOP_K over an AXIS_0 (class-sharded) input: each backend only holds a slice of the classes,
// the per-shard candidates are merged across backends during graph compute.
static bool ggml_backend_meta_topk_collective(enum ggml_backend_meta_split_axis src0_axis, enum ggml_type src0_type) {
    return src0_axis == GGML_BACKEND_SPLIT_AXIS_0 && src0_type == GGML_TYPE_F32;
}

// GET_ROWS from an AXIS_1 (row-sharded) input with replicated ids: each row is present on one
// backend only, the rows are collected across backends during graph compute.
static bool ggml_backend_meta_getrows_collective(enum ggml_backend_meta_split_axis src0_axis, enum ggml_backend_meta_split_axis src1_axis, enum ggml_type type) {
    return src0_axis == GGML_BACKEND_SPLIT_AXIS_1 && src1_axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && type == GGML_TYPE_F32;
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

// one event per simple device, a meta event records/waits on all of them
struct ggml_backend_meta_event_context {
    std::vector<ggml_backend_event_t> simple_events;
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // meta events are implemented, but the meta device does not advertise them.
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_meta_event_context * ev_ctx = new ggml_backend_meta_event_context;
    ev_ctx->simple_events.reserve(meta_dev_ctx->simple_devs.size());
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_event_t simple_event = ggml_backend_event_new(simple_dev);
        if (simple_event == nullptr) {
            for (ggml_backend_event_t prev : ev_ctx->simple_events) {
                ggml_backend_event_free(prev);
            }
            delete ev_ctx;
            return nullptr;
        }
        ev_ctx->simple_events.push_back(simple_event);
    }

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev_ctx,
    };
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_meta_event_context * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t simple_event : ev_ctx->simple_events) {
        ggml_backend_event_free(simple_event);
    }
    delete ev_ctx;
    delete event;
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    const ggml_backend_meta_event_context * ev_ctx = (const ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t simple_event : ev_ctx->simple_events) {
        ggml_backend_event_synchronize(simple_event);
    }
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

// Number of plan slots. Decode plans stay alive while prefill/checkpoint shapes rotate through the
// remaining slots.
static constexpr int GGML_META_N_PLANS = 6;

// Number of rotating "compute" containers: one per plan plus two that are being filled by the
// next graph allocations, so that a rebuild never evicts the views of all plans at once.
static constexpr int GGML_META_N_STC = GGML_META_N_PLANS + 2;

// One speculative round uses a plan for the target verification, the injection and the draft block, so
// plans and containers used by the last 3 computations are in flight and must not be evicted.
static constexpr uint64_t GGML_META_RECENT_TICKS = 3;

// use order for plan.last_use and stc_last_use, shared by all meta backend instances
static std::atomic<uint64_t> ggml_backend_meta_tick{0};

// UIDs of the plans that hold captured CUDA graphs (any meta backend instance): their containers are
// recycled last, so that the decode plans survive the prefill graphs of the next request
static std::mutex                   ggml_backend_meta_hot_mutex;
static std::unordered_set<uint64_t> ggml_backend_meta_hot_uids;

static void ggml_backend_meta_hot_add(uint64_t uid) {
    std::lock_guard<std::mutex> lock(ggml_backend_meta_hot_mutex);
    ggml_backend_meta_hot_uids.insert(uid);
}

static void ggml_backend_meta_hot_remove(uint64_t uid) {
    std::lock_guard<std::mutex> lock(ggml_backend_meta_hot_mutex);
    ggml_backend_meta_hot_uids.erase(uid);
}

static bool ggml_backend_meta_hot_has(uint64_t uid) {
    std::lock_guard<std::mutex> lock(ggml_backend_meta_hot_mutex);
    return uid != 0 && ggml_backend_meta_hot_uids.count(uid) != 0;
}

struct ggml_backend_meta_split_state_cache_hash {
    size_t operator()(const std::pair<const ggml_tensor *, bool> & key) const {
        // the low address bits of a tensor are always zero, mix in the higher bits
        const size_t h = std::hash<const ggml_tensor *>()(key.first) >> 4;
        return h ^ (key.second ? size_t(0x9e3779b97f4a7c15ULL) : 0);
    }
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 3 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[GGML_META_N_STC];
    uint64_t stc_owner_uid[GGML_META_N_STC] = {}; // plan UID owning each container, 0 = free
    uint64_t stc_last_use[GGML_META_N_STC] = {};  // use order, for choosing the container to recycle
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    struct split_state_cache_entry {
        ggml_backend_meta_split_state split_state;
        // for views of a tensor with explicit offsets: per-device byte offset of the view data
        // inside the source's simple tensor
        std::vector<size_t> local_offs;
        char tensor_copy[nbtc];
    };
    std::unordered_map<std::pair<const ggml_tensor *, bool>,
                       split_state_cache_entry,
                       ggml_backend_meta_split_state_cache_hash> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container * stc_compute_in,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)) {
        for (int i = 0; i < GGML_META_N_STC; i++) {
            stc_compute[i] = std::move(stc_compute_in[i]);
        }
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor);

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        if (tensor->data == nullptr) {
            return nullptr;
        }
        // the graph may be reused without reallocation, in which case the container for the current
        // index may not hold this tensor yet; rebuild it here
        ggml_backend_meta_simple_tensor_container & stc_compute = buf_ctx->stc_compute[buf_ctx->stc_compute_index];
        ggml_backend_meta_buffer_init_tensor_impl(stc_compute, const_cast<ggml_tensor *>(tensor));
        it = stc_compute.simple_tensors.find(tensor);
        if (it == stc_compute.simple_tensors.end()) {
            return nullptr;
        }
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

// A PAD node that evens out an uneven AXIS_0 split (see handle_pad): every backend pads its own
// slice up to node->ne[0]/n_bufs, so its per-device pad amount differs from the global one.
static bool ggml_backend_meta_pad_uneven(const struct ggml_tensor * node, const struct ggml_backend_meta_split_state & src_ss) {
    if (node->op != GGML_OP_PAD || node->type != GGML_TYPE_F32 || node->src[0] == nullptr ||
            node->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (src_ss.axis != GGML_BACKEND_SPLIT_AXIS_0 || src_ss.has_off || src_ss.n_segments != 1 || src_ss.nr[0] != 1) {
        return false;
    }
    // the per-device aux graph fills the end of the split axis only
    const int32_t * pad_params = (const int32_t *) node->op_params;
    if (pad_params[0] != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (pad_params[2*i + 0] != 0 || pad_params[2*i + 1] != 0) {
            return false;
        }
    }
    if (node->buffer == nullptr || !ggml_backend_buffer_is_meta(node->buffer)) {
        return false;
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(node->buffer);
    if (n_bufs < 2 || node->ne[0] % (int64_t) n_bufs != 0) {
        return false;
    }
    const int64_t ne_shard = node->ne[0] / (int64_t) n_bufs;
    bool uneven = false;
    for (size_t j = 0; j < n_bufs; j++) {
        if (src_ss.ne[j] <= 0 || src_ss.ne[j] > ne_shard) {
            return false;
        }
        uneven = uneven || src_ss.ne[j] != src_ss.ne[0];
    }
    return uneven;
}

// Derive the split state of a tensor from a source with explicit offsets: the window of the tensor
// along the source split axis is intersected with the per-device source slices and scaled to the
// units of the tensor's split axis. local_offs receives the per-device byte offset of the tensor
// data inside the source's simple tensor.
static size_t ggml_backend_meta_get_split_state_local_offs(const struct ggml_tensor * tensor, size_t j);

static ggml_backend_meta_split_state ggml_backend_meta_map_has_off_state(
        const ggml_tensor * tensor, const ggml_tensor * src, const ggml_backend_meta_split_state & src_ss,
        int split_axis, std::vector<size_t> & local_offs, size_t n_bufs) {
    GGML_ASSERT(src_ss.has_off);
    GGML_ASSERT(src_ss.nr[0] == 1);
    GGML_ASSERT(split_axis >= 0 && split_axis < GGML_MAX_DIMS);
    GGML_ASSERT(src_ss.axis >= 0 && src_ss.axis < GGML_MAX_DIMS);

    const int B = (int) src_ss.axis;
    const int A = split_axis;
    // one unit (element) of the source split axis covers ne[A]/ne[B] units of the tensor split axis
    const int64_t scale_num = tensor->ne[A];
    const int64_t scale_den = src->ne[B];
    GGML_ASSERT(scale_num > 0 && scale_den > 0);

    // scale a source offset/length to the units of the tensor split axis
    auto to_tensor_units = [&](int64_t x) -> int64_t {
        if ((x * scale_num) % scale_den != 0) {
            GGML_LOG_ERROR("%s: %s [%s] from %s [%s]: axis %d from source axis %d, %lld * %lld / %lld is not whole\n",
                __func__, tensor->name, ggml_op_name(tensor->op), src->name, ggml_op_name(src->op), A, B,
                (long long) x, (long long) scale_num, (long long) scale_den);
            GGML_ABORT("split state of a tensor derived from explicit offsets does not map to whole units");
        }
        return x * scale_num / scale_den;
    };

    ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(A), {0}, {1}, 1, {0}, true};
    local_offs.assign(n_bufs, 0);

    // A view of a view has the base tensor as view_src and an offset relative to it: work with the
    // offset relative to the source view, and place the local offsets relative to the base, i.e.
    // after the source view's own local offset.
    const bool is_view = tensor->op == GGML_OP_VIEW && tensor->view_src != nullptr &&
                         (tensor->view_src == src || src->view_src == tensor->view_src);
    // Any view-like op (view, permute, reshape of a view) whose source is itself a view shares the
    // source's base: its per-device data sits after the source's own local offset.
    const bool chained = tensor->view_src != nullptr && tensor->view_src != src && src->view_src == tensor->view_src;
    const size_t view_offs_rel = chained ? tensor->view_offs - src->view_offs : tensor->view_offs;
    auto src_local = [&](size_t j) -> size_t {
        return chained ? ggml_backend_meta_get_split_state_local_offs(src, j) : 0;
    };
    if (is_view && A != B) {
        // The view remaps the source split axis to a different tensor axis (e.g. a fused q+gate head
        // or a merge of several heads): rescale the units, the offsets stay inside the source slice.
        GGML_ASSERT(src_ss.n_segments == 1);
        for (size_t j = 0; j < n_bufs; j++) {
            ret.ne[j]  = to_tensor_units(src_ss.ne[j]);
            ret.off[j] = to_tensor_units(src_ss.off[j]);
            local_offs[j] = src_local(j) + view_offs_rel; // interior offsets are not rescaled
        }
        return ret;
    }

    // The tensor sees a window of the source, cut it at the device boundaries:
    // reshape/cpy/permute cover the full source axis, views start at their byte offset.
    int64_t w0 = 0;
    int64_t w1 = src->ne[B];
    if (is_view) {
        GGML_ASSERT(A == B);
        GGML_ASSERT(view_offs_rel % src->nb[B] == 0);
        w0 = view_offs_rel / src->nb[B];
        w1 = w0 + tensor->ne[A];
    }
    GGML_ASSERT(w0 >= 0 && w1 <= src->ne[B]);

    for (size_t j = 0; j < n_bufs; j++) {
        int64_t ne_before = 0;
        bool found = false;
        for (size_t s = 0; s < src_ss.n_segments; s++) {
            const int64_t off_s = src_ss.off[s*n_bufs + j];
            const int64_t ne_s  = src_ss.ne[s*n_bufs + j];
            const int64_t is = std::max(w0, off_s);
            const int64_t ie = std::min(w1, off_s + ne_s);
            if (ie > is) {
                GGML_ASSERT(!found); // at most one segment per device may intersect the window
                found = true;
                if (is_view) {
                    // views share the axis units of the source, no conversion is needed
                    ret.off[j] = is - w0;
                    ret.ne[j]  = ie - is;
                } else {
                    ret.off[j] = to_tensor_units(is - w0);
                    ret.ne[j]  = to_tensor_units(ie - is);
                }
                local_offs[j] = src_local(j) + ((is - off_s) + ne_before) * src->nb[B];
            }
            ne_before += ne_s;
        }
    }
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    // per-device local byte offsets of this tensor inside its source's simple tensor, for views
    // whose split state is derived from a source with explicit offsets
    std::vector<size_t> local_offs;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.has_off != b.has_off) {
            return false;
        }
        if (a.has_off) {
            if (a.n_segments != b.n_segments || a.axis != b.axis) {
                return false;
            }
            for (size_t s = 0; s < a.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    if (a.ne[s*n_bufs + j] != b.ne[s*n_bufs + j] || a.off[s*n_bufs + j] != b.off[s*n_bufs + j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1, {0}, false};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                GGML_LOG_ERROR("%s: %s [%s] src%zu %s [%s]: axis %s%s\n", __func__, tensor->name, ggml_op_name(tensor->op), i,
                    tensor->src[i]->name, ggml_op_name(tensor->src[i]->op),
                    ggml_backend_meta_split_axis_name(src_ss[i].axis), src_ss[i].has_off ? " (explicit offsets)" : "");
            }
            GGML_ABORT("no common split state for %s [%s]", tensor->name, ggml_op_name(tensor->op));
        }
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        if (src_ss[0].axis == concat_axis && src_ss[1].axis == concat_axis &&
                src_ss[0].has_off && src_ss[1].has_off) {
            // both sources are split along the concat axis with explicit offsets
            GGML_ASSERT(src_ss[0].n_segments == 1);
            GGML_ASSERT(src_ss[1].n_segments == 1);
            const int64_t global_a = tensor->src[0]->ne[concat_axis];
            ggml_backend_meta_split_state ret = {concat_axis, {0}, {1}, 1, {0}, true};
            for (size_t j = 0; j < n_bufs; j++) {
                const int64_t ne_a = src_ss[0].ne[j];
                const int64_t ne_b = src_ss[1].ne[j];
                ret.ne[j] = ne_a + ne_b;
                if (ne_a > 0) {
                    if (ne_b > 0) {
                        GGML_ASSERT(src_ss[0].off[j] + ne_a == global_a);
                    }
                    ret.off[j] = src_ss[0].off[j];
                } else if (ne_b > 0) {
                    ret.off[j] = global_a + src_ss[1].off[j];
                }
            }
            return ret;
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            if (ret.has_off) {
                // explicit offsets must be kept, the per-device slices may overlap and cannot be folded
                for (size_t s = 0; s < ret.n_segments; s++) {
                    ret.nr[s] = 1;
                }
            } else {
                ret.nr[0] = 1;
                ret.n_segments = 1;
            }
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1, {0}, false};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1, {0}, false};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, {0}, false};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1, {0}, false};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1, {0}, false};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1, {0}, false};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1, {0}, false};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1, {0}, false};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (ggml_backend_meta_getrows_collective(src_ss[0].axis, src_ss[1].axis, tensor->type)) {
            // rows spread across backends, collected during graph compute
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
        }
        // batched gather with the batch dimension split over the backends and the ids split alike
        // (ids dim d-1 indexes src0 dim d): every backend gathers from its own batches
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == src_ss[0].axis - 1 && src_ss[0].n_segments == 1 && src_ss[1].n_segments == 1) {
            bool same_split = true;
            for (size_t j = 0; j < n_bufs; j++) {
                same_split = same_split && src_ss[0].ne[j]*src_ss[0].nr[0] == src_ss[1].ne[j]*src_ss[1].nr[0];
            }
            if (same_split) {
                return src_ss[0];
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        if (!split_states_equal(src_ss[0], src_ss[2])) {
            auto ss_str = [&](const ggml_backend_meta_split_state & ss) {
                std::string str = std::string(ggml_backend_meta_split_axis_name(ss.axis)) + (ss.has_off ? " off" : "") +
                    " segs=" + std::to_string(ss.n_segments) + " {";
                for (size_t s = 0; s < ss.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        str += (s + j > 0 ? ", " : "") + std::to_string(ss.ne[s*n_bufs + j]) +
                            (ss.has_off ? "@" + std::to_string(ss.off[s*n_bufs + j]) : "x" + std::to_string(ss.nr[s]));
                    }
                }
                return str + "}";
            };
            GGML_LOG_ERROR("%s: SET_ROWS %s: split state of %s [%s] differs from %s [%s]\n", __func__, tensor->name,
                tensor->src[0]->name, ss_str(src_ss[0]).c_str(), tensor->src[2]->name, ss_str(src_ss[2]).c_str());
            GGML_ABORT("SET_ROWS source and destination split differently");
        }
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    // set by a handler that already filled the per-device layout itself, the ratio backfill then
    // has to keep its hands off the state (see handle_pad)
    bool split_state_fixed = false;

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_backend_meta_pad_uneven(tensor, src_ss[0])) {
            // every backend pads its own slice: the output is an even AXIS_0 split of the padded length
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
            ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, {0}, false};
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[j] = tensor->ne[0] / (int64_t) n_bufs;
            }
            split_state_fixed = true;
            return ret;
        }
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].has_off) {
            // Q is split into head windows (heads on axis 2 of the permuted Q, as in the path below):
            // the output, heads on axis 1, follows the Q window state
            GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
            return ret;
        }

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, {0}, false};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, {0}, false};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, {0}, false};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, {0}, false};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            GGML_ASSERT(ret.n_segments <= GGML_BACKEND_META_MAX_SEGMENTS);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                if (ret.has_off) {
                    // explicit source offsets allow the slices to overlap, so they only need to stay in bounds
                    for (size_t s = 0; s < ret.n_segments; s++) {
                        GGML_ASSERT(ret.nr[s] == 1);
                        for (size_t j = 0; j < n_bufs; j++) {
                            GGML_ASSERT(ret.off[s*n_bufs + j] >= 0);
                            GGML_ASSERT(ret.off[s*n_bufs + j] % granularity == 0);
                            GGML_ASSERT(ret.ne[s*n_bufs + j]  % granularity == 0);
                            GGML_ASSERT(ret.off[s*n_bufs + j] + ret.ne[s*n_bufs + j] <= tensor->ne[ret.axis]);
                        }
                    }
                } else {
                    int64_t ne_sum = 0;
                    for (size_t s = 0; s < ret.n_segments; s++) {
                        for (size_t j = 0; j < n_bufs; j++) {
                            GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                            ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                        }
                    }
                    GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
                }
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1, {0}, false});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_TOP_K: {
                if (ggml_backend_meta_topk_collective(src_ss[0].axis, tensor->src[0]->type)) {
                    // classes spread across backends, merged during graph compute
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, {0}, false};
                } else {
                    split_state = handle_per_row(src_ss);
                }
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
            } break;
        }
        if (!split_state_fixed && split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            bool has_off_src = false;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (src_ss[i].has_off) {
                    if (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_CONT ||
                            tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_PERMUTE) {
                        split_state = ggml_backend_meta_map_has_off_state(tensor, tensor->src[i], src_ss[i],
                            split_state.axis, local_offs, n_bufs);
                    }
                    has_off_src = true;
                    first_src_split_by_axis = false;
                    continue;
                }
                if (has_off_src) {
                    // the state derived from the source with explicit offsets already has the per-device ne/off
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.tensor_copy, (const char *) tensor, sizeof(it->second.tensor_copy)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    ggml_backend_meta_split_state ret;
    if (it == buf_ctx->split_state_cache.end()) {
        const ggml_backend_meta_split_state split_state = calculate_split_state();
        // calculate_split_state() can recurse into this function and insert entries, so insert afterwards
        it = buf_ctx->split_state_cache.try_emplace(key).first;
        it->second.split_state = split_state;
        it->second.local_offs  = std::move(local_offs);
        memcpy(it->second.tensor_copy, tensor, sizeof(it->second.tensor_copy));
        ret = split_state;
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                if (split_state.has_off) {
                    for (size_t s = 0; s < split_state.n_segments; s++) {
                        for (size_t j = 0; j < n_bufs; j++) {
                            if (!ne_info.empty()) {
                                ne_info += ", ";
                            }
                            ne_info += std::to_string(split_state.ne[s*n_bufs + j]) + "@" + std::to_string(split_state.off[s*n_bufs + j]);
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        if (!ne_info.empty()) {
                            ne_info += ", ";
                        }
                        ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                    }
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = ret;
                if (ss.has_off) {
                    ne_info += std::to_string(ss.ne[j]) + "@" + std::to_string(ss.off[j]);
                } else {
                    ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
                }
            }
            std::string local_info;
            for (size_t j = 0; j < it->second.local_offs.size(); j++) {
                local_info += (j > 0 ? ", " : " local {") + std::to_string(it->second.local_offs[j]);
            }
            if (!local_info.empty()) {
                local_info += "}";
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]%s\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(ret.axis), ne_info.c_str(), local_info.c_str());
        }
    } else {
        ret = it->second.split_state;
    }

    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        if (ret.has_off) {
            // explicit source offsets allow the slices to overlap, so they only need to stay in bounds
            for (size_t s = 0; s < ret.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    assert(ret.off[s*n_bufs + j] >= 0);
                    assert(ret.off[s*n_bufs + j] + ret.ne[s*n_bufs + j] <= tensor->ne[int(ret.axis)]);
                }
            }
        } else {
            int64_t ne_ret = 0;
            for (size_t s = 0; s < ret.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
                }
            }
            assert(ne_ret == tensor->ne[int(ret.axis)]);
        }
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

// Local byte offset of the tensor data inside its source's simple tensor, computed together with the
// split state of the tensor (see ggml_backend_meta_get_split_state).
static size_t ggml_backend_meta_get_split_state_local_offs(const struct ggml_tensor * tensor, size_t j) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const auto it = buf_ctx->split_state_cache.find(std::make_pair(tensor, true));
    GGML_ASSERT(it != buf_ctx->split_state_cache.end());
    if (j >= it->second.local_offs.size()) {
        // ops that view a whole tensor (e.g. SET_ROWS) carry no derived per-device offsets
        return tensor->view_offs;
    }
    return it->second.local_offs[j];
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= GGML_BACKEND_META_MAX_SEGMENTS);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    // For views of a tensor with explicit offsets the view data is a slice of the source's per-device
    // simple tensor, so the per-device strides must follow the source layout.
    const bool view_src_is_meta = tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer);
    ggml_backend_meta_split_state split_state_view_src = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, {0}, false};
    if (view_src_is_meta && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
        split_state_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true);
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            const ggml_tensor * view_src_simple = nullptr;
            if (split_state_view_src.has_off) {
                view_src_simple = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
                if (view_src_simple == nullptr) {
                    GGML_ABORT("fatal error: %s: view source %s has no simple tensor on device %zu\n",
                        tensor->name, tensor->view_src->name, j);
                }
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    if (view_src_simple != nullptr) {
                        // The view is a slice of the source's simple tensor: find the source dim the
                        // stride was created from and scale by its simple/global ratio on this device.
                        int dim_src = -1;
                        for (int d = 0; d < GGML_MAX_DIMS; d++) {
                            if (tensor->view_src->nb[d] == tensor->nb[i]) {
                                dim_src = d;
                                break;
                            }
                        }
                        if (dim_src < 0) {
                            GGML_ABORT("fatal error: %s: no source dimension matches nb[%d] = %zu\n",
                                tensor->name, i, tensor->nb[i]);
                        }
                        GGML_ASSERT((tensor->nb[i] * view_src_simple->nb[dim_src]) % tensor->view_src->nb[dim_src] == 0);
                        nb[i] = tensor->nb[i] * view_src_simple->nb[dim_src] / tensor->view_src->nb[dim_src];
                    } else {
                        nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                    }
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        if (tensor->op == GGML_OP_FLASH_ATTN_EXT && ggml_get_op_params_i32(tensor, 5) < 0) {
            // 3-card attention split: derive the per-device Q head boundary from the Q/K head windows
            const ggml_tensor * q_src = tensor->src[0];
            const ggml_tensor * k_src = tensor->src[1];
            GGML_ASSERT(q_src != nullptr && k_src != nullptr);

            const ggml_backend_meta_split_state q_ss = ggml_backend_meta_get_split_state(stc, q_src, /*assume_sync =*/ true);
            const ggml_backend_meta_split_state k_ss = ggml_backend_meta_get_split_state(stc, k_src, /*assume_sync =*/ true);
            GGML_ASSERT(q_ss.has_off && k_ss.has_off);
            GGML_ASSERT(q_ss.axis == GGML_BACKEND_SPLIT_AXIS_2 && k_ss.axis == GGML_BACKEND_SPLIT_AXIS_2);
            GGML_ASSERT(q_ss.n_segments == 1 && k_ss.n_segments == 1 && q_ss.nr[0] == 1 && k_ss.nr[0] == 1);

            const int64_t n_gqa = q_src->ne[2] / k_src->ne[2];
            const int64_t q0    = q_ss.off[j];
            const int64_t kv0   = k_ss.off[j];

            int64_t g0 = (kv0 + 1)*n_gqa - q0;
            if (g0 < 0 || g0 >= q_ss.ne[j]) {
                // the device uses a single KV head, uniform GQA selects the same one
                g0 = 0;
            } else {
                // both groups must be non-empty and fit the HEADS=6 slot layout
                GGML_ASSERT(g0 <= 6 && q_ss.ne[j] - g0 <= 6);
            }
            ggml_set_op_params_i32(t_ij, 5, (int32_t) g0);
        }
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                if (split_state_view_src.has_off) {
                    // with explicit offsets the view data sits at its per-device local byte offset;
                    // interior offsets (e.g. the gate half of a fused q+gate head) are not rescaled
                    ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true); // ensure the cached local offsets
                    t_ij->view_offs = ggml_backend_meta_get_split_state_local_offs(tensor, j);
                } else if (t_ij->view_offs > 0) {
                    GGML_ASSERT(tensor->ne[split_dim] != 0);
                    const int split_dim_view_src = split_state_view_src.axis;
                    GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                    // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                    // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                    bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                    for (int i = 0; i < GGML_MAX_DIMS; i++) {
                        const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                        if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                            split_internal_offset = true;
                            break;
                        }
                    }
                    if (!split_internal_offset) {
                        t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                    }
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            if (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation. A concatenation still
    // produces the other source's slice (the 3-card attention concatenates per-device head windows,
    // most of them empty on a given device), so it is only skipped where its own slice is empty.
    if (tensor->op == GGML_OP_CONCAT) {
        for (size_t j = 0; j < n_simple_bufs; j++) {
            if (ggml_nelements(simple_tensors[j]) == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
        stc.simple_tensors.insert_or_assign(tensor, std::move(simple_tensors));
        return GGML_STATUS_SUCCESS;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors.insert_or_assign(tensor, std::move(simple_tensors));

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    // the segment loops below only address the per-device slices, so they also handle
    // states with explicit source offsets (the offsets are only needed to move data)
    if (split_state.has_off || split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.has_off) {
        // every slice is read from the source at its explicit offset, so the slices may overlap
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
        std::vector<size_t> simple_offsets(n_bufs, 0);

        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const int64_t ne = split_state.ne[s*n_bufs + j];
                    GGML_ASSERT(ne % blck_size == 0);
                    const size_t nbytes     = ne/blck_size * tensor->nb[0];
                    const size_t src_offset = split_state.off[s*n_bufs + j]/blck_size * tensor->nb[0];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + src_offset,
                        simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                        row_count, simple_tensor->nb[1], tensor->nb[1]);
                    simple_offsets[j] += nbytes;
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const int64_t ne = split_state.ne[s*n_bufs + j];
                const size_t nbytes     = ne * tensor->nb[1];
                const size_t src_offset = split_state.off[s*n_bufs + j] * tensor->nb[1];
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + src_offset,
                    simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                    row_count, simple_tensor->nb[2], tensor->nb[2]);
                simple_offsets[j] += nbytes;
            }
        }
        return;
    }

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.has_off) {
        // every slice is written to the source at its explicit offset; where the slices overlap,
        // the data of the first slice that contains a source row is used
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
        std::vector<size_t> simple_offsets(n_bufs, 0);

        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            std::vector<uint8_t> covered(tensor->ne[0], 0);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const int64_t ne  = split_state.ne[s*n_bufs + j];
                    const int64_t off = split_state.off[s*n_bufs + j];
                    GGML_ASSERT(ne % blck_size == 0);
                    const size_t nbytes = ne/blck_size * tensor->nb[0];

                    int64_t i_start = off;
                    const int64_t i_stop = off + ne;
                    while (i_start < i_stop) {
                        if (covered[i_start]) {
                            i_start++;
                            continue;
                        }
                        int64_t i_end = i_start + 1;
                        while (i_end < i_stop && !covered[i_end]) {
                            i_end++;
                        }
                        const size_t src_offset = simple_offsets[j] + (i_start - off)/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + i_start/blck_size * tensor->nb[0],
                            src_offset, (i_end - i_start)/blck_size * tensor->nb[0],
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        std::fill(covered.begin() + i_start, covered.begin() + i_end, 1);
                        i_start = i_end;
                    }
                    simple_offsets[j] += nbytes;
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        std::vector<uint8_t> covered(tensor->ne[1], 0);
        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const int64_t ne  = split_state.ne[s*n_bufs + j];
                const int64_t off = split_state.off[s*n_bufs + j];
                const size_t nbytes = ne * tensor->nb[1];

                int64_t i_start = off;
                const int64_t i_stop = off + ne;
                while (i_start < i_stop) {
                    if (covered[i_start]) {
                        i_start++;
                        continue;
                    }
                    int64_t i_end = i_start + 1;
                    while (i_end < i_stop && !covered[i_end]) {
                        i_end++;
                    }
                    const size_t src_offset = simple_offsets[j] + (i_start - off) * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + i_start * tensor->nb[1],
                        src_offset, (i_end - i_start) * tensor->nb[1],
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    std::fill(covered.begin() + i_start, covered.begin() + i_end, 1);
                    i_start = i_end;
                }
                simple_offsets[j] += nbytes;
            }
        }
        return;
    }

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

//
// cross-shard collectives for ggml_backend_meta_topk_collective / ggml_backend_meta_getrows_collective
//

// Candidate slot padding, ranks after every real candidate under (value desc, id asc).
// Ids stay exact in F32 for id < 2^24 so the padding id only has to exceed all real ids.
static constexpr float GGML_META_COLLECT_PAD_VAL = -FLT_MAX;
static constexpr float GGML_META_COLLECT_PAD_ID  = 16777215.0f;

enum ggml_backend_meta_collect_kind {
    GGML_BACKEND_META_COLLECT_ALLREDUCE = 0, // PARTIAL node, summed in place
    GGML_BACKEND_META_COLLECT_TOPK,          // sharded TOP_K, candidates merged across backends
    GGML_BACKEND_META_COLLECT_GETROWS,       // sharded GET_ROWS, rows summed across backends
    GGML_BACKEND_META_COLLECT_PAD,           // unevenly split PAD, every backend pads its own slice
    GGML_BACKEND_META_COLLECT_ASSIST,        // idle devices compute the KV tail of a full attention node
};

// output that has to land on [dst] at [offset] instead of into the per-backend scratch buffer
struct ggml_backend_meta_aux_bind {
    ggml_tensor * tensor;
    ggml_tensor * dst;
    size_t        offset;
};

struct ggml_backend_meta_aux_graph {
    ggml_cgraph *                                graph = nullptr;
    std::vector<ggml_backend_meta_aux_bind>      binds;
};

struct ggml_backend_meta_collect {
    int                                      kind   = GGML_BACKEND_META_COLLECT_ALLREDUCE;
    int                                      i_node = -1;
    std::vector<ggml_backend_meta_aux_graph> local;
    std::vector<ggml_backend_meta_aux_graph> merge;
    std::vector<ggml_tensor *>               reduce; // per-backend tensors summed across backends
    std::vector<ggml_backend_meta_assist_rank> assist; // per-backend descriptors, kind == ASSIST only
};

static bool ggml_backend_meta_node_topk_collective(const struct ggml_tensor * node) {
    return node->op == GGML_OP_TOP_K && node->src[0] != nullptr && node->src[0]->buffer != nullptr &&
           ggml_backend_buffer_is_meta(node->src[0]->buffer) &&
           ggml_backend_meta_topk_collective(ggml_backend_meta_get_split_state(node->src[0], /*assume_sync =*/ true).axis, node->src[0]->type);
}

static bool ggml_backend_meta_node_getrows_collective(const struct ggml_tensor * node) {
    return node->op == GGML_OP_GET_ROWS && node->src[0] != nullptr && node->src[1] != nullptr &&
           node->src[0]->buffer != nullptr && ggml_backend_buffer_is_meta(node->src[0]->buffer) &&
           node->src[1]->buffer != nullptr && ggml_backend_buffer_is_meta(node->src[1]->buffer) &&
           ggml_backend_meta_getrows_collective(
               ggml_backend_meta_get_split_state(node->src[0], /*assume_sync =*/ true).axis,
               ggml_backend_meta_get_split_state(node->src[1], /*assume_sync =*/ true).axis, node->type);
}

static bool ggml_backend_meta_node_pad_uneven(const struct ggml_tensor * node) {
    return node->op == GGML_OP_PAD && node->src[0] != nullptr && node->src[0]->buffer != nullptr &&
           ggml_backend_buffer_is_meta(node->src[0]->buffer) &&
           ggml_backend_meta_pad_uneven(node, ggml_backend_meta_get_split_state(node->src[0], /*assume_sync =*/ true));
}

// Shapes of one collective: [k] rows over [n_cols] trailing columns. TOP_K additionally stores
// [n_slots] candidate slots of [k] rows in a [n_slots, n_cols, 2] (values, ids) buffer.
struct ggml_backend_meta_collect_plan {
    int64_t k;
    int64_t n_cols;
    int64_t n_slots;
    int64_t n_rows; // rows of the split input on this backend
    int64_t row0;   // first global row of this backend's slice
};

// copy [tensor] into [view] keeping the strides of [view] (ggml_cpy() resets them)
static ggml_tensor * ggml_backend_meta_aux_copy(ggml_context * ctx, ggml_tensor * tensor, ggml_tensor * view) {
    ggml_tensor * result = ggml_view_4d(ctx, view, view->ne[0], view->ne[1], view->ne[2], view->ne[3],
            view->nb[1], view->nb[2], view->nb[3], 0);
    result->op     = GGML_OP_CPY;
    result->src[0] = tensor;
    result->src[1] = result; // CUDA copies into src[1], self-reference as in ggml_cast()
    return result;
}

// Leaf with the shape, strides and data of the main graph tensor [t] but without its sources,
// so that building the aux graph does not pull in the main graph.
static ggml_tensor * ggml_backend_meta_aux_input(ggml_context * ctx, ggml_tensor * t, ggml_backend_meta_aux_graph & aux) {
    ggml_tensor * leaf = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
    std::copy(t->nb, t->nb + GGML_MAX_DIMS, leaf->nb);
    aux.binds.push_back({leaf, t, 0});
    return leaf;
}

// Per-backend local top-k over one vocab shard: zero the candidate buffer, fill (value, global id)
// into the shard's slot and pad the rest of the slot with markers that rank last.
static void ggml_backend_meta_build_topk_local(
        ggml_context * ctx, const ggml_backend_meta_collect_plan & p,
        ggml_tensor * src, ggml_tensor * cand, int64_t i_slot, ggml_backend_meta_aux_graph & aux) {
    const int64_t slot_off  = i_slot * p.k;
    const int64_t k_loc     = std::min(p.k, p.n_rows);
    const size_t  plane_off = (size_t) p.n_slots * (size_t) p.n_cols * sizeof(float);

    aux.graph = ggml_new_graph_custom(ctx, 64, /*grads =*/ false);
    src = ggml_backend_meta_aux_input(ctx, src, aux);

    // the candidate buffer still holds the summed result of the previous collective
    ggml_tensor * zero = ggml_fill(ctx, cand, 0.0f);
    aux.binds.push_back({zero, cand, 0});
    ggml_build_forward_expand(aux.graph, zero);

    if (k_loc > 0) {
        ggml_tensor * loc = ggml_top_k(ctx, ggml_reshape_2d(ctx, src, p.n_rows, p.n_cols), k_loc);
        ggml_tensor * gids = ggml_backend_meta_aux_copy(ctx,
                ggml_scale_bias(ctx, ggml_cast(ctx, loc, GGML_TYPE_F32), 1.0f, (float) p.row0),
                ggml_view_2d(ctx, cand, k_loc, p.n_cols, (size_t) p.n_slots * sizeof(float),
                    plane_off + (size_t) slot_off * sizeof(float)));
        ggml_tensor * vals = ggml_backend_meta_aux_copy(ctx,
                ggml_get_rows(ctx, ggml_reshape_3d(ctx, src, 1, p.n_rows, p.n_cols), ggml_reshape_3d(ctx, loc, k_loc, p.n_cols, 1)),
                ggml_view_3d(ctx, cand, 1, k_loc, p.n_cols, sizeof(float), (size_t) p.n_slots * sizeof(float),
                    (size_t) slot_off * sizeof(float)));

        ggml_build_forward_expand(aux.graph, gids);
        ggml_build_forward_expand(aux.graph, vals);
    }

    for (int i_plane = 0; i_plane < 2 && k_loc < p.k; i_plane++) {
        ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, p.k - k_loc, p.n_cols);
        ggml_tensor * view = ggml_view_3d(ctx, cand, 1, p.k - k_loc, p.n_cols, sizeof(float),
                (size_t) p.n_slots * sizeof(float),
                (size_t) i_plane * plane_off + (size_t) (slot_off + k_loc) * sizeof(float));
        ggml_tensor * pad = ggml_backend_meta_aux_copy(ctx,
                ggml_fill(ctx, tmpl, i_plane == 0 ? GGML_META_COLLECT_PAD_VAL : GGML_META_COLLECT_PAD_ID), view);
        ggml_build_forward_expand(aux.graph, pad);
    }
}

// Per-backend row gather for one row shard: ids outside the shard map to the zero row that ggml_pad
// appends, the sum over backends then yields the full result.
static void ggml_backend_meta_build_getrows_local(
        ggml_context * ctx, const ggml_backend_meta_collect_plan & p,
        ggml_tensor * src,         ggml_tensor * ids, ggml_tensor * dst, ggml_backend_meta_aux_graph & aux) {
    aux.graph = ggml_new_graph_custom(ctx, 64, /*grads =*/ false);

    if (p.n_rows == 0) {
        ggml_tensor * zero = ggml_fill(ctx, ggml_backend_meta_aux_input(ctx, dst, aux), 0.0f);
        aux.binds.push_back({zero, dst, 0});
        ggml_build_forward_expand(aux.graph, zero);
        return;
    }
    src = ggml_backend_meta_aux_input(ctx, src, aux);
    ids = ggml_backend_meta_aux_input(ctx, ids, aux);

    ggml_tensor * loc = ggml_scale_bias(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), 1.0f, -(float) p.row0);
    // in_shard = (loc >= 0) * (loc <= n_rows - 1); out-of-shard ids select the padded zero row
    ggml_tensor * lo = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sgn(ctx, loc), 1.0f, 1.0f), 0.0f, 1.0f);
    ggml_tensor * hi = ggml_clamp(ctx, ggml_scale_bias(ctx,
            ggml_sgn(ctx, ggml_scale_bias(ctx, loc, -1.0f, (float) (p.n_rows - 1))), 1.0f, 1.0f), 0.0f, 1.0f);
    ggml_tensor * in_shard = ggml_mul(ctx, lo, hi);
    // in_shard ? loc : n_rows
    ggml_tensor * sel = ggml_clamp(ctx,
            ggml_add(ctx, ggml_mul(ctx, in_shard, loc), ggml_scale_bias(ctx, in_shard, -(float) p.n_rows, (float) p.n_rows)),
            0.0f, (float) p.n_rows);
    ggml_tensor * out = ggml_get_rows(ctx, ggml_pad(ctx, src, 0, 1, 0, 0), ggml_cast(ctx, sel, GGML_TYPE_I32));
    aux.binds.push_back({out, dst, 0});
    ggml_build_forward_expand(aux.graph, out);
}

// Per-backend padding of one slice of an uneven AXIS_0 split: the global pad offsets do not apply,
// fill the whole output with the rank-last marker and copy the source slice over the front.
static void ggml_backend_meta_build_pad_local(ggml_context * ctx, ggml_tensor * dst, ggml_backend_meta_aux_graph & aux) {
    aux.graph = ggml_new_graph_custom(ctx, 64, /*grads =*/ false);

    // dummy input: keeps the main graph out of the aux graph, the fill does not read it
    ggml_tensor * tmpl = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    ggml_tensor * fill = ggml_fill(ctx, tmpl, GGML_META_COLLECT_PAD_VAL);
    aux.binds.push_back({fill, dst, 0});
    ggml_build_forward_expand(aux.graph, fill);

    ggml_tensor * src = dst->src[0];
    if (src != nullptr && src->ne[0] > 0) {
        src = ggml_backend_meta_aux_input(ctx, src, aux);
        ggml_tensor * view = ggml_view_4d(ctx, dst, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                dst->nb[1], dst->nb[2], dst->nb[3], 0);
        ggml_tensor * cpy = ggml_backend_meta_aux_copy(ctx, src, view);
        ggml_build_forward_expand(aux.graph, cpy);
    }
}

// Deterministic merge of the gathered candidates into the global top-k. The rank of a candidate is
// the number of candidates sorted before it under (value desc, id asc), the k smallest ranks are
// the result. The rank is unique per candidate so the output does not depend on the sort order.
static void ggml_backend_meta_build_topk_merge(
        ggml_context * ctx, const ggml_backend_meta_collect_plan & p,
        ggml_tensor * cand, ggml_tensor * dst, ggml_backend_meta_aux_graph & aux) {
    const int64_t M = p.n_slots;

    aux.graph = ggml_new_graph_custom(ctx, 64, /*grads =*/ false);

    ggml_tensor * vals = ggml_view_2d(ctx, cand, M, p.n_cols, (size_t) M * sizeof(float), 0);
    ggml_tensor * gids = ggml_view_2d(ctx, cand, M, p.n_cols, (size_t) M * sizeof(float),
            (size_t) M * (size_t) p.n_cols * sizeof(float));
    ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, M, p.n_cols);

    // clamp: +/-inf logits must not turn the differences below into NaN
    ggml_tensor * v = ggml_clamp(ctx, vals, -FLT_MAX, FLT_MAX);
    ggml_tensor * vq = ggml_repeat(ctx, ggml_reshape_3d(ctx, v, M, 1, p.n_cols), tmpl);          // val[q]
    ggml_tensor * vp = ggml_repeat(ctx, ggml_reshape_3d(ctx, v, 1, M, p.n_cols), tmpl);          // val[p]
    ggml_tensor * s  = ggml_sgn(ctx, ggml_sub(ctx, vq, vp));
    ggml_tensor * gt = ggml_clamp(ctx, s, 0.0f, 1.0f);                                           // val[q] >  val[p]
    ggml_tensor * eq = ggml_scale_bias(ctx, ggml_mul(ctx, s, s), -1.0f, 1.0f);                  // val[q] == val[p]
    ggml_tensor * iq = ggml_repeat(ctx, ggml_reshape_3d(ctx, gids, M, 1, p.n_cols), tmpl);       // id[q]
    ggml_tensor * ip = ggml_repeat(ctx, ggml_reshape_3d(ctx, gids, 1, M, p.n_cols), tmpl);       // id[p]
    ggml_tensor * lt = ggml_clamp(ctx, ggml_sgn(ctx, ggml_sub(ctx, ip, iq)), 0.0f, 1.0f);        // id[q] < id[p]
    ggml_tensor * rank = ggml_reshape_2d(ctx,
            ggml_sum_rows(ctx, ggml_add(ctx, gt, ggml_mul(ctx, eq, lt))), M, p.n_cols);
    ggml_tensor * perm = ggml_argsort(ctx, rank, GGML_SORT_ORDER_ASC);
    ggml_tensor * got = ggml_get_rows(ctx, ggml_reshape_3d(ctx, gids, 1, M, p.n_cols),
            ggml_view_2d(ctx, perm, p.k, p.n_cols, perm->nb[1], 0));
    ggml_tensor * out = ggml_cast(ctx, got, GGML_TYPE_I32);
    aux.binds.push_back({out, dst, 0});
    ggml_build_forward_expand(aux.graph, out);
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[GGML_META_N_STC];
    for (int i = 0; i < GGML_META_N_STC; i++) {
        stc_compute[i] = ggml_backend_meta_simple_tensor_container(params, n_simple_bufts);
    }

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute[GGML_META_N_STC];
    for (int i = 0; i < GGML_META_N_STC; i++) {
        stc_compute[i] = ggml_backend_meta_simple_tensor_container(params_compute, n_simple_bufts);
    }

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_cgraph_config {
    ggml_cgraph * cgraph_main = nullptr;
    int           offset      = 0; // Node offset vs. original graph

    std::vector<ggml_cgraph *> cgraphs_aux;
};

// Everything that is specific to one graph UID and can be reused when the same graph is computed again.
struct ggml_backend_meta_plan {
    uint64_t uid         = 0; // UID of the graph this plan was built for, 0 = invalid
    uint64_t last_use    = 0; // plan tick of the last graph_compute that used this plan
    size_t   n_subgraphs = 0;
    int      n_runs      = 0; // completed executions since the plan was built

    std::vector<std::vector<ggml_backend_meta_cgraph_config>> cgraphs; // per backend
    std::vector<std::vector<ggml_tensor *>>                   nodes;   // per backend

    std::vector<ggml_backend_meta_collect>            collects;
    std::vector<ggml_context_ptr>                     ctx_collect;
    std::vector<ggml_backend_buffer_ptr>              buf_collect; // per backend, scratch for the cross-shard collectives
    std::vector<ggml_backend_buffer_ptr>              buf_assist;  // per backend, assist slots, workspace and epoch words

    // Meta buffers used by the graph, with the simple tensor container slot holding its views.
    std::vector<std::pair<ggml_backend_buffer_t, int>> slots;

    // Whole-graph capture of every rank's launch sequence (see ggml_backend_meta_launchers), in segments
    // of consecutive subgraphs: 0 = not captured yet, 1 = captured, -1 = the capture failed.
    int                              full_state = 0;
    std::vector<size_t>              full_seg;  // first subgraph of every segment, then n_subgraphs
    std::vector<std::vector<void *>> full_exec; // per backend, one executable per segment

    // the executables must have been freed (ggml_backend_meta_context::plan_clear)
    void clear() {
        GGML_ASSERT(full_exec.empty());
        uid         = 0;
        last_use    = 0;
        n_subgraphs = 0;
        n_runs      = 0;
        full_state  = 0;
        full_seg.clear();
        collects.clear();
        ctx_collect.clear();
        buf_collect.clear();
        buf_assist.clear();
        slots.clear();
        // cgraphs and nodes keep their capacity and are overwritten on rebuild
    }
};

//
// Meta assist: an idle device computes the KV tail of a full attention node of two owner
// devices, the owners keep the head segment. See ggml_backend_meta_assist_rank.
//

// Consecutive shareable layers rotate between two partial slots on the owner so that a layer does
// not wait for the merge of its predecessor; the free epoch orders the reuse of a slot.
static constexpr int GGML_META_ASSIST_N_SLOTS   = 2;
static constexpr int GGML_META_ASSIST_HEAD_DIM  = 256;    // matches SM70_D256_D
static constexpr int GGML_META_ASSIST_N_OWNERS  = 2;      // owners served by one idle device

struct ggml_backend_meta_assist_layer {
    int     i_node    = -1;
    int     slot      = 0;
    int64_t q_len     = 0;
    int     q_pad     = 0;
    int     heads_q   = 0;
    int64_t kv_view   = 0;
    int     kv_type   = GGML_TYPE_F16;
    float   scale     = 1.0f;
    size_t  qs_offset = 0;
    int     idle[2]   = {-1, -1};
    int     owners[2][GGML_META_ASSIST_N_OWNERS] = {{-1, -1}, {-1, -1}};
    std::vector<ggml_backend_meta_assist_rank> desc;
};

// Per-backend buffer behind plan.buf_assist. Offsets are relative to `base`:
//   [owner slots: p_out, p_max, p_sum per slot][idle workspace][epochs][split points][last words]
struct ggml_backend_meta_assist_buf {
    ggml_backend_buffer_t buf = nullptr;
    char *   base             = nullptr;
    size_t   slot_off[GGML_META_ASSIST_N_SLOTS] = {0, 0};
    int64_t  rows             = 0;
    size_t   work_off         = 0;
    size_t   work_bytes       = 0;
    size_t   epochs_off       = 0;
    size_t   split_off        = 0;
    size_t   lasts_off        = 0;
};

static float * ggml_backend_meta_assist_p_out(const ggml_backend_meta_assist_buf & b, int slot) {
    return (float *) (b.base + b.slot_off[slot]);
}

static float * ggml_backend_meta_assist_p_max(const ggml_backend_meta_assist_buf & b, int slot) {
    return (float *) (b.base + b.slot_off[slot] + (size_t) b.rows * GGML_META_ASSIST_HEAD_DIM * sizeof(float));
}

static float * ggml_backend_meta_assist_p_sum(const ggml_backend_meta_assist_buf & b, int slot) {
    return (float *) (b.base + b.slot_off[slot] + (size_t) b.rows * (GGML_META_ASSIST_HEAD_DIM + 1) * sizeof(float));
}

// epoch words of an owner slot: 0 = epoch_kv, 1 = epoch_partial, 2 = epoch_free
static uint32_t * ggml_backend_meta_assist_epoch(const ggml_backend_meta_assist_buf & b, int slot, int which) {
    return (uint32_t *) (b.base + b.epochs_off + ((size_t) slot*3 + which)*sizeof(uint32_t));
}

static uint32_t * ggml_backend_meta_assist_last_partial(const ggml_backend_meta_assist_buf & b, int slot) {
    return (uint32_t *) (b.base + b.lasts_off + (size_t) slot * sizeof(uint32_t));
}

// device-side split point of (owner rank, slot), written by the owner and read by its idle
static ggml_backend_meta_assist_split * ggml_backend_meta_assist_split_at(const ggml_backend_meta_assist_buf & b, int slot) {
    return (ggml_backend_meta_assist_split *) (b.base + b.split_off + (size_t) slot * sizeof(ggml_backend_meta_assist_split));
}

// idle local words of (owner rank, slot): 0 = last_kv, 1 = last_free
static uint32_t * ggml_backend_meta_assist_last_idle(const ggml_backend_meta_assist_buf & b, int owner, int slot, int which) {
    const size_t i = GGML_META_ASSIST_N_SLOTS + ((size_t) owner*GGML_META_ASSIST_N_SLOTS + slot)*2 + which;
    return (uint32_t *) (b.base + b.lasts_off + i * sizeof(uint32_t));
}

// Returns true when the marked FA node can be shared: shapes, types and per-device K/V slices must
// match the transfer and sync assumptions of the CUDA side.
static bool ggml_backend_meta_assist_check(
        const ggml_cgraph * cgraph,
        const ggml_backend_meta_plan & plan,
        int i_node,
        size_t n_backends,
        const ggml_backend_assist_scratch_layout_t scratch_layout,
        const ggml_backend_assist_workspace_size_t workspace_size,
        ggml_backend_meta_assist_layer & layer) {
    const ggml_tensor * node = cgraph->nodes[i_node];
    if (node->op != GGML_OP_FLASH_ATTN_EXT || !ggml_flash_attn_ext_get_assist_capable(node)) {
        return false;
    }
    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
        return false;
    }
    if (mask->type != GGML_TYPE_I32) {
        return false; // range mask
    }
    if (q->ne[3] != 1) {
        return false; // one stream per rank
    }
    if (q->ne[1] < GGML_BACKEND_META_ASSIST_MIN_Q || k->ne[1] < GGML_BACKEND_META_ASSIST_MIN_KV) {
        return false; // host gate: the view is long enough, the real length is gated on the device
    }
    if (k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (v->type != k->type) {
        return false;
    }
    if (k->ne[0] != GGML_META_ASSIST_HEAD_DIM || v->ne[0] != GGML_META_ASSIST_HEAD_DIM || k->ne[3] != 1 || v->ne[3] != 1) {
        return false;
    }

    const size_t row_bytes = ggml_row_size(k->type, GGML_META_ASSIST_HEAD_DIM);
    int owners[8] = {0};
    int idles[8]  = {0};
    int n_owners  = 0;
    int n_idles   = 0;
    for (size_t j = 0; j < n_backends; j++) {
        const ggml_tensor * fa_j = plan.nodes[j][i_node];
        GGML_ASSERT(fa_j != nullptr);
        const ggml_tensor * q_j = fa_j->src[0];
        const ggml_tensor * k_j = fa_j->src[1];
        const ggml_tensor * v_j = fa_j->src[2];
        if (q_j == nullptr || k_j == nullptr || v_j == nullptr || fa_j->src[3] == nullptr) {
            return false;
        }
        if (k_j->ne[2] > 0) {
            // a KV head holder: the K/V rows of the per-device slice must be contiguous
            if (k_j->ne[0] != GGML_META_ASSIST_HEAD_DIM || k_j->ne[1] != k->ne[1] || k_j->ne[2] != 1 || k_j->ne[3] != 1) {
                return false;
            }
            if (v_j->ne[0] != GGML_META_ASSIST_HEAD_DIM || v_j->ne[1] != v->ne[1] || v_j->ne[2] != 1 || v_j->ne[3] != 1) {
                return false;
            }
            if (k_j->nb[1] != row_bytes || v_j->nb[1] != row_bytes) {
                return false;
            }
            // the idle device pulls the raw K/V bytes with 16 B vector loads
            if ((uintptr_t) k_j->data % 16 != 0 || (uintptr_t) v_j->data % 16 != 0) {
                return false;
            }
            if (q_j->ne[2] <= 0) {
                return false;
            }
            owners[n_owners++] = (int) j;
        } else {
            idles[n_idles++] = (int) j;
        }
    }
    if (n_owners != 4 || n_idles != 2) {
        return false;
    }

    const int64_t heads_q = plan.nodes[owners[0]][i_node]->src[0]->ne[2];
    for (int ii = 1; ii < n_owners; ii++) {
        if (plan.nodes[owners[ii]][i_node]->src[0]->ne[2] != heads_q) {
            return false;
        }
    }

    // every owner must take the windowed partial path, expose the same shape and have an idle
    // device with a workspace (the idle devices are the same across the plan, but check anyway)
    int q_pad = -1;
    size_t qs_offset = 0;
    for (int ii = 0; ii < n_owners; ii++) {
        ggml_backend_meta_assist_scratch scratch = {};
        if (!scratch_layout(plan.nodes[owners[ii]][i_node], &scratch)) {
            return false;
        }
        if (q_pad < 0) {
            q_pad     = scratch.q_pad;
            qs_offset = scratch.qs_offset;
        } else if (q_pad != scratch.q_pad) {
            return false;
        }
    }
    if (q_pad < GGML_BACKEND_META_ASSIST_MIN_Q) {
        return false;
    }

    // every idle device serves the two owners of its own group (rank / 3), every owner is served once
    for (int ii = 0; ii < n_idles; ii++) {
        const int r = idles[ii];
        int n_paired = 0;
        for (int oi = 0; oi < n_owners; oi++) {
            if (owners[oi] / 3 == r / 3) {
                if (n_paired >= GGML_META_ASSIST_N_OWNERS) {
                    return false; // more than two owners in one group
                }
                layer.owners[ii][n_paired++] = owners[oi];
            }
        }
        if (n_paired != GGML_META_ASSIST_N_OWNERS) {
            return false;
        }
        layer.idle[ii] = r;
        if (workspace_size(q_pad, (int) heads_q) == 0) {
            return false;
        }
    }
    for (int oi = 0; oi < n_owners; oi++) {
        int n_serves = 0;
        for (int ii = 0; ii < n_idles; ii++) {
            if (layer.owners[ii][0] == owners[oi] || layer.owners[ii][1] == owners[oi]) {
                n_serves++;
            }
        }
        if (n_serves != 1) {
            return false;
        }
    }

    layer.i_node    = i_node;
    layer.q_len     = q->ne[1];
    layer.q_pad     = q_pad;
    layer.heads_q   = (int) heads_q;
    layer.kv_view   = k->ne[1];
    layer.kv_type   = k->type;
    layer.qs_offset = qs_offset;
    memcpy(&layer.scale, node->op_params, sizeof(float));
    return true;
}

// Scan the graph for shareable layers, allocate their per-backend buffers and fill the per-rank
// descriptors. Returns false and leaves everything untouched when no layer can be shared.
// `before_alloc` runs after the scan but before the buffers are allocated, so the caller can
// release assist buffers of other plans first (they are too large to keep several around).
static bool ggml_backend_meta_assist_build(
        const ggml_cgraph * cgraph,
        ggml_backend_meta_plan & plan,
        const std::vector<ggml_backend_t> & simple_backends,
        const std::vector<int> & device_ordinal,
        const ggml_backend_assist_scratch_layout_t scratch_layout,
        const ggml_backend_assist_workspace_size_t workspace_size,
        const std::function<void()> & before_alloc,
        std::vector<ggml_backend_meta_assist_layer> & layers,
        std::vector<int> & at_node) {
    const size_t n_backends = simple_backends.size();
    at_node.assign(cgraph->n_nodes, -1);
    if (n_backends != 6 || scratch_layout == nullptr || workspace_size == nullptr || device_ordinal.size() != n_backends) {
        return false;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_backend_meta_assist_layer layer;
        if (!ggml_backend_meta_assist_check(cgraph, plan, i, n_backends, scratch_layout, workspace_size, layer)) {
            continue;
        }
        layer.slot = (int) (layers.size() % GGML_META_ASSIST_N_SLOTS);
        layer.desc.resize(n_backends);
        at_node[i] = (int) layers.size();
        layers.push_back(std::move(layer));
    }
    if (layers.empty()) {
        return false;
    }

    int64_t rows = 0;
    std::vector<size_t> work_bytes(n_backends, 0);
    for (const auto & layer : layers) {
        rows = std::max(rows, (int64_t) layer.q_pad * layer.heads_q);
        for (int ii = 0; ii < GGML_META_ASSIST_N_OWNERS; ii++) {
            work_bytes[layer.idle[ii]] = std::max(work_bytes[layer.idle[ii]],
                workspace_size(layer.q_pad, layer.heads_q));
        }
    }
    if (rows <= 0) {
        layers.clear();
        at_node.assign(cgraph->n_nodes, -1);
        return false;
    }

    before_alloc();

    const size_t n_epochs = GGML_META_ASSIST_N_SLOTS*3;
    const size_t n_lasts  = GGML_META_ASSIST_N_SLOTS + n_backends*GGML_META_ASSIST_N_SLOTS*2;

    std::vector<ggml_backend_meta_assist_buf> bufs(n_backends);
    plan.buf_assist.clear();
    plan.buf_assist.resize(n_backends);
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_meta_assist_buf & b = bufs[j];
        b.rows = rows;
        size_t off = 0;
        for (int s = 0; s < GGML_META_ASSIST_N_SLOTS; s++) {
            off = (off + 127) & ~size_t(127);
            b.slot_off[s] = off;
            off += (size_t) rows * GGML_META_ASSIST_HEAD_DIM * sizeof(float) + (size_t) rows * 2 * sizeof(float);
        }
        if (work_bytes[j] > 0) {
            off = (off + 127) & ~size_t(127);
            b.work_off   = off;
            b.work_bytes = work_bytes[j];
            off += work_bytes[j];
        }
        off = (off + 3) & ~size_t(3);
        b.epochs_off = off;
        off += n_epochs * sizeof(uint32_t);
        off = (off + 7) & ~size_t(7);
        b.split_off = off;
        off += GGML_META_ASSIST_N_SLOTS * sizeof(ggml_backend_meta_assist_split);
        off = (off + 3) & ~size_t(3);
        b.lasts_off = off;
        off += n_lasts * sizeof(uint32_t);

        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(simple_backends[j], off);
        GGML_ASSERT(buf != nullptr);
        b.buf  = buf;
        b.base = (char *) ggml_backend_buffer_get_base(buf);
        GGML_ASSERT((uintptr_t) b.base % 128 == 0);
        // epochs start at 0. The data epochs (KV ready, partial ready) start their last word at 0
        // so the first wait blocks until the peer signals; the free epoch starts at 0xFFFFFFFF
        // because there is no earlier use of the slot to wait for.
        ggml_backend_buffer_clear(buf, 0x00);
        {
            std::vector<uint8_t> lasts_init(n_lasts * sizeof(uint32_t), 0x00);
            for (size_t o = 0; o < n_backends; o++) {
                for (int s = 0; s < GGML_META_ASSIST_N_SLOTS; s++) {
                    const size_t i = GGML_META_ASSIST_N_SLOTS + (o*GGML_META_ASSIST_N_SLOTS + s)*2 + 1; // last_free
                    memset(&lasts_init[i * sizeof(uint32_t)], 0xFF, sizeof(uint32_t));
                }
            }
            const ggml_init_params params = {
                /*.mem_size   =*/ 2*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx(ggml_init(params));
            ggml_tensor * lasts = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, n_lasts * sizeof(uint32_t));
            ggml_backend_tensor_alloc(buf, lasts, b.base + b.lasts_off);
            ggml_backend_tensor_set(lasts, lasts_init.data(), 0, lasts_init.size());
        }
        plan.buf_assist[j].reset(buf);
    }

    for (auto & layer : layers) {
        const int slot = layer.slot;
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_meta_assist_rank d = {};

            int i_idle = -1;
            for (int ii = 0; ii < GGML_META_ASSIST_N_OWNERS; ii++) {
                if (layer.idle[ii] == (int) j) {
                    i_idle = ii;
                }
            }

            d.q_pad    = layer.q_pad;
            d.q_len    = (int32_t) layer.q_len;
            d.heads_q  = layer.heads_q;
            d.heads_kv = 1;
            d.batch    = 1;
            d.kv_view  = (int32_t) layer.kv_view;
            d.kv_window = GGML_BACKEND_META_ASSIST_KV_WINDOW;
            d.kv_type  = layer.kv_type;
            d.scale    = layer.scale;

            if (i_idle >= 0) {
                // idle device: it serves the two owners of its group
                d.role     = 1;
                d.n_owners = GGML_META_ASSIST_N_OWNERS;
                d.split_local = ggml_backend_meta_assist_split_at(bufs[j], slot);
                d.mask_is_range   = true;

                const ggml_tensor * mask = plan.nodes[j][layer.i_node]->src[3];
                d.mask = mask->data;
                // int2 units, as the CUDA range bounds kernel indexes the mask
                d.mask_row_stride   = (int64_t) mask->nb[1] / sizeof(int64_t);
                d.mask_batch_stride = mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int64_t);

                d.work       = bufs[j].base + bufs[j].work_off;
                d.work_bytes = bufs[j].work_bytes;

                for (int jj = 0; jj < GGML_META_ASSIST_N_OWNERS; jj++) {
                    const int o = layer.owners[i_idle][jj];
                    const ggml_tensor * fa_o = plan.nodes[o][layer.i_node];
                    d.owners[jj]   = o;
                    d.owner_dev[jj] = device_ordinal[o];
                    d.k_src[jj] = fa_o->src[1]->data;
                    d.v_src[jj] = fa_o->src[2]->data;
                    d.qs_src[jj] = (const char *) fa_o->data + ggml_nbytes(fa_o) + layer.qs_offset;
                    d.p_out_dst[jj] = ggml_backend_meta_assist_p_out(bufs[o], slot);
                    d.p_max_dst[jj] = ggml_backend_meta_assist_p_max(bufs[o], slot);
                    d.p_sum_dst[jj] = ggml_backend_meta_assist_p_sum(bufs[o], slot);
                    d.epoch_kv[jj]      = ggml_backend_meta_assist_epoch(bufs[o], slot, 0);
                    d.epoch_partial[jj] = ggml_backend_meta_assist_epoch(bufs[o], slot, 1);
                    d.epoch_free[jj]    = ggml_backend_meta_assist_epoch(bufs[o], slot, 2);
                    d.last_kv[jj]   = ggml_backend_meta_assist_last_idle(bufs[j], o, slot, 0);
                    d.last_free[jj] = ggml_backend_meta_assist_last_idle(bufs[j], o, slot, 1);
                    d.split_src[jj] = ggml_backend_meta_assist_split_at(bufs[o], slot);
                }
            } else {
                // owner device: it merges the tail partial of its single idle device
                int idle_rank = -1;
                for (int ii = 0; ii < GGML_META_ASSIST_N_OWNERS && idle_rank < 0; ii++) {
                    for (int jj = 0; jj < GGML_META_ASSIST_N_OWNERS; jj++) {
                        if (layer.owners[ii][jj] == (int) j) {
                            idle_rank = layer.idle[ii];
                            break;
                        }
                    }
                }
                GGML_ASSERT(idle_rank >= 0);

                d.role        = 0;
                d.n_owners    = 1;
                d.assist_rank = idle_rank;
                d.owner_dev[0] = device_ordinal[j];
                d.split_local = ggml_backend_meta_assist_split_at(bufs[j], slot);
                d.p_out_local = ggml_backend_meta_assist_p_out(bufs[j], slot);
                d.p_max_local = ggml_backend_meta_assist_p_max(bufs[j], slot);
                d.p_sum_local = ggml_backend_meta_assist_p_sum(bufs[j], slot);
                d.epoch_kv_local      = ggml_backend_meta_assist_epoch(bufs[j], slot, 0);
                d.epoch_partial_local = ggml_backend_meta_assist_epoch(bufs[j], slot, 1);
                d.last_partial_local  = ggml_backend_meta_assist_last_partial(bufs[j], slot);
                d.epoch_free_local    = ggml_backend_meta_assist_epoch(bufs[j], slot, 2);
            }
            layer.desc[j] = d;
        }
    }

    return true;
}

// One launcher thread per backend: with a single thread the launch overhead of every
// subgraph of every rank is serialized on the host and can keep the GPUs idle. The
// push AllReduce synchronizes on the GPU, so ranks may be driven independently as
// long as each submits the same collective sequence. The main thread waits for the
// enqueues before returning because the caller may use the same streams right after.
//
// Once a plan is warm, one run records every rank's launch sequence (subgraphs and all-reduces) into
// executables (ggml_backend_capture_*_t) that later runs replay, one launch per segment of about
// GGML_META_CAPTURE_SEGMENT_NODES nodes: launching a CUDA graph takes time proportional to its nodes,
// so the first segments already run while the later ones are being launched. Nothing runs during the
// capture, and every rank instantiates its executables before any rank launches, for the same reason
// as the barrier after each collective.
static constexpr int GGML_META_CAPTURE_SEGMENT_NODES = 256;

struct ggml_backend_meta_launchers {
    std::vector<ggml_backend_t> backends;
    void *                                  comm_ctx           = nullptr;
    ggml_backend_comm_allreduce_rank_can_t  allreduce_rank_can = nullptr;
    ggml_backend_comm_allreduce_rank_t      allreduce_rank     = nullptr;
    ggml_backend_set_assist_t               assist_set         = nullptr;
    ggml_backend_assist_run_t               assist_run         = nullptr;

    // optional whole-graph capture, all set or all nullptr
    ggml_backend_capture_begin_t  capture_begin  = nullptr;
    ggml_backend_capture_end_t    capture_end    = nullptr;
    ggml_backend_capture_launch_t capture_launch = nullptr;
    ggml_backend_capture_free_t   capture_free   = nullptr;

    std::vector<std::thread> threads;

    // Dispatch protocol: the main thread publishes a plan and then bumps `job`,
    // the workers run their launch sequence and bump `n_done`.
    std::atomic<uint64_t>   job{0};
    std::atomic<int>        n_done{0};
    std::atomic<bool>       stop{false};
    std::mutex              mutex;
    std::condition_variable cv;

    // Written by the main thread before `job` is bumped, read by the workers.
    ggml_backend_meta_plan * plan = nullptr;
    bool capture_run = false; // record the plan into plan->full_exec during this run

    // per rank result of the capture, exchanged through barrier()
    std::vector<char> capture_ok;

    // After enqueueing its part of a collective a worker waits until every rank has enqueued its
    // part. Otherwise a worker blocked in the driver (a lock holder that waits for its GPU, e.g.
    // during cudaGraphInstantiate) can keep a peer from enqueueing its part while the parts that
    // are already running spin on that peer, which deadlocked on V100.
    std::atomic<int>  bar_count{0};
    std::atomic<int>  bar_gen{0};
    std::atomic<bool> failed{false};

    std::vector<ggml_status> statuses;

    ggml_backend_meta_launchers(std::vector<ggml_backend_t> backends,
            void * comm_ctx,
            ggml_backend_comm_allreduce_rank_can_t allreduce_rank_can,
            ggml_backend_comm_allreduce_rank_t allreduce_rank,
            ggml_backend_set_assist_t assist_set,
            ggml_backend_assist_run_t assist_run) :
        backends(std::move(backends)), comm_ctx(comm_ctx),
        allreduce_rank_can(allreduce_rank_can), allreduce_rank(allreduce_rank),
        assist_set(assist_set), assist_run(assist_run),
        capture_ok(this->backends.size(), 0),
        statuses(this->backends.size(), GGML_STATUS_SUCCESS) {
        threads.reserve(this->backends.size());
        for (size_t j = 0; j < this->backends.size(); j++) {
            threads.emplace_back(&ggml_backend_meta_launchers::worker, this, j);
        }
    }

    ~ggml_backend_meta_launchers() {
        stop.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mutex);
            cv.notify_all();
        }
        for (std::thread & t : threads) {
            t.join();
        }
    }

    // Rank j's worker: spin for a while so a burst of jobs does not pay a
    // futex wakeup each time, then sleep on the condition variable.
    void worker(size_t j) {
        // `job` starts at 0: a job published before this thread got here must not be skipped
        uint64_t seen = 0;
        while (true) {
            uint64_t cur = job.load(std::memory_order_acquire);
            if (cur == seen && !stop.load(std::memory_order_acquire)) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
                int i_spin = 0;
                while (cur == seen && !stop.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < deadline) {
                    if ((++i_spin & 63) == 0) {
                        std::this_thread::yield();
                    }
                    cur = job.load(std::memory_order_acquire);
                }
            }
            if (cur == seen) {
                std::unique_lock<std::mutex> lock(mutex);
                while (job.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_acquire)) {
                    cv.wait(lock);
                }
                cur = job.load(std::memory_order_acquire);
            }
            if (stop.load(std::memory_order_acquire)) {
                // The destructor cannot race with a job: the main thread waits
                // for n_done before it returns from a dispatch.
                return;
            }

            seen = cur;
            statuses[j] = launch_rank(j);
            if (statuses[j] != GGML_STATUS_SUCCESS) {
                failed.store(true, std::memory_order_relaxed);
            }
            n_done.fetch_add(1, std::memory_order_release);
        }
    }

    // Returns false if another worker failed and will not arrive.
    bool barrier() {
        const int gen = bar_gen.load(std::memory_order_acquire);
        if (bar_count.fetch_add(1, std::memory_order_acq_rel) == (int) backends.size() - 1) {
            bar_count.store(0, std::memory_order_relaxed);
            bar_gen.fetch_add(1, std::memory_order_release);
            return true;
        }
        while (bar_gen.load(std::memory_order_acquire) == gen) {
            if (failed.load(std::memory_order_relaxed)) {
                return false;
            }
        }
        return true;
    }

    // Rank j's part of a run: replay the captured executables, record them, or launch the sequence.
    ggml_status launch_rank(size_t j) {
        ggml_backend_meta_plan * p = plan;
        ggml_backend_t backend = backends[j];

        if (p->full_state == 1) {
            for (void * exec : p->full_exec[j]) {
                if (!capture_launch(backend, exec)) {
                    return GGML_STATUS_FAILED;
                }
            }
            return GGML_STATUS_SUCCESS;
        }
        if (!capture_run) {
            return launch_sequence(j, 0, p->n_subgraphs);
        }

        // Record the segments. A stream capture only fails to begin on a broken stream; a rank that runs
        // its sequence while the others record would spin in the all-reduces, so this is fatal.
        std::vector<void *> execs;
        bool        ok     = true;
        ggml_status status = GGML_STATUS_SUCCESS;
        for (size_t s = 0; s + 1 < p->full_seg.size() && status == GGML_STATUS_SUCCESS; s++) {
            if (!capture_begin(backend)) {
                GGML_ABORT("%s: failed to begin the capture on backend %zu", __func__, j);
            }
            status = launch_sequence(j, p->full_seg[s], p->full_seg[s + 1]);
            execs.push_back(capture_end(backend));
            ok = ok && execs.back() != nullptr;
        }
        auto free_execs = [&]() {
            for (void * exec : execs) {
                capture_free(exec);
            }
        };
        if (status != GGML_STATUS_SUCCESS) {
            free_execs();
            return status;
        }

        // nothing was executed so far: launch the executables once all ranks have instantiated theirs,
        // or run the sequence without capture if any rank failed
        capture_ok[j] = ok;
        if (!barrier()) {
            free_execs();
            return GGML_STATUS_FAILED;
        }
        bool all_ok = true;
        for (char rank_ok : capture_ok) {
            all_ok = all_ok && rank_ok;
        }
        if (!all_ok) {
            free_execs();
            return launch_sequence(j, 0, p->n_subgraphs);
        }
        p->full_exec[j] = execs;
        for (void * exec : execs) {
            if (!capture_launch(backend, exec)) {
                return GGML_STATUS_FAILED;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    // Rank j's launch sequence for the subgraphs [i0, i1) and their collectives, the same work the
    // single-threaded path does for rank j.
    ggml_status launch_sequence(size_t j, size_t i0, size_t i1) {
        const ggml_backend_meta_plan * p = plan;
        const size_t n_backends = backends.size();
        ggml_backend_t backend = backends[j];

        for (size_t i = i0; i < i1; i++) {
            const ggml_backend_meta_collect & collect = p->collects[i];
            const bool assist = collect.kind == GGML_BACKEND_META_COLLECT_ASSIST;

            // mount the assist on the owner just for the FA node at the end of this subgraph
            if (assist && collect.assist[j].role == 0) {
                if (!assist_set(backend, &collect.assist[j])) {
                    return GGML_STATUS_FAILED;
                }
            }
            if (p->cgraphs[j][i].cgraph_main->n_nodes > 0) {
                const ggml_status status = ggml_backend_graph_compute_async(backend, p->cgraphs[j][i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            if (assist && collect.assist[j].role == 0) {
                if (!assist_set(backend, nullptr)) {
                    return GGML_STATUS_FAILED;
                }
            }

            if (assist) {
                // the idle rank copies and attends the owner's KV tail, the owner is a no-op here
                if (!assist_run(backend, &collect.assist[j]) || !barrier()) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            }
            if (collect.kind != GGML_BACKEND_META_COLLECT_ALLREDUCE) {
                if (collect.local[j].graph != nullptr) {
                    const ggml_status status = ggml_backend_graph_compute_async(backend, collect.local[j].graph);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
                const bool gather = collect.kind == GGML_BACKEND_META_COLLECT_TOPK ||
                                    collect.kind == GGML_BACKEND_META_COLLECT_GETROWS;
                if (gather && n_backends > 1) {
                    if (!allreduce_rank(comm_ctx, j, collect.reduce[j], true) || !barrier()) {
                        return GGML_STATUS_FAILED;
                    }
                }
                if (collect.kind == GGML_BACKEND_META_COLLECT_TOPK) {
                    const ggml_status status = ggml_backend_graph_compute_async(backend, collect.merge[j].graph);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
            } else if (n_backends > 1 && i < p->n_subgraphs - 1) {
                if (!allreduce_rank(comm_ctx, j, collect.reduce[j], false) || !barrier()) {
                    return GGML_STATUS_FAILED;
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    bool can(ggml_tensor ** tensors, bool exact) const {
        return allreduce_rank_can(comm_ctx, tensors, exact);
    }

    bool can_capture() const {
        return capture_begin != nullptr && capture_end != nullptr && capture_launch != nullptr && capture_free != nullptr;
    }

    // capture: record this run into p->full_exec, see ggml_backend_meta_launchers
    ggml_status compute(ggml_backend_meta_plan * p, bool capture) {
        plan        = p;
        capture_run = capture;
        if (capture) {
            GGML_ASSERT(p->full_state == 0 && p->full_exec.empty());
            p->full_exec.assign(backends.size(), {});

            // segment boundaries after a subgraph, i.e. after its collective, the same for all ranks
            p->full_seg.assign(1, 0);
            int n_nodes = 0;
            for (size_t i = 0; i + 1 < p->n_subgraphs; i++) {
                n_nodes += p->cgraphs[0][i].cgraph_main->n_nodes;
                if (n_nodes >= GGML_META_CAPTURE_SEGMENT_NODES) {
                    p->full_seg.push_back(i + 1);
                    n_nodes = 0;
                }
            }
            p->full_seg.push_back(p->n_subgraphs);
        }
        for (ggml_status & s : statuses) {
            s = GGML_STATUS_SUCCESS;
        }
        n_done.store(0, std::memory_order_relaxed);
        bar_count.store(0, std::memory_order_relaxed);
        failed.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex);
            job.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();

        // Wait for the enqueues only, not for the GPU work: the caller may read
        // or synchronize the same streams right after this returns.
        int i_spin = 0;
        while (n_done.load(std::memory_order_acquire) != (int) backends.size()) {
            if ((++i_spin & 63) == 0) {
                std::this_thread::yield();
            }
        }

        if (capture) {
            bool all_captured = true;
            for (const auto & execs : p->full_exec) {
                all_captured = all_captured && !execs.empty();
            }
            if (!all_captured) {
                // a rank that failed later than the others may leave executables behind
                for (const auto & execs : p->full_exec) {
                    for (void * exec : execs) {
                        capture_free(exec);
                    }
                }
                p->full_exec.clear();
            }
            p->full_state = all_captured ? 1 : -1;
            if (all_captured) {
                ggml_backend_meta_hot_add(p->uid);
            }
        }

        for (size_t j = 0; j < backends.size(); j++) {
            if (statuses[j] != GGML_STATUS_SUCCESS) {
                return statuses[j];
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

struct ggml_backend_meta_context {
    struct backend_config {
        ggml_backend_t backend;

        std::vector<ggml_backend_buffer_ptr> bufs;
        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                       name;
    std::vector<backend_config>       backend_configs;
    // retired temporary buffers: captured plans may still reference them, so they are kept until the context is destroyed
    std::vector<ggml_backend_buffer_ptr> tmp_bufs_retired;
    ggml_backend_meta_plan            plans[GGML_META_N_PLANS];
    ggml_context_ptr                  ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce_exact = nullptr;

    std::unique_ptr<ggml_backend_meta_launchers> launchers;

    // optional, see ggml_backend_meta_launchers
    ggml_backend_comm_allreduce_rank_capturable_t comm_allreduce_rank_capturable = nullptr;

    // Optional: meta assist, see ggml_backend_meta_assist_rank. All procs and the device
    // ordinals must be available for a plan to take the assist path.
    ggml_backend_set_assist_t            assist_set            = nullptr;
    ggml_backend_assist_run_t            assist_run            = nullptr;
    ggml_backend_assist_workspace_size_t assist_workspace_size = nullptr;
    ggml_backend_assist_scratch_layout_t assist_scratch_layout = nullptr;
    std::vector<int>                     device_ordinal;

    // Invalidate a plan. Its executables may still run, so the ranks are synchronized before they are freed.
    void plan_clear(ggml_backend_meta_plan & plan) {
        if (plan.uid != 0) {
            ggml_backend_meta_hot_remove(plan.uid);
        }
        if (!plan.full_exec.empty()) {
            for (auto & bc : backend_configs) {
                ggml_backend_synchronize(bc.backend);
            }
            for (const auto & execs : plan.full_exec) {
                for (void * exec : execs) {
                    launchers->capture_free(exec);
                }
            }
            plan.full_exec.clear();
        }
        plan.clear();
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            assist_set            = (ggml_backend_set_assist_t)            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_assist");
            assist_run            = (ggml_backend_assist_run_t)            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_assist_run");
            assist_workspace_size = (ggml_backend_assist_workspace_size_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_assist_workspace_size");
            assist_scratch_layout = (ggml_backend_assist_scratch_layout_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_assist_scratch_layout");
            ggml_backend_device_ordinal_t get_device_ordinal = (ggml_backend_device_ordinal_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_device_ordinal");

            if (assist_set != nullptr && assist_run != nullptr && assist_workspace_size != nullptr &&
                    assist_scratch_layout != nullptr && get_device_ordinal != nullptr) {
                device_ordinal.resize(n_devs);
                for (size_t i = 0; i < n_devs; i++) {
                    device_ordinal[i] = get_device_ordinal(simple_backends[i]);
                }
            } else {
                assist_set            = nullptr;
                assist_run            = nullptr;
                assist_workspace_size = nullptr;
                assist_scratch_layout = nullptr;
            }
        }

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            // Optional: the fallback reduction is exact, so a missing proc
            // only means that the collectives lose the fast path.
            comm_allreduce_exact = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor_exact");

            ggml_backend_comm_allreduce_rank_can_t comm_allreduce_rank_can = (ggml_backend_comm_allreduce_rank_can_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_rank_can");
            ggml_backend_comm_allreduce_rank_t comm_allreduce_rank = (ggml_backend_comm_allreduce_rank_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_rank");

            // Optional: per-rank launcher threads. The plan is only executed
            // this way when every collective in it can be driven per rank.
            if (n_devs > 1 && comm_allreduce_rank_can != nullptr && comm_allreduce_rank != nullptr) {
                launchers = std::make_unique<ggml_backend_meta_launchers>(
                    simple_backends, comm_ctx, comm_allreduce_rank_can, comm_allreduce_rank,
                    assist_set, assist_run);

                // Optional: whole-graph capture of the launch sequences.
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
                comm_allreduce_rank_capturable = (ggml_backend_comm_allreduce_rank_capturable_t)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_rank_capturable");
                launchers->capture_begin  = (ggml_backend_capture_begin_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_begin");
                launchers->capture_end    = (ggml_backend_capture_end_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_end");
                launchers->capture_launch = (ggml_backend_capture_launch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_launch");
                launchers->capture_free   = (ggml_backend_capture_free_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_free");
            }
        }
    }

    ~ggml_backend_meta_context() {
        for (auto & plan : plans) {
            if (!plan.full_exec.empty()) {
                plan_clear(plan);
            }
        }
        // Stop the launcher threads before the comm context they use is freed.
        launchers.reset();
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    // multi-segment and PARTIAL layouts are staged on the host (PARTIAL scales the values into a temporary buffer),
    // and slices at explicit offsets may overlap, so none of them can be spliced as consecutive chunks:
    // wait for the queued work of all ranks, then copy synchronously
    if (split_state.has_off || split_state.n_segments != 1 || split_state.nr[0] != 1 ||
            split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        ggml_backend_synchronize(backend);
        ggml_backend_meta_buffer_set_tensor(tensor->buffer, tensor, data, offset, size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.has_off) {
        // slices with explicit offsets may overlap, so they cannot be spliced into chunks: copy synchronously
        ggml_backend_synchronize(backend);
        ggml_backend_meta_buffer_get_tensor(tensor->buffer, tensor, data, offset, size);
        return;
    }
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static bool ggml_backend_meta_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);

    if (!ggml_backend_is_meta(backend_dst)) {
        return false;
    }
    if (dst->buffer == nullptr || !ggml_backend_buffer_is_meta(dst->buffer)) {
        return false;
    }
    if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    if (!ggml_are_same_layout(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (ggml_nelements(dst) == 0) {
        return true;
    }
    // only the layouts that set_tensor_async can upload, anything else takes the scheduler's synchronous copy
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(dst, /*assume_sync =*/ false);
    if (split_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED && split_state.axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
            (split_state.axis < 0 || split_state.axis > GGML_BACKEND_SPLIT_AXIS_2)) {
        return false;
    }

    // The host buffer of src is not overwritten until this graph is done (the caller synchronizes
    // the scheduler before writing inputs again), so the per-rank copies issued here may still be in
    // flight when this function returns. A pinned src makes them truly asynchronous.
    ggml_backend_meta_set_tensor_async(backend_dst, dst, src->data, 0, ggml_nbytes(dst));
    return true;
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const ggml_backend_meta_event_context * ev_ctx = (const ggml_backend_meta_event_context *) event->context;
    GGML_ASSERT(ev_ctx->simple_events.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_record(ev_ctx->simple_events[i], ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const ggml_backend_meta_event_context * ev_ctx = (const ggml_backend_meta_event_context *) event->context;
    GGML_ASSERT(ev_ctx->simple_events.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, i), ev_ctx->simple_events[i]);
    }
}

void ggml_backend_meta_copy_mirrored_async(ggml_backend_t backend,
        const struct ggml_tensor * src, size_t src_offs, struct ggml_tensor * dst, size_t dst_offs, size_t nbytes) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    GGML_ASSERT(src != nullptr && dst != nullptr);
    GGML_ASSERT(src->view_src == nullptr && dst->view_src == nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_meta(src->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(dst->buffer));
    GGML_ASSERT(ggml_backend_meta_get_split_state(src, /*assume_sync =*/ true).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    GGML_ASSERT(ggml_backend_meta_get_split_state(dst, /*assume_sync =*/ true).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    GGML_ASSERT(src_offs + nbytes <= ggml_nbytes(src));
    GGML_ASSERT(dst_offs + nbytes <= ggml_nbytes(dst));
    // the views below keep the tensor types, so the range has to be whole elements of both
    GGML_ASSERT(src->type == dst->type && !ggml_is_quantized(src->type));
    const size_t elsize = ggml_element_size(src);
    GGML_ASSERT(nbytes % elsize == 0 && src_offs % elsize == 0 && dst_offs % elsize == 0);

    const size_t n_backends = ggml_backend_meta_n_backends(backend);

    // temporary context holding the per-device views, they are no longer needed once the copies are enqueued
    const ggml_init_params params = {
        /*.mem_size   =*/ 2*n_backends*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));

    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);

        ggml_tensor * src_simple = ggml_backend_meta_buffer_simple_tensor(src, j);
        ggml_tensor * dst_simple = ggml_backend_meta_buffer_simple_tensor(dst, j);
        GGML_ASSERT(src_simple != nullptr && dst_simple != nullptr);

        // views on the static simple tensors, they are initialized by the generic view path (no meta buffer involved)
        ggml_tensor * src_view = ggml_view_1d(ctx.get(), src_simple, (int64_t) (nbytes / elsize), src_offs);
        ggml_tensor * dst_view = ggml_view_1d(ctx.get(), dst_simple, (int64_t) (nbytes / elsize), dst_offs);
        GGML_ASSERT(ggml_backend_view_init(src_view) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_view_init(dst_view) == GGML_STATUS_SUCCESS);

        ggml_backend_tensor_copy_async(simple_backend, simple_backend, src_view, dst_view);
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If a plan for this graph UID still owns all of its simple tensor containers it can be reused.
    ggml_backend_meta_plan * p = nullptr;
    if (cgraph->uid != 0) {
        for (int k = 0; k < GGML_META_N_PLANS && p == nullptr; k++) {
            ggml_backend_meta_plan & plan = backend_ctx->plans[k];
            if (plan.uid != cgraph->uid) {
                continue;
            }
            bool valid = true;
            for (const auto & slot : plan.slots) {
                ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) slot.first->context;
                if (buf_ctx->stc_owner_uid[slot.second] != cgraph->uid) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                const uint64_t tick = ++ggml_backend_meta_tick;
                plan.last_use = tick;
                for (const auto & slot : plan.slots) {
                    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) slot.first->context;
                    buf_ctx->stc_compute_index = slot.second;
                    buf_ctx->stc_last_use[slot.second] = tick;
                }
                p = &plan;
            }
        }
    }

    const bool needs_rebuild = p == nullptr;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (int k = 0; k < GGML_META_N_PLANS; k++) {
            backend_ctx->plans[k].nodes.resize(n_backends);
            backend_ctx->plans[k].cgraphs.resize(n_backends);
            for (size_t j = 0; j < n_backends; j++) {
                backend_ctx->plans[k].nodes[j].resize(cgraph->n_nodes);
                backend_ctx->plans[k].cgraphs[j].resize(cgraph->n_nodes);
            }
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        // Pick the slot to rebuild in: the stale plan of this same graph if there is one, an empty slot,
        // then the least recently used plan that is not in flight and has no captured CUDA graph, then
        // the least recently used plan that is not in flight, then any plan.
        const uint64_t now = ggml_backend_meta_tick.load();
        int k = -1;
        for (int i = 0; i < GGML_META_N_PLANS; i++) {
            if (backend_ctx->plans[i].uid == cgraph->uid) {
                k = i;
                break;
            }
        }
        if (k < 0) {
            for (int i = 0; i < GGML_META_N_PLANS; i++) {
                if (backend_ctx->plans[i].uid == 0) {
                    k = i;
                    break;
                }
            }
        }
        if (k < 0) {
            for (int i = 0; i < GGML_META_N_PLANS; i++) {
                if (now - backend_ctx->plans[i].last_use <= GGML_META_RECENT_TICKS) {
                    continue;
                }
                if (backend_ctx->plans[i].full_state == 1) {
                    continue;
                }
                if (k < 0 || backend_ctx->plans[i].last_use < backend_ctx->plans[k].last_use) {
                    k = i;
                }
            }
        }
        if (k < 0) {
            for (int i = 0; i < GGML_META_N_PLANS; i++) {
                if (now - backend_ctx->plans[i].last_use <= GGML_META_RECENT_TICKS) {
                    continue;
                }
                if (k < 0 || backend_ctx->plans[i].last_use < backend_ctx->plans[k].last_use) {
                    k = i;
                }
            }
        }
        if (k < 0) {
            for (int i = 0; i < GGML_META_N_PLANS; i++) {
                if (k < 0 || backend_ctx->plans[i].last_use < backend_ctx->plans[k].last_use) {
                    k = i;
                }
            }
        }
        GGML_ASSERT(k >= 0);

        ggml_backend_meta_plan & plan = backend_ctx->plans[k];
        backend_ctx->plan_clear(plan);

        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;

            // The simple tensors of this graph were placed into the container selected by init_tensor.
            const int s = buf_ctx->stc_compute_index_next;
            buf_ctx->stc_compute_index      = s;
            buf_ctx->stc_owner_uid[s]       = cgraph->uid;
            plan.slots.emplace_back(buf, s);

            // Prepare the container for the next graph allocation: prefer a free container, then the
            // least recently used one that is not in flight and holds no captured plan, then the least
            // recently used one that is not in flight, then any.
            int n = -1;
            for (int i = 0; i < GGML_META_N_STC; i++) {
                if (i != s && buf_ctx->stc_owner_uid[i] == 0) {
                    n = i;
                    break;
                }
            }
            if (n < 0) {
                for (int i = 0; i < GGML_META_N_STC; i++) {
                    if (i == s || ggml_backend_meta_hot_has(buf_ctx->stc_owner_uid[i])) {
                        continue;
                    }
                    if (now - buf_ctx->stc_last_use[i] <= GGML_META_RECENT_TICKS) {
                        continue;
                    }
                    if (n < 0 || buf_ctx->stc_last_use[i] < buf_ctx->stc_last_use[n]) {
                        n = i;
                    }
                }
            }
            if (n < 0) {
                for (int i = 0; i < GGML_META_N_STC; i++) {
                    if (i == s) {
                        continue;
                    }
                    if (now - buf_ctx->stc_last_use[i] <= GGML_META_RECENT_TICKS) {
                        continue;
                    }
                    if (n < 0 || buf_ctx->stc_last_use[i] < buf_ctx->stc_last_use[n]) {
                        n = i;
                    }
                }
            }
            if (n < 0) {
                for (int i = 0; i < GGML_META_N_STC; i++) {
                    if (i == s) {
                        continue;
                    }
                    if (n < 0 || buf_ctx->stc_last_use[i] < buf_ctx->stc_last_use[n]) {
                        n = i;
                    }
                }
            }
            GGML_ASSERT(n >= 0);

            if (buf_ctx->stc_owner_uid[n] != 0) {
                for (int kk = 0; kk < GGML_META_N_PLANS; kk++) {
                    if (backend_ctx->plans[kk].uid == buf_ctx->stc_owner_uid[n]) {
                        backend_ctx->plan_clear(backend_ctx->plans[kk]);
                    }
                }
            }
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[n];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            // clear() keeps the buckets; reserve them explicitly to avoid rehashing on the next graph
            const size_t n_simple_tensors = stc.simple_tensors.size();
            stc.simple_tensors.clear();
            stc.simple_tensors.reserve(n_simple_tensors);
            buf_ctx->stc_owner_uid[n]       = 0;
            buf_ctx->stc_compute_index_next = n;
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    plan.nodes[j][i] = node;
                    continue;
                }
                plan.nodes[j][i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(plan.nodes[j][i]);
            }
        }

        // Full attention nodes the model marked shareable: identify them before the collectives are
        // built and allocate the per-backend assist buffers (the boundary lands on the FA node).
        std::vector<ggml_backend_meta_assist_layer> assist_layers;
        std::vector<int> assist_at;
        {
            std::vector<ggml_backend_t> simple_backends(n_backends);
            for (size_t j = 0; j < n_backends; j++) {
                simple_backends[j] = backend_ctx->backend_configs[j].backend;
            }
            ggml_backend_meta_assist_build(cgraph, plan, simple_backends, backend_ctx->device_ordinal,
                backend_ctx->assist_scratch_layout, backend_ctx->assist_workspace_size,
                [&]() {
                    // one assist plan at a time: the workspace is hundreds of MB per device and
                    // the graphs that use it are submitted in sequence, so drop the captured
                    // plans of the previous graphs instead of keeping them all
                    for (auto & plan_other : backend_ctx->plans) {
                        if (&plan_other != &plan && !plan_other.buf_assist.empty()) {
                            backend_ctx->plan_clear(plan_other);
                        }
                    }
                },
                assist_layers, assist_at);
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_node_topk_collective(next) || ggml_backend_meta_node_getrows_collective(next) ||
                            ggml_backend_meta_node_pad_uneven(next)) {
                        // collective boundaries have to stay at subgraph ends
                        return i_delayed;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        const bool compute       = plan.nodes[j][i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = plan.nodes[j][i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            plan.collects.clear();
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                int collect_kind = GGML_BACKEND_META_COLLECT_ALLREDUCE;
                if (ggml_backend_meta_node_topk_collective(node)) {
                    collect_kind = GGML_BACKEND_META_COLLECT_TOPK;
                } else if (ggml_backend_meta_node_getrows_collective(node)) {
                    collect_kind = GGML_BACKEND_META_COLLECT_GETROWS;
                } else if (ggml_backend_meta_node_pad_uneven(node)) {
                    collect_kind = GGML_BACKEND_META_COLLECT_PAD;
                }
                // a shareable FA node ends its subgraph so that its assist sits right after it;
                // the FA node itself is computed by the subgraph, not by the collective
                const int i_assist = assist_at[i];
                if (i_assist >= 0) {
                    collect_kind = GGML_BACKEND_META_COLLECT_ASSIST;
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                                          collect_kind != GGML_BACKEND_META_COLLECT_ALLREDUCE;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = collect_kind == GGML_BACKEND_META_COLLECT_ALLREDUCE ? get_i_delayed(i) : i;

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        if ((plan.nodes[j][i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                plan.nodes[j][ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    plan.cgraphs[j][n_subgraphs].offset = i_start;
                }
                plan.collects.emplace_back();
                plan.collects.back().kind   = collect_kind;
                plan.collects.back().i_node = i;
                if (i_assist >= 0) {
                    plan.collects.back().assist = std::move(assist_layers[i_assist].desc);
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        plan.n_subgraphs = n_subgraphs;

        // the per-rank temporary buffers are only allocated by set_tmp_data when the fallback AllReduce actually runs
        backend_ctx->max_tmp_size = std::max(backend_ctx->max_tmp_size, max_tmp_size);

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            // the graph context is rebuilt: discard every other plan, its cgraphs live in the old context
            for (auto & plan_other : backend_ctx->plans) {
                if (&plan_other != &plan) {
                    backend_ctx->plan_clear(plan_other);
                }
            }
            // leave room for the assist boundaries: a graph whose FA nodes are marked shareable may be
            // rebuilt as an assist plan (prefill) with extra subgraphs, which must not discard the
            // plans of the decode graphs built before it
            size_t n_subgraphs_reserve = n_subgraphs;
            if (backend_ctx->assist_set != nullptr) {
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    const ggml_tensor * node = cgraph->nodes[i];
                    // every FA node, marked or not: the decode graphs are never marked but must
                    // leave room for a later assist plan of the same model
                    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                        n_subgraphs_reserve += 2;
                    }
                }
            }
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs_reserve);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = GGML_META_N_PLANS*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads); // one cgraph per subgraph and plan
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            // All cgraphs of all plans lived in the old context:
            for (int k = 0; k < GGML_META_N_PLANS; k++) {
                for (size_t j = 0; j < n_backends; j++) {
                    for (auto & cc : backend_ctx->plans[k].cgraphs[j]) {
                        cc.cgraph_main = nullptr;
                    }
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        plan.uid = cgraph->uid;

        for (size_t j = 0; j < n_backends; j++) {
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                if (plan.cgraphs[j][i_graph].cgraph_main == nullptr) {
                    plan.cgraphs[j][i_graph].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
                ggml_cgraph * cgraph_ij = plan.cgraphs[j][i_graph].cgraph_main;
                const size_t i_node_start = plan.cgraphs[j][i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? plan.cgraphs[j][i_graph + 1].offset : cgraph->n_nodes;
                size_t n_nodes_ij = i_node_stop - i_node_start;
                const int kind_ij = plan.collects[i_graph].kind;
                if (kind_ij != GGML_BACKEND_META_COLLECT_ALLREDUCE && kind_ij != GGML_BACKEND_META_COLLECT_ASSIST) {
                    // the boundary node itself is computed by the collective
                    GGML_ASSERT(plan.collects[i_graph].i_node == (int) i_node_stop - 1);
                    n_nodes_ij--;
                }
                cgraph_ij->n_nodes = n_nodes_ij;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_start + n_nodes_ij; i_node++) {
                    ggml_tensor * node_ij = plan.nodes[j][i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }

        // Build the auxiliary graphs of the cross-shard collectives and assign their memory.
        bool any_collective = false;
        for (const auto & collect : plan.collects) {
            any_collective = any_collective || collect.kind != GGML_BACKEND_META_COLLECT_ALLREDUCE;
        }
        plan.ctx_collect.clear();
        plan.ctx_collect.resize(n_backends);
        plan.buf_collect.resize(n_backends);
        if (any_collective) {
            const size_t collect_mem_size = plan.collects.size() *
                (64*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(64, /*grads =*/ false));
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_init_params params = {
                    /*.mem_size   =*/ collect_mem_size,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                plan.ctx_collect[j].reset(ggml_init(params));
            }
        }

        for (auto & collect : plan.collects) {
            collect.local.resize(n_backends);
            collect.merge.resize(n_backends);
            collect.reduce.resize(n_backends, nullptr);
            if (collect.kind == GGML_BACKEND_META_COLLECT_ALLREDUCE) {
                for (size_t j = 0; j < n_backends; j++) {
                    collect.reduce[j] = plan.nodes[j][collect.i_node];
                }
                continue;
            }
            if (collect.kind == GGML_BACKEND_META_COLLECT_ASSIST) {
                // the descriptors built with the buffers, no auxiliary graphs
                continue;
            }
            if (collect.kind == GGML_BACKEND_META_COLLECT_PAD) {
                // the node is computed by the per-backend local graphs only, no data crosses backends
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_context * ctx_j = plan.ctx_collect[j].get();
                    ggml_tensor  * node_j = plan.nodes[j][collect.i_node];
                    ggml_backend_meta_build_pad_local(ctx_j, node_j, collect.local[j]);
                }
                continue;
            }
            ggml_tensor * node = cgraph->nodes[collect.i_node];
            ggml_backend_meta_collect_plan collect_plan = {};
            collect_plan.k       = collect.kind == GGML_BACKEND_META_COLLECT_TOPK ? node->ne[0] : 0;
            collect_plan.n_cols  = ggml_nrows(node);
            collect_plan.n_slots = (int64_t) n_backends * collect_plan.k;

            const ggml_backend_meta_split_state src_ss = ggml_backend_meta_get_split_state(node->src[0], /*assume_sync =*/ true);
            // one contiguous row range per backend: piecewise segments would need per-row offsets
            GGML_ASSERT(src_ss.n_segments == 1 && src_ss.nr[0] == 1);
            GGML_ASSERT(node->src[0]->ne[src_ss.axis] < 0xFFFFFF); // ids must stay exact in F32

            for (size_t j = 0; j < n_backends; j++) {
                ggml_context * ctx_j = plan.ctx_collect[j].get();
                ggml_tensor  * node_j = plan.nodes[j][collect.i_node];
                collect_plan.n_rows = src_ss.ne[j];
                collect_plan.row0   = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    collect_plan.row0 += src_ss.ne[jj];
                }
                if (collect.kind == GGML_BACKEND_META_COLLECT_TOPK) {
                    ggml_tensor * cand = ggml_new_tensor_3d(ctx_j, GGML_TYPE_F32, collect_plan.n_slots, collect_plan.n_cols, 2);
                    // every backend contributes its own slot, so never zero the buffer during the all-reduce
                    cand->flags |= GGML_TENSOR_FLAG_COMPUTE;
                    collect.reduce[j] = cand;
                    ggml_backend_meta_build_topk_local(ctx_j, collect_plan, node_j->src[0], cand, (int64_t) j, collect.local[j]);
                    ggml_backend_meta_build_topk_merge(ctx_j, collect_plan, cand, node_j, collect.merge[j]);
                } else {
                    ggml_backend_meta_build_getrows_local(ctx_j, collect_plan, node_j->src[0], node_j->src[1], node_j, collect.local[j]);
                    collect.reduce[j] = node_j;
                }
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            if (plan.ctx_collect[j] == nullptr) {
                continue;
            }
            ggml_backend_t simple_backend = backend_ctx->backend_configs[j].backend;
            ggml_context * ctx_j = plan.ctx_collect[j].get();

            std::set<ggml_tensor *> bound;
            std::vector<ggml_backend_meta_aux_graph *> aux_graphs;
            for (auto & collect : plan.collects) {
                for (ggml_backend_meta_aux_graph * aux : { &collect.local[j], &collect.merge[j] }) {
                    if (aux->graph == nullptr) {
                        continue;
                    }
                    aux_graphs.push_back(aux);
                    for (const auto & b : aux->binds) {
                        bound.insert(b.tensor);
                    }
                }
            }

            size_t needed = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_j); t != nullptr; t = ggml_get_next_tensor(ctx_j, t)) {
                if (t->buffer == nullptr && t->view_src == nullptr) {
                    needed = (needed + 15) & ~size_t(15);
                    needed += ggml_nbytes(t);
                }
            }
            if (needed == 0) {
                // e.g. a plan whose only collectives are the assist ones
                continue;
            }
            if (!plan.buf_collect[j] || ggml_backend_buffer_get_size(plan.buf_collect[j].get()) < needed) {
                plan.buf_collect[j].reset(ggml_backend_alloc_buffer(simple_backend, needed));
            }

            size_t offset = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_j); t != nullptr; t = ggml_get_next_tensor(ctx_j, t)) {
                if (t->buffer != nullptr || t->view_src != nullptr || bound.count(t) != 0) {
                    continue;
                }
                offset = (offset + 15) & ~size_t(15);
                ggml_backend_tensor_alloc(plan.buf_collect[j].get(), t, (char *) ggml_backend_buffer_get_base(plan.buf_collect[j].get()) + offset);
                offset += ggml_nbytes(t);
            }
            for (ggml_backend_meta_aux_graph * aux : aux_graphs) {
                for (const auto & b : aux->binds) {
                    ggml_backend_tensor_alloc(b.dst->buffer, b.tensor, (char *) b.dst->data + b.offset);
                }
            }
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_j); t != nullptr; t = ggml_get_next_tensor(ctx_j, t)) {
                if (t->buffer == nullptr) {
                    GGML_ASSERT(t->view_src != nullptr);
                    ggml_backend_view_init(t);
                }
            }
        }

        const uint64_t tick = ++ggml_backend_meta_tick;
        plan.last_use = tick;
        for (const auto & slot : plan.slots) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) slot.first->context;
            buf_ctx->stc_last_use[slot.second] = tick;
        }
        p = &plan;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            // retire the old buffer: captured executables may still reference it
            if (buf_ptr) {
                backend_ctx->tmp_bufs_retired.push_back(std::move(buf_ptr));
            }
            // allocate a power of two so that repeated growth retires less than the current size in total
            size_t size = 1;
            while (size < backend_ctx->max_tmp_size) {
                size *= 2;
            }
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](std::vector<ggml_tensor *> & nodes) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = nodes[j];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = nodes[j_src];
            ggml_tensor * node_dst = nodes[j_dst];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, nodes[j - 2*offset_j_max], nodes[j]);
        }

        return GGML_STATUS_SUCCESS;
    };

    auto allreduce = [&](std::vector<ggml_tensor *> & nodes) -> ggml_status {
        if (backend_ctx->comm_ctx != nullptr && backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data())) {
            return GGML_STATUS_SUCCESS;
        }
        return allreduce_fallback(nodes);
    };

    auto allreduce_exact = [&](std::vector<ggml_tensor *> & nodes) -> ggml_status {
        if (backend_ctx->comm_ctx != nullptr && backend_ctx->comm_allreduce_exact != nullptr &&
            backend_ctx->comm_allreduce_exact(backend_ctx->comm_ctx, nodes.data())) {
            return GGML_STATUS_SUCCESS;
        }
        return allreduce_fallback(nodes);
    };


    GGML_ASSERT(p != nullptr);

    // Prefer the per-rank launcher threads when every collective of this plan can be
    // driven per rank. The check is cheap and done on every call because the plan can
    // change between calls. The first runs of a plan stay on this thread: they capture and
    // instantiate the CUDA graphs, load kernels and grow the memory pools, i.e. driver calls
    // that can wait for the GPU while holding driver locks (see ggml_backend_meta_launchers).
    //
    // After its first run (kernels loaded, pool memory grown) a plan is recorded into one executable per
    // rank if every collective can be recorded, i.e. none takes NCCL. The recording skips the CUDA graphs
    // of the individual subgraphs, and nothing runs until every rank has instantiated its executable.
    const bool try_capture = backend_ctx->launchers != nullptr && p->full_state == 0 && p->n_runs >= 1 &&
                             backend_ctx->launchers->can_capture() && backend_ctx->comm_allreduce_rank_capturable != nullptr;
    if (backend_ctx->launchers != nullptr && (p->n_runs >= 2 || try_capture)) {
        bool can_launch  = true;
        bool can_capture = try_capture;
        for (size_t i = 0; i < p->n_subgraphs && can_launch && p->full_state != 1; i++) {
            const ggml_backend_meta_collect & collect = p->collects[i];
            const bool gather = collect.kind == GGML_BACKEND_META_COLLECT_TOPK ||
                                collect.kind == GGML_BACKEND_META_COLLECT_GETROWS;
            const bool do_reduce = collect.kind == GGML_BACKEND_META_COLLECT_ALLREDUCE
                ? n_backends > 1 && i < p->n_subgraphs - 1
                : gather && n_backends > 1;
            if (!do_reduce) {
                continue;
            }
            const bool exact = gather;
            can_launch  = backend_ctx->launchers->can(p->collects[i].reduce.data(), exact);
            can_capture = can_capture && can_launch &&
                backend_ctx->comm_allreduce_rank_capturable(backend_ctx->comm_ctx, p->collects[i].reduce.data(), exact);
        }
        if (can_launch && (p->n_runs >= 2 || can_capture)) {
            const ggml_status status = backend_ctx->launchers->compute(p, can_capture);
            if (status == GGML_STATUS_SUCCESS) {
                p->n_runs++;
            }
            return status;
        }
    }

    for (size_t i = 0; i < p->n_subgraphs; i++) {
        ggml_backend_meta_collect & collect = p->collects[i];
        const bool assist = collect.kind == GGML_BACKEND_META_COLLECT_ASSIST;
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            if (assist && collect.assist[j].role == 0) {
                if (!backend_ctx->assist_set(bcj.backend, &collect.assist[j])) {
                    return GGML_STATUS_FAILED;
                }
            }
            if (p->cgraphs[j][i].cgraph_main->n_nodes > 0) {
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, p->cgraphs[j][i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            if (assist && collect.assist[j].role == 0) {
                if (!backend_ctx->assist_set(bcj.backend, nullptr)) {
                    return GGML_STATUS_FAILED;
                }
            }
        }

        if (assist) {
            // the idle rank copies and attends the owner's KV tail, the owner is a no-op here
            for (size_t j = 0; j < n_backends; j++) {
                if (!backend_ctx->assist_run(backend_ctx->backend_configs[j].backend, &collect.assist[j])) {
                    return GGML_STATUS_FAILED;
                }
            }
            continue;
        }
        if (collect.kind != GGML_BACKEND_META_COLLECT_ALLREDUCE) {
            for (size_t j = 0; j < n_backends; j++) {
                if (collect.local[j].graph == nullptr) {
                    continue;
                }
                const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, collect.local[j].graph);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            const bool gather = collect.kind == GGML_BACKEND_META_COLLECT_TOPK ||
                                collect.kind == GGML_BACKEND_META_COLLECT_GETROWS;
            if (gather && n_backends > 1) {
                // These all-reduces gather disjoint contributions, e.g. the TOP_K
                // candidates with their token ids stored as f32 (as f16, ids above
                // 65504 would become inf), so they must not take a lossy reduction.
                const ggml_status status = allreduce_exact(collect.reduce);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            if (collect.kind == GGML_BACKEND_META_COLLECT_TOPK) {
                for (size_t j = 0; j < n_backends; j++) {
                    const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, collect.merge[j].graph);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
            }
        } else if (n_backends > 1 && i < p->n_subgraphs - 1) {
            // Ordinary PARTIAL sums can take the lossy fast path.
            const ggml_status status = allreduce(collect.reduce);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    p->n_runs++;
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_meta_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
