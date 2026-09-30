#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

// tests for the speculative rejection core: the draft token is sampled from q and
// the output token must follow p; fixed seeds keep the test deterministic

static const int n_samples = 100000;

struct test_rng {
    std::mt19937 gen;

    explicit test_rng(uint32_t seed) : gen(seed) {}

    float uniform() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(gen);
    }
};

static float test_uniform(void * data) {
    return ((test_rng *) data)->uniform();
}

static llama_token sample_probs(const std::vector<float> & probs, test_rng & rng) {
    const float tgt = rng.uniform();

    float acc = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (acc >= tgt) {
            return (llama_token) i;
        }
    }

    return (llama_token) (probs.size() - 1);
}

static common_sampler_sparse_probs make_p(const std::vector<float> & probs) {
    common_sampler_sparse_probs p;
    for (size_t i = 0; i < probs.size(); ++i) {
        p.ids.push_back((llama_token) i);
        p.probs.push_back(probs[i]);
    }
    return p;
}

static common_sampler_draft_q make_q(const std::vector<float> & probs) {
    common_sampler_draft_q q;
    for (size_t i = 0; i < probs.size(); ++i) {
        q.ids.push_back((llama_token) i);
        q.q.push_back(probs[i]);
    }
    return q;
}

static double tv_distance(const std::vector<double> & counts, const std::vector<float> & p, int n) {
    double tv = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        tv += fabs(counts[i]/n - p[i]);
    }
    return 0.5*tv;
}

// single draft step: the first output token must follow p
static void test_single_step(const std::vector<float> & p, const std::vector<float> & q, uint32_t seed) {
    test_rng rng(seed);

    const std::vector<common_sampler_sparse_probs> p_rows  = { make_p(p), make_p(p) };
    const std::vector<common_sampler_draft_q>      draft_q = { make_q(q) };

    std::vector<double> counts(p.size(), 0.0);

    for (int i = 0; i < n_samples; ++i) {
        const llama_tokens draft = { sample_probs(q, rng) };

        const std::vector<llama_token> out = common_sampler_reject_core(p_rows, draft, draft_q, test_uniform, &rng);
        assert(out.size() == 1 || out.size() == 2);
        assert(out[0] >= 0 && (size_t) out[0] < p.size());

        counts[out[0]] += 1.0;
    }

    assert(tv_distance(counts, p, n_samples) < 0.01);
}

static void test_multi_step() {
    // q1 == p1, so the first draft token is always accepted
    const std::vector<float> q1 = { 0.05f, 0.10f, 0.15f, 0.20f, 0.20f, 0.15f, 0.10f, 0.05f };
    const std::vector<float> p1 = q1;
    // the second draft token is always token 0, which has p2[0] < 1
    const std::vector<float> q2 = { 1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f };
    const std::vector<float> p2 = { 0.30f, 0.20f, 0.10f, 0.10f, 0.10f, 0.10f, 0.05f, 0.05f };
    const std::vector<float> p3 = { 0.20f, 0.20f, 0.15f, 0.15f, 0.10f, 0.10f, 0.05f, 0.05f };

    test_rng rng(42);

    const std::vector<common_sampler_sparse_probs> p_rows  = { make_p(p1), make_p(p2), make_p(p3) };
    const std::vector<common_sampler_draft_q>      draft_q = { make_q(q1), make_q(q2) };

    std::vector<double> counts1(p1.size(), 0.0);
    std::vector<double> counts2(p2.size(), 0.0);
    std::vector<double> counts3(p3.size(), 0.0);
    int n_bonus = 0;

    for (int i = 0; i < n_samples; ++i) {
        const llama_tokens draft = { sample_probs(q1, rng), sample_probs(q2, rng) };

        const std::vector<llama_token> out = common_sampler_reject_core(p_rows, draft, draft_q, test_uniform, &rng);
        assert(out.size() >= 2 && out.size() <= 3);

        counts1[out[0]] += 1.0;
        counts2[out[1]] += 1.0;

        if (out.size() == 3) {
            counts3[out[2]] += 1.0;
            n_bonus++;
        }
    }

    assert(tv_distance(counts1, p1, n_samples) < 0.01);
    assert(tv_distance(counts2, p2, n_samples) < 0.01);

    // bonus tokens only occur when all draft tokens are accepted
    assert(n_bonus > 1000);
    assert(tv_distance(counts3, p3, n_bonus) < 0.03);
}

static std::vector<float> softmax(const llama_token_data * data, size_t n) {
    float max_l = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        max_l = std::max(max_l, data[i].logit);
    }

    std::vector<float> probs(n);

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        probs[i] = expf(data[i].logit - max_l);
        sum += probs[i];
    }
    for (size_t i = 0; i < n; ++i) {
        probs[i] = (float) (probs[i] / sum);
    }

    return probs;
}

static void test_sparse_penalties() {
    const int32_t n_vocab = 256;
    const int32_t top_k   = 20;
    const int32_t last_n  = 32;
    const float   freq    = 2.0f;
    const float   present = 5.0f;

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> logit_dist(-4.0f, 4.0f);

    std::vector<llama_token_data> row;
    row.reserve(n_vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        row.push_back({ id, logit_dist(gen), 0.0f });
    }

    // the penalty window holds the best logits, so that penalizing them pushes them out of the raw
    // top_k and replacements from below the cutoff are needed
    std::vector<llama_token_data> sorted = row;
    std::sort(sorted.begin(), sorted.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    const int32_t n_distinct = 24;

    std::vector<llama_token> prev;
    for (int32_t i = 0; i < n_distinct; ++i) {
        prev.push_back(sorted[i].id);
    }
    for (int32_t i = 0; prev.size() < (size_t) last_n; ++i) {
        prev.push_back(sorted[i % n_distinct].id);
    }

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, last_n, 1.0f, freq, present));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));

    for (llama_token id : prev) {
        llama_sampler_accept(chain, id);
    }

    // full vocabulary row through the chain
    std::vector<llama_token_data> full = row;
    llama_token_data_array cur_full = { full.data(), full.size(), -1, false };
    llama_sampler_apply(chain, &cur_full);

    // sparse row: the raw top_k + n_distinct candidates, as the rejection path builds them
    std::vector<llama_token_data> sparse = sorted;
    sparse.resize(top_k + n_distinct);
    llama_token_data_array cur_sparse = { sparse.data(), sparse.size(), -1, false };
    llama_sampler_apply(chain, &cur_sparse);

    assert(cur_full.size   == (size_t) top_k);
    assert(cur_sparse.size == (size_t) top_k);

    const std::vector<float> p_full   = softmax(cur_full.data,   cur_full.size);
    const std::vector<float> p_sparse = softmax(cur_sparse.data, cur_sparse.size);

    for (size_t i = 0; i < cur_full.size; ++i) {
        const llama_token id = cur_full.data[i].id;

        size_t j = 0;
        while (j < cur_sparse.size && cur_sparse.data[j].id != id) {
            ++j;
        }
        assert(j < cur_sparse.size);
        assert(fabs(cur_full.data[i].logit - cur_sparse.data[j].logit) < 1e-5f);
        assert(fabs(p_full[i] - p_sparse[j]) < 1e-5f);
    }

    // top_k candidates alone keep the penalized tokens: the window must be reserved
    std::vector<llama_token_data> short_row = sorted;
    short_row.resize(top_k);
    llama_token_data_array cur_short = { short_row.data(), short_row.size(), -1, false };
    llama_sampler_apply(chain, &cur_short);
    assert(cur_short.data[0].id != cur_full.data[0].id);

    llama_sampler_free(chain);

    printf("sparse penalties OK\n");
}

int main() {
    // p == q
    test_single_step(
            { 0.05f, 0.10f, 0.15f, 0.20f, 0.20f, 0.15f, 0.10f, 0.05f },
            { 0.05f, 0.10f, 0.15f, 0.20f, 0.20f, 0.15f, 0.10f, 0.05f }, 1);

    // disjoint supports
    test_single_step(
            { 0.40f, 0.30f, 0.20f, 0.10f, 0.00f, 0.00f, 0.00f, 0.00f },
            { 0.00f, 0.00f, 0.00f, 0.00f, 0.40f, 0.30f, 0.20f, 0.10f }, 2);

    // q is a point mass
    test_single_step(
            { 0.30f, 0.20f, 0.10f, 0.10f, 0.10f, 0.10f, 0.05f, 0.05f },
            { 1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f }, 3);

    // p_x == 0 for the drafted token
    test_single_step(
            { 0.00f, 0.50f, 0.30f, 0.20f, 0.00f, 0.00f, 0.00f, 0.00f },
            { 1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f }, 4);

    // p == q point mass
    test_single_step(
            { 0.00f, 0.00f, 0.00f, 1.00f, 0.00f, 0.00f, 0.00f, 0.00f },
            { 0.00f, 0.00f, 0.00f, 1.00f, 0.00f, 0.00f, 0.00f, 0.00f }, 5);

    test_multi_step();
    test_sparse_penalties();

    return 0;
}
