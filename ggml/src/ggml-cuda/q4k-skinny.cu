// Q4_K skinny GEMM for sm_70 (Volta).
//
// Design follows the QPN8 execution layout of q8-skinny.cu (itself adapted from 1Cat-vLLM
// fp8_qpn8_sm70.cu, Apache-2.0) and the int4 scale/bias folding of 1Cat's awq_qpn_m1_sm70.cu
// and ninfer-v100's q4 Volta kernels. The QPN8 layout is derived from dnv2003/v100-skinny
// (MIT) and its block-scale adaptation in haohervchb/sglang-V100. See LICENSE.v100-skinny in
// this directory and docs/design/v106-q4k-skinny.md for the layout and the numeric analysis.

#include "common.cuh"
#include "convert.cuh"
#include "q-skinny-common.cuh"
#include "q4k-skinny.cuh"

#include <algorithm>
#include <vector>

// sentinel object whose address tags a repacked Q4_K tensor (not const: it is stored in tensor->extra)
static char q4k_skinny_marker = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---- codec: per-type code decode plus scale/min parsing ----

// One loaded record: the record bytes as two little-endian 64-bit words, hi zero padded. The
// bits of one value sit at bit offset slot*bits_per_value of the record, slot = qpn8_physical_k.
struct qskinny_code {
    unsigned long long lo, hi;
};

// record buffers of the sizes that have no matching builtin load type
struct qskinny_rec6  { uint8_t b[6];  };
struct qskinny_rec10 { uint8_t b[10]; };
struct qskinny_rec12 { uint8_t b[12]; };

template <ggml_type T> struct qskinny_codec;

static __device__ __forceinline__ unsigned qskinny_code_get(const qskinny_code rec, const int off, const int bits) {
    const unsigned mask = (1u << bits) - 1;
    if (off + bits <= 64) {
        return (unsigned) (rec.lo >> off) & mask;
    }
    if (off >= 64) {
        return (unsigned) (rec.hi >> (off - 64)) & mask;
    }
    return (unsigned) ((rec.lo >> off) | (rec.hi << (64 - off))) & mask;
}

static __device__ __forceinline__ void qskinny_code_put(qskinny_code & rec, const unsigned code, const int off, const int bits) {
    if (off < 64) {
        rec.lo |= (unsigned long long) code << off;
        if (off + bits > 64) {
            rec.hi |= (unsigned long long) code >> (64 - off);
        }
    } else {
        rec.hi |= (unsigned long long) code << (off - 64);
    }
}

// get_scale_min_k4 from dequantize.cuh on two little-endian 64-bit words of scale bytes
static __device__ __forceinline__ uint8_t qskinny_scale_byte(const unsigned long long s_lo,
                                                             const unsigned long long s_hi, const int i) {
    return (uint8_t) ((i < 8 ? s_lo : s_hi) >> (8 * (i & 7)));
}

// One record holds values_per_record raw codes in qpn8 physical slot order; meta holds the raw
// super-block header. 16*record_bytes + meta_bytes == block size, so a repacked tensor keeps the
// byte count of its quantized type and the repack stays in place. decode() turns the two records
// of one 32-value sub-block plus the meta of its super-block into 16 half2 weights.
//
// Q4_K: w = d*sc*q - dmin*m folded into one half FMA. Q2_K/Q3_K/Q5_K/Q6_K fold the same way
// (their raw code lands exact in the 1024 + code bit pattern), so one hsub2 + one half FMA or
// multiply per value pair; the f16 result matches the float expression of dequantize.cuh up to
// the single f16 rounding of the scale, exactly as the Q4_K decoder has always done.

// ---- Q4_K: 8 code bytes per 16 K values and lane (16 nibbles, physical slot s in byte s/2,
// half s&1) and 16 meta bytes per 256 K values (the raw super-block header). ----

template <> struct qskinny_codec<GGML_TYPE_Q4_K> {
    using record_t = uint2;
    using meta_t   = uint4;
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 16;
    static constexpr int bits_per_value    = 4;
    static constexpr int block_bytes       = 144;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const uint2 r = __ldcs(p);
        return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint2 *>(p) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    // the raw 16-byte super-block header (dm + scales), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return *reinterpret_cast<const meta_t *>(block);
    }

    // a 32-value sub-block uses 32 consecutive qs bytes: low nibbles for even sub-blocks, high
    // nibbles for odd ones; group g covers the first (g&1 == 0) or second 16 values
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const uint8_t * qs = block + 16;
        const int qs_base  = 32 * (g >> 2) + ((g & 1) ? 16 : 0);
        const int qs_shift = (g & 2) ? 4 : 0;
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[qs_base + j] >> qs_shift) & 0xF, 4 * slot, 4);
        }
        return rec;
    }

    static __device__ __forceinline__ uint8_t meta_byte(const uint4 meta, const int i) {
        // scales[12] starts at byte 4 of the super-block header
        const unsigned word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w;
        return (uint8_t) (word >> (8 * (i & 3)));
    }

    // get_scale_min_k4 from dequantize.cuh, applied to the raw meta bytes
    static __device__ __forceinline__ void get_scale_min(const uint4 meta, const int j,
                                                         uint8_t & sc, uint8_t & m) {
        if (j < 4) {
            sc = meta_byte(meta, j) & 63;
            m  = meta_byte(meta, j + 4) & 63;
        } else {
            sc = (meta_byte(meta, j + 4) & 0x0F) | ((meta_byte(meta, j - 4) >> 6) << 4);
            m  = (meta_byte(meta, j + 4) >> 4)   | ((meta_byte(meta, j)     >> 6) << 4);
        }
    }

    // One record holds 16 nibbles in physical order. The m8n8k4 B fragment pairs physical
    // slots (s, s+4), exactly like the Q8_0 decoder pairs bytes (i, i+4).
    static __device__ __forceinline__ void decode_record(const qskinny_code rec, half2 out[8]) {
        const unsigned rx = (unsigned) rec.lo;
        const unsigned ry = (unsigned) (rec.lo >> 32);
        // the second byte goes to result byte 2, where the mask below reads it
        const unsigned pairs[4] = {
            __byte_perm(rx, 0, 0x0200),  // bytes (0, 2)
            __byte_perm(rx, 0, 0x0301),  // bytes (1, 3)
            __byte_perm(ry, 0, 0x0200),  // bytes (4, 6)
            __byte_perm(ry, 0, 0x0301),  // bytes (5, 7)
        };
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned lo = (pairs[i] & 0x000F000Fu) | offset_bits;
            const unsigned hi = ((pairs[i] >> 4) & 0x000F000Fu) | offset_bits;
            out[2*i + 0] = __hsub2(*reinterpret_cast<const half2 *>(&lo), offset);
            out[2*i + 1] = __hsub2(*reinterpret_cast<const half2 *>(&hi), offset);
        }
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const uint4 meta,
                                                  const int sub_block, half2 out[16]) {
        decode_record(records[0], out);
        decode_record(records[1], out + 8);

        uint8_t sc, m;
        get_scale_min(meta, sub_block, sc, m);
        const half2 dm = *reinterpret_cast<const half2 *>(&meta.x);
        const float d    = __half2float(__low2half(dm));
        const float dmin = __half2float(__high2half(dm));
        // f32 folding, rounded once to f16 (see 3.4 of the design note)
        const half2 scale2 = __float2half2_rn(d * (float) sc);
        const half2 bias2  = __float2half2_rn(-dmin * (float) m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hfma2(out[i], scale2, bias2);
        }
    }
};

// ---- Q5_K: same 16-byte meta as Q4_K (dm + scales), 5 code bits per value: the low 4 bits as
// in Q4_K, the high bit from qh at byte j_in_sub, bit sub_block. ----

template <> struct qskinny_codec<GGML_TYPE_Q5_K> {
    using record_t = qskinny_rec10;
    using meta_t   = uint4;
    static constexpr int record_bytes      = 10;
    static constexpr int meta_bytes        = 16;
    static constexpr int bits_per_value    = 5;
    static constexpr int block_bytes       = 176;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        qskinny_code rec = { 0, 0 };
        rec.lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                 ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        rec.hi = w[4];
        return rec;
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i] = (unsigned short) (rec.lo >> (16 * i));
        }
        w[4] = (unsigned short) rec.hi;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return *reinterpret_cast<const meta_t *>(block);
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * ql = block + 48;
        const uint8_t * qh = block + 16;
        const int ql_base  = 32 * (g >> 2) + ((g & 1) ? 16 : 0);
        const int ql_shift = (g & 2) ? 4 : 0;
        const int qh_base  = 16 * (g & 1);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned code = ((ql[ql_base + j] >> ql_shift) & 0xF) |
                                  (((qh[qh_base + j] >> sb) & 1) << 4);
            qskinny_code_put(rec, code, 5 * slot, 5);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        uint8_t sc, m;
        // the meta header has the Q4_K layout (dm + 12 scale bytes)
        qskinny_codec<GGML_TYPE_Q4_K>::get_scale_min(meta, sub_block, sc, m);
        const half2 dm = *reinterpret_cast<const half2 *>(&meta.x);
        const float dall = __half2float(__low2half(dm));
        const float dmin = __half2float(__high2half(dm));
        // f16 folding, rounded once to f16 like the Q4_K decoder: the 5-bit code lands exact
        // in the 1024 + code bit pattern, one hsub2 + one hfma2 per pair
        const half2 scale2 = __float2half2_rn(dall * (float) sc);
        const half2 bias2  = __float2half2_rn(-dmin * (float) m);
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                // the B fragment pairs physical slots (s, s+4), logical values (2i, 2i+1)
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned q0 = qskinny_code_get(records[r], 5 * s, 5);
                const unsigned q1 = qskinny_code_get(records[r], 5 * (s + 4), 5);
                const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
                out[8*r + i] = __hfma2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset),
                                       scale2, bias2);
            }
        }
    }
};

// ---- Q6_K: 18 meta bytes per 256 K values (16 int8 scales + d), 6 code bits per value: the
// low 4 bits as in Q4_K, the high 2 bits from qh at byte 32*(sub_block>>2) + j_in_sub. ----

template <> struct qskinny_codec<GGML_TYPE_Q6_K> {
    using record_t = qskinny_rec12;
    struct meta_t {
        unsigned long long s_lo, s_hi;
        unsigned short d;
    };
    static constexpr int record_bytes      = 12;
    static constexpr int meta_bytes        = 18;
    static constexpr int bits_per_value    = 6;
    static constexpr int block_bytes       = 210;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned * w = reinterpret_cast<const unsigned *>(p);
        return { (unsigned long long) w[0] | ((unsigned long long) w[1] << 32), w[2] };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                    ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        meta.s_hi = (unsigned long long) w[4] | ((unsigned long long) w[5] << 16) |
                    ((unsigned long long) w[6] << 32) | ((unsigned long long) w[7] << 48);
        meta.d = w[8];
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned * w = reinterpret_cast<unsigned *>(p);
        w[0] = (unsigned) rec.lo;
        w[1] = (unsigned) (rec.lo >> 32);
        w[2] = (unsigned) rec.hi;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i]     = (unsigned short) (meta.s_lo >> (16 * i));
            w[4 + i] = (unsigned short) (meta.s_hi >> (16 * i));
        }
        w[8] = meta.d;
    }

    // scales[16] + d, copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta(block + 192);
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * ql = block;
        const uint8_t * qh = block + 128;
        const int ql_base  = 32 * (sb & 1) + 16 * (g & 1) + 64 * (sb >> 2);
        const int ql_shift = (sb & 2) ? 4 : 0;
        const int qh_base  = 16 * (g & 1) + 32 * (sb >> 2);
        const int qh_shift = 2 * (sb & 3);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned code = ((ql[ql_base + j] >> ql_shift) & 0xF) |
                                  (((qh[qh_base + j] >> qh_shift) & 3) << 4);
            qskinny_code_put(rec, code, 6 * slot, 6);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float d = __half2float(__ushort_as_half(meta.d));
        // f16 folding: 1024 + q is exact for the 6-bit code, one hsub2 gives q - 32
        const unsigned offset_bits = 0x64206420u; // f16 1056 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int sc = (int8_t) qskinny_scale_byte(meta.s_lo, meta.s_hi, 2 * sub_block + r);
            const half2 scale2 = __float2half2_rn(d * (float) sc);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned q0 = qskinny_code_get(records[r], 6 * s, 6);
                const unsigned q1 = qskinny_code_get(records[r], 6 * (s + 4), 6);
                const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
                out[8*r + i] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset), scale2);
            }
        }
    }
};

// ---- Q2_K: 20 meta bytes per 256 K values (dm + 16 scale/min bytes), 2 code bits per value. ----

template <> struct qskinny_codec<GGML_TYPE_Q2_K> {
    using record_t = uint32_t;
    struct meta_t {
        unsigned long long s_lo, s_hi;
        unsigned dm;
    };
    static constexpr int record_bytes      = 4;
    static constexpr int meta_bytes        = 20;
    static constexpr int bits_per_value    = 2;
    static constexpr int block_bytes       = 84;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        return { __ldcs(p), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned * w = reinterpret_cast<const unsigned *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[1] | ((unsigned long long) w[2] << 32);
        meta.s_hi = (unsigned long long) w[3] | ((unsigned long long) w[4] << 32);
        meta.dm   = w[0];
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint32_t *>(p) = (uint32_t) rec.lo;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned * w = reinterpret_cast<unsigned *>(p);
        w[0] = meta.dm;
        w[1] = (unsigned) meta.s_lo;
        w[2] = (unsigned) (meta.s_lo >> 32);
        w[3] = (unsigned) meta.s_hi;
        w[4] = (unsigned) (meta.s_hi >> 32);
    }

    // dm lives after qs in the block, the scales before it: meta keeps dm first
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta_from(block + 80, block);
    }

    static __device__ __forceinline__ meta_t load_meta_from(const uint8_t * dm_src, const uint8_t * scales_src) {
        meta_t meta;
        meta.dm   = *reinterpret_cast<const unsigned *>(dm_src);
        meta.s_lo = (unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[0] |
                    ((unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[1] << 32);
        meta.s_hi = (unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[2] |
                    ((unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[3] << 32);
        return meta;
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * qs = block + 16;
        const int qs_base  = 32 * (sb >> 2) + 16 * (g & 1);
        const int qs_shift = 2 * (sb & 3);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[qs_base + j] >> qs_shift) & 3, 2 * slot, 2);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float dall = __half2float(__ushort_as_half((unsigned short) meta.dm));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (meta.dm >> 16)));
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int is = 2 * sub_block + r;
            const int sc = qskinny_scale_byte(meta.s_lo, meta.s_hi, is) & 0xF;
            const int mn = qskinny_scale_byte(meta.s_lo, meta.s_hi, is) >> 4;
            // f16 folding, rounded once to f16 like the Q4_K decoder
            const half2 scale2 = __float2half2_rn(dall * (float) sc);
            const half2 bias2  = __float2half2_rn(-dmin * (float) mn);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned q0 = qskinny_code_get(records[r], 2 * s, 2);
                const unsigned q1 = qskinny_code_get(records[r], 2 * (s + 4), 2);
                const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
                out[8*r + i] = __hfma2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset),
                                       scale2, bias2);
            }
        }
    }
};

// ---- Q3_K: 14 meta bytes per 256 K values (12 scale bytes + d), 3 code bits per value: the
// low 2 bits as in Q2_K, the sign bit from hmask at byte j_in_sub, bit sub_block. ----

template <> struct qskinny_codec<GGML_TYPE_Q3_K> {
    using record_t = qskinny_rec6;
    struct meta_t {
        unsigned long long s_lo, s_hi;
    };
    static constexpr int record_bytes      = 6;
    static constexpr int meta_bytes        = 14;
    static constexpr int bits_per_value    = 3;
    static constexpr int block_bytes       = 110;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        return { (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                 ((unsigned long long) w[2] << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                    ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        // scales[8..11] then d, zero padded into the second word
        meta.s_hi = (unsigned long long) w[4] | ((unsigned long long) w[5] << 16) |
                    ((unsigned long long) w[6] << 32);
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        w[0] = (unsigned short) rec.lo;
        w[1] = (unsigned short) (rec.lo >> 16);
        w[2] = (unsigned short) (rec.lo >> 32);
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i] = (unsigned short) (meta.s_lo >> (16 * i));
        }
        for (int i = 0; i < 3; ++i) {
            w[4 + i] = (unsigned short) (meta.s_hi >> (16 * i));
        }
    }

    // scales[12] + d, copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta(block + 96);
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * qs    = block + 32;
        const uint8_t * hmask = block;
        const int qs_base  = 32 * (sb >> 2) + 16 * (g & 1);
        const int qs_shift = 2 * (sb & 3);
        const int hm_base  = 16 * (g & 1);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned code = ((qs[qs_base + j] >> qs_shift) & 3) |
                                  (((hmask[hm_base + j] >> sb) & 1) << 2);
            qskinny_code_put(rec, code, 3 * slot, 3);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float d_all = __half2float(__ushort_as_half((unsigned short) (meta.s_hi >> 32)));
        // f16 folding: the effective code is code2 - (hbit ? 0 : 4), in [鈭?, 3]. 1024 + code2 + 8
        // (hbit set) or 1024 + code2 + 4 (hbit clear) minus 1032 gives it exactly, in one hsub2.
        const unsigned offset_bits = 0x64086408u; // f16 1032 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int is = 2 * sub_block + r;
            // the 6-bit scale unpack of dequantize.cuh
            const int8_t us = is <  4 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 0) & 0xF) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 8) >> 0) & 3) << 4) :
                              is <  8 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 0) & 0xF) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 4) >> 2) & 3) << 4) :
                              is < 12 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 8) >>  4) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 0) >> 4) & 3) << 4) :
                                        (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 8) >>  4) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 4) >> 6) & 3) << 4);
            const half2 scale2 = __float2half2_rn(d_all * (float) (us - 32));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned c0 = qskinny_code_get(records[r], 3 * s, 3);
                const unsigned c1 = qskinny_code_get(records[r], 3 * (s + 4), 3);
                const unsigned h0 = 0x6400u | ((c0 & 3) + ((c0 & 4) ? 8 : 4));
                const unsigned h1 = 0x6400u | ((c1 & 3) + ((c1 & 4) ? 8 : 4));
                const unsigned h = h0 | (h1 << 16);
                out[8*r + i] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset), scale2);
            }
        }
    }
};

static_assert(qskinny_codec<GGML_TYPE_Q2_K>::block_bytes == sizeof(block_q2_K), "q2_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q3_K>::block_bytes == sizeof(block_q3_K), "q3_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q4_K>::block_bytes == sizeof(block_q4_K), "q4_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q5_K>::block_bytes == sizeof(block_q5_K), "q5_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q6_K>::block_bytes == sizeof(block_q6_K), "q6_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q2_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q2_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q2_K>::block_bytes, "q2_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q3_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q3_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q3_K>::block_bytes, "q3_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q4_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q4_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q4_K>::block_bytes, "q4_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q5_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q5_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q5_K>::block_bytes, "q5_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q6_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q6_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q6_K>::block_bytes, "q6_K packing");

// ---- repack: row-major block -> codes + meta ----

// One CTA per tile. src points at the first tile of the range, codes at its codes
// destination and meta at the full meta array indexed by the global tile.
template <ggml_type T>
__global__ void qskinny_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ codes,
                                      uint8_t * __restrict__ meta, int n, int k, int tile0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    const int blocks_k = k >> 8;
    const int tile = tile0 + blockIdx.x;
    uint8_t * codes_tile = codes + (size_t) blockIdx.x * 32 * (k >> 4) * codec::record_bytes;
    for (int i = threadIdx.x; i < 32 * blocks_k; i += blockDim.x) {
        const int row = i / blocks_k;
        const int kb  = i - row * blocks_k;
        const uint8_t * block = src + ((size_t) blockIdx.x * 32 + row) * blocks_k * codec::block_bytes
                                         + (size_t) kb * codec::block_bytes;
        const int lane = qpn8_lane_from_col(row & 31);

        codec::store_meta(codec::pack_meta(block),
                          meta + ((size_t) kb * n + (size_t) tile * 32 + lane) * codec::meta_bytes);

        // 16 groups of 16 values per super-block, group g holds the first (g&1 == 0) or second
        // 16 values of sub-block g/2
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            codec::store_record(codec::pack_record(block, g),
                                codes_tile + (((size_t) kb * 16 + g) * 32 + lane) * codec::record_bytes);
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(src, codes, meta, n, k, tile0);
#endif
}

// ---- dequantize: codes + meta -> dense F16 [N][K], same layout as the Q4_K to F16 conversion ----

// One CTA per (tile of 32 rows, 256-K super-block). A warp decodes one 32-value sub-block per
// lane; a code record decodes to one output row, so the warps stage the decoded values in
// shared memory first. That keeps the stores to global memory coalesced along K instead of one
// 4-byte store per row.
template <ggml_type T>
__global__ void qskinny_to_f16_kernel(half * __restrict__ output, const uint8_t * __restrict__ codes,
                                      int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    // values_per_meta / 2 half2 per row plus 4 half2 of padding
    __shared__ __align__(16) half2 tile_smem[32][codec::values_per_meta / 2 + 4];
    __shared__ __align__(16) typename codec::meta_t meta_smem[32];

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int tile   = blockIdx.y;
    const int kb     = blockIdx.x;
    const int groups = k >> 4;

    // one meta per lane, for the row this lane holds the codes of
    if (warp == 0) {
        meta_smem[lane] = codec::load_meta(codes + codec::codes_bytes(n, k) +
                                           (size_t) codec::meta_bytes * ((size_t) kb * n + tile * 32 + lane));
    }
    __syncthreads();

    // sub-block `warp` of this super-block: records 2*warp and 2*warp+1
    const int group = kb * (codec::values_per_meta / codec::values_per_record) + 2 * warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes);
    const qskinny_code records[2] = {
        codec::load_record(code_ptr + ((size_t) tile * groups + group) * 32 + lane),
        codec::load_record(code_ptr + ((size_t) tile * groups + group + 1) * 32 + lane),
    };
    half2 weights[codec::values_per_sub_block / 2];
    codec::decode(records, meta_smem[lane], warp, weights);

    half2 * smem_row = tile_smem[qpn8_col_from_lane(lane)];
#pragma unroll
    for (int i = 0; i < codec::values_per_sub_block / 2; ++i) {
        smem_row[warp * (codec::values_per_sub_block / 2) + i] = weights[i];
    }
    __syncthreads();

    // 4 uint4 (four half2 each) per thread
    half * out_tile = output + (size_t) tile * 32 * k + (size_t) kb * codec::values_per_meta;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * 256;
        const int row = idx >> 5;
        const int col = (idx & 31) << 2;
        const uint4 value = *reinterpret_cast<const uint4 *>(&tile_smem[row][col]);
        *reinterpret_cast<uint4 *>(out_tile + (size_t) row * k + col * 2) = value;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, n, k);
#endif
}

// ---- 1Cat fp8_qpn8_sm70.cu:168-175 (mma.m8n8k4 macro, unchanged) ----

#define Q4K_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// ---- 1Cat fp8_qpn8_sm70.cu:207-432 (main kernel, adapted to the codec) ----

template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q4k_skinny_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    // one iteration per 32-value sub-block; SplitK divides the sub-block count, the meta is
    // reloaded whenever the 256-value super-block index changes
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    // MERGE-CODEC: meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec::meta_t meta = {};

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            // one meta per 256 values; shared by the 8 sub-blocks that follow
            meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const qskinny_code records[2] = { // MERGE-CODEC: record load per codec type
            codec::load_record(code_ptr + (size_t) (group + 0) * 32),
            codec::load_record(code_ptr + (size_t) (group + 1) * 32),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    // fixed reduction order over the split-K warps keeps the result reproducible
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

// ---- multiple projections of one input in one launch ----

// One segment is a repacked weight (N is a multiple of 32) plus its output tile range.
// first_tile is the running sum of n / 32 over the previous segments, so a CTA finds its
// segment from blockIdx.x with a short scan.
struct q4k_skinny_seg {
    const uint8_t * codes;
    int             n;
    float *         dst;
    int             first_tile;
};

// Passed by value. One CTA per 32-column tile, the segments cover the grid in order.
struct q4k_skinny_multi_params {
    q4k_skinny_seg seg[4];
    int n_seg;
    int n_tiles;
    const half * input;
    int k;
    int m;
};

// Same execution as q4k_skinny_kernel, but each CTA picks its segment from blockIdx.x and
// writes to that segment's dst with its own row length. launch_bounds keeps the register
// count at the single-kernel level so a row tile fits two CTAs per SM.
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1,
          bool PrefetchCodes = false>
__global__ void __launch_bounds__(32 * SplitK, RowTiles == 1 ? 2 : 1)
q4k_skinny_multi_kernel(const q4k_skinny_multi_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny multi supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny multi M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    int seg = 0;
    while (seg + 1 < p.n_seg && (int) blockIdx.x >= p.seg[seg + 1].first_tile) {
        ++seg;
    }
    const int n = p.seg[seg].n;
    const int k = p.k;
    const int m = p.m;
    const int tile = (int) blockIdx.x - p.seg[seg].first_tile;
    const uint8_t * codes = p.seg[seg].codes;
    const half * input = p.input;
    float * output = p.seg[seg].dst;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    // MERGE-CODEC: meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_meta_kb = -1;
        typename codec::meta_t meta = {};
        qskinny_code prefetched[2] = {};
        typename codec::meta_t prefetched_meta = {};
        int prefetched_meta_kb = -1;
        if constexpr (PrefetchCodes) {
            prefetched[0] = codec::load_record(code_ptr + (size_t) (sub_block_begin * 2 + 0) * 32);
            prefetched[1] = codec::load_record(code_ptr + (size_t) (sub_block_begin * 2 + 1) * 32);
        }

#pragma unroll 2
        for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
            const int kb = sub_block >> 3;
            if (kb != loaded_meta_kb) {
                meta = (PrefetchCodes && kb == prefetched_meta_kb) ? prefetched_meta
                    : codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
                loaded_meta_kb = kb;
            }

            const int group = sub_block << 1;
            qskinny_code records[2]; // MERGE-CODEC: record load per codec type
            if constexpr (PrefetchCodes) {
                records[0] = prefetched[0];
                records[1] = prefetched[1];
                // the next batch is loaded here so its latency hides under decode and mma
                if (sub_block + 1 < sub_block_begin + sub_blocks_per_warp) {
                    prefetched[0] = codec::load_record(code_ptr + (size_t) (group + 2) * 32);
                    prefetched[1] = codec::load_record(code_ptr + (size_t) (group + 3) * 32);
                    const int next_kb = (sub_block + 1) >> 3;
                    if (next_kb != loaded_meta_kb && next_kb != prefetched_meta_kb) {
                        prefetched_meta = codec::load_meta(meta_ptr + (size_t) next_kb * n * codec::meta_bytes);
                        prefetched_meta_kb = next_kb;
                    }
                }
            } else {
                records[0] = codec::load_record(code_ptr + (size_t) (group + 0) * 32);
                records[1] = codec::load_record(code_ptr + (size_t) (group + 1) * 32);
            }
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    // fixed reduction order over the split-K warps keeps the result reproducible
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED(p);
#endif
}

// ---- gated pair: the gate and up projections of one SwiGLU in one launch ----

// gate and up are two separate tensors: the block computes the two projections in two
// phases with the same split-K, accumulator chains and reduction order as two
// q4k_skinny_kernel launches, so each projection is bit-exact with its single-weight
// launch. Phase 0 keeps its reduced gate in registers; the phase 1 epilogue applies
// silu(gate) * up with the ggml_cuda_op_silu_single formula and writes float.
// launch_bounds keeps the register count at the single-kernel level so a row tile fits
// two CTAs per SM; the phase loop is not unrolled to keep the body shared.
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1,
          bool PrefetchCodes = false>
__global__ void __launch_bounds__(32 * SplitK, RowTiles == 1 ? 2 : 1)
q4k_skinny_gated_kernel(
    const uint8_t * __restrict__ gate_codes, const uint8_t * __restrict__ up_codes,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny gated pair supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny gated M=1 specialization uses one row tile");
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    __shared__ float partials[SplitK][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    // the reduced gate of phase 0 stays in registers: each thread keeps the values of its
    // own strided loop, so the shared footprint matches the single-weight kernel
    float gate_red[(kOutputElements + 32 * SplitK - 1) / (32 * SplitK)];

#pragma unroll 1
    for (int projection = 0; projection < 2; ++projection) {
        const uint8_t * codes = projection ? up_codes : gate_codes;
        const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                    (size_t) tile * groups_k16 * 32 + lane;
        // MERGE-CODEC: meta stride and offset are per codec type
        const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                                   (size_t) codec::meta_bytes * (tile * 32 + lane);

        float accum[RowTiles][NAcc][8];
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[row_tile][chain][i] = 0.0f;
                }
            }
        }
        int loaded_meta_kb = -1;
        typename codec::meta_t meta = {};
        qskinny_code prefetched[2] = {};
        typename codec::meta_t prefetched_meta = {};
        int prefetched_meta_kb = -1;
        if constexpr (PrefetchCodes) {
            prefetched[0] = codec::load_record(code_ptr + (size_t) (sub_block_begin * 2 + 0) * 32);
            prefetched[1] = codec::load_record(code_ptr + (size_t) (sub_block_begin * 2 + 1) * 32);
        }

#pragma unroll 2
        for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
            const int kb = sub_block >> 3;
            if (kb != loaded_meta_kb) {
                meta = (PrefetchCodes && kb == prefetched_meta_kb) ? prefetched_meta
                    : codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
                loaded_meta_kb = kb;
            }

            const int group = sub_block << 1;
            qskinny_code records[2]; // MERGE-CODEC: record load per codec type
            if constexpr (PrefetchCodes) {
                records[0] = prefetched[0];
                records[1] = prefetched[1];
                // the next batch is loaded here so its latency hides under decode and mma
                if (sub_block + 1 < sub_block_begin + sub_blocks_per_warp) {
                    prefetched[0] = codec::load_record(code_ptr + (size_t) (group + 2) * 32);
                    prefetched[1] = codec::load_record(code_ptr + (size_t) (group + 3) * 32);
                    const int next_kb = (sub_block + 1) >> 3;
                    if (next_kb != loaded_meta_kb && next_kb != prefetched_meta_kb) {
                        prefetched_meta = codec::load_meta(meta_ptr + (size_t) next_kb * n * codec::meta_bytes);
                        prefetched_meta_kb = next_kb;
                    }
                }
            } else {
                records[0] = codec::load_record(code_ptr + (size_t) (group + 0) * 32);
                records[1] = codec::load_record(code_ptr + (size_t) (group + 1) * 32);
            }
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta, sub_block & 7, weights);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                uint4 input01 = make_uint4(0, 0, 0, 0);
                uint4 input23 = make_uint4(0, 0, 0, 0);
                uint4 input45 = make_uint4(0, 0, 0, 0);
                uint4 input67 = make_uint4(0, 0, 0, 0);
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                    input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                    input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                    input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
                }

                const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
            }
        }

#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[row_tile][0][i] += accum[row_tile][chain][i];
                }
            }
        }

        if constexpr (M1Only) {
            if ((lane & 17) == 0) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                    for (int offset = 0; offset < 2; ++offset) {
                        const int i = pair * 4 + offset;
                        const int output_col =
                            offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                        partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                    }
                }
            }
        } else {
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int output_row =
                        row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                    const int output_col =
                        (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                    partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                        accum[row_tile][0][i];
                }
            }
        }
        __syncthreads();

        for (int element = threadIdx.x, slot = 0; element < kOutputElements;
             element += blockDim.x, ++slot) {
            float value = 0.0f;
#pragma unroll
            for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
                value += partials[k_warp][element];
            }
            if (projection == 0) {
                gate_red[slot] = value;
            } else {
                const float gate = gate_red[slot];
                const float silu = gate / (1.0f + expf(-gate));
                if constexpr (M1Only) {
                    output[tile * 32 + element] = silu * value;
                } else {
                    const int output_row = element >> 5;
                    const int output_col = element & 31;
                    if (output_row < m) {
                        output[(size_t) output_row * n + tile * 32 + output_col] = silu * value;
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, up_codes, input, output, n, k, m);
#endif
}

// Split-K 16 for the long K weights, 8 for the smallest ones (k = 256 has only 8 sub-blocks
// per tile). can_repack guarantees k % 256 == 0, so one of the two always divides the
// sub-block count. The long K weights use two accumulator chains, as in the Q8_0 path. No
// prefetch variant yet.
struct q4k_skinny_config {
    int  split_k;
    int  n_acc;
    bool prefetch;
};

static q4k_skinny_config q4k_skinny_config_for(const int64_t k) {
    // The prefetch variant of the multi/gated kernels is kept but not selected: on sm_70 it
    // measured a decode loss for every long-K shape of the Qwen3.8-27B weights (k =
    // 5120/6144/17408). Select a shape here only after re-measuring it.
    return { k % 512 == 0 ? 16 : 8, k >= 4096 ? 2 : 1, false };
}

template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_launch(const uint8_t * codes, const half * input, float * output,
                              int n, int k, int m, cudaStream_t stream) {
    q4k_skinny_kernel<T, SplitK, NAcc, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, input, output, n, k, m);
}

#define Q4K_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                               \
    do {                                                                                        \
        switch (config.split_k) {                                                               \
            case 16: q4k_skinny_launch<T, 16, NAcc, M1Only, RowTiles>(codes,                     \
                         input, output, n, k, m, stream); break;                                \
            default: q4k_skinny_launch<T, 8, NAcc, M1Only, RowTiles>(codes,                      \
                         input, output, n, k, m, stream); break;                                \
        }                                                                                       \
    } while (0)

// MERGE-CODEC: the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_mul_mat_launch_t(const uint8_t * codes, const half * input, float * output,
                                        int n, int k, int m, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_LAUNCH(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q4K_SKINNY_LAUNCH(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q4K_SKINNY_LAUNCH
// Same epilogue as q4k_skinny_gated_kernel, applied after both projections of an M > 16 run.
__global__ void q4k_skinny_swiglu_kernel(float * __restrict__ gate, const float * __restrict__ up,
                                         int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles, bool PrefetchCodes = false>
static void q4k_skinny_multi_launch(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    q4k_skinny_multi_kernel<T, SplitK, NAcc, M1Only, RowTiles, PrefetchCodes>
        <<<p.n_tiles, 32 * SplitK, 0, stream>>>(p);
}

#define Q4K_SKINNY_MULTI_LAUNCH(NAcc, M1Only, RowTiles)                                          \
    do {                                                                                         \
        switch (config.split_k) {                                                                \
            case 16: q4k_skinny_multi_launch<T, 16, NAcc, M1Only, RowTiles>(p, stream); break;    \
            default: q4k_skinny_multi_launch<T, 8, NAcc, M1Only, RowTiles>(p, stream); break;     \
        }                                                                                        \
    } while (0)

// MERGE-CODEC+PERF
#define Q4K_SKINNY_MULTI_LAUNCH_PF(NAcc, M1Only, RowTiles)                                       \
    do {                                                                                         \
        switch (config.split_k) {                                                                \
            case 16: q4k_skinny_multi_launch<T, 16, NAcc, M1Only, RowTiles, true>(p, stream); break; \
            default: q4k_skinny_multi_launch<T, 8, NAcc, M1Only, RowTiles, true>(p, stream); break;  \
        }                                                                                        \
    } while (0)

// MERGE-CODEC: the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_multi_mul_mat_launch_t(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(p.k);
    if (config.prefetch) {
        if (config.n_acc == 2) {
            if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH_PF(2, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH_PF(2, false, 1); }
            else { Q4K_SKINNY_MULTI_LAUNCH_PF(2, false, 2); }
        } else {
            if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH_PF(1, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH_PF(1, false, 1); }
            else { Q4K_SKINNY_MULTI_LAUNCH_PF(1, false, 2); }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (config.n_acc == 2) {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(2, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(2, false, 2); }
    } else {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(1, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(1, false, 2); }
    }
}

#undef Q4K_SKINNY_MULTI_LAUNCH
#undef Q4K_SKINNY_MULTI_LAUNCH_PF

static void q4k_skinny_multi_mul_mat_launch(ggml_type type, const q4k_skinny_multi_params & p,
                                            cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q2_K>(p, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q3_K>(p, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q4_K>(p, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q5_K>(p, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q6_K>(p, stream); break;
        default: GGML_ASSERT(!"q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// Same split-K/accumulator-chain table as the single-weight kernel, so a projection is
// bit-exact with its own q4k_skinny_kernel launch.
template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles, bool PrefetchCodes = false>
static void q4k_skinny_gated_launch(const uint8_t * gate_codes, const uint8_t * up_codes,
                                    const half * input, float * output, int n, int k, int m,
                                    cudaStream_t stream) {
    q4k_skinny_gated_kernel<T, SplitK, NAcc, M1Only, RowTiles, PrefetchCodes>
        <<<n / 32, 32 * SplitK, 0, stream>>>(gate_codes, up_codes, input, output, n, k, m);
}

#define Q4K_SKINNY_GATED_LAUNCH(NAcc, M1Only, RowTiles)                                        \
    do {                                                                                       \
        switch (config.split_k) {                                                              \
            case 16: q4k_skinny_gated_launch<T, 16, NAcc, M1Only, RowTiles>(                    \
                         gate_codes, up_codes, input, output, n, k, m, stream); break;         \
            default: q4k_skinny_gated_launch<T, 8, NAcc, M1Only, RowTiles>(                     \
                         gate_codes, up_codes, input, output, n, k, m, stream); break;         \
        }                                                                                      \
    } while (0)

// MERGE-CODEC+PERF
#define Q4K_SKINNY_GATED_LAUNCH_PF(NAcc, M1Only, RowTiles)                                     \
    do {                                                                                       \
        switch (config.split_k) {                                                              \
            case 16: q4k_skinny_gated_launch<T, 16, NAcc, M1Only, RowTiles,                    \
                         true>(gate_codes, up_codes, input, output, n, k, m, stream); break;  \
            default: q4k_skinny_gated_launch<T, 8, NAcc, M1Only, RowTiles,                     \
                         true>(gate_codes, up_codes, input, output, n, k, m, stream); break;  \
        }                                                                                      \
    } while (0)

// MERGE-CODEC: the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_gated_mul_mat_launch_t(const uint8_t * gate_codes, const uint8_t * up_codes,
                                              const half * input, float * output, int n, int k, int m,
                                              cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    if (config.prefetch) {
        if (config.n_acc == 2) {
            if (m == 1) { Q4K_SKINNY_GATED_LAUNCH_PF(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH_PF(2, false, 1); }
            else { Q4K_SKINNY_GATED_LAUNCH_PF(2, false, 2); }
        } else {
            if (m == 1) { Q4K_SKINNY_GATED_LAUNCH_PF(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH_PF(1, false, 1); }
            else { Q4K_SKINNY_GATED_LAUNCH_PF(1, false, 2); }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_GATED_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_GATED_LAUNCH(1, false, 2); }
    }
}

#undef Q4K_SKINNY_GATED_LAUNCH
#undef Q4K_SKINNY_GATED_LAUNCH_PF

static void q4k_skinny_gated_mul_mat_launch(ggml_type type, const uint8_t * gate_codes,
                                            const uint8_t * up_codes, const half * input,
                                            float * output, int n, int k, int m,
                                            cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q2_K>(gate_codes, up_codes, input, output, n, k, m, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q3_K>(gate_codes, up_codes, input, output, n, k, m, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q4_K>(gate_codes, up_codes, input, output, n, k, m, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q5_K>(gate_codes, up_codes, input, output, n, k, m, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q6_K>(gate_codes, up_codes, input, output, n, k, m, stream); break;
        default: GGML_ASSERT(!"q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- M=32 two-phase kernel (1Cat fp8_qpn8_sm70.cu:445-573, adapted to the codec) ----

// M=32 needs four 8-row accumulator tiles. Keeping all logical split-K warps resident would
// need 64 KiB of reduction storage for split-16, so half of the physical warps run the
// original logical warp ranges in two ordered phases. The compact first-half sum keeps the
// final reduction in the p0 + ... + p(SplitK-1) order at 36 KiB of shared memory.
template <ggml_type T, int SplitK>
__global__ void q4k_skinny_m32_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(SplitK == 12 || SplitK == 16,
                  "Q4_K skinny M=32 two-phase kernel supports split-12 or split-16");
    constexpr int kPhysicalWarps = SplitK / 2;
    constexpr int kRowTiles = 4;
    constexpr int kOutputElements = kRowTiles * 256;
    __shared__ float reduction_storage[kPhysicalWarps + 1][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const typename codec::record_t * code_ptr = reinterpret_cast<const typename codec::record_t *>(codes) +
                                                (size_t) tile * groups_k16 * 32 + lane;
    // MERGE-CODEC: meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        float accum[kRowTiles][2][8];
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
                for (int index = 0; index < 8; ++index) {
                    accum[row_tile][chain][index] = 0.0f;
                }
            }
        }

        const int logical_warp = warp + phase * kPhysicalWarps;
        const int sub_block_begin = logical_warp * sub_blocks_per_warp;
        int loaded_meta_kb = -1;
        typename codec::meta_t meta = {};
#pragma unroll 2
        for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
            const int kb = sub_block >> 3;
            if (kb != loaded_meta_kb) {
                meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
                loaded_meta_kb = kb;
            }

            const int group = sub_block << 1;
            const qskinny_code records[2] = { // MERGE-CODEC: record load per codec type
                codec::load_record(code_ptr + (size_t) (group + 0) * 32),
                codec::load_record(code_ptr + (size_t) (group + 1) * 32),
            };
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta, sub_block & 7, weights);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
                uint4 input01 = make_uint4(0, 0, 0, 0);
                uint4 input23 = make_uint4(0, 0, 0, 0);
                uint4 input45 = make_uint4(0, 0, 0, 0);
                uint4 input67 = make_uint4(0, 0, 0, 0);
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                    input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                    input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                    input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
                }

                const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a3[2], a3[3], b[14], b[15]);
            }
        }

#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int index = 0; index < 8; ++index) {
                accum[row_tile][0][index] += accum[row_tile][1][index];
                const int output_row =
                    row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
                reduction_storage[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][index];
            }
        }
        __syncthreads();

        for (int element = threadIdx.x; element < kOutputElements;
             element += blockDim.x) {
            float value = phase == 0 ? 0.0f : reduction_storage[kPhysicalWarps][element];
#pragma unroll
            for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
                value += reduction_storage[k_warp][element];
            }
            if (phase == 0) {
                reduction_storage[kPhysicalWarps][element] = value;
            } else {
                const int output_row = element >> 5;
                const int output_col = element & 31;
                if (output_row < m) {
                    output[(size_t) output_row * n + tile * 32 + output_col] = value;
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

#undef Q4K_SKINNY_MMA_8N8K4

// Only split-16 is enabled in the launcher below: the shapes that reach M > 16 all select
// split 16, and the kernel keeps the 1Cat split-12 variant unused.
template <ggml_type T>
static void q4k_skinny_m32_mul_mat_launch(const uint8_t * codes, const half * input,
                                          float * output, int n, int k, int m,
                                          cudaStream_t stream) {
    constexpr int kSplitK = 16;
    q4k_skinny_m32_kernel<T, kSplitK><<<n / 32, 32 * (kSplitK / 2), 0, stream>>>(
        codes, input, output, n, k, m);
    CUDA_CHECK(cudaGetLastError());
}

// Dispatches an already F16-converted input: M <= 16 keeps the single-stage kernel, M = 17..32
// uses the two-phase M=32 kernel and M = 33..64 runs it on two row blocks. Shapes whose
// split-K is not 16 have no M > 16 kernel and return false so the caller can fall back.
template <ggml_type T>
static bool q4k_skinny_mul_mat_dispatch_t(const uint8_t * codes, const half * input, float * output,
                                          int n, int k, int m, cudaStream_t stream) {
    GGML_ASSERT(m >= 1 && m <= 64);
    if (m <= 16) {
        q4k_skinny_mul_mat_launch_t<T>(codes, input, output, n, k, m, stream);
        return true;
    }
    if (q4k_skinny_config_for(k).split_k != 16) {
        return false;
    }
    if (m <= 32) {
        q4k_skinny_m32_mul_mat_launch<T>(codes, input, output, n, k, m, stream);
    } else {
        q4k_skinny_m32_mul_mat_launch<T>(codes, input, output, n, k, 32, stream);
        q4k_skinny_m32_mul_mat_launch<T>(codes, input + (size_t) 32 * k,
                                        output + (size_t) 32 * n, n, k, m - 32, stream);
    }
    return true;
}

// MERGE-CODEC: the dispatch is instantiated per codec type
static bool q4k_skinny_mul_mat_dispatch(ggml_type type, const uint8_t * codes, const half * input,
                                        float * output, int n, int k, int m, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q2_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q3_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q3_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q4_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q4_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q5_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q5_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q6_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q6_K>(codes, input, output, n, k, m, stream);
        default: GGML_ASSERT(!"q4k skinny: unknown type"); return false;
    }
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_q4k_skinny_is_repacked(src0));
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    return q4k_skinny_mul_mat_dispatch(src0->type, (const uint8_t *) src0->data, input.get(),
                                       (float *) dst->data, (int) n, (int) k, (int) m,
                                       ctx.stream());
}
bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                        const ggml_tensor * up_w, const ggml_tensor * src1,
                                        ggml_tensor * dst) {
    if (!ggml_cuda_q4k_skinny_is_repacked(gate_w) || !ggml_cuda_q4k_skinny_is_repacked(up_w)) {
        return false;
    }
    // MERGE-CODEC: the gated kernel runs one codec type for both projections
    if (gate_w->type != up_w->type) {
        return false;
    }
    const int64_t k = gate_w->ne[0];
    const int64_t n = gate_w->ne[1];
    const int64_t m = src1->ne[1];
    if (!ggml_are_same_shape(gate_w, up_w) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            m < 1 || m > 64) { // MERGE-PERF: M=32 kernel
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    if (m <= 16) {
        q4k_skinny_gated_mul_mat_launch(gate_w->type, (const uint8_t *) gate_w->data, // MERGE-CODEC
                                        (const uint8_t *) up_w->data, input.get(), (float *) dst->data,
                                        (int) n, (int) k, (int) m, ctx.stream());
        return true;
    }

    // MERGE-PERF: M = 17..64 runs both projections with the M=32 kernel, then the SwiGLU
    // epilogue, with the same silu formula as the fused kernel
    float * gate = (float *) dst->data;
    ggml_cuda_pool_alloc<float> up(ctx.pool(), n * m);
    q4k_skinny_mul_mat_dispatch(gate_w->type, (const uint8_t *) gate_w->data, input.get(), gate,
                                (int) n, (int) k, (int) m, ctx.stream());
    q4k_skinny_mul_mat_dispatch(up_w->type, (const uint8_t *) up_w->data, input.get(), up.get(),
                                (int) n, (int) k, (int) m, ctx.stream());

    const int count = (int) (n * m);
    q4k_skinny_swiglu_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(gate, up.get(), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                        int n_nodes, const ggml_tensor * src1) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) { // MERGE-PERF: M=32 kernel
        return false;
    }

    q4k_skinny_multi_params p = {};
    p.k = (int) k;
    p.m = (int) m;
    p.n_seg = n_nodes;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        ggml_tensor * dst = dsts[i];
        const int64_t n = w->ne[1];
        if (!ggml_cuda_q4k_skinny_is_repacked(w) ||
                // MERGE-CODEC: the multi kernel runs one codec type for all segments
                w->type != src0s[0]->type ||
                w->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
        q4k_skinny_seg & seg = p.seg[i];
        seg.codes = (const uint8_t *) w->data;
        seg.n = (int) n;
        seg.dst = (float *) dst->data;
    }
    for (int s = 0; s < p.n_seg; ++s) {
        p.seg[s].first_tile = p.n_tiles;
        p.n_tiles += p.seg[s].n / 32;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());
    p.input = input.get();

    if (m <= 16) {
        q4k_skinny_multi_mul_mat_launch(src0s[0]->type, p, ctx.stream()); // MERGE-CODEC
        return true;
    }

    // MERGE-PERF: M = 17..64 has no segment dispatch in the M=32 kernel, so each repacked
    // weight gets its own launch from the one converted input (the group is homogeneous)
    for (int s = 0; s < p.n_seg; ++s) {
        const q4k_skinny_seg & seg = p.seg[s];
        q4k_skinny_mul_mat_dispatch(src0s[s]->type, seg.codes, p.input, seg.dst, seg.n, p.k, (int) m,
                                    ctx.stream());
    }
    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t) {
    return t->extra == (const void *) &q4k_skinny_marker;
}

bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(t);
    return false;
#else
    // MERGE-CODEC: the codec family serves Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (see qskinny_codec) and all
    // five are repacked at runtime. Their compute moves from the MMVQ q8_1 path to F16, which
    // changes greedy tokens at ulp level; that is the intended cost of covering the mixed types.
    switch (t->type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }
    if (t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
        return false;
    }
    if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ggml_cuda_get_device())) {
        return false;
    }
    if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (!ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    // a whole number of 256-value super-blocks (k % 256 == 0) and whole 32-row tiles; k >= 256
    // keeps the split-K, the M=1 specialization and the to_f16 tile loop non-empty. Under
    // tensor split a device can hold an empty K slice, such a weight is never multiplied.
    return k >= 256 && k % 256 == 0 && n % 32 == 0 && n >= 32;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// MERGE-CODEC: the repack kernel is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_repack_tiles(const void * staging, uint8_t * codes, uint8_t * meta,
                                    int n, int k, int t0, int t1, cudaStream_t stream) {
    qskinny_repack_kernel<T><<<(unsigned) (t1 - t0), 256, 0, stream>>>(
        (const uint8_t *) staging, codes, meta, n, k, t0);
}

static void q4k_skinny_repack_tiles_dispatch(ggml_type type, const void * staging, uint8_t * codes,
                                             uint8_t * meta, int n, int k, int t0, int t1,
                                             cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_repack_tiles<GGML_TYPE_Q2_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_repack_tiles<GGML_TYPE_Q3_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_repack_tiles<GGML_TYPE_Q4_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_repack_tiles<GGML_TYPE_Q5_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_repack_tiles<GGML_TYPE_Q6_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        default: GGML_ASSERT(!"q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    const int64_t ntiles = n / 32;
    GGML_ASSERT(k >= 256 && ntiles >= 1);
    // one codec record per 16 K values and row, one meta per 256 K values and row;
    // 16*record_bytes + meta_bytes is the block size, so codes + meta fill the tensor
    const size_t record_bytes = t->type == GGML_TYPE_Q2_K ? 4  :
                                t->type == GGML_TYPE_Q3_K ? 6  :
                                t->type == GGML_TYPE_Q5_K ? 10 :
                                t->type == GGML_TYPE_Q6_K ? 12 : 8;
    const size_t meta_bytes = t->type == GGML_TYPE_Q2_K ? 20 :
                              t->type == GGML_TYPE_Q3_K ? 14 :
                              t->type == GGML_TYPE_Q6_K ? 18 : 16;
    const size_t tile_src_bytes = (size_t) k * (16 * record_bytes + meta_bytes) / 8;
    const size_t tile_dst_bytes = 2 * record_bytes * (size_t) k;
    const size_t meta_total = (size_t) n * k * meta_bytes / 256;
    const size_t codes_total = (size_t) n * k * record_bytes / 16;

    constexpr size_t staging_max = 32ull * 1024 * 1024;
    size_t tiles_per_chunk = staging_max / tile_src_bytes;
    if (tiles_per_chunk < 1) {
        tiles_per_chunk = 1;
    }
    if (tiles_per_chunk > (size_t) ntiles) {
        tiles_per_chunk = (size_t) ntiles;
    }

    cudaStream_t stream = ctx.stream();
    uint8_t * data = (uint8_t *) t->data;

    void * staging = nullptr;
    void * meta_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, tiles_per_chunk * tile_src_bytes));
    CUDA_CHECK(cudaMalloc(&meta_tmp, meta_total));

    // Chunk [t0, t1) writes codes to [t0*tile_dst, t1*tile_dst), which ends before the source
    // [t1*tile_src, ...) of the remaining tiles (tile_dst <= tile_src), and its own source was
    // copied to staging first. The meta region [codes_total, codes_total + meta_total) overlaps
    // the sources of the last tiles, so it is written only after all codes are done.
    for (int64_t t0 = 0; t0 < ntiles; t0 += (int64_t) tiles_per_chunk) {
        const int64_t t1 = std::min(t0 + (int64_t) tiles_per_chunk, ntiles);
        CUDA_CHECK(cudaMemcpyAsync(staging, data + t0 * tile_src_bytes, (t1 - t0) * tile_src_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        q4k_skinny_repack_tiles_dispatch(t->type, staging, data + t0 * tile_dst_bytes, (uint8_t *) meta_tmp,
                                         (int) n, (int) k, (int) t0, (int) t1, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(data + codes_total, meta_tmp, meta_total,
                               cudaMemcpyDeviceToDevice, stream));

    // no pool: the staging must not stay resident after the one-time repack
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(meta_tmp));

    t->extra = &q4k_skinny_marker;
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    // The idle counter is shared with the Q8_0 prepass, which controls the scan rate for both:
    // once it reaches 1024, neither prepass finds anything new, so the graph walk is skipped.
    // This prepass never modifies the counter so that the Q8_0 behaviour is unchanged.
    if (ctx.q8_skinny_idle_scans >= 1024) {
        return;
    }

    // candidates: Q4_K weights used as MUL_MAT src0; none in steady state
    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        ggml_tensor * src0 = node->src[0];
        if (!ggml_cuda_q4k_skinny_can_repack(src0) || ggml_cuda_q4k_skinny_is_repacked(src0)) {
            continue;
        }
        if (std::find(candidates.begin(), candidates.end(), src0) == candidates.end()) {
            candidates.push_back(src0);
        }
    }
    if (candidates.empty()) {
        return;
    }

    // repacking cannot be recorded into a CUDA graph; the capture is checked only here so that
    // the steady state costs no CUDA API call
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    // Repacked data can only be read by the new kernel, so drop candidates that are read
    // anywhere else (non-MUL_MAT src, MUL_MAT src1) or viewed by another tensor.
    for (int i = 0; i < cgraph->n_nodes && !candidates.empty(); ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->view_src != nullptr) {
            candidates.erase(std::remove(candidates.begin(), candidates.end(), node->view_src),
                             candidates.end());
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            if (src->view_src != nullptr) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src->view_src),
                                 candidates.end());
            }
            if (node->op != GGML_OP_MUL_MAT || j != 0) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src),
                                 candidates.end());
            }
        }
    }

    for (ggml_tensor * t : candidates) {
        ggml_cuda_q4k_skinny_repack_inplace(ctx, t);
    }
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int n = (int) src0->ne[1];
    const int k = (int) src0->ne[0];
    const uint8_t * data = (const uint8_t *) src0->data;
    const dim3 grid((unsigned) (k / 256), (unsigned) (n / 32), 1);
    // MERGE-CODEC: the dequantize kernel is instantiated per codec type
    switch (src0->type) {
        case GGML_TYPE_Q2_K: qskinny_to_f16_kernel<GGML_TYPE_Q2_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q3_K: qskinny_to_f16_kernel<GGML_TYPE_Q3_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q4_K: qskinny_to_f16_kernel<GGML_TYPE_Q4_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q5_K: qskinny_to_f16_kernel<GGML_TYPE_Q5_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q6_K: qskinny_to_f16_kernel<GGML_TYPE_Q6_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        default: GGML_ASSERT(!"q4k skinny to_f16: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
