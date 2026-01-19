#pragma once

#include <cstdint>
#include <immintrin.h>

namespace hnswlib {

class AlpSIMDHelper {
public:
    // Unpack varying bit-width integers (1-16 bits) to 64-bit integers
    // Using simple switch-case for now, can be optimized with AVX2 expand/shuffle later if needed for unpacking
    // Since we decode to double, we want int64 eventually.
    // However, fastlanes/unffor usually does bit-unpacking.
    // For simplicity and portability, let's assume we unpack to a temporary buffer first or use a scalar loop for unpacking 
    // if bit-unpacking AVX is too complex to implement from scratch right now.
    // 
    // Wait, the goal is high performance. Scalar unpacking might bottleneck.
    // But let's look at alp_simd_helper logic.
    // 
    // Helper to add base, scale by factor and store.
    
    // min_val + delta * factor ?? Check original logic.
    // From file read:
    // double min_val = reader.readDouble();
    // double factor = reader.readDouble();
    // ...
    // double val = min_val + (double)delta * factor;
    
    static inline void simd_add_and_scale(const int64_t* deltas, double min_val, double factor, double* output, size_t count) {
        size_t i = 0;
#ifdef __AVX2__
        __m256d v_min = _mm256_set1_pd(min_val);
        __m256d v_factor = _mm256_set1_pd(factor);
        
        // Process 4 doubles at a time
        for (; i + 4 <= count; i += 4) {
            // Load 4 int64s
            __m256i v_deltas_i64 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(deltas + i));
            
            // Convert int64 to double. _mm256_cvtepi64_pd is AVX512DQ? No, VS2019 might not support easily.
            // _mm256_cvtepi64_pd does not exist in standard AVX2, only AVX512.
            // AVX2 only has _mm256_cvtepi32_pd (int32 -> double).
            // So we need a fast int64->double conversion or assume deltas fit in int32?
            // FFOR deltas usually fit in smaller types, but here we use int64 buffer.
            
            // If we are sure deltas are small (e.g. < 32 bit), we can pack them to 32bit then convert.
            // But let's use the magic number trick or extract halves.
            
            // Let's use the scalar fallback loop for conversion if AVX2 instructions are missing,
            // OR use the SSE instruction _mm_cvtsi64_sd for each? No.
            
            // Let's look at `decoder.hpp` attachment. It has `int64_to_double_fast_precise` using AVX2.
            // I will copy that function.
            
            __m256d v_deltas_fp = int64_to_double_fast_precise(v_deltas_i64);
            
            // result = min + delta * factor
            // FMA: result = min + (delta * factor) -> _mm256_fmadd_pd(v_deltas_fp, v_factor, v_min) (Available in FMA3)
            __m256d v_res = _mm256_fmadd_pd(v_deltas_fp, v_factor, v_min);
            
            _mm256_storeu_pd(output + i, v_res);
        }
#endif
        // Scalar fallback
        for (; i < count; ++i) {
            output[i] = min_val + static_cast<double>(deltas[i]) * factor;
        }
    }

private:
#ifdef __AVX2__
    static inline __m256d int64_to_double_fast_precise(const __m256i v) {
        // Copied from alp/decoder.hpp
        
        __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000); /* 2^52 */
        __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); /* 2^84 + 2^63 */
        __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000080100000); /* 2^84 + 2^63 + 2^52 */
        __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

        __m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
        __m256i v_hi = _mm256_srli_epi64(v, 32);
        v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);
        __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
        __m256d result   = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
        return result;
    }
#endif
};

} // namespace hnswlib
