#pragma once

#include <cstdint>
#include <immintrin.h>

namespace hnswlib {

class AlpSIMDHelper {
public:
    static inline void simd_add_and_scale(const int64_t* deltas, double min_val, double factor, double* output, size_t count) {
        size_t i = 0;

#ifdef __AVX2__
        if (count >= 4) {
            __m256d v_min = _mm256_set1_pd(min_val);
            __m256d v_factor = _mm256_set1_pd(factor);
            
            // 使用更简单的方法：分别加载高32位和低32位
            __m256i mask_low = _mm256_set1_epi64x(0xFFFFFFFF);
            
            for (; i <= count - 4; i += 4) {
                // 加载4个int64_t
                __m256i v_int64 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(deltas + i));
                
                // 方法1：使用标量转换（简单且正确）
                // 这是最简单的方法，虽然不是纯SIMD转换，但混合方法也能获得不错的效果
                // __m256d v_double = _mm256_set_pd(
                //     static_cast<double>(deltas[i+3]),
                //     static_cast<double>(deltas[i+2]),
                //     static_cast<double>(deltas[i+1]),
                //     static_cast<double>(deltas[i])
                // );
                // 方法2：使用分离高低32位的方法转换
                __m256d v_double = int64_to_double_avx2(v_int64);
                
                // 计算: min + delta * factor
                __m256d v_scaled = _mm256_mul_pd(v_double, v_factor);
                __m256d v_res = _mm256_add_pd(v_scaled, v_min);
                
                // 存储
                _mm256_storeu_pd(output + i, v_res);
            }
        }
#elif defined(__SSE2__)
        if (count >= 2) {
            __m128d v_min = _mm_set1_pd(min_val);
            __m128d v_factor = _mm_set1_pd(factor);
            
            for (; i <= count - 2; i += 2) {
                // 加载2个int64_t
                __m128i v_int64 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(deltas + i));
                
                // 使用标量转换
                // __m128d v_double = _mm_set_pd(
                //     static_cast<double>(deltas[i+1]),
                //     static_cast<double>(deltas[i])
                // );
                // 使用分离高低32位的方法转换
                __m128d v_double = int64_to_double_sse2(v_int64);
                
                __m128d v_scaled = _mm_mul_pd(v_double, v_factor);
                __m128d v_res = _mm_add_pd(v_scaled, v_min);
                
                _mm_storeu_pd(output + i, v_res);
            }
        }
#endif

        // 标量回退（处理剩余元素）
        for (; i < count; ++i) {
            output[i] = min_val + static_cast<double>(deltas[i]) * factor;
        }
    }

private:
#ifdef __AVX2__
    static inline __m256d int64_to_double_avx2(const __m256i v) {
        // 关键修正：使用 permute 将分散的32位整数 Pack 到一起，避免稀疏排列导致的0值和错位
        
        // 提取 High Parts (indices: 1, 3, 5, 7) 到低128位用于转换
        // [v0_l, v0_h, v1_l, v1_h, v2_l, v2_h, v3_l, v3_h] -> [v0_h, v1_h, v2_h, v3_h, ...]
        __m256i indices_hi = _mm256_setr_epi32(1, 3, 5, 7, 0, 0, 0, 0);
        __m256i v_hi_packed = _mm256_permutevar8x32_epi32(v, indices_hi);
        __m256d d_hi = _mm256_cvtepi32_pd(_mm256_castsi256_si128(v_hi_packed));
        
        // 提取 Low Parts (indices: 0, 2, 4, 6) 到低128位用于转换
        // [v0_l, v0_h, v1_l, v1_h, v2_l, v2_h, v3_l, v3_h] -> [v0_l, v1_l, v2_l, v3_l, ...]
        __m256i indices_lo = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
        __m256i v_lo_packed = _mm256_permutevar8x32_epi32(v, indices_lo);
        __m256d d_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(v_lo_packed));
        
        // 修正 Low Part 的符号问题：cvtepi32 将其视为有符号数，如果最高位为1会变成负数
        // 我们需要将其视为无符号数：if (d_lo < 0) d_lo += 2^32
        __m256d thresh = _mm256_set1_pd(4294967296.0); // 2^32
        __m256d mask = _mm256_cmp_pd(d_lo, _mm256_setzero_pd(), _CMP_LT_OQ);
        d_lo = _mm256_add_pd(d_lo, _mm256_and_pd(mask, thresh));
        
        // 组合：High * 2^32 + Low
        return _mm256_add_pd(_mm256_mul_pd(d_hi, thresh), d_lo);
    }
#endif

#if defined(__SSE2__)
    static inline __m128d int64_to_double_sse2(const __m128i v) {
        // v = [v0_l, v0_h, v1_l, v1_h]
        
        // 使用 Shuffle 将 Low Parts (0, 2) 放在前，High Parts (1, 3) 放在后
        // _MM_SHUFFLE(3, 1, 2, 0) -> 目标序列：[v0_l (0), v1_l (2), v0_h (1), v1_h (3)]
        __m128i packed = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 1, 2, 0));
        
        // 转换 Low Parts (前64位)
        __m128d d_lo = _mm_cvtepi32_pd(packed);
        
        // 右移8字节，将 High Parts 移到低位进行转换
        __m128i packed_hi = _mm_srli_si128(packed, 8);
        __m128d d_hi = _mm_cvtepi32_pd(packed_hi);
        
        // 修正 Low Part 的符号问题
        __m128d thresh = _mm_set1_pd(4294967296.0);
        __m128d mask = _mm_cmplt_pd(d_lo, _mm_setzero_pd());
        d_lo = _mm_add_pd(d_lo, _mm_and_pd(mask, thresh));
        
        // 组合
        return _mm_add_pd(_mm_mul_pd(d_hi, thresh), d_lo);
    }
#endif
};

} // namespace hnswlib