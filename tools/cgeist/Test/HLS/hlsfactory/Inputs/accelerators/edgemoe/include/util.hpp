#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include "hls_math.h"

#define _LABEL_FOR_EACH_HELPER(line, var) _ln ## line ## _for_each_ ## var
#define _LABEL_FOR_EACH(line, var) _LABEL_FOR_EACH_HELPER(line, var)
#define FOR_EACH(var, limit) _LABEL_FOR_EACH(__LINE__, var): for (unsigned int var = 0; var < limit; var++)

#define _LABEL_FOR_BLOCK_HELPER(line, var) _ln ## line ## _for_block_ ## var
#define _LABEL_FOR_BLOCK(line, var) _LABEL_FOR_BLOCK_HELPER(line, var)
#define FOR_BLOCK(var, limit, block_size) \
    constexpr unsigned int var##_step = (block_size); \
    constexpr unsigned int var##_limit = (limit); \
    constexpr unsigned int var##_iters = ((limit) + (block_size) - 1) / (block_size); \
    _LABEL_FOR_BLOCK(__LINE__, var): for ( \
        unsigned int var##_base = 0, var##_block = 0; \
        var##_base < var##_limit; \
        var##_base += var##_step, var##_block++ \
    )

#define _LABEL_FOR_OFFSET_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET(line, var) _LABEL_FOR_OFFSET_HELPER(line, var)
#define FOR_OFFSET(var) \
    _LABEL_FOR_OFFSET(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    ) \
    if (var##_limit % var##_step == 0 || var < var##_limit)

#define _LABEL_FOR_OFFSET_NOCHK_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET_NOCHK(line, var) _LABEL_FOR_OFFSET_NOCHK_HELPER(line, var)
#define FOR_OFFSET_NOCHK(var) \
    static_assert(var##_limit % var##_step == 0, "Cannot use FOR_OFFSET_NOCHK; use FOR_OFFSET instead."); \
    _LABEL_FOR_OFFSET_NOCHK(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    )

#define _LABEL_FOR_OFFSET_UNSAFE_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET_UNSAFE(line, var) _LABEL_FOR_OFFSET_UNSAFE_HELPER(line, var)
#define FOR_OFFSET_UNSAFE(var) \
    _LABEL_FOR_OFFSET_UNSAFE(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    )

template <typename T>
static constexpr T ceildiv(T dividend, T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
static constexpr T roundup(T dividend, T divisor)
{
    return ceildiv(dividend, divisor) * divisor;
}

template <typename T>
static constexpr T max(T a, T b)
{
    return (a > b) ? a : b;
}

template <typename T>
static constexpr T roundup_p2(T num)
{
    return (num == 0 || num == 1) ? 1 : 2 * roundup_p2((num + 1) / 2);
}

template <typename T>
static constexpr T bitcount(T num)
{
    return (num == 0 || num == 1) ? num : 1 + bitcount(num >> 1);
}

template <typename T>
static constexpr T ap_fixed_relu(T x)
{
    return hls::signbit(x) ? T(0) : x;
}

template <typename T>
static constexpr T ap_fixed_epsilon()
{
    return T(1.0 / (1 << (T::width - T::iwidth)));
}

template <typename T>
static constexpr T ap_fixed_min()
{
    return T(-(1 << (T::iwidth - 1)));
}

#endif
