#ifndef __LOAD_UTIL_HPP__
#define __LOAD_UTIL_HPP__

#include <istream>
#include <ostream>
#include <ap_fixed.h>
#include <hls_vector.h>
#include "linear.hpp"

template<int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
std::istream& read(std::istream& stream, ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& value)
{
    union {
        float value;
        char bytes[sizeof(float)];
    } readable;
    stream.read(readable.bytes, sizeof(float));
    value = readable.value;
    return stream;
}

template<typename T, size_t N>
std::istream& read(std::istream& stream, hls::vector<T, N>& vector)
{
    for (size_t i = 0; i < N; i++)
    {
        read(stream, vector[i]);
    }
    return stream;
}

template<typename T, size_t N>
std::istream& read(std::istream& stream, T(&array)[N])
{
    for (size_t i = 0; i < N; i++)
    {
        read(stream, array[i]);
    }
    return stream;
}

// template<typename T>
// void load_array(T& array, std::ifstream& file)
// {
//     if (is_fixedtype<T>())
//     {
//         union {
//             T value;
//             char bytes[sizeof(T)];
//         } data;

//         file.read(data.bytes, sizeof(T));
//         array = data.value;
//     }
//     else
//     {
//         union {
//             decltype(array[0]) value;
//             char bytes[sizeof(decltype(array[0]))];
//         } data;

//         for (unsigned int i = 0; i < array.size(); i++)
//         {
//             file.read(data.bytes, sizeof(decltype(array[0])));
//             array[i] = data.value;
//         }
//     }
// }

// template<typename T, size_t N, int dummy>
// void load_array(T(&array)[N], std::ifstream& file)
// {
//     for (unsigned int i = 0; i < N; i++)
//     {
//         load_array(array[i], file);
//     }
// }

// Template instantiations for debugging
template double fm_t::Base::to_double() const;
template double wt_linear_t::Base::to_double() const;
template double wt_attn_bias_t::Base::to_double() const;
template double wt_bias_t::Base::to_double() const;
template double wt_wbias_t::Base::to_double() const;
// template double wt_norm_t::Base::to_double() const;
template double wt_patch_embed_t::Base::to_double() const;
template double pixel_t::Base::to_double() const;

#endif
