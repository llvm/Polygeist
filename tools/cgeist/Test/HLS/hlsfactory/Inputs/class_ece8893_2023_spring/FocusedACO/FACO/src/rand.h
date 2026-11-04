#ifndef RAND_H
#define RAND_H

#include <cstdint>
#include <limits>

/**
 * This is based on xoroshiro+ algorithm by David Blackman and Sebastiano Vigna
 * as described at: http://xoroshiro.di.unimi.it/
 */
struct Random {
    typedef uint64_t result_type;

    Random() = default;

    /**
    Initializes a PRNG's state using the given "seed" value.
    Uses the SplitMix64 generator as described in:
    http://xoshiro.di.unimi.it/splitmix64.c
    */
    void init(uint64_t seed) {
        auto splitmix64_next = [=](uint64_t x) {
            uint64_t z = (x + 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        };
        state_[0] = splitmix64_next(seed);
        state_[1] = splitmix64_next(state_[0]);
    }

    /**
     * @return A floating point number drawn randomly with uniform probability from range [0, 1)
     */
    double next_float() {
        auto x = next();
        return static_cast<double>(x >> 11) * (1. / (UINT64_C(1) << 53));
    }

    /**
     * See: Lemire, Daniel. "Fast random integer generation in an interval."
     * ACM Transactions on Modeling and Computer Simulation (TOMACS) 29.1 (2019): 1-12.
     *
     * @return an unsigned number in range [0, max_exclusive) drawn randomly with uniform
     * probability.
     */
    inline uint32_t next_uint32(uint32_t max_exclusive) {
        const auto range = max_exclusive;
        uint32_t x = next();
        uint64_t m = uint64_t(x) * uint64_t(range);
        auto l = uint32_t(m);
        if (l < range) {
            uint32_t t = -range;
            if (t >= range) {
                t -= range;
                if (t >= range)
                    t %= range;
            }
            while (l < t) {
                x = next();
                m = uint64_t(x) * uint64_t(range);
                l = uint32_t(m);
            }
        }
        return m >> 32;
    }

    /*
    This is the jump function for the generator. It is equivalent
    to 2^64 calls to next(); it can be used to generate 2^64
    non-overlapping subsequences for parallel computations.
    */
    void jump() {
        static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for (auto s : JUMP) {
            for (int b = 0; b < 64; b++) {
                if (s & UINT64_C(1) << b) {
                    s0 ^= state_[0];
                    s1 ^= state_[1];
                }
                next();
            }
        }
        state_[0] = s0;
        state_[1] = s1;
    }

    /* Same as call to next() */
    uint64_t operator()() noexcept { return next(); }

    [[nodiscard]] static uint64_t min() noexcept { return 0u; }

    [[nodiscard]] static uint64_t max() noexcept { return std::numeric_limits<uint64_t>::max(); }

private:

    /**
     * Advances the state of the generator.
     * @return uint64_t drawn randomly with uniform probability.
     */
    uint64_t next() noexcept {
        const uint64_t s0 = state_[0];
        uint64_t s1 = state_[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        state_[1] = rotl(s1, 36); // c

        return result;
    }

    static constexpr uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t state_[2];
};


/**
 * This is used to get a default (pseudo-)random number generator (RNG).
 * Every OpenMP thread has its own private instance that is
 * initialized by calling init_random_number_generators().
 * All threads generators share the same sequence of the random numbers,
 * however they are looking at different (non-overlapping) parts of it.
 * This assumes that each thread does not generates more than 2^64 
 * numbers.
 *
 * @return Instance of the RNG for the calling thread.
 */
Random &get_rng();

/**
 * This should be called before drawing any random numbers.
 */
void init_random_number_generators(uint64_t seed);

#endif
