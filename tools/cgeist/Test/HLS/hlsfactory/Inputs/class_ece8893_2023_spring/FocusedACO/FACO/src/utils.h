#pragma once

#include <chrono>
#include <random>
#include <ostream>
#include <cassert>
#include <string>

#include "rand.h"


// Utility for time measurements
struct Timer {
    using Clock = std::chrono::high_resolution_clock;

    Timer() : start_time_(Clock::now()) {}

    // Time since the constructor was called
    inline double get_elapsed_seconds() const noexcept {
        return get_elapsed_nanoseconds() * 1e-9;
    }

    inline double operator()() const noexcept {
        return get_elapsed_nanoseconds() * 1e-9;
    }

    inline int64_t get_elapsed_nanoseconds() const noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (Clock::now() - start_time_).count();
    }

    friend std::ostream &operator<<(std::ostream &out, const Timer &timer);

    Clock::time_point start_time_;
};


/**
 * Loads a map of the best-known or optimal solution values from the
 * JSON file at the given path.
 */
void load_best_known_solutions(const std::string &path);


/**
 * Returns the best-known or optimal solution value for the TSP
 * instance with the given name or default_value if the value is not
 * known.
 */
double get_best_known_value(const std::string &instance_name,
                            double default_value);


struct Bitmask {
    uint32_t size_ = 0;
    std::vector<uint32_t> mask_;

    Bitmask() {}

    Bitmask(uint32_t size)
        : size_(size),
          mask_(size / 32 + (size % 32 != 0 ? 1 : 0), 0)
    {}

    void resize(uint32_t size) {
        if (size != size_) {
            size_ = size;
            mask_.clear();
            uint32_t words_needed = size_ / 32 + (size_ % 32 != 0 ? 1 : 0);
            mask_.resize(words_needed, 0);
        }
    }

    void set_bit(uint32_t bit_pos) {
        assert(bit_pos < size_);
        mask_[ (bit_pos / 32) ] |= (1u << bit_pos % 32);
    }

    void clear_bit(uint32_t bit_pos) {
        assert(bit_pos < size_);
        mask_[ (bit_pos / 32) ] &= ~(1u << bit_pos % 32);
    }

    bool get_bit(uint32_t bit_pos) const {
        assert(bit_pos < size_);
        return (mask_[ (bit_pos / 32) ] & (1u << bit_pos % 32)) != 0;
    }

    bool operator[](uint32_t bit_pos) const {
        return get_bit(bit_pos);
    }

    void clear() {
        std::fill(mask_.begin(), mask_.end(), 0);
    }
};


inline double round(double value, int decimal_places) {
    const double s = std::pow(10.0, decimal_places);
    return std::round(value * s) / s;
}

double sample_mean(const std::vector<double> &vec);

double sample_stdev(const std::vector<double> &vec);


/*
 * Returns a string with the current date & time,
 * e.g. 2021-12-31 15:45:59
 */
std::string get_current_datetime_string(const char *date_sep = "-",
                                        const char *time_sep = ":",
                                        const char *between_sep = " ",
                                        bool include_ns=false);
