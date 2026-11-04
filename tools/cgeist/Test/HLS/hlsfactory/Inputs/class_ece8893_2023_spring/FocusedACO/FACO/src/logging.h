/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/

#pragma once

#include <string>
#include <ostream>
#include <utility>
#include <vector>

#include "utils.h"  // round()

#define FMT_HEADER_ONLY
#include "include/fmt/format.h"
#include "include/fmt/os.h"

template<typename LogMap_t>
struct ComputationsLog {
    LogMap_t &log_;
    std::ostream &out_;

    ComputationsLog(LogMap_t &log_map, std::ostream &out)
        : log_(log_map), out_(out)
    {}

    template<typename T>
    void operator()(const std::string &key, const T &value, bool is_update=false) {
        out_ << key << ": "  << value << '\n';
        if (!is_update) {
            log_[key] = value;
        }
    }

    template<typename T>
    void operator()(const std::string &key, const std::vector<T> &vec) {
        log_[key] = vec;
    }
};


// Trace class is designed to record changes of a single variable values
// during an algorithm execution
template<typename ComputationsLog_t, typename T>
class Trace {
    std::vector<T> values_;
    std::vector<double> times_;
    std::vector<int32_t> iterations_;

    ComputationsLog_t &parent_;
    std::string key_;
    int32_t record_every_ith_iter_ = 1;

    // Computation log can receive periodic updates if enough time has passed
    // since the previous update
    bool   record_times_ = false;
    double prev_update_time_ = 0;
    double seconds_between_updates_;

public:

    Trace(ComputationsLog_t &parent,
          std::string key,
          size_t iterations,
          int32_t record_every_ith_iter = 1,
          bool trace_times=false,
          double seconds_between_updates = std::numeric_limits<double>::max())

        : parent_(parent),
          key_(std::move(key)),
          record_every_ith_iter_(record_every_ith_iter),
          record_times_(trace_times),
          seconds_between_updates_(seconds_between_updates)
    {
        auto capacity = iterations / std::max(1, record_every_ith_iter);
        values_.reserve(capacity);
        iterations_.reserve(capacity);
        if (record_times_) {
            times_.reserve(capacity);
        }
    }

    ~Trace() {
        parent_(key_ + " values", values_);

        int32_t gap = record_every_ith_iter_; 
        bool regular_gaps = true;

        if (iterations_.size() >= 2) {
            gap = iterations_[1] - iterations_[0];
            for (size_t i = 2 ; i < iterations_.size() ; ++i) {
                if (iterations_[i-1] + gap != iterations_[i]) {
                    regular_gaps = false;
                    break ;
                }
            }
        }
        if (regular_gaps) {
            parent_(key_ + " recorded every ith iter", gap);
        } else {
            parent_(key_ + " iterations", iterations_);
        }

        if (record_times_) {
            parent_(key_ + " times", times_);
        }
    }

    void add(const T &value, int32_t iteration, double time = 0) {
        if (iteration % record_every_ith_iter_ == 0) {
            values_.push_back(value); 
            iterations_.push_back(iteration);

            if (record_times_) {
                times_.push_back(round(time, 6));  // Microsecond precision
            }

            if (time - prev_update_time_ > seconds_between_updates_) {
                auto msg = fmt::format("[{}, {:.1f}s] {}", iteration, time, value);
                parent_(key_, msg, true);
                prev_update_time_ = time;
            }
        }
    }
};


struct SolutionCost {
    double cost_ = 0;
    // Optional error value, e.g. relative to the current best known result
    // in percents
    double error_ = -1;

    template<typename Json>
    friend void to_json(Json& j, const SolutionCost& p) {
        j = Json(p.cost_);
    }
};


template <>
struct fmt::formatter<SolutionCost> {
    static constexpr auto parse(format_parse_context& ctx) { return ctx.end(); }

    template <typename FormatContext>
    auto format(const SolutionCost& p, FormatContext& ctx) {
        return format_to(ctx.out(), "{:.0f} ({:.2f}%)", p.cost_, p.error_);
    }
};
