/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cassert>

#include "utils.h"
#include <iostream>


struct MatrixPheromone {
    uint32_t dimension_ = 0;
    std::vector<double> trails_; // For every edge (a,b),
                                 // where 0 <= a, b < dimension_
    bool is_symmetric_ = true;

    MatrixPheromone(uint32_t dimension, double initial_pheromone, bool is_symmetric)
        : dimension_(dimension),
          trails_(dimension * dimension, initial_pheromone),
          is_symmetric_(is_symmetric) {
    }

    [[nodiscard]] double get(uint32_t from, uint32_t to) const {
        assert((from < dimension_) && (to < dimension_));
        return trails_[from * dimension_ + to];
    }

    void evaporate(double evaporation_rate, double min_pheromone_value) {
        const auto n = trails_.size();

        // #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            trails_[i] = std::max(min_pheromone_value, trails_[i] * (1 - evaporation_rate));
        }
    }

    void increase(uint32_t from, uint32_t to, double deposit,
                  double max_pheromone_value) {

        assert((from < dimension_) && (to < dimension_));

        auto &value = trails_[from * dimension_ + to];
        value = std::min(max_pheromone_value, value + deposit);

        if (is_symmetric_) {
            trails_[to * dimension_ + from] = value;
        }
    }
};


// Pheromone values are stored only for the nodes which are on candidate lists
struct CandListPheromone {
    // We store cl_size_ trails for every node but serialized
    std::vector<uint32_t> nodes_;  // neighboring nodes
    std::vector<double>   trails_; // corresponding pheromone trails

    uint32_t dimension_ = 0;
    uint32_t cl_size_ = 0;
    bool is_symmetric_ = true;
    double default_pheromone_value_ = 0;

    template<typename NodeList_t>
    CandListPheromone(const std::vector<NodeList_t> &cand_lists,
                      double initial_pheromone,
                      bool is_symmetric=true)
        : dimension_(cand_lists.size()),
          cl_size_(cand_lists.at(0).size()),
          is_symmetric_(is_symmetric),
          default_pheromone_value_(initial_pheromone)
    {
        trails_.resize(dimension_ * cl_size_, initial_pheromone);
        for (auto &list : cand_lists) {
            assert(list.size() == cl_size_);
            for (auto &node : list) {
                nodes_.push_back(node);
            }
        }
    }

    [[nodiscard]] double get(uint32_t from, uint32_t to) const {
        assert((from < dimension_) && (to < dimension_));

        auto offset = from * cl_size_;
        for (uint32_t i = offset; i < offset + cl_size_; ++i) {
            if (nodes_[i] == to) {
                return trails_[i];
            }
        }
        return default_pheromone_value_;
    }

    void increase_helper(uint32_t from, uint32_t to,
                         double deposit,
                         double max_pheromone_value) {
        assert((from < dimension_) && (to < dimension_));

        auto offset = from * cl_size_;
        for (uint32_t i = offset; i < offset + cl_size_; ++i) {
            if (nodes_[i] == to) {
                trails_[i] = std::min(max_pheromone_value, deposit + trails_[i]);
                break ;
            }
        }
    }

    void increase(uint32_t from, uint32_t to,
                  double delta,
                  double max_pheromone_value) {

        increase_helper(from, to, delta, max_pheromone_value);
        if (is_symmetric_) {
            increase_helper(to, from, delta, max_pheromone_value);
        }
    }

    void evaporate(double evaporation_rate, double min_pheromone_value) {
        const auto n = trails_.size();

        // #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            trails_[i] = std::max(min_pheromone_value, trails_[i] * (1 - evaporation_rate));
        }

        // #pragma omp single
        default_pheromone_value_ = std::max(default_pheromone_value_ * (1 - evaporation_rate),
                                            min_pheromone_value);
    }

    void set_all_trails(double pheromone_value) {
        for (auto &t : trails_) {
            t = pheromone_value;
        }
    }

    void print_stats() {
        std::vector<double> ratios;
        for (uint32_t node = 0; node < dimension_; ++node) {
            auto low = *std::min_element(trails_.begin() + (node * cl_size_),
                              trails_.begin() + ((node + 1) * cl_size_));
            auto high = *std::max_element(trails_.begin() + (node * cl_size_),
                              trails_.begin() + ((node + 1) * cl_size_));
            if (low > 0) {
                ratios.push_back(high / low);
            }
        }
        std::cout << "Pher. ratio: " << sample_mean(ratios) 
                  << "\tMin ratio: " << *min_element(begin(ratios), end(ratios))
                  << "\tMax ratio: " << *max_element(begin(ratios), end(ratios))
                  << "\n";
    }
};
