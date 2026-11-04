/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
 */
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

#include "kd_tree.h"
#include "utils.h"


// Various types of edge weights used in TSPLIB
enum EdgeWeightType { EUC_2D, EXPLICIT, GEO, ATT, CEIL_2D };


inline int32_t euc2d_distance(const Vec2d &p1, const Vec2d &p2) {
    return static_cast<int32_t>((p2 - p1).length() + 0.5);
}


inline int32_t ceil_distance(const Vec2d &p1, const Vec2d &p2) {
    return static_cast<int32_t>(std::ceil((p2 - p1).length()));
}


/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
inline int32_t att_distance (const Vec2d &p1, const Vec2d &p2) {
    auto real = std::sqrt((p1 - p2).length_squared() / 10.0);
    auto trun = static_cast<int32_t>(real);
    return static_cast<int32_t>((trun < real) ? trun + 1 : trun);
}


/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
inline int32_t geo_distance (const Vec2d &p1, const Vec2d &p2) {
    double deg, min;
    double lati, latj, longi, longj;
    double q1, q2, q3;

    deg = static_cast<int32_t>(p1.x_);  // Truncate
    min = p1.x_ - deg;
    lati = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    deg = static_cast<int32_t>(p2.x_);
    min = p2.x_ - deg;
    latj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    deg = static_cast<int32_t>(p1.y_);
    min = p1.y_ - deg;
    longi = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    deg = static_cast<int32_t>(p2.y_);
    min = p2.y_ - deg;
    longj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    q1 = cos (longi - longj);
    q2 = cos (lati - latj);
    q3 = cos (lati + latj);
    return static_cast<int32_t>(6378.388 * acos (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
}


/*
 * This is used to implement nearest neighbor lists.
 *
 * Essentially, it wraps a raw pointer to nodes with length and a few utility
 * methods allowing to iterate over the nodes using range-based for loop.
 */
class NodeList {
public:
    class iterator {
        public:
            explicit iterator(const uint32_t *ptr): ptr(ptr){}
            iterator operator++() { ++ptr; return *this; }
            bool operator!=(const iterator & other) const { return ptr != other.ptr; }
            const uint32_t& operator*() const { return *ptr; }
        private:
            const uint32_t* ptr;
    };
private:
    const uint32_t *nodes_ = nullptr;
    uint32_t length_ = 0;
public:

    NodeList(const uint32_t *nodes, uint32_t length)
        : nodes_(nodes),
        length_(length) {}

    [[nodiscard]] uint32_t size() const { return length_; }

    [[nodiscard]] iterator begin() const { return iterator(nodes_); }

    [[nodiscard]] iterator end() const { return iterator(nodes_ + length_); }

    uint32_t operator[](uint32_t index) const {
        assert(index < length_);
        return nodes_[index];
    }
};


struct ProblemInstance {
    using Point = Vec2d;

    uint32_t dimension_;
    EdgeWeightType edge_weight_type_ = EUC_2D;
    std::vector<Point> coords_;  // Locations of the instance cities
    std::vector<double> distance_matrix_;
    // This stores a specified number of nearest neighbors for every node
    std::vector<uint32_t> all_nearest_neighbors_;
    uint32_t total_nn_per_node_ = 0;
    bool is_symmetric_ = true;
    std::string name_;  // Optional name of the instance
    double best_known_cost_ = -1;
    // k-d tree instance for efficient computation of the nearest neighbors
    mutable std::unique_ptr<KDTree> kdtree_ = nullptr;


    ProblemInstance(uint32_t dimension,
                    EdgeWeightType edge_weight_type,
                    std::vector<Point> coords,
                    const std::vector<double> &distance_matrix,
                    bool is_symmetric,
                    std::string name = "Unknown",
                    double best_known_cost = -1)
        : dimension_(dimension),
          coords_(std::move(coords)),
          distance_matrix_(distance_matrix),
          is_symmetric_(is_symmetric),
          name_(std::move(name)),
          best_known_cost_(best_known_cost) {

        assert(dimension >= 2);

        edge_weight_type_ = edge_weight_type;

        if (edge_weight_type == EUC_2D || edge_weight_type == CEIL_2D) {
            // We can use kd-tree to speed up nearest neighbor calculations
            kdtree_ = std::make_unique<KDTree>(coords_);
        }
        // For smaller instances we can pre-calculate distances for faster
        // computations
        if (distance_matrix.empty() && dimension_ < 2000) {
            std::vector<double> mat(dimension_ * dimension_);
            for (uint32_t i = 0; i < dimension_; ++i) {
                for (uint32_t j = 0; j < dimension_; ++j) {
                    mat[i * dimension_ + j] = get_distance(i, j);
                }
            }
            distance_matrix_ = mat;
        }
    }

    void compute_nn_lists(uint32_t nn_count) {
        total_nn_per_node_ = std::min(nn_count, dimension_ - 1);
        all_nearest_neighbors_.reserve(dimension_ * total_nn_per_node_);

        if (kdtree_ != nullptr) {
            auto &kdtree = *kdtree_;
            uint32_t n = kdtree.get_points_count();

            for (uint32_t node = 0; node < n; ++node) {
                for (uint32_t j = 0; j < total_nn_per_node_; ++j) {
                    auto pt_idx = kdtree.nn_bottom_up(node);
                    all_nearest_neighbors_.push_back(pt_idx);
                    kdtree.delete_point(pt_idx);
                }

                // Revert changes so that kdtree can be reused for other
                // lists
                for (auto pt_idx : get_nearest_neighbors(node, total_nn_per_node_)) {
                    kdtree.undelete_point(pt_idx);
                }
            }
        } else {
            for (uint32_t node = 0; node < dimension_; ++node) {
                std::vector<uint32_t> neighbors;
                neighbors.reserve(dimension_);
                for (uint32_t i = 0; i < dimension_; ++i) {
                    if (i != node) {
                        neighbors.push_back(i);
                    }
                }
                // This puts the closest cand_list_size + 1 nodes in front of
                // the array (and sorted)
                partial_sort(neighbors.begin(),
                            neighbors.begin() + std::min(total_nn_per_node_, dimension_),
                            neighbors.end(),
                            [this, node](uint32_t a, uint32_t b) {
                                return this->get_distance(node, a) < this->get_distance(node, b);
                            });

                for (uint32_t i = 0; i < total_nn_per_node_; ++i) {
                    all_nearest_neighbors_.push_back(neighbors.at(i));
                }
            }
        }
    }

    NodeList get_nearest_neighbors(uint32_t node, uint32_t nn_length) const {
        assert(nn_length <= total_nn_per_node_);
        return NodeList{ &all_nearest_neighbors_[node * total_nn_per_node_], nn_length };
    }

    std::vector<NodeList> get_nn_lists(uint32_t nn_length) const {
        assert(nn_length < total_nn_per_node_);
        std::vector<NodeList> lists;
        lists.reserve(dimension_);
        for (uint32_t node = 0; node < dimension_; ++node) {
            lists.emplace_back(get_nearest_neighbors(node, nn_length));
        }
        return lists;
    }

    // Backup neighbors follow the nearest neighbors
    NodeList get_backup_neighbors(uint32_t node, uint32_t nn_size, uint32_t backup_nn_size) const {
        assert(nn_size + backup_nn_size <= total_nn_per_node_);
        return NodeList{ &all_nearest_neighbors_[node * total_nn_per_node_] + nn_size, backup_nn_size };
    }

    double get_distance(uint32_t from, uint32_t to) const {
        assert((from < dimension_) && (to < dimension_));

        if (!distance_matrix_.empty()) {
            return distance_matrix_[from * dimension_ + to];
        }

        auto a = coords_[from];
        auto b = coords_[to];
        if (edge_weight_type_ == EUC_2D) {
            return euc2d_distance(a, b);
        }
        if (edge_weight_type_ == CEIL_2D) {
            return ceil_distance(a, b);
        }
        if (edge_weight_type_ == GEO) {
            return geo_distance(a, b);
        }
        if (edge_weight_type_ == ATT) {
            return att_distance(a, b);
        }
        // else edge_weight_type_ == EXPLICIT
        assert(!distance_matrix_.empty());
        return 0;
    }

    double calculate_route_length(const std::vector<uint32_t> &route) const {
        double distance = 0;
        if (!route.empty()) {
            auto prev_node = route.back();
            for (auto node : route) {
                distance += get_distance(prev_node, node);
                prev_node = node;
            }
        }
        return distance;
    }

    bool is_route_valid(const std::vector<uint32_t> &route) const {
        if (route.size() != dimension_) {
            return false;
        }
        Bitmask visited(dimension_);
        for (auto node : route) {
            if (node >= dimension_ || visited[node]) {
                return false;
            }
            visited.set_bit(node);
        }
        return true;
    }

    std::vector<uint32_t> build_nn_tour(uint32_t start_node) const {
        std::vector<uint32_t> tour(dimension_);
        tour.clear();
        tour.push_back(start_node);

        if (kdtree_ != nullptr) {
            auto &kdtree = *kdtree_;
            kdtree.delete_point(start_node);

            for (uint32_t i = 1; i < dimension_; ++i) {
                auto pt_idx = kdtree.nn(tour.back());
                tour.push_back(pt_idx);
                kdtree.delete_point(pt_idx);
            }

            for (auto it = tour.rbegin(); it != tour.rend(); ++it) {
                kdtree.undelete_point(*it);
            }
        } else {  // Use NN lists
            Bitmask visited(dimension_);
            visited.set_bit(start_node);
            for (uint32_t i = 1; i < dimension_; ++i) {
                auto prev = tour.back();
                auto next = prev;
                for (auto node : get_nearest_neighbors(prev, total_nn_per_node_)) {
                    if (!visited[node]) {
                        next = node;
                        break ;
                    }
                }
                if (next == prev) {
                    double min_cost = std::numeric_limits<double>::max();
                    for (uint32_t node = 0; node < dimension_; ++node) {
                        if (!visited[node] && get_distance(prev, node) < min_cost) {
                            min_cost = get_distance(prev, node);
                            next = node;
                        }
                    }
                }
                assert(next != prev);
                visited.set_bit(next);
                tour.push_back(next);
            }
        }
        return tour;
    }

    // Based on the given cost, calculates error relative to the best known
    // result in percents [%]
    double calc_relative_error(double cost) const {
        if (best_known_cost_ > 0) {
            return 100 * (cost - best_known_cost_) / best_known_cost_;
        }
        return -1;
    }
};


/**
 * Tries to load a Traveling Salesman Problem (or ATSP) instance in TSPLIB
 * format from file at 'path'. Only the instances with 'EDGE_WEIGHT_TYPE:
 * EUC_2D' or 'EXPLICIT' are supported.
 *
 * Throws runtime_error if the file is in unsupported format or if an error was
 * encountered.
 *
 * Returns the loaded problem instance.
 */
ProblemInstance load_tsplib_instance(const char *path);


void route_to_svg(const ProblemInstance &instance,
                  const std::vector<uint32_t> &route,
                  const std::string &path);
