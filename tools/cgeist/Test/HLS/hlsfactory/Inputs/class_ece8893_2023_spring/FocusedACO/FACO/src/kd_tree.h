/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/
#pragma once

#include <limits>
#include <utility>
#include <cstdint>
#include <vector>
#include <cassert>
#include <ostream>
#include <cmath>
#include <algorithm>

struct Vec2d {
    double x_;
    double y_;

    Vec2d& operator-=(const Vec2d &other) {
        x_ -= other.x_;
        y_ -= other.y_;
        return *this;
    }

    friend Vec2d operator-(const Vec2d &a, const Vec2d &b) {
        return { a.x_ - b.x_, a.y_ - b.y_ };
    }

    [[nodiscard]] double length() const { return std::sqrt(x_ * x_ + y_ * y_); }

    [[nodiscard]] double length_squared() const { return x_ * x_ + y_ * y_; }
};


/**
 * Simple implementation of the K-d tree as described in
 *
 * Bentley, Jon Louis. "K-d trees for semidynamic point sets." Proceedings of
 * the sixth annual symposium on Computational geometry. ACM, 1990.
 *
 * This implements only the most basic variant of the structure without
 * many optimizations mentioned in the paper.
 */
class KDTree {
public:

    using Point = Vec2d;

    enum : uint32_t { Sentinel = std::numeric_limits<uint32_t>::max() };

    struct Bounds {
        double x_min_ = std::numeric_limits<double>::min();
        double x_max_ = std::numeric_limits<double>::max();
        double y_min_ = std::numeric_limits<double>::min();
        double y_max_ = std::numeric_limits<double>::max();
    };

    struct Node {
        uint32_t left_ = Sentinel;
        uint32_t right_ = Sentinel;
        uint32_t parent_ = Sentinel;
        int bucket_start_ = -1;  // start index in buckets_points_
        int bucket_end_ = -1;    // end index in bucket_points_
        double cutval_ = 0;
        Bounds bounds_;
        int8_t cutdim_ = 0;
        bool is_empty_ = true;

        [[nodiscard]] bool is_bucket() const { return bucket_start_ != -1; }
    };

    std::vector<Node> nodes_;
    std::vector<Point> points_;
    std::vector<uint32_t> bucket_points_;  // A list of all points stored in
                                           // the "bucket" nodes
    uint32_t cutoff_ = 16;
    uint32_t root_ = Sentinel;
    std::vector<uint32_t> point_idx_to_node_;

    uint32_t nn_target_pt_idx_ = 0;
    int nn_pt_idx_ = 0;
    double nn_dist_ = 0;


    explicit KDTree(const std::vector<Point> &points) {
        points_ = points;
        const auto n = points_.size();
        bucket_points_.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            bucket_points_[i] = i;
        }
        point_idx_to_node_.clear();
        point_idx_to_node_.resize(n, Sentinel);

        nodes_.reserve(n / 2);

        Bounds inf;
        root_ = build(0, n - 1, inf);
    }

    uint32_t build(uint32_t l, uint32_t u, const Bounds &bounds) {
        uint32_t node_id = nodes_.size();

        nodes_.push_back(Node{});

        auto *p = &nodes_.back();
        p->is_empty_ = false;
        p->bounds_ = bounds;

        if (u - l + 1 <= cutoff_) {
            p->bucket_start_ = static_cast<int32_t>(l);
            p->bucket_end_ = static_cast<int32_t>(u);

            for (auto i = l; i <= u; ++i) {
                point_idx_to_node_[ bucket_points_[i] ] = node_id;
            }
        } else {
            p->cutdim_ = find_max_spread_dimension(l, u);
            const auto m = (l + u) / 2;
            select(l, u, m, p->cutdim_);
            p->cutval_ = get_coordinate(points_[bucket_points_[m]],
                                        p->cutdim_);

            Bounds bounds_left { bounds };
            if (p->cutdim_ == 0) {
                bounds_left.x_max_ = p->cutval_;
            } else {
                bounds_left.y_max_ = p->cutval_;
            }
            p->left_ = build(l, m, bounds_left);

            Bounds bounds_right { bounds };
            if (p->cutdim_ == 0) {
                bounds_right.x_min_ = p->cutval_;
            } else {
                bounds_right.y_min_ = p->cutval_;
            }
            p->right_ = build(m+1, u, bounds_right);

            nodes_.at(p->left_).parent_ = node_id;
            nodes_.at(p->right_).parent_ = node_id;
        }
        return node_id;
    }

    [[nodiscard]] size_t get_points_count() const {
        return points_.size();
    }

    int8_t find_max_spread_dimension(uint32_t low, uint32_t up) {
        auto pt = points_.at( bucket_points_.at(low) );

        auto min_x = pt.x_;
        auto max_x = min_x;

        auto min_y = pt.y_;
        auto max_y = min_y;

        for (uint32_t i = low + 1; i <= up; ++i) {
            pt = points_[ bucket_points_[i] ];

            min_x = std::min(min_x, pt.x_);
            max_x = std::max(max_x, pt.x_);

            min_y = std::min(min_y, pt.y_);
            max_y = std::max(max_y, pt.y_);
        }
        return (max_x - min_x) > (max_y - min_y) ? 0 : 1;
    }


    static double get_coordinate(const Point &p, int8_t dim) {
        assert(dim == 0 || dim == 1);
        return (dim == 0) ? p.x_ : p.y_;
    }


    void select(uint32_t lower, uint32_t upper, uint32_t middle, int8_t dim) {
        std::nth_element(
                bucket_points_.begin() + lower,
                bucket_points_.begin() + middle,
                bucket_points_.begin() + upper + 1,
                [this, dim](size_t a, size_t b) {
                    auto &pa = this->points_[a];
                    auto &pb = this->points_[b];
                    return get_coordinate(pa, dim) < get_coordinate(pb, dim);
                });
    }


    uint32_t nn(uint32_t point_idx) {
        assert(point_idx < points_.size());
        assert(root_ != Sentinel);

        nn_target_pt_idx_ = point_idx;

        nn_dist_ = std::numeric_limits<double>::max();
        rnn(root_);

        return nn_pt_idx_;
    }


    /**
     * Outputs tree in Graphviz format for an easy visualization of the tree.
     * Example:
     * 
     * If the function has written tree repr. to "tree.gv" file
     * 
     * then
     * 
     * dot -Tpdf tree.gv -o tree.pdf
     * 
     * will produce PDF file with the tree visualization.
     */
    void print_in_dot_format(uint32_t node_id, std::ostream &out) {
        if (node_id == root_) {
            out << "digraph G {\n";
        }

        auto &node = nodes_.at(node_id);
        if ( node.is_bucket() ) {
            out << "\t" << node_id << " [label=\"";
            for (auto i = node.bucket_start_; i <= node.bucket_end_; i++) {
                auto pt_id = bucket_points_.at(i);
                auto pt = points_.at(pt_id);
                out << "(" << pt.x_ << ", " << pt.y_ << ")\\n";
            }
            out << "\"];\n";
        } else {
            auto split_dim = node.cutdim_ == 0 ? 'x' : 'y';

            out << "\t" << node_id << " -> " << "" << node.left_;
            out << " [label=\""
                << split_dim << " <= " << node.cutval_
                << "\"];\n";
            out << "\t" << node_id << " -> " << "" << node.right_;
            out << " [label=\""
                << split_dim << " > " << node.cutval_
                << "\"];\n";
            print_in_dot_format(node.left_, out);
            print_in_dot_format(node.right_, out);
        }

        if (node_id == root_) {
            out << "}" << std::flush;
        }
    }


    void rnn(uint32_t node_id) {
        auto *p = &nodes_.at(node_id);

        if (p->is_bucket()) {
            for (auto i = p->bucket_start_; i <= p->bucket_end_; i++) {
                auto pt_id = bucket_points_[i];
                if (pt_id != nn_target_pt_idx_) {
                    auto dist = get_distance(pt_id, nn_target_pt_idx_);
                    if (dist < nn_dist_) {
                        nn_dist_ = dist;
                        nn_pt_idx_ = static_cast<int32_t>(pt_id);
                    }
                }
            }
        } else {
            auto val = p->cutval_;
            auto coord = get_coordinate(points_[nn_target_pt_idx_], p->cutdim_);
            if (coord < val) {
                rnn(p->left_);
                if (coord + nn_dist_ > val) {
                    rnn(p->right_);
                }
            } else {
                rnn(p->right_);
                if (coord - nn_dist_ < val) {
                    rnn(p->left_);
                }
            }
        }
    }


    uint32_t nn_bottom_up(uint32_t point_idx) {
        assert(point_idx < points_.size());
        assert(root_ != Sentinel);

        nn_target_pt_idx_ = point_idx;
        nn_dist_ = std::numeric_limits<double>::max();
        const auto node_id = point_idx_to_node_.at(point_idx);
        rnn(node_id);

        auto *p = &nodes_.at(node_id);
        while (true) {
            auto *lastp = p;
            if (p->parent_ == Sentinel) {
                break ;
            }
            p = &nodes_.at(p->parent_);
            auto coord = get_coordinate(points_[nn_target_pt_idx_], p->cutdim_);
            auto diff = coord - p->cutval_;
            if (nn_dist_ >= fabs(diff)) {
                if (lastp == &nodes_.at(p->left_)) {
                    rnn(p->right_);
                } else {
                    rnn(p->left_);
                }
            }
            // We use + 1 so that TSPLIB rounded distance calculation can be
            // handled properly
            if (ball_in_bounds(p->bounds_, nn_target_pt_idx_, nn_dist_ + 1)) {
                break ;
            }
        }

        return nn_pt_idx_;
    }


    [[nodiscard]] double get_distance(size_t first_pt_idx, size_t second_pt_idx) const {
        const auto &a = points_[first_pt_idx];
        const auto &b = points_[second_pt_idx];

        auto dx = a.x_ - b.x_;
        auto dy = a.y_ - b.y_;

        // return int(std::sqrt(dx*dx + dy*dy) + 0.5);
        return static_cast<double>(lround(std::sqrt(dx*dx + dy*dy)));
    }


    Node &get_node(uint32_t node_id) {
        assert(node_id < nodes_.size());
        return nodes_[node_id];
    }


    Node *get_parent(Node &node) {
        return (node.parent_ != Sentinel)
             ? &get_node(node.parent_)
             : nullptr;
    }


    void delete_point(uint32_t point_idx) {
        assert(point_idx < point_idx_to_node_.size());
        auto node_id = point_idx_to_node_.at(point_idx);

        assert(node_id < nodes_.size());
        auto *p = &nodes_.at(node_id);
        auto j = p->bucket_start_;
        while (bucket_points_[j] != point_idx) {
            ++j;
        }
        std::swap( bucket_points_[j], bucket_points_[p->bucket_end_] );
        --p->bucket_end_;
        if (p->bucket_start_ > p->bucket_end_) {
            p->is_empty_ = true;

            while ( (p = get_parent(*p)) != nullptr
                    && nodes_.at(p->left_).is_empty_
                    && nodes_.at(p->right_).is_empty_ ) {
                p->is_empty_ = true;
            }
        }
    }


    void undelete_point(uint32_t point_idx) {
        auto node_id = point_idx_to_node_.at(point_idx);
        auto *p = &nodes_.at(node_id);
        auto j = p->bucket_start_;
        while (bucket_points_[j] != point_idx) {
            ++j;
        }
        ++p->bucket_end_;
        std::swap( bucket_points_[j], bucket_points_[p->bucket_end_] );
        if (p->is_empty_) {
            p->is_empty_ = false;

            while ( (p = get_parent(*p)) != nullptr
                    && p->is_empty_ ) {
                p->is_empty_ = false;
            }
        }
    }


    [[nodiscard]] bool ball_in_bounds(const Bounds &bounds, uint32_t point_idx,
                        double radius) const {
        assert(radius >= 0);

        const auto &pt = points_.at(point_idx);
        auto x = pt.x_;
        auto y = pt.y_;

        return (bounds.x_min_ <= x - radius)
            && (bounds.x_max_ >= x + radius)
            && (bounds.y_min_ <= y - radius)
            && (bounds.y_max_ >= y + radius);
    }

    [[maybe_unused]] void fixed_radius_nn(uint32_t point_idx, double radius,
                         std::vector<uint32_t> &result) {
        nn_target_pt_idx_ = point_idx;
        nn_dist_ = radius;

        const auto node_id = point_idx_to_node_.at(point_idx);
        auto *p = &nodes_.at(node_id);
        auto p_id = node_id;

        result.clear();

        fixed_radius_nn_helper(p_id, result);

        while (true) {
            auto prev_p = p_id;
            if (p->parent_ == Sentinel) {
                break ;
            }
            p_id = p->parent_;
            p = &nodes_.at(p->parent_);
            auto coord = get_coordinate(points_[nn_target_pt_idx_], p->cutdim_);
            auto diff = coord - p->cutval_;
            if (prev_p == p->left_) {
                if (nn_dist_ >= -diff) {
                    fixed_radius_nn_helper(p->right_, result);
                }
            } else {
                if (nn_dist_ >= diff) {
                    fixed_radius_nn_helper(p->left_, result);
                }
            }
            if (ball_in_bounds(p->bounds_, nn_target_pt_idx_, nn_dist_ + 1)) {
                break ;
            }
        }
    }

    void fixed_radius_nn_helper(uint32_t node_id, std::vector<uint32_t> &result) {
        auto *p = &nodes_.at(node_id);
        if (p->is_empty_) {
            return ;
        }
        if (p->is_bucket()) {
            for (auto i = p->bucket_start_; i <= p->bucket_end_; i++) {
                auto dist = get_distance(bucket_points_[i], nn_target_pt_idx_);
                if (dist <= nn_dist_ && bucket_points_[i] != nn_target_pt_idx_) {
                    result.push_back(bucket_points_[i]);
                }
            }
        } else {
            auto coord = get_coordinate(points_[nn_target_pt_idx_], p->cutdim_);
            auto diff = coord - p->cutval_;
            if (diff < 0.0) {
                fixed_radius_nn_helper(p->left_, result);
                if (nn_dist_ >= -diff) {
                    fixed_radius_nn_helper(p->right_, result);
                }
            } else {
                fixed_radius_nn_helper(p->right_, result);
                if (nn_dist_ >= diff) {
                    fixed_radius_nn_helper(p->left_, result);
                }
            }
        }
    }
};
