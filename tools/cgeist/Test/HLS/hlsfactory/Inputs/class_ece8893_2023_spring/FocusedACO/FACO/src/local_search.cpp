/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/
#include <array>
#include <algorithm>
#include <numeric>

#include "local_search.h"
#include "utils.h"

/*
 * This performs a 2-opt move by flipping a section of the route.
 * The boundaries of the section are given by first and last.
 * 
 * It may happen that the section is very long compared to the remaining part
 * of the route. In such case, the remaining part is flipped, to speed things
 * up as the result of such flip results in equivalent solution.
 *
 * The positions of the nodes inside the route are also updated to match
 * the order after the flip.
 */
void flip_route_section(std::vector<uint32_t> &route,
                        std::vector<uint32_t> &pos_in_route,
                        int32_t first, int32_t last) {

    if (first > last) {
        std::swap(first, last);
    }

    const auto length = static_cast<int32_t>(route.size());
    const int32_t segment_length = last - first;
    const int32_t remaining_length = length - segment_length;

    if (segment_length <= remaining_length) {  // Reverse the specified segment
        std::reverse(route.begin() + first, route.begin() + last);

        for (auto k = first; k < last; ++k) {
            pos_in_route[ route[k] ] = k;
        }
    } else {  // Reverse the rest of the route, leave the segment intact
        first = (first > 0) ? first - 1 : length - 1;
        last = last % length;
        std::swap(first, last);
        int32_t l = first;
        int32_t r = last;
        int32_t i = 0;
        int32_t j = length - first + last + 1;
        while(i++ < j--) {
            std::swap(route[l], route[r]);
            pos_in_route[route[l]] = l;
            pos_in_route[route[r]] = r;
            l = (l+1) % length;
            r = (r > 0) ? r - 1 : length - 1;
        }
    }
}

int32_t random_fpga_function(int32_t first, int32_t length) {
#pragma HLS interface m_axi depth=1 port=first bundle=mem
#pragma HLS interface m_axi depth=1 port=length bundle=mem
#pragma HLS interface s_axilite register port=return

    first = (first > 0) ? first - 1 : length - 1;
    return first;
}

void flip_route_section(std::vector<uint32_t> &route,
                        int32_t first, int32_t last) {

    if (first > last) {
        std::swap(first, last);
    }

    const int32_t length = static_cast<int32_t>(route.size());
    const int32_t segment_length = last - first;
    const int32_t remaining_length = length - segment_length;

    if (segment_length <= remaining_length) {
        std::reverse(route.begin() + first, route.begin() + last);
    } else {
        first = random_fpga_function(first, length); // (first > 0) ? first - 1 : length - 1;
        last = last % length;
        std::swap(first, last);
        int32_t l = first;
        int32_t r = last;
        int32_t i = 0;
        int32_t j = length - first + last + 1;
        while(i++ < j--) {
            std::swap(route[l], route[r]);
            l = (l+1) % length;
            r = (r > 0) ? r - 1 : length - 1;
        }
    }
}


/**
 * This is an implementation of an approximate 2-opt heuristic which uses the
 * nearest neighbor lists to limit the search for an improving move.
 *
 * Returns a number of changes (moves) applied to the route.
 */
int64_t two_opt_nn(const ProblemInstance &instance,
                   std::vector<uint32_t> &route,
                   bool use_dont_look_bits,
                   uint32_t nn_count) {
// #pragma HLS disaggregate variable = instance
// #pragma HLS disaggregate variable = route
    // We assume symmetry so that the order of the nodes does not matter
    assert(instance.is_symmetric_);

    const auto route_size = route.size();

    Bitmask dont_look_bits(route_size);

    std::vector<uint32_t> pos_in_route(route_size);
    for (uint32_t i = 0; i < route_size; ++i) {
        pos_in_route[ route[i] ] = i;
    }

    // Setting maximum number of allowed route changes prevents very long-running times
    // for very hard to solve TSP instances.
    const int64_t MaxChanges = UINT64_C(10) * route_size;
    int64_t changes_count = 0;

    bool improvement_found;

    do {
        improvement_found = false;

        for (uint32_t i = 0; i < route_size; ++i) {
            auto a = route[i];

            if (use_dont_look_bits && dont_look_bits[a]) {
                continue ;
            }

            auto a_next = (i + 1 < route_size) ? route[i+1] : route[0];
            auto a_prev = (i > 0) ? route[i-1] : route[route_size-1];

            auto dist_a_to_next = instance.get_distance(a, a_next);
            auto dist_a_to_prev = instance.get_distance(a, a_prev);

            double max_diff = -1;
            uint32_t left = 0;
            uint32_t right = 0;

            uint32_t b_index = 0;
            for (auto b : instance.get_nearest_neighbors(a, nn_count)) {
                auto dist_ab = instance.get_distance(a, b);
                ++b_index;

                auto b_pos = pos_in_route[b];

                if (dist_a_to_next > dist_ab) {
                    auto b_next = (b_pos + 1 < route_size) ? route[b_pos + 1] : route[0];

                    auto diff = dist_a_to_next
                              + instance.get_distance(b, b_next)
                              - dist_ab
                              - instance.get_distance(a_next, b_next);

                    if (diff > max_diff) {
                        left = std::min(i, b_pos) + 1;
                        right = std::max(i, b_pos) + 1;
                        max_diff = diff;
                    }
                }
            }

            b_index = 0;
            for (auto b : instance.get_nearest_neighbors(a, nn_count)) {
                auto dist_ab = instance.get_distance(a, b);
                ++b_index;

                auto b_pos = pos_in_route[b];
                if (dist_a_to_prev > dist_ab) {
                    auto b_prev = (b_pos > 0) ? route[b_pos-1] : route[route_size-1];

                    auto diff = dist_a_to_prev
                              + instance.get_distance(b_prev, b)
                              - dist_ab
                              - instance.get_distance(a_prev, b_prev);

                    if (diff > max_diff) {
                        left = std::min(i, b_pos);
                        right = std::max(i, b_pos);
                        max_diff = diff;
                    }
                }
            }

            if (max_diff > 0) {
                flip_route_section(route, pos_in_route,
                                   static_cast<int32_t>(left), static_cast<int32_t>(right));

                dont_look_bits.clear_bit(route[left]);
                dont_look_bits.clear_bit(route[right-1]);

                const auto left_prev = (left > 0) ? left-1 : route_size-1;
                dont_look_bits.clear_bit(route[left_prev]);

                const auto right_next = (right < route_size) ? right : 0;
                dont_look_bits.clear_bit(route[right_next]);

                improvement_found = true;

                ++changes_count;

                break ;
            } else if (use_dont_look_bits) {
                dont_look_bits.set_bit(a);
            }
        }
    } while (improvement_found && changes_count < MaxChanges);

    assert(instance.is_route_valid(route));
    return changes_count;
}


/**
 * This impl. of the 2-opt heuristic uses a queue of nodes to check for an
 * improving move, i.e. checklist. This is useful to speed up computations
 * if the route was 2-optimal but a few new edges were introduced -- endpoints
 * of the new edges should be inserted into checklist.
 */
int64_t two_opt_nn(const ProblemInstance &instance,
                   std::vector<uint32_t> &route,
                   std::vector<uint32_t> &checklist,
                   uint32_t nn_list_size) {

    // We assume symmetry so that the order of the nodes does not matter
    assert(instance.is_symmetric_);

    const auto route_size = route.size();
    std::vector<uint32_t> pos_in_route(route_size);

    for (uint32_t i = 0; i < route_size; ++i) {
        pos_in_route[ route[i] ] = i;
    }
    const auto n = route.size();

    // Setting maximum number of allowed route changes prevents very long-running times
    // for very hard to solve TSP instances.
    const uint32_t MaxChanges = route_size;
    uint32_t changes_count = 0;

    size_t checklist_pos_pos = 0;
    while (checklist_pos_pos < checklist.size() && changes_count < MaxChanges) {
        auto a = checklist[checklist_pos_pos++];
        assert(a < instance.dimension_);
        auto i = pos_in_route[a];

        auto a_next = (i + 1 < n) ? route[i+1] : route[0];
        auto a_prev = (i > 0) ? route[i-1] : route[route_size-1];

        auto dist_a_to_next = instance.get_distance(a, a_next);
        auto dist_a_to_prev = instance.get_distance(a, a_prev);

        double max_diff = -1;
        uint32_t left = 0;
        uint32_t right = 0;

        const auto &nn_list = instance.get_nearest_neighbors(a, nn_list_size);

        for (auto b : nn_list) {
            auto dist_ab = instance.get_distance(a, b);
            if (dist_a_to_next > dist_ab) {
                // We rotate the section between a and b_next so that
                // two new (undirected) edges are created: { a, b } and { a_next, b_next }
                //
                // a -> a_next ... b -> b_next
                // a -> b ... a_next -> b_next
                //
                // or
                //
                // b -> b_next ... a -> a_next
                // b -> a ... b_next -> a_next
                auto b_pos = pos_in_route[b];
                auto b_next = (b_pos + 1 < route_size) ? route[b_pos + 1] : route[0];

                auto diff = dist_a_to_next
                          + instance.get_distance(b, b_next)
                          - dist_ab
                          - instance.get_distance(a_next, b_next);

                if (diff > max_diff) {
                    left  = std::min(i, b_pos) + 1;
                    right = std::max(i, b_pos) + 1;
                    max_diff = diff;
                }
            } else {
                break ;
            }
        }

        for (auto b : nn_list) {
            auto dist_ab = instance.get_distance(a, b);
            if (dist_a_to_prev > dist_ab) {
                // We rotate the section between a_prev and b so that
                // two new (undirected) edges are created: { a, b } and { a_prev, b_prev }
                //
                // a_prev -> a ... b_prev -> b
                // a_prev -> b_prev ... a -> b
                //
                // or
                //
                // b_prev -> b ... a_prev -> a
                // b_prev -> a_prev ... b -> a
                auto b_pos = pos_in_route[b];
                auto b_prev = (b_pos > 0) ? route[b_pos-1] : route[route_size-1];

                auto diff = dist_a_to_prev
                          + instance.get_distance(b_prev, b)
                          - dist_ab
                          - instance.get_distance(a_prev, b_prev);

                if (diff > max_diff) {
                    left  = std::min(i, b_pos);
                    right = std::max(i, b_pos);
                    max_diff = diff;
                }
            } else {
                break ;
            }
        }

        if (max_diff > 0) {
            flip_route_section(route, pos_in_route, static_cast<int32_t>(left), static_cast<int32_t>(right));

            // Add nodes at the beginning/end of the flipped segment
            // and the non-flipped part
            uint32_t endpoints[] = {
                route[left],
                route[right-1],
                route[(left > 0) ? left-1 : route_size-1],
                route[(right < route_size) ? right : 0]
            };

            for (auto x : endpoints) {
                if (std::find(checklist.begin() + static_cast<int32_t>(checklist_pos_pos),
                              checklist.end(), x) == checklist.end()) {
                    checklist.push_back(x);
                }
            }
            ++changes_count;
        }
    }
    assert(instance.is_route_valid(route));
    return changes_count;
}


/*
 * Segment corresponds to a fragment (segment) of a route (vector), i.e. a
 * sequence of consecutive indices of the vector.
 */
struct Segment {
    uint32_t first_{ 0 }; // Index of the first element belonging to a segment
    uint32_t last_{ 0 };  // Index of the last element of the segment
    uint32_t len_{ 0 };   // Length of a whole route (vector)
    uint32_t id_{ 0 };    // We can give each segment an id / index
    bool is_reversed_{ false }; // This is used to denote that the order of elements
                                // within segment should be reversed

    /* Returns length of a segment */
    [[nodiscard]] uint32_t size() const {
        if (is_reversed_) {
            return get_reversed().size();
        }
        if (first_ <= last_) {
            return last_ - first_ + 1;
        }
        return len_ - first_ + last_ + 1;
    }

    void reverse() {
        std::swap(first_, last_);
        is_reversed_ = !is_reversed_;
    }

    [[nodiscard]] Segment get_reversed() const {
        return Segment{ last_, first_, len_, id_, !is_reversed_ };
    }

    [[nodiscard]] int32_t first() const { return static_cast<int32_t>(first_); }
    [[nodiscard]] int32_t last() const { return static_cast<int32_t>(last_); }
    [[nodiscard]] int32_t isize() const { return static_cast<int32_t>(size()); }
};


struct RelativeIndex {
    int32_t offset_;
    int32_t length_;

    RelativeIndex(int32_t offset, int32_t length) :
        offset_(offset), length_(length)
    {}

    inline int32_t operator()(int32_t index) const {
        return index + offset_ < length_
             ? index + offset_
             : index + offset_ - length_;
    }
};


void perform_2_opt_move(std::vector<uint32_t> &route, int32_t i, int32_t j) {
    flip_route_section(route, i+1, j+1);
}


/**
 * Performs route modifications required by a 3-opt move, i.e. segments
 * reversals and swaps.
 *
 * The longest of the { s0, s1, s2 } segments is not modified, instead the
 * remaining segments are reversed if necessary.
 *
 * This works only for the symmetric version of the TSP.
 */
void perform_3_opt_move(std::vector<uint32_t> &route,
                        Segment s0, Segment s1, Segment s2) {
    // Sort segments so that the longest one is the first - it
    // will be kept without changes
    if (s0.size() < s1.size()) {
        std::swap(s0, s1);
    }
    if (s0.size() < s2.size()) {
        std::swap(s0, s2);
    }
    if (s1.size() < s2.size()) {
        std::swap(s1, s2);
    }
    // Segments should be sorted
    assert((s0.size() >= s1.size()) && (s1.size() >= s2.size()));

    bool swap_needed = false;  // Do we need to swap shorter segments?

    // We do not want to touch the longest (first) segment so
    // if it is reversed we reverse the other two instead and do a
    // swap
    if (s0.is_reversed_) {
        s1.reverse();
        s2.reverse();
        swap_needed = true;
    }
    // segment[0] is OK, so touch only segment[1] and [2]
    if (s1.is_reversed_) {
        s1.reverse();

        RelativeIndex idx(static_cast<int32_t>(s1.first_), static_cast<int32_t>(route.size()));
        for (int32_t l = 0, r = s1.isize() - 1; l < r; ++l, --r) {
            std::swap(route[idx(l)], route[idx(r)]);
        }
    }
    if (s2.is_reversed_) {
        s2.reverse();

        RelativeIndex idx(static_cast<int32_t>(s2.first_), static_cast<int32_t>(route.size()));
        for (int32_t l = 0, r = s2.isize() - 1; l < r; ++l, --r) {
            std::swap(route[idx(l)], route[idx(r)]);
        }
    }
    if (swap_needed) {
        auto beg = route.begin();

        // Now perform the swap of s1 and s2, we use std::rotate
        if (s1.id_ == 2 && s2.id_ == 1) { // 0 2 1, easy case
            rotate(beg + s2.first(),
                   beg + s1.first(),
                   beg + s1.last() + 1);
        } else if (s1.id_ == 1 && s2.id_ == 2) { // 0 1 2, easy case
            rotate(beg + s1.first(),
                   beg + s2.first(),
                   beg + s2.last() + 1);
        } else {
            auto left = 0;
            auto middle = 0;
            auto right = s1.isize() + s2.isize();

            if ( (s1.id_ == 0 && s2.id_ == 2) // 1 0 2
              || (s1.id_ == 1 && s2.id_ == 0) ) {  // 2 1 0
                left = s2.first();
                middle = static_cast<int>(s2.size());
            } else if ( (s1.id_ == 2 && s2.id_ == 0) // 1 2 0
                     || (s1.id_ == 0 && s2.id_ == 1) ) {  // 2 0 1
                left = s1.first();
                middle = static_cast<int>(s1.size());
            }

            int32_t First = left;
            auto Middle = static_cast<int32_t>((left + middle) % route.size());
            auto Last =static_cast<int32_t>((left + right) % route.size());

            auto N = static_cast<int32_t>(route.size());

            auto Next = Middle;
            while (First != Next) {
                std::swap (route[First], route[Next]);

                First = (First + 1) < N ? First + 1 : 0;
                Next = (Next + 1) < N ? Next + 1 : 0;

                if (Next == Last) {
                    Next = Middle;
                } else if (First == Middle) {
                    Middle = Next;
                }
            }
        }
    }
}


/*
 * Impl. of the 3-opt heuristic. Tries to change the order of nodes in
 * solution to shorten the travel distance.
 *
 * During the search for an improvement only edges connecting nn_count nearest
 * neighbors are taken into account to cut the overall search time.
 *
 * The solution's route is modified only if a better order was found.
 *
 * Function returns improvement over the previous route length (travel
 * distance).
 *
 * This implementation is based on the ideas proposed in:
 * Bentley, Jon Jouis. "Fast algorithms for geometric traveling
 * salesman problems." ORSA Journal on computing 4.4 (1992): 387-411.
*/
int64_t three_opt_nn(const ProblemInstance &instance,
                     std::vector<uint32_t> &sol,
                     bool use_dont_look_bits,
                     uint32_t nn_count) {
    using namespace std;

    assert( instance.is_symmetric_ );

    const auto len = static_cast<uint32_t>(sol.size());
    auto &route = sol;

    Bitmask dont_look_bits(len);
    vector<uint32_t> pos_in_route(len);

    int64_t two_opt_changes = 0;
    int64_t three_opt_changes = 0;

    bool found_improvement;

    do {
        found_improvement = false;

        for (auto i = 0u; i < len; ++i) {
            pos_in_route[ route[i] ] = i;
        }

        for (auto i = 0u; i < len && !found_improvement; ++i) {
            const auto at_i = route[i];

            if (use_dont_look_bits && dont_look_bits[at_i]) {
                continue ;  // Do not check, it probably won't find an
                            // improvement
            }
            const auto &i_nn_list = instance.get_nearest_neighbors(at_i, nn_count);

            for (auto i_nn_idx = 0u; i_nn_idx < nn_count && !found_improvement; ++i_nn_idx) {
                const auto at_j = i_nn_list[i_nn_idx];
                const auto j = pos_in_route[at_j];

                // Check for 2-opt move
                const auto i_1 = (i + 1) % len;
                const auto j_1 = (j + 1) % len;
                const auto at_i_1 = route[i_1];

                const auto dist_i_to_next = instance.get_distance(at_i, at_i_1);
                const auto dist_i_to_j = instance.get_distance(at_i, at_j);

                // This shortens time considerably although results in longer tours
                if (dist_i_to_next < dist_i_to_j) {
                    break ;
                }

                const auto at_j_1 = route[j_1];

                auto cost_before_2opt = dist_i_to_next
                                      + instance.get_distance(at_j, at_j_1);

                auto cost_after_2opt = dist_i_to_j
                                     + instance.get_distance(at_i_1, at_j_1);

                if (cost_after_2opt < cost_before_2opt) {
                    perform_2_opt_move(route, static_cast<int32_t>(i), static_cast<int32_t>(j));

                    found_improvement = true;

                    dont_look_bits.clear_bit(at_i);
                    dont_look_bits.clear_bit(at_i_1);

                    dont_look_bits.clear_bit(at_j);
                    dont_look_bits.clear_bit(at_j_1);

                    ++two_opt_changes;

                    continue ;
                }

                const auto &j_nn_list = instance.get_nearest_neighbors(at_j, nn_count);

                assert(at_i != at_j);  // These two should be different

                for (auto j_nn_idx = 0u; j_nn_idx < nn_count && !found_improvement ; ++j_nn_idx) {
                    const auto at_k = j_nn_list[j_nn_idx];
                    const auto k = pos_in_route[at_k];

                    if (k == len || k == i) {  // Unlikely but possible, we want at_i != at_j != at_k
                        continue ;
                    }

                    uint32_t x = i;
                    uint32_t y = j;
                    uint32_t z = k;
                    uint32_t at_x = at_i;
                    uint32_t at_y = at_j;
                    uint32_t at_z = at_k;

                    // Sort (x, y, z)
                    if (x > y) { swap(x, y); swap(at_x, at_y); }
                    if (x > z) { swap(x, z); swap(at_x, at_z); }
                    if (y > z) { swap(y, z); swap(at_y, at_z); }

                    const auto x_1 = (x + 1) % len;
                    const auto y_1 = (y + 1) % len;
                    const auto z_1 = (z + 1) % len;

                    const auto at_x_1 = route[x_1];
                    const auto at_y_1 = route[y_1];
                    const auto at_z_1 = route[z_1];

                    const auto curr = instance.get_distance(at_x, at_x_1)
                                    + instance.get_distance(at_y, at_y_1)
                                    + instance.get_distance(at_z, at_z_1);

                    // 4 sets of possible new edges to check
                    const array<pair<uint32_t, uint32_t>, 4 * 3> edges{{
                        { at_y, at_x   }, { at_z_1, at_y_1 }, {   at_z, at_x_1 },
                        { at_y, at_z_1 }, {   at_x, at_y_1 }, {   at_z, at_x_1 },
                        { at_y, at_z_1 }, {   at_x, at_z   }, { at_y_1, at_x_1 },
                        { at_y, at_z   }, { at_y_1, at_x   }, { at_z_1, at_x_1 }
                    }};

                    // Which segments do we need to reverse in order to transform
                    // route so that the new edges are created properly
                    const array<bool, 4 * 3> segment_reversals{{
                        false, true, true,
                        true, true, true,
                        true, true, false,
                        true, false, true
                    }};

                    Segment seg[3] = {
                        { z_1, x, len, 0 },
                        { x_1, y, len, 1 },
                        { y_1, z, len, 2 }
                    };

                    for (auto l = 0u; l < 4 * 3 && !found_improvement; l += 3) {
                        auto e1 = edges[l + 0];
                        auto e2 = edges[l + 1];
                        auto e3 = edges[l + 2];

                        const auto cost = instance.get_distance(e1.first, e1.second)
                                        + instance.get_distance(e2.first, e2.second)
                                        + instance.get_distance(e3.first, e3.second);


                        if (cost < curr) {
                            found_improvement = true;

                            if (segment_reversals[l + 0]) {
                                seg[0].reverse();
                            }
                            if (segment_reversals[l + 1]) {
                                seg[1].reverse();
                            }
                            if (segment_reversals[l + 2]) {
                                seg[2].reverse();
                            }
                            dont_look_bits.clear_bit(e1.first);
                            dont_look_bits.clear_bit(e1.second);

                            dont_look_bits.clear_bit(e2.first);
                            dont_look_bits.clear_bit(e2.second);

                            dont_look_bits.clear_bit(e3.first);
                            dont_look_bits.clear_bit(e3.second);

                            ++three_opt_changes;
                        }
                    }
                    if (found_improvement) {
                        perform_3_opt_move(route, seg[0], seg[1], seg[2]);
                    }
                }
            }
            if (!found_improvement && use_dont_look_bits) {
                dont_look_bits.set_bit(at_i);
            }
        }
    } while(found_improvement);

    return two_opt_changes + three_opt_changes;
}
