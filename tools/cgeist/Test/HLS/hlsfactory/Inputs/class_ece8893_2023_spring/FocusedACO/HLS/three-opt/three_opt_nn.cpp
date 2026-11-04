// #include <vector>
// #include <stdint.h>
// #include <cmath>
// #include <cassert>
// #include <algorithm>
// #include <array>

#include  "hls_math.h"

#define NODES 2392
#define NN_LIST_SIZE 20
#define CHECKLIST_CAPACITY 500

template <typename T>
void swap(T& a, T& b)
{
    #pragma HLS INLINE
    T temp = a;
    a = b;
    b = temp;
}

int32_t get_distance(uint32_t a, uint32_t b, int32_t coords[NODES][2])
{
    #pragma HLS INLINE
    int32_t p1[2] = {coords[a][0], coords[a][1]};
    int32_t p2[2] = {coords[b][0], coords[b][1]};
	float val = sqrt(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])));
    return static_cast<int32_t>(val + 0.5);
}

// int32_t get_distance(uint32_t a, uint32_t b, int32_t coords[NODES][2])
// {
//     #pragma HLS INLINE
//     int32_t p1[2] = {coords[a][0], coords[a][1]};
//     int32_t p2[2] = {coords[b][0], coords[b][1]};
// 	float val = sqrt(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])));
//     return static_cast<int32_t>(val + 0.5);
// }

void flip_route_section(uint32_t route[NODES],
                        uint32_t pos_in_route[NODES],
                        int32_t first, int32_t last) {

    if (first > last) {
        swap(first, last);
    }

    const int32_t length = static_cast<int32_t>(NODES);
    const int32_t segment_length = last - first;
    const int32_t remaining_length = length - segment_length;

    if (segment_length <= remaining_length) {  // Reverse the specified segment

        int32_t i = first;
        int32_t j = last - 1;
        // while (i < j) {
        for (i = first; i < j; i++) {
            swap(route[i], route[j]);
            // i++;
            j--;
        }
        for (int32_t k = first; k < last; ++k) {
            pos_in_route[ route[k] ] = k;
        }
    } else {  // Reverse the rest of the route, leave the segment intact
        first = (first > 0) ? first - 1 : length - 1;
        last = last % length;
        swap(first, last);
        int32_t l = first;
        int32_t r = last;
        int32_t i = 0;
        int32_t j = length - first + last + 1;
        while(i++ < j--) {
            swap(route[l], route[r]);
            pos_in_route[route[l]] = l;
            pos_in_route[route[r]] = r;
            l = (l+1) % length;
            r = (r > 0) ? r - 1 : length - 1;
        }
    }
}

// void flip_route_section(uint32_t route[NODES],
//                         uint32_t pos_in_route[NODES],
//                         int32_t first, int32_t last) {

//     if (first > last) {
//         std::swap(first, last);
//     }

//     const int32_t length = static_cast<int32_t>(NODES);
//     const int32_t segment_length = last - first;
//     const int32_t remaining_length = length - segment_length;

//     if (segment_length <= remaining_length) {  // Reverse the specified segment

//         int32_t i = first;
//         int32_t j = last - 1;
//         // while (i < j) {
//         for (i = first; i < j; i++) {
//             std::swap(route[i], route[j]);
//             // i++;
//             j--;
//         }
//         for (int32_t k = first; k < last; ++k) {
//             pos_in_route[ route[k] ] = k;
//         }
//     } else {  // Reverse the rest of the route, leave the segment intact
//         first = (first > 0) ? first - 1 : length - 1;
//         last = last % length;
//         std::swap(first, last);
//         int32_t l = first;
//         int32_t r = last;
//         int32_t i = 0;
//         int32_t j = length - first + last + 1;
//         while(i++ < j--) {
//             std::swap(route[l], route[r]);
//             pos_in_route[route[l]] = l;
//             pos_in_route[route[r]] = r;
//             l = (l+1) % length;
//             r = (r > 0) ? r - 1 : length - 1;
//         }
//     }
// }

void rotate(int first, int middle, int last, uint32_t route[NODES]) {
    uint32_t adjusted_route[NODES];
    for (int i = 0; i < last - middle; i++) {
        adjusted_route[i + first] = route[i + middle];
    }
    for (int i = 0; i < middle - first; i++) {
        adjusted_route[i + first + last - middle] = route[first + i];
    }

    for (int i = first; i < last; i++) {
        route[i] = adjusted_route[i];
    }
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
        // std::swap(first_, last_);
        swap(first_, last_);
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


// void perform_2_opt_move(std::vector<uint32_t> &route, int32_t i, int32_t j) {
//     flip_route_section(route, i+1, j+1);
// }


/**
 * Performs route modifications required by a 3-opt move, i.e. segments
 * reversals and swaps.
 *
 * The longest of the { s0, s1, s2 } segments is not modified, instead the
 * remaining segments are reversed if necessary.
 *
 * This works only for the symmetric version of the TSP.
 */
void perform_3_opt_move(uint32_t route[NODES],//std::vector<uint32_t> &route,
                        Segment s0, Segment s1, Segment s2) {
    // Sort segments so that the longest one is the first - it
    // will be kept without changes
    if (s0.size() < s1.size()) {
        // std::swap(s0, s1);
        swap(s0, s1);
    }
    if (s0.size() < s2.size()) {
        // std::swap(s0, s2);
        swap(s0, s2);
    }
    if (s1.size() < s2.size()) {
        // std::swap(s1, s2);
        swap(s1, s2);
    }
    // Segments should be sorted
    // assert((s0.size() >= s1.size()) && (s1.size() >= s2.size()));

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

        // RelativeIndex idx(static_cast<int32_t>(s1.first_), static_cast<int32_t>(route.size()));
        RelativeIndex idx(static_cast<int32_t>(s1.first_), static_cast<int32_t>(NODES));
        for (int32_t l = 0, r = s1.isize() - 1; l < r; ++l, --r) {
            // std::swap(route[idx(l)], route[idx(r)]);
            swap(route[idx(l)], route[idx(r)]);
        }
    }
    if (s2.is_reversed_) {
        s2.reverse();

        // RelativeIndex idx(static_cast<int32_t>(s2.first_), static_cast<int32_t>(route.size()));
        RelativeIndex idx(static_cast<int32_t>(s2.first_), static_cast<int32_t>(NODES));
        for (int32_t l = 0, r = s2.isize() - 1; l < r; ++l, --r) {
            // std::swap(route[idx(l)], route[idx(r)]);
            swap(route[idx(l)], route[idx(r)]);
        }
    }
    if (swap_needed) {
        // auto beg = route.begin();
        int32_t beg = 0;
        // Now perform the swap of s1 and s2, we use std::rotate
        if (s1.id_ == 2 && s2.id_ == 1) { // 0 2 1, easy case
            // std::rotate(beg + s2.first(),
            //        beg + s1.first(),
            //        beg + s1.last() + 1);
            rotate(beg + s2.first(), beg + s1.first(), beg + s1.last() + 1, route);
        } else if (s1.id_ == 1 && s2.id_ == 2) { // 0 1 2, easy case
            // std::rotate(beg + s1.first(),
            //        beg + s2.first(),
            //        beg + s2.last() + 1);
            rotate(beg + s1.first(), beg + s2.first(), beg + s2.last() + 1, route);
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
            // auto Middle = static_cast<int32_t>((left + middle) % route.size());
            auto Middle = static_cast<int32_t>((left + middle) % NODES);
            // auto Last =static_cast<int32_t>((left + right) % route.size());
            auto Last =static_cast<int32_t>((left + right) % NODES);

            // auto N = static_cast<int32_t>(route.size());
            auto N = static_cast<int32_t>(NODES);

            auto Next = Middle;
            while (First != Next) {
                // std::swap (route[First], route[Next]);
                swap(route[First], route[Next]);

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
int64_t three_opt_nn(int32_t coords_in[NODES][2],
                   uint32_t neighbors_in[NODES][NN_LIST_SIZE],
                   uint32_t route_in[NODES],
                   uint32_t checklist_in[CHECKLIST_CAPACITY],
                   uint32_t first_in,
                   uint32_t CHECKLIST_SIZE_IN) {

    #pragma hls interface m_axi depth=1 port=coords_in bundle=mem
    #pragma hls interface m_axi depth=1 port=route_in bundle=mem
    #pragma hls interface m_axi depth=1 port=checklist_in bundle=mem
    #pragma hls interface m_axi depth=1 port=neighbors_in bundle=mem

    #pragma hls interface s_axilite register port=CHECKLIST_SIZE_IN
    #pragma hls interface s_axilite register port=first_in
    #pragma HLS INTERFACE s_axilite register port=return

    // using namespace std;

    // assert( instance.is_symmetric_ );

    // const auto len = static_cast<uint32_t>(sol.size());
    const auto len = NODES;
    // auto &route = route_in;
    static int32_t coords[NODES][2];
    uint32_t route[NODES];
    uint32_t checklist[CHECKLIST_CAPACITY];
    static uint32_t neighbors[NODES][NN_LIST_SIZE];
    uint32_t first = first_in;
    uint32_t CHECKLIST_SIZE = CHECKLIST_SIZE_IN;

    if (first)
    {
        for (int i = 0; i < NODES; i++) {
            coords[i][0] = coords_in[i][0];
            coords[i][1] = coords_in[i][1];
            for (int j = 0; j < NN_LIST_SIZE; j++)
            {
                neighbors[i][j] = neighbors_in[i][j];
            }
        }
    }



    for (int i = 0; i < NODES; i++)
    {
        route[i] = route_in[i];
    }
    for (int j = 0; j < CHECKLIST_CAPACITY; j++)
    {
        checklist[j] = checklist_in[j];
    }

    // Bitmask dont_look_bits(len);
    // vector<uint32_t> pos_in_route(len);
    uint32_t pos_in_route[NODES];
    // for (uint32_t i = 0; i < NODES; ++i) {
    //     pos_in_route[ route[i] ] = i;
    // }

    const int nn_count = NN_LIST_SIZE;
    // const int route_size = NODES;
    int64_t two_opt_changes = 0;
    int64_t three_opt_changes = 0;

    bool found_improvement;

    int loop_count = 0;
    do {
        found_improvement = false;
        loop_count++;

        for (auto i = 0u; i < len; ++i) {
            pos_in_route[ route[i] ] = i;
        }

        size_t checklist_pos_pos = 0;
        // for (auto i = 0u; i < len && !found_improvement; ++i) {
        for (int m = 0; m < CHECKLIST_CAPACITY; m++) {
            uint32_t at_i = checklist[checklist_pos_pos++];
            // const auto at_i = route[i];
            if (checklist_pos_pos >= CHECKLIST_SIZE)
            {
                break;
            }
            uint32_t i = pos_in_route[at_i];
            // if (use_dont_look_bits && dont_look_bits[at_i]) {
            //     continue ;  // Do not check, it probably won't find an
            //                 // improvement
            // }
            // const auto &i_nn_list = instance.get_nearest_neighbors(at_i, nn_count);
            const auto &i_nn_list = neighbors_in[at_i];
            for (auto i_nn_idx = 0u; i_nn_idx < nn_count && !found_improvement; ++i_nn_idx) {
            // for (auto i_nn_idx = 0u; i_nn_idx < nn_count; ++i_nn_idx) {
                const auto at_j = i_nn_list[i_nn_idx];
                const auto j = pos_in_route[at_j];

                // Check for 2-opt move
                const auto i_1 = (i + 1) % len;
                const auto j_1 = (j + 1) % len;
                const auto at_i_1 = route[i_1];

                // const auto dist_i_to_next = instance.get_distance(at_i, at_i_1);
                // const auto dist_i_to_j = instance.get_distance(at_i, at_j);

                const auto dist_i_to_next = get_distance(at_i, at_i_1, coords);
                const auto dist_i_to_j = get_distance(at_i, at_j, coords);

                // This shortens time considerably although results in longer tours
                if (dist_i_to_next < dist_i_to_j) {
                    break ;
                }

                const auto at_j_1 = route[j_1];

                // auto cost_before_2opt = dist_i_to_next
                //                       + instance.get_distance(at_j, at_j_1);

                // auto cost_after_2opt = dist_i_to_j
                //                      + instance.get_distance(at_i_1, at_j_1);

                auto cost_before_2opt = dist_i_to_next
                                      + get_distance(at_j, at_j_1, coords);

                auto cost_after_2opt = dist_i_to_j
                                     + get_distance(at_i_1, at_j_1, coords);

                if (cost_after_2opt < cost_before_2opt) {
                    // perform_2_opt_move(route, static_cast<int32_t>(i), static_cast<int32_t>(j));
                    flip_route_section(route, pos_in_route, static_cast<int32_t>(i) + 1, static_cast<int32_t>(j) + 1);
                    found_improvement = true;

                    // dont_look_bits.clear_bit(at_i);
                    // dont_look_bits.clear_bit(at_i_1);

                    // dont_look_bits.clear_bit(at_j);
                    // dont_look_bits.clear_bit(at_j_1);
                    uint32_t endpoints[] = {at_i, at_i_1, at_j, at_j_1};
                    for (int idx = 0; idx < 4; idx++) {
                        bool exists = false;
                        for (int jdx = checklist_pos_pos; jdx < CHECKLIST_SIZE; jdx++)
                        {
                            if (checklist[jdx] == endpoints[idx])
                            {
                                exists = true;
                            }
                        }
                        if (!exists)
                        {
                            checklist[CHECKLIST_SIZE++] = endpoints[idx];
                        }
                    }

                    ++two_opt_changes;

                    continue ;
                }

                // const auto &j_nn_list = instance.get_nearest_neighbors(at_j, nn_count);
                const auto &j_nn_list = neighbors[at_j];

                // assert(at_i != at_j);  // These two should be different

                for (auto j_nn_idx = 0u; j_nn_idx < nn_count && !found_improvement ; ++j_nn_idx) {
                // for (auto j_nn_idx = 0u; j_nn_idx < nn_count ; ++j_nn_idx) {
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

                    // const auto curr = instance.get_distance(at_x, at_x_1)
                    //                 + instance.get_distance(at_y, at_y_1)
                    //                 + instance.get_distance(at_z, at_z_1);

                    const auto curr = get_distance(at_x, at_x_1, coords)
                                    + get_distance(at_y, at_y_1, coords)
                                    + get_distance(at_z, at_z_1, coords);

                    // 4 sets of possible new edges to check
                    // const std::array<std::pair<uint32_t, uint32_t>, 4 * 3> edges{{
                    //     { at_y, at_x   }, { at_z_1, at_y_1 }, {   at_z, at_x_1 },
                    //     { at_y, at_z_1 }, {   at_x, at_y_1 }, {   at_z, at_x_1 },
                    //     { at_y, at_z_1 }, {   at_x, at_z   }, { at_y_1, at_x_1 },
                    //     { at_y, at_z   }, { at_y_1, at_x   }, { at_z_1, at_x_1 }
                    // }};

                    const uint32_t edges[4 * 3][2] = {
                        { at_y, at_x   }, { at_z_1, at_y_1 }, {   at_z, at_x_1 },
                        { at_y, at_z_1 }, {   at_x, at_y_1 }, {   at_z, at_x_1 },
                        { at_y, at_z_1 }, {   at_x, at_z   }, { at_y_1, at_x_1 },
                        { at_y, at_z   }, { at_y_1, at_x   }, { at_z_1, at_x_1 }
                    };

                    // Which segments do we need to reverse in order to transform
                    // route so that the new edges are created properly
                    // const std::array<bool, 4 * 3> segment_reversals{{
                    //     false, true, true,
                    //     true, true, true,
                    //     true, true, false,
                    //     true, false, true
                    // }};

                    const bool segment_reversals[12] = {
                        false, true, true,
                        true, true, true,
                        true, true, false,
                        true, false, true
                    };

                    Segment seg[3] = {
                        { z_1, x, len, 0 },
                        { x_1, y, len, 1 },
                        { y_1, z, len, 2 }
                    };

                    int32_t best_cost = (1 << 31) - 1;
                    auto ll = 0u;
                    for (auto l = 0u; l < 4 * 3 && !found_improvement; l += 3) {
                    // for (auto l = 0u; l < 4 * 3; l += 3) {
                        auto e1 = edges[l + 0];
                        auto e2 = edges[l + 1];
                        auto e3 = edges[l + 2];

                        // const auto cost = instance.get_distance(e1.first, e1.second)
                        //                 + instance.get_distance(e2.first, e2.second)
                        //                 + instance.get_distance(e3.first, e3.second);

                        const auto cost = get_distance(e1[0], e1[1], coords)
                                        + get_distance(e2[0], e2[1], coords)
                                        + get_distance(e3[0], e3[1], coords);

                        // if (cost < curr) if (cost < best_cost) {best_cost = cost; ll = l;}
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
                            // dont_look_bits.clear_bit(e1.first);
                            // dont_look_bits.clear_bit(e1.second);

                            // dont_look_bits.clear_bit(e2.first);
                            // dont_look_bits.clear_bit(e2.second);

                            // dont_look_bits.clear_bit(e3.first);
                            // dont_look_bits.clear_bit(e3.second);
                            uint32_t endpoints[] = {e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
                            for (int idx = 0; idx < 6; idx++) {
                                bool exists = false;
                                for (int jdx = checklist_pos_pos; jdx < CHECKLIST_SIZE; jdx++)
                                {
                                    if (checklist[jdx] == endpoints[idx])
                                    {
                                        exists = true;
                                    }
                                }
                                if (!exists)
                                {
                                    checklist[CHECKLIST_SIZE++] = endpoints[idx];
                                }
                            }
                            ++three_opt_changes;
                        }
                    }
                    // auto e1 = edges[ll + 0];
                    // auto e2 = edges[ll + 1];
                    // auto e3 = edges[ll + 2];
                    // if (best_cost < curr) {
                    //     found_improvement = true;

                    //     if (segment_reversals[ll + 0]) {
                    //         seg[0].reverse();
                    //     }
                    //     if (segment_reversals[ll + 1]) {
                    //         seg[1].reverse();
                    //     }
                    //     if (segment_reversals[ll + 2]) {
                    //         seg[2].reverse();
                    //     }
                    //     // dont_look_bits.clear_bit(e1.first);
                    //     // dont_look_bits.clear_bit(e1.second);

                    //     // dont_look_bits.clear_bit(e2.first);
                    //     // dont_look_bits.clear_bit(e2.second);

                    //     // dont_look_bits.clear_bit(e3.first);
                    //     // dont_look_bits.clear_bit(e3.second);
                    //     uint32_t endpoints[] = {e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
                    //     for (int idx = 0; idx < 6; idx++) {
                    //         bool exists = false;
                    //         for (int jdx = checklist_pos_pos; jdx < CHECKLIST_SIZE; jdx++)
                    //         {
                    //             if (checklist[jdx] == endpoints[idx])
                    //             {
                    //                 exists = true;
                    //             }
                    //         }
                    //         if (!exists)
                    //         {
                    //             checklist[CHECKLIST_SIZE++] = endpoints[idx];
                    //         }
                    //     }
                    //     ++three_opt_changes;
                    //     perform_3_opt_move(route, seg[0], seg[1], seg[2]);
                    // }
                    if (found_improvement) {
                        perform_3_opt_move(route, seg[0], seg[1], seg[2]);
                    }
                }
            }
            // if (!found_improvement && use_dont_look_bits) {
            //     dont_look_bits.set_bit(at_i);
            // }
        }
    } while(found_improvement && loop_count < NODES);

    for (int i = 0; i < NODES; i++)
    {
        route_in[i] = route[i];
    }

    return two_opt_changes + three_opt_changes;
}