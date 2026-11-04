// #include <cstdint>
// #include <cmath>
// #include <algorithm>
#include  "hls_math.h"


#define NODES 2392
#define NN_LIST_SIZE 20
#define CHECKLIST_CAPACITY 500

using namespace std;


int32_t get_distance(uint32_t a, uint32_t b, int32_t coords[NODES][2])
{
    #pragma HLS INLINE
    int32_t p1[2] = {coords[a][0], coords[a][1]};
    int32_t p2[2] = {coords[b][0], coords[b][1]};
	float val = sqrt(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])));
    return static_cast<int32_t>(val + 0.5);
}

void swap(int32_t &a, int32_t &b)
{
    #pragma HLS INLINE
    int32_t temp = a;
    a = b;
    b = temp;
}

void swap(uint32_t &a, uint32_t &b)
{
    #pragma HLS INLINE
    uint32_t temp = a;
    a = b;
    b = temp;
}

uint32_t min(uint32_t a, uint32_t b)
{
    #pragma HLS INLINE
    if (a < b) return a;
    return b;
}

uint32_t max(uint32_t a, uint32_t b)
{
    #pragma HLS INLINE
    if (a > b) return a;
    return b;
}

void flip_route_section(uint32_t route[NODES],
                        uint32_t pos_in_route[NODES],
                        int32_t first, int32_t last) {

    if (first > last) {
        swap(first, last);
    }

    // std::cout << "First: " << first << ", Last: " << last << std::endl;

    const int32_t length = static_cast<int32_t>(NODES);
    const int32_t segment_length = last - first;
    const int32_t remaining_length = length - segment_length;

    if (segment_length <= remaining_length) {  // Reverse the specified segment
        
        int32_t i = first;
        int32_t j = last - 1;
        
        FLIP_1:
        for (i = first; i < j; i++) { // 3 cycles per iteration?
            // #pragma HLS unroll
            swap(route[i], route[j]);            
            j--;
        }
        FLIP_2: // 1 cycle per iteration
        for (int32_t k = first; k < last; ++k) {
            pos_in_route[ route[k] ] = k;
        }
    } else {  // Reverse the rest of the route, leave the segment intact
        first = (first > 0) ? first - 1 : length - 1;
        last = last % length;
        swap(first, last);
        int32_t l = first;
        int32_t r = last;
        int32_t i; //= 0;
        int32_t j = length - first + last + 1;

        FLIP_3:
        for (i = 0; i < j; i += 2) { // 16 cycles per iteration
            swap(route[l], route[r]);
            pos_in_route[route[l]] = l;
            pos_in_route[route[r]] = r;
            l = (l+1) % length;
            r = (r > 0) ? r - 1 : length - 1;
        }
    }
}

uint32_t two_opt_nn(int32_t coords_in[NODES][2],
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
    // We assume symmetry so that the order of the nodes does not matter
    
    
    int32_t coords[NODES][2];
    uint32_t route[NODES];
    uint32_t checklist[CHECKLIST_CAPACITY];
    uint32_t neighbors[NODES][NN_LIST_SIZE];
    uint32_t first = first_in;
    uint32_t CHECKLIST_SIZE = CHECKLIST_SIZE_IN;
    uint32_t endpoints[4];

    #pragma HLS bind_storage variable=route type=ram_1wnr
    #pragma HLS array_partition variable=route dim=1 type=cyclic factor=2
    #pragma HLS array_partition variable=coords dim=1 type=cyclic factor=2
    // #pragma HLS array_partition variable=neighbors dim=2 type=cyclic factor=2
    // #pragma HLS array_partition variable=checklist dim=1 type=cyclic factor=500
    // #pragma HLS array_partition variable=endpoints dim=1 type=complete

    // We assume symmetry so that the order of the nodes does not matter
    const uint32_t route_size = NODES;
    uint32_t pos_in_route[route_size];
    // #pragma HLS bind_storage variable=pos_in_route type=ram_1wnr
    // #pragma HLS array_partition variable=pos_in_route dim=1 type=cyclic factor=2

    // Data transfer from DRAM to BRAM
    if (first)
    {
        COORDS_TRANSFER:
        for (int i = 0; i < NODES; i++) { // 22 cycles per iteration
            coords[i][0] = coords_in[i][0];
            coords[i][1] = coords_in[i][1];
            NEIGHBORS_TRANSFER:
            for (int j = 0; j < NN_LIST_SIZE; j++)
            {
                neighbors[i][j] = neighbors_in[i][j];
            }
        }
    }
    
    // Data transfer from DRAM to BRAM
    ROUTE_TRANSFER:
    for (int i = 0; i < NODES; i++) // 1 cycle per iteration
    {
        route[i] = route_in[i];
    }
    
    // Data transfer from DRAM to BRAM
    CHECKLIST_TRANSFER:
    for (int j = 0; j < CHECKLIST_CAPACITY; j++) // 1 cycle per iteration
    {
        checklist[j] = checklist_in[j];
    }

    POS_IN_ROUTE_I:
    for (uint32_t i = 0; i < NODES / 13; ++i) {
        // #pragma HLS pipeline II=1
        POS_IN_ROUTE_J:
        for (uint32_t j = 13 * i; j < 13 * (i + 1); j++) { // 1 cycle per iteration
            // #pragma HLS unroll
            // std::cout << j << std::endl;
            pos_in_route[ route[j] ] = j;
        }     
    }
    const uint32_t n = NODES;

    // Setting maximum number of allowed route changes prevents very long-running times
    // for very hard to solve TSP instances.
    const uint32_t MaxChanges = route_size;
    uint32_t changes_count = 0;

    size_t checklist_pos_pos = 0;
    
    CHECKLIST_LOOP:
    for (int m = 0; m < CHECKLIST_CAPACITY; m++) {
        // #pragma HLS pipeline II=5
        uint32_t a = checklist[checklist_pos_pos++];
        if (checklist_pos_pos >= CHECKLIST_SIZE)
        {
            break;
        } 
        uint32_t i = pos_in_route[a];

        uint32_t a_next = (i + 1 < n) ? route[i+1] : route[0];
        uint32_t a_prev = (i > 0) ? route[i-1] : route[route_size-1];

        int32_t dist_a_to_next = get_distance(a, a_next, coords);
        int32_t dist_a_to_prev = get_distance(a, a_prev, coords);

        int32_t max_diff = -1;
        int32_t left = 0;
        int32_t right = 0;
        int32_t sum = 0;

        DIST_A_OUTER:
        for (uint32_t k = 0; k < NN_LIST_SIZE / 2; k++) { // 1 cycle per iteration
            #pragma HLS pipeline II=1
            DIST_A_INNER:
            for (uint32_t kk = 2 * k; kk < 2 * (k + 1); kk++) {
                #pragma HLS unroll
                uint32_t b = neighbors[a][kk]; 
                int32_t dist_ab = get_distance(a, b, coords);

                uint32_t b_pos = pos_in_route[b];
                uint32_t b_next = (b_pos + 1 < route_size) ? route[b_pos + 1] : route[0];
                uint32_t b_prev = (b_pos > 0) ? route[b_pos-1] : route[route_size-1];

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

                    int32_t diff = dist_a_to_next
                            + get_distance(b, b_next, coords)
                            - dist_ab
                            - get_distance(a_next, b_next, coords);

                    if (diff > max_diff) {
                        left  = min(i, b_pos) + 1;
                        right = max(i, b_pos) + 1;
                        max_diff = diff;
                    }
                } else if (dist_a_to_prev > dist_ab) {
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
                    int32_t diff = dist_a_to_prev
                            + get_distance(b_prev, b, coords)
                            - dist_ab
                            - get_distance(a_prev, b_prev, coords);

                    if (diff > max_diff) {
                        left  = min(i, b_pos);
                        right = max(i, b_pos);
                        max_diff = diff;
                    }
                }
            }            
        }        

        if (max_diff > 0) {
            flip_route_section(route, pos_in_route, static_cast<int32_t>(left), static_cast<int32_t>(right));
            uint32_t endpoints[] = {
                route[left],
                route[right-1],
                route[(left > 0) ? left-1 : route_size-1],
                route[(right < route_size) ? right : 0]
            };
            ENDPOINTS_1:            
            for (int i = 0; i < 4; i++) {
                // #pragma HLS pipeline II=1
                bool exists = false; 
                ENDPOINTS_2:
                for (int j = checklist_pos_pos; j < CHECKLIST_SIZE; j++) // 1 cycle per iteration
                {
                    // #pragma HLS unroll                    
                    // if (j >= CHECKLIST_SIZE) break;
                    if (checklist[j] == endpoints[i])
                    {
                        exists = true;
                    }
                }
                if (!exists)
                {
                    checklist[CHECKLIST_SIZE++] = endpoints[i];
                }
            }
            ++changes_count;
        }
    }

    // Write back from BRAM to DRAM
    WRITE_ROUTE:
    for (int i = 0; i < NODES; i++) // 1 cycle per iteration
    {
        route_in[i] = route[i];
    }
    return changes_count;
}