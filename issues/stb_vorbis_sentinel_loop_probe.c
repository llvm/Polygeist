#include <stdint.h>

typedef uint8_t uint8;

int isolate_sentinel_scalar_loop(int n, const uint8 * q) {
    int s = -1;
    int total = 0;

    for (; s == -1;) {
        for (s = 0; s < n; ++s) {
            total += q[s];
            if (q[s] < 255) {
                break;
            }
        }
        if (s == n) {
            s = -1;
        }
    }

    return total + s;
}

int isolate_sentinel_pointer_loop(uint8 * p, const uint8 * q, int n) {
    int s = -1;

    for (; s == -1;) {
        for (s = 0; s < n; ++s) {
            p += q[s];
            if (q[s] < 255) {
                break;
            }
        }
        if (s == n) {
            s = -1;
        }
    }

    return (int)(p - q) + s;
}

int isolate_sentinel_pointer_checked_loop(uint8 * p, uint8 * end, const uint8 * q, int n) {
    int s = -1;

    for (; s == -1;) {
        if (p + 26 >= end) {
            return -1;
        }
        for (s = 0; s < n; ++s) {
            p += q[s];
            if (q[s] < 255) {
                break;
            }
        }
        if (s == n) {
            s = -1;
        }
        if (p > end) {
            return -2;
        }
    }

    return (int)(p - q) + s;
}
