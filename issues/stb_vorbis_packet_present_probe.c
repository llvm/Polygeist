#include <stdint.h>
#include <string.h>

typedef uint8_t uint8;

#define TRUE 1
#define PAGEFLAG_continued_packet 1

enum {
    VORBIS_need_more_data = 1,
    VORBIS_invalid_stream = 2,
};

static const uint8 ogg_page_header[4] = { 'O', 'g', 'g', 'S' };

struct stb_vorbis_probe {
    int next_seg;
    int segment_count;
    int previous_length;
    uint8 segments[255];
    uint8 * stream;
    uint8 * stream_end;
};

static int error_probe(struct stb_vorbis_probe * f, int e) {
    (void)f;
    return -e;
}

int isolate_packet_present_first_segment(struct stb_vorbis_probe * f) {
    int s = f->next_seg, first = TRUE;
    uint8 * p = f->stream;

    if (s != -1) {
        for (; s < f->segment_count; ++s) {
            p += f->segments[s];
            if (f->segments[s] < 255) {
                break;
            }
        }
        if (s == f->segment_count) {
            s = -1;
        }
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    return s + first;
}

int isolate_packet_present_cross_page(struct stb_vorbis_probe * f) {
    int s = -1, first = TRUE;
    uint8 * p = f->stream;

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        if (memcmp(p, ogg_page_header, 4)) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (p[4] != 0) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (first) {
            if (f->previous_length) {
                if ((p[5] & PAGEFLAG_continued_packet)) {
                    return error_probe(f, VORBIS_invalid_stream);
                }
            }
        } else {
            if (!(p[5] & PAGEFLAG_continued_packet)) {
                return error_probe(f, VORBIS_invalid_stream);
            }
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    return TRUE;
}

int isolate_packet_present_cross_page_core(struct stb_vorbis_probe * f) {
    int s = -1;
    uint8 * p = f->stream;

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
    }

    return TRUE;
}

int isolate_packet_present_cross_page_no_memcmp(struct stb_vorbis_probe * f) {
    int s = -1, first = TRUE;
    uint8 * p = f->stream;

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        if (p[0] != 'O' || p[1] != 'g' || p[2] != 'g' || p[3] != 'S') {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (p[4] != 0) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (first) {
            if (f->previous_length) {
                if ((p[5] & PAGEFLAG_continued_packet)) {
                    return error_probe(f, VORBIS_invalid_stream);
                }
            }
        } else {
            if (!(p[5] & PAGEFLAG_continued_packet)) {
                return error_probe(f, VORBIS_invalid_stream);
            }
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    return TRUE;
}

int isolate_packet_present_cross_page_flags_only(struct stb_vorbis_probe * f) {
    int s = -1, first = TRUE;
    uint8 * p = f->stream;

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        if (first) {
            if (f->previous_length) {
                if ((p[5] & PAGEFLAG_continued_packet)) {
                    return error_probe(f, VORBIS_invalid_stream);
                }
            }
        } else {
            if (!(p[5] & PAGEFLAG_continued_packet)) {
                return error_probe(f, VORBIS_invalid_stream);
            }
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    return TRUE;
}

int isolate_packet_present_cross_page_no_flags(struct stb_vorbis_probe * f) {
    int s = -1;
    uint8 * p = f->stream;

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        if (memcmp(p, ogg_page_header, 4)) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (p[4] != 0) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
    }

    return TRUE;
}

int isolate_packet_present_full(struct stb_vorbis_probe * f) {
    int s = f->next_seg, first = TRUE;
    uint8 * p = f->stream;

    if (s != -1) {
        for (; s < f->segment_count; ++s) {
            p += f->segments[s];
            if (f->segments[s] < 255) {
                break;
            }
        }
        if (s == f->segment_count) {
            s = -1;
        }
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    for (; s == -1;) {
        uint8 * q;
        int n;

        if (p + 26 >= f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        if (memcmp(p, ogg_page_header, 4)) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (p[4] != 0) {
            return error_probe(f, VORBIS_invalid_stream);
        }
        if (first) {
            if (f->previous_length) {
                if ((p[5] & PAGEFLAG_continued_packet)) {
                    return error_probe(f, VORBIS_invalid_stream);
                }
            }
        } else {
            if (!(p[5] & PAGEFLAG_continued_packet)) {
                return error_probe(f, VORBIS_invalid_stream);
            }
        }
        n = p[26];
        q = p + 27;
        p = q + n;
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
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
        if (p > f->stream_end) {
            return error_probe(f, VORBIS_need_more_data);
        }
        first = 0;
    }

    return TRUE;
}
