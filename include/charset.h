#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>

// Maps characters ↔ integer token ids
struct Charset {
    char idx_to_char[128] = {};
    int  char_to_idx[256] = {};
    int  size = 0;

    Charset() { memset(char_to_idx, -1, sizeof(char_to_idx)); }

    void build(const char* text, int len) {
        for (int i = 0; i < len; ++i) {
            unsigned char c = (unsigned char)text[i];
            if (char_to_idx[c] == -1) {
                assert(size < 128);
                char_to_idx[c] = size;
                idx_to_char[size++] = (char)c;
            }
        }
    }

    int  encode(char c)  const { return char_to_idx[(unsigned char)c]; }
    char decode(int idx) const { return idx_to_char[idx]; }
};
