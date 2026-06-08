#pragma once
#include "net.h"
#include <cstdio>
#include <cstdlib>

// Read exactly n bytes or abort
static void read_exact(FILE* f, void* dst, size_t n) {
    size_t got = fread(dst, 1, n, f);
    if (got != n) { fprintf(stderr, "read_exact: got %zu/%zu\n", got, n); abort(); }
}

// Load weights.bin → MLP
// Format: [num_layers: i32] then per layer: [in: i32][out: i32][W: floats][b: floats]
inline MLP load_mlp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); abort(); }

    int32_t num_layers;
    read_exact(f, &num_layers, sizeof(num_layers));

    struct LayerData { int in_f, out_f; std::vector<float> W, b; };
    std::vector<LayerData> ld(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        int32_t in_f, out_f;
        read_exact(f, &in_f, sizeof(in_f));
        read_exact(f, &out_f, sizeof(out_f));
        ld[i] = {in_f, out_f, std::vector<float>((size_t)in_f * out_f), std::vector<float>((size_t)out_f)};
        read_exact(f, ld[i].W.data(), ld[i].W.size() * sizeof(float));
        read_exact(f, ld[i].b.data(), ld[i].b.size() * sizeof(float));
    }
    fclose(f);

    std::vector<int> sizes;
    sizes.push_back(ld[0].in_f);
    for (auto& l : ld) sizes.push_back(l.out_f);

    MLP net(sizes);
    for (int i = 0; i < num_layers; ++i) {
        memcpy(net.layers[i].W.data, ld[i].W.data(), ld[i].W.size() * sizeof(float));
        memcpy(net.layers[i].b.data, ld[i].b.data(), ld[i].b.size() * sizeof(float));
    }

    printf("Loaded %d layers from %s\n", num_layers, path);
    for (auto& l : ld) printf("  Linear(%d -> %d)\n", l.in_f, l.out_f);
    return net;
}

// Load a flat float32 binary file into a [rows, cols] tensor
inline Tensor load_tensor(const char* path, int rows, int cols) {
    Tensor t({rows, cols});
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); abort(); }
    read_exact(f, t.data, (size_t)rows * cols * sizeof(float));
    fclose(f);
    return t;
}
