#pragma once
#include "tensor.h"
#include <cstring>

// Lookup table: token id → float vector of size embed_dim
struct Embedding {
    Tensor E; // [vocab_size, embed_dim]
    int vocab_size, embed_dim;

    Embedding(int vocab, int dim)
        : E({vocab, dim}), vocab_size(vocab), embed_dim(dim) { E.alloc_grad(); }

    // Copy embedding rows into out [seq_len, embed_dim]
    void forward(const int* tokens, int seq_len, Tensor& out) const {
        assert(out.rows() == seq_len && out.cols() == embed_dim);
        for (int i = 0; i < seq_len; ++i)
            memcpy(&out.at(i, 0), &E.at(tokens[i], 0), embed_dim * sizeof(float));
    }

    // Accumulate upstream gradient into embedding rows
    void backward(const int* tokens, int seq_len, const Tensor& grad_out) {
        for (int i = 0; i < seq_len; ++i)
            for (int d = 0; d < embed_dim; ++d)
                E.gat(tokens[i], d) += grad_out.at(i, d);
    }

    void update(float lr) {
        for (size_t i = 0; i < E.numel; ++i) E.data[i] -= lr * E.grad[i];
        E.zero_grad();
    }

    void zero_grad() { E.zero_grad(); }
};
