#pragma once
#include "ops.h"

// Single-head self-attention.
// input:  [seq, d_model]
// output: [seq, d_model]
//
// Q = input @ W_Q
// K = input @ W_K
// V = input @ W_V
// scores = Q @ K.T / sqrt(d_model)
// out = softmax(scores) @ V @ W_O
struct Attention {
    Tensor W_Q, W_K, W_V, W_O;  // [d_model, d_model] each
    int d_model;

    Attention(int d)
        : W_Q({d, d}), W_K({d, d}), W_V({d, d}), W_O({d, d}), d_model(d) {}

    // Forward: returns [seq, d_model]
    Tensor forward(const Tensor& input) {
        const int seq = input.rows(), d = d_model;

        // Q, K, V = input @ W_*
        Tensor Q({seq, d}), K({seq, d}), V({seq, d});
        matmul(input, W_Q, Q);
        matmul(input, W_K, K);
        matmul(input, W_V, V);

        // scores = Q @ K.T / sqrt(d)   [seq, seq]
        float scale = 1.f / sqrtf((float)d);
        Tensor scores({seq, seq});
        scores.zero();
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < seq; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += Q.at(i, k) * K.at(j, k);
                scores.at(i, j) = dot * scale;
            }

        // Causal mask: position i can only attend to j <= i
        for (int i = 0; i < seq; ++i)
            for (int j = i + 1; j < seq; ++j)
                scores.at(i, j) = -1e9f;

        // softmax over each row
        softmax(scores);

        // context = scores @ V   [seq, d]
        Tensor context({seq, d});
        matmul(scores, V, context);

        // output = context @ W_O   [seq, d]
        Tensor out({seq, d});
        matmul(context, W_O, out);
        return out;
    }
};
