#include "attention.h"
#include "charset.h"
#include "embedding.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

static const int D_MODEL = 32;
static const int SEQ_LEN = 8;

static void xavier(Tensor& W, int fan_in, int fan_out) {
    float limit = sqrtf(6.f / (fan_in + fan_out));
    for (size_t i = 0; i < W.numel; ++i)
        W.data[i] = limit * (2.f * (float)rand() / (float)RAND_MAX - 1.f);
}

int main() {
    srand(42);

    const char* text = "the cat sat on the mat";
    Charset cs;
    cs.build(text, (int)strlen(text));
    std::cout << "Vocab size: " << cs.size << "\n";

    int seq[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i) seq[i] = cs.encode(text[i]);

    std::cout << "Input sequence: \"";
    for (int i = 0; i < SEQ_LEN; ++i) std::cout << cs.decode(seq[i]);
    std::cout << "\"\n\n";

    Embedding emb(cs.size, D_MODEL);
    xavier(emb.E, 1, D_MODEL);

    Tensor emb_out({SEQ_LEN, D_MODEL});
    emb.forward(seq, SEQ_LEN, emb_out);

    Attention attn(D_MODEL);
    xavier(attn.W_Q, D_MODEL, D_MODEL);
    xavier(attn.W_K, D_MODEL, D_MODEL);
    xavier(attn.W_V, D_MODEL, D_MODEL);
    xavier(attn.W_O, D_MODEL, D_MODEL);

    Tensor out = attn.forward(emb_out);

    std::cout << "Attention output shape: [" << out.rows() << ", " << out.cols() << "]\n";
    std::cout << "First token output (first 8 values): ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(3) << out.at(0, i) << " ";
    std::cout << "\n\n";

    std::cout << "Attention weight matrix (row i attends to col j):\n\n";

    Tensor Q({SEQ_LEN, D_MODEL}), K({SEQ_LEN, D_MODEL});
    matmul(emb_out, attn.W_Q, Q);
    matmul(emb_out, attn.W_K, K);

    float scale = 1.f / sqrtf((float)D_MODEL);
    Tensor scores({SEQ_LEN, SEQ_LEN});
    scores.zero();
    for (int i = 0; i < SEQ_LEN; ++i)
        for (int j = 0; j < SEQ_LEN; ++j) {
            float dot = 0.f;
            for (int k = 0; k < D_MODEL; ++k) dot += Q.at(i, k) * K.at(j, k);
            scores.at(i, j) = dot * scale;
        }
    for (int i = 0; i < SEQ_LEN; ++i)
        for (int j = i + 1; j < SEQ_LEN; ++j)
            scores.at(i, j) = -1e9f;
    softmax(scores);

    std::cout << "     ";
    for (int j = 0; j < SEQ_LEN; ++j) std::cout << "  '" << cs.decode(seq[j]) << "'  ";
    std::cout << "\n";

    for (int i = 0; i < SEQ_LEN; ++i) {
        std::cout << "'" << cs.decode(seq[i]) << "'  ";
        for (int j = 0; j < SEQ_LEN; ++j) {
            float w = scores.at(i, j);
            if (w < 0.01f) std::cout << "  .    ";
            else           std::cout << " " << std::fixed << std::setprecision(2) << w << "  ";
        }
        std::cout << "\n";
    }

    std::cout << "\nEach row shows where that character is attending.\n";
    std::cout << "1.00 = full attention, . = masked or near zero.\n";

    return 0;
}
