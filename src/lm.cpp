#include "embedding.h"
#include "net.h"
#include "charset.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>

static const int   CTX       = 8;
static const int   EMBED_DIM = 16;
static const int   HIDDEN    = 128;
static const int   EPOCHS    = 150;
static const int   BATCH     = 32;
static const float LR        = 0.05f;

static std::vector<char> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { std::cerr << "Cannot open " << path << "\n"; exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    std::vector<char> buf(sz + 1);
    fread(buf.data(), 1, sz, f); buf[sz] = '\0';
    fclose(f);
    return buf;
}

// Sample next token with temperature scaling
static int sample(const float* probs, int vocab_size, float temperature = 0.8f) {
    std::vector<float> p(vocab_size);
    float sum = 0.f;
    for (int i = 0; i < vocab_size; ++i) { p[i] = powf(probs[i], 1.f / temperature); sum += p[i]; }
    float r = (float)rand() / (float)RAND_MAX * sum;
    for (int i = 0; i < vocab_size; ++i) { r -= p[i]; if (r <= 0.f) return i; }
    return vocab_size - 1;
}

static void generate(Embedding& emb, MLP& net, const Charset& cs,
                     const char* seed, int gen_len) {
    int ctx[CTX] = {};
    int seed_len = (int)strlen(seed);
    for (int i = 0; i < CTX; ++i) ctx[i] = cs.encode(' ');
    for (int i = 0; i < seed_len && i < CTX; ++i)
        ctx[CTX - seed_len + i] = cs.encode(seed[i]);

    std::cout << "\n--- Generated (seed: \"" << seed << "\") ---\n" << seed;

    for (int t = 0; t < gen_len; ++t) {
        Tensor emb_out({CTX, EMBED_DIM});
        emb.forward(ctx, CTX, emb_out);

        int flat_dim = CTX * EMBED_DIM;
        Tensor input({1, flat_dim});
        memcpy(input.data, emb_out.data, flat_dim * sizeof(float));

        Tensor probs = net.forward(input);
        int next = sample(probs.data, cs.size);
        std::cout << cs.decode(next) << std::flush;

        for (int i = 0; i < CTX - 1; ++i) ctx[i] = ctx[i + 1];
        ctx[CTX - 1] = next;
    }
    std::cout << "\n---\n";
}

int main() {
    srand((unsigned)time(nullptr));

    auto text_buf = read_file("data.txt");
    const char* text = text_buf.data();
    int text_len = (int)strlen(text);

    Charset cs;
    cs.build(text, text_len);
    std::cout << "Text: " << text_len << " chars  Vocab: " << cs.size << "\n";

    std::vector<int> tokens(text_len);
    for (int i = 0; i < text_len; ++i) tokens[i] = cs.encode(text[i]);

    int input_dim = CTX * EMBED_DIM;
    Embedding emb(cs.size, EMBED_DIM);
    MLP net({input_dim, HIDDEN, HIDDEN, cs.size});

    auto xavier = [](Tensor& W, int fan_in, int fan_out) {
        float limit = sqrtf(6.f / (fan_in + fan_out));
        for (size_t i = 0; i < W.numel; ++i)
            W.data[i] = limit * (2.f * (float)rand() / (float)RAND_MAX - 1.f);
    };
    xavier(emb.E, 1, EMBED_DIM);
    for (auto& l : net.layers) xavier(l.W, l.in_features, l.out_features);

    int num_samples = text_len - CTX;
    std::cout << "Samples: " << num_samples << "\n\n";

    std::vector<std::vector<int>> all_x(num_samples, std::vector<int>(CTX));
    std::vector<int>              all_y(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < CTX; ++j) all_x[i][j] = tokens[i + j];
        all_y[i] = tokens[i + CTX];
    }

    std::cout << "Epoch   Loss\n-----   ------\n";
    int steps = num_samples / BATCH;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.f;
        for (int step = 0; step < steps; ++step) {
            int offset = step * BATCH;
            Tensor batch_x({BATCH, input_dim});
            int    batch_y[BATCH];
            for (int b = 0; b < BATCH; ++b) {
                Tensor emb_out({CTX, EMBED_DIM});
                emb.forward(all_x[offset + b].data(), CTX, emb_out);
                memcpy(&batch_x.at(b, 0), emb_out.data, input_dim * sizeof(float));
                batch_y[b] = all_y[offset + b];
            }
            Tensor probs = net.forward(batch_x);
            Tensor grad_input({BATCH, input_dim});
            epoch_loss += net.backward_and_update(probs, batch_y, LR, grad_input);

            emb.zero_grad();
            for (int b = 0; b < BATCH; ++b) {
                Tensor g_emb({CTX, EMBED_DIM});
                memcpy(g_emb.data, &grad_input.at(b, 0), input_dim * sizeof(float));
                emb.backward(all_x[offset + b].data(), CTX, g_emb);
            }
            emb.update(LR);
        }
        if ((epoch + 1) % 5 == 0 || epoch == 0)
            std::cout << "  " << std::setw(3) << epoch + 1
                      << "     " << std::fixed << std::setprecision(4) << epoch_loss / steps << "\n";
    }

    generate(emb, net, cs, "the ", 200);
    generate(emb, net, cs, "in the", 200);
    return 0;
}
