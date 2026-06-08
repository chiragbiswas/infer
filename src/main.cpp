#include "net.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// 4 classes, each a cluster of points in 16-D space
static const int INPUT_DIM    = 16;
static const int NUM_CLASSES  = 4;
static const int DATASET_SIZE = 400;

static float X[DATASET_SIZE][INPUT_DIM];
static int   Y[DATASET_SIZE];

static void make_dataset() {
    srand(42);
    for (int c = 0; c < NUM_CLASSES; ++c)
        for (int n = 0; n < DATASET_SIZE / NUM_CLASSES; ++n) {
            int idx = c * (DATASET_SIZE / NUM_CLASSES) + n;
            Y[idx] = c;
            for (int d = 0; d < INPUT_DIM; ++d) {
                float noise = (float)rand() / (float)RAND_MAX * 0.5f;
                X[idx][d] = (d == c * 4 || d == c * 4 + 1) ? 2.f + noise : noise;
            }
        }
}

static float accuracy(MLP& net) {
    int correct = 0;
    for (int i = 0; i < DATASET_SIZE; ++i) {
        Tensor inp({1, INPUT_DIM});
        memcpy(inp.data, X[i], INPUT_DIM * sizeof(float));
        Tensor out = net.forward(inp);
        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; ++c)
            if (out.at(0, c) > out.at(0, pred)) pred = c;
        if (pred == Y[i]) ++correct;
    }
    return 100.f * correct / DATASET_SIZE;
}

int main() {
    make_dataset();

    MLP net({INPUT_DIM, 32, 32, NUM_CLASSES});

    srand(123);
    for (auto& l : net.layers) {
        float limit = sqrtf(6.f / (l.in_features + l.out_features));
        for (size_t i = 0; i < l.W.numel; ++i)
            l.W.data[i] = limit * (2.f * (float)rand() / (float)RAND_MAX - 1.f);
    }

    const int   EPOCHS = 50, BATCH_SIZE = 32, STEPS = DATASET_SIZE / BATCH_SIZE;
    const float LR = 0.01f;

    printf("Training  input=%d  classes=%d  batch=%d  lr=%.3f\n\n",
           INPUT_DIM, NUM_CLASSES, BATCH_SIZE, LR);
    printf("Epoch   Loss     Accuracy\n-----   ------   --------\n");

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.f;
        for (int step = 0; step < STEPS; ++step) {
            int offset = step * BATCH_SIZE;
            Tensor batch_x({BATCH_SIZE, INPUT_DIM});
            int    batch_y[BATCH_SIZE];
            for (int i = 0; i < BATCH_SIZE; ++i) {
                memcpy(&batch_x.at(i, 0), X[offset + i], INPUT_DIM * sizeof(float));
                batch_y[i] = Y[offset + i];
            }
            Tensor probs = net.forward(batch_x);
            epoch_loss += net.backward_and_update(probs, batch_y, LR);
        }
        if ((epoch + 1) % 5 == 0 || epoch == 0)
            printf("  %3d     %.4f   %.1f%%\n", epoch + 1, epoch_loss / STEPS, accuracy(net));
    }

    printf("\nDone.\n");
    return 0;
}
