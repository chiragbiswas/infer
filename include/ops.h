#pragma once
#include "tensor.h"
#include <cmath>
#include <thread>
#include <vector>

// NEON on ARM64, AVX2 on x86
#if defined(__aarch64__) || defined(__arm__)
  #include <arm_neon.h>
  #define USE_NEON 1
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define USE_AVX2 1
#endif

// Compute one row slice of C = A @ B (called per thread)
inline void matmul_range(const Tensor& A, const Tensor& B, Tensor& C,
                         int row_start, int row_end) {
    const int K = A.cols(), N = B.cols();
    for (int i = row_start; i < row_end; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a_ik = A.at(i, k);
            int j = 0;
#if USE_NEON
            float32x4_t va = vdupq_n_f32(a_ik);
            for (; j <= N - 4; j += 4) {
                float32x4_t vb = vld1q_f32(&B.at(k, j));
                float32x4_t vc = vld1q_f32(&C.at(i, j));
                vst1q_f32(&C.at(i, j), vmlaq_f32(vc, va, vb));
            }
#elif USE_AVX2
            __m256 va256 = _mm256_set1_ps(a_ik);
            for (; j <= N - 8; j += 8) {
                __m256 vb = _mm256_loadu_ps(&B.at(k, j));
                __m256 vc = _mm256_loadu_ps(&C.at(i, j));
                _mm256_storeu_ps(&C.at(i, j), _mm256_fmadd_ps(va256, vb, vc));
            }
#endif
            for (; j < N; ++j)
                C.at(i, j) += a_ik * B.at(k, j);
        }
    }
}

// C = A @ B, split across threads
static const int NUM_THREADS = 4;

inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = A.rows(), K = A.cols(), N = B.cols();
    assert(B.rows() == K);
    assert(C.rows() == M && C.cols() == N);
    C.zero();

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    int chunk = (M + NUM_THREADS - 1) / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int rs = t * chunk, re = std::min(rs + chunk, M);
        if (rs >= M) break;
        threads.emplace_back([&A, &B, &C, rs, re]() { matmul_range(A, B, C, rs, re); });
    }
    for (auto& t : threads) t.join();
}

// out[i][j] += bias[j]
inline void add_bias(Tensor& out, const Tensor& bias) {
    const int M = out.rows(), N = out.cols();
    assert((int)bias.numel == N);
    for (int i = 0; i < M; ++i) {
        int j = 0;
#if USE_NEON
        for (; j <= N - 4; j += 4)
            vst1q_f32(&out.at(i, j), vaddq_f32(vld1q_f32(&out.at(i, j)), vld1q_f32(&bias.data[j])));
#elif USE_AVX2
        for (; j <= N - 8; j += 8)
            _mm256_storeu_ps(&out.at(i, j), _mm256_add_ps(_mm256_loadu_ps(&out.at(i, j)), _mm256_loadu_ps(&bias.data[j])));
#endif
        for (; j < N; ++j) out.at(i, j) += bias.data[j];
    }
}

// max(0, x) in-place
inline void relu(Tensor& t) {
    size_t i = 0;
#if USE_NEON
    float32x4_t zero = vdupq_n_f32(0.f);
    for (; i + 4 <= t.numel; i += 4)
        vst1q_f32(&t.data[i], vmaxq_f32(vld1q_f32(&t.data[i]), zero));
#elif USE_AVX2
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= t.numel; i += 8)
        _mm256_storeu_ps(&t.data[i], _mm256_max_ps(_mm256_loadu_ps(&t.data[i]), zero));
#endif
    for (; i < t.numel; ++i) t.data[i] = t.data[i] > 0.f ? t.data[i] : 0.f;
}

// relu backward: grad passes through where input > 0
inline void relu_backward(Tensor& act, const float* grad_out) {
    for (size_t i = 0; i < act.numel; ++i)
        act.grad[i] = act.data[i] > 0.f ? grad_out[i] : 0.f;
}

// backward for y = x @ W + b
// accumulates grad_W, grad_b; writes grad_x
inline void linear_backward(const Tensor& grad_out, const Tensor& x,
                            Tensor& W, Tensor& b, Tensor& grad_x) {
    const int batch = grad_out.rows(), out_f = grad_out.cols(), in_f = x.cols();

    // grad_W += x.T @ grad_out
    for (int i = 0; i < in_f; ++i)
        for (int k = 0; k < batch; ++k) {
            float x_ki = x.at(k, i);
            for (int j = 0; j < out_f; ++j)
                W.grad[i * out_f + j] += x_ki * grad_out.at(k, j);
        }

    // grad_b += sum(grad_out, axis=0)
    for (int k = 0; k < batch; ++k)
        for (int j = 0; j < out_f; ++j)
            b.grad[j] += grad_out.at(k, j);

    // grad_x = grad_out @ W.T
    grad_x.zero();
    for (int k = 0; k < batch; ++k)
        for (int j = 0; j < out_f; ++j) {
            float g = grad_out.at(k, j);
            for (int i = 0; i < in_f; ++i)
                grad_x.at(k, i) += g * W.at(i, j);
        }
}

// Combined softmax + cross-entropy backward.
// grad = probs - one_hot(label), returns mean loss.
inline float cross_entropy_backward(const Tensor& probs, const int* labels, Tensor& grad) {
    const int batch = probs.rows(), C = probs.cols();
    float loss = 0.f;
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < C; ++j) grad.at(i, j) = probs.at(i, j);
        grad.at(i, labels[i]) -= 1.f;
        loss -= logf(probs.at(i, labels[i]) + 1e-9f);
    }
    float inv = 1.f / batch;
    for (size_t i = 0; i < grad.numel; ++i) grad.data[i] *= inv;
    return loss / batch;
}

// per-row stable softmax, in-place
inline void softmax(Tensor& t) {
    const int M = t.rows(), N = t.cols();
    for (int i = 0; i < M; ++i) {
        float* row = &t.at(i, 0);
        float mx = row[0];
        for (int j = 1; j < N; ++j) if (row[j] > mx) mx = row[j];
        float sum = 0.f;
        for (int j = 0; j < N; ++j) { row[j] = expf(row[j] - mx); sum += row[j]; }
        float inv = 1.f / sum;
        int j = 0;
#if USE_NEON
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; j <= N - 4; j += 4)
            vst1q_f32(&row[j], vmulq_f32(vld1q_f32(&row[j]), vinv));
#elif USE_AVX2
        __m256 vinv256 = _mm256_set1_ps(inv);
        for (; j <= N - 8; j += 8)
            _mm256_storeu_ps(&row[j], _mm256_mul_ps(_mm256_loadu_ps(&row[j]), vinv256));
#endif
        for (; j < N; ++j) row[j] *= inv;
    }
}
