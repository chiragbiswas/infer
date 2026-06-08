#pragma once
#include "ops.h"
#include <vector>

// One fully-connected layer. Saves input and pre-relu output for backprop.
struct Linear {
    Tensor W, b;
    Tensor act, inp; // saved during forward
    int in_features, out_features;

    Linear(int in, int out)
        : W({in, out}), b({out}), act({1, out}), inp({1, in}),
          in_features(in), out_features(out) {
        W.alloc_grad(); b.alloc_grad();
    }

    void forward(const Tensor& src, Tensor& dst) {
        // Resize saved buffers if batch changed
        if (inp.rows() != src.rows()) {
            int d[2];
            d[0] = src.rows(); d[1] = in_features;  inp.~Tensor(); new(&inp) Tensor(d,2);
            d[1] = out_features;                     act.~Tensor(); new(&act) Tensor(d,2);
        }
        memcpy(inp.data, src.data, src.numel * sizeof(float));
        matmul(src, W, dst);
        add_bias(dst, b);
        memcpy(act.data, dst.data, dst.numel * sizeof(float));
    }

    void zero_grad() { W.zero_grad(); b.zero_grad(); }
};

// Stack of Linear layers with ReLU between them, Softmax at the end.
struct MLP {
    std::vector<Linear> layers;

    MLP(std::initializer_list<int> sizes) {
        auto it = sizes.begin(); int prev = *it++;
        while (it != sizes.end()) { int cur = *it++; layers.emplace_back(prev, cur); prev = cur; }
    }

    explicit MLP(const std::vector<int>& sizes) {
        for (size_t i = 0; i + 1 < sizes.size(); ++i)
            layers.emplace_back(sizes[i], sizes[i+1]);
    }

    Tensor forward(const Tensor& input) {
        int batch = input.rows();
        Tensor buf_a({batch, layers[0].out_features});
        layers[0].forward(input, buf_a);
        if (layers.size() == 1) { softmax(buf_a); return buf_a; }
        relu(buf_a);
        Tensor buf_b({batch, layers[1].out_features});
        for (size_t i = 1; i < layers.size(); ++i) {
            Tensor& src = (i % 2 == 1) ? buf_a : buf_b;
            Tensor& dst = (i % 2 == 1) ? buf_b : buf_a;
            if (dst.cols() != layers[i].out_features) {
                int d[2] = {batch, layers[i].out_features};
                dst.~Tensor(); new(&dst) Tensor(d, 2);
            }
            layers[i].forward(src, dst);
            if (i + 1 < layers.size()) relu(dst);
        }
        Tensor& out = (layers.size() % 2 == 0) ? buf_b : buf_a;
        softmax(out);
        return std::move(out);
    }

    // Backprop + SGD. Returns loss.
    float backward_and_update(Tensor& probs, const int* labels, float lr) {
        const int batch = probs.rows(), n = (int)layers.size();
        std::vector<Tensor> grads;
        grads.reserve(n + 1);
        for (auto& l : layers) { int d[2] = {batch, l.out_features}; grads.emplace_back(d, 2); }
        { int d[2] = {batch, layers[0].in_features}; grads.emplace_back(d, 2); }
        for (auto& l : layers) l.zero_grad();

        float loss = cross_entropy_backward(probs, labels, grads[n - 1]);

        for (int i = n - 1; i >= 0; --i) {
            if (i < n - 1) {
                layers[i].act.alloc_grad();
                relu_backward(layers[i].act, grads[i].data);
                memcpy(grads[i].data, layers[i].act.grad, grads[i].numel * sizeof(float));
            }
            linear_backward(grads[i], layers[i].inp, layers[i].W, layers[i].b, grads[i+1]);
        }

        for (auto& l : layers) {
            for (size_t i = 0; i < l.W.numel; ++i) l.W.data[i] -= lr * l.W.grad[i];
            for (size_t i = 0; i < l.b.numel; ++i) l.b.data[i] -= lr * l.b.grad[i];
        }
        return loss;
    }

    // Same as above but also returns gradient w.r.t. input (for embedding backprop)
    float backward_and_update(Tensor& probs, const int* labels, float lr, Tensor& grad_input) {
        const int batch = probs.rows(), n = (int)layers.size();
        std::vector<Tensor> grads;
        grads.reserve(n + 1);
        for (auto& l : layers) { int d[2] = {batch, l.out_features}; grads.emplace_back(d, 2); }
        { int d[2] = {batch, layers[0].in_features}; grads.emplace_back(d, 2); }
        for (auto& l : layers) l.zero_grad();

        float loss = cross_entropy_backward(probs, labels, grads[n - 1]);

        for (int i = n - 1; i >= 0; --i) {
            if (i < n - 1) {
                layers[i].act.alloc_grad();
                relu_backward(layers[i].act, grads[i].data);
                memcpy(grads[i].data, layers[i].act.grad, grads[i].numel * sizeof(float));
            }
            linear_backward(grads[i], layers[i].inp, layers[i].W, layers[i].b, grads[i+1]);
        }

        memcpy(grad_input.data, grads[n].data, grads[n].numel * sizeof(float));

        for (auto& l : layers) {
            for (size_t i = 0; i < l.W.numel; ++i) l.W.data[i] -= lr * l.W.grad[i];
            for (size_t i = 0; i < l.b.numel; ++i) l.b.data[i] -= lr * l.b.grad[i];
        }
        return loss;
    }
};
