#pragma once
#include <cstddef>
#include <cstring>
#include <cassert>
#include <initializer_list>
#include <memory>

// Flat float array + shape metadata. grad is allocated on demand.
struct Tensor {
    float*  data   = nullptr;
    float*  grad   = nullptr;
    int     shape[4] = {};
    int     ndim   = 0;
    size_t  numel  = 0;

    Tensor() = default;
    explicit Tensor(std::initializer_list<int> dims) { init(dims.begin(), (int)dims.size()); }
    Tensor(const int* dims, int nd) { init(dims, nd); }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept
        : data(o.data), grad(o.grad), ndim(o.ndim), numel(o.numel) {
        memcpy(shape, o.shape, sizeof(shape));
        o.data = nullptr; o.grad = nullptr; o.numel = 0; o.ndim = 0;
    }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            free(data); free(grad);
            data = o.data; grad = o.grad;
            ndim = o.ndim; numel = o.numel;
            memcpy(shape, o.shape, sizeof(shape));
            o.data = nullptr; o.grad = nullptr; o.numel = 0; o.ndim = 0;
        }
        return *this;
    }

    ~Tensor() { free(data); free(grad); }

    float& at(int r, int c)             { return data[r * shape[1] + c]; }
    const float& at(int r, int c) const { return data[r * shape[1] + c]; }
    float& gat(int r, int c)            { return grad[r * shape[1] + c]; }
    float& operator[](size_t i)         { return data[i]; }
    float  operator[](size_t i) const   { return data[i]; }

    int rows() const { assert(ndim >= 1); return shape[0]; }
    int cols() const { assert(ndim >= 2); return shape[1]; }

    void zero()      { memset(data, 0, numel * sizeof(float)); }
    void zero_grad() { if (grad) memset(grad, 0, numel * sizeof(float)); }

    void alloc_grad() {
        if (!grad) {
            size_t bytes = ((numel * sizeof(float) + 63) / 64) * 64;
            grad = (float*)aligned_alloc(64, bytes);
            assert(grad);
        }
        zero_grad();
    }

private:
    void init(const int* dims, int nd) {
        ndim = nd; numel = 1;
        for (int i = 0; i < nd; ++i) { shape[i] = dims[i]; numel *= dims[i]; }
        size_t bytes = ((numel * sizeof(float) + 63) / 64) * 64;
        data = (float*)aligned_alloc(64, bytes);
        assert(data);
        zero();
    }
    void init_shape(const int* dims, int nd) {
        ndim = nd; numel = 1;
        for (int i = 0; i < nd; ++i) { shape[i] = dims[i]; numel *= dims[i]; }
    }
};
