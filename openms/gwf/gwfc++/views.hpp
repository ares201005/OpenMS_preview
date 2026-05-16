#ifndef VIEWS_HPP
#define VIEWS_HPP

#include <Eigen/Dense>
#include <complex>
#include <vector>

typedef std::complex<double> complex128;

template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using MatRM = Eigen::Matrix<
    T,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor
>;

template <typename T>
struct VecView {
    T* data;
    Eigen::Index size;

    VecView() = default;
    VecView(T* data_, Eigen::Index size_): data(data_), size(size_) {}

    Eigen::Map<Vec<T>> eigen() {
        return Eigen::Map<Vec<T>>(data, size);
    }

    Eigen::Map<Vec<T>> eigen() const {
        return Eigen::Map<Vec<T>>(data, size);
    }

    T& operator[](Eigen::Index i) {
        return data[i];
    }
};

template <typename T>
struct MatView {
    T* data = nullptr;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;

    MatView() = default;
    MatView(T* data_, Eigen::Index rows_, Eigen::Index cols_) : data(data_), rows(rows_), cols(cols_) {}

    Eigen::Map<MatRM<T>> eigen() {
        return Eigen::Map<MatRM<T>>(data, rows, cols);
    }

    Eigen::Map<MatRM<T>> eigen() const {
        return Eigen::Map<MatRM<T>>(data, rows, cols);
    }

    T& operator()(Eigen::Index i, Eigen::Index j) {
        return data[i*cols + j];
    }

    T& operator()(Eigen::Index i, Eigen::Index j) const {
        return data[i*cols + j];
    }
};

template <typename T>
struct TensorView {
    T* data = nullptr;
    Eigen::Index d0 = 0;
    Eigen::Index d1 = 0;
    Eigen::Index d2 = 0;
    Eigen::Index d3 = 0;

    TensorView() = default;
    TensorView(T* data_, Eigen::Index d0_, Eigen::Index d1_, Eigen::Index d2_, Eigen::Index d3_) :
        data(data_), d0(d0_), d1(d1_), d2(d2_), d3(d3_) {}

    T& operator()(Eigen::Index a, Eigen::Index b, Eigen::Index c, Eigen::Index d) {
        return data[((a*d1 + b)*d2 + c)*d3 + d];
    }

    T& operator()(Eigen::Index a, Eigen::Index b, Eigen::Index c, Eigen::Index d) const {
        return data[((a*d1 + b)*d2 + c)*d3 + d];
    }
};

struct InputView {
    std::vector<MatView<complex128>> psiarr;
    std::vector<MatView<complex128>> L;
    std::vector<MatView<complex128>> Lc;
    std::vector<VecView<double>> n;
    VecView<double> Ec;

    std::vector<MatView<complex128>> harr;
    MatView<complex128> tblock;
    std::vector<TensorView<complex128>> Uarr;
};

struct GradientOutput {
    std::vector<MatRM<complex128>> psiarr;
    std::vector<MatRM<complex128>> L;
    std::vector<MatRM<complex128>> Lc;
    std::vector<Vec<double>> n;
    std::vector<double> Ec;

    GradientOutput() = default;

    explicit GradientOutput(std::size_t N)
        : psiarr(N),
          L(N),
          Lc(N),
          n(N),
          Ec(N)
    {}
};

#endif