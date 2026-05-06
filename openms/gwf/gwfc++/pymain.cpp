#include <pybind11/numpy.h>
#include <pybind11/stl.h>     
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include "gafermion.hpp"

#include <iostream>

namespace py = pybind11;

int FermionGACPP::get_N() {
    return N;
}

int FermionGACPP::get_Ne() {
    return Ne;
}

std::vector<int> FermionGACPP::get_M() {
    return Mvec;
}

std::vector<int> FermionGACPP::get_Moff() {
    return Moff_;
}

template <typename T>
using ArrayIn = py::array_t<T, py::array::c_style>;

template <typename T>
using ArrayOut = py::array_t<T>;

template <typename T>
static ArrayIn<T> require_array(py::handle obj) {
    ArrayIn<T> arr = ArrayIn<T>::ensure(obj);
    if (!arr) {
        throw std::runtime_error("Expected C-contiguous array");
    }
    return arr;
}

template <typename T, int Flags>
static VecView<T> make_vec_view(py::array_t<T, Flags>& arr) {
    return VecView<T> (
        arr.mutable_data(),
        static_cast<Eigen::Index>(arr.shape(0))
    );
}

template <typename T, int Flags>
static MatView<T> make_mat_view(py::array_t<T, Flags>& arr) {
    return MatView<T> {
        arr.mutable_data(),
        static_cast<Eigen::Index>(arr.shape(0)),
        static_cast<Eigen::Index>(arr.shape(1))
    };
}

template <typename T, int Flags>
static TensorView<T> make_tensor_view(py::array_t<T, Flags>& arr) {
    return TensorView<T> {
        arr.mutable_data(),
        static_cast<Eigen::Index>(arr.shape(0)),
        static_cast<Eigen::Index>(arr.shape(1)),
        static_cast<Eigen::Index>(arr.shape(2)),
        static_cast<Eigen::Index>(arr.shape(3))
    };
}

static py::tuple py_compute_gradient(FermionGACPP& self,
        py::sequence psiarr,
        py::sequence L,
        py::sequence Lc,
        py::sequence n,
        py::object Ec_obj,
        py::sequence harr,
        py::object tblock_obj,
        py::sequence Uarr
) {
    const py::ssize_t N = self.get_N();

    // configure input format
    GradientInputView input;
    input.psiarr.reserve(N);
    input.L.reserve(N);
    input.Lc.reserve(N);
    input.n.reserve(N);
    input.harr.reserve(N);
    input.Uarr.reserve(N);

    // match with output format
    GradientOutputView output;
    output.psiarr.reserve(N);
    output.L.reserve(N);
    output.Lc.reserve(N);
    output.n.reserve(N);
    py::list grad_psiarr;
    py::list grad_L;
    py::list grad_Lc;
    py::list grad_n;
    
    // match format for input and output data
    for (py::ssize_t I = 0; I < N; I++) {
        auto psi = require_array<complex128>(psiarr[I]);
        input.psiarr.push_back(make_vec_view<complex128>(psi));
        ArrayOut<complex128> gpsi(psi.shape(0));
        output.psiarr.push_back(make_vec_view<complex128>(gpsi));
        grad_psiarr.append(gpsi);

        auto LM = require_array<complex128>(L[I]);
        input.L.push_back(make_mat_view<complex128>(LM));
        ArrayOut<complex128> gL({LM.shape(0), LM.shape(1)});
        output.L.push_back(make_mat_view<complex128>(gL));
        grad_L.append(gL);

        auto LcM = require_array<complex128>(Lc[I]);
        input.Lc.push_back(make_mat_view<complex128>(LcM));
        ArrayOut<complex128> gLc({LcM.shape(0), LcM.shape(1)});
        output.Lc.push_back(make_mat_view<complex128>(gLc));
        grad_Lc.append(gLc);
        
        auto nvec = require_array<double>(n[I]);
        input.n.push_back(make_vec_view(nvec));
        ArrayOut<double> gn(nvec.shape(0));
        output.n.push_back(make_vec_view<double>(gn));
        grad_n.append(gn);

        auto h = require_array<complex128>(harr[I]);
        input.harr.push_back(make_mat_view<complex128>(h));

        auto U = require_array<complex128>(Uarr[I]);
        input.Uarr.push_back(make_tensor_view<complex128>(U));
    }
    auto Ecvec = require_array<double>(Ec_obj);
    input.Ec = make_vec_view<double>(Ecvec);
    ArrayOut<double> grad_Ec(Ecvec.shape(0));
    output.Ec = make_vec_view(grad_Ec);

    auto tblock = require_array<complex128>(tblock_obj);
    input.tblock = make_mat_view<complex128>(tblock);

    self.compute_gradient(input, output);

    return py::make_tuple(
        grad_psiarr,
        grad_L,
        grad_Lc,
        grad_n,
        grad_Ec
    );
}

PYBIND11_MODULE(gafermion, m, py::mod_gil_not_used()) {
    py::class_<FermionGACPP>(m, "FermionGACPP")
        .def(py::init<std::vector<int>, int>())
        .def_property_readonly("N", &FermionGACPP::get_N)
        .def_property_readonly("Ne", &FermionGACPP::get_Ne)
        .def_property_readonly("M", &FermionGACPP::get_M)
        .def_property_readonly("_Moff", &FermionGACPP::get_Moff)
        .def("_compute_gradient", &py_compute_gradient);
}