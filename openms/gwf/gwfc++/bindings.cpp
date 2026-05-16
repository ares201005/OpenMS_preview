#include <pybind11/numpy.h>
#include <pybind11/stl.h>     
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include "gafermion.hpp"

namespace py = pybind11;

int FermionGACPP::get_N() const {
    return N;
}

int FermionGACPP::get_Ne() const {
    return Ne;
}

std::vector<int> FermionGACPP::get_M() const {
    return Mvec;
}

std::vector<int> FermionGACPP::get_Moff() const {
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

static py::tuple py_compute_initial_guess(FermionGACPP& self,
        py::sequence psiarr,
        py::sequence L,
        py::sequence n,
        py::object corr_obj,
        py::sequence harr,
        py::object tblock_obj,
        py::sequence Uarr,
        int maxiter,
        double tol
) {
    const py::ssize_t N = self.get_N();

    // configure input format
    InputView input;
    input.psiarr.reserve(N);
    input.L.reserve(N);
    input.n.reserve(N);
    input.harr.reserve(N);
    input.Uarr.reserve(N);

    // match with output format
    InitialGuessResult result;
    
    // match format for input and output data
    for (py::ssize_t I = 0; I < N; I++) {
        auto psi = require_array<complex128>(psiarr[I]);
        input.psiarr.push_back(make_mat_view<complex128>(psi));

        auto LM = require_array<complex128>(L[I]);
        input.L.push_back(make_mat_view<complex128>(LM));
        
        auto nvec = require_array<double>(n[I]);
        input.n.push_back(make_vec_view(nvec));

        auto h = require_array<complex128>(harr[I]);
        input.harr.push_back(make_mat_view<complex128>(h));

        auto U = require_array<complex128>(Uarr[I]);
        input.Uarr.push_back(make_tensor_view<complex128>(U));
    }

    auto tblock = require_array<complex128>(tblock_obj);
    input.tblock = make_mat_view<complex128>(tblock);

    auto corr = require_array<complex128>(corr_obj);
    auto corr_view = make_mat_view<complex128>(corr);
    
    {
        py::gil_scoped_release release;
        result = self.compute_initial_guess(input, corr_view, maxiter, tol);
    }

    return py::make_tuple(
        result.psiarr,
        result.Lc,
        result.Ec
    );
}

static double py_compute_lagrangian(FermionGACPP& self,
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
    InputView input;
    input.psiarr.reserve(N);
    input.L.reserve(N);
    input.Lc.reserve(N);
    input.n.reserve(N);
    input.harr.reserve(N);
    input.Uarr.reserve(N);

    // match with output format
    GradientOutput output(N);
    
    // match format for input and output data
    for (py::ssize_t I = 0; I < N; I++) {
        auto psi = require_array<complex128>(psiarr[I]);
        input.psiarr.push_back(make_mat_view<complex128>(psi));

        auto LM = require_array<complex128>(L[I]);
        input.L.push_back(make_mat_view<complex128>(LM));

        auto LcM = require_array<complex128>(Lc[I]);
        input.Lc.push_back(make_mat_view<complex128>(LcM));
        
        auto nvec = require_array<double>(n[I]);
        input.n.push_back(make_vec_view(nvec));

        auto h = require_array<complex128>(harr[I]);
        input.harr.push_back(make_mat_view<complex128>(h));

        auto U = require_array<complex128>(Uarr[I]);
        input.Uarr.push_back(make_tensor_view<complex128>(U));
    }
    auto Ecvec = require_array<double>(Ec_obj);
    input.Ec = make_vec_view<double>(Ecvec);

    auto tblock = require_array<complex128>(tblock_obj);
    input.tblock = make_mat_view<complex128>(tblock);

    py::gil_scoped_release release;
    return self.compute_lagrangian(input);
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
    InputView input;
    input.psiarr.reserve(N);
    input.L.reserve(N);
    input.Lc.reserve(N);
    input.n.reserve(N);
    input.harr.reserve(N);
    input.Uarr.reserve(N);

    // match with output format
    GradientOutput output(N);
    
    // match format for input and output data
    for (py::ssize_t I = 0; I < N; I++) {
        auto psi = require_array<complex128>(psiarr[I]);
        input.psiarr.push_back(make_mat_view<complex128>(psi));

        auto LM = require_array<complex128>(L[I]);
        input.L.push_back(make_mat_view<complex128>(LM));

        auto LcM = require_array<complex128>(Lc[I]);
        input.Lc.push_back(make_mat_view<complex128>(LcM));
        
        auto nvec = require_array<double>(n[I]);
        input.n.push_back(make_vec_view(nvec));

        auto h = require_array<complex128>(harr[I]);
        input.harr.push_back(make_mat_view<complex128>(h));

        auto U = require_array<complex128>(Uarr[I]);
        input.Uarr.push_back(make_tensor_view<complex128>(U));
    }
    auto Ecvec = require_array<double>(Ec_obj);
    input.Ec = make_vec_view<double>(Ecvec);

    auto tblock = require_array<complex128>(tblock_obj);
    input.tblock = make_mat_view<complex128>(tblock);

    {
        py::gil_scoped_release release;
        self.compute_gradient(input, output);
    }

    return py::make_tuple(
        output.psiarr,
        output.L,
        output.Lc,
        output.n,
        output.Ec
    );
}

static MatRM<complex128> py_compute_1body_corr(FermionGACPP& self,
        py::sequence ppsiarr,
        py::sequence pL,
        py::sequence pn,
        py::object tblock_obj
) {
    const py::ssize_t N = self.get_N();

    // configure input format
    std::vector<MatView<complex128>> psiarr;
    std::vector<MatView<complex128>> L;
    std::vector<VecView<double>> n;
    
    // match format for input and output data
    for (py::ssize_t I = 0; I < N; I++) {
        auto psi = require_array<complex128>(ppsiarr[I]);
        psiarr.push_back(make_mat_view<complex128>(psi));

        auto LM = require_array<complex128>(pL[I]);
        L.push_back(make_mat_view<complex128>(LM));
        
        auto nvec = require_array<double>(pn[I]);
        n.push_back(make_vec_view(nvec));
    
    }

    auto tblock = require_array<complex128>(tblock_obj);
    auto tblock_view = make_mat_view<complex128>(tblock);

    py::gil_scoped_release release;
    return self.compute_1body_corr(psiarr, L, n, tblock_view);
}

struct CdictAccessor {
    FermionGACPP* parent;
    std::vector<MatRM<complex128>> getitem(int M) const {
        return parent->return_Cdict(M);
    }
};

PYBIND11_MODULE(_gafermion, m) {
    m.def("_get_annihilation_operators", &get_annihilation_operators);

    py::class_<CdictAccessor>(m, "_CdictAccessor")
        .def("__getitem__", &CdictAccessor::getitem);

    py::class_<FermionGACPP>(m, "FermionGACPP")
        .def(py::init<std::vector<int>, int>())
        .def_property_readonly("N", &FermionGACPP::get_N)
        .def_property_readonly("Ne", &FermionGACPP::get_Ne)
        .def_property_readonly("M", &FermionGACPP::get_M)
        .def_property_readonly("_Moff", &FermionGACPP::get_Moff)
        .def_property_readonly(
            "_Cdict",
            [](FermionGACPP& self) {
                return CdictAccessor{&self};
            }
        )

        .def("_get_initial_guess", &py_compute_initial_guess)
        .def("_compute_lagrangian", &py_compute_lagrangian)
        .def("_compute_gradient", &py_compute_gradient)
        .def("_compute_1body_correlations", &py_compute_1body_corr);
}