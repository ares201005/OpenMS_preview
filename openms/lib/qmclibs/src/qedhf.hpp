#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

// function for qedhf

namespace py = pybind11;


/**
 * Compute the displacement matrix and its contraction with a photon density matrix.
 *
 * @param nboson_states: vector of boson dimensions per mode
 * @param mode: the active boson mode index
 * @param imode: index to use for freq/vsq arrays
 * @param freq: 1D NumPy array of mode frequencies
 * @param eta: 2D NumPy array [nmode x nao]
 * @param pdm: 2D NumPy array [mdim x mdim], symmetric
 * @param vsq: 1D NumPy array of squeezing amplitudes
 * @param shift: scalar real displacement
 * @return tuple (disp_mat [packed_dim x nao x nao], exp_val [nao x nao])
 */
std::pair<py::array_t<double>, py::array_t<double>> displacement_matrix_cpp(
    const std::vector<int>& nboson_states,
    int imode,
    py::array_t<double, py::array::c_style | py::array::forcecast> freq,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> pdm,
    py::array_t<double, py::array::c_style | py::array::forcecast> vsq,
    double shift = 0.0
);

// optimized Hartree-Fock J/K builder for single photonic mode
// returns tuple ``(vj, vk)`` with both arrays real-valued
std::pair<py::array_t<double>, py::array_t<double>> get_JK_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> ltensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> dm_do,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta_imode,
    double tau,
    double omega,
    int mdim,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> pdm
);


/*

py::array_t<double> displacement_val(
    int imode,
    py::array_t<int, py::array::c_style | py::array::forcecast> nboson_states,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> brho,
    float shift,
    );


py::array_t<double> displacement_deriv_kernel(
    int imode,
    py::array_t<int, py::array::c_style | py::array::forcecast> nboson_states,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> brho,
    float shift,
    );
*/
