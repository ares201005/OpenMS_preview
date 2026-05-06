#include <complex>
#include <Eigen/Dense>
#include <iostream>

#include "gafermion.hpp"

static std::vector<int> compute_Moff(std::vector<int> M) {
    std::vector<int> Moff(M.size() + 1);
    Moff[0] = 0;
    for (size_t i = 0; i < M.size(); i++) {
        Moff[i+1] = Moff[i] + M[i];
    }
    return Moff;
}

// static std::vector<MatrixType> get_annihilation_operators(int M) {

// }


// check zero copy
FermionGACPP::FermionGACPP(std::vector<int> M, int Ne_) : 
    Mvec(std::move(M)),
    N(static_cast<int>(Mvec.size())),
    Ne(Ne_),
    Moff_(compute_Moff(Mvec))
{}

void FermionGACPP::compute_gradient(const GradientInputView& input, GradientOutputView& output) {
    std::cout << "FermionGACPP::compute_gradient called" << std::endl;

    // placeholder: return all zeroes
    for (auto& gpsi : output.psiarr) {
        gpsi.eigen().setZero();
    }
    for (auto& gL : output.L) {
        gL.eigen().setZero();
    }
    for (auto& gLc : output.Lc) {
        gLc.eigen().setZero();
    }
    for (auto& gn: output.n) {
        gn.eigen().setZero();
    }
    output.Ec.eigen().setZero();
}