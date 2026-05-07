#include <complex>
#include <Eigen/Dense>
#include <unordered_map>
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

static std::vector<MatRM<int>> get_annihilation_operators(int M) {
    std::vector<MatRM<int>> ops;
    ops.reserve(M);
    const int dim = 1 << M;
    for (int i = 0; i < M; i++) {
        MatRM<int> op(dim, dim);
        op.setZero();
        for (int state = 0; state < dim; state++) {
            if (state & (1 << i)) {
                const int parity = __builtin_popcount(state >> (i+1));
                op(state & ~(1 << i), state) = (parity % 2 == 0) ? +1 : -1;
            }
        }
        ops.push_back(std::move(op));
    }
    return ops;
}

std::map<int, std::vector<MatRM<int>>> FermionGACPP::get_Cdict() {
    std::unordered_map<int, int> counts;
    for (int x : Mvec) {
        counts[x]++;
    }
    std::map<int, std::vector<MatRM<int>>> Cdict;
    for (auto const& item : counts) {
        const int M = item.first;
        Cdict.emplace(M, get_annihilation_operators(M));
    }
    return Cdict;
}

FermionGACPP::FermionGACPP(std::vector<int> M, int Ne_) : 
    Mvec(std::move(M)),
    N(static_cast<int>(Mvec.size())),
    Ne(Ne_),
    Moff_(compute_Moff(Mvec)),
    Cdict_(get_Cdict())
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