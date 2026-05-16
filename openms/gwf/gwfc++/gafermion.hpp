#ifndef GAFERMION_HPP
#define GAFERMION_HPP

#include <map>
#include "views.hpp"

struct InitialGuessResult {
    std::vector<MatRM<complex128>> psiarr;
    std::vector<MatRM<complex128>> Lc;
    Vec<double> Ec;
};

class FermionGACPP {
public:
    FermionGACPP(std::vector<int> M, int Ne);
    int get_N() const;
    int get_Ne() const;
    std::vector<int> get_M() const;
    std::vector<int> get_Moff() const;

    InitialGuessResult compute_initial_guess(
        const InputView& input,
        const MatView<complex128>& corr_view,
        int max_iter,
        double tol
    ) const;

    std::vector<MatRM<complex128>> compute_renormalizations(
        const std::vector<MatView<complex128>>& psiarr,
        const std::vector<VecView<double>>& n
    ) const;
    void compute_gradient(
        const InputView& input,
        GradientOutput& output
    ) const;

private:
    const std::vector<int> Mvec;
    const int N;
    const int Ne;
    const std::vector<int> Moff_;

    std::map<int, std::vector<MatRM<complex128>>> get_Cdict() const;
    const std::map<int, std::vector<MatRM<complex128>>> Cdict_;

    MatRM<complex128> compute_qp_corr(
        const std::vector<MatView<complex128>>& L,
        const std::vector<MatRM<complex128>>& R,
        const MatView<complex128>& tblock
    ) const;

    MatRM<complex128> compute_Hloc(
        const MatView<complex128>& h,
        const TensorView<complex128>& U
    ) const;

    MatRM<complex128> compute_G(
        int K,
        const MatView<complex128>& tblock,
        const std::vector<MatRM<complex128>>& R,
        Eigen::Ref<const MatRM<complex128>> corr
    ) const;

};

#endif