#ifndef GAFERMION_HPP
#define GAFERMION_HPP

#include <map>
#include "views.hpp"

class FermionGACPP {
public:
    FermionGACPP(std::vector<int> M, int Ne);
    int get_N();
    int get_Ne();
    std::vector<int> get_M();
    std::vector<int> get_Moff();

    void compute_gradient(const GradientInputView& input, GradientOutputView& output);

private:
    const std::vector<int> Mvec;
    const int N;
    const int Ne;
    const std::vector<int> Moff_;

    std::map<int, std::vector<MatRM<int>>> get_Cdict();
    const std::map<int, std::vector<MatRM<int>>> Cdict_;
};

// use psimatrix instead of psivec if HK is unnecessary

#endif