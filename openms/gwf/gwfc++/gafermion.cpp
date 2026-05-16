#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unordered_map>
#include <omp.h>

#include "gafermion.hpp"

static std::vector<int> compute_Moff(std::vector<int> M) {
    std::vector<int> Moff(M.size() + 1);
    Moff[0] = 0;
    for (size_t i = 0; i < M.size(); i++) {
        Moff[i+1] = Moff[i] + M[i];
    }
    return Moff;
}

std::vector<MatRM<complex128>> get_annihilation_operators(int M) {
    std::vector<MatRM<complex128>> ops;
    ops.reserve(M);
    const int dim = 1 << M;
    for (int i = 0; i < M; i++) {
        MatRM<complex128> op(dim, dim);
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

static complex128 matrix_ip(const MatRM<complex128>& A, const MatRM<complex128>& B) {
    return (A.conjugate().array() * B.array()).sum();
}

template <typename T>
static MatRM<T> kron(const MatRM<T>& A, const MatRM<T>& B) {
    return Eigen::kroneckerProduct(A, B).eval();
}

std::map<int, std::vector<MatRM<complex128>>> FermionGACPP::get_Cdict() const {
    std::unordered_map<int, int> counts;
    for (int x : Mvec) {
        counts[x]++;
    }
    std::map<int, std::vector<MatRM<complex128>>> Cdict;
    for (auto const& item : counts) {
        const int M = item.first;
        Cdict.emplace(M, get_annihilation_operators(M));
    }
    return Cdict;
}

static void check_eigen_status(Eigen::ComputationInfo info, std::string identifier) {
    if (info != Eigen::Success) {
        std::string msg = identifier + " diagonalization failed: ";
        if (info == Eigen::NumericalIssue) {
            throw std::runtime_error(msg + "Numerical issue (e.g., singular matrix)!");
        } else if (info == Eigen::InvalidInput) {
            throw std::runtime_error(msg + "Invalid input!");
        } else if (info == Eigen::NoConvergence) {
            throw std::runtime_error(msg + "Did not converge!");
        }
    }
}

std::map<int, std::vector<MatRM<complex128>>> FermionGACPP::return_Cdict() const {
    return Cdict_;
}

std::vector<MatRM<complex128>> FermionGACPP::return_Cdict(int M) const {
    return Cdict_.at(M);
}

FermionGACPP::FermionGACPP(std::vector<int> M, int Ne_) : 
    Mvec(std::move(M)),
    N(static_cast<int>(Mvec.size())),
    Ne(Ne_),
    Moff_(compute_Moff(Mvec)),
    Cdict_(get_Cdict())
{}

std::vector<MatRM<complex128>> FermionGACPP::compute_renormalizations(
    const std::vector<MatView<complex128>>& psiarr,
    const std::vector<VecView<double>>& n
) const {
    std::vector<MatRM<complex128>> R(N);
    Eigen::setNbThreads(1);
    #pragma omp parallel for schedule(static)
    for (int I = 0; I < N; I++) {
        const int M = Mvec[I];
        const auto& C = Cdict_.at(M);
        auto psi = psiarr[I].eigen();
        auto nI = n[I];
        MatRM<complex128> RI(M, M);
        for (int a = 0; a < M; a++) {
            double nIa = nI[a];
            for (int alpha = 0; alpha < M; alpha++) {
                MatRM<complex128> op = C[alpha].transpose() * psi * C[a];
                RI(alpha, a) = matrix_ip(psi, op) / std::sqrt(nIa * (1.0 - nIa));
            }
        }
        R[I] = std::move(RI);
    }
    return R;
}

MatRM<complex128> FermionGACPP::compute_Hqp(
    const std::vector<MatView<complex128>>& L,
    const std::vector<MatRM<complex128>>& R,
    const MatView<complex128>& tblock
) const {
    const int Mtot = Moff_.back();
    MatRM<complex128> Hqp = MatRM<complex128>::Zero(Mtot, Mtot);
    auto t = tblock.eigen();

    // construct QP Hamiltonian
    Eigen::setNbThreads(1);
    #pragma omp parallel for schedule(static)
    for (int I = 0; I < N; I++) {
        const int i0 = Moff_[I];
        const int Mi = Mvec[I];
        for (int J = 0; J < N; J++) {
            const int j0 = Moff_[J];
            const int Mj = Mvec[J];
            auto tIJ = t.block(i0, j0, Mi, Mj);
            Hqp.block(i0, j0, Mi, Mj).noalias() = R[I].transpose() * tIJ * R[J].conjugate();
        }
    }
    #pragma omp parallel for schedule(static)
    for (int I = 0; I < N; I++) {
        const int i0 = Moff_[I];
        const int Mi = Mvec[I];
        auto LI = L[I].eigen();
        Hqp.block(i0, i0, Mi, Mi).noalias() += LI + LI.adjoint();
    }

    return Hqp;
}

MatRM<complex128> FermionGACPP::compute_qp_corr(
    const std::vector<MatView<complex128>>& L,
    const std::vector<MatRM<complex128>>& R,
    const MatView<complex128>& tblock
) const {
    // diagonalize and compute 1-body QP correlation matrix
    auto Hqp = compute_Hqp(L, R, tblock);
    Eigen::setNbThreads(omp_get_max_threads());
    Eigen::SelfAdjointEigenSolver<MatRM<complex128>> solver(Hqp);
    check_eigen_status(solver.info(), "Hqp");
    const auto& V = solver.eigenvectors();
    auto Vocc = V.leftCols(Ne);
    MatRM<complex128> corr = Vocc.conjugate() * Vocc.transpose();
    return corr;
}

MatRM<complex128> FermionGACPP::compute_1body_corr(
    const std::vector<MatView<complex128>>& psiarr,
    const std::vector<MatView<complex128>>& L,
    const std::vector<VecView<double>>& n,
    const MatView<complex128>& tblock
) const {
    auto R = compute_renormalizations(psiarr, n);
    auto corr = compute_qp_corr(L, R, tblock);
    const int size = Moff_.back();
    MatRM<complex128> expcorr(size, size);

    Eigen::setNbThreads(1);
    #pragma omp parallel for schedule(static)
    for (int I = 0; I < N; I++) {
        const int i0 = Moff_[I];
        const int Mi = Mvec[I];
        auto& C = Cdict_.at(Mi);
        auto psi = psiarr[I].eigen();
        for (int J = 0; J < N; J++) {
            if (I != J) {
                const int j0 = Moff_[J];
                const int Mj = Mvec[J];
                auto corrIJ = corr.block(i0, j0, Mi, Mj);
                expcorr.block(i0, j0, Mi, Mj).noalias() = R[I] * corrIJ * R[J].adjoint();
            }
            else {
                for (int a = 0; a < Mi; a++) {
                    for (int b = 0; b < Mi; b++) {
                        MatRM<complex128> op = C[a].transpose() * C[b] * psi;
                        expcorr(i0 + a, i0 + b) = matrix_ip(psi, op);
                    }
                }
            }
        }
    }

    return expcorr;
}

MatRM<complex128> FermionGACPP::compute_Hloc(
    const MatView<complex128>& h,
    const TensorView<complex128>& U
) const {
    const int M = h.eigen().rows();
    const int dim = 1 << M;
    const auto& C = Cdict_.at(M);
    MatRM<complex128> Hloc = MatRM<complex128>::Zero(dim, dim);
    for (int a = 0; a < M; a++) {
        for (int b = 0; b < M; b++) {
            Hloc += h(a,b)*C[a].transpose()*C[b];
            for (int c = 0; c < M; c++) {
                for (int d = 0; d < M; d++) {
                    Hloc += U(a,b,c,d)*C[a].transpose()*C[b].transpose()*C[c]*C[d];
                }
            }
        }
    }
    return Hloc;
}

MatRM<complex128> FermionGACPP::compute_G(
    int K,
    const MatView<complex128>& tblock,
    const std::vector<MatRM<complex128>>& R,
    Eigen::Ref<const MatRM<complex128>> corr
) const {
    const int k0 = Moff_[K];
    const int Mk = Mvec[K];
    MatRM<complex128> GK = MatRM<complex128>::Zero(Mk, Mk);
    auto t = tblock.eigen();
    for (int I = 0; I < N; I++) {
        const int i0 = Moff_[I];
        const int Mi = Mvec[I];
        auto tIK = t.block(i0, k0, Mi, Mk);
        auto DeltaIK = corr.block(i0, k0, Mi, Mk);
        MatRM<complex128> temp = tIK.transpose() * R[I];
        GK.noalias() += temp * DeltaIK;
    }
    return GK;
}

template <typename T>
static MatView<T> make_mat_view_from_eigen(MatRM<T>& A) {
    return MatView<T>(
        A.data(),
        A.rows(),
        A.cols()
    );
}

MatRM<complex128> FermionGACPP::get_1body_fock(Eigen::Ref<const MatRM<complex128>> A) const {
    const int M = A.rows();
    const auto& C = Cdict_.at(M);
    const int dim = 1 << M;
    MatRM<complex128> F = MatRM<complex128>::Zero(dim, dim);
    for (int a = 0; a < M; a++) {
        for (int b = 0; b < M; b++) {
            F.noalias() += A(a, b) * C[a].transpose() * C[b];
        }
    }
    return F;
}

InitialGuessResult FermionGACPP::compute_initial_guess(
    const InputView& input,
    const MatView<complex128>& corr_view,
    int max_iter,
    double tol
) const {
    // initialize memory
    std::vector<MatRM<complex128>> psi_old(N);
    std::vector<MatRM<complex128>> psi_new(N);
    std::vector<MatRM<complex128>> Lc(N);
    Vec<double> Ec(N);

    Eigen::setNbThreads(1);
    #pragma omp parallel for
    for (int I = 0; I < N; ++I) {
        psi_old[I] = input.psiarr[I].eigen();

        const int M = Mvec[I];
        const int dim = 1 << M;

        psi_new[I].resize(dim, dim);
        psi_new[I].setZero();
    }

    for (int it = 0; it < max_iter; it++) {
        std::vector<MatView<complex128>> psi_views;
        psi_views.reserve(N);
        for (int I = 0; I < N; I++) {
            psi_views.push_back(make_mat_view_from_eigen(psi_old[I]));
        }
        auto R = compute_renormalizations(psi_views, input.n);

        Eigen::setNbThreads(1);
        #pragma omp parallel for schedule(static)
        for (int I = 0; I < N; I++) {
            const int M = Mvec[I];
            const int dim = 1 << M;
            auto& C = Cdict_.at(M);
            auto nI = input.n[I].eigen();
            auto LI = input.L[I].eigen();

            MatRM<complex128> GI = compute_G(I, input.tblock, R, corr_view.eigen());

            // get Lc
            MatRM<complex128> r(M, M);
            for (int a = 0; a < M; a++) {
                const double factor = 0.5 * (1.0/(1.0 - nI[a]) - 1.0/nI[a]);
                r.col(a) = factor * R[I].col(a);
            }
            Lc[I] = (r.adjoint() * GI) - LI;

            // get embedding Hamiltonian
            auto Hloc = compute_Hloc(input.harr[I], input.Uarr[I]);
            const auto& LcI = Lc[I];
            MatRM<complex128> HL = get_1body_fock(LcI);
            Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(dim, dim);
            MatRM<complex128> HI = kron<complex128>(Hloc.transpose(), id) + kron<complex128>(id, HL + HL.adjoint());
            // construct HI
            for (int a = 0; a < M; a++) {
                double b = std::sqrt(nI[a] * (1-nI[a]));
                for (int alpha = 0; alpha < M; alpha++) {
                    complex128 factor = GI(alpha, a) / b;
                    HI.noalias() += factor * kron<complex128>(C[alpha].transpose(), C[a].transpose());
                    HI.noalias() += std::conj(factor) * kron<complex128>(C[alpha], C[a]);
                }
            }
            // diagonalize and get groundstate
            Eigen::SelfAdjointEigenSolver<MatRM<complex128>> solver(HI);
            check_eigen_status(solver.info(), "Initial guess H" + std::to_string(I));
            Ec[I] = solver.eigenvalues()[0];
            auto v0 = solver.eigenvectors().col(0);
            Eigen::Map<const MatRM<complex128>> psi_mat(v0.data(), dim, dim);
            psi_new[I] = psi_mat;
        }

        // check convergence

        psi_old.swap(psi_new);
    }

    InitialGuessResult result;
    result.psiarr = std::move(psi_old);
    result.Lc = std::move(Lc);
    result.Ec = std::move(Ec);
    return result;
}

double FermionGACPP::compute_lagrangian(const InputView& input) const {
    // diagonalize Hqp and get first Ne energies
    auto R = compute_renormalizations(input.psiarr, input.n);
    auto Hqp = compute_Hqp(input.L, R, input.tblock);
    Eigen::setNbThreads(omp_get_max_threads());
    Eigen::SelfAdjointEigenSolver<MatRM<complex128>> solver(Hqp);
    check_eigen_status(solver.info(), "Hqp");
    double Lag = solver.eigenvalues().head(Ne).sum();

    double site_sum = 0.0;
    Eigen::setNbThreads(1);
    #pragma omp parallel for reduction(+:site_sum) schedule(static)
    for (int I = 0; I < N; I++) {
        auto psi = input.psiarr[I].eigen();
        auto nI = input.n[I].eigen();
        auto LI = input.L[I].eigen();
        auto LcI = input.Lc[I].eigen();
        const double EcI = input.Ec.eigen()[I];

        auto Hloc = compute_Hloc(input.harr[I], input.Uarr[I]);
        MatRM<complex128> HL = get_1body_fock(LcI);
        MatRM<complex128> Hemb_psi = (Hloc * psi) + (psi * (HL + HL.adjoint()));
        site_sum += std::real(matrix_ip(psi, Hemb_psi));
        site_sum += EcI * (1.0 - psi.squaredNorm());

        complex128 LmixI = ((LI + LcI).diagonal().array() * nI.array()).sum();
        site_sum -= 2*std::real(LmixI);
    }
    Lag += site_sum;

    return Lag;
}

void FermionGACPP::compute_gradient(
    const InputView& input,
    GradientOutput& output
) const {
    // get renormalizations and qp correlation matrix
    auto R = compute_renormalizations(input.psiarr, input.n);
    auto corr = compute_qp_corr(input.L, R, input.tblock);

    Eigen::setNbThreads(1);
    #pragma omp parallel for schedule(static)
    for (int I = 0; I < N; I++) {
        auto nI = input.n[I].eigen();
        auto psi = input.psiarr[I].eigen();
        const int M = Mvec[I];
        const int dim = 1 << M;
        const auto& C = Cdict_.at(M);
        const double EcI = input.Ec.eigen()[I];

        output.psiarr[I].resize(dim, dim);
        output.L[I].resize(M, M);
        output.Lc[I].resize(M, M);
        output.n[I].resize(M);

        // get Ec gradient
        output.Ec[I] = 1.0 - psi.squaredNorm();

        // get L gradient
        const int i0 = Moff_[I];
        auto& gL = output.L[I];
        gL.noalias() = corr.block(i0, i0, M, M);
        gL.diagonal().array() -= nI.array().template cast<complex128>();

        // get Lc gradient
        auto& gLc = output.Lc[I];
        MatRM<complex128> A = psi.adjoint() * psi;
        for (int a = 0; a < M; a++) {
            for (int b = 0; b < M; b++) {
                auto cbA = C[b] * A;
                gLc(a, b) = (C[a].array() * cbA.array()).sum();
            }
        }
        gLc.diagonal().array() -= nI.array().template cast<complex128>();

        MatRM<complex128> GI = compute_G(I, input.tblock, R, corr);

        // compute psi gradient
        auto& gpsi = output.psiarr[I];
        auto Hloc = compute_Hloc(input.harr[I], input.Uarr[I]);
        auto LcI = input.Lc[I].eigen();
        MatRM<complex128> HL = get_1body_fock(LcI);
        gpsi.noalias() = (Hloc * psi) + (psi * (HL + HL.adjoint()));
        gpsi.noalias() -= EcI*psi;
        for (int a = 0; a < M; a++) {
            double b = std::sqrt(nI[a] * (1-nI[a]));
            for (int alpha = 0; alpha < M; alpha++) {
                complex128 factor = GI(alpha, a) / b;
                gpsi.noalias() += factor * (C[alpha] * psi * C[a].transpose());
                gpsi.noalias() += std::conj(factor) * (C[alpha].transpose() * psi * C[a]);
            }
        }

        // compute n gradient
        auto& gn = output.n[I];
        MatRM<complex128> r(M, M);
        for (int a = 0; a < M; a++) {
            const double factor = 0.5 * (1.0/(1.0 - nI[a]) - 1.0/nI[a]);
            r.col(a) = factor * R[I].col(a);
        }
        auto LI = input.L[I].eigen();
        MatRM<complex128> An = r.adjoint() * GI;
        An.noalias() -= LI + LcI;
        for (int z = 0; z < M; z++) {
            gn[z] = 2.0 * std::real(An(z,z));
        }
    }
}