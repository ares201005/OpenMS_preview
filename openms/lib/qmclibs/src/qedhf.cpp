#include "qedhf.hpp"

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
#else
    #error "No BLAS backend defined. Define either USE_ACCELERATE or USE_OPENBLAS."
#endif
#include <omp.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_gamma.h>  // For gsl_sf_fact

inline int packed_index(int i, int j) {
    if (i > j) std::swap(i, j);
    return j * (j + 1) / 2 + i;
}


std::vector<double> pack_symmetric(const py::array_t<double>& pdm) {
    auto r = pdm.unchecked<2>();
    const ssize_t dim = r.shape(0);
    std::vector<double> packed;
    packed.reserve(dim * (dim + 1) / 2);

    for (ssize_t i = 0; i < dim; ++i) {
        for (ssize_t j = 0; j <= i; ++j) {
            packed.push_back(r(i, j));
        }
    }
    return packed;
}



std::pair<py::array_t<double>, py::array_t<double>> displacement_matrix_cpp(
    const std::vector<int>& nboson_states,
    int imode,
    py::array_t<double, py::array::c_style | py::array::forcecast> freq,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> pdm,
    py::array_t<double, py::array::c_style | py::array::forcecast> vsq,
    double shift
) {
    const double* freq_ptr = freq.data();
    const double* vsq_ptr = vsq.data();
    const double* eta_ptr = eta.data();
    const double* pdm_ptr = pdm.data();

    ssize_t nao = eta.shape(1);
    ssize_t mdim = nboson_states.at(imode);
    ssize_t packed_dim = mdim * (mdim + 1) / 2;

    py::array_t<double> disp_mat({packed_dim, nao, nao});
    py::array_t<double> exp_val({nao, nao});

    auto* disp_data = disp_mat.mutable_data();
    auto* exp_data = exp_val.mutable_data();

    const double tau = std::exp(vsq_ptr[imode]);
    const double tmp = tau / freq_ptr[imode];

    // Precompute factor matrix (nao x nao)
    std::vector<double> factor(nao * nao);
    for (ssize_t p = 0; p < nao; ++p) {
        for (ssize_t q = 0; q < nao; ++q) {
            double diff_eta = eta_ptr[imode * nao + p] - eta_ptr[imode * nao + q] + shift;
            factor[p * nao + q] = tmp * diff_eta;
        }
    }

    // Fill displacement matrix
    for (int m = 0; m < mdim; ++m) {
        for (int n = 0; n <= m; ++n) {
            int idx = packed_index(m, n);
            double* slice = disp_data + idx * nao * nao;

            for (ssize_t i = 0; i < nao * nao; ++i) {
                double f = factor[i];
                double val = 0.0;

                if (m == n) {
                    val = gsl_sf_laguerre_n(m, 0.0, f * f);
                } else {
                    double ratio = gsl_sf_fact(n) / gsl_sf_fact(m);
                    val = 2.0 * std::sqrt(ratio)
                        * std::pow(-f, m - n)
                        * gsl_sf_laguerre_n(n, m - n, f * f);
                }

                slice[i] = val;
            }
        }
    }

    // Multiply by exp(-0.5 * factor^2)
    for (ssize_t i = 0; i < nao * nao; ++i) {
        double g = std::exp(-0.5 * factor[i] * factor[i]);
        for (int idx = 0; idx < packed_dim; ++idx) {
            disp_data[idx * nao * nao + i] *= g;
        }
    }

    // Contract with symmetric PDM
    for (ssize_t pq = 0; pq < nao * nao; ++pq) {
        double acc = 0.0;
        for (int idx = 0; idx < packed_dim; ++idx) {
            acc += pdm_ptr[idx] * disp_data[idx * nao * nao + pq];
        }
        exp_data[pq] = acc;
    }

    return {disp_mat, exp_val};
}


// helper that computes FC factor matrix for a single (p,q) pair
static void compute_fc_matrix(
    int nao,
    const double *base_diff,
    double tmp,
    int mdim,
    const double *packed_pdm,
    double shift,
    double *fc_out
) {
    if (mdim == 1) {
        // vacuum gaussian case
        for (int r = 0; r < nao; ++r) {
            for (int s = 0; s < nao; ++s) {
                double A = tmp * (base_diff[r*nao + s] + shift);
                fc_out[r*nao + s] = std::exp(-0.5 * A * A);
            }
        }
        return;
    }
    int packed_dim = mdim * (mdim + 1) / 2;
    // compute A array and exponentials
    std::vector<double> Aarr(nao * nao);
    for (int r = 0; r < nao; ++r)
        for (int s = 0; s < nao; ++s)
            Aarr[r*nao + s] = tmp * (base_diff[r*nao + s] + shift);

    for (int r = 0; r < nao; ++r) {
        for (int s = 0; s < nao; ++s) {
            double A = Aarr[r*nao + s];
            double exponential = std::exp(-0.5 * A * A);
            double acc = 0.0;
            int idx = 0;
            for (int m = 0; m < mdim; ++m) {
                for (int n = 0; n <= m; ++n) {
                    double val;
                    if (m == n) {
                        val = gsl_sf_laguerre_n(m, 0.0, A*A);
                    } else {
                        double ratio = gsl_sf_fact(n) / gsl_sf_fact(m);
                        val = 2.0 * std::sqrt(ratio)
                              * std::pow(-A, m - n)
                              * gsl_sf_laguerre_n(n, m - n, A*A);
                    }
                    acc += packed_pdm[idx++] * val;
                }
            }
            fc_out[r*nao + s] = acc * exponential;
        }
    }
}


std::pair<py::array_t<double>, py::array_t<double>> get_JK_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> ltensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> dm_do,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta_imode,
    double tau,
    double omega,
    int mdim,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> pdm
) {
    auto lt = ltensor.unchecked<3>();
    auto dm = dm_do.unchecked<2>();
    auto eta = eta_imode.unchecked<1>();

    int nao = dm.shape(0);
    int X = lt.shape(0);

    py::array_t<double> vj({nao, nao});
    py::array_t<double> vk({nao, nao});
    auto vj_buf = vj.mutable_unchecked<2>();
    auto vk_buf = vk.mutable_unchecked<2>();

    // precompute base_diff
    std::vector<double> base_diff(nao * nao);
    for (int p = 0; p < nao; ++p) {
        for (int q = 0; q < nao; ++q) {
            base_diff[p*nao + q] = eta(p) - eta(q);
        }
    }

    // pack photon density matrix (real part only)
    std::vector<double> packed_pdm;
    if (mdim > 1) {
        packed_pdm.reserve(mdim * (mdim + 1) / 2);
        auto pdm_buf = pdm.unchecked<2>();
        for (int m = 0; m < mdim; ++m) {
            for (int n = 0; n <= m; ++n) {
                packed_pdm.push_back(std::real(pdm_buf(m, n)));
            }
        }
    }

    double tmp = tau / omega;

    // main loops with OpenMP parallelization over p,q
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int p = 0; p < nao; ++p) {
        for (int q = p; q < nao; ++q) {
            double shift = eta(p) - eta(q);
            // compute FC factor matrix for this pair
            std::vector<double> fc(nao * nao);
            compute_fc_matrix(nao, base_diff.data(), tmp, mdim,
                              packed_pdm.empty() ? nullptr : packed_pdm.data(),
                              shift, fc.data());

            double accJ = 0.0;
            double accK = 0.0;
            for (int r = 0; r < nao; ++r) {
                for (int s = 0; s < nao; ++s) {
                    double dotJ = 0.0;
                    double dotK = 0.0;
                    for (int x = 0; x < X; ++x) {
                        dotJ += lt(x, p, q) * lt(x, r, s);
                        dotK += lt(x, p, s) * lt(x, r, q);
                    }
                    double w = dm(r, s) * fc[r*nao + s];
                    accJ += dotJ * w;
                    accK += dotK * w;
                }
            }
            vj_buf(p, q) = accJ;
            vj_buf(q, p) = accJ;
            vk_buf(p, q) = accK;
            vk_buf(q, p) = accK;
        }
    }

    return {vj, vk};
}


//std::pair<py::array_t<double>, py::array_t<double>> displacement_matrix(
//    const std::vector<int>& nboson_states,
//    int mode,
//    py::array_t<double> factor,
//    py::array_t<double> pdm
//) {
//    const int mdim = nboson_states[mode];
//    const size_t packed_mdim = mdim * (mdim + 1) / 2;
//
//    auto factor_buf = factor.unchecked<1>();  // assuming 1D for simplicity
//    ssize_t fsize = factor_buf.shape(0);
//
//    py::array_t<double> disp_mat({packed_mdim, fsize});
//    auto disp = disp_mat.mutable_unchecked<2>();
//
//    for (int i_m = 0; i_m < mdim; ++i_m) {
//        for (int i_n = 0; i_n <= i_m; ++i_n) {
//            size_t idx = packed_index(i_m, i_n);
//
//            for (ssize_t k = 0; k < fsize; ++k) {
//                double A = factor_buf(k);
//                double val = 0.0;
//
//                if (i_m == i_n) {
//                    val = gsl_sf_laguerre_n(i_m, 0.0, A * A);
//                } else {
//                    double ratio = gsl_sf_fact(i_n) / gsl_sf_fact(i_m);
//                    val = 2.0 * std::sqrt(ratio)
//                        * std::pow(-A, i_m - i_n)
//                        * gsl_sf_laguerre_n(i_n, i_m - i_n, A * A);
//                }
//
//                disp(idx, k) = val;
//            }
//        }
//    }
//
//    // Multiply by exp(-0.5 * A^2)
//    for (size_t idx = 0; idx < packed_mdim; ++idx) {
//        for (ssize_t k = 0; k < fsize; ++k) {
//            double A = factor_buf(k);
//            disp(idx, k) *= std::exp(-0.5 * A * A);
//        }
//    }
//
//    // Contract with photon density matrix
//    auto packed_pdm = pack_symmetric(pdm);
//    py::array_t<double> exp_val({fsize});
//    auto exp = exp_val.mutable_unchecked<1>();
//
//    for (ssize_t k = 0; k < fsize; ++k) {
//        double sum = 0.0;
//        for (size_t idx = 0; idx < packed_mdim; ++idx) {
//            sum += packed_pdm[idx] * disp(idx, k);
//        }
//        exp(k) = sum;
//    }
//
//    return std::make_pair(disp_mat, exp_val);
//}

/*
py::array_t<double> displacement_val(
    int imode,
    py::array_t<int, py::array::c_style | py::array::forcecast> nboson_states,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> brho,
    float shift,
) {
    ssize_t nmode = eta.shape(0);
    ssize_t nao = eta.shape(1);
    ssize_t mdim = nboson_states(imode);
    ssize_t packed_mdim = mdim * (mdim + 1) / 2; //

    // compute factor
    py::array_t<double> factor({nao, nao});
    py::array_t<double> disp_mat({packed_mdim, nao, nao});

    for (ssize_t p = 0; p < nao, ++im){
        for (ssize_t q = 0; q < nao; ++in){
            factor(p, q) = eta(imode, p) - eta(imode, q);
         }
    }
    factor += shift;

    for (ssize_t im = 0; im < nao, ++im){
        for (ssize_t in = im; in < nao; ++in){
            // TODO: implement factorial
            if (im == in){
                val = genlaguerre(n=i_m, alpha=0)(factor**2);
            } esle {
                ratio = 1.0; // factorial(i_n, exact=True) / factorial(i_m, exact=True);
                // Matrix elements
                val = 2.0 * numpy.sqrt(ratio) * (-factor)**(i_m - i_n) \
                      * genlaguerre(n=i_n, alpha=(i_m - i_n))(factor**2);
            }
            disp_mat[idx] = val;
        }
    }

}

*/
