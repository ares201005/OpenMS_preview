#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Authors:   Yu Zhang    <zhy@lanl.gov>
#          Ilia Mazin <imazin@lanl.gov>
#

import time
import numpy
from scipy import linalg

from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.scf import addons as scf_addons

import openms
from openms.lib import mathlib
from openms.lib import scipy_helper
from openms.mqed import diis
from openms.mqed import qedhf
from openms import __config__

TIGHT_GRAD_CONV_TOL = getattr(__config__, "TIGHT_GRAD_CONV_TOL", True)
LINEAR_DEP_THRESHOLD = getattr(__config__, "LINEAR_DEP_THRESHOLD", 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, "CHOLESKY_THRESHOLD", 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, "FORCE_PIVOTED_CHOLESKY", False)
LINEAR_DEP_TRIGGER = getattr(__config__, "LINEAR_DEP_TRIGGER", 1e-10)


def unitary_transform(U, A):
    r"U^T A U"
    B = numpy.einsum("ik, kj->ij", A, U)
    B = numpy.einsum("ki, kj->ij", U, B)
    return B


def eigh(h, s):
    r"""Solver for generalized eigenvalue problem."""
    e, c = linalg.eigh(h, s)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c


def cholesky_diag_fock_rao(mf, h1e):
    r"""Diagonalize the Fock matrix in RAO basis."""
    F_rao = ao2rao(h1e, mf.P)
    S_rao = get_reduced_overlp(mf.L)
    mo_energy, mo_coeff = eigh(F_rao, S_rao)
    mo_coeff = get_orbitals_from_rao(mo_coeff, mf.P)

    return mo_energy, mo_coeff

def ao2rao(A_ao, P):
    r"""Transforms a matrix from the AO basis to the reduced AO (RAO) basis

    .. math::

       A_{RAO} = P^T A_{AO} P

    where P is the projection onto the linarly independent AO basis
    """

    AP = numpy.einsum("ik, kj->ij", A_ao, P)
    A_rao = numpy.einsum("ik, kj->ij", P.conj().T, AP)
    del AP
    return A_rao


def get_reduced_overlp(L):
    return numpy.einsum("ik, kj->ij", L, L.conj().T)


def get_orbitals_from_rao(c, P):
    r"""Sets orbital coefficients to RAO orbital coefficients.
    """
    return numpy.einsum("ik, kj->ij", P, c)


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    r"""
    SCF kernel: the main QED-HF driver.

    Modified version of :external:func:`hf.kernel <pyscf.scf.hf.kernel>`
    from PySCF. The main difference is:

    - one- and two-body terms are scaled by Gaussian factors in the
      polarized dipole operator basis for all modes.

    Parameters
    ----------
    mf : :class:`RHF <mqed.qedhf.RHF>`
        Instance of OpenMS mean-field class.

    Keyword Arguments
    -----------------
    conv_tol : float
        Energy convergence threshold.
        **Optional**, ``default = 1e-10``.
    conv_tol_grad : float
        Energy gradients convergence threshold.
        **Optional**, ``default = sqrt(conv_tol)``.
    dump_chk : bool
        Whether to save SCF intermediate results
        in the checkpoint file.
        **Optional**, ``default = True``.
    dm0 : :class:`~numpy.ndarray`
        Initial guess density matrix. If not given (the default),
        the kernel takes the density matrix generated by
        :external:meth:`~pyscf.scf.hf.SCF.get_init_guess`.
    callback : function(envs_dict) => None
        callback function takes one ``dict`` as the argument which
        is generated by built-in function :func:`locals`, so that
        the callback function can access all local variables in the
        current environment.
    conv_check : bool
        Whether to perform an additional SCF cycle after convergence
        criteria are met. **Optional**, ``default = True``.

    Return
    ------
    scf_conv : bool
        Whether SCF has converged.
    e_tot : float
        QED Hartree-Fock energy of last iteration.
    mo_energy : :class:`~numpy.ndarray`
        Orbital energies. Depending on the ``eig`` function provided by
        ``mf`` object, the orbital energies may **NOT** be sorted.
    mo_coeff : :class:`~numpy.ndarray`
        Orbital coefficients.
    mo_occ : :class:`~numpy.ndarray`
        Orbital occupancies. The occupancies may **NOT** be sorted from
        large to small.
    """

    if 'init_dm' in kwargs:
        err_msg = "You see this error message because of the API " + \
                  "updates in pyscf v0.11. Keyword argument 'init_dm' " + \
                  "is replaced by 'dm0'"
        raise RuntimeError(err_msg)

    cput0 = (logger.process_clock(), logger.perf_counter())

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log_msg = f"Set gradient convergence threshold to {conv_tol_grad}"
        logger.info(mf, log_msg)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = scf.hf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    # Update initial guess
    mo_energy, mo_coeff = cholesky_diag_fock_rao(mf, dm)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)

    h1e = mf.bare_h1e = scf.hf.get_hcore(mol)
    vhf = scf.hf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    # Create initial photonic eigenvector guess for each mode
    for a in range(mf.qed.nmodes):
        mf.qed.boson_coeff[a] = mf.qed.update_boson_coeff(e_tot, dm, a)

    mo_energy, mo_coeff = cholesky_diag_fock_rao(mf, h1e + vhf)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    #logger.debug(mf, "trace of ao_density: %.8f", numpy.trace(dm))

    if mf.qed.use_cs:
        mf.qed.update_cs(dm)

    mf.init_var_params(dm)

    # construct h1e, gmat in DO representation (used in SC/VT-QEDHF)
    mf.get_h1e_DO(mol, dm=dm)

    h1e = mf.get_hcore(mol, dm, dress=True)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        scf.hf.chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = logger.timer(mf, 'initialize scf', *cput0)

    for cycle in range(mf.max_cycle):
        time0 = time.time()
        dm_last = dm
        last_hf_e = e_tot

        time1 = time.time()
        # update h1e in DO
        mf.get_h1e_DO(mol, dm=dm)
        time_h1e_do = time.time() - time1

        # compute gradient of eta
        time1 = time.time()
        mf.grad_var_params(mf.dm_do, mf.g_dipole, dm=dm)
        time_etagrad = time.time() - time1

        # use DIIS to update eta (in get_fock)
        time1 = time.time()
        h1e = mf.get_hcore(mol, dm, dress=True)
        time_hcore = time.time() - time1

        time1 = time.time()
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        time_fock = time.time() - time1

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)

        # update energy
        time1 = time.time()
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        time_veff = time.time() - time1

        h1e = mf.get_hcore(mol, dm, dress=True)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        time1 = time.time()
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        time_fock += time.time() - time1

        norm_gorb = linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_eta = mf.norm_var_params()
        norm_gorb += norm_eta

        norm_ddm = linalg.norm(dm-dm_last)
        logger.info(mf, '\ncycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |g_var|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_eta, norm_ddm)
        logger.debug(mf, "cycle= %d times: h1e_do = %.6g eta_grad = %.6g hcore = %.6g veff = %.6g  fock = %.6g scf = %.6g",
                    cycle+1, time_h1e_do, time_etagrad, time_hcore, time_veff, time_fock, time.time()-time0)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        h1e = mf.get_hcore(mol, dm, dress=True)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel()
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class RHF(qedhf.RHF):
    r"""Non-relativistic SC-QED-RHF subclass."""
    def __init__(
        self, mol, qed=None, xc=None, **kwargs):

        super().__init__(mol, qed, xc, **kwargs)

        if "scqedhf" not in openms.runtime_refs:
            openms.runtime_refs.append("scqedhf")

        self.mo2dipole = numpy.zeros_like(self.qed.gmat)
        self.ao2dipole = numpy.zeros_like(self.qed.gmat)

        self.dm_do = numpy.zeros_like(self.qed.gmat)
        self.h1e_DO = numpy.zeros_like(self.qed.gmat)
        self.eri_DO = numpy.zeros((self.qed.nmodes, self.nao, self.nao, self.nao, self.nao))

        self.eta = numpy.zeros((self.qed.nmodes, self.nao))
        self.eta_grad = numpy.zeros((self.qed.nmodes, self.nao))
        self.g_dipole = numpy.zeros((self.qed.nmodes, self.nao))

        self.precond = 0.1

        # Parameters for dipole moment basis set degeneracy
        self.dipole_degen_thresh = 1.0e-8
        self.dipole_fock_shift = 1.0e-3

        s1e = self.get_ovlp(mol)
        self.X = scf_addons.partial_cholesky_orth_(s1e, canthr=1.0e-7,
                                                  cholthr=CHOLESKY_THRESHOLD)

        # Check condition of overlap and e-p interaction matrices
        scipy_helper.remove_linear_dep(self, threshold=1.0e-7, lindep=1.0e-7,
                                       cholesky_threshold=CHOLESKY_THRESHOLD,
                                       force_pivoted_cholesky=FORCE_PIVOTED_CHOLESKY)

        self.L, self.P = mathlib.full_cholesky_orth(s1e, threshold=1.e-7)
        self.n_oao = self.P.shape[1]

        # will replace it with our general DIIS
        self.diis_space = 20
        self.DIIS = diis.SCF_DIIS


    def initialize_bare_fock(self, dm=None):
        r"""Return non-QED Fock matrix from provided or initial guess ``dm``."""
        mol = self.mol
        if dm is None: dm = scf.hf.get_init_guess(mol, self.init_guess)

        h1e = scf.hf.get_hcore(mol)
        vhf = scf.hf.get_veff(mol, dm)

        return h1e + vhf


    def get_bare_mo_coeff(self, dm):

        s1e = self.get_ovlp(self.mol)
        fock = self.initialize_bare_fock(dm)
        mo_energy, mo_coeff = self._eigh(fock, s1e)

        return mo_energy, mo_coeff


    def ao2mo(self, A):
        r"""Transform AO into MO

        .. math::

            A_{MO} =& C^T_{MO} A_{AO} C_{MO} \\
            A_{pq} =& \sum_{uv} C^T_{pu} A_{uv} C_{vq} = \sum_{pq}
            C^*_{up} A_{uv} C_{vq}, \text{ and } C^T_{pu} = C_{up}
        """

        Amo = numpy.einsum("uv, vq->uq", A, self.mo_coeff)
        Amo = numpy.einsum("up, uq->pq", self.mo_coeff, Amo)
        return Amo


    def check_n_resolve_degeneracy(self, evals, mo2dipole, dm):

        degeneracy = 1
        ediff = 0

        for p in range(self.nao - 1):

            ediff = evals[p + 1] - evals[p]

            if abs(ediff) < self.dipole_degen_thresh:
                degeneracy += 1

            elif abs(ediff) > self.dipole_degen_thresh and degeneracy > 1:
                r = p + 1 - degeneracy
                s = p + 1

                # non-QED Fock matrix, AO basis
                fock = self.initialize_bare_fock(dm = dm)

                # Transform to MO basis
                fock = self.ao2mo(fock)

                # Dipole matrix, AO basis
                r_ao = self.qed.get_dipole_ao(self.mol, self.qed.add_nuc_dipole)
                sum_dipole_ao = numpy.sum(r_ao, axis=0)
                sum_dipole_mo = self.ao2mo(sum_dipole_ao)

                # plus shift: f_pq += shift * r_pq
                fock += self.dipole_fock_shift * sum_dipole_mo

                # transform into dipole basis
                fock = unitary_transform(mo2dipole, fock)
                deg_fock = fock[r : r + degeneracy, r : r  + degeneracy]

                del sum_dipole_ao, sum_dipole_mo, r_ao
                del fock

                # diagonalize deg_fock
                evecs = linalg.eigh(deg_fock)[1]

                # the basis of the degenerate space --> the new basis
                # new = vector * deg_fock
                vectors = mo2dipole[:, r : s]
                vectors = numpy.einsum("ik, kj-> ij", vectors, evecs)
                mo2dipole[:, r : s] = vectors
                del vectors, deg_fock

                degeneracy = 1

        return self


    def get_dm_do(self, dm, U):
        r"""Transform ``dm`` density matrix with unitary matrix ``U``."""
        su = numpy.einsum("ik,kj->ij", self.get_ovlp(self.mol), U)
        return unitary_transform(su, dm)


    def initialize_eta(self, dm):
        r"""Initialize the eta parameters and dipole basis sets."""

        self.mo_coeff = self.get_bare_mo_coeff(dm)[1]

        # diagonalize the gmat in MO; then get ao2dipole basis transformaiton
        gmo = numpy.zeros_like(self.qed.gmat)  # gmat*sqrt(w/2) in MO

        for a in range(self.qed.nmodes):

            gmo[a] = self.qed.gmat[a] * numpy.sqrt(self.qed.omega[a] / 2.0) \
                     * self.qed.couplings_var[a]
            gmo[a] = self.ao2mo(gmo[a])  # transform into MO

            # create dipole basis
            evals, evecs = linalg.eigh(gmo[a])

            # check degeneracy
            self.check_n_resolve_degeneracy(evals, evecs, dm)
            self.mo2dipole[a] = evecs
            self.eta[a] = evals

            # Creating the basis change matrix from ao to dipole basis
            self.ao2dipole[a] = lib.einsum("ui, ip-> up",
                                           self.mo_coeff, self.mo2dipole[a])

        return self


    def get_eta_gradient(self, dm_do, g_DO, dm=None):
        r"""Compute the gradient of energy with respect to eta."""

        onebody_deta = numpy.zeros(self.nao)
        twobody_deta = numpy.zeros(self.nao)

        for a in range(self.qed.nmodes):

            for p in range(self.nao):
                onebody_deta[p] -= 2.0 * dm_do[a, p,p] * g_DO[a, p] / self.qed.omega[a]

            fc_derivative = self.gaussian_derivative_vectorized(self.eta, a)
            tmp1 = 2.0 * self.h1e_DO[a] * dm_do[a] * fc_derivative
            tmp2 = (2.0 * dm_do[a].diagonal().reshape(-1, 1) * dm_do[a].diagonal() \
                   - dm_do[a] * dm_do[a].T) \
                   * g_DO[a].reshape(1, -1) / self.qed.omega[a]
            onebody_deta += numpy.sum(tmp1 - tmp2, axis=1)
            del fc_derivative, tmp1, tmp2

            fc_derivative = self.gaussian_derivative_vectorized(self.eta, a, onebody=False)
            fc_derivative *= (2.0 * self.eri_DO[a] - self.eri_DO[a].transpose(0, 3, 2, 1))

            tmp = lib.einsum('pqrs, rs-> pq', fc_derivative, dm_do[a])
            twobody_deta = lib.einsum('pq, pq-> p', tmp, dm_do[a])
            del fc_derivative, tmp

            self.eta_grad[a] = onebody_deta + twobody_deta
        return self
    # variable gradients, here we only have eta
    grad_var_params = get_eta_gradient


    def construct_eri_DO(self, U):
        r"""Repulsion integral modifier according to dipole self-energy terms."""
        eri_ao = self.mol.intor("int2e", aosym="s1")
        eri = lib.einsum("pu, qv, rw, st, pqrs->uvwt", U, U, U, U, eri_ao)
        return eri


    def FC_factor(self, eta, imode, onebody=True):
        r"""Compute Franck-Condon (or renormalization) factor

        FIXME: check the whether eta has sqrt{\omega/2} or not

        .. math::
           \chi^\alpha_{pq} = \exp[-\frac{f^2_\alpha(\eta_{\alpha,p}-\eta_{\alpha,q})^2}
                                         {4\omega_\alpha}]
        """

        if onebody:
            p, q = numpy.ogrid[:self.nao, :self.nao]
            diff_eta = eta[imode, p] - eta[imode, q]
        else:
            p, q, r, s = numpy.ogrid[:self.nao, :self.nao, :self.nao, :self.nao]
            diff_eta = eta[imode, p] - eta[imode, q] +  eta[imode, r] - eta[imode, s]

        tmp = 1 / self.qed.omega[imode]
        ph_exp_val = self.qed.get_bdag_plus_b_sq_expval(imode)

        factor = numpy.exp((-0.5 * (tmp * diff_eta) ** 2) * (ph_exp_val + 1))

        if onebody:
            return factor.reshape(self.nao, self.nao)
        else:
            return factor.reshape(self.nao, self.nao, self.nao, self.nao)


    def gaussian_derivative_vectorized(self, eta, imode, onebody=True):

        if onebody:
            p, q = numpy.ogrid[:self.nao, :self.nao]
            diff_eta = eta[imode, q] - eta[imode, p]
        else:
            p, q, r, s = numpy.ogrid[:self.nao, :self.nao, :self.nao, :self.nao]
            diff_eta = eta[imode, q] - eta[imode, p] +  eta[imode, s] - eta[imode, r]

        tmp = 1 / self.qed.omega[imode]
        ph_exp_val = self.qed.get_bdag_plus_b_sq_expval(imode)

        # Apply the derivative formula
        derivative = numpy.exp((-0.5 * (tmp * diff_eta) ** 2) * (ph_exp_val + 1)) \
                     * (tmp ** 2) * diff_eta

        if onebody:
            return derivative.reshape(self.nao, self.nao)
        else:
            return derivative.reshape(self.nao, self.nao, self.nao, self.nao)


    def get_h1e_DO(self, mol=None, dm=None):
        r"""QED variational transformaiton dressed one-body integral.

        .. math::

            h_{uv} = h_{u'v'} \prod_\alpha U^\alpha_{up} U^\alpha_{vq}
                    \exp[-\chi^\alpha_{pq}] U^\alpha_{pu'}U^\alpha_{qv'}

        where

        .. math::

            \chi^\alpha_{pq} = -\frac{f^2_\alpha(\eta_{\alpha,p}-\eta_{\alpha,q})^2}
                                     {4\omega_\alpha}.

        .. note::
            considering moving the DSE correciton to this function
        """

        if mol is None: mol = self.mol

        for a in range(self.qed.nmodes):

            gtmp = self.qed.gmat[a] * numpy.sqrt(self.qed.omega[a] / 2.0)
            gtmp *= self.qed.couplings_var[a]
            gtmp = unitary_transform(self.ao2dipole[a], gtmp)

            # h1e in dipole basis
            self.h1e_DO[a] = unitary_transform(self.ao2dipole[a], self.bare_h1e)

            # one-body operator h1e_pq = h1e_pq + g_pq(p, l) * g_pq(l, p)
            for p in range(self.nao):
                self.g_dipole[a, p] = gtmp[p, p] - self.eta[a, p]
                self.h1e_DO[a, p, p] += self.g_dipole[a, p] ** 2 / self.qed.omega[a]
            del gtmp

            # transform DM from AO to DO
            self.dm_do[a] = self.get_dm_do(dm, self.ao2dipole[a])
            self.eri_DO[a] = self.construct_eri_DO(self.ao2dipole[a])

        return self


    def get_hcore(self, mol=None, dm=None, dress=True):
        r"""QED variational transformaiton dressed one-body integral.

        .. math::

            h_{uv} = h_{u'v'} \prod_\alpha U^\alpha_{up} U^\alpha_{vq}
                    \exp[-\chi^\alpha_{pq}] U^\alpha_{pu'}U^\alpha_{qv'}

        where

        .. math::

            \chi^\alpha_{pq} = -\frac{f^2_\alpha(\eta_{\alpha,p}-\eta_{\alpha,q})^2}{4\omega_\alpha}.

        .. note::
            considering moving the DSE correction to this function
        """

        if mol is None: mol = self.mol

        if dress == False:
            return self.bare_h1e

        else:

            h1e_tot = numpy.zeros((self.nao, self.nao))

            for a in range(self.qed.nmodes):

                factor = self.FC_factor(self.eta, a)
                h1e = lib.einsum("pq, pq-> pq", self.h1e_DO[a], factor)

                Uinv = linalg.inv(self.ao2dipole[a])
                h1e_tot += unitary_transform(Uinv, h1e)

        return h1e_tot


    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""QED Hartree-Fock potential matrix for the given density matrix

        .. math::
            V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

        DSE-mediated one-electron parts:

         2 * \title{g}_{pp} * sum_{q} [D_{qq} \title{g}_{qq}]
                                         mean_value
         -D_{qp}\tidle{g}_{pq} * \tilde{g}_{qq} (diagonal element is then g_pq(p)**2)
        """
        # we also need to update hcore as it's dressed by photon displacement

        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vhf_tot = numpy.zeros((self.nao, self.nao))

        for a in range(self.qed.nmodes):

            dm_do = self.dm_do[a]
            g_dot_D = numpy.diagonal(dm_do) @ self.g_dipole[a, :]

            vhf_do = numpy.zeros((self.nao, self.nao))
            # vectorized code
            p_indices = numpy.arange(self.nao)
            vhf_do[p_indices, p_indices] += (2.0 * self.g_dipole[a, p_indices] * g_dot_D -
                                             numpy.square(self.g_dipole[a, p_indices]) * dm_do[p_indices, p_indices]) / self.qed.omega[a]

            vhf_do_offdiag = numpy.zeros_like(vhf_do)
            # Calculate off-diagonal elements
            p, q = numpy.triu_indices(self.nao, k = 1)
            vhf_do_offdiag[p, q] -= self.g_dipole[a, p] * self.g_dipole[a, q] * dm_do[q, p] / self.qed.omega[a]
            vhf_do_offdiag[q, p] = vhf_do_offdiag[p, q]  # Exploit symmetry
            vhf_do += vhf_do_offdiag

            # vectorized code
            fc_factor = self.FC_factor(self.eta, a, onebody = False)
            fc_factor *= (1.0 * self.eri_DO[a] - 0.5 * self.eri_DO[a].transpose(0, 3, 2, 1))
            vhf = 0.5 * lib.einsum('pqrs, rs->pq', fc_factor, dm_do)
            vhf += 0.5 * lib.einsum('qprs, rs->pq', fc_factor, dm_do)
            vhf_do += vhf

            # transform back to AO
            Uinv = linalg.inv(self.ao2dipole[a])
            vhf_tot += unitary_transform(Uinv, vhf_do)

        return vhf_tot


    def norm_var_params(self):
        var_norm = linalg.norm(self.eta_grad) / numpy.sqrt(self.eta.size)
        return var_norm


    def init_var_params(self, dm=None):
        r"""Initialize additional variational parameters"""
        self.initialize_eta(dm)


    def update_var_params(self):
        self.eta -= self.precond * self.eta_grad


    def pre_update_var_params(self):
        variables = self.eta
        gradients = self.eta_grad

        return variables, gradients


    def set_params(self, params, fock_shape=None):

        fsize = numpy.prod(fock_shape)
        f = params[:fsize].reshape(fock_shape)
        etasize = self.eta.size

        if params.size > fsize:
            self.eta = params[fsize:fsize+etasize].reshape(self.eta_grad.shape)

        return f


    def scf(self, dm0=None, **kwargs):

        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')


if __name__ == "__main__":
    import numpy
    from pyscf import gto

    atom = """
           H          0.86681        0.60144       0.00000
           F         -0.86681        0.60144       0.00000
           O          0.00000       -0.07579       0.00000
           He         0.00000        0.00000       2.50000
           """

    mol = gto.M(atom = atom,
                basis = "sto-3g",
                unit = "Angstrom",
                symmetry = True,
                verbose = 3)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

    qedmf = RHF(mol, omega = cavity_freq, vec = cavity_mode)
    qedmf.max_cycle = 500
    qedmf.kernel()
