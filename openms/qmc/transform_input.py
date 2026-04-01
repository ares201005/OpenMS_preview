#
# @ 2026. Triad National Security, LLC. All rights reserved.
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
#         Jacob Williams <jzw@lanl.gov>
#
# Created:          2026-03-19
# Last modified:    2026-03-19

import numpy as np
from pyscf.gto import Mole as Mol
from pyscf.scf.hf import RHF as HF
from openms.mqed.qedhf import RHF as QEDHF

from scipy.linalg import svd
from openms.lib.mathlib import loewdin_orth  # from pyscf.lo.orth import lowdin
from openms.qmc.tools import bilinear_decomposition


class AFQMCSystem:
    r"""System transformed for input to AFQMC.
    Note that this class doesn't do any processing: for instance,
    * in cavity QED systems, the one- and two-electron integrals should already
      be shifted by the dipole self-energy;
    * the electron repulsion integrals and bilinear tensors should already be
      decomposed into L_gamma^2 form.

    The goal is for this to act as a drop-in replacement for QMCbase.system,
    which is currently an extended pyscf.gto.Mole or openms.lib.Boson object.

    In coupled fermion-boson systems, we currently assume the form
    gmat[i, :, :] * (a^+[i] + a[i]) for mode i's bilinear coupling:
    that is, the electronic operator gmat couples to boson displacement,
    and all constants (like sqrt(omega[i]/2) in cavity QED) are in gmat.

    Since we also replace openms.qmc.QMCbase.get_integrals(), there are also
    some attributes from openms.qmc.QMCbase that are replicated here.

    Parameters
    ----------
    n_ao: int
        The number of orbitals/basis functions for the electronic part.
    h1e : np.ndarray
        The one-electron integrals; h1e.shape = (nspin, n_ao, n_ao)
    ltensor : np.ndarray
        The two-electron integrals and (electronic part of the) bilinear/DSE
        terms, in the QMC decomposed form; ltensor.shape = (nchol, nao, nao)
    ovlp : np.ndarray
        The electronic overlap matrix; ovlp.shape = (n_ao, n_ao)
    elec_wf : np.ndarray
        The electronic trial wavefunction; elec_wf.shape = (n_ao, )
    nelectron : int
        The total number of electrons.
    spin : int
        The total spin, equal to nelec[0] - nelec[1].
    nelec : list[int] (Optional)
        The number of [alpha, beta] electrons.
    nuc_energy : float
        The (constant) electron-nuclear energy.
    nmodes : int (Optional)
        The number of boson modes.
    dim_fock : int (Optional)
        The dimension of the Fock space for each boson mode.
    boson_freq : np.ndarray (Optional)
        The boson frequencies. boson_freq.shape = (nmode,)
    boson_wf : np.ndarary (Optional)
        The bosons' trial wavefunction; boson_wf.shape = (nmode, dim_fock)
    verbose : int
        Flag for verbosity (1 is lowest; 5 for debugging)
    stdout : int
        The standard output buffer. (Used by openms.qmc.{trial, generic_walkers}.)

    Fermionic attributes
    ----------
    n_ao: int
        The number of orbitals/basis functions for the electronic part.
        (Parameter passed through unchanged.)
    h1e : np.ndarray
        The one-electron integrals; h1e.shape = (nspin, n_ao, n_ao)
        (Parameter passed through unchanged.)
    ltensor : np.ndarray
        The two-electron integrals and (electronic part of the) bilinear/DSE
        terms, in the QMC decomposed form; ltensor.shape = (nchol, nao, nao)
        (Parameter passed through unchanged.)
    ovlp : np.ndarray
        The electronic overlap matrix; ovlp.shape = (n_ao, n_ao)
        (Parameter passed through unchanged.)
    elec_wf : np.ndarray
        The electronic trial wavefunction; elec_wf.shape = (n_ao, )
        (Parameter passed through unchanged.)
    nelectron : int
        The total number of electrons.
        (Parameter passed through unchanged.)
    spin : int
        The total spin, equal to nelec[0] - nelec[1].
        (Parameter passed through unchanged.)
    nelec : list[int] (Optional)
        The number of [alpha, beta] electrons. Computed if not provided.
    nuc_energy : float
        The (constant) electron-nuclear energy.

    Bosonic attributes
    ------------------
    nmodes : int (Optional)
        The number of boson modes.
        (Parameter passed through unchanged.)
    dim_fock : int (Optional)
        The dimension of the Fock space for each boson mode.
        (Parameter passed through unchanged.)
    nboson_states : int (Optional)
        The total size of the boson degrees of freedom, dim_fock * nmodes.
        Should be DEPRECATED.
    boson_freq : np.ndarray (Optional)
        The boson frequencies. boson_freq.shape = (nmode,)
        (Parameter passed through unchanged.)
    boson_wf : np.ndarary (Optional)
        The bosons' trial wavefunction; boson_wf.shape = (nmode, dim_fock)
        (Parameter passed through unchanged.)

    General attributes
    ------------------
    verbose : int
        Flag for verbosity (1 is lowest; 5 for debugging)
        (Parameter passed through unchanged.)
    stdout : int
        The standard output buffer. (Used by openms.qmc.{trial, generic_walkers}.)
        (Parameter passed through unchanged.)

    Methods
    -------
    nao_nr(): None -> int
        The number of electronic orbitals/basis functions as a function
        (matches pyscf syntax).
    energy_nuc(): None -> float
        The nuclear energy (matches pyscf syntax)


    """

    def __init__(
        self,
        n_ao: int,
        h1e: np.ndarray,
        ltensor: np.ndarray,
        ovlp: np.ndarray,
        elec_wf: np.ndarray,
        nelectron: int,
        spin: int = 0,
        nelec: list[int] | None = None,
        nuc_energy: float = 0.0,
        nmodes: int | None = None,
        dim_fock: int | None = None,
        boson_freq: np.ndarray | None = None,
        boson_wf: np.ndarray | None = None,
        verbose: int = 3,
        stdout: int = 1,
    ):
        # Electron quantities
        self.n_ao = n_ao
        self.h1e = h1e
        self.ltensor = ltensor
        self.ovlp = ovlp
        self.elec_wf = elec_wf
        self.nelectron = nelectron
        self.spin = spin
        if nelec is None:
            self.nelec[0] = (self.nelectron + self.spin) // 2
            self.nelec[1] = self.nelectron - self.nelec[0]
        else:
            self.nelec = nelec

        self.nuc = nuc_energy

        # Methods (to replicate pyscf syntax. TODO: deprecate)
        self.nao_nr = lambda *args: n_ao
        self.energy_nuc = lambda *args: nuc_energy

        # Boson quantities
        self.nmodes = nmodes
        self.dim_fock = dim_fock
        self.nboson_states = self.nmodes * self.dim_fock
        self.boson_freq = boson_freq
        self.boson_wf = boson_wf

        # General quantities
        self.verbose = verbose
        self.stdout = stdout

    def intor(self, which: str = "int1e_ovlp") -> np.ndarray:
        if which.lower() != "int1e_ovlp":
            raise NotImplementedError("Only overlap implemented for now")
        return self.ovlp


class QMCMeanField:
    r"""Mostly a dataclass for passing electronic mean-field properties to QMC.
    Templated from PySCF's scf.hf.RHF object.

    Parameters
    ----------
    h1e : np.ndarray
        The one-electron Hamiltonian after dressing; h1e.shape = (n_ao, n_ao)
    ovlp : np.ndarray
        The atomic orbital overlap matrix; ovlp.shape = (n_ao, n_ao)
    mo_coeff : np.ndarray
        The molecular orbital coefficients; mo_coeff.shape = (nspin, n_ao, n_ao)
    mo_occ : np.ndarray
        The occupations of the molecular orbitals; mo_occ.shape = (nspin, n_ao)
    _eri : np.ndarray
        The electron repulsion integrals; _eri.shape = (n_ao, n_ao, n_ao, n_ao)
    E_mf : float
        The mean-field energy, computed previously.

    Attributes
    ----------
    h1e : np.ndarray
        The one-electron Hamiltonian after dressing; h1e.shape = (n_ao, n_ao)
    ovlp : np.ndarray
        The atomic orbital overlap matrix; ovlp.shape = (n_ao, n_ao)
    mo_coeff : np.ndarray
        The molecular orbital coefficients; mo_coeff.shape = (nspin, n_ao, n_ao)
    mo_occ : np.ndarray
        The occupations of the molecular orbitals; mo_occ.shape = (nspin, n_ao)
    _eri : np.ndarray
        The electron repulsion integrals; _eri.shape = (n_ao, n_ao, n_ao, n_ao)
    E_mf : float
        The mean-field energy, computed previously.

    Methods
    -------
    get_hcore(): None -> np.ndarray = h1e
    get_ovlp(): None -> np.ndarray = ovlp
    kernel(): None -> float = E_mf
    """

    def __init__(
        self,
        h1e: np.ndarray,
        ovlp: np.ndarray,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        eri: np.ndarray,
        E_mf: float = 0.0,
    ):
        self.h1e = h1e
        self.ovlp = ovlp
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._eri = eri
        self.E_mf = E_mf

    def get_hcore(self) -> np.ndarray:
        r"""To be deprecated. Returns one-electron Hamiltonian."""
        return self.h1e

    def get_ovlp(self) -> np.ndarray:
        r"""To be deprecated. Returns overlap matrix."""
        return self.ovlp

    def kernel(self) -> float:
        r"""To be deprecated. Returns (but DOES NOT compute) the mean-field energy."""
        return self.E_mf


def orth_overlap(S: np.ndarray, oao: str = "oao") -> np.ndarray:
    r"""Return the orthogonalized overlap matrix S.

    This matrix is used to orthogonalize:
        h1e -> S.conj().T @ h1e @ S
        gmat -> S.conj().T @ gmat @ S
        Lgamma -> S.conj().T @ Lgamma @ S

    Parameters
    ----------
    S : np.ndarray
        The metric matrix: either the AO overlap matrix if oao == "oao",
        or the matrix of MO coefficients in the AO basis.
    oao : str
        Whether to use Loewdin orthogonalization (oao == "oao": default),
        or the MO basis (oao != "oao").

    Returns
    -------
    S : np.ndarray
        The MO coefficient matrix (unchanged),
        or the Loewdin orthogonalization of the AO overlap matrix.
    """
    return loewdin_orth(S) if oao.lower() == "oao" else S


def dress_h1e_photon(
    h1e_bare: np.ndarray, S: np.ndarray, lambda_dot_mu: np.ndarray
) -> np.ndarray:
    r"""Dress the one-electron integrals by the photon degrees of freedom.
    Assumes the Pauli-Fierz Hamiltonian in the length gauge and
    under the long-wavelength approximation, so that

    .. math::
        H = H_e + \sum_\alpha \omega_\alpha (a^\dagger_\alpha a_\alpha + \frac12) +
            \sum_\alpha \sqrt{\omega_\alpha/2} \lambda_\alpha \cdot \mu +
            \sum_\alpha \left( \lambda_\alpha \cdot \mu \right)^2,

    where :math:`\lambda_\alpha` is the coupling strength (a 3-vector),
    and :math:`\mu` is the molecular dipole (also a 3-vector).

    Parameters
    ----------
    h1e_bare : np.ndarray
        The bare electronic Hamiltonian, in the AO basis (not yet orthogonalized);
        h1e_bare.shape = (nao, nao)
    S : np.ndarray
        The metric matrix for orthogonalizaiton; S.shape = (nao, nao)
    lambda_dot_mu : np.ndarray
        The coupling matrix :math:`\lambda_\alpha \cdot \mu` (without the
        factor :math:`\sqrt{\omega_\alpha/2}`);
        lambda_dot_mu.shape = (nmode, nao, nao)

    Returns
    -------
    h1e : np.ndarray
        The dressed one-electron integrals.
        h1e.shape = (nao, nao) (TODO: (nspin, nao, nao)?)
    """
    h1e = S.conj().T @ h1e_bare @ S
    ldmu_orth = np.einsum(
        "pi, nij, jq -> npq", S.conj().T, lambda_dot_mu, S, optimize=True
    )
    h1e += 0.5 * np.einsum("npq, nqs -> qs", ldmu_orth, ldmu_orth)

    return h1e


def decompose_tensor_eri(
    eri: np.ndarray,
    S: np.ndarray,
    thresh: float = 1.0e-12,
    lambda_dot_mu: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    r"""Decompose the electron repulsion integrals into Cholesky form.

    A modified copy of openms.qmc.tools.chols_full()
    that does not require a PySCF Mole object.

    Allows folding of the DSE tensor lambda_dot_mu into the ERI;
    this is not used at the moment.

    Note that this holds the entire ERI object in memory.
    TODO: re-implement qmc.tools.chols_blocked() without requiring Mole object.

    Parameters
    ----------
    eri : np.ndarray
        The electron repulsion integrals; eri.shape = (nao, nao, nao, nao)
    S : np.ndarray
        The orthogonalization matrix; S.shape = (nao, nao)
    thresh : float
        The SVD threshold; discard singular values < thresh.
    lambda_dot_mu : np.ndarray (Optional)
        If present, add the DSE terms into the electron repulsion integrals.

    Returns
    -------
    L_eri : np.ndarray
        The electron repulsion integral tensors; L_eri = (n_eri, nao, nao)
    n_eri : int
        The number of tensors thus generated.
    """
    if lambda_dot_mu is not None:
        eri += np.einsum("npq, nrs->pqrs", lambda_dot_mu, lambda_dot_mu)

    nao = eri.shape[0]
    eri = eri.reshape((nao**2, -1))
    u, s, _ = svd(eri)
    del eri

    idx = s > thresh
    ltensor = (u[:, idx] * np.sqrt(s[idx])).T
    ltensor = ltensor.reshape(ltensor.shape[0], nao, nao)

    # Orthogonalize
    L_eri = np.einsum("pi, nij, jq -> npq", S.conj().T, ltensor, S, optimize=True)
    n_eri = L_eri.shape[0]

    return L_eri, n_eri


def decompose_tensor_dse(
    lambda_dot_mu: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, int]:
    r"""Return the dipole self-energy tensor to concatenate to the
    Cholesky tensors :math:`L_\gamma`, in the orthogonalized basis.

    Note: The dipole self-energy is

    .. math::
        \sum_\alpha (\lambda_\alpha \cdot \mu)^2 =
        \sum_\alpha L_\alpha^2,

    so the Cholesky tensors are simply the (orthogonalized)
    :math:`\lambda_\alpha \cdot \mu`.

    Parameters
    ----------
    lambda_dot_mu : np.ndarray
        The operator :math:`\lambda_\alpha \cdot \mu`, of shape (nmode, nao, nao)
    S : np.ndarray
        The orthogonalization matrix, of shape (nao, nao)

    Returns
    -------
    L_dse : np.ndarray
        The DSE tensors, of shape (nmode, nao, nao).
    n_dse : int
        The number of additional tensors added by the DSE (== nmode).
    """

    L_dse = np.einsum("pi, nij, jq -> npq", S.conj().T, lambda_dot_mu, S, optimize=True)
    n_dse = L_dse.shape[0]

    return L_dse, n_dse


def decompose_tensor_bilinear(
    g_bil: np.ndarray, S: np.ndarray, scheme: int = 2
) -> tuple[list[np.ndarray], int]:
    r"""Return the Cholesky tensors for the (photonic and phononic)
    bilinear coupling terms, in the orthogonalized basis.

    The input tensors should be structured such that

    .. math::
        H_{eb} = \sum_\alpha g_\alpha \otimes Q_\alpha,

    where :math:`g_\alpha` is g_bil[alpha, :, :], a fermionic operator of
    shape (nao, nao), and :math:`Q_\alpha` is the bosonic displacement
    operator on mode :math:`\alpha`. In other words, g_bil is in the
    (nonorthogonalized) AO basis, and already contains factors such as
    the :math:`\sqrt{\omega_\alpha/2}` for the cavity QED coupling.

    Parameters
    ----------
    g_bil : np.ndarray
        The bilinear coupling tensors for both photons and phonons,
        in the AO basis; g_bil.shape = (nmode, nao, nao)
    S : np.ndarray
        The matrix for orthogonalization, of shape (nao, nao)
    scheme : int
        The bilinear decomposition scheme; 1 or 2.
        scheme == 1: 3 * nmode tensors are returned;
        scheme == 2: 2 * nmode tensors are returned.

    Returns
    -------
    L_bil : list[np.ndarray]
        The bilinear tensors [L_elec, L_bose].
        L_elec.shape = (n_bil, nao, nao);
        L_bose.shape = (n_bil,).
    n_bil : int
        The number of bilinear tensors so generated.
    """

    if scheme != 1 or scheme != 2:
        raise ValueError(
            "Only bilinear decomposition schemes 1 (threefold) and 2 (twofold) are supported."
        )

    nmode = g_bil.shape[0]
    Afac = np.ones((nmode,))  # factors are within g_bil already
    Bfac = np.ones((nmode,))

    # Orthogonalize
    g_orth = np.einsum("pi, nij, jq -> npq", S.conj().T, g_bil, S, optimize=True)

    L_bil = bilinear_decomposition(Afac, Bfac, g_orth, scheme)
    n_bil = L_bil[0].shape[0]  # elec/bose tensors should have same length

    return L_bil, n_bil


def combine_boson(
    omega_photon: np.ndarray,
    omega_phonon: np.ndarray,
    lambda_dot_mu: np.ndarray,
    g_phonon: np.ndarray,
    dim_fock_photon: int,
    dim_fock_phonon: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Combine the photon and phonon quantities:
    frequencies, bilinear couplings, and Fock space dimensions.

    In this implementation we also multiply the photonic coupling matrix
    :math:`\lambda_\alpha \cdot \mu`, which is used in both the bilinear
    and dipole self-energy terms, by the bilinear factor :math:`\sqrt{\omega_\alpha/2}`.

    Parameters
    ----------
    omega_photon : np.ndarray
        The photon frequencies, of shape (nphoton,)
    omega_phonon : np.ndarray
        The phonon frequencies, of shape (nphonon,)
    lambda_dot_mu : np.ndarray
        The photon coupling matrix, of shape (nphoton, nao, nao)
    g_phonon : np.ndarray
        The phonon coupling matrix, of shape (nphonon, nao, nao)
    dim_fock_photon : int
        The photon Fock space dimension
    dim_fock_phonon : int
        The phonon Fock space dimension

    Returns
    -------
    omega : np.ndarray
        The boson frequencies, of shape (nmode,) == (nphonon + nphonon,)
    g_bilinear : np.ndarray
        The bilinear coupling matrix, of shape (nmode, nao, nao)
    dim_fock : int
        The Fock space dimension == max(dim_fock_photon, dim_fock_phonon)
    """

    omega = np.concatenate((omega_photon, omega_phonon), axis=0)

    g_photon = np.einsum("n, npq -> npq", np.sqrt(omega_photon / 2), lambda_dot_mu)
    g_bilinear = np.concatenate((g_photon, g_phonon), axis=0)

    dim_fock = max(dim_fock_photon, dim_fock_phonon)

    return omega, g_bilinear, dim_fock


def holstein_coupling(
    rdm1: np.ndarray,
    g: float | np.ndarray,
    omega: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    From a one-electron reduced density matrix,
    boson frequency (or frequencies) omega, and coupling constant(s) g,
    return the frequencies as an array and the bilinear coupling matrix
    (g_i * n_i).

    Parameters
    ----------
    rdm1: np.ndarray
        The one-electron reduced density matrix; rdm1.shape = (nao, nao)
    g : float | np.ndarray
        The Holstein coupling strength;
        optionally different for different modes (sites)
    omega : float | np.ndarray
        The phonon frequency (optionally differing by mode)

    Returns
    -------
    omega : np.ndarray
        The Holstein phonon frequencies, converted to an array if necessary;
        omega.shape = (nmode,)
    gmat : np.ndarray
        The Holstein coupling matrix; gmat.shape = (nmode, nao, nao)


    """
    nao = rdm1.shape[0]
    nmode = nao

    if isinstance(omega, float):
        omega = np.array([omega for _ in range(nmode)])
    elif omega.shape != (nmode,):
        raise ValueError("Pass a float or an array of size nmode for omega!")

    # Holstein coupling: g[i] * n[i, i] * (a^+[i] + a[i])
    # displacement vector is implicit in AFQMC object
    if isinstance(g, float):
        g = np.array([g for _ in range(nmode)])
    elif g.shape != (nmode,):
        raise ValueError("Pass a float or an array of size nmode for g!")

    gmat = np.zeros((nmode, nao, nao))
    for i in range(nmode):
        gmat[i, i, i] = g[i] * rdm1[i, i]

    return omega, gmat
