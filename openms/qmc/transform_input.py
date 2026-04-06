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
from openms.lib.boson import Boson
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

    TODO: Consider whether to include elec_wf, boson_wf here, in QMCMeanField,
    or compute in openms.qmc.trial.py (as currently done). At the momemnt,
    stick with the current implementation.

    Parameters
    ----------
    n_ao: int
        The number of orbitals/basis functions for the electronic part.
    h1e : np.ndarray
        The one-electron integrals; h1e.shape = (nspin, n_ao, n_ao)
    ltensor : np.ndarray
        The two-electron integrals and (electronic part of the) bilinear/DSE
        terms, in the QMC decomposed form; ltensor.shape = (nchol, nao, nao)
    nbarefields : int
        The number of electron-only auxiliary fields required
        (includes dipole self-energy if present).
    nfields : int
        The total number of auxiliary fields: nbarefields + n_bilinear_fields
    ovlp : np.ndarray
        The electronic overlap matrix; ovlp.shape = (n_ao, n_ao)
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
    ovlp : np.ndarray
        The electronic overlap matrix; ovlp.shape = (n_ao, n_ao)
        (Parameter passed through unchanged.)
    ltensor : np.ndarray
        The two-electron integrals and (electronic part of the) bilinear/DSE
        terms, in the QMC decomposed form; ltensor.shape = (nchol, nao, nao)
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

    Bosonic attributes (Optional)
    -----------------------------
    nmodes : int
        The number of boson modes.
        (Parameter passed through unchanged.)
    dim_fock : int
        The dimension of the Fock space for each boson mode.
        (Parameter passed through unchanged.)
    nboson_states : int
        The total size of the boson degrees of freedom, dim_fock * nmodes.
        Should be DEPRECATED.
    boson_freq : np.ndarray
        The boson frequencies. boson_freq.shape = (nmode,)
        (Parameter passed through unchanged.)
    geb : np.ndarray
        The electronic part of the bilinear (electron-boson) coupling matrix,
        including all necessary coefficients, before decomposition into chol_bilinear.
        That is, H_bilinear = \sum_a^{modes} geb[a, :, :] * Q_a[:, :],
        where Q_a is the bosonic displacement operator for mode a.
        geb.shape = (nmodes, n_ao, n_ao).
        (Parameter passed through unchanged.)
    chol_bilinear : list[np.ndarray]
        The bilinear tensors: [chol_bilinear_e, chol_bilinear_b].
        chol_bilinear_e.shape = (n_fields_bilinear, nao, nao);
        chol_bilinear_b.shape = (n_fields_bilinear).
        That is, chol_bilinear_b is just an array of coefficients; the actual operators,
        which are currently assumed to be displacement operators, are computed on-the-fly
        inside the AFQMC routines.
        (Parameter passed through unchanged.)

    General attributes
    ------------------
    nbarefields : int
        The number of electron-only Cholesky tensors (includes dipole self-energy if present).
    nfields : int
        The total number of auxiliary fields: nbarefields + n_bilinear_fields
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
        ovlp: np.ndarray,
        ltensor: np.ndarray,
        nbarefields: int,
        nfields: int,
        nelectron: int,
        spin: int = 0,
        nelec: list[int] | None = None,
        nuc_energy: float = 0.0,
        nmodes: int | None = None,
        dim_fock: int | None = None,
        boson_freq: np.ndarray | None = None,
        chol_bilinear: list[np.ndarray] | None = None,
        geb: np.ndarray | None = None,
        verbose: int = 3,
        stdout: int = 1,
    ):
        # Electron quantities
        self.n_ao = n_ao
        self.h1e = h1e
        self.ovlp = ovlp
        self.ltensor = ltensor
        self.nbarefields = nbarefields
        self.nfields = nfields
        self.nelectron = nelectron
        self.spin = spin
        if nelec is None:
            self.nelec[0] = (self.nelectron + self.spin) // 2
            self.nelec[1] = self.nelectron - self.nelec[0]
        else:
            self.nelec = nelec

        self.nuc_energy = nuc_energy

        # Methods (to replicate pyscf syntax. TODO: deprecate)
        self.nao_nr = lambda *args: n_ao
        self.energy_nuc = lambda *args: nuc_energy

        # Boson quantities
        is_boson = nmodes is not None and dim_fock is not None
        self.nmodes = nmodes
        self.dim_fock = dim_fock
        self.nboson_states = self.nmodes * self.dim_fock if is_boson else None
        self.boson_freq = boson_freq
        self.chol_bilinear = chol_bilinear
        self.geb = geb

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
        The molecular orbital coefficients; mo_coeff.shape = (n_ao, n_mo)
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
        The molecular orbital coefficients; mo_coeff.shape = (n_ao, n_mo)
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


def orth_overlap(S: np.ndarray) -> np.ndarray:
    r"""Orthogonalize S via the Loewdin (symmetric) procedure.

    This matrix is used to orthogonalize:
        h1e -> S_orth.conj().T @ h1e @ S_orth
        gmat -> S_orth.conj().T @ gmat @ S_orth
        Lgamma -> S_orth.conj().T @ Lgamma @ S_orth

    Parameters
    ----------
    S : np.ndarray
        The metric matrix, typically the overlaps between atomic orbitals.

    Returns
    -------
    S_orth : np.ndarray
        The Loewdin orthogonalization of S.
    """
    return loewdin_orth(S)


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
        The bare electronic Hamiltonian, in the orthogonalized AO basis;
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
        h1e.shape = (nao, nao)
    """
    # h1e = S.conj().T @ h1e_bare @ S
    ldmu_orth = np.einsum(
        "pi, nij, jq -> npq", S.conj().T, lambda_dot_mu, S, optimize=True
    )
    return h1e_bare + 0.5 * np.einsum("npq, nqs -> qs", ldmu_orth, ldmu_orth)


def decompose_tensor_eri(
    eri: np.ndarray,
    S: np.ndarray,
    thresh: float = 1.0e-12,
    lambda_dot_mu: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    r"""Decompose the electron repulsion integrals into Cholesky form,
    and orthogonalize them with the overlap matrix S.

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


def holstein_coupling(
    rdm1: np.ndarray,
    g: float | np.ndarray,
    omega: float | np.ndarray,
    nmode: int | None = None,
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
    nmode: int (Optional)
        The number of phonon modes. Defaults to the number of atomic orbitals.

    Returns
    -------
    nmode_phonon : int
        The number of phonon modes. Defaults to the number of atomic orbitals.
    omega : np.ndarray
        The Holstein phonon frequencies, converted to an array if necessary;
        omega.shape = (nmode,)
    gmat : np.ndarray
        The Holstein coupling matrix; gmat.shape = (nmode, nao, nao)


    """
    nao = rdm1.shape[0]
    if nmode is None:
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

    return nmode, omega, gmat


def combine_boson(
    nmodes_photon: int | None = None,
    nmodes_phonon: int | None = None,
    omega_photon: np.ndarray | None = None,
    omega_phonon: np.ndarray | None = None,
    g_photon: np.ndarray | None = None,
    g_phonon: np.ndarray | None = None,
    dim_fock_photon: int | None = None,
    dim_fock_phonon: int | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    r"""Combine the photon and phonon quantities:
    frequencies, bilinear couplings, and Fock space dimensions.

    Allows photon, phonon, or both parameters to be specified.

    Parameters (All optional)
    -------------------------
    nmodes_photon: int
        The number of photon modes
    nmodes_phonon: int
        The number of phonon modes
    omega_photon : np.ndarray
        The photon frequencies, of shape (nphoton,)
    omega_phonon : np.ndarray
        The phonon frequencies, of shape (nphonon,)
    g_photon : np.ndarray
        The photon coupling matrix, of shape (nphoton, nao, nao).
        Includes factor of sqrt(omega_photon/2) already
    g_phonon : np.ndarray
        The phonon coupling matrix, of shape (nphonon, nao, nao)
    dim_fock_photon : int
        The photon Fock space dimension
    dim_fock_phonon : int
        The phonon Fock space dimension

    Returns
    -------
    nmodes : int
        The total number of boson modes == nmodes_photon + nmodes_phonon
    dim_fock : int
        The Fock space dimension == max(dim_fock_photon, dim_fock_phonon)
    omega : np.ndarray
        The boson frequencies, of shape (nmode,) == (nphonon + nphonon,)
    g_bilinear : np.ndarray
        The bilinear coupling matrix, of shape (nmode, nao, nao)
    """

    if omega_photon is None and omega_phonon is None:
        raise ValueError("Supply photon or phonon components (or both)!")

    if omega_photon is None:
        nmodes = nmodes_phonon
        omega = omega_phonon
        g_bilinear = g_phonon
        dim_fock = dim_fock_phonon
        pass
    elif omega_phonon is None:
        nmodes = nmodes_photon
        omega = omega_photon
        g_bilinear = g_photon
        dim_fock = dim_fock_photon
    else:
        nmodes = nmodes_photon + nmodes_phonon
        omega = np.concatenate((omega_photon, omega_phonon), axis=0)
        g_bilinear = np.concatenate((g_photon, g_phonon), axis=0)
        dim_fock = max(dim_fock_photon, dim_fock_phonon)

    return nmodes, dim_fock, omega, g_bilinear


def pyscf_openms_to_qmc(
    mol: Mol | Boson,
    mf: HF | QEDHF,
    use_oao: bool = True,
    chol_thresh: float = 1.0e-6,
    ncomponents: int = 1,
    has_photon: bool = False,
    has_phonon: bool = False,
    dress_eri_dse: bool = False,
    bilinear_scheme: int = 2,
    verbose: int | None = None,
    stdout: int | None = None,
    photon_gauge: str | None = None,
    long_wave_approx: bool = True,
    nmodes_photon: int | None = None,
    dim_fock_photon: int | None = None,
    omega_photon: float | np.ndarray | None = None,
    lambda_dot_mu: np.ndarray | None = None,
    phonon_type: str | None = None,
    dim_fock_phonon: int | None = None,
    coupling_phonon: float | np.ndarray | None = None,
    omega_phonon: float | np.ndarray | None = None,
) -> tuple[AFQMCSystem, QMCMeanField]:
    r"""Wrapper to convert pyscf/OpenMS objects to AFQMC form:
    A molecule object, either pyscf.gto.Mole or openms.lib.boson.Boson,
    and a mean-field object, either pyscf.scf.hf.RHF or openms.mqed.qedhf.RHF.
    Includes, optionally, photonic and phononic components.

    Note that if mf is a QEDHF object, then it includes a Boson object
    as the mf.qed attribute.

    pyscf.gto.Mole and pyscf.hf.RHF objects to AFQMCSystem form.
    This part is electron-only.

    Parameters
    ----------
    mol : pyscf.gto.Mole | openms.lib.boson.Boson
        The molecular object.
    mf : pyscf.scf.hf.RHF | openms.mqed.qedhf.RHF
        The mean-field object.
    use_oao: bool
        Whether to use the orthogonalized atomic orbital representation
        for the QMC Hamiltonian (default: True). If not use_oao, then
        use the molecular orbital representation.
    chol_thresh : float
        The threshold for the Cholesky decomposition of the two-electron integrals.
        Default: 1e-6
    ncomponents : int
        The number of spin components to compute separately. Default: 1
    has_photon: bool
        Whether the system is coupled to an optical cavity.
    has_phonon: bool
        Whether to include electron-phonon coupling.
    dress_eri_dse : bool
        For photon-copuled calculations, whether to dress the 2-e integrals
        with the dipole self-energy. Default: False
    bilinear_scheme : int
        The scheme to decompose the bilinear coupling:
            1 (generates 3 * nmodes auxiliary fields),
            2 (generates 2 * nmodes auxiliary fields).
        Default: 2
    verbose: int (Optional)
        The verbosity level (1 lowest, 5 for debugging). If none provided, use mol.verbose
    stdout: int (Optional)
        The standard output buffer. If none provided, use mol.stdout

    Optional parameters (bosonic)
    -----------------------------
    photon_gauge : str
        The gauge for the photon coupling. UNUSED; we assume length gauge.
    long_wave_approx : bool
        Whether to use the long-wavelength approximation for the photons.
        UNUSUED, but currently assumed True.
    nmode_photon : int
        The number of photon modes.
    dim_fock_photon : int
        The Fock space dimension of the photon modes
    omega_photon : np.ndarray
        The photon frequencies; if an array, omega_photon.shape = (nmode_photon,)
    lambda_dot_mu : np.ndarray
        The photon coupling in the length gauge; lambda_dot_mu.shape = (nmode_photon, n_ao, n_ao)
    phonon_type : str
        What type of phonons to compute. For now, only "holstein" is supported.
    dim_fock_phonon : int
        The Fock space dimension of the phonon modes
    coupling_phonon: float | np.ndarray
        The Holstein coupling; if an array, coupling_phonon.shape = (n_ao,)
    omega_phonon : float | np.ndarray
        The phonon frequencies; if an array, omega_phonon.shape = (n_ao,)

    """
    from pyscf.ao2mo import restore

    # Read the electron-only quantities and orthogonalize hcore
    n_ao = mol.nao_nr()
    ovlp = orth_overlap(mf.get_ovlp()) if use_oao else mf.mo_coeff
    hcore = ovlp.T @ mf.get_hcore() @ ovlp
    eri = restore(1, mf._eri, n_ao)

    if verbose is None:
        verbose = mol.verbose

    if stdout is None:
        stdout = mol.stdout

    # Read the photonic quantities
    if has_photon:
        if isinstance(mol, Boson):
            lambda_dot_mu = mol.gmat.copy()
            freq_photon = mol.boson_freq
            nmodes_photon = lambda_dot_mu.shape[0]
            dim_fock_photon = max(mol.nboson_states)
            pass
        elif omega_photon is None:
            raise ValueError("Either supply a Boson object or photonic parameters!")
        else:
            if omega_photon is float:
                freq_photon = np.repeat(omega_photon, nmodes_photon)
        hcore = dress_h1e_photon(hcore, ovlp, lambda_dot_mu)
        gmat_photon = np.einsum(
            "n, npq -> npq", np.sqrt(0.5 * omega_photon), lambda_dot_mu
        )

        if dress_eri_dse:
            eri += np.einsum("npq, nrs -> pqrs", lambda_dot_mu, lambda_dot_mu)
            nfields_dse = 0
        else:
            L_dse, nfields_dse = decompose_tensor_dse(
                lambda_dot_mu=lambda_dot_mu, S=ovlp
            )

    # Decompose the two-electron integrals, which are optionally dressed by DSE
    ltensor, nbarefields = decompose_tensor_eri(eri=eri, S=ovlp, thresh=chol_thresh)
    if has_photon and not dress_eri_dse:
        ltensor = np.concatenate(ltensor, L_dse, axis=0)
        nbarefields += nfields_dse

    # Read the phonon quantities
    if has_phonon:
        if phonon_type is not None:
            if phonon_type.lower() == "holstein":
                rdm = mf.make_rdm1()
                nmodes_phonon, freq_phonon, gmat_phonon = holstein_coupling(
                    rdm1=rdm, g=coupling_phonon, omega=omega_phonon
                )
            else:
                raise NotImplementedError("Only Holstein phonons implemented for now.")

    # Combine photon and phonon quantities
    nfields_bilinear = 0
    nmodes = dim_fock = boson_freq = geb = chol_bilinear = None
    if has_photon or has_phonon:
        nmodes, dim_fock, boson_freq, geb = combine_boson(
            nmodes_photon,
            nmodes_phonon,
            freq_photon,
            freq_phonon,
            gmat_photon,
            gmat_phonon,
            dim_fock_photon,
            dim_fock_phonon,
        )
        chol_bilinear, nfields_bilinear = decompose_tensor_bilinear(
            g_bil=geb, S=ovlp, scheme=bilinear_scheme
        )
        ltensor = np.concatenate(ltensor, chol_bilinear[0], axis=0)
    nfields = nbarefields + nfields_bilinear

    h1e = np.array([hcore for _ in range(ncomponents)])

    qmcsys = AFQMCSystem(
        n_ao=mol.nao_nr(),
        h1e=h1e,
        ovlp=ovlp,
        ltensor=ltensor,
        nbarefields=nbarefields,
        nfields=nfields,
        nelec=mol.nelec,
        nelectron=mol.nelectron,
        spin=mol.spin,
        nuc_energy=mf.energy_nuc(),
        nmodes=nmodes,
        dim_fock=dim_fock,
        boson_freq=boson_freq,
        chol_bilinear=chol_bilinear,
        geb=geb,
        verbose=verbose,
        stdout=stdout,
    )

    qmcmf = QMCMeanField(
        h1e=h1e,
        ovlp=ovlp,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        eri=eri,
        E_mf=mf.energy_tot(),
    )

    return qmcsys, qmcmf
