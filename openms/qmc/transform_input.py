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
from scipy.linalg import svd
from dataclasses import dataclass
from pyscf.gto import Mole as Mol
from pyscf.scf.hf import RHF as HF
from pyscf.lo.orth import lowdin
from openms.mqed.qedhf import RHF as QEDHF
from openms.qmc.tools import bilinear_decomposition


@dataclass
class AFQMCInput:
    r"""Input transformed for AFQMC.
    Note that this class doesn't do any processing; for example,
    the one- and two-electron integrals should already be shifted by the
    dipole self-energy.

    Assumes the form gmat[i, :, :] * (a^+[i] + a[i]) for mode i's
    bilinear coupling: that is, the electronic operator gmat couples to the
    boson displacement operator, and all constants (such as the cavity QED
    sqrt(omega[i]/2)) are included in gmat.

    Attributes
    ----------
    n_ao: int
        The number of orbitals/basis functions for the electronic part.
    h_1e : np.ndarray
        The one-electron integrals; h_1e.shape = (n_ao, n_ao)
    h_2e : np.ndarray
        The two-electron integrals; h_2e.shape = (n_ao, n_ao, n_ao, n_ao)
    elec_wf : np.ndarray
        The electronic trial wavefunction; elec_wf.shape = (n_ao, )
    nmode : int (Optional)
        The number of boson modes.
    dim_fock : int (Optional)
        The dimension of the Fock space for each boson mode.
    gmat : np.ndarray (Optional)
        The electron-boson bilinear coupling matrix.
        gmat.shape = (nmode, n_ao, n_ao)
    boson_freq : np.ndarray (Optional)
        The boson frequencies. boson_freq.shape = (nmode,)
    boson_wf : np.ndarary (Optional)
        The bosons' trial wavefunction; boson_wf.shape = (nmode, dim_fock)

    """

    n_ao: int
    h_1e: np.ndarray
    h_2e: np.ndarray
    ovlp: np.ndarray
    elec_wf: np.ndarray
    nmode: int | None = None
    dim_fock: int | None = None
    gmat: np.ndarray | None = None
    boson_freq: np.ndarray | None = None
    boson_wf: np.ndarray | None = None


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
    return lowdin(S) if oao.lower() == "oao" else S


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
) -> tuple[np.ndarray, int]:
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


def holstein_phonon_nocavity(
    rdm1: np.ndarray,
    g: float | np.ndarray,
    omega: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Couple an electronic mean-field calculation mf to Holstein phonons for AFQMC.
    TODO: Add support for a photonic cavity as well.

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
