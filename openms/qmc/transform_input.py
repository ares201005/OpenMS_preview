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
from dataclasses import dataclass
from pyscf.gto import Mole as Mol
from pyscf.scf.hf import RHF as HF
from openms.mqed.qedhf import RHF as QEDHF


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


def holstein_phonon_nocavity(
    mol: Mol,
    rdm1: np.ndarray,
    g: float | np.ndarray,
    omega: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Couple an electronic mean-field calculation mf to Holstein phonons for AFQMC.
    TODO: Add support for a photonic cavity as well.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The PySCF molecule object.
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
    nao = mol.nao_nr()

    nmode = mol.nao_nr()  # one Holstein mode per site/AO

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
