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
# Created:          2025-10-01
# Last modified:    2026-03-11

import numpy as np
from pyscf.gto import Mole as PySCFMole
from pyscf.scf.hf import RHF as HF
from openms.mqed.qedhf import RHF as QEDHF

# from openms.mqed.scqedhf import RHF as SCQEDHF # these inherit from QEDHF
# from openms.mqed.vtqedhf import RHF as VTQEDHF
from openms.lib import logger


class FermiBoseSystem:
    r"""A general container class for fermions coupled to
    noninteracting bosons: phonons, photons, or both.

    We can read QEDHF output or build the photonic input from scratch,
    and include phonons either way.

    In other words, in the context of quantum Monte Carlo,
    this class should be thought of as 'downstream' from the mean-field
    calculation to prepare the trial state.


    TODO: Decide whether to fold the dipole self-energy into the
    one-electron integrals here! 2026-03-13: leaning toward doing so.

    Parameters
    ----------
    mol             : pyscf.gto.Mole (or similar)
        The fermionic part of the system.
    mf              : pyscf.scf.hf.RHF, openms.mqed.qedhf.QEDHF
        The mean-field calculation.
    photon_gauge    : str
        The gauge in which to compute the photons.
        Currently only "dipole" == "length" is supported.
    phonon_type     : str
        The type of electron-phonon interaction.
        Currently only "holstein" is supported.
    freq_photon     : np.ndarray | None
        The photon frequencies; freq_photon.shape = (nphoton,)
    freq_phonon     : np.ndarray | None
        The phonon frequencies; freq_photon.shape = (nphonon,)
    dim_fock_photon : int | None
        The number of Fock states in each photonic mode.
    dim_fock_phonon : int | None
        The number of Fock states in each phononic mode.
    coupling_photon : np.ndarray | None
        The photonic coupling matrix; coupling_photon.shape = (nphoton, nao, nao),
        where nao is the number of fermionic basis functions (atomic orbitals).
    coupling_phonon : np.ndarray
        The phononic coupling matrix; coupling_phonon.shape = (nphonon, nao, nao),
        where nao is the number of fermionic basis functions (atomic orbitals).
    wf_photon       : np.ndarray | None
        A photonic wavefunction can be specified:
        wf_photon.shape = (nphoton, nfock_photon)
    wf_phonon       : np.ndarray | None
        A phononic wavefunction can be specified:
        wf_phonon.shape = (nphonon, nfock_phonon)
    include_zpe     : bool
        Whether to include the bosonic zero-point energy. Default: False

    Attributes
    ----------
    mol            : pyscf.gto.Mole-like
        The fermionic object passed through.
    mf              : pyscf.scf.hf.RHF, openms.mqed.qedhf.QEDHF
        The mean-field calculation, passed through.
    nmodes_photon   : int
        The number of photon modes.
    nmodes_phonon   : int
        The number of phonon modes.
    nmodes          : int
        The total number of modes; = nmodes_photon + nmodes_phonon
    nao             : float
        The number of fermionic basis states
        (atomic orbitals, lattice sites, &c.)
    dim_fock_photon : int
        The number of Fock states in each photonic mode.
    dim_fock_phonon : int
        The number of Fock states in each phononic mode.
    dim_fock        : int
        The Fock space dimension of each bosonic mode;
        the larger of nfock_photon and nfock_phonon
    nboson_states   : list[int] (TO BE DEPRECATED)
        The Fock space dimension of each bosonic mode.
    freq_photon     : np.ndarray
        The photon frequencies; freq_photon.shape = (nphoton,)
    freq_phonon     : np.ndarray
        The phonon frequencies; freq_photon.shape = (nphonon,)
    omega           : np.ndarray
        The combined bosonic frequencies; omega.shape = (nmodes, )
    gmat            : np.ndarray
        The combined fermion-boson coupling matrices;
        gmat.shape = (nmodes, nao, nao)
    dse             : np.ndarray
        The dipole self-energy to add to the one-electron integrals;
        dse.shape = (nao, nao)
    photon_psi      : np.ndarray
        The photonic wavefunction;
        photon_psi.shape = (nmodes_photon, dim_fock_photon)
    phonon_psi      : np.ndarray
        The phononic wavefunction;
        phonon_psi.shape = (nmodes_phonon, dim_fock_phonon)
    boson_psi       : np.ndarray
        The full bosonic wavefunction; boson_psi.shape = (nmodes, dim_fock)
    wf_kind_photon  : str
        If computing the photon wavefunction, what kind?
        Options: "thermal", "equal_weight", "vacuum"
    wf_kind_phonon  : str
        If computing the phonon wavefunction, what kind?
        Options: "thermal", "equal_weight", "vacuum"
    zpe             : float
        The zero-point energy shift, :math:`\sum_\alpha \omega_\alpha / 2`

    Notes
    -----
    Inherits from lib.boson.Boson for reasons of compatibility:
        code in openms.qmc checks whether the system is a Boson instance
        to decide whether to do coupled fermion-boson quantum Monte Carlo.
    """

    def __init__(
        self,
        mol: PySCFMole,
        mf: HF | QEDHF | None,
        freq_photon: np.ndarray | None,
        freq_phonon: np.ndarray | None,
        dim_fock_photon: int | None,
        dim_fock_phonon: int | None,
        coupling_photon: np.ndarray | None,
        coupling_phonon: np.ndarray | None,
        wf_photon: np.ndarray | None,
        wf_phonon: np.ndarray | None,
        photon_gauge: str = "length",
        phonon_type: str = "holstein",
        include_zpe: bool = False,
    ):

        # Fermion-only quantities
        self.mol = mol
        self.nao = mol.nao_nr()

        # The mean-field object
        if mf is not None:
            self.mf = mf
        else:
            logger.note(
                self,
                "No mean-field object specified. "
                + "Please ensure this is what you want!",
            )

        # Photon quantities
        if isinstance(mf, QEDHF):
            # QEDHF object contains the photonic information
            self.nmodes_photon = mf.qed.nmodes
            self.dim_fock_photon = max(mf.qed.nboson_states)
            self.freq_photon = mf.qed.omega
            self.coupling_photon = mf.qed.gmat  # DOESN'T include sqrt(omega/2)
        elif freq_photon is not None:
            if photon_gauge.lower() != "length" or photon_gauge.lower() != "dipole":
                raise ValueError("Only dipole/length gauge photons supported for now!")

            self.freq_photon = freq_photon
            self.nmodes_photon = freq_photon.shape[0]

            if dim_fock_photon is not None:
                self.dim_fock_photon = dim_fock_photon
            else:
                raise ValueError("Photon Fock space dimension not specified!")

            if coupling_photon is not None:
                self.coupling_photon = coupling_photon
            else:
                raise ValueError("Electron-photon coupling not specified!")

            # Dipole self-energy!
            self.dse = 0.5 * np.einsum(
                "apq, aqs -> ps",
                self.coupling_photon,
                self.coupling_photon,
                optimize=True,
            )
        else:
            self.nmodes_photon = 0

        # Phonon quantities
        if freq_phonon is not None:
            if phonon_type.lower() != "holstein":
                raise ValueError("Only Holstein phonons supported for now!")

            self.freq_phonon = freq_phonon
            self.nmodes_phonon = freq_phonon.shape[0]

            if dim_fock_phonon is not None:
                self.dim_fock_phonon = dim_fock_phonon
            else:
                raise ValueError("Phonon Fock space dimension not specified!")

            if coupling_phonon is not None:
                self.coupling_phonon = coupling_phonon
            else:
                raise ValueError("Electron-phonon coupling not specified!")
        else:
            self.nmodes_phonon = 0

        # Combined bosonic quantities
        self.nmodes = self.nmodes_photon + self.nmodes_phonon
        logger.note(self, f"There are {self.nmodes} bosonic modes.")
        if self.nmodes == 0:
            logger.note(self, "No bosonic modes! Please ensure this is what you want.")
        self.dim_fock = max(self.dim_fock_photon, self.dim_fock_phonon)
        logger.debug(
            self, f"dim_fock = max(dim_fock_photon, dim_fock_phonon) = {self.dim_fock}"
        )

        self.boson_freq = np.concat(self.freq_photon, self.freq_phonon, axis=0)
        self.gmat = np.concat(self.coupling_photon, self.coupling_phonon, axis=0)

        # Zero-point energy
        if include_zpe:
            self.zpe = 1.0
        else:
            self.zpe = 0.0
