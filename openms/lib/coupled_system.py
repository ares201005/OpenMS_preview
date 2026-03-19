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
from openms.lib.boson import Boson
from openms.lib import logger


class CoupledSystem(Boson):
    r"""A general container class for a fermionic system coupled
    to phonons, photons, or both.

    The first nphoton modes are assumed to be photonic; the remainder,
    of which there are nphonon, are assumed to be phononic.

    We assume that all preprocessing has been done beforehand: the
    coupling matrices are passed through as given.

    In other words, in the context of quantum Monte Carlo,
    this class should be thought of as 'downstream' from the mean-field
    calculation to prepare the trial state.

    Parameters
    ----------
    mol             : pyscf.gto.Mole (or similar)
        The fermionic part of the system.
    model_system    : bool
        True if the system is described by a model Hamiltonian
    nphoton         : int
        The number of photon modes.
    nphonon         : int
        The number of phonon modes.
    nmodes          : int
        The total number of modes (must be equal to nphoton + nphonon).
    nfock_photon   : int
        The number of Fock states in each photonic mode
    nfock_phonon   : int
        The number of Fock states in each phononic mode
    freq_photon     : np.ndarray
        The photon frequencies (length == nphoton).
    freq_phonon     : np.ndarray
        The phonon frequencies (length == nphonon).
    polariz_photon  : np.ndarray
        The photonic polarization vectors (shape = (3, nphoton)).
    polariz_phonon  : np.ndarray
        The phononic polarization vectors (shape = (3, nphonon)).
    coupling_photon : np.ndarray
        The photonic coupling matrices (shape = (nphoton, nao, nao)).
        Here nao is the number of fermionic basis functions (atomic orbitals).
    coupling_phonon : np.ndarray
        The phononic coupling matrices (shape = (nphonon, nao, nao)).
        Here nao is the number of fermionic basis functions (atomic orbitals).
    wf_photon       : np.ndarray
        A photonic wavefunction can be specified:
        wf_photon.shape = (nphoton, nfock_photon)
    wf_phonon       : np.ndarray
        A phononic wavefunction can be specified:
        wf_phonon.shape = (nphonon, nfock_phonon)
    incl_zpe        : bool
        If true, include zero-point energy for the bosonic modes.

    Attributes
    ----------
    _mol            : pyscf.gto.Mole-like
        The fermionic object, renamed slightly.
    nao             : float
        The number of fermionic basis states (atomic orbitals, lattice sites, &c.)
    dim_fock        : int
        The Fock space dimension of each bosonic mode;
        the larger of nfock_photon and nfock_phonon
    nboson_states   : list[int] (DEPRECATED)
        The Fock space dimension of each bosonic mode.
    omega           : np.ndarray
        The combined bosonic frequencies; omega.shape = (nmodes, )
        (length == nphoton + nphoton == nmodes).
    vec             : np.ndarray
        The combined bosonic mode polarizations; vec.shape = (nmodes, 3)
        a NumPy array of length 3.
    gmat            : np.ndarray
        The combined fermion-boson coupling matrices; gmat.shape = (nmodes, nao, nao)
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
        mol,
        model_system=False,
        nphoton=None,
        nphonon=None,
        nmodes=None,
        nfock_photon=None,
        nfock_phonon=None,
        freq_photon=None,
        freq_phonon=None,
        polariz_photon=None,
        polariz_phonon=None,
        coupling_photon=None,
        coupling_phonon=None,
        wf_photon=None,
        wf_phonon=None,
        wf_kind_photon=None,
        wf_kind_phonon=None,
        incl_zpe=False,
        *args,
        **kwargs,
    ):

        super().__init__(mol, omega=[0.0], vec=[[1.0, 0.0, 0.0]], *args, **kwargs)

        # Fermionic quantities
        self._set_fermion(mol)

        if model_system:
            logger.note(self, "This is a model system!")
        self.model_system = model_system

        # Number of bosonic modes
        nphoton = 0 if nphoton is None else nphoton
        nphonon = 0 if nphonon is None else nphonon
        nmode_msg = (
            "The number of bosonic modes must be equal to "
            + "the sum of the number of photonic and phononic modes!"
        )
        if nmodes is not None:
            assert nmodes == nphoton + nphonon, nmode_msg

        self.nphoton = nphoton
        self.nphonon = nphonon
        self.nmodes = self.nphoton + self.nphonon

        # Set photonic and phononic quantities
        # self.nfock_photon = self._set_dim_fock(self.nphoton, nfock_photon)
        self.nfock_phonon = self._set_dim_fock(self.nphonon, nfock_phonon)

        self.freq_photon = self._set_frequencies(self.nphoton, freq_photon)
        self.freq_phonon = self._set_frequencies(self.nphonon, freq_phonon)

        self.polariz_photon = self._set_polarization(self.nphoton, polariz_photon)
        self.polariz_phonon = self._set_polarization(self.nphonon, polariz_phonon)

        self.coupling_photon = self._set_coupling(
            self.nphoton, self.nao, coupling_photon
        )
        self.coupling_phonon = self._set_coupling(
            self.nphonon, self.nao, coupling_phonon
        )

        self.wf_kind_photon = (
            "thermal" if wf_kind_photon is None else wf_kind_photon.lower()
        )
        self.wf_kind_phonon = (
            "thermal" if wf_kind_phonon is None else wf_kind_phonon.lower()
        )

        self.wf_photon = self._set_boson_wavefunction(
            wf_photon, self.nphoton, self.nfock_photon, kind=self.wf_kind_photon
        )
        self.wf_phonon = self._set_boson_wavefunction(
            wf_phonon, self.nphonon, self.nfock_phonon, kind=self.wf_kind_phonon
        )

        # Combined quantities: photonic degrees of freedom come first
        self.nboson_states = self.nfock_photon + self.nfock_phonon
        self.omega = self.boson_freq = self.freq_photon + self.freq_phonon
        self.vec = self.polariz_photon + self.polariz_phonon
        self.gmat = self._combine_coupling(self.coupling_photon, self.coupling_phonon)
        # self.gmat = self.coupling_photon + self.coupling_phonon

        self.boson_psi = self.wf_photon + self.wf_phonon

        # Zero-point energy, if any
        self.zpe = 0.0
        if incl_zpe is not None:
            if incl_zpe:
                logger.note(self, "Including boson zero-point energy.")
            self.incl_zpe = incl_zpe
            self.zpe = sum([0.5 * omega for omega in self.boson_freq])

        """
        # Photonic quantities
        if self.nphoton > 0:
            # Frequencies
            if isinstance(freq_photon, list):
                freq_photon = np.asarray(freq_photon, dtype = float)
            if not isinstance(freq_photon, (float, np.ndarray)):
                err_msg = f"Specify the photon frequencies as a float, " + \
                f"a list of floats, or a NumPy array of floats."
                logger.error(self, err_msg)
                raise ValueError(err_msg)
            else:
                if isinstance(freq_photon, float):
                    freq_list = [freq_photon for _ in range(self.nphoton)]
                    self.freq_photon = np.asarray(freq_list, dtype = float)
                else:
                    self.freq_photon = freq_photon.copy()

            # Fock state dimensions
            if nfock_photon is not None:
                if isinstance(nfock_photon, int):
                    self.nfock_photon = [nfock_photon] * self.nphoton
                elif isinstance(nfock_photon, np.ndarray):
                    raise ValueError("Give the Fock state dimension as an integer or a list of integers!")
                else:
                    assert len(nfock_photon) == self.nphoton, (
                    "Specify the same number of Fock state dimensions as photon modes.")
                    self.nfock_photon = nfock_photon
            else:
                self.nfock_phonon = [1] * self.nphonon

            # Polarizations
            if polariz_photon is not None:
                if isinstance(polariz_photon, list):
                    polariz_photon = np.asarray(polariz_photon, dtype = float)
                if (not isinstance(polariz_photon, np.ndarray)
                    or polariz_photon.shape != (self.nphoton, 3)):
                    err_msg = f"Photon polarizations should have dimensions " + \
                              f"(len(nphoton), 3)."
                    logger.error(self, err_msg)
                    raise ValueError(err_msg)
            else:
                # Default: all modes polarize along x
                self.polariz_photon = [np.array([1., 0., 0.]) 
                                       for _ in range(self.nphoton)]
            
            # Electron-photon coupling matrices
            if coupling_photon is not None:
                if isinstance(coupling_photon, list):
                    coupling_photon = np.asarray(coupling_photon, dtype = float)
                assert coupling_photon.shape == (self.nmodes, self.nao, self.nao), (
                    "The size of the coupling matrix must be "
                        "(nmodes x nao x nao)!")
                
                # We do not preprocess the coupling matrices
                self.coupling_photon = coupling_photon.copy()
            else:
                # Assume no coupling
                self.coupling_photon = np.zeros(self.nphoton, 
                                                   self.mol.nao, self.mol.nao)


        # Phononic quantities
        if self.nphonon > 0:

            # Frequencies
            if isinstance(freq_phonon, list):
                freq_phonon = np.asarray(freq_phonon, dtype = float)
            if not isinstance(freq_phonon, (float, np.ndarray)):
                err_msg = f"Specify the phonon frequencies as a float, " + \
                f"a list of floats, or a NumPy array of floats."
                logger.error(self, err_msg)
                raise ValueError(err_msg)
            else:
                if isinstance(freq_phonon, float):
                    freq_list = [freq_phonon for _ in range(self.nphonon)]
                    self.freq_phonon = np.asarray(freq_list, dtype = float)
                else:
                    self.freq_phonon = freq_phonon.copy()

            # Fock state dimensions
            if nfock_phonon is not None:
                if isinstance(nfock_phonon, int):
                    self.nfock_phonon = [nfock_phonon] * self.nphonon
                elif isinstance(nfock_phonon, np.ndarray):
                    raise ValueError("Give the Fock state dimension as an integer or a list of integers!")
                else:
                    assert len(nfock_phonon) == self.nphonon, (
                    "Specify the same number of Fock state dimensions as phonon modes.")
                    self.nfock_phonon = nfock_phonon
            else:
                self.nfock_phonon = [1] * self.nphonon

            # Polarizations
            if polariz_phonon is not None:
                if isinstance(polariz_phonon, list):
                    polariz_phonon = np.asarray(polariz_phonon, dtype = float)
                if (not isinstance(polariz_phonon, np.ndarray)
                    or polariz_phonon.shape != (self.nphonon, 3)):
                    err_msg = f"phonon polarizations should have dimensions " + \
                              f"(len(nphonon), 3)."
                    logger.error(self, err_msg)
                    raise ValueError(err_msg)
            else:
                # Default: all modes polarize along x
                self.polariz_phonon = [np.array([1., 0., 0.]) 
                                       for _ in range(self.nphonon)]
                
            # Electron-phonon coupling matrices
            if coupling_phonon is not None:
                if isinstance(coupling_phonon, list):
                    coupling_phonon = np.asarray(coupling_phonon, dtype = float)
                assert coupling_phonon.shape == (self.nmodes, self.nao, self.nao), (
                    "The size of the coupling matrix must be "
                        "(nmodes x nao x nao)!")
                
                # We do not preprocess the coupling matrices
                self.coupling_phonon = coupling_phonon.copy()
            else:
                # Assume no coupling
                self.coupling_phonon = np.zeros(self.nphonon, 
                                                   self.mol.nao, self.mol.nao)

        # Combined quantities
        # jzw 2025-10-22: run _set_*() even if nmode == 0; they'll return
        #   empty lists if so, and we can just concat regardless
        if nphoton == 0 and nphonon > 0:
            self.omega          = self.freq_phonon
            self.vec            = self.polariz_phonon
            self.nboson_states  = self.nfock_phonon
            self.gmat           = self.coupling_phonon
        elif nphonon == 0 and nphoton > 0:
            self.omega          = self.freq_photon
            self.vec            = self.polariz_photon
            self.nboson_states  = self.nfock_photon
            self.gmat           = self.coupling_photon
        elif nphoton > 0 and nphonon > 0:
            self.omega          = np.concat((self.freq_photon,
                                                self.freq_phonon))
            self.vec            = np.concat((self.polariz_photon, 
                                                self.polariz_phonon))
            self.nboson_states  = np.concat((self.nfock_photon,
                                                self.nfock_phonon))
            self.gmat           = np.concat((self.coupling_photon,
                                                self.coupling_phonon))
        else:
            self.omega = None
            self.vec = None
            self.nboson_states = None
            self.gmat = None

        # omega is to be deprecated in favor of boson_freq
        self.boson_freq     = self.omega

        # Zero-point energy, if any
        self.zpe = 0.0
        if incl_zpe is not None:
            if incl_zpe:
                logger.note("Including boson zero-point energy.")
            self.incl_zpe = incl_zpe
            self.zpe = np.sum(0.5 * self.omega)
        """

    def _set_fermion(self, mol):
        r"""Set the fermionic quantities.

        For now, we assume a PySCF Mole object. In later development,
            we will enable I/O from other electronic structure theory codes,
            and simply standardize the necessary information about the
            fermionic portion of the system here.

        """
        from pyscf import gto

        if not isinstance(mol, gto.mole.MoleBase):
            err_msg = (
                f"Parameter 'mol' is not an instance "
                + f"of MoleBase class, part of the PySCF "
                + f"quantum chemistry software package."
            )
            logger.error(self, err_msg)
            raise ValueError(err_msg)
        else:
            self._mol = mol
            self.verbose = mol.verbose
            self.stdout = mol.stdout
            self.nelectron = mol.nelectron
            self.nao = mol.nao_nr()

            self.nao_nr = mol.nao_nr
            self.spin = mol.spin
            self.nelec = mol.nelec

        return

    def _set_dim_fock(self, nmode=1, dim=None):
        r"""Set the Fock space dimensions of each mode.

        Parameters
        ----------
        nmode       : int
            The number of modes.
        dim         : int, list[int]
            The Fock space dimension for each mode; if a single integer,
            the dimension is assumed to be the same for all modes.

        Returns
        -------
        dim_fock    : list[int]
            The Fock space dimension for each mode in list form.

        """
        dim_fock = []
        if nmode > 0:
            if dim is None:
                # default: only zero excitations allowed in each mode
                dim_fock = [1] * nmode
            else:
                if isinstance(dim, int):
                    dim_fock = [dim] * nmode
                elif not isinstance(dim, list):
                    err_msg = (
                        "Give the Fock state dimension "
                        "as an integer or as a list of integers!"
                    )
                    raise ValueError(err_msg)
                else:
                    err_msg = "Specify one or nmode Fock space dimensions."
                    assert len(dim) == nmode, err_msg
                    dim_fock = dim

        return dim_fock

    def _set_frequencies(self, nmode=1, omega=1.0):
        r"""Return the list of mode frequencies.

        Arguments
        ---------
        omega       : float, list[float]
            The mode frequencies (assumed all to be identical if one float).
        nmode       : int
            The number of modes.

        Returns
        -------
        frequencies : list[float]
            The mode frequencies, len(freqs) == nmode.

        """
        frequencies = []
        if nmode > 0:
            if isinstance(omega, float):
                frequencies = [omega for _ in range(nmode)]
            else:
                assert len(omega) == nmode, "Specify one or nmode frequencies."
                frequencies = omega
        return frequencies

    def _set_polarization(self, nmode=1, pvec=None):
        r"""Return a list of mode polarization vectors.

        Parameters
        ----------
        nmode       : int
            The number of bosonic modes.
        pvec        : list[float], np.array, list[list[float]], list[np.array]
            The polarization vector(s).
            If only one vector is given, assume all modes share the
                same polarization.
            Otherwise, a list of (3-)vectors for each mode must be given;
                each can be a list or a NumPy array.

        Returns
        -------
        polarizations   : list[np.array]
            The polarization vectors cast into our standard form:
                a list of length nmode, whose elements are
                1d NumPy arrays (each of shape 3 and unit norm).
        """

        polarizations = []
        if nmode > 0:
            if pvec is None:
                # Default to polarization along x.
                polarizations = [np.array([1.0, 0.0, 0.0]) for _ in range(nmode)]
            elif isinstance(pvec, np.array):
                err_msg = "Ensure that the polarization is given as [x, y, z]."
                assert pvec.shape == (3,), err_msg

                pvec = pvec / np.linalg.norm(pvec)  # normalize
                polarizations = [pvec for _ in range(nmode)]
            elif isinstance(pvec, list):
                if len(pvec) == 3:  # one polarization passed as a list
                    pvec = np.array(pvec) / np.linalg.norm(pvec)
                    polarizations = [pvec for _ in range(nmode)]
                elif len(pvec) == nmode:  # all polarizations passed as a list
                    err_msg = "Ensure that each polarization is given as [x, y, z]."
                    polarizations = []
                    for e in pvec:  # could be of lists or of NumPy arrays
                        if isinstance(e, list):
                            assert len(e) == 3, err_msg
                            e = np.array(e) / np.linalg.norm(e)
                        elif isinstance(e, np.array):
                            assert e.shape == (3,), err_msg
                            e = e / np.linalg.norm(e)
                        else:
                            err_msg = (
                                "Specify each polarization vector "
                                + "as a list or as a NumPy array."
                            )
                            raise ValueError(err_msg)
                        polarizations.append(e)
            else:
                raise ValueError("Unexpected format for polarization vectors.")

        return polarizations

    def _set_coupling(self, nmode=1, nao=1, coupling=None):
        r"""Return the fermion-boson coupling matrices for each mode.

        Parameters
        ----------
        nmode       : int
            The number of bosonic modes.
        nao         : int
            The number of electronic basis states (e.g., atomic orbitals).
        coupling    : float, np.ndarray, list[np.ndarray]
            The fermion-boson coupling matrix.
            If input as a float g,
                assume all boson modes couple to the electron density
                with strength g (as in the Hubbard-Holstein model).
            If input as a nao x nao NumPy array,
                assume all boson modes couple with the same array.
            If input as a list of nao x nao NumPy arrays of length nmode,
                set that as the coupling.

        Returns
        -------
        coupling_fb : np.ndarray
            The fermion-boson coupling, with coupling_fb.shape = (nmode, nao, nao).

        NOTE: At the moment, this includes the coupling strength and
              the fermionic operator part of the coupling, in the AO basis.
              The bosonic operator is assumed to be the mode displacement;
              it is calculated in the openms.qmc modules and not stored
              here explicitly. This behavior may change in the future
              to enable more general coupling!
        """
        coupling_fb = None
        if nmode > 0:
            coupling_fb = np.zeros((nmode, nao, nao))
            # coupling_fb = [np.zeros((nao, nao)) for _ in range(nmode)]
            if coupling is None:
                pass
            elif isinstance(coupling, float):
                # Assume g * (c+_i c_i) * X
                for a in range(nmode):
                    for i in range(nao):
                        coupling_fb[a][i][i] = coupling
            elif isinstance(coupling, list):
                assert len(coupling) == nmode and all(
                    c.shape == (nao, nao) for c in coupling
                ), (
                    "Ensure that the list of coupling matrices is of length nmode, "
                    "and that each matrix is of shape (nao, nao)."
                )
                coupling_fb = np.asarray(coupling)
            elif isinstance(coupling, np.ndarray) and coupling.shape == (nao, nao):
                for a in range(nmode):
                    coupling_fb[a][:, :] = coupling
            elif isinstance(coupling, np.ndarray) and coupling.shape == (
                nmode,
                nao,
                nao,
            ):
                coupling_fb = coupling.copy()
            else:
                err_msg = (
                    f"Specify coupling as a float, "
                    + f"a single nao x nao NumPy array, "
                    + f"a list of such arrays of length nmode, "
                    + f"or a (nmode x nao x nao) NumPy ndarray."
                )
                raise ValueError(err_msg)

        return coupling_fb

    def _combine_coupling(self, g_photon, g_phonon):
        r"""Combine the coupling matrices
        for the photonic and phononic degrees of freedom
        into the full coupling matrix.

        Parameters
        ----------
        g_photon : numpy.ndarray
            The photonic coupling matrix; shape = (nmode_photon, nao, nao)
            if nmode_photon == 0, then g_photon is None.
        g_phonon : numpy.ndarray
            The phononic coupling matrix; shape = (nmode_phonon, nao, nao)
            if nmode_phonon == 0, then g_phonon is None.

        Returns
        -------
        gmat : numpy.ndarray
            The full coupling matrix:
            if there are no photonic modes, gmat = g_phonon,
            if there are no phononic modes, gmat = g_photon,
            if there are both, gmat = np.concat(g_photon, g_phonon),
            and if there are neither, then gmat = None.
        """

        if g_photon is None:
            if g_phonon is None:
                return None
            else:
                return g_phonon.copy()
        elif g_phonon is None:
            return g_photon.copy()
        else:
            return np.concat((g_photon, g_phonon), axis=0)

    def _set_boson_wavefunction(self, wf=None, nmode=0, dim_fock=[], kind="thermal"):
        r"""Set the bosonic wavefunction for each mode.

        Parameters
        ----------
        wf          : np.array
            If a wavefunction is provided, check that it's the right size
            and return it.
        nmode       : int
            The number of bosonic modes.
        dim_fock    : list[int]
            The Fock space dimension of each mode.
        kind        : str
            The sort of wavefunction:
                "thermal", with n excitations proportional to exp(-n);
                "equal_weight", with equal weight on each Fock state; or
                "vacuum", with only the ground state |100...0> occupied.
        """
        if wf is not None:
            pass
        else:
            wf = []
            if nmode > 0:
                if kind.lower() == "thermal":
                    wf = [wf_mode_thermal(dim_fock[a], beta=1.0) for a in range(nmode)]
                elif kind.lower() == "equal_weight":
                    pass
                elif kind.lower() == "vacuum":
                    wf = [wf_mode_vacuum(dim_fock[a]) for a in range(nmode)]
                else:
                    err_msg = "Unsupported wavefunction type!"
                    raise ValueError(err_msg)
        return wf


def wf_mode_thermal(dim_fock, beta=1.0):
    occ = np.array([i for i in range(dim_fock)])
    psi = np.exp(-beta * occ)
    return psi / np.linalg.norm(psi)


def wf_mode_equal_weight(dim_fock):
    psi = np.repeat(1.0 / np.sqrt(dim_fock), dim_fock)
    return psi


def wf_mode_vacuum(dim_fock):
    psi = np.zeros(dim_fock)
    psi[0] = 1.0
    return psi
