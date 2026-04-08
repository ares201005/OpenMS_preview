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
# Created:          2025-03-11
# Last modified:    2026-03-20

import unittest
import numpy as np
from pyscf import gto, scf, ao2mo
from openms.models.hubbard import Hubbard
from openms.qmc.transform_input import pyscf_openms_to_qmc, AFQMCSystem, QMCMeanField
from openms.qmc.afqmc import AFQMC
from openms.qmc.tools import analysis_autocorr

np.set_printoptions(precision=3)

r"""The Hubbard-Holstein model in OpenMS."""


def setup_mol_mf(hub: Hubbard, verbose: int = 1):
    mol = gto.M(verbose=verbose)
    mol.nelectron = hub.N
    mol.tot_electrons = lambda *args: hub.N
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: hub.L_tot
    mol.spin = hub.S_z
    mol.energy_nuc = lambda *args: 0.0

    mf = scf.RHF(mol)
    mf.max_cycle = 2000
    mf.get_hcore = lambda *args: hub.tmat
    mf.get_ovlp = lambda *args: hub.ovlp
    mf._eri = ao2mo.restore(
        1, hub.Umat, hub.L_tot
    )  # 8-fold symmetry if real, symmetric U(x, x') = U(x', x)
    mf.init_guess = "1e"

    return mol, mf


def setup_qmc(
    mol: gto.Mole | AFQMCSystem,
    mf: scf.hf.RHF | QMCMeanField,
    t_max: float = 10.0,
    dt: float = 0.005,
    num_walker: int = 500,
    E_scheme: str = "hybrid",
    verbose: int = 1,
) -> AFQMC:
    return AFQMC(
        mol,
        mf=mf,
        dt=dt,
        total_time=t_max,
        num_walkers=num_walker,
        energy_scheme=E_scheme,
        verbose=verbose,
    )


def run_qmc(qmc: AFQMC) -> tuple[float, float]:
    _, energies = qmc.kernel()
    output = analysis_autocorr(energies)
    E_qmc = output["etot"][0]
    std = output["etot_error"][0]

    return E_qmc, std


class TestHubbardHolstein1d(unittest.TestCase):
    def test_hubbard_holstein_1d(self):
        verbose = 1

        #### Reference energy, via Block2
        # E_noph = -2.875942809002934  # t = -1, U = 2, no phonon
        E_noph = -4.462146352704201  # t = -1, U = 0.01, no phonon
        # E_dmrg = (
        #    -3.039521251791332
        # )  # t = -1, U = 2, g = 0.1, omega = 0.25; dim_fock = 11
        # E_dmrg = (
        #    -2.876743192368235
        # )  # t = -1, U = 2, L = N = 4, g = 0.001, omega = 0.005; dim_fock = 11
        E_dmrg = (
            -4.462947370880658
        )  # t = -1, U = 0.01, L = N = 4, g = 0.001, omega = 0.005, dim_fock = 11

        #### Lattice parameters
        L = 4
        dim = 1
        shape = "square"
        periodic = False

        #### QMC parameters
        verbose = 1
        dt = 0.001
        t_max = 20.0
        num_walker = 2000
        E_scheme = "hybrid"

        #### Hubbard parameters
        t = -1.0
        U = 0.01
        N = 4
        nspin = 1

        #### Holstein parameters
        g = 0.001
        omega = 0.005
        dim_fock = 11

        #### Set up the Hubbard model
        hub = Hubbard(
            t=t,
            U=U,
            N=N,
            nspin=nspin,
            verbose=verbose,
            L=L,
            dim=dim,
            periodic=periodic,
            shape=shape,
        )

        mol, mf = setup_mol_mf(hub, verbose)
        E_hf = mf.kernel()

        #### Previous version of QMC
        qmc_old = setup_qmc(
            mol=mol,
            mf=mf,
            t_max=t_max,
            dt=dt,
            num_walker=num_walker,
            E_scheme=E_scheme,
            verbose=verbose,
        )

        #### New version of QMC
        qmcmol, qmcmf = pyscf_openms_to_qmc(
            mol=mol,
            mf=mf,
            use_oao="True",
            has_phonon=True,
            phonon_type="holstein",
            bilinear_scheme=2,
            coupling_phonon=g,
            omega_phonon=omega,
            dim_fock_phonon=dim_fock,
        )

        qmc_new = setup_qmc(
            mol=qmcmol,
            mf=qmcmf,
            t_max=t_max,
            dt=dt,
            num_walker=num_walker,
            E_scheme=E_scheme,
            verbose=verbose,
        )

        print("Holstein quantities debug")
        # print(f"mean-field density matrix: =\n {mf.make_rdm1()}")
        print("old:")
        print(f"ltensor: shape = {qmc_old.ltensor.shape}")
        print(f"norm = {np.linalg.norm(qmc_old.ltensor)}")
        print("new:")
        print(f"geb = \n {qmcmol.geb}\n")
        print(f"elec ltensor: shape = {qmcmol.ltensor[0:4, :, :].shape}\n")
        print(f"{np.linalg.norm(qmcmol.ltensor[0:4, :, :])}")
        print(f"full ltensor: shape = {qmcmol.ltensor.shape}\n")
        print(f"{np.linalg.norm(qmcmol.ltensor)}")

        #### Run QMC
        E_old, std_old = run_qmc(qmc_old)
        E_new, std_new = run_qmc(qmc_new)

        print(f"HF energy    (no phonon)    = {E_hf:.6f}")
        print(f"AFQMC energy (no phonon)    = {E_old:.6f} ± {std_old:.6f}")
        print(f"DMRG energy  (no phonon)    = {E_noph:.6f}")
        print(f"AFQMC energy (w/ phonon)    = {E_new:.6f} ± {std_new:.6f}")
        print(f"DMRG energy  (w/ phonon)    = {E_dmrg:.6f}")

        err = abs(E_new - E_dmrg)

        self.assertLess(err, 0.001, msg=f"|E_qmc - E_dmrg| = {err:.6e}")


if __name__ == "main":
    unittest.main()
