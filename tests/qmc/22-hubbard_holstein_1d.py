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
from openms.qmc.afqmc import AFQMC
from openms.qmc.tools import analysis_autocorr

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


def run_afqmc(
    mol: gto.Mole,
    mf: scf.hf.RHF,
    t_max: float = 10.0,
    dt: float = 0.005,
    num_walker: int = 500,
    E_scheme: str = "hybrid",
    verbose: int = 1,
):
    qmc = AFQMC(
        mol,
        mf=mf,
        dt=dt,
        total_time=t_max,
        num_walkers=num_walker,
        energy_scheme=E_scheme,
        verbose=verbose,
    )

    _, energies = qmc.kernel()
    output = analysis_autocorr(energies)
    E_qmc = output["etot"][0]
    std = output["etot_error"][0]

    return E_qmc, std


class TestHubbardHolstein1d(unittest.TestCase):
    def test_hubbard_holstein_1d(self):
        verbose = 1

        #### Reference energy, via Block2
        E_dmrg = -2.875942809002934

        #### Lattice parameters
        L = 4
        dim = 1
        shape = "square"
        periodic = False

        #### Hubbard parameters
        t = -1.0
        U = 2.0
        N = 4
        g = 0.5
        nspin = 1
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

        #### Set up the Holstein part

        #### QMC parameters
        verbose = 3
        dt = 0.01
        t_max = 20.0
        num_walker = 5000
        E_scheme = "hybrid"

        E_qmc, std = run_afqmc(mol, mf, t_max, dt, num_walker, E_scheme, verbose)
        print(f"HF energy       = {E_hf:.6f}")
        print(f"AFQMC energy    = {E_qmc:.6f} ± {std:.6f}")
        print(f"DMRG energy     = {E_dmrg:.6f}")

        err = abs(E_qmc - E_dmrg)

        self.assertLess(err, 0.1, msg=f"|E_qmc - E_dmrg| = {err:.6e}")


if __name__ == "main":
    unittest.main()
