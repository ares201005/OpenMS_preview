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
# Last modified:    2026-03-11

import unittest
import numpy as np
from pyscf import gto, scf, ao2mo
from openms.models.hubbard import Hubbard
from openms.qmc.afqmc import AFQMC
from openms.qmc.tools import analysis_autocorr

r"""Tight-binding model (Hubbard with U = 0) with AFQMC.
    Here, the RHF method is exact, so AFQMC is superfluous;
    still, we should get the right answer, since we can even
    solve analytically.
"""


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


class TestTightBind(unittest.TestCase):
    def lattice(self, dim: int = 1, periodic: bool = False):
        self.verbose = 1
        self.L = 4
        self.dim = dim
        self.periodic = periodic

    def hubbard_param(self, L: int, dim: int):
        self.t = -1.0
        self.U = 0.0
        self.N = 4**dim  # hardcode L = 4
        self.nspin = 1

    def hubbard(self):
        return Hubbard(
            t=self.t,
            U=self.U,
            N=self.N,
            nspin=self.nspin,
            verbose=1,
            L=self.L,
            dim=self.dim,
            periodic=self.periodic,
        )

    def test_tb_1d(self):
        self.lattice(dim=1, periodic=False)
        self.hubbard_param(L=4, dim=1)
        hub = self.hubbard()
        mol, mf = setup_mol_mf(hub, self.verbose)
        E_hf = mf.kernel()
        print(f"HF energy   = {E_hf:.6f}")

    def test_tb_1d_periodic(self):
        self.lattice(dim=1, periodic=True)
        self.hubbard_param(L=4, dim=1)
        hub = self.hubbard()
        mol, mf = setup_mol_mf(hub, self.verbose)
        E_hf = mf.kernel()
        print(f"HF energy   = {E_hf:.6f}")

    def test_tb_2d(self):
        self.lattice(dim=2, periodic=False)
        self.hubbard_param(L=4, dim=2)
        hub = self.hubbard()
        mol, mf = setup_mol_mf(hub, self.verbose)
        E_hf = mf.kernel()
        print(f"HF energy   = {E_hf:.6f}")

    def test_tb_2d_periodic(self):
        self.lattice(dim=2, periodic=True)
        self.hubbard_param(L=4, dim=2)
        hub = self.hubbard()
        mol, mf = setup_mol_mf(hub, self.verbose)
        E_hf = mf.kernel()
        print(f"HF energy   = {E_hf:.6f}")


if __name__ == "main":
    unittest.main()
