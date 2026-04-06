import unittest
import numpy as np
from pyscf import scf
from pyscf.gto import Mole as Mol
from pyscf.ao2mo import restore
from pyscf.scf.hf import RHF as HF
from openms.qmc.afqmc import AFQMC
from openms.qmc import transform_input
from openms.qmc.transform_input import AFQMCSystem, QMCMeanField
from openms.qmc.tools import get_mean_std
from molecules import get_mol


# System is that from openms.tests.qmc.01-qmc_rhf.py
def get_qmc_object(
    mol: Mol | AFQMCSystem,
    mf: HF | QMCMeanField,
    dt: float = 0.005,
    time: float = 6.0,
    num_walkers: int = 200,
    uhf: bool = False,
    energy_scheme: str = "hybrid",
    chol_thresh: float = 1.0e-6,
) -> AFQMC:
    r"""Note the number of walkers here is small, in order to do fast test"""
    return AFQMC(
        mol,
        mf,
        dt=dt,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        chol_thresh=chol_thresh,
        property_calc_freq=1,
        verbose=mol.verbose,
    )


# Global quantities
mean_ref = -1.13981  # FCI energy (RHF)
mean_ref_u = -1.13998  # FCI energy (UHF)
std_ref = 0.003  # standard deviation ref

bond = 1.6 * 0.5291772
basis = "sto6g"
verbose = 1
mol = get_mol(2, bond, basis=basis, verbose=verbose, name="Hchain")
mf = scf.RHF(mol)
E_hf = mf.kernel()

# AFQMC quantities
dt = 0.005
time = 6.0
num_walkers = 1  # some statistical error accumulates over walkers
uhf = False
energy_scheme = "hybrid"
chol_thresh = 1.0e-20

qmc_default = get_qmc_object(
    mol=mol,
    mf=mf,
    dt=dt,
    time=time,
    num_walkers=num_walkers,
    energy_scheme=energy_scheme,
    uhf=uhf,
    chol_thresh=chol_thresh,
)

tlist, Elist_default = qmc_default.kernel()


class TestTransformInputElectron(unittest.TestCase):
    r"""Unit tests of transform_input methods: electron-only system."""

    def test_decompose_eri(self):
        eri = restore("s1", mf._eri, mol.nao_nr())
        S = transform_input.loewdin_orth(mf.get_ovlp())
        L_eri, n_eri = transform_input.decompose_tensor_eri(eri, S, thresh=1e-20)

        np.testing.assert_equal(n_eri, qmc_default.nbarefields)
        np.testing.assert_allclose(L_eri, qmc_default.ltensor, rtol=1e-7)


class TestQMCWithTransformedInput(unittest.TestCase):
    r"""Test full transform_input'ed QMC calculations."""

    def test_qmc_rhf(self):
        qmcsys, qmcmf = transform_input.pyscf_openms_to_qmc(
            mol,
            mf,
            chol_thresh=chol_thresh,
        )

        qmc_runner = get_qmc_object(
            mol=qmcsys,
            mf=qmcmf,
            dt=dt,
            time=time,
            num_walkers=num_walkers,
            energy_scheme=energy_scheme,
            uhf=uhf,
            chol_thresh=chol_thresh,
        )

        E_default, std_default = get_mean_std(Elist_default)
        _, Elist_qmc = qmc_runner.kernel()
        E_qmc, std_qmc = get_mean_std(Elist_qmc)

        print("Integration test: 01-qmc_rhf vs. new QMCSystem")
        print(f"HF energy       = {E_hf:.6f}")
        print(f"Original QMC    = {E_default:.6f} ± {std_default:.6f}")
        print(f"New AFQMCSystem = {E_qmc:.6f} ± {std_qmc:.6f}")

        print("Difference in quantities:")
        print(f"One-electron: {np.linalg.norm(qmc_runner.h1e - qmc_default.h1e):.6e}")
        print(
            f"Two-electron: {np.linalg.norm(qmc_runner.ltensor - qmc_default.ltensor):.6e}"
        )
        print(f"Constant     : {qmc_runner.nuc_energy - qmc_default.nuc_energy}")

        err = abs(E_qmc - E_default)
        print(f"New system: error = {err:.6e}")
        self.assertLess(err, 1.0e-4, msg=f"|E_default - E_newsys| = {err:.6e}")


if __name__ == "main":
    unittest.main()
