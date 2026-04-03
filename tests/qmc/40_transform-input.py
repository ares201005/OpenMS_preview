import unittest
import numpy as np
from scipy.linalg import svd
from pyscf import gto, scf, fci
from pyscf.gto import Mole as Mol
from pyscf.ao2mo import restore
from pyscf.scf.hf import RHF as HF
from openms.mqed import qedhf
from openms.qmc.afqmc import AFQMC
from openms.qmc import transform_input
from openms.qmc.transform_input import AFQMCSystem, QMCMeanField
from openms.lib import boson
from molecules import get_mol


# System is that from openms.tests.qmc.01-qmc_rhf.py
def get_qmc_object(
    mol: Mol,
    mf: HF,
    time: float = 6.0,
    num_walkers: int = 200,
    uhf: bool = False,
    energy_scheme: str = "hybrid",
):
    r"""Note the number of walkers here is small, in order to do fast test"""
    return AFQMC(
        mol,
        mf,
        dt=0.005,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        chol_thresh=1.0e-20,
        property_calc_freq=1,
        verbose=mol.verbose,
    )


def get_mean_std(energies, ratio=10):
    # Compute the mean and standard deviation
    # Extract the real parts of the last m elements
    m = max(1, len(energies) // ratio)
    last_m_real = np.asarray(energies[-m:]).real
    mean = np.mean(last_m_real)
    std_dev = np.std(last_m_real)

    return mean, std_dev


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

qmc_default = get_qmc_object(mol, mf)


if __name__ == "__main__":
    unittest.main()


class TestTransformInputElectron(unittest.TestCase):
    r"""Unit tests of transform_input methods: electron-only system."""

    def test_decompose_eri(self):
        eri = restore("s1", mf._eri, mol.nao_nr())
        S = transform_input.loewdin_orth(mf.get_ovlp())
        L_eri, n_eri = transform_input.decompose_tensor_eri(eri, S, thresh=1e-20)

        _eri = eri.reshape(mol.nao_nr() ** 2, -1)
        u, s, v = svd(_eri)

        np.testing.assert_equal(n_eri, qmc_default.nbarefields)
        np.testing.assert_allclose(L_eri, qmc_default.ltensor, rtol=1e-4)


class TestQMCWithTransformedInput(unittest.TestCase):
    r"""Test full transform_input'ed QMC calculations."""

    pass


if __name__ == "main":
    unittest.main()
