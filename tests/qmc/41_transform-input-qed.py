import unittest
import numpy as np
from scipy.linalg import svd
from pyscf import gto, scf
from pyscf.gto import Mole as Mol
from pyscf.ao2mo import restore
from pyscf.scf.hf import RHF as HF
from openms.mqed import qedhf
from openms.qmc.afqmc import AFQMC
from openms.qmc import transform_input
from openms.qmc.transform_input import AFQMCSystem, QMCMeanField
from openms.qmc.tools import get_mean_std
from openms.lib import boson
from molecules import get_mol, get_cavity


# System is that from openms.tests.qmc.10_ebAFQMC.py
def get_qmc_object(
    mol: Mol | AFQMCSystem,
    mf: HF | QMCMeanField,
    cavity_freq: np.ndarray,
    gmat: np.ndarray | None = None,
    qed: bool = False,
    dt: float = 0.005,
    time: float = 6.0,
    num_walkers: int = 200,
    uhf: bool = False,
    energy_scheme: str = "hybrid",
    decouple: bool = True,
    chol_thresh: float = 1.0e-6,
    verbose: int = 1,
) -> AFQMC:

    nmode = len(cavity_freq)
    # add num_fake field in order to have same number of random numbers in the comparison
    nfake = 0 if qed else nmode
    propagator_options = {
        "decouple_bilinear": decouple,
        "decouple_scheme": 2,
        "num_fake_fields": nfake,
    }

    return AFQMC(
        mol,
        mf,
        dt=dt,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        gmat=gmat,
        boson_freq=cavity_freq,
        chol_thresh=chol_thresh,
        propagator_options=propagator_options,
        property_calc_freq=1,
        verbose=verbose,
    )


class TestIntegratedWithBoson(unittest.TestCase):

    def test1(self):
        # Molecular properties
        bond = 2.0
        mol = get_mol(bond=bond)
        mf = scf.RHF(mol)
        E_hf = mf.kernel()
        mean_ref = -7.9203411  # for reference; from 10_ebAFQMC.py

        # Cavity quantities
        gfac = 0.1
        cavity_freq, cavity_mode = get_cavity(1, gfac, omega=1.0, pol_axis=2)
        # for omega = 0.5, sqrt(0.5 * omega) = 0.5 as well
        dip_mat = boson.get_dipole_ao(mol)
        gmat_ao = np.einsum("nx, xuv->nuv", cavity_mode, dip_mat)

        # AFQMC quantities
        dt = 0.005
        time = 10.0
        num_walkers = 1000

        qmc_default = get_qmc_object(
            mol=mol,
            mf=mf,
            cavity_freq=cavity_freq,
            gmat=gmat_ao,
            qed=True,
            dt=dt,
            time=time,
            num_walkers=num_walkers,
            decouple=True,
        )
        _, Elist = qmc_default.kernel()
        means, stds = get_mean_std(Elist)

        qmcmol, qmcmf = transform_input.pyscf_openms_to_qmc(
            mol,
            mf,
            has_photon=True,
            nmodes_photon=len(cavity_freq),
            dim_fock_photon=2,
            omega_photon=cavity_freq,
            lambda_dot_mu=gmat_ao,
        )

        qmc_runner = get_qmc_object(
            mol=qmcmol,
            mf=qmcmf,
            cavity_freq=cavity_freq,
            gmat=gmat_ao,
            qed=True,
            dt=dt,
            time=time,
            num_walkers=num_walkers,
            decouple=True,
        )

        _, Erunner = qmc_runner.kernel()
        mean_new, std_new = get_mean_std(Erunner)

        print("The QMC quantities:")
        print("h1e: shape | norm")
        print(f"old {qmc_default.h1e.shape} | {np.linalg.norm(qmc_default.h1e):.6e}")
        print(f"new {qmc_runner.h1e.shape}  | {np.linalg.norm(qmc_runner.h1e):.6e}")
        print("ltensor: shape | norm")
        print(
            f"old {qmc_default.ltensor.shape} | {np.linalg.norm(qmc_default.ltensor):.6e}"
        )
        print(
            f"new {qmc_runner.ltensor.shape}  | {np.linalg.norm(qmc_runner.ltensor):.6e}"
        )
        print(
            f"nuc: old = {qmc_default.nuc_energy:.6f} | new = {qmc_runner.nuc_energy:.6f}"
        )
        print("Bilinear tensors: shape | norm")
        print("New multiplied by sqrt(omega/2); different partitioning of Afac/Bfac")
        print("for one mode:")
        bil_old = qmc_default.chol_bilinear_e
        bil_old *= np.sqrt(0.5 * cavity_freq[0])
        print(
            f"old {qmc_default.chol_bilinear_e.shape} | {np.linalg.norm(bil_old):.6e}"
        )
        print(
            f"new {qmc_runner.chol_bilinear_e.shape} | {np.linalg.norm(qmc_runner.chol_bilinear_e):.6e}"
        )

        print(f"HF energy           + {E_hf:.6f}")
        print(f"Original energy     = {means:.6f} ± {stds:.6f}")
        print(f"New system energy   = {mean_new:.6f} ± {std_new:.6f}")
        print(f"Reference energy    = {mean_ref:.6f}")
        self.assertLess(
            abs(mean_new - means),
            1.0e-3,
            msg="New and old system don't match!.",
        )


class TestTransformInputBoson(unittest.TestCase):
    pass


if __name__ == "main":
    unittest.main()
