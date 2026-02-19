import unittest
import copy

import numpy
import pyscf.gto, pyscf.cc
import openms.mqed


def build_molecule():
    r"""Create molecule object."""

    atom = f"Li   0.0  0.0  0.0  ;  F   0.0  0.0  1.5"
    mol = pyscf.gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        symmetry=True,
        verbose=4,
    )

    return mol


class TestCCSDEquivalence(unittest.TestCase):
    r"""Unit tests verifying QED-CCSD with nf1=nf2=0 matches PySCF CCSD correlation."""

    def build_qed_mf(self, molecule, g_factor):
        """Build QED-HF mean-field object."""
        cavity_freq = numpy.zeros(1)
        cavity_mode = numpy.zeros((1, 3))
        cavity_freq[0] = 3.0 / 27.211386245988  # 3 eV -> Ha
        cavity_mode[0, :] = g_factor * numpy.asarray([0, 0, 1])

        qedmf = openms.mqed.qedhf.RHF(
            mol=molecule,
            cavity_mode=cavity_mode,
            cavity_freq=cavity_freq,
            add_nuc_dipole=False
        )

        qedmf.max_cycle = 1500
        qedmf.conv_tol = 1e-9
        qedmf.diis_space = 15
        qedmf.precond = 5e-3

        return qedmf


    def build_qedcc_object(self, mean_field, nf1=0, nf2=0, u2n=False, max_cycles=500):
        """Build QED-CCSD object."""
        qed_photon = None
        if mean_field.qed:
            qed_photon = copy.copy(mean_field.qed)
        else:
            raise Exception("Mean-field object doesn't have a QED object attribute.")

        qedcc = openms.mqed.ccsd.CCSD(qed_photon, nfock1=nf1, nfock2=nf2, add_U2n=u2n)
        qedcc.max_cycle = max_cycles

        return qedcc


    def run_qedccsd_and_pyscf_ccsd(self, g_factor):
        """Run QED-CCSD and PySCF CCSD on the same mean-field and compare."""
        mol = build_molecule()

        # Run QED mean-field
        qedmf = self.build_qed_mf(mol, g_factor)
        qedmf.kernel()

        # Run QED-CCSD with nf1=nf2=0, u2n=False
        qedcc = self.build_qedcc_object(qedmf, nf1=0, nf2=0, u2n=False)
        qedcc.kernel()
        qedcc_corr = qedcc.e_corr

        # Run PySCF CCSD on the same mean-field
        pyscf_cc = pyscf.cc.CCSD(qedmf)
        pyscf_cc.max_cycle = 500
        pyscf_cc.kernel()
        pyscf_corr = pyscf_cc.e_corr

        return qedcc_corr, pyscf_corr


    def test_qedhf_nf0_equivalence(self):
        """Test QED-HF with QED-CCSD(nf=0) matches PySCF CCSD correlation."""
        ref_corr_energy = -0.054452119098

        qedcc_corr, pyscf_corr = self.run_qedccsd_and_pyscf_ccsd(g_factor=0.0)

        # Check both correlation energies match reference
        self.assertAlmostEqual(
            qedcc_corr,
            ref_corr_energy,
            places=5,
            msg="QED-CCSD(nf=0) correlation energy does not match reference"
        )
        self.assertAlmostEqual(
            pyscf_corr,
            ref_corr_energy,
            places=5,
            msg="PySCF CCSD correlation energy does not match reference"
        )


if __name__ == '__main__':
    unittest.main()
