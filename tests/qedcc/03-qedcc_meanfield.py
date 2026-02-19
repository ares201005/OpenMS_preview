import unittest

import numpy
import pyscf.gto, pyscf.cc
import openms.mqed


def build_molecule():
    r"""Create molecule object."""

    atom = f"Li   0.0  0.0  0.0  ;  F   0.0  0.0  1.5"
    mol = pyscf.gto.M(
        atom = atom,
        basis = "sto-3g",
        unit = "Angstrom",
        symmetry = True,
        verbose = 4,
    )

    return mol


class TestMeanFieldCCSD(unittest.TestCase):
    r"""Unit tests for QED mean-field methods with PySCF CCSD correlation."""

    def build_qed_mf(self, molecule, mf_type, g_factor):
        """Build QED mean-field object."""
        cavity_freq = numpy.zeros(1)
        cavity_mode = numpy.zeros((1, 3))
        cavity_freq[0] = 3.0 / 27.211386245988  # 3 eV -> Ha
        cavity_mode[0, :] = g_factor * numpy.asarray([0, 0, 1])

        if mf_type == "qedhf":
            qedmf = openms.mqed.qedhf.RHF(
                mol=molecule,
                cavity_mode=cavity_mode,
                cavity_freq=cavity_freq,
                add_nuc_dipole=False
            )
        elif mf_type == "scqedhf":
            qedmf = openms.mqed.scqedhf.RHF(
                mol=molecule,
                cavity_mode=cavity_mode,
                cavity_freq=cavity_freq,
                add_nuc_dipole=False
            )
        elif mf_type == "vtqedhf":
            qedmf = openms.mqed.vtqedhf.RHF(
                mol=molecule,
                cavity_mode=cavity_mode,
                cavity_freq=cavity_freq,
                add_nuc_dipole=False
            )
        else:
            raise ValueError(f"Unknown mean field type: {mf_type}")

        qedmf.max_cycle = 1500
        qedmf.conv_tol = 1e-9
        qedmf.diis_space = 15
        qedmf.precond = 5e-3

        return qedmf


    def run_mf_and_ccsd(self, mf_type):
        """Run mean-field and CCSD calculations for a given MF type."""
        mol = build_molecule()
        mf_energies = []
        ccsd_energies = []
        corr_energies = []

        gfac_values = [0.0, 0.01, 0.05]
        for gfac in gfac_values:
            # Run QED mean-field
            qedmf = self.build_qed_mf(mol, mf_type, gfac)
            qedmf.kernel()
            mf_energies.append(qedmf.e_tot)

            # Run PySCF CCSD on top of QED mean-field
            cc = pyscf.cc.CCSD(qedmf)
            cc.max_cycle = 500
            cc.kernel()
            ccsd_energies.append(cc.e_tot)
            corr_energies.append(cc.e_corr)

        return numpy.array(mf_energies), numpy.array(ccsd_energies), numpy.array(corr_energies)


    def test_qedhf(self):
        """Test QED-HF mean-field and CCSD correlation energies."""
        ref_qedhf_energies = numpy.array([
            -105.368886426602,
            -105.368569909424,
            -105.361026254419,
        ])
        ref_qedhf_ccsd_corr_energies = numpy.array([
            -0.054452119098,
            -0.054389011203,
            -0.052881625255,
        ])

        mf_energies, _, corr_energies = self.run_mf_and_ccsd("qedhf")

        for i, (computed, reference) in enumerate(zip(mf_energies, ref_qedhf_energies)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"QED-HF mean-field energy {i} does not match reference"
            )
        for i, (computed, reference) in enumerate(zip(corr_energies, ref_qedhf_ccsd_corr_energies)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"QED-HF/CCSD correlation energy {i} does not match reference"
            )


    def test_scqedhf(self):
        """Test SC-QED-HF mean-field and CCSD correlation energies."""
        ref_scqedhf_energies = numpy.array([
            -105.368886426602,
            -105.368751573936,
            -105.365529962629,
        ])
        ref_scqedhf_ccsd_corr_energies = numpy.array([
            -0.054452104850,
            -0.054432082317,
            -0.053948544813,
        ])

        mf_energies, _, corr_energies = self.run_mf_and_ccsd("scqedhf")

        for i, (computed, reference) in enumerate(zip(mf_energies, ref_scqedhf_energies)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"SC-QED-HF mean-field energy {i} does not match reference"
            )
        for i, (computed, reference) in enumerate(zip(corr_energies, ref_scqedhf_ccsd_corr_energies)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"SC-QED-HF/CCSD correlation energy {i} does not match reference"
            )


    def test_vtqedhf(self):
        """Test VT-QED-HF mean-field and CCSD correlation energies."""

        ref_vtqedhf_vals = numpy.array([
            -105.368886426602,
            -105.368791853552,
            -105.366528459557,
        ])
        ref_vtqedhf_ccsd_corr_vals = numpy.array([
            -0.054452104850,
            -0.054436886114,
            -0.054070422716,
        ])

        mf_energies, _, corr_energies = self.run_mf_and_ccsd("vtqedhf")

        for i, (computed, reference) in enumerate(zip(mf_energies, ref_vtqedhf_vals)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"VT-QED-HF mean-field energy {i} does not match reference"
            )
        for i, (computed, reference) in enumerate(zip(corr_energies, ref_vtqedhf_ccsd_corr_vals)):
            self.assertAlmostEqual(
                computed,
                reference,
                places=5,
                msg=f"VT-QED-HF/CCSD correlation energy {i} does not match reference"
            )


if __name__ == '__main__':
    unittest.main()
