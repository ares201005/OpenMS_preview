import unittest
import copy
import time

import numpy
from pyscf import gto, scf, cc
from openms.mqed import qedhf, ccsd


def LiF_ref(complete=True):
    # Reference energies, complete basis assumption (quadrupole/Q terms)
    if complete:
        refs_list = [-0.0558287, -0.05584209, -0.05584289, -0.05585593,
                     -0.056926, -0.05693954, -0.05695403, -0.05697047]
    # Reference energies, no basis assumption (no Q terms)
    else:
        refs_list = [-0.05734083, -0.05735435, -0.05735539, -0.05736854,
                     -0.05846713, -0.05848054, -0.05849606, -0.05851241]
    qedccsd_refs = numpy.asarray(refs_list)

    # Create molecule object
    atom = f"Li   0.0  0.0  0.0  ;  F   0.0  0.0  1.5"
    mol = gto.M(
        atom = atom,
        basis = "sto-3g",
        unit = "Angstrom",
        symmetry = True,
        verbose = 3,
    )

    return mol, qedccsd_refs


class TestQEDCC(unittest.TestCase):

    def test_qedcc_ref1(self):
        r"""Test with complete basis set assumption (i.e., quadrupole terms)"""
        # Molecule object and reference energies
        mol, qedccsd_refs = LiF_ref()

        # HF kernel
        mf = scf.RHF(mol)
        mf.kernel()

        ## CC kernel
        mycc = cc.CCSD(mf)
        mycc.diis_space = 10

        start_time = time.time()
        mycc.kernel()
        print("\nCCSD kernel time: ", time.time() - start_time)
        print("\nCCSD correlation energy: ", mycc.e_corr, "\n")

        # Create cavity
        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 3.0 / 27.211386245988
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 0, 1])

        # QED-HF kernel
        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                         add_nuc_dipole=False)
        qedmf.max_cycle = 500
        qedmf.kernel()

        # Copy Boson object from QED-HF kernel
        qed = copy.copy(qedmf.qed)
        qed.verbose = 5
        qed.kernel()

        # Scan multiple QED-CC parameters
        e_corrs_list = []
        for add_U2n in [False, True]:
            for nfock1 in range(1, 3):
                for nfock2 in range(1, 3):
                    print("\n", "=" * 60)

                    # QED-CC kernel
                    myqedccsd = ccsd.CCSD(qed, nfock1=nfock1, nfock2=nfock2, add_U2n=add_U2n)
                    myqedccsd.max_cycle = 500
                    myqedccsd.verbose = 3

                    start_time = time.time()
                    __, e_corr = myqedccsd.kernel()
                    print("\nQED-CCSD kernel time: ", time.time() - start_time)
                    print("\nQED-CCSD correlation energy: ", e_corr)

                    # Append correlation energy to list
                    e_corrs_list.append(e_corr)

        # Compare correlation energy values
        e_corrs = numpy.asarray(e_corrs_list)
        for e_corr, qedccsd_ref in zip(e_corrs, qedccsd_refs):
            self.assertAlmostEqual(e_corr, qedccsd_ref, places=5,
                msg="Correlation energy does not match the reference value.")

        return


    def test_qedcc_ref2(self):
        r"""Test without complete basis set assumption (i.e., no Q)"""
        # Molecule object and reference energies
        mol, qedccsd_refs = LiF_ref(complete=False)

        # HF kernel
        mf = scf.RHF(mol)
        mf.kernel()

        # CC kernel
        mycc = cc.CCSD(mf)
        mycc.diis_space = 10

        start_time = time.time()
        mycc.kernel()
        print("\nCCSD kernel time: ", time.time() - start_time)
        print("\nCCSD correlation energy: ", mycc.e_corr, "\n")

        # Create cavity
        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 3.0 / 27.211386245988
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 0, 1])

        # QED-HF kernel
        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                         add_nuc_dipole=False, complete_basis=False)
        qedmf.max_cycle = 500
        qedmf.kernel()

        # Copy Boson object from QED-HF kernel
        qed = copy.copy(qedmf.qed)
        qed.verbose = 5
        qed.kernel()

        # Scan multiple QED-CC parameters
        e_corrs_list = []
        for add_U2n in [False, True]:
            for nfock1 in range(1, 3):
                for nfock2 in range(1, 3):
                    print("\n", "=" * 60)

                    # QED-CC kernel
                    myqedccsd = ccsd.CCSD(qed, nfock1=nfock1, nfock2=nfock2, add_U2n=add_U2n)
                    myqedccsd.max_cycle = 500
                    myqedccsd.verbose = 3

                    start_time = time.time()
                    __, e_corr = myqedccsd.kernel()
                    print("\nQED-CCSD kernel time: ", time.time() - start_time)
                    print("\nQED-CCSD correlation energy: ", e_corr)

                    # Append correlation energy to list
                    e_corrs_list.append(e_corr)

        # Compare correlation energy values
        e_corrs = numpy.asarray(e_corrs_list)
        for e_corr, qedccsd_ref in zip(e_corrs, qedccsd_refs):
            self.assertAlmostEqual(e_corr, qedccsd_ref, places=5,
                msg="Correlation energy does not match the reference value.")

        return


    # def test_qedcc_isomer(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
