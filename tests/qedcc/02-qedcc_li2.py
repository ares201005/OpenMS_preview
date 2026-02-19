import unittest
import copy
import time

import numpy
from pyscf import gto, scf, cc
from openms.mqed import qedhf, ccsd


class TestQEDCC(unittest.TestCase):

    def test_qedcc_ref(self):
        # Reference energies
        refs_list = [-14.593881582106865, -14.64207137319351]
        qedccsd_refs = numpy.asarray(refs_list)

        # Scan multiple bond distances
        qed_energies_list = []
        for bond in numpy.arange(2.0, 2.21, 0.20):
            print("\n", "=" * 60)

            # Create molecule object
            atom = f"Li   0.0  0.0  0.0  ;  Li   0.0  0.0  {bond}"
            mol = gto.M(
                atom = atom,
                basis = "6-31g",
                unit = "Bohr",
                symmetry = True,
                verbose = 3,
            )

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
            cavity_mode[0, :] = 1.e-1 * numpy.asarray([1, 1, 1])

            # QED-HF kernel
            qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                            add_nuc_dipole=False)
            qedmf.max_cycle = 500
            qedmf.kernel()

            # Copy Boson object from QED-HF kernel
            qed = copy.copy(qedmf.qed)
            qed.verbose = 5
            qed.kernel()

            # QED-CC kernel
            myqedccsd = ccsd.CCSD(qed, nfock1=2, nfock2=2, add_U2n=True)
            myqedccsd.max_cycle = 200
            myqedccsd.verbose = 3

            start_time = time.time()
            e_tot, e_corr = myqedccsd.kernel()
            print("\nQED-CCSD kernel time: ", time.time() - start_time)
            print("\nQED-CCSD correlation energy: ", e_corr)

            # Append total energy to list
            qed_energies_list.append(e_tot)

        # Test total energy values
        qed_energies = numpy.asarray(qed_energies_list)
        for e_tot, qedccsd_ref in zip(qed_energies, qedccsd_refs):
            self.assertAlmostEqual(e_tot, qedccsd_ref, places=5,
                msg="Total energy does not match the reference value.")

        return


if __name__ == '__main__':
    unittest.main()
