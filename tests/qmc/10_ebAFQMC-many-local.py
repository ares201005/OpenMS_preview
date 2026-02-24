import unittest
import contextlib
import numpy
from pyscf import gto, scf
from openms.mqed import qedhf
from openms.qmc.afqmc import AFQMC
from openms.lib import boson
from openms.qmc.tools import get_mean_std
from molecules import get_mol, get_cavity

# Test eb-AFQMC using bare molecule object


def afqmc_energy_vs_lambda(
    mol,
    verbose,
    cavity_freq,
    cavity_mode,
    nmode,
    nmode_max,
    time=5.0,
    nwalkers=100,
    uhf=False,
    gmat=None,
):
    dt = 0.005
    # nmode = len(cavity_freq)
    # add num_fake field in order to have same number of random numbers in the comparison
    nfake = 2 * nmode_max - 2 * nmode
    propagator_options = {
        "decouple_bilinear": True,
        "decouple_scheme": 2,
        "num_fake_fields": nfake,
    }
    afqmc = AFQMC(
        mol,
        dt=dt,
        total_time=time,
        num_walkers=nwalkers,
        energy_scheme="hybrid",
        uhf=uhf,
        boson_freq=cavity_freq,
        gmat=gmat,
        propagator_options=propagator_options,
        verbose=verbose,
        # random_seed=123456,
    )
    times, energies = afqmc.kernel()
    return energies


def test_nmode(
    mol,
    verbose,
    nmode=1,
    nmode_max=8,
    g_bare=0.1,
    time=5.0,
    nwalkers=100,
):
    gfac = g_bare / numpy.sqrt(nmode)
    cavity_freq, cavity_mode = get_cavity(nmode, gfac, pol_axis=2)
    # set gmat
    dip_mat = boson.get_dipole_ao(mol)
    gmat_ao = numpy.einsum("nx, xuv->nuv", cavity_mode, dip_mat)
    energies = afqmc_energy_vs_lambda(
        mol,
        verbose,
        cavity_freq,
        cavity_mode,
        nmode=nmode,
        nmode_max=nmode_max,
        time=time,
        nwalkers=nwalkers,
        gmat=gmat_ao,
    )
    means, stds = get_mean_std(energies)
    print("[nmode, nwalker, time], means, stds = ", nmode, nwalkers, time, means, stds)
    return means, stds


class Test_ebAFQMC(unittest.TestCase):
    def test_modes(self):
        verbose = 5
        g_bare = 0.1
        nmode_max = 8
        bond = 2.0  # essentially arbitrary
        mol = get_mol(bond=bond)
        means = numpy.zeros(3, dtype=float)
        for i, nmode in enumerate([1, 2, 4]):
            means[i], _ = test_nmode(
                mol=mol,
                nmode=nmode,
                nmode_max=nmode_max,
                g_bare=g_bare,
                time=20.0,
                nwalkers=10000,
                verbose=verbose,
            )

        first_mean = numpy.zeros_like(means)
        first_mean[:] = means[0]
        numpy.testing.assert_allclose(means, first_mean, atol=1e-3)
        # self.assertLess(
        #    abs(means - mean_ref),
        #    1.0e-3,
        #    msg="E_mean does not match the reference value.",
        # )


if __name__ == "__main__":
    unittest.main()
