#
# @ 2023. Triad National Security, LLC. All rights reserved.
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
# Author: Yu Zhang <zhy@lanl.gov>
#


r"""

Theoretical background
----------------------

Phaseless formalism for complex auxiliary-fields :cite:`zhang2013af, zhang2021jcp`:

Imaginary time evolution
~~~~~~~~~~~~~~~~~~~~~~~~

Most ground-state QMC methods are based on the
imaginary time evolution

.. math::

  \ket{\Psi}\propto\lim_{\tau\rightarrow\infty} e^{-\tau\hat{H}}\ket{\Psi_T}.

Numerically, the ground state can be projected out iteratively,

.. math::

   \ket{\Psi^{(n+1)}}=e^{-\Delta\tau \hat{H}}\ket{\Psi^{(n)}}.

To evaluate the imaginary time propagation, Trotter decomposition is used to
to break the evolution operator :math:`e^{-\Delta\tau H} \approx   e^{-\Delta\tau H_1}e^{-\Delta\tau H_2}`.


Hubbard-Stratonovich transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cholesky decomposition of eri:

.. math::

     (ij|kl) = \sum_\gamma L^*_{\gamma,il} L_{\gamma,kj}

The two-body interactions becomes

.. math::

     H_2 = & \sum_{ijkl} V_{ijkl} c^\dagger_i c^\dagger_j c_k c_l \\
         = & \sum_{ijkl} V_{ijkl} c^\dagger_i c_l c^\dagger_j c_k- \sum_{ijkl} c^\dagger_i c_l \delta_{jl} \\
         = & \sum_{ijkl}\sum_\gamma (L^*_{\gamma,il} c^\dagger_i c_l) (L_{\gamma,kj}c^\dagger_j c_k)
            - \sum_{ijkj} V_{ijkj} c^\dagger_i c_k

Hence, the last term in above equation is a single-particle operator, which is defined as the
shifted_h1e in the code.

Finally,

.. math::

  e^{-\Delta\tau H} = \int d x p(x) B(x)

where :math:`B(x)` is the Auxilary field.

Importance sampling
~~~~~~~~~~~~~~~~~~~

With importance sampling, the global wave function is a weighted statical sum over
:math:`N_w` walkers

.. math::

  \Psi^n = \sum_w


Program overview
----------------

"""

import sys, os
from pyscf import tools, lo, scf, fci, ao2mo
from pyscf.lib import logger
import numpy
import scipy
import itertools
import h5py

from openms.mqed.qedhf import RHF as QEDRHF
from openms.lib.boson import Photon

from openms.qmc import qmc


class AFQMC(qmc.QMCbase):


    def __init__(self, system, *args, **kwargs):

        super().__init__(system, *args, **kwargs)
        self.exp_h1e = None

    def dump_flags(self):
        r"""
        Dump flags
        """
        print(f"\n========  AFQMC simulation using OpenMS package ========\n")

    def hs_transform(self, h1e):
        r"""
        Perform Hubbard-Stratonovich (HS) decomposition

        .. math::

            e^{-\Delta\tau \hat{H}} = \int d\boldsymbol{x} p(\boldsymbol{x})\hat{B}(\boldsymbol{x}).

        """
        hs_fields = None
        return hs_fields

    def build_propagator(self, h1e, eri, ltensor):
        r"""Pre-compute the propagators
        """

        # shifted h1e
        self.shifted_h1e = numpy.zeros(h1e.shape)
        rho_mf = self.trial.psi.dot(self.trial.psi.T.conj())
        self.mf_shift = 1j * numpy.einsum("npq,pq->n", ltensor, rho_mf)
        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            self.shifted_h1e[p, q] = h1e[p, q] - 0.5 * numpy.trace(eri[p, :, :, q])
        self.shifted_h1e = self.shifted_h1e - numpy.einsum("n, npq->pq", self.mf_shift, 1j*ltensor)

        self.TL_tensor = numpy.einsum("pr, npq->nrq", self.trial.psi.conj(), ltensor)
        self.exp_h1e = scipy.linalg.expm(-self.dt/2 * self.shifted_h1e)

    def propagation_onebody(self, phi_w):
        r"""Propgate one-body term
        """
        return numpy.einsum('pq, zqr->zpr', self.exp_h1e, phi_w)

    def propagation_twobody(self, vbias, phi_w):
        r"""Propgate two-body term
        Which is the major computational bottleneck.

        TODO: move the two-body propagation into this function
        TODO: improve the efficiency
        of this part with a) MPI, b) GPU, c) tensor hypercontraction

        """
        pass

    def propagation(self, walkers, xbar, ltensor):
        r"""
        Eqs 50 - 51 of Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf

        Trotter decomposition of the imaginary time propagator:

        .. math::

            e^{-\Delta\tau/2 H_1} e^{-\Delta\tau \sum_\gamma L^2_\gamma /2 } e^{-\Delta\tau H_1/2}

        where the two-body propagator in HS form

        .. math::

            e^{-\Delta\tau L^2_\gamma} \rightarrow  \exp[x\sqrt{-\Delta\tau}L_\gamma]
            = \sum_n \frac{1}{n!} [x\sqrt{-\Delta\tau}L_\gamma]^n
        """

        # a) 1-body propagator propagation :math:`e^{-dt/2*H1e}`
        walkers.phiw = self.propagation_onebody(walkers.phiw)

        # b): 2-body propagator propagation :math:`\exp[(x-\bar{x}) * L]`
        # normally distributed AF
        xi = numpy.random.normal(0.0, 1.0, self.nfields * self.num_walkers)
        xi = xi.reshape(self.num_walkers, self.nfields)

        xshift = xi - xbar
        # TODO: further improve the efficiency of this part
        two_body_op_power = 1j * numpy.sqrt(self.dt) * numpy.einsum('zn, npq->zpq', xshift, ltensor)

        # \sum_n 1/n! (j\sqrt{\Delta\tau) xL)^n
        temp = walkers.phiw.copy()
        for order_i in range(self.taylor_order):
            temp = numpy.einsum('zpq, zqr->zpr', two_body_op_power, temp) / (order_i + 1.0)
            walkers.phiw += temp

        # c):  1-body propagator propagation e^{-dt/2*H1e}
        walkers.phiw = self.propagation_onebody(walkers.phiw)
        # walkers.phiw = numpy.exp(-self.dt * nuc) * walkers.phiw

        # (x*\bar{x} - \bar{x}^2/2)
        cfb = numpy.einsum("zn, zn->z", xi, xbar) - 0.5*numpy.einsum("zn, zn->z", xbar, xbar)
        cmf = -numpy.sqrt(self.dt) * numpy.einsum('zn, n->z', xshift, self.mf_shift)
        return cfb, cmf


class QEDAFQMC(AFQMC):

    def __init__(self, system, mf, *args, **kwargs):

        super().__init__(system, *args, **kwargs)

        # create qed object

    def get_integral(self):
        r"""
        TODO: 1) add DSE-mediated eri and oei
              2) bilinear coupling term (gmat)
        """
        pass


    def dump_flags(self):
        r"""
        Dump flags
        """
        print(f"\n========  QED-AFQMC simulation using OpenMS package ========\n")


if __name__ == "__main__":
    from pyscf import gto, scf, fci
    bond = 1.6
    natoms = 2
    atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
    mol = gto.M(atom=atoms, basis='sto-6g', unit='Bohr', verbose=3)

    num_walkers = 500
    afqmc = AFQMC(mol, dt=0.005, total_time=2.0,
                 num_walkers=num_walkers, energy_scheme="hybrid",
                 verbose=3)

    times, energies = afqmc.kernel()

    # HF energy
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()

    # FCI energy
    fcisolver = fci.FCI(mf)
    fci_energy = fcisolver.kernel()[0]

    print(fci_energy)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # time = numpy.arange(0, 5, 0.)
    ax.plot(times, energies, '--', label='afqmc (my code)')
    ax.plot(times, [hf_energy] * len(times), '--')
    ax.plot(times, [fci_energy] * len(times), '--')
    ax.set_ylabel("Ground state energy")
    ax.set_xlabel("Imaginary time")
    plt.savefig("afqmc_gs_h2_sto6g.pdf")
    #plt.show()
