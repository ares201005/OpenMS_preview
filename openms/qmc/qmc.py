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
QMC method
----------

DF or Cholesky decomposition:

.. math::

    I_{pqrs} \simeq L_{\lambda, pq} L_{\lambda, rs}

Tensor hypercontraction:

.. math::

    I_{pqrs} \simeq X_{\mu p} X_{\mu q} Z_{\mu\nu} X_{\nu r} X_{\nu s}

it means that the :math:`L` is effectively decomposed into :math:`X`

.. math::

    L_{\lambda pq} = X_{\lambda p} X_{\lambda q}

"""

import sys
from abc import abstractmethod
from pyscf import lo, scf, fci
from pyscf import tools as pyscftools
from pyscf.gto import mole
import numpy as backend
import scipy
import itertools
import logging
import h5py
import time
import warnings

from openms import runtime_refs, _citations
from . import generic_walkers as gwalker
from . import tools

from pyscf.lib import logger
from openms.lib.logger import task_title
from openms.lib.boson import Boson
from openms.qmc.trial import make_trial, multiCI
from openms.qmc.estimators import local_eng_elec_chol
from openms.qmc.estimators import local_eng_elec_chol_new

from openms.qmc.propagators import Phaseless, PhaselessElecBoson


def qr_ortho(phiw):
    r"""
    phiw size is [nao, nalpha/nbeta]
    """
    Qmat, Rmat = backend.linalg.qr(phiw)
    Rdiag = backend.diag(Rmat)
    ## don't need to work on the sign
    # signs = backend.sign(Rdiag)
    # Qmat = backend.dot(Qmat, backend.diag(signs))
    log_det = backend.sum(backend.log(backend.abs(Rdiag)))

    return Qmat, log_det


def qr_ortho_batch(phiw):
    r"""
    phiw size is [nwalker, nao, nalpha/nbeta]
    """
    Qmat, Rmat = backend.linalg.qr(phiw)
    Rdiag = backend.einsum("wii->wi", Rmat)
    log_det = backend.einsum("wi->w", backend.log(abs(Rdiag)))
    return Qmat, log_det


def kernel(mc, propagator=None, trial=None):
    r"""An universal kernel for qmc propagations"""

    mc.dump_flags()

    logger.note(mc, task_title("Entering the main kernel of AFQMC"))
    # prepare propagation
    logger.info(mc, f"\n Random seed is {mc.random_seed}\n")
    backend.random.seed(mc.random_seed)

    # integrals
    h1e = mc.h1e
    ltensor = mc.ltensor
    if propagator is None:
        propagator = mc.propagator
    if trial is None:
        trial = mc.trial
    walkers = mc.walkers

    # setup propagator
    # generalize the propagator build for any typs
    # (fermions, bosons, or fermion-boson mixture).
    # mc.build_propagator(h1e, eri, ltensor)
    propagator.build(h1e, ltensor, trial, mc.geb)

    logger.debug(mc, f"Debug: the initial orthogonalise in walker")
    mc.orthogonalization()

    # initialize property calculations
    mc.property_stack(walkers, -1)

    # start the propagation
    tt = 0.0
    energy_list = []
    time_list = []
    wall_t0 = time.time()
    logstring = f"{'Step':^8}{'Etot':^16}{'Raw_Etot':^16}{'Norm':^14}{'E1':^16}{'E2':^16}"
    if isinstance(propagator, PhaselessElecBoson):
        logstring += f"{'Eb':^16}{'Eg':^16}"
    logstring += "  Wall_time"
    logger.info(mc, logstring)

    # while tt <= mc.total_time:
    for step in range(mc.nsteps):
        t0 = time.time()
        tt = mc.dt * step
        dump_result = step % mc.print_freq == 0
        logger.debug(
            mc, f"\nDebug: -------------- qmc step {step} -----------------"
        )

        # step 0): periodic re-orthogonalization
        # (FIXME: whether put this at the begining or end, in principle, should not matter)
        if (step + 1) % mc.renorm_freq == 0:
            wall_t1 = time.time()
            mc.orthogonalization()
            logger.debug(mc, f"Debug: orthogonalise at step {step}")
            mc.wt_ortho += time.time() - wall_t1

        # step 1: propagate walkers
        wall_t1 = time.time()
        propagator.propagate_walkers(
            trial, walkers, vbias, ltensor, eshift=mc.eshift, verbose=int(dump_result)
        )
        mc.wt_propagator += time.time() - wall_t1

        # step 2) weight control
        wall_t1 = time.time()
        mc.walkers.weight_control(step)
        mc.wt_weight_control += time.time() - wall_t1

        # step 3): estimate energies and other properties if needed
        # We store weights, energies, and other properties of each estimator in local
        # buffer_variables and compute the properties at every print_freq
        wall_t1 = time.time()
        mc.property_stack(walkers, step)

        # mc.measurements(walkers, step)
        if (step + 1) % mc.property_calc_freq == 0:
            # Compute energies and other observables
            energies = propagator.local_energy(h1e, ltensor, walkers, trial, enuc=mc.nuc_energy)
            energy = energies[0] / energies[1]

            # Append time and energy to respective lists
            time_list.append(tt)
            energy_list.append(energy)

            # Log the computed energy and other properties
            logstring = (
                f"Step: {step:5d}  {energy:14.7e}  "
                f"{energies[0]:14.7e}  {energies[1]:9.5e}  "
                f"{backend.sum(walkers.weights_org):14.7e}  "
                f"{energies[2]:14.7e}  {energies[3]:14.7e}  "
            )
            if len(energies) > 4:
                logstring += f"{energies[4]:14.7e}  {energies[5]:14.7e}  "
            logstring += f"{time.time() - t0:10.4f}s"

            logger.info(mc, logstring)
            sys.stdout.flush()
        mc.wt_observables += time.time() - wall_t1

        # step 5): TODO: code of checkpoint
        wall_t1 = time.time()
        # if dump_result:
        #     mc.save_checkpoint()
        #     logger.debug(mc, f"local_energy:   {walkers.eloc}")
        mc.wt_io += time.time() - wall_t1

    #
    # TODO: code of analysis, post processing, etc.
    #
    mc.wt_tot = time.time() - wall_t0

    # finalize the propagations
    mc.post_kernel()
    return time_list, energy_list


class QMCbase(object):
    r"""
    Basic QMC class
    """

    def __init__(
        self,
        system,  # or molecule
        mf=None,
        dt=0.005,
        # nsteps=25,
        total_time=5.0,
        num_walkers=100,
        renorm_freq=5,
        random_seed=1,
        taylor_order=6,
        energy_scheme=None,
        propagator_options=None,
        *args,
        **kwargs,
    ):
        r"""

        Args:

           system:      (or molecule) that contains the information about
                        number of electrons, orbitals, Hamiltonain, etc.
           propagator:  propagator class that deals with the way of propagating walkers.
           walker:      Walkers used for open ended random walk.
           renorm_freq: renormalization frequency
           nblocks:     Number of blocks
           nsteps:      Number of steps per block
           OAO:         use OAO representation or not
        """
        # import openms
        # logger.info(self, openms.__logo__)

        if "pra2024" not in runtime_refs:
            runtime_refs.append("pra2024")

        # io params
        self.stdout = sys.stdout
        self.verbose = kwargs.get("verbose", 1)
        self.print_freq = 10

        # system parameters
        self.system = self.mol = system
        self.uhf = kwargs.get("uhf", False)
        self.OAO = kwargs.get("OAO", True)
        #
        # TODO: when uhf is turned offf, check whehter the system is really a closed-shell system
        #
        self.use_so = kwargs.get("use_so", False)
        if "use_so" not in kwargs:
            self.use_so = kwargs.get("use_spinorbital", False)
        if self.uhf:  # set use_so true if uhf is used.
            self.use_so = True

        # TODO: when the eri size is larger than 50% of the available memory,
        # turn on the block_decompose_eri anyway!
        self.block_decompose_eri = kwargs.get("block_decompose_eri", False)
        self.chol_thresh = kwargs.get("chol_thresh", 1.0e-6)

        # check whether it is a eb-AFQMC case
        # Two ways of turning on fermion-boson interactions
        # 1) pass a boson object
        # 2) pass a bare molecule object but with fermion-boson coupling matrix

        self.geb = None  # TODO: optimize the handling of geb
        self.fbinteraction = False # whether this is fermion-boson mixture
        if not isinstance(self.system, Boson): # only check boson_freq is system itself not a boson object
            boson_freq = kwargs.get("boson_freq", None)
            if boson_freq is not None:
                self.system.boson_freq = boson_freq
                self.system.nmodes = len(self.system.boson_freq)
                nphoton = kwargs.get("nphoton", 3)
                self.system.gmat = kwargs.get("gmat", None)
                self.system.nboson_states =  [nphoton for i in range(self.system.nmodes)]
                # print("boson_freq = ", self.system.boson_freq.shape, self.system.boson_freq)
                # print("gmat = ", self.system.gmat)
                self.fbinteraction = True
        else:
            self.fbinteraction = True

        # propagator params
        self.dt = dt
        self.total_time = total_time
        # FIXME: in the future, we will use nsteps to control the propagation,
        # and deprecate the total_time

        self.nsteps = int(total_time / dt)  # nsteps
        # self.nsteps = nsteps  #
        self.propagator = None
        self.nblocks = 500  #
        self.eq_time = 2.0  # time of equilibration phase
        self.eq_steps = int(
            self.eq_time / self.dt
        )  # Number of time steps for the equilibration phase
        self.stablize_freq = 5  # Frequency of stablization(re-normalization) steps
        self.energy_scheme = energy_scheme

        # trial parameters
        self.trial = kwargs.get("trial", None)
        self.mf = mf

        # walker parameters
        # TODO: move these variables into walker object
        self.__dict__.update(kwargs)

        self.taylor_order = taylor_order
        # self.num_walkers = num_walkers
        self.random_seed = random_seed
        self.renorm_freq = renorm_freq
        self.stablize_freq = self.renorm_freq

        # walker_tensors/coeff are moved into walkers class (phiw, weights) respectively
        # self.walker_coeff = None
        # self.walker_tensors = None

        self.hybrid_energy = None
        self.batched = kwargs.get("batched", True)

        # parameters for walker weight control
        self.pop_control_freq = 5  # weight control frequency

        # other variables for Hamiltonian
        self.chol_Xa = None  # TODO: optimize the handling of this term

        # update propagator options
        self.propagator_options = {
            "verbose": self.verbose,
            "stdout": self.stdout,
            "num_fake_fields": 0,
            "energy_scheme": self.energy_scheme,
            "taylor_order": self.taylor_order,
            # electron-boson_mixture
            "decouple_bilinear" : False,
            "turnoff_bosons" : False,
            "quantizaiton": "second",
        }

        self.walker_options = gwalker.default_walker_options
        tmp_walker_options = kwargs.get("walker_options", None)
        # print("tmp_walker_options = ", tmp_walker_options)
        if tmp_walker_options is not None:
            self.walker_options.update(tmp_walker_options)
        if num_walkers is not None:
            self.walker_options["nwalkers"] = num_walkers

        if propagator_options is not None:
            self.propagator_options.update(propagator_options)

        # property calculations
        self.property_calc_freq = kwargs.get("property_calc_freq", 10)
        self.default_properties = ["energy"]  # TODO: more property calculators
        self.stacked_variables = [
            "weights",
            "unscaled_weights",
            "walker_hybrid_energies",
            "walker_local_energies",
        ]  # may add more variables
        self.property_buffer = backend.zeros(
            (len(self.stacked_variables),), dtype=backend.complex128
        )
        self.eshift = 0.0

        # set up calculations
        self.build()  # setup calculations

        # wall time analysis variables
        self.wt_propagator = 0.0
        self.wt_observables = 0.0
        self.wt_weight_control = 0.0
        self.wt_ortho = 0.0
        self.wt_io = 0.0
        self.wt_tot = 0.0


    def build(self):
        r"""
        Build up the afqmc calculations, including:

        1) setup trial WF
        2) setup integrals
        3) setup walkers
        4) setup propagators

        TODO: add the options to get integrals and trials from files
        """
        # 1) set up trial wavefunction
        logger.note(self, task_title("Initialize Trial WF and Walker"))

        # number of spin components
        self.ncomponents = 1
        if self.uhf or self.use_so:
            self.ncomponents = 2
        # if trial is not None, get ncomponents from trial

        self.spin_fac = 1.0 / self.ncomponents

        # FIXME: determine how to better handle trial options from qmc kwargs
        #        may directly make trial_options as a qmc class argument.
        # make a trial_options for creating trail WF.
        trial_options = {
            "verbose": self.verbose,
            "stdout": self.stdout,
            "trail_type": "RHF",
            "numdets": 1,
            "OAO": self.OAO,
            "uhf": self.uhf,
            "ncomponents": self.ncomponents,
        }

        # TODO: 1) simplify this the construction of trial
        #      2) handle different trial, RHF, UHF, GHF, ROHF, and MCSCF.
        if self.trial is None:
            self.trial = make_trial(self.mol, mf=self.mf, **trial_options)
            if self.mf is None:
                self.mf = self.trial.mf
            logger.debug(self, f"Debug: self.mf = {self.mf}")
            logger.debug(self, f"Debug: trail.mf = {self.trial.mf}")
        else:
            logger.info(self, "Trial WF is set from the input")

        # 2) make h1e in Spin orbital
        t0 = time.time()
        logger.note(self, task_title("Get integrals"))
        self.h1e, self.ltensor = self.get_integrals()

        # half-rotate integrals
        self.trial.half_rotate_integrals(self.h1e, self.ltensor)
        logger.note(
            self,
            task_title(f"Get integrals ... Done! Time used: {time.time()-t0: 7.3f} s"),
        )
        if self.geb is not None:
            #trial.init_boson_trial_with_z(self.geb)
            zalpha = backend.einsum("Xpq, pq->X", self.geb, self.trial.Gf[0])
            self.trial.initialize_boson_trial_with_z(zalpha, self.mol.nboson_states)
            logger.debug(self, f"Debug: initial coherent state is  : {zalpha}")
            logger.debug(self, f"Debug: initial bosonic trial WF is: {self.trial.boson_psi}")

        # 3) set up walkers
        t0 = time.time()
        logger.note(self, task_title("Set up walkers"))
        # self.walkers = gwalker.Walkers_so(self.trial, nwalkers=self.num_walkers)
        self.walkers = gwalker.make_walkers(self.trial, self.walker_options)

        # calculate green's function
        ovlp = self.trial.ovlp_with_walkers_gf(self.walkers)
        logger.debug(self, f"Debug: initial trial_walker overlap is\n {ovlp}")
        logger.note(self, f"Setup walkers ... Done! Time used: {time.time()-t0: 7.3f} s")

        # 4) prepare the propagator
        # TODO: may use a dict to switch different propagator
        t0 = time.time()
        logger.note(self, task_title("Prepare propagator"))
        if self.fbinteraction:
            logger.note(
                self,
                "\nsystem is a electron-boson coupled system!"
                + "\nPhaselessElecBoson propagator is to be used!\n",
            )
            self.propagator_options["nbarefields"] = self.nbarefields
            self.propagator = PhaselessElecBoson(self.dt, **self.propagator_options)
            self.propagator.chol_Xa = self.chol_Xa
        else:
            logger.note(
                self,
                "\nsystem is a bare electronic system!"
                + "\nPhaseless propagator is to be used!\n",
            )
            self.propagator = Phaseless(dt=self.dt, **self.propagator_options)
        # FIXME: temporarily assign system to propgator as well
        # FIXME: need to decide how to handle access system from the propagation
        self.propagator.system = self.system

        self.propagator.build(self.h1e, self.ltensor, self.trial, self.geb)

        logger.note(
            self,
            task_title(
                f"Prepare propagator ... Done! Time used: {time.time()-t0: 7.3f} s"
            ),
        )


    @abstractmethod
    def cast2backend(self):
        r"""cast the tensors to backend for following simulation"""
        raise NotImplementedError


    def dump_flags(self):
        r"""dump flags"""

        title = f"{self.__class__.__name__} simulation using OpenMS package"
        logger.note(self, task_title(title, level=0))

        representation = "OAO" if self.OAO else "MO"
        logger.note(self, f" {representation} representation is used")
        logger.note(self, f" Time step              : {self.dt:7.3f}")
        logger.note(self, f" Total time             : {self.total_time:7.3f}")
        logger.note(self, f" Number of steps        : {self.nsteps:7d}")
        logger.note(self, f" Energy scheme          : {self.energy_scheme}")
        logger.note(self, f" Number of chols        : {self.ltensor.shape[0]:5d}")
        logger.note(self, f" Threshold of chols     : {self.chol_thresh:7.3e}")
        logger.note(self, f" Use Spin orbital?      : {self.use_so}")
        logger.note(self, f" Unrestricted spin?     : {self.uhf}")
        logger.note(self, f" No. of spin components : {self.ncomponents:5d}")

        # flags of propagators (walkers)
        logger.note(
            self, f" Propagator is          : {self.propagator.__class__.__name__}"
        )
        self.propagator.dump_flags()

        # flags of trial WF and walkers
        self.trial.dump_flags()
        self.walkers.dump_flags()


    def measurements(self, walkers, step):
        r"""masurement of physical observables, e.g., energies"""

        pass


    def get_integrals(self):
        r"""return oei, eri, and cholesky tensors in OAO or MO

        .. note::

           this part will be replaced by the code in tools which provide
           either full of block decomposition of the eri.
        """

        # with h5py.File("input.h5") as fa:
        #    ao_coeff = fa["ao_coeff"][()]

        hcore = scf.hf.get_hcore(self.mol)
        logger.debug(self, f"Debug: norm of hcore = {backend.linalg.norm(hcore)}")
        if self.OAO:
            # Lowdin orthogonalization S^{-/2} -> X
            overlap = self.mol.intor("int1e_ovlp")
            Xmat = lo.orth.lowdin(overlap)
        else:
            Xmat = self.mf.mo_coeff
        logger.debug(self, f"Debug: basis_transform_matrix = \n{Xmat}")
        norb = Xmat.shape[0]

        #
        # old code, will remove this part, as we don't need the eri
        #

        # get h1e, and ori in OAO representation
        mol = self.mol._mol if isinstance(self.mol, Boson) else self.mol
        if not self.block_decompose_eri:
            import tempfile
            ftmp = tempfile.NamedTemporaryFile()

            pyscftools.fcidump.from_mo(mol, ftmp.name, Xmat)
            hcore, eri, self.nuc_energy = tools.read_fcidump(ftmp.name, norb)

            # Cholesky decomposition of eri
            # here eri uses chemist's notation
            # [il|kj]  a^\dag_i a^\dag_j a_k a_l  = <ij|kl> a^\dag_i a^\dag_j a_k a_l
            # [] -- chemist, <> physicist notations

            # Cholesky decomposition of eri (ij|kl) -> L_{\gamma,ij} L_{\gamma,kl}
            eri_2d = eri.reshape((norb**2, -1))
            u, s, v = scipy.linalg.svd(eri_2d)
            mask = s > self.chol_thresh
            # ltensor = u * backend.sqrt(s)
            ltensor = u[:, mask] * backend.sqrt(s[mask])
            ltensor = ltensor.T
            ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        else:
            # ---------------------------------------
            #  use block decomposition of eri in tools
            # ---------------------------------------
            # get h1e, eri, ltensors in OAO/MO representation
            hcore, ltensor, self.nuc_energy = tools.get_h1e_chols(
                mol, Xmat=Xmat, thresh=self.chol_thresh
            )

        # shape of h1e [nspin, nao, nao]
        h1e = backend.array([hcore for _ in range(self.ncomponents)])
        logger.debug(self, f"\nDebug: h1e.shape = {h1e.shape}")
        logger.debug(self, f"Debug: hcore norm in OAO/MO = {backend.linalg.norm(h1e[0])}")
        logger.debug(self, f"Debug: chol norm in OAO/MO  = {backend.linalg.norm(ltensor)}")

        self.nbarefields = ltensor.shape[0]
        # with h5py.File("input.h5", "r+") as fa:
        #    fa["h1e"] = h1e
        #    fa["nuc_energy"] = self.nuc_energy
        #    fa["cholesky"] = ltensor

        if self.fbinteraction: # isinstance(self.system, Boson):
            system = self.system
            # Add boson-mediated oei and eri:
            # shape [nm, nao, nao]

            chol_eb = system.gmat.copy()
            nmodes = system.gmat.shape[0]

            # transform into OAO
            chol_eb = backend.einsum(
                "ik, mkj, jl -> mil", Xmat.conj().T, chol_eb, Xmat, optimize=True
            )

            # add DSE contribution to h1e
            oei_dse = 0.5 * backend.einsum('Xpq, Xqs->ps', chol_eb, chol_eb)
            h1e += backend.array([oei_dse for _ in range(self.ncomponents)])

            # geb is the bilinear coupling term
            tmp = (system.boson_freq * 0.5) ** 0.5
            self.geb = chol_eb * tmp[:, backend.newaxis, backend.newaxis]

            logger.info(self, f"size of chol before adding DSE: {ltensor.shape[0]}")
            # DSE-mediated eri
            if backend.linalg.norm(chol_eb) > 1.0e-10:
                ltensor = backend.concatenate((ltensor, chol_eb), axis=0)
                self.nbarefields += chol_eb.shape[0] # add N_v more chols

            # add terms due to decoupling of bilinear term
            if self.propagator_options["decouple_bilinear"]:
                logger.debug(
                    self, f"creating chol due to decomposition of bilinear term"
                )
                zalpha = backend.einsum("Xpq, pq->X", self.geb, self.trial.Gf[0])

                # TODO: set Afac as input variables as long as
                # Decompose \sqrt{w/2} (\lambda\cdot D}(a^\dagger_v + a_v) as (A_v * \lambda\cdot D) * (O_v * X_v)
                # where X_v = a^\dagger_v + a_v
                # where A_v * O_v = sqrt{w_v/2}
                decoup_Afac = backend.ones(nmodes)
                decoup_Ofac = (system.boson_freq * 0.5) ** 0.5 / decoup_Afac

                # Add the chols due to the decomposition of bilinear term:
                #
                # 1) Original DSE + terms due tot the decomposition is:
                #     - 1/2 * (A_v\lambda_v\cdot D)^2 + 1/2 * (A_v\lambda_v\cdot D + O_v * X_v)^2
                #                            |                              |
                #           a): \sqrt{1-A_v} chol_eb           b): \sqrt{A_v} * chol_eb
                #
                #  i.e., at most 2N_b more tensors will be appended (N_b number of bosonic modes)
                #
                # 2) Bosonic part from the decomposition:
                #   1/2 * (A_v \lambda_v \cdot D + O_v * X_v)^2 - 1/2 * (O_v * X_v)^2
                #                                     |                     |
                #                           a)   sqrt{O_v}X_v    b)  j * sqrt{O_v} X_v
                # factors = [backend.sqrt(backend.ones(nmodes) - decoup_Afac, dtype=complex), backend.sqrt(decoup_Afac)]
                factors = [(1 + 1j) * decoup_Afac]
                factors = [-1j*decoup_Afac, decoup_Afac]
                for factor in factors:
                    Lga = chol_eb * factor[:, backend.newaxis, backend.newaxis]
                    for imode in range(nmodes):
                        if backend.linalg.norm(Lga[imode]) > 1.0e-10:
                            ltensor = backend.concatenate((ltensor, Lga[imode][backend.newaxis,...]), axis=0)

                # [1, 1j] * Ofac * (a^\dagger_v + a_v)
                chol_Xa =  decoup_Ofac
                self.chol_Xa =  backend.array([chol_Xa * (1 + 1j)])
                self.chol_Xa =  backend.array([chol_Xa, -1j * chol_Xa])
                logger.debug(self, f"Debug: chol_Xa.shape = {chol_Xa.shape}")
                logger.debug(self, f"Debug: chol_Xa.shape = {self.chol_Xa.shape}\n")
                logger.debug(self, f"size of bosonic chol {self.chol_Xa.shape[0]}\n")

            for i in range(ltensor.shape[0]):
                evals, evecs = scipy.linalg.eigh(ltensor[i])
                logger.debug(self, f"evals of ltensor[{i}] in OAO: {evals}")

            logger.info(self, f"size of chol after adding DSE:  {ltensor.shape[0]}")
            logger.debug(self, f"Norm of ltensor (bare):  {backend.linalg.norm(ltensor[:-1])}")
            logger.debug(self, f"Norm of ltensor (L[0]):  {backend.linalg.norm(ltensor[0])}")
            logger.debug(self, f"Norm of ltensor (eb):    {backend.linalg.norm(ltensor[-1])}")

        self.nfields = ltensor.shape[0]
        return h1e, ltensor


    def propagate_walkers(self, walkers, xbar, ltensor):
        pass


    def measure_observables(self, operator):
        r"""Placeholder for measure_observables.
        According to the operator, we measure the expectation values

        TODO: may not used this one!
        """
        observables = None
        return observables


    def walker_trial_overlap(self):
        r"""
        Compute the overlap between trial and walkers:

        .. math::

            \langle \Psi_T \ket{\Psi_w} = \det[S]

        where

        .. math::

            S = C^*_{\psi_T} C_{\psi_w}

        and :math:`C_{\psi_T}` and :math:`C_{\psi_w}` are the coefficient matrices
        of Trial and Walkers, respectively.
        """

        warnings.warn(
            "The 'walker_trial_overlap' function is deprecated and will be removed in a future version. "
            "Please use the 'trial.ovlp_with_walkers' function instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return backend.einsum("pr, zpq->zrq", self.trial.psi.conj(), self.walkers.phiw)

    def orthogonalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers

        .. note::

            Since the orthogonaization depends on the type of walkers, we are moving
            this function into walkers. Hence, the function here is to be deprecated!
        """

        # 1) old code
        # ortho_walkers = backend.zeros_like(self.walkers.phiwa)
        detR = backend.zeros(self.walkers.nwalkers, dtype=backend.complex128)
        for iw in range(self.walkers.phiwa.shape[0]):
            self.walkers.phiwa[iw], log_det = qr_ortho(self.walkers.phiwa[iw])

            if self.walkers.ncomponents > 1:
                self.walkers.phiwb[iw], log_det_b = qr_ortho(self.walkers.phiwb[iw])
                log_det += log_det_b

            detR[iw] = backend.exp(log_det - self.walkers.detR_shift[iw])
            self.walkers.log_detR[iw] += backend.log(detR[iw])
            self.walkers.detR[iw] = detR[iw]
            self.walkers.ovlp[iw] = self.walkers.ovlp[iw] / detR[iw]

        """
        # 2) batched code
        self.walkers.phiwa, log_det = qr_ortho_batch(self.walkers.phiwa)
        if self.walkers.ncomponents > 1:
            self.walkers.phiwb, log_det_b = qr_ortho_batch(self.walkers.phiwb)
            log_det += log_det_b

        self.walkers.detR = backend.exp(log_det - self.walkers.detR_shift)
        self.walkers.ovlp = self.walkers.ovlp / self.walkers.detR
        """


        if self.walkers.boson_phiw is not None:
            ortho_walkers = backend.zeros_like(self.walkers.boson_phiw)
            norms = backend.einsum('ij,ij->i', self.walkers.boson_phiw, self.walkers.boson_phiw.conj())
            norms = backend.sqrt(norms)
            self.walkers.boson_phiw = self.walkers.boson_phiw / norms[:, None]
            # for iw in range(self.walkers.boson_phiw.shape[0]):
            #    ortho_walkers[iw] = backend.linalg.qr(self.walkers.boson_phiw[iw])[0]
            # self.walkers.boson_phiw = ortho_walkers

    # renormalization is to be deprecated
    orthonormalization = orthogonalization

    def local_energy_spin(self, h1e, eri, G1p):
        r"""Compute local energy

        .. math::

             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}
        """
        # E_coul
        tmp = 2.0 * backend.einsum("prqs,zSpr->zqs", eri, G1p) * self.spin_fac
        ecoul = backend.einsum("zqs,zSqs->z", tmp, G1p)
        # E_xx
        tmp = backend.einsum("prqs,zSps->zSqr", eri, G1p)
        exx = backend.einsum("zSqs,zSqs->z", tmp, G1p)
        e2 = (ecoul - exx) * self.spin_fac

        e1 = 2.0 * backend.einsum("zSpq,pq->z", G1p, h1e) * self.spin_fac

        energy = e1 + e2 + self.nuc_energy
        return energy

    def local_energy(self, TL_theta, h1e, eri, vbias, gf):
        r"""Compute local energy from oei, eri and GF.

        Warning: this function is Deprecaeted and moved to propagator to handle
        the cases of different trial and walkers.

        Args:
            gf: green function

        .. math::

             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}

        if :math:`L_\gamma` tensor is used
        [PS: need to rotate Ltensor into (nocc, norb) shape since G's shape is (nocc, norb)],

        .. math::

             E = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{ps\sigma} G_{qr\sigma} \\
               = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{\gamma,pq\sigma\sigma'} (L_\gamma G_\sigma)_{pq} (L_\gamma G_{\sigma'})_{pq}
                 - \frac{1}{2}\sum_{\gamma,\sigma} [\sum_{pq} L_{\gamma,pq} G_{pq\sigma}]^2

        i.e. the Ecoul is :math:`\left[\frac{\bra{\Psi_T}L\ket{\Psi_w}}{\bra{\Psi_T}\Psi_w\rangle}\right]^2`,
        which is the TL_Theta tensor in the code
        """

        warnings.warn(
            "The qmc.local_energy function is deprecated and will be removed in a future version. "
            "Please use 'estimators.local_eng_elec or local_eng_elec_chol' function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # approach 0) : most efficient way to compute the energy: use Ltensors instead of eri
        vbias2 = vbias * vbias
        ej = 2.0 * backend.einsum("zn->z", vbias2)
        ek = backend.einsum("znpr, znrp->z", TL_theta, TL_theta)
        e2 = ej - ek

        # approach 1) : most inefficient way
        # e2 = 2.0 * backend.einsum("prqs, zpr, zqs->z", eri, gf, gf)
        # e2 -= backend.einsum("prqs, zps, zqr->z", eri, gf, gf)

        # approach 3): use normal way without using ltensors
        # vjk = 2.0 * backend.einsum("prqs, zpr->zqs", eri, gf) # E_coulomb
        # vjk -= backend.einsum("prqs, zps->zqr", eri, gf)  # exchange
        # e2 = backend.einsum("zqs, zqs->z", vjk, gf)

        e1 = 2.0 * backend.einsum("zpq, pq->z", gf, h1e)
        energy = e1 + e2 + self.nuc_energy
        return energy

    # we use the following function to deal with property calculations
    def property_stack(self, walkers, step):
        r"""
        The function handles the accumulation and periodic reduction of properties.

        Parameters:
            walkers: object
                Contains the walker states and associated properties.
            step: int
                The current simulation step.

        """
        # property_list = self.default_properties
        if step < 0:
            # Initial setup for the property buffer
            logger.debug(
                self, f"Debug: initial buffer shape (in accumulator) is {self.property_buffer.shape}"
            )
            # TODO: compute the initial values of the properties
            return

        # Dictionary linking property names to their computed values
        _data_dict = {
            "weights": backend.sum(walkers.weights),
            "unscaled_weights": backend.sum(walkers.weights_org),
            "walker_hybrid_energies": backend.sum(walkers.ehybrid * walkers.weights),
            "walker_local_energies": backend.sum(walkers.eloc * walkers.weights),
        }

        # Accumulate values for the specified properties
        tmp = [
            _data_dict.get(key, 0.0 + 0.0j) for key in self.stacked_variables
        ]
        self.property_buffer += backend.array(tmp)

        # logger.debug(self, f"Debug: updated buffer shape is {self.property_buffer.shape}")
        # logger.debug(self, f"Debug: updated buffer is {self.property_buffer}")

        # Perform periodic property reduction and normalization
        if (step + 1) % self.property_calc_freq == 0:

            # Normalize energies by weights
            weights_idx = self.stacked_variables.index("weights")
            norm = self.property_buffer[weights_idx]

            for idx, name in enumerate(self.stacked_variables):
                if "energies" in name:
                    self.property_buffer[idx] /= norm

            # Note: dont' combine the following loop with above one, for energies,
            # we only need to normalized it against weights
            # Normalize weights over the calculation frequency
            for idx, name in enumerate(self.stacked_variables):
                if "weights" in name:
                    self.property_buffer[idx] /= self.property_calc_freq

            # Update the energy shift using normalized hybrid energies
            idx = self.stacked_variables.index("walker_hybrid_energies")
            self.eshift = self.property_buffer[idx]
            logger.debug(self, f"Debug: update eshift = {self.eshift}")

            # Reset the property buffer for the next accumulation cycle
            self.property_buffer.fill(0.0 + 0.0j)


    def kernel(self, trial_wf=None):
        r"""main function for QMC time-stepping

        trial_wf: trial wavefunction
        walkers: walker function

        """

        self.dump_flags()

        # Note of the rename
        # rename precomputed_ltensor -> TL_tensor
        # rename ltheta -> TL_theta
        # trace_ltheta -> trace_tltheta -> vbias

        # prepare propagation
        logger.info(self, f"\n Random seed is {self.random_seed}\n")
        backend.random.seed(self.random_seed)

        # print("YZ: walkers WF      =", self.walkers.phiw)
        # print("YZ: walkers weights =", self.walkers.weights)

        h1e = self.h1e
        # eri = self.eri
        ltensor = self.ltensor
        propagator = self.propagator

        trial = self.trial if trial_wf is None else trial_wf
        walkers = self.walkers

        # setup propagator
        # self.build_propagator(h1e, eri, ltensor)
        propagator.build(h1e, ltensor, trial, self.geb)

        logger.debug(self, f"Debug: the initial orthogonalise in walker")
        self.orthogonalization()

        # initialize property calculations
        self.property_stack(walkers, -1)

        # start the propagation
        tt = 0.0
        energy_list = []
        time_list = []
        wall_t0 = time.time()
        logstring = f"{'Step':^10}{'Etot':^16}{'Raw_Etot':^16}{'Norm':^14}{'Raw_Norm':^14}{'E1':^16}{'E2':^16}"
        if isinstance(propagator, PhaselessElecBoson):
            logstring += f"{'Eb':^16}{'Eg':^16}"
        logstring += "  Wall_time"
        logger.info(self, logstring)

        # while tt <= self.total_time:
        for step in range(self.nsteps):
            t0 = time.time()
            tt = self.dt * step
            dump_result = step % self.print_freq == 0
            logger.debug(
                self, f"\nDebug: -------------- qmc step {step} -----------------"
            )

            # step 3): periodic re-orthogonalization
            # (FIXME: whether put this at the begining or end, in principle, should not matter)
            if (step + 1) % self.renorm_freq == 0:
                wall_t1 = time.time()
                self.orthogonalization()
                logger.debug(self, f"Debug: orthogonalise at step {step}")
                self.wt_ortho += time.time() - wall_t1

            vbias = None
            # step 1): get force bias (note: TL_tensor and mf_shift moved into propagator.atrributes)
            # store Gf in walkers in order to recycle it in the propagators
            # gf, vbias = trial.get_vbias(walkers, ltensor, verbose=dump_result)
            """
            gf, TL_theta = trial.force_bias(
                walkers, propagator.TL_tensor, verbose=dump_result
            )

            # trace[TL_theta] is the force_bias
            vbias = backend.einsum("znpp->zn", TL_theta)

            # imaginary time propagation
            # TODO: may apply bias bounding
            xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - propagator.mf_shift)
            """

            wall_t1 = time.time()
            # step 3): propagate walkers and update weights
            # self.propagate_walkers(walkers, xbar, ltensor)
            propagator.propagate_walkers(
                trial, walkers, vbias, ltensor, eshift=self.eshift, verbose=int(dump_result)
            )
            self.wt_propagator += time.time() - wall_t1

            # step 2) weight control
            wall_t1 = time.time()
            self.walkers.weight_control(step)
            self.wt_weight_control += time.time() - wall_t1

            # moved phaseless approximation to propagation
            # since it is associated with propagation type
            # self.update_weight(overlap, cfb, cmf)

            # step 4): estimate energies and other properties if needed
            # We store weights, energies, and other properties of each estimator in local
            # buffer_variables and compute the properties at every print_freq
            wall_t1 = time.time()
            self.property_stack(walkers, step)

            # self.measurements(walkers, step)
            if (step + 1) % self.property_calc_freq == 0:
                # Compute energies and other observables
                energies = propagator.local_energy(h1e, ltensor, walkers, trial, enuc=self.nuc_energy)
                energy = energies[0] / energies[1]

                # Append time and energy to respective lists
                time_list.append(tt)
                energy_list.append(energy)

                # Log the computed energy and other properties
                logstring = (
                    f"Step {step:5d}  {energy:14.7e}  "
                    f"{energies[0]:14.7e}  {energies[1]:9.5e}  "
                    f"{backend.sum(walkers.weights_org):14.7e}  "
                    f"{energies[2]:14.7e}  {energies[3]:14.7e}  "
                )
                if len(energies) > 4:
                    logstring += f"{energies[4]:15.7e}  {energies[5]:15.7e}  "
                logstring += f"{time.time() - t0:10.4f}s"

                logger.info(self, logstring)
                sys.stdout.flush()
            self.wt_observables += time.time() - wall_t1

            # step 5): TODO: code of checkpoint
            wall_t1 = time.time()
            # if dump_result:
            #     self.save_checkpoint()
            #     logger.debug(self, f"local_energy:   {walkers.eloc}")
            self.wt_io += time.time() - wall_t1

        #
        # TODO: code of analysis, post processing, etc.
        #
        self.wt_tot = time.time() - wall_t0

        # finalize the propagations
        self.post_kernel()
        return time_list, energy_list

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""

        # TODO: print summary & post-processing data, etc.


        logger.note(self, task_title("Wall time analysis"))
        logger.note(self, f" Total             : {self.wt_tot: 9.3f}")
        logger.note(self, f" IO                : {self.wt_io: 9.3f}")
        logger.note(self, f" Measuremnets      : {self.wt_observables: 9.3f}")
        logger.note(self, f" Weight control    : {self.wt_weight_control: 9.3f}")
        logger.note(self, f" Orthogonalization : {self.wt_ortho: 9.3f}")
        logger.note(self, f" Propagator        : {self.wt_propagator: 9.3f}")
        logger.note(self, f"\n Breakdown of propagator:")
        logger.note(self, f"   Overlap & GF    : {self.propagator.wt_ovlp: 9.3f}")
        logger.note(self, f"   Updte weights   : {self.propagator.wt_weight: 9.3f}")
        logger.note(self, f"   Onebody term    : {self.propagator.wt_onebody: 9.3f}")
        logger.note(self, f"   Build h1e term  : {self.propagator.wt_buildh1e: 9.3f}")
        logger.note(self, f"   Twobody term    : {self.propagator.wt_twobody: 9.3f}")
        logger.note(self, f"   Bilinear term   : {self.propagator.wt_bilinear: 9.3f}")
        logger.note(self, f"   Bosonic term    : {self.propagator.wt_boson: 9.3f}")
        logger.note(self, f"   Breakdown of twobody:")
        logger.note(self, f"     Force bias    : {self.propagator.wt_fbias: 9.3f}")
        logger.note(self, f"     HS of twobody : {self.propagator.wt_hs: 9.3f}")
        logger.note(self, f"     build HS      : {self.propagator.wt_chs: 9.3f}")
        logger.note(self, f"     Propagate HS  : {self.propagator.wt_phs: 9.3f}")
        # more wall times TBA.
        logger.note(self, f"")


    def post_kernel(self):
        r"""Prints relevant citation information for calculation."""
        breakline = "=" * 86
        logger.note(self, f"\n{breakline}")
        logger.note(self, f"*  Hoollary, the job is done!\n")

        self._finalize()

        logger.note(self, task_title("Citations"))
        for i, key in enumerate(runtime_refs):
            logger.note(self, f"[{i+1}]. {_citations[key]}")
        logger.note(self, f"{breakline}\n")
