#
# @ 2026. Triad National Security, LLC. All rights reserved.
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
# Authors:  Yu Zhang            <zhy@lanl.gov>
#           Jacob Z. Williams   <jzw@lanl.gov>
#
# Created:          2026-03-17
# Last modified:    2026-03-17
#

import unittest
import numpy as np
from pyscf import gto, scf
from openms.mqed import qedhf as QEDHF
from openms.qmc.afqmc import AFQMC
from openms.lib.boson import get_dipole_ao
from openms.qmc.tools import get_mean_std
from molecules import get_mol, get_cavity

# Molecular system
atom = """
    H   0.0000      0.0000      0.0000
    F   0.0000      0.0000      1.1000
    """

mol = gto.M(atom=atom, basis="sto-3g")


# Couple to a cavity
g_bare = 0.1
omega = 0.5
nfield_max = 20
nmode_max = 6

mf = scf.RHF(mol)
E_hf = mf.kernel()

# AFQMC parameters
dt = 0.05
total_time = 10.0
num_walkers = 100
energy_scheme = "hybrid"

# Bare AFQMC
qmc_bare = AFQMC(
    mol,
    mf=mf,
    dt=dt,
    total_time=total_time,
    num_walkers=num_walkers,
    energy_scheme=energy_scheme,
)
tlist, Elist = qmc_bare.kernel()
E_qmc, std = get_mean_std(Elist)
print(
    "[nmode, nwalker, time], means, stds = ",
    0,
    num_walkers,
    total_time,
    E_qmc,
    std,
)


# Couple to multiple modes: strength scales as g/sqrt(N_mode)
E_mode = [E_qmc]
std_mode = [std]
for nmode in range(1, nmode_max + 1):

    num_fake_fields = nfield_max - 2 * nmode
    g_eff = g_bare / np.sqrt(nmode)

    cavity_mode = np.zeros((nmode, 3))
    cavity_mode[:, 1] = g_eff

    dipole = get_dipole_ao(mol)
    gmat = np.einsum("ax, xij -> aij", cavity_mode, dipole)

    test_gmat = np.zeros_like(gmat)

    boson_freq = omega * np.ones(nmode, dtype=float)

    qmc = AFQMC(
        mol,
        verbose=1,
        mf=mf,
        dt=dt,
        nmode=nmode,
        boson_freq=boson_freq,
        gmat=gmat,
        num_walkers=num_walkers,
        # gmat=test_gmat,
        energy_scheme=energy_scheme,
        propagator_options={
            "decouple_bilinear": True,
            "decouple_scheme": 2,
            "num_fake_fields": num_fake_fields,
        },
    )
    _, Elist = qmc.kernel()

    E_qmc, std = get_mean_std(Elist)
    print(
        "[nmode, nwalker, time], means, stds = ",
        nmode,
        num_walkers,
        total_time,
        E_qmc,
        std,
    )
    E_mode.append(E_qmc)
    std_mode.append(std)


print(f"HF energy    = {E_hf:.6f}\n")
print("QMC parameters:")
print(f"dt          = {dt}")
print(f"T_max       = {total_time}")
print(f"num_walkers = {num_walkers}")
for i in range(nmode_max):
    print(f"{i} modes: E = {E_mode[i]:.6f} ± {std_mode[i]:.6f}")


# TODO: Add a flag to make the auxiliary fields the same for each bosonic mode
