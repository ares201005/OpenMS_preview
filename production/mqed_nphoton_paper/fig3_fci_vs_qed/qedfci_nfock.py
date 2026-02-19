import os
import sys
import h5py
import numpy as np

from pyscf import gto

from openms.lib.boson import get_integrals4fci
from openms.mqed import qedfci


def get_mol(name="LiH", basis="sto-3g"):
    """Return Molecule object."""

    atom = None

    if name.lower() == "lih":
        atom = f"Li    0.000    0.000    0.000;\
                  H    0.000    0.000    1.600"

    elif name.lower() == "lif":
        atom = f"Li    0.000    0.000    0.000;\
                  F    0.000    0.000    1.560"

    return gto.Mole(
        atom = atom,
        basis = basis,
        unit = "Angstrom",
        symmetry = True,
        verbose = 1
    )


def get_cavity(nmode=1, gfac=0.0, omega=0.5):
    """Return tuple of cavity parameters."""

    cavity_freq = np.zeros(nmode)
    cavity_mode = np.zeros((nmode, 3))

    cavity_freq[0] = omega
    cavity_mode[0, :] = gfac * np.asarray([0, 0, 1])

    return cavity_freq, cavity_mode


def run_qedfci(
    molecule = "LiH",
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
):
    """Perform QED-FCI calculation."""

    # Molecule
    mol = get_mol(name=molecule, basis=basis_set)
    mol.verbose = 3
    mol.build()

    # System values
    nao = mol.nao_nr()
    nelec = mol.nelectron

    # Cavity
    nmodes = 1
    cavity_freq, cavity_mode = get_cavity(nmode=nmodes, gfac=lmbda, omega=freq)

    # FCI integrals
    enuc, __, H1, H2, Hep, __ = get_integrals4fci(
        mol = mol,
        cavity_freq = cavity_freq,
        cavity_mode = cavity_mode,
    )

    # FCI kernel
    efci_elec, efci, ci = qedfci.kernel(
        H1, H2, nao, nelec, nmodes, nfock,
        Hep, cavity_freq,
        coherent_state = False,
        ecore = enuc,
        tol = 1e-8,
        verbose = 1,
        #max_cycle = 100,
        max_cycle = 1000,
    )

    ph_dm = None
    #ph_dm = qedfci.make_rdm1p(ci, nao, nelec, nmodes, nfock)

    print(f"\nQED-FCI energy = {efci:.13f}\n")
    sys.stdout.flush()

    return efci_elec, efci, ph_dm


if __name__ == "__main__":

    # Molecule
    #molecule = "LiF"
    molecule = "LiH"

    # Basis set
    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "cc-pvdz"

    # Create/check for directory
    mol_dir = f"{basis}_{molecule}"
    if not os.path.exists(mol_dir):
        os.makedirs(mol_dir)

    # Datafile name
    h5_name = f"{mol_dir}/qed_e_vs_fci_lambda.h5"

    # Frequency
    omega = 0.5

    # Command-line argument
    la_vals = None
    if len(sys.argv) > 1:
        la_vals = [float(sys.argv[1])]

    # Scan range of values
    else:
        la_start = 0.000
        la_end = 0.300
        la_skip = 0.010

        la_vals = np.arange(la_start, la_end + la_skip, la_skip)

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "QEDFCI"
        skip_calc = False

        # Datafile exists
        if os.path.exists(h5_name):
            with h5py.File(h5_name, "r") as f:

                # Method run for provided lambda value
                if lambda_group in f:
                    lam_grp = f[lambda_group]
                    if method_group in lam_grp:
                        print(f"{method_group} data exists for lambda = {la:.3f} in file {h5_name}. Skipping...\n")
                        skip_calc = True

        # Only run new coupling values
        if skip_calc:
            sys.stdout.flush()
            continue

        # Create initial file
        if not os.path.exists(h5_name):
            with h5py.File(h5_name, 'w') as f: pass
            print(f"Created empty file: {h5_name}.\n Starting calculation...\n")
            sys.stdout.flush()

        # Energy differences
        e_prev = 0.0
        e_diff = 1000.

        # Energy lists
        etot = []
        efci = []
        eboson = []
        photon_occ = []

        # Run QED-HF with increasing nFock
        nf = -1
        while e_diff >= 1.0e-6:

            # Increase number of Fock states
            nf += 1
            nf_list = [nf]

            # Run QED-FCI
            fci_elec, fci_energy, ph_dm = run_qedfci(
                molecule = molecule,
                basis_set = basis,
                lmbda = la,
                nfock = nf_list,
                freq = omega,
            )
            #print (ph_dm)

            # Save energies to list
            etot.append(fci_energy)
            efci.append(fci_elec)
            eboson.append(fci_energy - fci_elec)
            #photon_occ.append(p_occ)

            # Update energy difference if SCF converges
            e_diff = np.abs(fci_energy - e_prev)
            e_prev = 0.0 + fci_energy

        # Ediff criteria met
        print (f"\nQED-FCI CONVERGED W/ NUMBER OF FOCK STATES!\n "
                f"GFAC={la:.3f}  NFOCK={nf}\n")
        sys.stdout.flush()

        # write data to file
        with h5py.File(h5_name, "a") as fa:
            lambda_grp = fa.require_group(lambda_group)
            method_grp = lambda_grp.require_group(method_group)

            method_grp.create_dataset(f"fci_energy", data=np.array(etot))
            method_grp.create_dataset(f"fci_elec", data=np.array(efci))
            method_grp.create_dataset(f"fci_boson_energy", data=np.array(eboson))
            method_grp.create_dataset(f"fci_nfock", data=nf)

