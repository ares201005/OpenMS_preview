import os
import sys
import h5py
import numpy as np

from pyscf import gto

from openms.mqed import scqedhf


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


def run_scqedhf(
    molecule = "LiH",
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    guess_dm = None,
    guess_eta = None
):
    """Perform SC-QED-HF calculation."""

    # Molecule
    mol = get_mol(name=molecule, basis=basis_set)
    mol.verbose = 3
    mol.build()

    # Cavity
    nmodes = 1
    cavity_freq, cavity_mode = get_cavity(nmode=nmodes, gfac=lmbda, omega=freq)

    # SC-QED-HF kernel
    scqedmf = scqedhf.RHF(
        mol = mol,
        cavity_mode = cavity_mode,
        cavity_freq = cavity_freq,
        add_nuc_dipole = True,
        nboson_states = nfock,
    )
    scqedmf.max_cycle = 500
    scqedmf.init_guess = "hcore"
    #scqedmf.conv_tol = 1e-8
    #scqedmf.diis_space = 30
    #scqedmf.precond = 8e-3

    if guess_dm is not None:
        if guess_eta is not None:
            scqedmf.kernel(dm0=guess_dm, init_params=guess_eta)
        else:
            scqedmf.kernel(dm0=guess_dm)
    else:
        scqedmf.kernel()

    # Energies
    e_tot = scqedmf.e_tot
    e_boson = scqedmf.qed.e_boson

    # Density matrix
    return_dm = None
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Eta parameters
    return_eta_val = None
    if guess_eta is not None:
        return_eta_val = guess_eta.copy()

    # Only create guess if previous calculation converged
    conv = scqedmf.converged
    if conv:
        return_dm = scqedmf.make_rdm1()
        return_eta_val = scqedmf.eta

    photon_occ = None
    if conv:
        photon_occ = scqedmf.qed.get_boson_occ()

    del scqedmf
    return return_dm, return_eta_val, e_tot, e_boson, photon_occ, conv


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

    # Store guess data
    scqed_dm = None
    scqed_eta = None

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "SCQEDHF"
        skip_calc = False

        # Datafile exists
        if os.path.exists(h5_name):
            with h5py.File(h5_name, "a") as f:

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

        # Value lists
        etot = []
        eboson = []
        photon_occ = []
        eta = []

        # Run SC-QED-HF with increasing nFock
        nf = 0
        while e_diff >= 1.0e-6:

            # Increase number of Fock states
            nf += 1
            nf_list = [nf]

            # Run SC-QED-HF
            scqed_dm, scqed_eta, e_tot, e_boson, p_occ, conv = run_scqedhf(
                molecule = molecule,
                basis_set = basis,
                lmbda = la,
                nfock = nf_list,
                freq = omega,
                guess_dm = scqed_dm,
                guess_eta = scqed_eta
            )

            # Append values to list
            etot.append(e_tot)
            eboson.append(e_boson)
            photon_occ.append(p_occ)
            eta.append(scqed_eta)

            # Update energy difference if SCF converges
            if conv:
                e_diff = np.abs(e_tot - e_prev)
                e_prev = 0.0 + e_tot

        # Ediff criteria met
        print (f"\nSC-QED-HF CONVERGED W/ NUMBER OF FOCK STATES!\n "
               f"GFAC={la:.3f}  NFOCK={nf}\n")
        sys.stdout.flush()

        # write data to file
        with h5py.File(h5_name, "a") as fa:
            lambda_grp = fa.require_group(lambda_group)
            method_grp = lambda_grp.require_group(method_group)

            method_grp.create_dataset(f"energy", data=np.array(etot))
            method_grp.create_dataset(f"boson_energy", data=np.array(eboson))
            method_grp.create_dataset(f"photon_occ", data=np.array(photon_occ))
            method_grp.create_dataset(f"eta", data=np.array(eta))
            method_grp.create_dataset(f"nfock", data=nf)
