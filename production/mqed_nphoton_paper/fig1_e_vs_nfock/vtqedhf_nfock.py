import os
import sys
import h5py
import numpy as np

from pyscf import gto

from openms.mqed import vtqedhf


def get_mol(basis="sto-3g"):
    """Return Molecule object."""

    atom = f"C   0.00000000   0.00000000    0.0000;\
             O   0.00000000   1.23456800    0.0000;\
             H   0.97075033  -0.54577032    0.0000;\
             C  -1.21509881  -0.80991169    0.0000;\
             H  -1.15288176  -1.89931439    0.0000;\
             C  -2.43440063  -0.19144555    0.0000;\
             H  -3.37262777  -0.75937214    0.0000;\
             O  -2.62194056   1.12501165    0.0000;\
             H  -1.71446384   1.51627790    0.0000"

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


def run_vtqedhf(
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    guess_dm = None,
    guess_eta = None,
    guess_fopt = None
):
    """Perform VT-QED-HF calculation."""

    # Molecule
    mol = get_mol(basis_set)
    mol.verbose = 4
    mol.build()

    # Cavity
    nmodes = 1
    cavity_freq, cavity_mode = get_cavity(nmode=nmodes, gfac=lmbda, omega=freq)

    # VT-QED-HF kernel
    vtqedmf = vtqedhf.RHF(
        mol = mol,
        cavity_mode = cavity_mode,
        cavity_freq = cavity_freq,
        add_nuc_dipole = True,
        nboson_states = nfock
    )
    vtqedmf.max_cycle = 500
    vtqedmf.init_guess = "hcore"
    #vtqedmf.conv_tol = 1e-8
    #vtqedmf.diis_space = 30
    #vtqedmf.precond = 8e-3

    if guess_dm is not None:
        if (guess_eta is not None) and (guess_fopt is not None):
            guess_params = [guess_eta, guess_fopt]
            vtqedmf.kernel(dm0=guess_dm, init_params=guess_params)
        else:
            vtqedmf.kernel(dm0=guess_dm)
    else:
        vtqedmf.kernel()

    # Energies
    e_tot = vtqedmf.e_tot
    e_boson = vtqedmf.qed.e_boson

    # Density matrix
    return_dm = None
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Eta parameters
    return_eta_val = None
    if guess_eta is not None:
        return_eta_val = guess_eta.copy()

    # Variational f parameters
    return_fopt_val = None
    if guess_fopt is not None:
        return_fopt_val = guess_fopt.copy()

    # Only create guess if previous calculation converged
    conv = vtqedmf.converged
    if conv:
        return_dm = vtqedmf.make_rdm1()
        return_eta_val = vtqedmf.eta
        return_fopt_val = vtqedmf.qed.couplings_var

    del vtqedmf
    return return_dm, return_eta_val, return_fopt_val, e_tot, e_boson, conv


if __name__ == "__main__":

    # Basis set
    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "cc-pvdz"

    # Datafile name
    h5_name = f"{basis}/vtqed_e_vs_nfock.h5"

    # Frequency
    omega = 0.5

    # Command-line argument
    la_vals = None
    if len(sys.argv) > 1:
        la_vals = [float(sys.argv[1])]

    # Scan range of values
    else:
        la_start = 0.200
        la_end = 0.300
        la_skip = 0.005
        la_vals = np.arange(la_start, la_end + la_skip, la_skip)

    # Store guess data
    vtqed_dm = None
    vtqed_eta = None
    vtqed_fopt = None

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "VTQEDHF"
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
        eta = []
        fopt = []

        # Run VT-QED-HF with increasing nFock
        nf = 0
        while e_diff >= 1.0e-6:

            # Increase number of Fock states
            nf += 1
            nf_list = [nf]

            # Run VT-QED-HF
            vtqed_dm, vtqed_eta, vtqed_fopt, e_tot, e_boson, conv = run_vtqedhf(
                basis_set = basis,
                lmbda = la,
                nfock = nf_list,
                freq = omega,
                guess_dm = vtqed_dm,
                guess_eta = vtqed_eta,
                guess_fopt = vtqed_fopt
            )

            # Append values to list
            etot.append(e_tot)
            eboson.append(e_boson)
            eta.append(vtqed_eta)
            fopt.append(vtqed_fopt)

            # Update energy difference if SCF converges
            if conv:
                e_diff = np.abs(e_tot - e_prev)
                e_prev = 0.0 + e_tot

        # Ediff criteria met
        print (f"\nVT-QED-HF CONVERGED W/ NUMBER OF FOCK STATES!\n "
               f"GFAC={la:.3f}  NFOCK={nf}\n")
        sys.stdout.flush()

        # write data to file
        with h5py.File(h5_name, "a") as fa:
            lambda_grp = fa.require_group(lambda_group)
            method_grp = lambda_grp.require_group(method_group)

            method_grp.create_dataset(f"energy", data=np.array(etot))
            method_grp.create_dataset(f"boson_energy", data=np.array(eboson))
            method_grp.create_dataset(f"eta", data=np.array(eta))
            method_grp.create_dataset(f"fopt", data=np.array(fopt))
            method_grp.create_dataset(f"nfock", data=nf)
