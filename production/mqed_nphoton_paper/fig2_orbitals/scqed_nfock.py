import os
import sys
import h5py
import numpy as np

from pyscf import gto
from pyscf.tools import molden

from openms.mqed import scqedhf


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
    cavity_mode[0, :] = gfac * np.asarray([0, 1, 0])

    return cavity_freq, cavity_mode


def run_scqedhf(
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    guess_dm = None,
    guess_eta = None
):
    """Perform SC-QED-HF calculation."""

    # Molecule
    mol = get_mol(basis_set)
    mol.verbose = 4
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
        nboson_states = nfock
    )
    scqedmf.max_cycle = 1000
    scqedmf.init_guess = "hcore"
    #scqedmf.conv_tol = 1e-8
    scqedmf.diis_space = 30
    scqedmf.precond = 0.01

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
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Eta parameters
    if guess_eta is not None:
        return_eta_val = guess_eta.copy()

    # Only create guess if previous calculation converged
    conv = scqedmf.converged
    if conv:
        return_dm = scqedmf.make_rdm1()
        return_eta_val = scqedmf.eta

    return scqedmf, return_dm, return_eta_val, e_tot, e_boson, conv


if __name__ == "__main__":

    # Basis set
    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "cc-pvdz"

    # Frequency
    omega = 0.5

    # Command-line argument
    la_vals = None
    if len(sys.argv) > 1:
        la_vals = [float(sys.argv[1])]

    # Scan range of values
    else:
        la_start = 0.0
        la_end = 0.3
        la_skip = 0.15
        la_vals = np.arange(la_start, la_end + la_skip, la_skip)

    # Store guess data
    scqed_dm = None
    scqed_eta = None

    # Datafile name
    h5_name = f"{basis}/scqed_e_vs_nfock.h5"

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "SCQEDHF"

        # Save datafile to these variables
        nf = None

        # Datafile exists
        if os.path.exists(h5_name):
            with h5py.File(h5_name, "a") as f:

                # Method run for provided lambda value
                if lambda_group in f:
                    lam_grp = f[lambda_group]

                    if method_group in lam_grp:
                        method_grp = lam_grp[method_group]

                    # Check for density matrix
                    if "elec_dm" in method_grp:
                        scqed_dm = np.array(method_grp["elec_dm"])
                        print ("Loaded 'elec_dm' from file.")

                    # Check for eta values
                    if "eta" in method_grp:
                        scqed_eta = np.array(method_grp["eta"])
                        print ("Loaded 'eta' from file.")

                    # Check for nfock value
                    if "nfock" in method_grp:
                        nf = method_grp["nfock"]
                        print ("Loaded 'nfock' from file.")

        # Create initial file
        if not os.path.exists(h5_name):
            with h5py.File(h5_name, 'w') as f: pass
            print(f"Created empty file: {h5_name}.\n Starting calculation...\n")
            sys.stdout.flush()


        # Run SC-QED-HF
        conv = False
        nf_list = None

        if isinstance(nf, int):
            nf_list = [nf]
        else:
            if la == 0.0:
                nf_list = [1]
            else:
                nf_list = [4]

        # Run SC-QED-HF
        scqed, scqed_dm, scqed_eta, e_tot, e_boson, conv = run_scqedhf(
            basis_set = basis,
            lmbda = la,
            nfock = nf_list,
            freq = omega,
            guess_dm = scqed_dm,
            guess_eta = scqed_eta
        )

        new_mo_coeff = scqed.compute_fock_state_basis_mos()
        dm_tot, dm_elec, dm_ph = scqed.transform_to_original_basis()

        ovlp = scqed.get_ovlp()
        new_mo_energy, new_mo_coeff = scqed.eig(dm_elec, ovlp)

        # Save .molden file
        if conv:
            with open(f"c3h4o2_{la:.3f}.molden", "w") as f1:
                molden.header(scqed.mol, f1)
                molden.orbital_coeff(
                    mol = scqed.mol,
                    fout = f1,
                    mo_coeff = new_mo_coeff,
                    ene = scqed.mo_energy,
                    occ = scqed.mo_occ
                )

        else:
            print ("CALCULATION NOT CONVERGED")
