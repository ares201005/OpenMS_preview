import os
import sys
import h5py
import numpy as np

from pyscf import gto
from pyscf.tools import molden

from openms.mqed import qedhf


def inv_quartic_for_mo(mo_coeff, mo_num):
    col = mo_coeff[:, mo_num]
    x = np.abs(col)
    return 1.0 / np.sum(x**4)

def inv_quartic_all(mo_coeff):
    x = np.sum(np.abs(mo_coeff)**4, axis=0)
    return 1.0 / x


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


def run_qedhf(
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    coherent_state = False,
    guess_dm = None,
):
    """Perform QED-HF calculation."""

    # Molecule
    mol = get_mol(basis_set)
    mol.verbose = 4
    mol.build()

    # Cavity
    nmodes = 1
    cavity_freq, cavity_mode = get_cavity(nmode=nmodes, gfac=lmbda, omega=freq)

    # QED-HF kernel
    qedmf = qedhf.RHF(
        mol = mol,
        cavity_mode = cavity_mode,
        cavity_freq = cavity_freq,
        add_nuc_dipole = True,
        nboson_states = nfock,
        use_cs = coherent_state,
    )
    qedmf.max_cycle = 1000
    qedmf.init_guess = "hcore"
    #qedmf.conv_tol = 1e-8
    #qedmf.diis_space = 30

    if guess_dm is not None:
        qedmf.kernel(dm0=guess_dm)
    else:
        qedmf.kernel()

    # Energies
    e_tot = qedmf.e_tot
    e_boson = qedmf.qed.e_boson

    # Density matrix
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Only create guess if previous calculation converged
    conv = qedmf.converged
    if conv:
        return_dm = qedmf.make_rdm1()

    return qedmf, return_dm, e_tot, e_boson, conv


if __name__ == "__main__":

    # Basis set
    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "cc-pvdz"

    # Datafile name
    h5_name = f"{basis}/qedhf_e_vs_nfock.h5"

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
    qedhf_dm = None

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "QEDHF"

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
                        qedhf_dm = np.array(method_grp["elec_dm"])
                        print ("Loaded 'elec_dm' from file.")

                    # Check for nfock value
                    if "nfock" in method_grp:
                        nf = method_grp["nfock"]
                        print ("Loaded 'nfock' from file.")

        # # Create initial file
        # if not os.path.exists(h5_name):
        #     with h5py.File(h5_name, 'w') as f: pass
        #     print(f"Created empty file: {h5_name}.\n Starting calculation...\n")
        #     sys.stdout.flush()

        # Run QED-HF
        conv = False
        nf_list = None

        # if isinstance(nf, int):
        #     nf_list = [nf]
        # else:
        #     if la == 0.0:
        #         nf_list = [1]
        #     elif la == 0.1:
        #         nf_list = [4]
        #     elif la == 0.2:
        #         nf_list = [5]
        #     else:
        #         nf_list = [7]

        # Run QED-HF
        qed, qedhf_dm, e_tot, e_boson, conv = run_qedhf(
            basis_set = basis,
            lmbda = la,
            nfock = [1],
            freq = omega,
            guess_dm = qedhf_dm,
        )

        ovlp = qed.get_ovlp()
        e_val, e_vec = np.linalg.eigh(ovlp)
        shalf = e_vec @ np.diag(np.sqrt(e_val)) @ e_vec.T
        t_mo_coeff = shalf @ qed.mo_coeff

        ##########

        ipr_orth = inv_quartic_for_mo(mo_coeff=t_mo_coeff, mo_num=14)
        print (f"{ipr_orth = }")

        ipr_orth_all = inv_quartic_all(mo_coeff=t_mo_coeff)
        print ("ipr_orth_all_sum = ", np.sum(ipr_orth_all))
        exit()


        # Save .molden file
        if conv:
            with open(f"{basis}/c3h4o2_{la:.3f}.molden", "w") as f1:
                molden.header(qed.mol, f1)
                molden.orbital_coeff(qed.mol, f1, qed.mo_coeff, ene=qed.mo_energy, occ=qed.mo_occ)

        else:
            print ("CALCULATION NOT CONVERGED")

        # Delete QEDHF object
        del qed
