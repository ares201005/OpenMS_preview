
import os
import sys
import h5py
import numpy as np

from pyscf import gto
from pyscf.tools import molden

from openms.mqed import vtqedhf


def inv_quartic_for_mo(mo_coeff, mo_num):
    col = mo_coeff[:, mo_num]
    x = np.abs(col)
    return 1.0 / np.sum(x**4)

def inv_quartic_all(mo_coeff):
    x = np.sum(np.abs(mo_coeff)**4, axis=0)
    return 1.0 / x


def get_mol(basis="sto-3g"):
    """Return Molecule object."""
    atom = f"S      0.000000    0.000000   -0.000000;\
             O      0.698015    1.304897   -0.244990;\
             Cl     1.508897   -1.239426   -0.323263;\
             Br    -0.000000   -0.000000    2.126247"
    # atom = f"Li    0.000    0.000    0.000;\
    #           H    0.000    0.000    1.600"
    # # atom = f"C   0.00000000   0.00000000    0.0000;\
    #          O   0.00000000   1.23456800    0.0000;\
    #          H   0.97075033  -0.54577032    0.0000;\
    #          C  -1.21509881  -0.80991169    0.0000;\
    #          H  -1.15288176  -1.89931439    0.0000;\
    #          C  -2.43440063  -0.19144555    0.0000;\
    #          H  -3.37262777  -0.75937214    0.0000;\
    #          O  -2.62194056   1.12501165    0.0000;\
    #          H  -1.71446384   1.51627790    0.0000"

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
    #cavity_mode[0, :] = gfac * np.asarray([0, 1, 0])
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
        nboson_states = nfock,
        #couplings_var = [0.0] * nmodes
    )
    vtqedmf.max_cycle = 500
    vtqedmf.init_guess = "hcore"
    #vtqedmf.conv_tol = 1e-8
    vtqedmf.diis_space = 30
    vtqedmf.precond = 8e-3

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
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Eta parameters
    if guess_eta is not None:
        return_eta_val = guess_eta.copy()

    # Variational f parameters
    if guess_fopt is not None:
        return_fopt_val = guess_fopt.copy()

    # Only create guess if previous calculation converged
    conv = vtqedmf.converged
    if conv:
        return_dm = vtqedmf.make_rdm1()
        return_eta_val = vtqedmf.eta
        return_fopt_val = vtqedmf.qed.couplings_var

    return vtqedmf, return_dm, return_eta_val, return_fopt_val, e_tot, e_boson, conv


if __name__ == "__main__":

    # Basis set
    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "3-21g*"
    #basis = "cc-pvdz"

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
        la_skip = 0.1
        la_vals = np.arange(la_start, la_end + la_skip, la_skip)

    # Store guess data
    vtqed_dm = None
    vtqed_eta = None
    vtqed_fopt = None

    # Datafile name
    #h5_name = f"{basis}/vtqed_e_vs_nfock.h5"
    #h5_name = "lih_vtqed_test.h5"
    h5_name = "soclbr_vtqed_test.h5"

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "VTQEDHF"

        nf = None

        # # Datafile exists
        # if os.path.exists(h5_name):
        #     with h5py.File(h5_name, "a") as f:

        #         # Method run for provided lambda value
        #         if lambda_group in f:
        #             lam_grp = f[lambda_group]

        #             if method_group in lam_grp:
        #                 method_grp = lam_grp[method_group]

        #             # Check for density matrix
        #             if "elec_dm" in method_grp:
        #                 vtqed_dm = np.array(method_grp["elec_dm"])
        #                 print ("Loaded 'elec_dm' from file.")

        #             # Check for eta values
        #             if "eta" in method_grp:
        #                 vtqed_eta = np.array(method_grp["eta"])
        #                 print ("Loaded 'eta' from file.")

        #             # Check for fopt value
        #             if "fopt" in method_grp:
        #                 vtqed_fopt = np.array(method_grp["fopt"])
        #                 print ("Loaded 'fopt' from file.")

        #             # Check for nfock value
        #             if "nfock" in method_grp:
        #                 nf = method_grp["nfock"]
        #                 print ("Loaded 'nfock' from file.")

        # # Create initial file
        # if not os.path.exists(h5_name):
        #     with h5py.File(h5_name, 'w') as f: pass
        #     print(f"Created empty file: {h5_name}.\n Starting calculation...\n")
        #     sys.stdout.flush()

        # Run VT-QED-HF
        conv = False
        nf_list = None

        # if isinstance(nf, int):
        #     nf_list = [nf]
        # else:
        #     if la == 0.0:
        #         nf_list = [1]
        #     else:
        #         nf_list = [4]

        # Run VT-QED-HF
        vtqed, vtqed_dm, vtqed_eta, vtqed_fopt, e_tot, e_boson, conv = run_vtqedhf(
            basis_set = basis,
            lmbda = la,
            nfock = [1],
            freq = omega,
            guess_dm = vtqed_dm,
            guess_eta = vtqed_eta,
            guess_fopt = vtqed_fopt
        )

        new_mo_coeff = vtqed.compute_fock_state_basis_mos()
        exit()

        # ovlp = vtqed.get_ovlp()
        # e_val, e_vec = np.linalg.eigh(ovlp)
        # shalf = e_vec @ np.diag(np.sqrt(e_val)) @ e_vec.T
        # t_mo_coeff = shalf @ new_mo_coeff
        # #t_mo_coeff = shalf @ vtqed.mo_coeff

        # ipr_orth = inv_quartic_for_mo(mo_coeff=t_mo_coeff, mo_num=14)
        # print (f"{ipr_orth = }")

        # ipr_orth_all = inv_quartic_all(mo_coeff=t_mo_coeff)
        # print ("ipr_orth_all_sum = ", np.sum(ipr_orth_all))
        # exit()

        if conv:

            # Write data to file
            with h5py.File(h5_name, "a") as fa:
                lambda_grp = fa.require_group(lambda_group)
                method_grp = lambda_grp.require_group(method_group)

                method_grp.create_dataset(f"energy", data=np.array(e_tot))
                method_grp.create_dataset(f"boson_energy", data=np.array(e_boson))
                method_grp.create_dataset(f"elec_dm", data=np.array(vtqed_dm))
                method_grp.create_dataset(f"eta", data=np.array(vtqed_eta))
                method_grp.create_dataset(f"fopt", data=np.array(vtqed_fopt))
                #method_grp.create_dataset(f"nfock", data=nf)


            # # Save .molden file
            # with open(f"{mol_name}_{la:.3f}.molden", "w") as f1:
            #     molden.header(vtqed.mol, f1)
            #     molden.orbital_coeff(
            #         mol = vtqed.mol,
            #         fout = f1,
            #         mo_coeff = new_mo_coeff,
            #         ene = vtqed.mo_energy,
            #         occ = vtqed.mo_occ
            #     )
        dm_tot, dm_elec, dm_ph = vtqed.transform_to_original_basis()

        ovlp = vtqed.get_ovlp()
        new_mo_energy, new_mo_coeff = vtqed.eig(dm_elec, ovlp)

        else:
            print ("CALCULATION NOT CONVERGED")

