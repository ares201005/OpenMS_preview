import os
import sys
import h5py
import numpy as np

from pyscf import gto

from openms.mqed import qedhf, vtqedhf


def embed_dm(dm_small, dim_large):
    """Embed density matrix into larger Hilbert space."""
    m = dm_small.shape[0]
    n = dim_large
    rho_large = np.zeros((n, n), dtype=np.complex128)
    rho_large[:m, :m] = dm_small
    return rho_large


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


def run_qedhf(
    molecule = "LiH",
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    coherent_state = False,
    guess_dm = None
):
    """Perform QED-HF calculation."""

    # Molecule
    mol = get_mol(name=molecule, basis=basis_set)
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
        use_cs = coherent_state
    )
    qedmf.max_cycle = 1000
    qedmf.init_guess = "hcore"

    if guess_dm is not None:
        qedmf.kernel(dm0=guess_dm)
    else:
        qedmf.kernel()

    # Energies
    e_tot = qedmf.e_tot
    e_boson = qedmf.qed.e_boson

    # Density matrix
    return_dm = None
    if guess_dm is not None:
        return_dm = guess_dm.copy()

    # Only create guess if previous calculation converged
    conv = qedmf.converged
    if conv:
        return_dm = qedmf.make_rdm1()

    ph_dm = None
    if not coherent_state:
        ph_dm = qedmf.qed.get_boson_dm()

    del qedmf
    return return_dm, e_tot, e_boson, ph_dm, conv

def run_vtqedhf(
    molecule = "LiH",
    basis_set = "sto-3g",
    lmbda = 0.0,
    nfock = [1],
    freq = 0.5,
    guess_dm = None,
    guess_eta = None,
    guess_fopt = None,
):
    """Perform VT-QED-HF calculation."""

    # Molecule
    mol = get_mol(name=molecule, basis=basis_set)
    mol.verbose = 4
    mol.build()

    # Cavity
    nmodes = 1
    cavity_freq, cavity_mode = get_cavity(nmode=nmodes, gfac=lmbda, omega=freq)

    # SC-QED-HF kernel
    vtqedmf = vtqedhf.RHF(
        mol = mol,
        cavity_mode = cavity_mode,
        cavity_freq = cavity_freq,
        add_nuc_dipole = True,
        nboson_states = nfock
    )
    vtqedmf.max_cycle = 1000
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

    ph_dm = None
    if conv:
        ph_dm = vtqedmf.qed.get_boson_dm()

    del vtqedmf
    return return_dm, return_eta_val, return_fopt_val, e_tot, e_boson, ph_dm, conv


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
        la_start = 0.3
        la_end = 0.3
        la_skip = 0.010

        la_vals = np.arange(la_start, la_end + la_skip, la_skip)

    # Loop through lambda values
    for la in la_vals:
        print (f"\nCURRENT LAMBDA VALUE = {la:.3f}\n")

        # QEDHF data
        qedhf_dm = None
        qed_nf = None

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "QEDHF"

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
                        qed_nf = int(method_grp["nfock"][()])
                        print ("Loaded 'nfock' from file.")


        # VTQED data
        vtqed_dm = None
        vtqed_eta = None
        vtqed_fopt = None
        vtqed_nf = None

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "VTQEDHF"

        if os.path.exists(h5_name):
            with h5py.File(h5_name, "a") as f:

                # Method run for provided lambda value
                if lambda_group in f:
                    lam_grp = f[lambda_group]

                    if method_group in lam_grp:
                        method_grp = lam_grp[method_group]

                    # Check for density matrix
                    if "elec_dm" in method_grp:
                        vtqed_dm = np.array(method_grp["elec_dm"])
                        print ("Loaded 'elec_dm' from file.")

                    # Check for eta values
                    if "eta" in method_grp:
                        vtqed_eta = np.array(method_grp["eta"])
                        print ("Loaded 'eta' from file.")

                    # Check for fopt values
                    if "fopt" in method_grp:
                        vtqed_fopt = np.array(method_grp["fopt"])
                        print ("Loaded 'fopt' from file.")

                    # Check for nfock value
                    if "nfock" in method_grp:
                        vtqed_nf = int(method_grp["nfock"][()])
                        print ("Loaded 'nfock' from file.")

        # Convergence and nfock variables
        conv = False
        nf_list = None

        vt_conv = False
        vtqed_nf_list = None

        if isinstance(qed_nf, int):
            nf_list = [qed_nf]
        else:
            raise TypeError("Invalid 'nfock' value. Must be an integer.")

        if isinstance(vtqed_nf, int):
            vtqed_nf_list = [vtqed_nf]

        # Run VT-QED-HF
        print ("PERFORMING VT-QED-HF CALCULATION...\n")
        vtqed_dm, vtqed_eta, vtqed_fopt, e_tot, e_boson, vtqed_ph_dm, vt_conv = run_vtqedhf(
            molecule = molecule,
            basis_set = basis,
            lmbda = la,
            nfock = vtqed_nf_list,
            freq = omega,
            guess_dm = vtqed_dm,
            guess_eta = vtqed_eta,
            guess_fopt = vtqed_fopt,
        )

        # Run QED-HF
        print ("\n\nPERFORMING QED-HF CALCULATION...\n")
        qedhf_dm, e_tot, e_boson, qed_ph_dm, conv = run_qedhf(
            molecule = molecule,
            basis_set = basis,
            lmbda = la,
            nfock = nf_list,
            freq = omega,
            guess_dm = qedhf_dm
        )

        if la > 0.0:

            n_qed = qed_ph_dm.shape[0]
            n_vtqed = vtqed_ph_dm.shape[0]

            n0 = np.array(range(n_qed))
            qed_ph = np.einsum("x, xx", n0, qed_ph_dm)
            print (f"\nPHOTON OCCUPATION; QEDHF: {qed_ph:.6f}")

            n0 = np.array(range(n_vtqed))
            vtqed_ph = np.einsum("x, xx", n0, vtqed_ph_dm)
            print (f"\nPHOTON OCCUPATION; VTQEDHF: {vtqed_ph:.6f}")

            print ("PRE-TRANSFORMATION SHAPES:")
            print (vtqed_ph_dm.shape, qed_ph_dm.shape)

            # Embed QED photon density matrix into larger Hilbert space
            if n_qed < n_vtqed:
                qed_ph_dm = embed_dm(qed_ph_dm, n_vtqed)
            # Embed SCQED photon density matrix into larger Hilbert space
            elif n_qed > n_vtqed:
                vtqed_ph_dm = embed_dm(vtqed_ph_dm, n_qed)

            print ("POST-TRANSFORMATION SHAPES:")
            print (vtqed_ph_dm.shape, qed_ph_dm.shape)

            # Diagonalize the photon density matrices
            evals_vt, vt_vec = np.linalg.eigh(vtqed_ph_dm)
            evals_qed, qed_vec = np.linalg.eigh(qed_ph_dm)

            # Transformation matrix
            Umat = qed_vec @ vt_vec.conj().T
            assert np.allclose(Umat @ Umat.conj().T, np.eye(Umat.shape[0]))

            # VTQED photon density matrix in QED basis
            tvt_ph_dm = Umat @ vtqed_ph_dm @ Umat.conj().T

            # Transformed photon occupation range
            n0 = np.array(range(tvt_ph_dm.shape[0]))
            tvtqed_ph = np.einsum("x, xx", n0, tvt_ph_dm)
            print (f"\nPHOTON OCCUPATION; transformed-VTQEDHF: {tvtqed_ph:.6f}")
