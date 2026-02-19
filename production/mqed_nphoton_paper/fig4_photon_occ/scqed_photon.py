import os
import sys
import h5py
import numpy as np

from pyscf import gto

from openms.mqed import qedhf, scqedhf


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
    scqedmf.precond = 8e-3

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

    ph_dm = None
    if conv:
        ph_dm = scqedmf.qed.get_boson_dm()

    del scqedmf
    return return_dm, return_eta_val, e_tot, e_boson, ph_dm, conv


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
        la_start = 0.0
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


        # SCQED data
        scqed_dm = None
        scqed_eta = None
        scqed_nf = None

        # Datafile group names
        lambda_group = f"lambda_{la:.3f}"
        method_group = "SCQEDHF"

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

                    # Check for nfock value
                    if "nfock" in method_grp:
                        scqed_nf = int(method_grp["nfock"][()])
                        print ("Loaded 'nfock' from file.")

                    # Check for eta values
                    if "eta" in method_grp:
                        scqed_eta = np.array(method_grp["eta"])
                        print ("Loaded 'eta' from file.")

        # Convergence and nfock variables
        conv = False
        nf_list = None

        sc_conv = False
        scqed_nf_list = None

        if isinstance(qed_nf, int):
            nf_list = [qed_nf]
        else:
            raise TypeError("Invalid 'nfock' value. Must be an integer.")

        if isinstance(scqed_nf, int):
            scqed_nf_list = [scqed_nf]

        # Run SC-QED-HF
        print ("PERFORMING SC-QED-HF CALCULATION...\n")
        scqed_dm, scqed_eta, e_tot, e_boson, scqed_ph_dm, sc_conv = run_scqedhf(
            molecule = molecule,
            basis_set = basis,
            lmbda = la,
            nfock = scqed_nf_list,
            freq = omega,
            guess_dm = scqed_dm,
            guess_eta = scqed_eta
        )

        # Run QED-HF
        print ("PERFORMING QED-HF CALCULATION...\n")
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
            n_scqed = scqed_ph_dm.shape[0]

            n0 = np.array(range(n_qed))
            qed_ph = np.einsum("x, xx", n0, qed_ph_dm)
            print (f"\nPHOTON OCCUPATION; QEDHF: {qed_ph:.6f}")

            n0 = np.array(range(n_scqed))
            scqed_ph = np.einsum("x, xx", n0, scqed_ph_dm)
            print (f"\nPHOTON OCCUPATION; SCQEDHF: {scqed_ph:.6f}")

            print ("PRE-TRANSFORMATION SHAPES:")
            print (scqed_ph_dm.shape, qed_ph_dm.shape)

            # Embed QED photon density matrix into larger Hilbert space
            if n_qed < n_scqed:
                qed_ph_dm = embed_dm(qed_ph_dm, n_scqed)
            # Embed SCQED photon density matrix into larger Hilbert space
            elif n_qed > n_scqed:
                scqed_ph_dm = embed_dm(scqed_ph_dm, n_qed)

            print ("POST-TRANSFORMATION SHAPES:")
            print (scqed_ph_dm.shape, qed_ph_dm.shape)

            # Diagonalize the photon density matrices
            evals_sc, sc_vec = np.linalg.eigh(scqed_ph_dm)
            evals_qed, qed_vec = np.linalg.eigh(qed_ph_dm)

            # Transformation matrix
            Umat = qed_vec @ sc_vec.conj().T
            assert np.allclose(Umat @ Umat.conj().T, np.eye(Umat.shape[0]))

            # SCQED photon density matrix in QED basis
            tsc_ph_dm = Umat @ scqed_ph_dm @ Umat.conj().T

            # Transformed photon occupation range
            n0 = np.array(range(tsc_ph_dm.shape[0]))
            tscqed_ph = np.einsum("x, xx", n0, tsc_ph_dm)
            print (f"\nPHOTON OCCUPATION; transformed-SCQEDHF: {tscqed_ph:.6f}")
