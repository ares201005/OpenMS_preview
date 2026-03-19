import numpy as np
from openms.lib.lattice import Lattice


class Hubbard(Lattice):
    r"""Hubbard model on a Lattice.

    Note: In the x_matrix_spin() methods, the AOs are ordered
    (1a, 2a, ... La, 1b, 2b, ..., Lb),
    where (1, 2, ..., L) are sites and (a, b) are spins.
    So they have a block form: [[aa, ab], [ba, bb]], with each block L x L

    Passed members
    --------------
    t           : Kinetic hopping parameter
    U           : On-site Coulomb repulsion
    N           : Total number of electrons (optionally, of each spin component)
                  (TODO: filling fraction or chemical potential instead?)
    nspin       : Number of spin components (1 or 2)
    verbose     : integer verbosity flag (default: 3)

    Computed members
    ----------------
    tmat        : Kinetic energy matrix, in site(-spin) basis
    Umat        : On-site repulsion matrix, in site(-spin) basis
    ovlp        : Overlap matrix (identity), in site basis (N.B.: no noncollinear orbitals!)
    N_spin      : Number of electrons of each spin component (if nspin == 2)
    S_z         : Total spin

    Class methods
    -------------
    t_matrix_site(t) -> np.array(L_tot x L_tot)
        Return the hopping matrix in the site basis

    U_matrix_site(U) -> np.array(L_tot x L_tot x L_tot x L_tot)
        Return the interaction matrix U in the site basis

    overlap_site() -> np.array(L_tot x L_tot)
        Return the overlap (identity) matrix in the site basis

    t_matrix_spin(t) -> np.array(2L_tot x 2L_tot)
        Return the hopping matrix in the spin-orbital basis

    U_matrix_spin(U) -> np.array(2L_tot x 2L_tot x 2L_tot x 2L_tot)
        Return the interaction matrix U in the spin-orbital basis

    overlap_spin() -> np.array(2L_tot x 2L_tot)
        Return the overlap (identity) matrix in the spin-orbital basis

    hubbard_info() -> None
        Print basic information about the Hubbard lattice.
    """

    def __init__(self, t, U, N, nspin=2, verbose=3, *args, **kwargs):
        super().__init__(**kwargs)

        self.verbose = verbose

        self.t = t
        self.U = U

        self.nspin = nspin

        nelec = N if N is list else [N]
        assert len(nelec) <= 2, "Input number of electrons, or of [up, down] electrons."
        if len(nelec) == 1:
            self.N = nelec[0]
            self.N_spin = [self.N - (self.N // 2), self.N // 2]
        else:
            self.N_spin[0] = nelec[0]
            self.N_spin[1] = nelec[1]
            self.N = sum(self.N_spin)

        # Total spin
        self.S_z = self.N_spin[0] - self.N_spin[1]

        if self.nspin == 1:
            self.tmat = self.t_matrix_site(t)  # L x L
            self.Umat = self.U_matrix_site(U)  # L x L x L x L
        if self.nspin == 2:
            self.tmat = self.t_matrix_spin(t)  # 2L x 2L
            self.Umat = self.U_matrix_spin(U)  # 2L x 2L x 2L x 2L
            # self.ovlp = self.overlap_spin()  # 2L x 2L

        self.ovlp = self.overlap_site()  # L x L

    def t_matrix_site(self, t: float) -> np.array:
        tmat = np.zeros((self.L_tot, self.L_tot))
        for i in range(self.L_tot):
            for n in self.nn[i]:
                tmat[i][n] = t

        return tmat

    def U_matrix_site(self, U: float) -> np.array:
        # Spatial repulsion matrix
        Ltot = self.L_tot
        umat = np.zeros([Ltot, Ltot, Ltot, Ltot])

        for i in range(Ltot):
            umat[i, i, i, i] = U

        return umat

    def overlap_site(self) -> np.array:
        return np.eye(self.L_tot)

    def t_matrix_spin(self, t: float) -> np.array:
        t_site = self.t_matrix_site(t)

        if self.nspin == 1:
            tmat = t_site
        else:
            z_site = np.zeros((self.L_tot, self.L_tot))
            tmat = np.block([[t_site, z_site], [z_site, t_site]])

        return tmat

    def U_matrix_spin(self, U: float) -> np.array:
        # <ab|U|cd>, where a = (i, σ)
        # only nonzero when all site indices are the same
        Ltot = self.L_tot
        umat = np.zeros((2 * Ltot, 2 * Ltot, 2 * Ltot, 2 * Ltot))

        for i in range(Ltot):  # i: site indices
            umat[i, i + Ltot, i, i + Ltot] = +U  # <iα, iβ|U|iα, iβ>
            umat[i + Ltot, i, i + Ltot, i] = +U  # <iβ, iα|U|iβ, iα>
            umat[i, i + Ltot, i + Ltot, i] = -U  # <iα, iβ|U|iβ, iα>
            umat[i + Ltot, i, i, i + Ltot] = -U  # <iβ, iα|U|iα, iβ>

        return umat

    def overlap_spin(self) -> np.array:
        I_site = np.eye(self.L_tot)

        if self.nspin == 2:
            ovlp = np.broadcast_to(I_site, (self.nspin, self.L_tot, self.L_tot))
        else:
            ovlp = I_site

        return ovlp

    def hubbard_info(self):
        self.lattice_info()

        print("--------------------- Hubbard Parameters --------------------")
        print("-------------------------------------------------------------")
        print("Hopping strength (t)             : ", self.t)
        print("Dimensionless repulsion (-U/t)   : ", -self.U / self.t)
        print("Electron number (N)              : ", self.N)
        print("Number of spin components        : ", self.nspin)
        print("Total spin                       : ", self.S_z)
        print("Electrons up/down                : ", self.N_spin[0], self.N_spin[1])
        print("-------------------------------------------------------------")
        print()
