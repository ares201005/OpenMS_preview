import numpy as np
import unittest
from openms.models.hubbard import Hubbard

# Test cases:
#   1d: 4-site lattice, nspin = 1; U = 4, t = -1
#       In both cases, U[i][i][i][i] = 4, else U[i][j][k][l] = 0
#       OBC: T_true = [[0, -1., 0, 0], [-1., 0, -1., 0], [0, -1., 0, -1.], [0, 0, -1., 0]]
#       PBC: T_true = [[0, -1., 0, -1.], [-1., 0, -1., 0], [0, -1., 0, -1.], [-1., 0, -1., 0]]
#   2d: 3x3 lattice; nspin = 1, U = 4, t = -1
#       As in 1d, U[i][j][k][l] = 4 \delta_{ij} \delta_{ik} \delta_{il}
#       OBC: T_true = [[0, -1., 0, -1., 0, 0, 0, 0, 0],
#                      [-1., 0, -1., 0, -1., 0, 0, 0, 0],
#                      [0, -1., 0, 0, 0, -1., 0, 0, 0],
#                      [-1., 0, 0, 0, -1., 0, -1., 0, 0],
#                      [0, -1., 0, -1., 0, -1., 0, -1., 0],
#                      [0, 0, -1., 0, -1., 0, 0, 0, -1.],
#                      [0, 0, 0, -1., 0, 0, 0, -1., 0],
#                      [0, 0, 0, 0, -1., 0, -1., 0, -1.],
#                      [0, 0, 0, 0, 0, -1., 0, -1., 0]]
#       PBC: T_true =   [[0, -1., -1., -1., 0, 0, -1., 0, 0],
#                        [-1., 0, -1., 0, -1., 0, 0, -1., 0],
#                        [-1., -1., 0, 0, 0, -1., 0, 0, -1.],
#                        [-1., 0, 0, 0, -1., -1., -1., 0, 0],
#                        [0, -1., 0, -1., 0, -1., 0, -1., 0],
#                        [0, 0, -1., -1., -1., 0, 0, 0, -1.],
#                        [-1., 0, 0, -1., 0, 0, 0., -1., -1.],
#                        [0, -1., 0, 0, -1., 0, -1., 0, -1.],
#                        [0, 0, -1., 0, 0, -1., -1., -1., 0]]
#   2d: As above, but a 2x2 lattice with nspin = 2 and N = 5
#       This changes the hopping matrix to look like [[T 0], [0 T]]
#       and the electron repulsion matrix to
#           U[iα][iβ][iα][iβ] = U[iβ][iα][iβ][iα]
#            = -U[iα][iβ][iβ][iα] = -U[iβ][iα][iα][iβ] = U.
#       For each spin component,
#           T_true =    [[0, -1., -1., 0],
#                        [-1., 0, 0, -1.],
#                        [-1., 0, 0, -1.],
#                        [0, -1., -1., 0]]
#       The spin occupancy is [Nα, Nβ] = [3, 2], with S_z = 1


class Test1d(unittest.TestCase):
    def test_1d_open(self):
        # Lattice
        dim = 1
        L = Ltot = 4
        periodic = False
        verbose = 1

        # Hubbard
        t = -1.0
        U = 4.0
        N = 4
        nspin = 1

        hub = Hubbard(
            t=t, U=U, N=N, nspin=nspin, verbose=verbose, dim=dim, L=L, periodic=periodic
        )

        T_true = [
            [0, -1.0, 0, 0],
            [-1.0, 0, -1.0, 0],
            [0, -1.0, 0, -1.0],
            [0, 0, -1.0, 0],
        ]
        U_true = np.zeros((Ltot, Ltot, Ltot, Ltot))
        for i in range(Ltot):
            U_true[i][i][i][i] = U

        np.testing.assert_allclose(
            hub.tmat, T_true, err_msg="Kinetic energy matrix wrong"
        )
        np.testing.assert_allclose(
            hub.Umat, U_true, err_msg="Electron repulsion matrix wrong"
        )

    def test_1d_pbc(self):
        # Lattice
        dim = 1
        L = Ltot = 4
        periodic = True
        verbose = 1

        # Hubbard
        t = -1.0
        U = 4.0
        N = 4
        nspin = 1

        hub = Hubbard(
            t=t, U=U, N=N, nspin=nspin, verbose=verbose, dim=dim, L=L, periodic=periodic
        )

        T_true = [
            [0, -1.0, 0, -1.0],
            [-1.0, 0, -1.0, 0],
            [0, -1.0, 0, -1.0],
            [-1.0, 0, -1.0, 0],
        ]
        U_true = np.zeros((Ltot, Ltot, Ltot, Ltot))
        for i in range(Ltot):
            U_true[i][i][i][i] = U

        np.testing.assert_allclose(
            hub.tmat, T_true, err_msg="Kinetic energy matrix wrong"
        )
        np.testing.assert_allclose(
            hub.Umat, U_true, err_msg="Electron repulsion matrix wrong"
        )


class Test2dSquare(unittest.TestCase):
    def test_2d_open(self):
        # Lattice
        dim = 2
        L = 3
        Ltot = L**2
        periodic = False
        shape = "square"
        verbose = 1

        # Hubbard
        t = -1.0
        U = 4.0
        N = 4
        nspin = 1

        hub = Hubbard(
            t=t,
            U=U,
            N=N,
            nspin=nspin,
            verbose=verbose,
            dim=dim,
            L=L,
            shape=shape,
            periodic=periodic,
        )

        T_true = [
            [0, -1.0, 0, -1.0, 0, 0, 0, 0, 0],
            [-1.0, 0, -1.0, 0, -1.0, 0, 0, 0, 0],
            [0, -1.0, 0, 0, 0, -1.0, 0, 0, 0],
            [-1.0, 0, 0, 0, -1.0, 0, -1.0, 0, 0],
            [0, -1.0, 0, -1.0, 0, -1.0, 0, -1.0, 0],
            [0, 0, -1.0, 0, -1.0, 0, 0, 0, -1.0],
            [0, 0, 0, -1.0, 0, 0, 0, -1.0, 0],
            [0, 0, 0, 0, -1.0, 0, -1.0, 0, -1.0],
            [0, 0, 0, 0, 0, -1.0, 0, -1.0, 0],
        ]

        U_true = np.zeros((Ltot, Ltot, Ltot, Ltot))
        for i in range(Ltot):
            U_true[i][i][i][i] = U

        np.testing.assert_allclose(
            hub.tmat, T_true, err_msg="Kinetic energy matrix wrong"
        )
        np.testing.assert_allclose(
            hub.Umat, U_true, err_msg="Electron repulsion matrix wrong"
        )

    def test_2d_pbc(self):
        # Lattice
        dim = 2
        L = 3
        Ltot = L**2
        periodic = True
        shape = "square"
        verbose = 1

        # Hubbard
        t = -1.0
        U = 4.0
        N = 4
        nspin = 1

        hub = Hubbard(
            t=t,
            U=U,
            N=N,
            nspin=nspin,
            verbose=verbose,
            dim=dim,
            L=L,
            shape=shape,
            periodic=periodic,
        )

        T_true = [
            [0, -1.0, -1.0, -1.0, 0, 0, -1.0, 0, 0],
            [-1.0, 0, -1.0, 0, -1.0, 0, 0, -1.0, 0],
            [-1.0, -1.0, 0, 0, 0, -1.0, 0, 0, -1.0],
            [-1.0, 0, 0, 0, -1.0, -1.0, -1.0, 0, 0],
            [0, -1.0, 0, -1.0, 0, -1.0, 0, -1.0, 0],
            [0, 0, -1.0, -1.0, -1.0, 0, 0, 0, -1.0],
            [-1.0, 0, 0, -1.0, 0, 0, 0.0, -1.0, -1.0],
            [0, -1.0, 0, 0, -1.0, 0, -1.0, 0, -1.0],
            [0, 0, -1.0, 0, 0, -1.0, -1.0, -1.0, 0],
        ]

        U_true = np.zeros((Ltot, Ltot, Ltot, Ltot))
        for i in range(Ltot):
            U_true[i][i][i][i] = U

        np.testing.assert_allclose(
            hub.tmat, T_true, err_msg="Kinetic energy matrix wrong"
        )
        np.testing.assert_allclose(
            hub.Umat, U_true, err_msg="Electron repulsion matrix wrong"
        )


class Test2dSquareSpin(unittest.TestCase):
    def test_2d_pbc(self):
        # Lattice
        dim = 2
        L = 2
        Ltot = L**2
        periodic = True
        shape = "square"
        verbose = 1

        # Hubbard
        t = -1.0
        U = 4.0
        N = 5
        nspin = 2

        hub = Hubbard(
            t=t,
            U=U,
            N=N,
            nspin=nspin,
            verbose=verbose,
            dim=dim,
            L=L,
            shape=shape,
            periodic=periodic,
        )

        N_true = [3, 2]

        Sz_true = 1  # N_true[0] - N_true[1]

        T_site = np.array(
            [
                [0, -1.0, -1.0, 0],
                [-1.0, 0, 0, -1.0],
                [-1.0, 0, 0, -1.0],
                [0, -1.0, -1.0, 0],
            ]
        )
        Z_site = np.zeros((Ltot, Ltot))
        T_true = np.block([[T_site, Z_site], [Z_site, T_site]])

        U_true = np.zeros((2 * Ltot, 2 * Ltot, 2 * Ltot, 2 * Ltot))
        for i in range(Ltot):
            U_true[i, i + Ltot, i, i + Ltot] = U
            U_true[i + Ltot, i, i + Ltot, i] = U
            U_true[i, i + Ltot, i + Ltot, i] = -U
            U_true[i + Ltot, i, i, i + Ltot] = -U

        self.assertEqual(hub.N_spin, N_true, "Number of up/dw electrons wrong")
        self.assertEqual(hub.S_z, Sz_true, "Total spin wrong")

        np.testing.assert_allclose(
            hub.tmat, T_true, err_msg="Kinetic energy matrix wrong"
        )
        np.testing.assert_allclose(
            hub.Umat, U_true, err_msg="Electron repulsion matrix wrong"
        )


if __name__ == "main":
    unittest.main()
