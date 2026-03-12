import unittest
from openms.lib.lattice import Lattice

# Test cases: Indices run left-right across x, then up y
#   1d: L = 4 = L_tot
#       NN(open): [[1], [0, 2], [1, 3], [2]]
#       NN(PBC) : [[1,3], [0,2], [1,3], [0,2]]
#   2d: L = 2 or [2, 2], L_tot = 4
#       NN(open): [[1,2], [0,3], [0,3], [1,2]]
#       NN(PBC) : [[1,2], [0,3], [0,3], [1,2]]
#   2d: L = 3 or [3, 3], L_tot = 9
#       NN(open): [[1,3], [0,2,4], [1,5],
#                  [0,4,6], [1,3,5,7], [2,4,8],
#                  [3,7], [4,6,8], [5,7]]
#       NN(PBC) : [[1,2,3,6], [0,2,4,7], [0,1,5,8],
#                  [0,4,5,6], [1,3,5,7], [2,3,4,8],
#                  [0,3,7,8], [1,4,6,8], [2,5,6,7]]
#   2d: L = [4, 3], L_tot = 12
#       NN(open): [[1,4], [0,2,5], [1,3,6], [2,7],
#                  [0,5,8], [1,4,6,9], [2,5,7,10], [3,6,11],
#                  [4,9], [5,8,10], [6,9,11], [7,10]]
#       NN(PBC) : [[1,3,4,8], [0,2,5,9], [1,3,6,10], [0,2,7,11],
#                  [0,5,7,8], [1,4,6,9], [2,5,7,10], [3,4,6,11],
#                  [0,4,9,11],[1,5,8,10], [2,6,9,11], [3,7,8,10]]


class Test1d(unittest.TestCase):
    def test_1d_open(self):
        L = Ltot = 4
        lat = Lattice(dim=1, L=L, periodic=False)
        nn_true = [[1], [0, 2], [1, 3], [2]]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")

    def test_1d_pbc(self):
        L = Ltot = 4
        lat = Lattice(dim=1, L=L, periodic=True)
        nn_true = [[1, 3], [0, 2], [1, 3], [0, 2]]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")


class Test2x2(unittest.TestCase):
    def test_2x2_open(self):
        L, Ltot = 2, 4
        lat = Lattice(dim=2, L=L, periodic=False)
        nn_true = [[1, 2], [0, 3], [0, 3], [1, 2]]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")

    def test_2x2_pbc(self):
        L, Ltot = [2, 2], 4
        lat = Lattice(dim=2, L=L, periodic=True)
        nn_true = [[1, 2], [0, 3], [0, 3], [1, 2]]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")


class Test3x3(unittest.TestCase):
    def test_3x3_open(self):
        L, Ltot = [3, 3], 9
        lat = Lattice(dim=2, L=L, periodic=False)
        nn_true = [
            [1, 3],
            [0, 2, 4],
            [1, 5],
            [0, 4, 6],
            [1, 3, 5, 7],
            [2, 4, 8],
            [3, 7],
            [4, 6, 8],
            [5, 7],
        ]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")

    def test_3x3_pbc(self):
        L, Ltot = 3, 9
        lat = Lattice(dim=2, L=L, periodic=True)
        nn_true = [
            [1, 2, 3, 6],
            [0, 2, 4, 7],
            [0, 1, 5, 8],
            [0, 4, 5, 6],
            [1, 3, 5, 7],
            [2, 3, 4, 8],
            [0, 3, 7, 8],
            [1, 4, 6, 8],
            [2, 5, 6, 7],
        ]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")


class Test4x3(unittest.TestCase):
    def test_4x3_open(self):
        L, Ltot = [4, 3], 12
        lat = Lattice(dim=2, L=L, periodic=False)
        nn_true = [
            [1, 4],
            [0, 2, 5],
            [1, 3, 6],
            [2, 7],
            [0, 5, 8],
            [1, 4, 6, 9],
            [2, 5, 7, 10],
            [3, 6, 11],
            [4, 9],
            [5, 8, 10],
            [6, 9, 11],
            [7, 10],
        ]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")

    def test_4x3_pbc(self):
        L, Ltot = [4, 3], 12
        lat = Lattice(dim=2, L=L, periodic=True)
        nn_true = [
            [1, 3, 4, 8],
            [0, 2, 5, 9],
            [1, 3, 6, 10],
            [0, 2, 7, 11],
            [0, 5, 7, 8],
            [1, 4, 6, 9],
            [2, 5, 7, 10],
            [3, 4, 6, 11],
            [0, 4, 9, 11],
            [1, 5, 8, 10],
            [2, 6, 9, 11],
            [3, 7, 8, 10],
        ]
        self.assertEqual(lat.L_tot, Ltot, "L_tot wrong")
        self.assertEqual(nn_true, lat.nn, "Nearest neighbors wrong")


if __name__ == "main":
    unittest.main()
