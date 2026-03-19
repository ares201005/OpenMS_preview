from operator import mul
from functools import reduce


class Lattice(object):
    r"""Hamiltonian-independent parts of a lattice model.

    ...
    Attributes
    ----------
    dim         : int
        The lattice dimension (default: 2)
    nsite       : int
        The number of sites in each unit cell (default: 1)
    L           : int[dim]
        The size of the lattice in each dimension.
        Can be specified as a single int, in which case we write [L, ..., L]
    L_tot       : int
        The total lattice size (default: L^dim)
    shape       : str
        The lattice in consideration (default: 'square')
    periodic    : bool
        True : periodic boundary conditions
    nn          : [[int1], [int2], ..., [intL_tot]]
        For each site index i, list of indices of i's nearest neighbors

    Methods
    -------
    nearest_neighbors(int i) -> [int]
        Return: list of indices of index i's nearest neighbors
    list_nearest_neighhbors() -> [[int]]
        Return: [nearest_neighbors(i) for each lattice site i]
    next_nearest_neighbors(int i) -> [int]
        Return: list of indices of index i's next-nearest neighbors
    list_next_nearest_neighhbors() -> [[int]]
        Return: [next_nearest_neighbors(i) for each lattice site i]
    unravel(list[int] [i1, ..., id], int d, str shape) -> int
        Return: unraveled index i from shape lattice's d-dimensional index (i1, ..., id)
    lattice_info -> None
        Print information about the lattice
    """

    def __init__(self, *, dim=1, L=2, shape="square", periodic=True, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        assert dim < 3, "3+ dimensions are not yet implemented"

        assert (
            isinstance(L, int) or len(L) == dim
        ), "Specify either a single L or [L_1, ..., L_dim]"
        if isinstance(L, int):
            self.L = [L for i in range(dim)]
        else:
            self.L = L

        if self.dim > 1 and any(Li < 2 for Li in self.L):
            raise ValueError("Length along each nontrivial dimension should be > 1")

        self.L_tot = reduce(mul, self.L, 1)

        self.shape = shape
        # TODO: implement more lattice shapes
        if shape != "square":
            raise NotImplementedError("Only square lattice for now")

        self.periodic = periodic

        self.nn = self.list_nearest_neighbors()
        # self.nnn    = self.list_next_nearest_neighbors()

    def nearest_neighbors(self, i: int) -> list[int]:
        assert i < self.L_tot, "_nn: index out of range"

        if self.dim == 1:
            nn = self._nn_1d(i)
        elif self.dim == 2:
            if self.shape == "square":
                nn = self._nn_2d_square(i)

        return nn

    def list_nearest_neighbors(self) -> list[list[int]]:
        nn_all = []
        for i in range(self.L_tot):
            nn_all.append(self.nearest_neighbors(i))
        return nn_all

    def next_nearest_neighbors(self, i: int) -> list[int]:
        raise NotImplementedError("Next-nearest-neighbors not implemented yet.")

    def list_next_nearest_neighbors(self) -> list[list[int]]:
        nnn_all = []
        for i in range(self.L_tot):
            nnn_all.append(self.next_nearest_neighbors(i))

    def unravel(self, ilist: list[int], dim: int, shape: str) -> int:
        assert (
            len(ilist) == dim
        ), "List of coordinates should be the same as lattice dimension"

        if dim == 1:
            return ilist[0]
        elif dim == 2:
            if shape == "square":
                return self._unravel_square(ilist[0], ilist[1])
            else:
                raise NotImplementedError("Only square lattices in 2-D so far")
        else:
            raise NotImplementedError("Only 1- and 2-D lattices so far")

    def _nn_1d(self, i: int) -> list[int]:
        Ltot = self.L_tot
        if not self.periodic:
            if i == 0:
                return [i + 1]
            if i == Ltot - 1:
                return [i - 1]

        # sorted() handles PBC (L - 1 + 1 === 0)
        return sorted(list(set([(i - 1) % Ltot, (i + 1) % Ltot])))

    def _nn_2d_square(self, i):
        Lx = self.L[0]
        Ly = self.L[1]

        # Index i = ix + Lx * iy (x-major order)
        ix = i % Lx
        iy = i // Lx

        ix_plus = (ix + 1) % Lx
        iy_plus = (iy + 1) % Ly
        ix_minus = (ix - 1) % Lx
        iy_minus = (iy - 1) % Ly

        north = self._unravel_square(ix, iy_plus)
        south = self._unravel_square(ix, iy_minus)
        east = self._unravel_square(ix_plus, iy)
        west = self._unravel_square(ix_minus, iy)

        # Handle the edges and corners
        if not self.periodic:
            if ix == 0:
                if iy == 0:
                    return sorted([north, east])
                elif iy == self.L[1] - 1:
                    return sorted([south, east])
                else:
                    return sorted([north, south, east])
            elif ix == self.L[0] - 1:
                if iy == 0:
                    return sorted([north, west])
                elif iy == self.L[1] - 1:
                    return sorted([south, west])
                else:
                    return sorted([north, south, west])
            elif iy == 0:
                return sorted([north, east, west])
            elif iy == self.L[1] - 1:
                return sorted([south, east, west])

        # Sorting handles the edges in PBC
        # Need unique elements if L = 2
        return sorted(list(set([north, south, east, west])))

    def _unravel_square(self, ix: int, iy: int) -> int:
        return ix + self.L[0] * iy

    def lattice_info(self):
        size_str = str(self.L[0])
        for Li in self.L[1:]:
            size_str += " x " + str(Li)

        print()
        print("-------------------- Lattice Information --------------------")
        print("-------------------------------------------------------------")
        print("Dimension                        : ", self.dim)
        print("Size                             : ", size_str)
        print("Shape                            : ", self.shape.lower())
        print("PBC                              : ", str(self.periodic).lower())
        print("-------------------------------------------------------------")
        print()
