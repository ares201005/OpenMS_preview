import numpy as np
from sys import argv
from ga_mband import *

class FermiHubbard(FermionGASCF):
    def __init__(self, n=4, filling=0.25, U=1.0, t=-1.0, J=0.0, PBC=False):
        if (n <= 0):
            raise ValueError("lattice size must be positive")
        if ((filling <= 0) or (filling >= 1)):
            raise ValueError("filling must be in between 0 and 1 exclusive")
        self.n = n
        N = n**2
        Ne = int(filling * N * 2)
        self.U = np.zeros((2,2,2,2))
        self.U[0,1,0,1] = -U
        self.t = (t*np.eye(2)) + (J*np.array([[-1, 1], [1, -1]]))
        self.PBC = PBC
        self.msg = f"n={n}, N={N}, t={t}, U={U}, J={J} PBC={PBC}, filling={Ne/(N*2)}"
        super().__init__(N*[2], Ne)

    def get_ht(self, I):
        return np.zeros((2,2))
    
    def get_U(self, I):
        return self.U
    
    def tuple_to_idx(self, i, j):
        return i*self.n + j
    
    def idx_to_tuple(self, idx):
        return divmod(idx, self.n)
    
    def is_nn(self, x1, x2):
        i1, j1 = x1
        i2, j2 = x2
        di = abs(i1 - i2)
        dj = abs(j1 - j2)
        if self.PBC:
            di = min(di, self.n - di)
            dj = min(dj, self.n - dj)
        return (di + dj == 1)
    
    def get_tt(self, I, J):
        if self.is_nn(self.idx_to_tuple(I), self.idx_to_tuple(J)):
            return self.t
        return np.zeros((2,2))
    
    def kernel(self, verbose=True, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)
        res = super().kernel(verbose=verbose, **kwargs)
        print(f"self.Ne = {self.Ne}")
        print(f"Delta computed Ne = {sum(np.trace(res.Delta(I)) for I in range(self.N))}")
        print(f"correlation computed Ne = {sum(np.trace(res.get_1body_corr(I, I)) for I in range(self.N))}")
        breakpoint()

if __name__ == '__main__':
    PBC = False
    if (len(argv) <= 3):
        gamf = FermiHubbard(PBC=PBC)
        if (len(argv) == 1):
            gamf.kernel()
        elif (len(argv) == 2):
            gamf.kernel(method=argv[1])
        elif (len(argv) == 3):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = FermiHubbard(n=int(argv[3]), PBC=PBC)
        if (len(argv) == 4):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")  