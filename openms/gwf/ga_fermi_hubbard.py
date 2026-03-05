import numpy as np
from sys import argv
from ga_mband import FermionGASCF

class FermiHubbard(FermionGASCF):
    def __init__(self, N=12, filling=0.5, U=1.0, t=-1.0, J=0.0, PBC=True):
        if (N <= 0):
            raise ValueError("number of sites must be positive")
        if ((filling <= 0) or (filling >= 1)):
            raise ValueError("filling must be in between 0 and 1 exclusive")
        Ne = int(filling * N * 2)
        self.U = U
        self.t = t
        self.J = J
        self.PBC = PBC
        super().__init__(N, Ne, 2)

    def get_ht(self, I):
        return np.zeros((2,2))
    
    def get_U(self, I):
        res = np.zeros((2,2,2,2))
        res[0, 1, 0, 1] = -self.U
        return res
    
    def get_tt(self, I, J):
        if (abs(I - J) == 1):
            return self.t * np.eye(2) + self.J * np.diag([-1, 1])
        if (self.PBC and ({I, J} == {0, self.N-1})):
            return self.t * np.eye(2) + self.J * np.diag([-1, 1])
        return  np.zeros((2,2))
    
    def kernel(self, method="krylov", maxiter=None, tolerance=1e-6, verbose=True):
        res = super().kernel(method=method, maxiter=maxiter, tolerance=tolerance, verbose=verbose)
        print(f"Computed Ne = {sum(np.trace(res['Delta'][I]) for I in range(self.N))}, self.Ne = {self.Ne}")
        breakpoint()

if __name__ == '__main__':
    if (len(argv) <= 3):
        gamf = FermiHubbard(PBC=True)
        if (len(argv) == 1):
            gamf.kernel()
        elif (len(argv) == 2):
            gamf.kernel(method=argv[1])
        elif (len(argv) == 3):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = FermiHubbard(N=int(argv[3]), PBC=True)
        if (len(argv) == 4):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")  