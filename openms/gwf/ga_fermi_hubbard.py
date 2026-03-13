import numpy as np
from sys import argv
from ga_mband import FermionGASCF

class FermiHubbard(FermionGASCF):
    def __init__(self, N=12, filling=0.5, U=1.0, t=-1.0, J=-1.0, PBC=False):
        if (N <= 0):
            raise ValueError("number of sites must be positive")
        if ((filling <= 0) or (filling >= 1)):
            raise ValueError("filling must be in between 0 and 1 exclusive")
        Ne = int(filling * N * 2)
        self.U = np.zeros((2,2,2,2))
        self.U[0,1,0,1] = -U
        self.t = (t*np.eye(2)) + (J*np.array([[-1, 1], [1, -1]]))
        self.PBC = PBC
        self.msg = f"N={N}, t={t}, U={U}, J={J}, PBC={PBC}, filling={Ne/(N*2)}"
        super().__init__(N*[2], Ne)

    def get_ht(self, I):
        return np.zeros((2,2))
    
    def get_U(self, I):
        return self.U
    
    def get_tt(self, I, J):
        link = False
        if (abs(I-J) == 1):
            link = True
        elif (self.PBC and ({I, J} == {0, self.N-1})):
            link = True
        if link:
            return self.t
        return np.zeros((2,2))
    
    def kernel(self, verbose=True, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)
        res = super().kernel(verbose=verbose, **kwargs)
        print(f"Computed Ne = {sum(np.trace(res['Delta'][I]) for I in range(self.N))}, self.Ne = {self.Ne}")
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
        gamf = FermiHubbard(N=int(argv[3]), PBC=PBC)
        if (len(argv) == 4):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            gamf.kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")  