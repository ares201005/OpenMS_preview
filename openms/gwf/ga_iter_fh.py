import numpy as np
from sys import argv
import matplotlib.pyplot as plt
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
        d = abs(I - J)
        if self.PBC:
            d = min(d, self.N - d)
        if (d == 1):
            return self.t
        return np.zeros((2,2))
    
    def kernel(self, verbose=True, maxiter=30, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)

        Earr = np.zeros(maxiter)
        max = maxiter-1
        for it in range(maxiter):
            print(f"maxiter = {it+1}")
            res = super().kernel(verbose=verbose, maxiter=it+1, **kwargs)
            Earr[it] = res.E
            if res.result.success:
                max = it
                break     

        plt.figure()
        plt.title(self.msg)
        plt.xticks(range(max-1))
        plt.plot(range(max-1), Earr[0:max-1])
        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.savefig("results/energy_plot.png")
        plt.show()

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