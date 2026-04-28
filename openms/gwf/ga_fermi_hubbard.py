import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from ga_mband import FermionGASCF

class FermiHubbard(FermionGASCF):
    def __init__(self, N=12, filling=0.5, U=2.0, t=-1.0, J=+1.0, PBC=True):
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

    def _compute_structure_factor(self, A):
        # Returns S(k_m) on the natural OBC/PBC momentum grid.
        # For OBC: m=1..N, k_m = m*pi/(N+1)
        # For PBC: m=0..N-1, k_m = 2*pi*m/N

        N = self.N
        if self.PBC:
            ms = np.arange(N)
            qs = 2*np.pi*ms/N
            data = np.zeros(N, dtype=float)
            for idx, q in enumerate(qs):
                v = np.exp(1j * q * np.arange(N))
                data[idx] = (1/N) * (v.conj().T @ A @ v).real
            return qs, data
        else:
            ms = np.arange(1, N+1)
            qs = np.pi * ms / (N + 1)
            data = np.zeros(N, dtype=float)
            sites = np.arange(1, N+1)  # 1..N
            for idx, q in enumerate(qs):
                # v is real, so $v^\dagger = v^T$
                v = np.sin(q * sites)
                data[idx] = (2/(N+1)) * (v.T @ A @ v).real
            return qs, data

    def _charge_structure_factor(self, res):
        N = self.N
        A = np.zeros((N, N))
        for I in range(N):
            for J in range(N):
                M = res.get_number_corr(I, J) - np.outer(np.diag(res.get_1body_corr(I, I)), np.diag(res.get_1body_corr(J, J)))
                A[I, J] = M.real.sum()
                
        print(f"number (CDW) A sum = {A.sum()}")
        return self._compute_structure_factor(A)

    def _spin_structure_factor(self, res):
        N = self.N
        A = np.zeros((N, N))
        for I in range(N):
            for J in range(N):
                nIJ = res.get_number_corr(I, J)
                A[I, J] = 0.25 * (nIJ[0,0] + nIJ[1,1] - nIJ[0,1] - nIJ[1,0])
      
        return self._compute_structure_factor(A)
    
    def _plot_structure_factors(self, qS, S, qN, N):
        plt.figure()
        plt.title(self.msg)
        plt.plot(qS/np.pi, S, label="Spin structure factor S(q)")
        plt.plot(qN/np.pi, N, label="Charge structure factor N(q)")
        plt.xlabel(r"q/$\pi$")
        plt.ylabel("Structure Factor")
        plt.legend()
        plt.savefig("results/sf_plot.png")
        plt.show()
    
    def kernel(self, verbose=True, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)
        res = super().kernel(verbose=verbose, **kwargs)
        if verbose:
            print(f"self.Ne = {self.Ne}")
            print(f"Delta computed Ne = {sum(np.trace(res.Delta(I)) for I in range(self.N))}")
            print(f"correlation computed Ne = {sum(np.trace(res.get_1body_corr(I, I)) for I in range(self.N))}")
            print(f"E = {res.E}")
        return res
    
    def do_kernel(self, verbose=True, **kwargs):
        res = self.kernel(verbose=verbose, **kwargs)
        if verbose:
            print(f"<Sz> = {sum(np.trace(res.get_1body_corr(I, I) @ np.diag([1,-1])) for I in range(self.N))}")
            qS, S = self._spin_structure_factor(res)
            qN, N = self._charge_structure_factor(res)
            print(f"S(q) maxq/pi={qS[np.argmax(S)]/np.pi}, minq/pi={qS[np.argmin(S)]/np.pi}")
            print(f"N(q) maxq/pi={qN[np.argmax(N)]/np.pi}, minq/pi={qN[np.argmin(N)]/np.pi}")
        self._plot_structure_factors(qS, S, qN, N)

if __name__ == '__main__':
    if (len(argv) <= 3):
        gamf = FermiHubbard()
        if (len(argv) == 1):
            gamf.do_kernel()
        elif (len(argv) == 2):
            gamf.do_kernel(method=argv[1])
        elif (len(argv) == 3):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = FermiHubbard(N=int(argv[3]))
        if (len(argv) == 4):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")  
