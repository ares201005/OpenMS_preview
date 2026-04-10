import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from ga_mband_args import FermionGASCF, FermionGASCFResult

# return boson annihilation operator with M excitations
def _get_annihilation_operator(M):
    B = np.zeros((M, M))
    for n in range(1, M):
        B[n-1, n] = np.sqrt(n)
    return B

class FermiBoseGASCFResult(FermionGASCFResult):
    def __init__(self, c, *args, **kwargs):
        self.c = c
        super().__init(*args, **kwargs)


class FermiBoseGASCF(FermionGASCF):
    def __init__(self, BM, omega, *args):
        self.BN = len(BM)
        if (len(omega) != self.BN):
            raise ValueError("Number of modes is not consistent")
        self.BM = BM
        self.omega = [omega]
        self._nfock = np.prod(BM)
        super().__init__(*args)

    def boson_tuple_to_idx(self, tup):
        BN = self.BN
        if (len(tup) != BN):
            raise ValueError("Tuple does not correspond to a valid Fock state")
        idx = tup[-1]
        if not (0 <= idx < self.BM[-1]):
            raise ValueError(f"Invalid occupation number {idx} for mode {BN-1}")
        for i in range(BN - 1):
            j = BN - 2 - i
            n = tup[j]
            b = self.BM[j]
            if not (0 <= n < b):
                raise ValueError(f"Invalid occupation number {n} for mode {j}")
            idx = n + (idx * b)
        return idx

    def boson_idx_to_tuple(self, idx):
        if not (0 <= idx < self._nfock):
            raise ValueError("Index does not correspond to a valid Fock state")
        BN = self.BN
        tup = [0] * BN
        for nu in range(BN):
            idx, tup[nu] = divmod(idx, self.BM[nu])
        return tuple(tup)

    @abstractmethod  
    def get_g(self, nu, I, J):
        pass

    def _get_g(self, nu, I, J):
        res = self.get_g(nu, I, J)
        if (res.shape != (self.M[I], self.M[J])):
            raise ValueError(f"g{nu}[{I}, {J}] has incorrect shape")
        return res

    @abstractmethod
    def get_h(self, I):
        pass
    
    def get_ht(self, I):
        return self.get_h(I)
    
    @abstractmethod
    def get_t(self, I, J):
        pass
    
    def get_tt(self, I, J):
        return self.get_t(I, J)
    
    def _get_annahilation_operator(self, nu):
        B = np.eye(int(np.prod(self.BM[:nu])))
        B = np.kron(_get_annihilation_operator(self.BM[nu]), B)
        B = np.kron(np.eye(int(np.prod(self.BM[nu+1:]))), B)
        return B
    
    def _pack_boson_vector(self, x, c):
        assert (len(c) == self._nfock)
        return np.concatenate([x, c.real, c.imag])
    
    def _unpack_boson_vector(self, y):
        nfock = self._nfock
        assert (len(y) > 2*nfock)
        x = y[:-2*nfock]
        re = y[-2*nfock:-nfock]
        im = y[-nfock:]
        c = re + 1j*im
        return x, c
    
    # def _compute_lagrangian(self, y):
    #     x, c = self._unpack_boson_vector(y)
    #     Lag = super()._compute_lagrangian(x, c)
    #     return Lag
    
    # def kernel(self, *args, **kwargs):
    #     breakpoint()


class HubbardHolstein(FermiBoseGASCF):
    def __init__(self, N=12, filling=0.5, U=1.0, t=-1.0, J=-1.0, g=0.0, omega=1.0, nstates=3, PBC=False):
        if (N <= 0):
            raise ValueError("number of sites must be positive")
        if ((filling <= 0) or (filling >= 1)):
            raise ValueError("filling must be in between 0 and 1 exclusive")
        Ne = int(filling * N * 2)
        self.U = np.zeros((2,2,2,2))
        self.U[0,1,0,1] = -U
        self.t = (t*np.eye(2)) + (J*np.array([[-1, 1], [1, -1]]))
        self.g = g * np.sqrt(omega/2) * np.eye(2)
        self.PBC = PBC
        self.msg = f"N={N}, t={t}, U={U}, J={J}, PBC={PBC}, filling={Ne/(N*2)}"
        super().__init__([nstates], [omega], N*[2], Ne)

    def get_h(self, I):
        return np.zeros((2,2))
    
    def get_U(self, I):
        return self.U
    
    def get_t(self, I, J):
        d = abs(I - J)
        if self.PBC:
            d = min(d, self.N - d)
        if (d == 1):
            return self.t
        return np.zeros((2,2))

    def get_g(self, nu, I, J):
        return self.g
    
    def _compute_structure_factor(self, A):
        # Returns S(k_m) on the natural OBC/PBC momentum grid.
        # For OBC: m=1..N, k_m = m*pi/(N+1)
        # For PBC: m=0..N-1, k_m = 2*pi*m/N

        N = self.N
        if self.PBC:
            ms = np.arange(N)
            ks = 2*np.pi*ms/N
            data = np.zeros(N, dtype=float)
            for idx, k in enumerate(ks):
                v = np.exp(1j * k * np.arange(N))
                data[idx] = (1/N) * (v.conj().T @ A @ v).real
            return ks, data
        else:
            ms = np.arange(1, N+1)
            ks = np.pi * ms / (N + 1)
            data = np.zeros(N, dtype=float)
            sites = np.arange(1, N+1)  # 1..N
            for idx, k in enumerate(ks):
                # v is real, so $v^\dagger = v^T$
                v = np.sin(k * sites)
                data[idx] = (2/(N+1)) * (v @ A @ v).real
            return ks, data

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
    
    def kernel(self, verbose=True, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)
        res = super().kernel(verbose=verbose, **kwargs)
        print(f"self.Ne = {self.Ne}")
        print(f"Delta computed Ne = {sum(np.trace(res.Delta(I)) for I in range(self.N))}")
        print(f"correlation computed Ne = {sum(np.trace(res.get_1body_corr(I, I)) for I in range(self.N))}")
        print(f"E = {res.E}")
        
        qS, S = self._spin_structure_factor(res)
        qN, N = self._charge_structure_factor(res)

        plt.figure()
        plt.title(self.msg)
        plt.plot(qS, S, label="Spin structure factor S(q)")
        plt.plot(qN, N, label="Charge structure factor N(q)")
        plt.xlabel("q")
        plt.ylabel("Structure factor")
        plt.legend()
        plt.savefig("results/sf_plot.png")
        plt.show()

        breakpoint()

if __name__ == '__main__':
   gamf = HubbardHolstein()
   gamf.kernel()  