import numpy as np
from pyscf import lib, gto

from pdb import set_trace as st

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

class GASCF(lib.StreamObject):
    def __init__(self, h1e, eri, N, Ne):
        # sanity check inputs
        if (Ne > N):
            raise ValueError
        self.N = N
        self.Ne = Ne

        h1e_shape = h1e.shape
        if (len(h1e_shape) != 2):
            raise ValueError
        if (h1e_shape[0] != h1e_shape[1]):
            raise ValueError
        
        eri_shape = eri.shape
        if (len(eri_shape) != 4):
            raise ValueError
        if (not (eri_shape[0] == eri_shape[1] == eri_shape[2] == eri_shape[3] == h1e_shape[0])):
            raise ValueError
        
        if ((h1e_shape[0] % N) != 0):
            raise ValueError
        
        # extract 1-electron coefficients
        M = int(h1e_shape[0] / N)
        self.M = M
        h = np.zeros((N, M, M))
        h_int = np.zeros((N*M, N*M))
        for I in range(N):
            idx = slice(I*M, (I+1)*M)
            h[I] = h1e[idx, idx]
            h_int[idx, idx] = h1e[idx, idx]
        self.h = h
        t = h1e - h_int
        self.t = np.zeros((N, N, M, M))
        for I in range(N):
            for J in range(N):
                self.t[I, J] = t[slice(I*M, (I+1)*M), slice(J*M, (J+1)*M)]

        # extract 2-electron coefficients
        U = np.zeros((N, M, M, M, M))
        for I in range(N):
            idx = slice(I*M, (I+1)*M)
            U[I] = eri[idx, idx, idx, idx]
            self.U = U

        # TODO: include nonlocal interactions and corresponding renormalizations

        # initialize renormalizations and Lagrange multipliers
        # how to get initial guess?
        self.R = np.ones((N, M, M))
        self.D = np.ones((N, M, M))
        self.Lc = np.zeros((N, M, M))
        self.L = np.zeros((M, M))

        # everything is done in terms of
        # \ket{\Psi_0^e} given by self.mo_occ[I]
        # $\ket{\Psi_I}$ given by self.phi[I]
        self.mo_energy = np.zeros((N*M))
        self.mo_coeff = np.zeros((N*M, N*M))
        self.phi = np.ones((N, 4**M))

        # self.C is the correlation matrix
        self.C = np.zeros((N, N, M, M))

        # set mo_energy, mo_coeff, rho, by solving preliminary Hqp
        self._update_solve_qp()
        st()

    def _get_ht(self, I):
        return self.h[I]
    
    def _get_U(self, I):
        return self.U[I]
    
    def _get_tt(self, I, J):
        return self.t[I, J]
    
    def _update_solve_qp(self):
        N = self.N
        M = self.M

        # construct qp Hamiltonian as an N*M x N*M matrix
        teff = np.zeros((N*M, N*M))
        for I in range (N):
            for J in range(N):
                teff_IJ = self.R[I].T @ self._get_tt(I, J) @ self.R[J].conj()
                for a in range(M):
                    for b in range(M):
                        teff[I*M + a, I*M + b] = teff_IJ[a, b]
        Hqp = teff

        # diagonalize Hqp and calculate correlation matrix
        self.mo_energy, self.mo_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self.Ne) for i in range(N*M)]
        C = self.mo_coeff[:, occ].conj() @ self.mo_coeff[:, occ].T
        for I in range(N):
            for J in range(N):
                self.C[I, J] = C[slice(I*M, (I+1)*M), slice(J*M, (J+1)*M)]


    def _update_solve_eb(self):
        pass

    def kernel(self):
        pass
        self._post_kernel()

    def _post_kernel(self):
        pass


def get_ga_model(N=12, filling=0.5, U=2.0, t=-1.0, PBC=True):
    dim = N * 2
    Ne = int(N * filling)

    # 1-electron interactions
    h1e = np.zeros((dim, dim))
    for I in range(N-1):
        for a in range(2):
            h1e[I*2 + a, (I+1)*2 + a] = t
    if PBC:
        for a in range(2):
            h1e[(N-1)*2 + a, a] = t

    # 2-electron interactions
    eri = np.zeros((dim, dim, dim, dim))
    for I in range(N):
        eri[I*2, I*2, I*2+1, I*2+1] = U

    return GASCF(h1e, eri, N, Ne)

if __name__ == '__main__':
    gamf = get_ga_model(PBC=False)
    gamf.kernel()