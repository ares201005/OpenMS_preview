import numpy as np
import scipy.linalg
from pyscf import lib

from pdb import set_trace as st

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

class GASCF(lib.StreamObject):
    def __init__(self, h1e, eri, N, Ne, embedding_cutoff=1e-14):
        # sanity check inputs
        if (Ne > h1e.shape[0]):
            raise ValueError
        self._N = N
        self._Ne = Ne
        self._cutoff = embedding_cutoff

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
        self._M = M
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
        self.R = np.broadcast_to(np.eye(M), (N, M, M)).copy()

        # everything is done in terms of
        # \ket{\Psi_0^e} given by self.mo_occ[I]
        # $\ket{\Psi_I}$ given by self.phi[I]
        self.mo_energy = np.zeros((N*M))
        self.mo_coeff = np.zeros((N*M, N*M))

        # initialize all phi to uniformly id on each block (unentangled)
        phi = np.zeros(4**M)
        for Gamma in range(2**M):
            phi[Gamma*(2**M) + Gamma] = 1

        self.phi = np.zeros((N, 4**M))
        for I in range(N):
            self.phi[I] = phi

        # self._C is the correlation matrix
        self._C = np.zeros((N, N, M, M))

    def _get_ht(self, I):
        return self.h[I]
    
    def _get_U(self, I):
        return self.U[I]
    
    def _get_tt(self, I, J):
        return self.t[I, J]

    def _get_annahilation_operators(self, M):
        C = np.zeros((M, 2**M, 2**M))
        for i in range(M):
            for state in range(2**M):
                if (state & (1 << i)):
                    C[i, state & ~(1 << i), state] = (-1)**(bin(state >> (i+1)).count('1'))
        return C
    
    def _update_renormalizations(self):
        for I in range(self._N):
            # get local state and correlation
            M = self._M
            C = self._get_annahilation_operators(M)
            Delta = self._C[I, I]
            mat = scipy.linalg.sqrtm(np.linalg.inv(Delta @ (np.eye(M) - Delta)))
            phi = self.phi[I]

            # compute R
            opmat = np.zeros((M, M))
            for alpha in range(M):
                for b in range(M):
                    opmat[alpha, b] = phi.conj().T @ (np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[b].T)) @ phi
            self.R[I] = opmat @ mat
            
    
    def _update_solve_qp(self):
        N = self._N
        M = self._M

        # construct qp Hamiltonian as an N*M x N*M matrix
        teff = np.zeros((N*M, N*M))
        for I in range (N):
            for J in range(N):
                teff_IJ = self.R[I].T @ self._get_tt(I, J) @ self.R[J].conj()
                for a in range(M):
                    for b in range(M):
                        teff[I*M + a, J*M + b] = teff_IJ[a, b]
        Hqp = teff

        # missing logic for lambda

        # diagonalize Hqp and calculate correlation matrix
        self.mo_energy, self.mo_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self._Ne) for i in range(N*M)]
        C = self.mo_coeff[:, occ].conj() @ self.mo_coeff[:, occ].T
        for I in range(N):
            for J in range(N):
                self._C[I, J] = C[slice(I*M, (I+1)*M), slice(J*M, (J+1)*M)]
        
    def get_phi_matrix(self, phi):
        M = self._M
        matrix = np.zeros((2**M, 2**M))
        for Gamma in range(2**M):
            for n in range(2**M):
                matrix[Gamma, n] = phi[Gamma*(2**M) + n]
        return matrix
    
    def _update_solve_eb(self):
        N = self._N
        for I in range(N):
            M = self._M
            phi = self.phi[I]

            # calculate Lagrange multipliers
            D_bare = sum((self._get_tt(I, J) @ self.R[J].conj() @ self._C[I, J].T) for J in range(N))

            # get coefficients for embedding Hamiltonian
            Delta_T = self._C[I, I].T
            h = self._get_ht(I)
            U = self._get_U(I)
            D = D_bare @ (scipy.linalg.sqrtm(np.linalg.inv(Delta_T @ (np.eye(M) - Delta_T))))

            C = self._get_annahilation_operators(M)
            
            # construct local Hamiltonian
            Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
            Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
            
            # construct lambda part of the Hamiltonian
            HL = np.zeros((2**M, 2**M))
            for a in range(M):
                for b in range(M):
                    X = np.zeros((M, M))
                    X[a, b] = 1
                    H = (X @ (np.eye(M) - Delta_T.T)) - (Delta_T.T @ X)
                    B = Delta_T.T @ (np.eye(M) - Delta_T.T)
                    Y = scipy.linalg.solve_lyapunov(scipy.linalg.sqrtm(B), -H)
                    LC = sum(D[alpha, c] * (phi.T.conj() @ np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[d].T) @ phi) * Y[d, c] for alpha in range(M) for c in range(M) for d in range(M))
                    HL += LC * C[b].T @ C[a]
            HL = np.kron(np.eye(2**M), HL)
            
            # construct Lagrange multiplier part of the Hamiltonian
            HM = sum(D[a, b] * np.kron(C[a].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[b].T) for a in range(M) for b in range(M)) 
            
            # construct and solve embedding Hamiltonian
            Hemb = np.kron(Hloc, np.eye(2**M)) + (HM + HM.conj().T) + (HL + HL.conj().T)
            E, eigvec = np.linalg.eigh(Hemb)
            phiC = np.where(np.abs(eigvec) < self._cutoff, 0, eigvec)

            # TODO check that groundstate is nondegenerate
            if (E[0] == E[1]):
                # groundstate is degenerate
                st()
            self.phi[I] = phiC[:,0]

    def kernel(self):
        for i in range(100):
            prev_coeff = self.mo_coeff
            # set mo_energy, mo_coeff, C, by solving  Hqp
            self._update_solve_qp()
            # set phi by solving Hemb
            self._update_solve_eb()

            self._update_renormalizations()

            print(f"Iteration {i}")   
            
        st()

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
            h1e[(I+1)*2 + a, I*2 + a] = t
    if PBC:
        for a in range(2):
            h1e[(N-1)*2 + a, a] = t
            h1e[a, (N-1)*2 + a] = t

    # 2-electron interactions
    eri = np.zeros((dim, dim, dim, dim))
    for I in range(N):
        eri[I*2, I*2+1, I*2, I*2+1] = -U

    return GASCF(h1e, eri, N, Ne)

if __name__ == '__main__':
    gamf = get_ga_model(PBC=False)
    gamf.kernel()