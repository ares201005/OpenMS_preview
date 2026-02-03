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
        self.N = N
        self.Ne = Ne
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
        for I in range(self.N):
            # get local state and correlation
            M = self.M
            C = self._get_annahilation_operators(M)
            Delta = self._C[I, I]
            B = scipy.linalg.sqrtm(np.linalg.inv(Delta @ (np.eye(M) - Delta)))
            phi = self.phi[I]

            # compute R
            opmat = np.zeros((M, M))
            for alpha in range(M):
                for b in range(M):
                    opmat[alpha, b] = phi.conj().T @ (np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[b].T)) @ phi
            self.R[I] = opmat @ B
            
    
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
                        teff[I*M + a, J*M + b] = teff_IJ[a, b]
        Hqp = teff

        # missing logic for lambda

        # diagonalize Hqp and calculate correlation matrix
        self.mo_energy, self.mo_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self.Ne) for i in range(N*M)]
        C = self.mo_coeff[:, occ].conj() @ self.mo_coeff[:, occ].T
        for I in range(N):
            for J in range(N):
                self._C[I, J] = C[slice(I*M, (I+1)*M), slice(J*M, (J+1)*M)]
        
    def get_phi_matrix(self, phi):
        M = self.M
        matrix = np.zeros((2**M, 2**M))
        for Gamma in range(2**M):
            for n in range(2**M):
                matrix[Gamma, n] = phi[Gamma*(2**M) + n]
        return matrix
    
    def _update_solve_eb(self):
        N = self.N
        for I in range(N):
            M = self.M

            # get coefficients for embedding Hamiltonian
            h = self._get_ht(I)
            U = self._get_U(I)
            Delta = self._C[I, I]
            A = scipy.linalg.sqrtm(Delta @ (np.eye(M) - Delta))
            B = np.linalg.inv(A)

            # calculate Lagrange multipliers
            D = sum((self._get_tt(I, J) @ self.R[J].conj() @ self._C[I, J].T @ B.T) for J in range(N))

            C = self._get_annahilation_operators(M)
            
            # construct local Hamiltonian
            Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
            Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
            
            # construct lambda part of the Hamiltonian
            HL = np.zeros((2**M, 2**M))
            for a in range(M):
                for b in range(M):
                    E = np.zeros((M, M))
                    E[a, b] = 1
                    H = (E @ (np.eye(M) - Delta)) - (Delta @ E)
                    Z = scipy.linalg.solve_continuous_lyapunov(A, H)
                    LC = -np.trace(D.T @ self.R[I] @ Z)
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

    
    # phi is a vector of size N, containing embedding matrix states each of size 2**M x 2**M
    # L is a vector of size N, containing matrices of size MxM
    # Lc is a vector of size N, containing matrices of size MxM
    # Delta is a vector of size N, containing Hermitian matrices of size MxM
    # Ec is a vector of size N containing real scalar energies
    def _pack_vector(self, phiarr, L, Lc, Delta, Ec):
        parts = []
        N = self.N
        M = self.M
        for i in range(N):
            parts.append(phiarr[i].real.ravel())
            parts.append(phiarr[i].imag.ravel())
        for i in range(N):
            parts.append(L[i].real.ravel())
            parts.append(L[i].imag.ravel())
        for i in range(N):
            parts.append(Lc[i].real.ravel())
            parts.append(Lc[i].imag.ravel())
        for i in range(N):
            parts.append(Delta[i].real.ravel())
            parts.append(Delta[i].imag.ravel())
        parts.append(np.asarray(Ec).ravel())

        print("phi sizes:", [phiarr[i].size for i in range(N)])
        print("L sizes:  ", [L[i].size for i in range(N)])
        print("Lc sizes: ", [Lc[i].size for i in range(N)])
        print("Delta sizes:", [Delta[i].size for i in range(N)])
        print("Ec size:", np.asarray(Ec).ravel().size)

        assert all(phiarr[i].size == 4**M for i in range(N))
        assert all(L[i].size == M*M for i in range(N))
        assert all(Lc[i].size == M*M for i in range(N))
        assert all(Delta[i].size == M*M for i in range(N))
        assert np.asarray(Ec).ravel().size == N

        return np.concatenate(parts)
    
    def _unpack_vector(self, x):
        idx = 0
        N, M = self.N, self.M

        phiarr = []
        size_phi = 4**M
        for _ in range(N):
            Re = x[idx:(idx + size_phi)].reshape(size_phi)
            idx += size_phi
            Im = x[idx:(idx + size_phi)].reshape(size_phi)
            idx += size_phi
            phiarr.append(Re + 1j*Im)
        print(f"idx = {idx}")

        L = []
        size_sq = M*M
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            L.append(Re + 1j*Im)
        print(f"idx = {idx}")
        Lc = []
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Lc.append(Re + 1j*Im)
        print(f"idx = {idx}")

        Delta = []
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Delta.append(Re + 1j*Im)
        print(f"idx = {idx}")

        Ec = x[idx:(idx+N)]
        idx += N
        print(f"idx = {idx}")

        assert idx == len(x), "Unpack error: leftover elements in input array"
        return phiarr, L, Lc, Delta, Ec

    def _compute_renormalizations(self, phiarr, Delta):
        R = []
        for I in range(self.N):
            # get local state and correlation
            M = self.M
            C = self._get_annahilation_operators(M)
            B = scipy.linalg.sqrtm(np.linalg.inv(Delta[I] @ (np.eye(M) - Delta[I])))
            phi = phiarr[I]

            # compute R
            opmat = np.zeros((M, M), dtype=np.complex128)
            for alpha in range(M):
                for b in range(M):
                    opmat[alpha, b] = phi.conj().T @ (np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[b].T)) @ phi
            R.append(opmat @ B)

        return R
    
    def _compute_Hqp(self, phiarr, Delta, L):
        N, M = self.N, self.M
        R = self._compute_renormalizations(phiarr, Delta)

        teff = np.zeros((N*M, N*M), dtype=np.complex128)
        for I in range (N):
            for J in range(N):
                teff_IJ = R[I].T @ self._get_tt(I, J) @ R[J].conj()
                for a in range(M):
                    for b in range(M):
                        teff[I*M + a, J*M + b] = teff_IJ[a, b]
        Hqp = teff
        lqp = np.zeros((N*M, N*M), dtype=np.complex128)
        for I in range(N):
            for a in range(M):
                for b in range(M):
                   lqp[I*M + a, I*M + b] = L[I][a, b] 
        Hqp += lqp + lqp.T.conj()

        return Hqp
    
    def _compute_Hemb(self, I, Lc):
        M = self.M
        # get coefficients for embedding Hamiltonian
        h = self._get_ht(I)
        U = self._get_U(I)
        C = self._get_annahilation_operators(M)
        # construct local Hamiltonian
        Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
        Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
        # construct lambda part of the Hamiltonian
        HL = np.kron(np.eye(2**M), sum(Lc[I][a, b] * C[b].T @ C[a] for a in range(M) for b in range(M)))
        # construct and solve embedding Hamiltonian
        Hemb = np.kron(Hloc, np.eye(2**M)) + HL + HL.T.conj()
        return Hemb
    
    # this function assumes that the Hqp eigenenergies are sorted from lowest to highest
    def _compute_firstorder_perturubation_energy(self, eigvals, eigmatrix, DH):
        # get the degenerate subspaces of Hqp
        boundaries = np.where(np.diff(eigvals) != 0)[0] + 1
        indices = np.arange(len(eigvals))
        degenerate_indices = np.split(indices, boundaries)

        # project onto the eigenbasis and calculate 1st order perturbations
        DH_eig = eigmatrix.conj().T @ DH @ eigmatrix
        E1 = np.zeros(len(eigvals))
        c = []
        for indices in degenerate_indices:
            H1 = DH_eig[np.ix_(indices, indices)]
            energies, coeff = np.linalg.eigh(H1)
            E1[indices] = energies
            c.append(coeff)

        return sum(E1[:self.Ne]), E1, c
    
    def _compute_derivative_energy(self, eigmatrix, DH):
        return sum((eigmatrix[:, i].T.conj() @ DH @ eigmatrix[:, i]) for i in range(self.Ne))

    def _compute_lagrangian(self, x):
        N = self.N
        phiarr, L, Lc, Delta, Ec = self._unpack_vector(x)
        Lag = 0
        
        # calculate Hqp and energies of filled eigenstates
        Hqp = self._compute_Hqp(phiarr, Delta, L)
        mo_energy, mo_coeff = np.linalg.eigh(Hqp)
        Lag += sum(mo_energy[i] for i in range(self.Ne))

        # get embedding Hamiltonian expectations and normalization terms
        for I in range(N):
            Hemb = self._compute_Hemb(I, Lc)
            phi = phiarr[I]
            Lag += phi.T.conj() @ Hemb @ phi
            # calculate Ec contribution
            Lag += Ec[I] * (1 - (phi.T.conj() @ phi))

        #  calculate Lmix
        Lmix = -sum(np.trace((L[I] + Lc[I]) @ Delta[I].T) for I in range(N))
        Lag += Lmix + Lmix.conj()

        return Lag

    def _compute_gradient(self, x):
        N = self.N
        phiarr, L, Lc, Delta, Ec = self._unpack_vector(x)

        # get energy gradients
        grad_Ec = [1 - (phiarr[I].T.conj() @ phiarr[I]) for I in range(N)]

        # get quasiparticle and embedding Hamiltonians
        Hqp = self._compute_Hqp(phiarr, Delta, L)
        mo_energy, mo_coeff = np.linalg.eigh(Hqp)
        Hemb = [self._compute_Hemb(I, Lc) for I in range(N)]
        R = self._compute_renormalizations(phiarr, Delta)

        occ = [(i < self.Ne) for i in range(N*self.M)]
        full_correlation = mo_coeff[:, occ].conj() @ mo_coeff[:, occ].T

        # get |phi_I> gradients, each of size 4**M
        grad_phiarr = []
        for K in range(N):
            M = self.M
            C = self._get_annahilation_operators(M)
            A = scipy.linalg.sqrtm(Delta[K] @  (np.eye(M) - Delta[K]))
            B = np.linalg.inv(A)
            
            # construct derivative Hamiltonian
            HD = Hemb[K] - Ec[K]*np.eye(4**M)
            HDqp = np.zeros((4**M, 4**M), dtype=np.complex128)
            for I in range(N):
                ttIK = self._get_tt(I, K)
                Delta_mixed = full_correlation[I*M:(I+1)*M, K*M:(K+1)*M]
                M1 = ttIK.T @ R[I] @ Delta_mixed @ B
                HDqp += sum((M1[alpha, gamma] * np.kron(C[alpha], np.eye(2**M)) @ np.kron(np.eye(2**M), C[gamma])) for alpha in range(M) for gamma in range(M))
            HDqp = HDqp + HDqp.T.conj()
            HD += HDqp

            grad_phiarr.append(HD @ phiarr[K])

        # get lambda gradients
        grad_L = []
        for I in range(N):
            M = self.M
            Lm = np.zeros((M,M))
            for a in range(M):
                for b in range(M):
                    DH = np.zeros(Hqp.shape)
                    DH[I*M + a, I*M + b] = 1
                    Lm[a, b] = self._compute_derivative_energy(mo_coeff, DH)
            grad_L.append(Lm - Delta[I])

        grad_Lc = []
        for I in range(N):
            M = self.M
            Lm = np.zeros((M,M))
            C = self._get_annahilation_operators(M)
            for a in range(M):
                for b in range(M):
                    Lm[a,b] = phiarr[I].conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ phiarr[I]
            grad_Lc.append(Lm - Delta[I])

        # get Delta gradients
        grad_Delta = []
        for K in range(N):
            M = self.M

            # get Lmix contribution
            Dm = -L[K] - Lc[K]

            # calculate renormalization based matrix
            P = np.zeros((M,M))
            C = self._get_annahilation_operators(M)
            for alpha in range(M):
                for gamma in range(M):
                    P[alpha, gamma] = phiarr[K].conj().T @ np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[gamma].T) @ phiarr[K]
            
            A = scipy.linalg.sqrtm(Delta[K] @  (np.eye(M) - Delta[K]))
            B = np.linalg.inv(A)

            # calculate Hqp contribution element by element
            for y in range(M):
                for z in range(M):
                    # calculate helper matrices for Hqp contribution
                    E = np.zeros((M, M))
                    E[y, z] = 1
                    H = (E @ (np.eye(M) - Delta[K])) - (Delta[K] @ E)
                    Z = scipy.linalg.solve_continuous_lyapunov(A, H)
                    Y = -B @ Z @ B
                    # calculate $\frac{\partial{H_{qp}}}{\partial{\Delta^K_{yz}}}$
                    DH = np.zeros(Hqp.shape)
                    for I in range(N):
                        M1 = Y.T @ P.T @ self._get_tt(K, I) @ R[I].conj()
                        M2 = R[I].T @ self._get_tt(I, K) @ P.conj() @ Y.T
                        for a in range(M):
                            for b in range(M):
                                DH[K*M + a, I*M + b] += M1[a,b]
                                DH[I*M + a, K*M + b] += M2[a, b]
                    # calculate contribution to the full derivative
                    Dm[y, z] += self._compute_derivative_energy(mo_coeff, DH)

            grad_Delta.append(Dm)

        # return packed vector
        return self._pack_vector(grad_phiarr, grad_L, grad_Lc, grad_Delta, grad_Ec)

    def naive_kernel(self):
        for i in range(100):
            prev_coeff = self.mo_coeff
            # set mo_energy, mo_coeff, C, by solving  Hqp
            self._update_solve_qp()
            # set phi by solving Hemb
            self._update_solve_eb()

            self._update_renormalizations()

            print(f"Iteration {i+1}: diff {np.linalg.norm(prev_coeff - self.mo_coeff)}")
            
        st()

        self._post_kernel()

    def kernel(self):
        # construct initial guess
        N = self.N
        phiarr = [self.phi[I] for I in range(N)]
        Ec = np.zeros(N)

        M = self.M
        Delta = [(self.Ne/M)*np.eye(M) for I in range(N)]

        L = [np.zeros((M,M)) for I in range(N)]
        Lc = [np.zeros((M,M)) for I in range(N)]

        x0 = self._pack_vector(phiarr, L, Lc, Delta, Ec)
        
        result = scipy.optimize.root(self._compute_gradient, x0, method="hybr" or "krylov")

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