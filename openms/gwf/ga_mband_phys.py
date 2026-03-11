import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

class FermionGASCF(ABC):
    def __init__(self, N, Ne, M):
        self.N = N
        self.Ne = Ne
        self.M = M
        self.filling = Ne / (N * M)

    @abstractmethod
    def get_ht(self, I):
        pass

    def _get_ht(self, I):
        M = self.M
        res = self.get_ht(I)
        if (res.shape != (M, M)):
            raise ValueError(f"ht[{I}] has incorrect shape")
        return res
    
    @abstractmethod
    def get_U(self, I):
        pass

    def _get_U(self, I):
        M = self.M
        res = self.get_U(I)
        if (res.shape != (M, M, M, M)):
            raise ValueError(f"U[{I}] has incorrect shape")
        return res
    
    @abstractmethod
    def get_tt(self, I, J):
        pass

    def _get_tt(self, I, J):
        res = self.get_tt(I, J)
        if (res.shape != (self.M, self.M)):
            raise ValueError(f"tt[{I}, {J}] has incorrect shape")
        return res

    def _get_annahilation_operators(self, M):
        C = np.zeros((M, 2**M, 2**M))
        for i in range(M):
            for state in range(2**M):
                if (state & (1 << i)):
                    C[i, state & ~(1 << i), state] = (-1)**(bin(state >> (i+1)).count('1'))
        return C           
    
    # psiarr is a vector of size N, containing embedding matrix states each of size 2**M x 2**M encoded as a 4**M component vector
    # L is a vector of size N, containing matrices of size MxM
    # Lc is a vector of size N, containing matrices of size MxM
    # Delta is a vector of size N, containing Hermitian matrices of size MxM
    # Ec is a vector of size N containing real scalar energies
    def _pack_vector(self, psiarr, L, Lc, Delta, Ec):
        parts = []
        N = self.N
        M = self.M
        for i in range(N):
            parts.append(psiarr[i].real.ravel())
            parts.append(psiarr[i].imag.ravel())
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

        assert all(psiarr[i].size == 4**M for i in range(N))
        assert all(L[i].size == M**2 for i in range(N))
        assert all(Lc[i].size == M**2 for i in range(N))
        assert all(Delta[i].size == M**2 for i in range(N))
        assert np.asarray(Ec).ravel().size == N

        result = np.concatenate(parts)
        assert all(result.imag == 0)
        return result.real
    
    def _unpack_vector(self, x):
        idx = 0
        N, M = self.N, self.M

        psiarr = []
        size_psi = 4**M
        for _ in range(N):
            Re = x[idx:(idx + size_psi)].reshape(size_psi)
            idx += size_psi
            Im = x[idx:(idx + size_psi)].reshape(size_psi)
            idx += size_psi
            psiarr.append(Re + 1j*Im)

        L = []
        size_sq = M*M
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            L.append(Re + 1j*Im)
        Lc = []
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Lc.append(Re + 1j*Im)

        Delta = []
        for _ in range(N):
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Delta.append(Re + 1j*Im)

        Ec = x[idx:(idx+N)]
        idx += N

        assert idx == len(x), "Unpack error: leftover elements in input array"
        return psiarr, L, Lc, Delta, Ec
    
    def get_psi_matrix(self, psi):
        psi_size = int(np.sqrt(psi.size))
        matrix = np.zeros((psi_size, psi_size), dtype=psi.dtype)
        for Gamma in range(psi_size):
            for n in range(psi_size):
                matrix[Gamma, n] = psi[Gamma*(psi_size) + n]
        return matrix
    
    def get_psi_vector(self, psi):
        a, b = psi.shape
        vec = np.zeros(psi.size, dtype=np.complex128)
        for Gamma in range(a):
            for n in range(b):
                vec[Gamma*a + n] = psi[Gamma, n]
        return vec     

    def _compute_renormalizations(self, psiarr, Delta):
        R = []
        for I in range(self.N):
            # get local state and correlation
            M = self.M
            C = self._get_annahilation_operators(M)
            B = scipy.linalg.sqrtm(np.linalg.inv(Delta[I] @ (np.eye(M) - Delta[I])))
            psi = psiarr[I]

            # compute R
            opmat = np.zeros((M, M), dtype=np.complex128)
            for alpha in range(M):
                for b in range(M):
                    opmat[alpha, b] = psi.conj().T @ (np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[b].T)) @ psi
            R.append(opmat @ B)

        return R
    
    def _compute_Hqp(self, psiarr, Delta, L):
        N, M = self.N, self.M
        R = self._compute_renormalizations(psiarr, Delta)

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
    def _compute_firstorder_perturbation_energy(self, eigvals, eigmatrix, DH):
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
    
    def _compute_rdm(self, Delta):
        M, _ = Delta.shape
        K = scipy.linalg.logm((np.eye(M) - Delta) @ np.linalg.inv(Delta))
        C = self._get_annahilation_operators(M)
        rho = scipy.linalg.expm(sum(-K[a, b] * C[a].T @ C[b] for a in range(M) for b in range(M)))
        return (1/np.trace(rho)) * rho

    def _compute_lagrangian(self, x):
        N = self.N
        psiarr, L, Lc, Delta, Ec = self._unpack_vector(x)
        Lag = 0
        
        # calculate Hqp and energies of filled eigenstates
        Hqp = self._compute_Hqp(psiarr, Delta, L)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        Lag += sum(qp_energy[i] for i in range(self.Ne))

        # get embedding Hamiltonian expectations and normalization terms
        for I in range(N):
            Hemb = self._compute_Hemb(I, Lc)
            psi = psiarr[I]
            Lag += psi.T.conj() @ Hemb @ psi
            # calculate Ec contribution
            Lag += Ec[I] * (1 - (psi.T.conj() @ psi))

        #  calculate Lmix
        Lmix = -sum(np.trace((L[I] + Lc[I]) @ Delta[I].T) for I in range(N))
        Lag += Lmix + Lmix.conj()

        return Lag

    def _compute_gradient(self, x):
        N = self.N
        psiarr, L, Lc, Delta, Ec = self._unpack_vector(x)

        # get energy gradients
        grad_Ec = [1 - (psiarr[I].T.conj() @ psiarr[I]) for I in range(N)]

        # get quasiparticle and embedding Hamiltonians
        Hqp = self._compute_Hqp(psiarr, Delta, L)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        Hemb = [self._compute_Hemb(I, Lc) for I in range(N)]
        R = self._compute_renormalizations(psiarr, Delta)

        occ = [(i < self.Ne) for i in range(N*self.M)]
        full_correlation = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T

        # get <psi_K| gradients, each of size 4**M
        grad_psiarr = []
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
                DeltaIK = full_correlation[I*M:(I+1)*M, K*M:(K+1)*M]
                M1 = ttIK.T @ R[I] @ DeltaIK @ B.conj().T
                HDqp += sum((M1[alpha, gamma] * np.kron(C[alpha], np.eye(2**M)) @ np.kron(np.eye(2**M), C[gamma])) for alpha in range(M) for gamma in range(M))
            HD += HDqp + HDqp.conj().T

            grad_psiarr.append(HD @ psiarr[K])

        # get lambda gradients
        grad_L = []
        for I in range(N):
            M = self.M
            Lm = np.zeros((M,M), dtype=np.complex128)
            for a in range(M):
                for b in range(M):
                    DH = np.zeros(Hqp.shape)
                    DH[I*M + a, I*M + b] = 1
                    Lm[a, b] = self._compute_derivative_energy(qp_coeff, DH)
            grad_L.append(Lm - Delta[I])

        grad_Lc = []
        for I in range(N):
            M = self.M
            Lm = np.zeros((M,M), dtype=np.complex128)
            C = self._get_annahilation_operators(M)
            for a in range(M):
                for b in range(M):
                    Lm[a,b] = psiarr[I].conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psiarr[I]
            grad_Lc.append(Lm - Delta[I])

        # get Delta gradients
        grad_Delta = []
        for K in range(N):
            M = self.M

            # get Lmix contribution
            Dm = -L[K] - Lc[K]

            # calculate renormalization based matrix
            P = np.zeros((M,M), dtype=np.complex128)
            C = self._get_annahilation_operators(M)
            for alpha in range(M):
                for gamma in range(M):
                    P[alpha, gamma] = psiarr[K].conj().T @ np.kron(C[alpha].T, np.eye(2**M)) @ np.kron(np.eye(2**M), C[gamma].T) @ psiarr[K]
            
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
                    DH = np.zeros(Hqp.shape, dtype=np.complex128)
                    for I in range(N):
                        M1 = Y.T @ P.T @ self._get_tt(K, I) @ R[I].conj()
                        for a in range(M):
                            for b in range(M):
                                DH[K*M + a, I*M + b] += M1[a,b]
                    # calculate contribution to the full derivative
                    Dm[y, z] += self._compute_derivative_energy(qp_coeff, DH)

            grad_Delta.append(Dm)

        # return packed vector wrt x and y derivatives
        f = lambda grad: [2*G.conj() for G in grad]
        g = lambda grad: [2*G for G in grad]
        return self._pack_vector(g(grad_psiarr), f(grad_L), f(grad_Lc), f(grad_Delta), grad_Ec)
    
    def _get_Delta_psivec(self, Delta, p):
        M, _ = Delta.shape
        diag = np.zeros(2**M)
        for G in range(2**M):
            m = bin(G).count('1')
            diag[G] = np.sqrt((p**m) * ((1-p)**(M-m)))
        proj = np.diag(diag)
        psimat = proj @ scipy.linalg.sqrtm(self._compute_rdm(Delta))
        return self.get_psi_vector(psimat / np.linalg.norm(psimat))
    
    def _compute_initial_residual(self, D):
        M = int(np.sqrt(len(D)))
        Delta = D.reshape((M,M))
        psivec = self._get_Delta_psivec(Delta, self.filling)
        
        norm_residual = 1 - (np.linalg.norm(psivec)**2)
        num_residual = p*M - np.trace(Delta)

        corr_residual = np.zeros((M,M))
        C = self._get_annahilation_operators(M)
        for a in range(M):
            for b in range(M):
                corr_residual[a, b] += abs((psivec.conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psivec) - Delta[a, b])

        return 1*(np.concatenate([[norm_residual, num_residual], corr_residual.reshape(M*M)]))
        # return corr_residual.reshape(M*M)
    
    def _get_initial_guess(self):
        # initialize all psi to uniformly id on each block (unentangled)
        # L and Lc to 0 and Delta uniform identity on all sites
        # Ec to 0
        psiarr = []
        L = []
        Lc = []
        Delta = []
        N = self.N

        for I in range(N):
            M = self.M

            ZM = np.eye(M, dtype=np.complex128)
            L.append(ZM.copy())
            Lc.append(ZM.copy())
            D = self.filling*np.eye(M)

            # get projector and normalized phi matrix for the evenly filled state
            # res = scipy.optimize.least_squares(self._compute_initial_residual, list(D.reshape(M*M)), method="lm")
            # if res.success:
            #     D = np.array(res.x).reshape((M,M))
            Delta.append(D)
            # psivec = self._get_Delta_psivec(D, p)
            psivec = self.get_psi_vector(np.eye(2**M))
            psivec = psivec / np.linalg.norm(psivec)
            psiarr.append(psivec)

            # Dtest = np.zeros((M, M))
            # C = self._get_annahilation_operators(M)
            # for a in range(M):
            #     for b in range(M):
            #         Dtest[a, b] = psivec.conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psivec
            # breakpoint()

        Ec = np.zeros(N)
        return self._pack_vector(psiarr, L, Lc, Delta, Ec)

    def kernel(self, method="krylov", x0=None, maxiter=None, tolerance=1e-6, verbose=True):
        N = self.N

        if (x0 == None):
            x0 = self._get_initial_guess()

        options = {}
        if maxiter:
            options['maxiter'] = maxiter
        if verbose:
            options['disp'] = True

        options['fatol'] = tolerance
        result = scipy.optimize.root(self._compute_gradient, x0, method=method, options=options)
        
        psiarr, L, Lc, Delta, Ec = self._unpack_vector(result.x)
        E = self._compute_lagrangian(result.x)
        result.pop("x")
        projectors = []
        for I in range(N):
            rho = self._compute_rdm(Delta[I])
            projectors.append(self.get_psi_matrix(psiarr[I]) @ np.linalg.inv(scipy.linalg.sqrtm(rho)))

        Hqp = self._compute_Hqp(psiarr, Delta, L)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self.Ne) for i in range(N*self.M)]
        full_correlation = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T
        corr = np.empty((N, N), dtype=object)
        for I in range(N):
            for J in range(N):
                M = self.M
                corr[I, J] = full_correlation[I*M:(I+1)*M, J*M:(J+1)*M]
            
        return {
            "result":result,
            "E": E,
            "psiarr":[self.get_psi_matrix(psiarr[I]) for I in range(N)],
            "L":L,
            "Lc":Lc,
            "Delta":Delta,
            "Ec":Ec,
            "projectors":projectors,
            "corr":corr
        }
