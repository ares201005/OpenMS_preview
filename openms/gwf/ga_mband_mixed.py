import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

def get_psi_matrix(psi):
    psi_size = int(np.sqrt(psi.shape)[0])
    matrix = np.zeros((psi_size, psi_size), dtype=psi.dtype)
    for Gamma in range(psi_size):
        for n in range(psi_size):
            matrix[Gamma, n] = psi[Gamma*(psi_size) + n]
    return matrix

def get_psi_vector(psi):
    a, b = psi.shape
    vec = np.zeros(psi.size, dtype=psi.dtype)
    for Gamma in range(a):
        for n in range(b):
            vec[Gamma*a + n] = psi[Gamma, n]
    return vec

def _get_annahilation_operators(M):
    C = np.zeros((M, 2**M, 2**M))
    for i in range(M):
        for state in range(2**M):
            if (state & (1 << i)):
                C[i, state & ~(1 << i), state] = (-1)**(bin(state >> (i+1)).count('1'))
    return C

class FermionGASCF(ABC):
    def __init__(self, M, Ne):
        self.N = len(M)
        self.Ne = Ne
        self.M = M

        idx = 0
        Moff = []
        for I in range(self.N):
            Moff.append(idx)
            idx += M[I]
        Moff.append(idx)
        self._Moff = Moff

        self.filling = Ne/Moff[-1]

    @abstractmethod
    def get_ht(self, I):
        pass

    def _get_ht(self, I):
        M = self.M[I]
        res = self.get_ht(I)
        if (res.shape != (M, M)):
            raise ValueError(f"ht[{I}] has incorrect shape")
        return res
    
    @abstractmethod
    def get_U(self, I):
        pass

    def _get_U(self, I):
        M = self.M[I]
        res = self.get_U(I)
        if (res.shape != (M, M, M, M)):
            raise ValueError(f"U[{I}] has incorrect shape")
        return res
    
    @abstractmethod
    def get_tt(self, I, J):
        pass

    def _get_tt(self, I, J):
        res = self.get_tt(I, J)
        if (res.shape != (self.M[I], self.M[J])):
            raise ValueError(f"tt[{I}, {J}] has incorrect shape")
        return res

    def get_block(self, Mat, I, J):
        return Mat[self._Moff[I]:self._Moff[I+1], self._Moff[J]:self._Moff[J+1]]           
    
    # psiarr is a vector of size N, containing embedding matrix states each of size 2**M x 2**M encoded as a 4**M component vector
    # L is a vector of size N, containing matrices of size MxM
    # Lc is a vector of size N, containing matrices of size MxM
    # n is a vector of size N, containing vectors of size M
    # Ec is a vector of size N containing real scalar energies
    def _pack_vector(self, psiarr, L, Lc, n, Ec):
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
            parts.append(np.asarray(n[i]).ravel())
        parts.append(np.asarray(Ec).ravel())

        assert all(psiarr[i].size == 4**M[i] for i in range(N))
        assert all(L[i].size == M[i]**2 for i in range(N))
        assert all(Lc[i].size == M[i]**2 for i in range(N))
        assert all(len(n[i]) == M[i] for i in range(N))
        assert np.asarray(Ec).ravel().size == N

        result = np.concatenate(parts)
        assert all(result.imag == 0)
        return result.real
    
    def _unpack_vector(self, x):
        idx = 0
        N = self.N

        psiarr = []
        for I in range(N):
            size_psi = 4**self.M[I]
            Re = x[idx:(idx + size_psi)].reshape(size_psi)
            idx += size_psi
            Im = x[idx:(idx + size_psi)].reshape(size_psi)
            idx += size_psi
            psiarr.append(Re + 1j*Im)

        L = []
        for I in range(N):
            M = self.M[I]
            size_sq = M*M
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            L.append(Re + 1j*Im)
        Lc = []
        for I in range(N):
            M = self.M[I]
            size_sq = M*M
            Re = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Im = x[idx:(idx + size_sq)].reshape(M, M)
            idx += size_sq
            Lc.append(Re + 1j*Im)

        n = []
        for I in range(N):
            M = self.M[I]
            n.append(x[idx:(idx + M)])
            idx += M

        Ec = x[idx:(idx+N)]
        idx += N

        assert idx == len(x), "Unpack error: leftover elements in input array"
        return psiarr, L, Lc, n, Ec  
    
    def _get_tarr(self):
        N = self.N
        tarr = np.zeros((N, N), dtype=object)
        for I in range(N):
            for J in range(N):
                tarr[I, J] = self._get_tt(I, J)
        return tarr

    def _compute_renormalizations(self, psiarr, Delta):
        R = []
        for I in range(self.N):
            # get local state and correlation
            M = self.M[I]
            C = _get_annahilation_operators(M)
            B = scipy.linalg.sqrtm(np.linalg.inv(Delta[I] @ (np.eye(M) - Delta[I])))
            psi = psiarr[I]

            # compute R
            opmat = np.zeros((M, M), dtype=np.complex128)
            for alpha in range(M):
                for b in range(M):
                    opmat[alpha, b] = psi.conj().T @ (np.kron(C[alpha].T, C[b].T)) @ psi
            R.append(opmat @ B)

        return R
    
    def _compute_Hqp(self, psiarr, Delta, L, tarr=None, R=None):
        N = self.N
        if (tarr is None):
            tarr = self._get_tarr()
        if (R is None):
            R = self._compute_renormalizations(psiarr, Delta)
        t = np.block(tarr.tolist())
        Rblock = scipy.linalg.block_diag(*R)
        
        Hqp = Rblock.T @ t @ Rblock.conj()

        LH = []
        for I in range(N):
            LH.append(L[I] + L[I].conj().T)
        Hqp += scipy.linalg.block_diag(*LH)

        return Hqp
    
    def _compute_Hemb(self, I, Lc):
        M = self.M[I]
        # get coefficients for embedding Hamiltonian
        h = self._get_ht(I)
        U = self._get_U(I)
        C = _get_annahilation_operators(M)
        # construct local Hamiltonian
        Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
        Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
        # construct lambda part of the Hamiltonian
        HL = np.kron(np.eye(2**M), sum(Lc[I][a, b] * C[b].T @ C[a] for a in range(M) for b in range(M)))
        # construct and solve embedding Hamiltonian
        Hemb = np.kron(Hloc, np.eye(2**M)) + HL + HL.T.conj()
        return Hemb
    
    def _compute_derivative_energy(self, eigmatrix, DH):
        return sum((eigmatrix[:, i].T.conj() @ DH @ eigmatrix[:, i]) for i in range(self.Ne))
    
    def _compute_rdm(self, Delta):
        M, _ = Delta.shape
        K = scipy.linalg.logm((np.eye(M) - Delta) @ np.linalg.inv(Delta))
        C = _get_annahilation_operators(M)
        rho = scipy.linalg.expm(sum(-K[a, b] * C[a].T @ C[b] for a in range(M) for b in range(M)))
        return (1/np.trace(rho)) * rho

    def _compute_lagrangian(self, x):
        N = self.N
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)
        Delta = []
        for K in range(N):
            Delta.append(np.diag(n[K]))
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
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)
        Delta = []
        for K in range(N):
            Delta.append(np.diag(n[K]))
        tarr = self._get_tarr()

        # get energy gradients
        grad_Ec = [1 - (psiarr[I].T.conj() @ psiarr[I]) for I in range(N)]

        # get quasiparticle and embedding Hamiltonians
        R = self._compute_renormalizations(psiarr, Delta)
        Hqp = self._compute_Hqp(psiarr, Delta, L, tarr=tarr, R=R)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        Hemb = [self._compute_Hemb(I, Lc) for I in range(N)]

        # compute single particle correlations
        occ = [(i < self.Ne) for i in range(self._Moff[-1])]
        corr = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T

        # get <psi_K| gradients, each of size 4**M
        grad_psiarr = []
        for K in range(N):
            M = self.M[K]
            C = _get_annahilation_operators(M)
            nK = n[K]
            B = np.diag(1/np.sqrt(nK*(1- nK)))
            
            # construct derivative Hamiltonian
            HD = Hemb[K] - Ec[K]*np.eye(4**M)
            HDqp = np.zeros((4**M, 4**M), dtype=np.complex128)
            for I in range(N):
                M1 = tarr[I, K].T @ R[I] @ self.get_block(corr, I, K) @ B
                HDqp += sum((M1[alpha, gamma] * np.kron(C[alpha], C[gamma])) for alpha in range(M) for gamma in range(M))
            HD += HDqp + HDqp.conj().T

            grad_psiarr.append(HD @ psiarr[K])

        # get lambda gradients
        grad_L = []
        for I in range(N):
            M = self.M[I]
            Lm = np.zeros((M,M), dtype=np.complex128)
            for a in range(M):
                for b in range(M):
                    DH = np.zeros(Hqp.shape)
                    DH[self._Moff[I] + a, self._Moff[I] + b] = 1
                    Lm[a, b] = self._compute_derivative_energy(qp_coeff, DH)
            grad_L.append(Lm - Delta[I])

        grad_Lc = []
        for I in range(N):
            M = self.M[I]
            Lm = np.zeros((M,M), dtype=np.complex128)
            C = _get_annahilation_operators(M)
            for a in range(M):
                for b in range(M):
                    Lm[a,b] = psiarr[I].conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psiarr[I]
            grad_Lc.append(Lm - Delta[I])

        # get n gradients
        grad_n = []
        for K in range(N):
            nK = n[K]
            r = 0.5 * R[K] @ np.diag(1/(1-nK) - 1/nK)
            M1 = sum((r.conj().T @ tarr[I, K].T @ R[I] @ self.get_block(corr, I, K)) for I in range(N))
            M1 -= L[K] + Lc[K]
            grad_n.append(2*np.diag(M1).real)

        # return packed vector wrt x and y derivatives
        f = lambda grad: [2*G.conj() for G in grad]
        g = lambda grad: [2*G for G in grad]
        return self._pack_vector(g(grad_psiarr), f(grad_L), f(grad_Lc), grad_n, grad_Ec)

    def _get_initial_guess(self):
        # initialize all psi to uniformly id on each block (unentangled)
        # L and Lc to 0 and Delta uniform identity on all sites
        # Ec to 0
        psiarr = []
        L = []
        Lc = []
        narr = []
        N = self.N

        for I in range(N):
            M = self.M[I]

            ZM = np.eye(M, dtype=np.complex128)
            L.append(ZM.copy())
            Lc.append(ZM.copy())

            # get projector and normalized phi matrix for the evenly filled state   
            narr.append(M*[self.filling])
            psivec = get_psi_vector(np.eye(2**M))
            psivec = psivec / np.linalg.norm(psivec)
            psiarr.append(psivec)
        
        Ec = np.zeros(N)
        return self._pack_vector(psiarr, L, Lc, narr, Ec)

    def kernel(self, method="krylov", maxiter=None, x0=None, tolerance=1e-6, verbose=True):
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

        psiarr, L, Lc, n, Ec = self._unpack_vector(result.x)
        Delta = [np.diag(n[K]) for K in range(N)]
        projectors = []
        for I in range(N):
            projectors.append(get_psi_matrix(psiarr[I]) @ scipy.linalg.sqrtm(self._compute_rdm(Delta[I])))

        Hqp = self._compute_Hqp(psiarr, Delta, L)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self.Ne) for i in range(self._Moff[-1])]
        corr = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T
        
        E = self._compute_lagrangian(result.x)
        result.pop("x")
        return {
            "result":result,
            "E": E,
            "psiarr":[get_psi_matrix(psiarr[I]) for I in range(N)],
            "L":L,
            "Lc":Lc,
            "Delta":Delta,
            "Ec":Ec,
            "projectors":projectors,
            "corr":corr
        }