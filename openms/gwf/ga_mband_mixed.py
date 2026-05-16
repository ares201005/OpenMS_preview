import numpy as np
import scipy.linalg
import scipy.optimize
from abc import ABC, abstractmethod
from pyscf import scf, gto
import time

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

def get_psi_matrix(psi):
    r"""
    Convert state vector :math:`\ket{\Psi_I}` to matrix form :math:`\phi_I`.
    """
    psi_size = int(np.sqrt(psi.shape)[0])
    matrix = np.zeros((psi_size, psi_size), dtype=psi.dtype)
    for Gamma in range(psi_size):
        for n in range(psi_size):
            matrix[Gamma, n] = psi[Gamma*(psi_size) + n]
    return matrix

def get_psi_vector(psi):
    r"""
    Convert matrix :math:`\phi_I` to state vector :math:`\ket{\Psi_I}`.
    """
    a, b = psi.shape
    vec = np.zeros(psi.size, dtype=psi.dtype)
    for Gamma in range(a):
        for n in range(b):
            vec[Gamma*a + n] = psi[Gamma, n]
    return vec

def _get_matrix_ip(psi, A):
    M, _ = A.shape
    psic = psi.conj().T
    return sum(np.dot(psic[a,:], A[:,a]) for a in range(M))

def _get_annahilation_operators(M):
    C = np.zeros((M, 2**M, 2**M))
    for i in range(M):
        for state in range(2**M):
            if (state & (1 << i)):
                C[i, state & ~(1 << i), state] = (-1)**(bin(state >> (i+1)).count('1'))
    return C

def _compute_rdm(Delta):
    r"""
    Given local correlation matrix :math:`\Delta` in the single-particle space, return

    .. math::

            \rho^0_I = \frac{1}{Z}\exp\left(
            -\sum_{\alpha\beta}\left[ \ln\left(\frac{\mathbb{I}-\Delta}{\Delta}\right)_{\alpha\beta} c^\dagger_{I\alpha} c_{I\beta}\right]
        \right)

    in the full Fock space.
    """
    M, _ = Delta.shape
    K = scipy.linalg.logm((np.eye(M) - Delta) @ np.linalg.inv(Delta))
    C = _get_annahilation_operators(M)
    rho = scipy.linalg.expm(sum(-K[a, b] * C[a].T @ C[b] for a in range(M) for b in range(M)))
    return (1/np.trace(rho)) * rho

def _get_block(A, Moff, I, J):
    return A[Moff[I]:Moff[I+1], Moff[J]:Moff[J+1]]

class FermionGASCF(ABC):
    r"""
    The electronic Hamiltonian being minimized under :math:`\ket{\Psi_G}` can be written as follows:

    .. math::
        \hat{H}_e &= \sum_I \left[ \sum_{\alpha\beta} \tilde{h}^{I}_{\alpha\beta}c^\dagger_{I\alpha}c_{I\beta} + \sum_{\alpha\beta\gamma\delta} U^I_{\alpha\beta\gamma\delta} c^\dagger_{I\alpha} c_{I\beta}^\dagger c_{I\gamma} c_{I\delta} \right] \\ 
        &+ \sum_{I \neq J} \sum_{\alpha\beta} \left[ \tilde{t}^{IJ}_{\alpha\beta}c^\dagger_{I\alpha}c_{J\beta} \right]

    :param M: array describing the number of states at each site
    :param Ne: number of electrons present in the system
    """
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

        self._Cdict = {}
        for M in set(self.M):
            self._Cdict[M] = _get_annahilation_operators(M)

        self._put_harr()
        self._put_tblock()
        self._put_Uarr()

        self._grad_time = []
        self._ren_time = []
        self._qp_time = []
        self._L_time = []
        self._Lc_time = []
        self._psi_time = []
        self._n_time = []

    @abstractmethod
    def get_ht(self, I):
        r"""
        Return the matrix :math:`\tilde{h}^I_{ab}` for site :math:`I`.
        Must be implemented by the  concrete class.
        """
        pass

    def _get_ht(self, I):
        M = self.M[I]
        res = self.get_ht(I)
        if (res.shape != (M, M)):
            raise ValueError(f"ht[{I}] has incorrect shape")
        self._harr[I] = res
        return res
    
    def _get_harr(self):
        if (self._harr == [None]*self.N):
            for I in range(self.N):
                self._harr[I] = self._get_ht(I)
    
    def _put_harr(self):
        self._harr = [None] * self.N
    
    @abstractmethod
    def get_U(self, I):
        r"""
        Return the tensor :math:`U^I_{abcd}` for site :math:`I`.
        Must be implemented by the  concrete class.
        """
        pass

    def _get_U(self, I):
        M = self.M[I]
        res = self.get_U(I)
        if (res.shape != (M, M, M, M)):
            raise ValueError(f"U[{I}] has incorrect shape")
        return res
    
    def _get_Uarr(self):
        if (self._Uarr == [None]*self.N):
            for I in range(self.N):
                self._Uarr[I] = self._get_U(I)

    def _put_Uarr(self):
        self._Uarr = [None]*self.N
    
    @abstractmethod
    def get_tt(self, I, J):
        r"""
        Return the matrix :math:`\tilde{t}^{IJ}_{ab}` between sites :math:`I \neq J`.
        Must be implemented by the  concrete class.
        The concrete class is expected to return  :math:`t^{II} = 0` for all sites :math:`I`.
        """
        pass

    def _get_tt(self, I, J):
        shape = (self.M[I], self.M[J])
        if (I == J):
            return np.zeros(shape)
        res = self.get_tt(I, J)
        if (res.shape != shape):
            raise ValueError(f"tt[{I}, {J}] has incorrect shape")
        return res
    
    def _get_tblock(self):
        if (self._tblock is None):
            N = self.N
            tarr = np.zeros((N, N), dtype=object)
            for I in range(N):
                for J in range(N):
                    tarr[I, J] = self._get_tt(I, J)
            self._tblock = np.block(tarr.tolist())
    
    def _put_tblock(self):
        self._tblock = None

    def _get_block(self, Mat, I, J):
        return _get_block(Mat, self._Moff, I, J)           
    
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
            size_psi = 2**self.M[I]
            Re = x[idx:(idx + size_psi**2)].reshape(size_psi, size_psi)
            idx += size_psi**2
            Im = x[idx:(idx + size_psi**2)].reshape(size_psi, size_psi)
            idx += size_psi**2
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

    def _get_n_slice(self):
        idx = 0
        for M in self.M:
            idx += 2 * (4**M)
        for M in self.M:
            idx += 2 * (M*M)
        for M in self.M:
            idx += 2 * (M*M)
        n_start = idx
        for M in self.M:
            idx += M
        return slice(n_start, idx)

    def _get_qp_occupations(self, qp_energy, degeneracy_tol=1e-10):
        ret = np.zeros(self._Moff[-1])
        ret[:self.Ne] = [1]*self.Ne
        return ret
        # occ = np.zeros(len(qp_energy), dtype=float)
        # if self.Ne <= 0:
        #     return occ
        # if self.Ne >= len(qp_energy):
        #     occ[:] = 1.0
        #     return occ

        # fermi = qp_energy[self.Ne - 1]
        # tol = degeneracy_tol * max(1.0, abs(fermi))
        # below = qp_energy < fermi - tol
        # shell = np.abs(qp_energy - fermi) <= tol

        # occ[below] = 1.0
        # remaining = self.Ne - np.count_nonzero(below)
        # if remaining > 0:
        #     occ[shell] = remaining / np.count_nonzero(shell)
        # return occ

    def _compute_qp_density(self, qp_coeff, qp_energy):
        occ = [(i < self.Ne) for i in range(self._Moff[-1])]
        return qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T

    def _compute_renormalizations(self, psiarr, n):
        r"""
        Compute all 1-body renormalization factors, given by

        .. math::

             \mathcal{R}^I_{\alpha a} = \dfrac{\text{Tr} \left[ \phi_I^\dagger c_\alpha^\dagger \phi_I c_a \right]}{\sqrt{n_a^I(1-n_a^I)}} 

        """
        t = time.time()
        R = []
        for I in range(self.N):
            # get local state and correlation
            M = self.M[I]
            C = _get_annahilation_operators(M)
            nI = n[I]
            psi = psiarr[I]

            # compute R
            opmat = np.zeros((M, M), dtype=np.complex128)
            for alpha in range(M):
                for a in range(M):
                    opmat[alpha, a] = _get_matrix_ip(psi, C[alpha].T @ psi @ C[a])
            R.append(opmat * (1/np.sqrt(nI*(1-nI))))
        
        self._ren_time[-1] += time.time() - t
        return R

    def _compute_density_renormalizations(self, psiarr, n):
        r"""
        Compute all 2-body (number-preserving) renormalization factors, given by

        .. math::

            \mathcal{T}^I_{\alpha\beta,ab} &= \dfrac{\text{Tr} \left[ \hi_I^\dagger c_{\alpha}^\dagger c_\beta \phi_I (c_a^\dagger c_b - \delta_{ab}n^I_a \mathbb{I}) \right]}{\sqrt{n^I_an^I_b(1-n^I_a)(1-n^I_b)}} \\
            \mathcal{T}^I_{\alpha\beta} &= \text{Tr} \left[ c_\alpha^\dagger c_\beta \right] - \sum_{a} \mathcal{T}^I_{\alpha\beta, aa}n^I_{a}

        """
        Xdict = {}
        for M in set(self.M):
            C = _get_annahilation_operators(M)
            X = np.zeros((M,M,2**M, 2**M))
            for a in range(M):
                for b in range(M):
                    X[a, b] = C[a].T @ C[b]
            Xdict[M] = X

        T = []
        Tsrc = []
        for I in range(self.N):
            psi = psiarr[I]
            M = self.M[I]
            X =  Xdict[M] 
            G = np.zeros((M,M,M,M), dtype=np.complex128)
            P = np.zeros((M,M), dtype=np.complex128)
            for alpha in range(M):
                for beta in range(M):
                    M1 = X[alpha, beta]
                    P[alpha, beta] = _get_matrix_ip(psi, M1 @ psi)
                    for a in range(M):
                        for b in range(M):
                            G[alpha, beta, a, b] = _get_matrix_ip(psi, M1 @ psi @ X[a, b])

            nI = n[I]
            b = 1/np.sqrt(nI*(1-nI))
            bb = np.outer(b, b)
            T4 = (G - np.einsum('ab,ij->abij', P, np.diag(nI))) * bb[None, None, :, :]
            T.append(T4)
            Tsrc.append(P - np.einsum('abii,i->ab', T4, nI))

        return T, Tsrc

    def _get_A(self, n, R, corr):
        arr = []
        N = self.N
        for K in range(N):
            nK = n[K]
            r = 0.5 * R[K] * (1/(1-nK) - 1/nK)
            arr.append(sum((r.conj().T @ self._get_block(self._tblock, I, K).T @ R[I] @ self._get_block(corr, I, K)) for I in range(N)))
        return arr

    def _compute_Hqp(self, L, R):
        r"""
        Compute

        .. math::

            \hat{H}_{qp} = & \sum_{IJ ab} \left[ \sum_{\alpha\beta}\tilde{t}^{IJ}_{\alpha\beta} \mathcal{R}^{I}_{\alpha a} \mathcal{R}^{J*}_{\beta b} \right] f^\dagger_{Ia}f_{Jb}  \\
            & + \sum_{I ab} \left[ \lambda^I_{ab}f_{Ia}^\dagger f_{Ib} + h.c.  \right]

        in the single-particle basis.
        """
        N = self.N
        Rblock = scipy.linalg.block_diag(*R)
        Hqp = Rblock.T @ self._tblock @ Rblock.conj()

        LH = []
        for I in range(N):
            LH.append(L[I] + L[I].conj().T)
        Hqp += scipy.linalg.block_diag(*LH)

        return Hqp

    def _compute_Hemb(self, I, Lc):
        r"""
        Compute

        .. math::

            \hat{H}_{emb}^I &= \displaystyle\sum_{\alpha\beta} \tilde{h}^{I}_{\alpha\beta}c^\dagger_{\beta}c_{\alpha} + \sum_{\alpha\beta\gamma\delta}U^{I}_{\alpha\beta\gamma\delta}c^\dagger_\delta c_\gamma^\dagger c_\beta c_\alpha \\
            &+ \displaystyle\sum_{ab} \left[ (\lambda_c^I)_{ab}f_a^\dagger f_b + h.c. \right]

        in the mixed tensor-product basis.
        """
        M = self.M[I]
        # get coefficients for embedding Hamiltonian
        h = self._harr[I]
        U = self._Uarr[I]
        C = self._Cdict[M]
        # construct local Hamiltonian
        Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
        Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
        # construct lambda part of the Hamiltonian
        HL = np.kron(np.eye(2**M), sum(Lc[I][a, b] * C[a].T @ C[b] for a in range(M) for b in range(M)))
        # construct and solve embedding Hamiltonian
        Hemb = np.kron(Hloc.T, np.eye(2**M)) + HL + HL.T.conj()
        return Hemb
    
    def _get_HK(self, K, n, Lc, R, corr):
        N = self.N
        M = self.M[K]
        C = self._Cdict[M]
        nK = n[K]
        Bv = 1/np.sqrt(nK*(1- nK))
        
        # construct derivative Hamiltonian
        HD = self._compute_Hemb(K, Lc)
        HDqp = np.zeros((4**M, 4**M), dtype=np.complex128)
        for I in range(N):
            M1 = self._get_block(self._tblock, I, K).T @ R[I] @ self._get_block(corr, I, K) * Bv
            HDqp += sum((M1[alpha, gamma] * np.kron(C[alpha].T, C[gamma].T)) for alpha in range(M) for gamma in range(M))
        HD += HDqp + HDqp.conj().T
        return HD
    
    def _compute_Hemb_psi(self, I, Lc, psiarr):
        M = self.M[I]
        # get coefficients for embedding Hamiltonian
        h = self._harr[I]
        U = self._Uarr[I]
        C = self._Cdict[M]
        # construct local Hamiltonian
        Hloc = sum((h[a, b] * C[a].T @ C[b]) for a in range(M) for b in range(M))
        Hloc += sum(U[a, b, c, d] * C[a].T @ C[b].T @ C[c] @ C[d] for a in range(M) for b in range(M) for c in range(M) for d in range(M))
        # construct lambda part of the Hamiltonian
        HL = sum(Lc[I][a, b] * C[a].T @ C[b] for a in range(M) for b in range(M))
        HL += HL.conj().T
        # return matrix vector product
        P = psiarr[I]
        return (Hloc @ P) + (P @ HL)
    
    def _compute_HKD_psi(self, K, n, R, corr, psiarr):
        M = self.M[K]
        C = self._Cdict[M]
        nK = n[K]
        Bv = 1/np.sqrt(nK*(1- nK))

        # construct derivative Hamiltonian vector product
        psi = psiarr[K]
        HD = np.zeros_like(psi)
        M1 = sum(self._get_block(self._tblock, I, K).T @ R[I] @ self._get_block(corr, I, K) for I in range(self.N)) * Bv
        for alpha in range(M):
            for a in range(M):
                HD += M1[alpha, a] * (C[alpha] @ psi @ C[a].T)
                HD += M1[alpha, a].conj() * (C[alpha].T @ psi @ C[a])
        return HD
    
    def _compute_derivative_energy(self, eigmatrix, DH, occ):
        return sum(occ[i] * (eigmatrix[:, i].T.conj() @ DH @ eigmatrix[:, i]) for i in np.flatnonzero(occ))

    def _compute_lagrangian(self, x):
        r"""
        Compute

        .. math::

            &\mathcal{L}_e \left(\left\{ \ket{\Psi_I}, n^I, \lambda^I, \lambda_c^I, E_c^I \right\}\right) \\
            &= \bra{\Psi_0^e} \hat{H}_{qp} \left( \left\{ \ket{\Psi_I}, n^I, \lambda^I \right\} \right) \ket{\Psi_0^e} \\
            &+ \sum_I \left[\bra{\Psi_I} \hat{H}_{emb}^I \left(  \lambda^I_c \right) \ket{\Psi_I} + E^I_c \left( 1 - \braket{\Psi_I}\right)\right] \\
            &+ \sum_I \left[ \mathcal{L}^I_{mix} \left( \left\{ \lambda^I, \lambda_c^I, n^I \right\} \right) + c.c. \right]

        where

        .. math::

                \mathcal{L}_{mix}^I = -\displaystyle\sum_{aa} \left(\lambda^I + \lambda_c^I \right)_{aa}n^I_a

        Here, :math:`\ket{\Psi_0^e}` is a Slater determinant of the first ``self.Ne`` single-particle eigenstates of :math:`\hat{H}_{qp}` and is not an independent variable.
        """
        N = self.N
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)
        Lag = 0

        # calculate Hqp and energies of filled eigenstates
        R = self._compute_renormalizations(psiarr, n)
        Hqp = self._compute_Hqp(L, R)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        qp_occ = self._get_qp_occupations(qp_energy)
        Lag += np.dot(qp_occ, qp_energy)

        # get embedding Hamiltonian expectations and normalization terms
        for I in range(N):
            Hemb_psi = self._compute_Hemb_psi(I, Lc, psiarr)
            psi = psiarr[I]
            Lag += np.trace(psi.T.conj() @ Hemb_psi)
            # calculate Ec contribution
            Lag += Ec[I] * (1 - np.trace(psi.T.conj() @ psi))

        #  calculate Lmix
        Lmix = -sum(np.trace((L[I] + Lc[I]) @ np.diag(n[I])) for I in range(N))
        Lag += Lmix + Lmix.conj()
        
        return Lag.real

    def _compute_gradient(self, x):
        r"""
        Compute the first derivatives

        .. math::

            \dfrac{\partial \mathcal{L}_e}{\partial E_c^K} &= 1 - \braket{\Psi_K} \\
            \dfrac{\partial \mathcal{L}_e}{\partial \lambda^K_{ab}} &= \bra{\Psi_0^e} c_{Ka}^\dagger c_{Kb} \ket{\Psi_0^e} - \Delta^K_{ab} \\
            \dfrac{\partial \mathcal{L}_e}{\partial (\lambda^K_c)_{ab}} &= \bra{\Psi_K} f_a^\dagger f_b \ket{\Psi_K} - \Delta^K_{ab} \\
            \dfrac{\partial{\mathcal{L}_e}}{\partial n^K_z} &= 2 \text{Re} \mathcal{A}^K_{zz} \\
            \dfrac{\partial \mathcal{L}_e}{\partial \bra{\Psi_K}} &= \hat{H}^K \ket{\Psi_K}

        where

        .. math::

            \mathcal{A}^K_{yz} &= -(\lambda^K + \lambda^K_c)_{yz} \\
            &+ \sum_I \left[ r^{K\dagger} \tilde{t}^{IK}\mathcal{R}^I\Delta^{IK} \right]_{yz} \\
            r^K_{\alpha a} \equiv \dfrac{\partial \mathcal{R}^K_{\alpha a}}{\partial n^K_a} &= \dfrac{1}{2} \left[ \dfrac{1}{1- n^K_a} - \dfrac{1}{n^K_a} \right] \mathcal{R}^K_{\alpha a}

        and the Hermitian operator

        .. math::

            \hat{H}^K &= \hat{H}^K_{emb} - E_c^K \mathbb{I} \\
            &+ \sum_{\alpha\gamma} \left[ \left[\sum_I \tilde{t}^{{IK}^T} \mathcal{R}^I \Delta^{IK}  {B^K} \right]_{\alpha\gamma} c_\alpha^\dagger f_\gamma^\dagger \right] + h.c. \\

       Here, :math:`B^K` is a diagonal matrix with

        .. math::

            (B^K)_{ab} = \dfrac{\delta_{ab}}{\sqrt{n^K_a(1-n^K_a)}}

        """
        self._grad_time.append(0)
        self._ren_time.append(0)
        self._qp_time.append(0)
        self._L_time.append(0)
        self._Lc_time.append(0)
        self._psi_time.append(0)
        self._n_time.append(0)
        t = time.time()
        N = self.N
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)

        # get energy gradients
        grad_Ec = [1 - np.trace(psiarr[I].T.conj() @ psiarr[I]) for I in range(N)]

        # get quasiparticle and embedding Hamiltonians
        R = self._compute_renormalizations(psiarr, n)
        tqp = time.time()
        Hqp = self._compute_Hqp(L, R)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        qp_occ = self._get_qp_occupations(qp_energy)
        # compute single particle correlations
        corr = self._compute_qp_density(qp_coeff, qp_energy)
        self._qp_time[-1] += time.time() - tqp

        # get <psi_K| gradients, each of size 4**M
        tpsi = time.time()
        grad_psiarr = []
        for K in range(N):
            HD = self._compute_Hemb_psi(K, Lc, psiarr) - Ec[K]*psiarr[K]
            HD += self._compute_HKD_psi(K, n, R, corr, psiarr)
            grad_psiarr.append(HD)
        self._psi_time[-1] += time.time() - tpsi

        # get lambda_c gradients
        tLc = time.time()
        grad_Lc = []
        for I in range(N):
            M = self.M[I]
            Lm = np.zeros((M,M), dtype=np.complex128)
            C = self._Cdict[M]
            psi = psiarr[I]
            A = psi.conj().T @ psi
            for a in range(M):
                for b in range(M):
                    Lm[a,b] = np.sum(C[a] * (C[b] @ A))
            idx = np.arange(M)
            Lm[idx, idx] -= n[I]
            grad_Lc.append(Lm)
        self._Lc_time[-1] += time.time() - tLc

        # get n gradients
        tn = time.time()
        grad_n = []
        Aarr = self._get_A(n, R, corr)
        for K in range(N):
            M1 = Aarr[K]
            M1 -= L[K] + Lc[K]
            grad_n.append(2*np.diag(M1).real)
        self._n_time[-1] += time.time() - tn

        # get lambda gradients
        tL = time.time()
        idx = np.arange(self._Moff[-1])
        Lm = corr
        Lm[idx, idx] -= np.concatenate(n)
        grad_L = [self._get_block(Lm, I, I) for I in range(N)]
        self._L_time[-1] += time.time() - tL

        # return packed vector wrt x and y derivatives
        f = lambda grad: [2*G.conj() for G in grad]
        g = lambda grad: [2*G for G in grad]
        self._grad_time[-1] += time.time() - t
        return self._pack_vector(g(grad_psiarr), f(grad_L), f(grad_Lc), grad_n, grad_Ec)

    def _get_static_initial_guess(self):
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
            psi = np.eye(2**M)
            psi = psi / np.linalg.norm(psi)
            psiarr.append(psi)

        Ec = np.zeros(N)
        return self._pack_vector(psiarr, L, Lc, narr, Ec)

    class _GHF(scf.ghf.GHF):
        def __init__(self, gascf, verbose=True):
            self.gascf = gascf

            vint = 3 if verbose else 0
            mol = gto.M(verbose=vint)
            mol.nelectron = gascf.Ne
            mol.incore_anyway = False
            mol.nao = 0
            super().__init__(mol)

            self.conv_tol = 1e-10
            self.max_cycle = 5000
            self._init_guess = '1e'
            self.direct_scf = False

            self._h1e = scipy.linalg.block_diag(*gascf._harr) + gascf._tblock

        def get_hcore(self, mol=None):
            return self._h1e

        def get_ovlp(self, mol=None):
            return np.eye(self.gascf._Moff[-1])

        def energy_nuc(self):
            return 0.0

        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            gascf = self.gascf
            if (dm is None):
                dm = self.make_rdm1()

            vHF = np.zeros_like(dm)
            for I in range(self.gascf.N):
                sl = slice(gascf._Moff[I], gascf._Moff[I+1])
                CI = dm[sl, sl]
                U = gascf._Uarr[I]
                vHF[sl, sl] += (
                    np.einsum('iabj,ab->ij', U, CI)
                  + np.einsum('aijb,ab->ij', U, CI)
                  - np.einsum('iajb,ab->ij', U, CI)
                  - np.einsum('aibj,ab->ij', U, CI)
                )

            return vHF

    def _get_initial_guess(self, verbose=True, hf=True):
        self._ren_time = [0]
        # get GHF 1-body correlations in the natural basis
        mf = self._GHF(self, verbose=verbose)
        mf.kernel()
        pcorr = self._compute_qp_density(mf.mo_coeff, mf.mo_energy)
        Uarr = []
        narr = []
        N = self.N
        for I in range(N):
            n, U = np.linalg.eigh(self._get_block(pcorr, I, I))
            Uarr.append(U)
            narr.append(n)
        U = scipy.linalg.block_diag(*Uarr)
        corr = U.conj().T @ pcorr @ U

        psiarr = []
        for I in range(N):
            M = self.M[I]
            mat = np.zeros((2**M, 2**M), dtype=np.complex128)
            n = narr[I]
            for state in range(2**M):
                p = 1
                for i in range(M):
                    if ((state & (1 << i)) == 0):
                        p *= 1 - n[i]
                    else:
                        p *= n[i]
                mat[state, state] = np.sqrt(p)
            psiarr.append(get_psi_vector(mat))

        L = [None] * N
        Lc = [None] * N
        Ec = np.zeros(N)
        fock = mf.get_hcore() + mf.get_veff()
        fock = U.conj().T @ fock @ U
        L = [0.5*self._get_block(fock, I, I) for I in range(N)]
        for _ in range(50):
            prev_psiarr = [psi.copy() for psi in psiarr]
            psiarr = [get_psi_matrix(psi) for psi in psiarr]
            R = self._compute_renormalizations(psiarr, narr)
            Aarr = self._get_A(narr, R, corr)
            for I in range(N):
                Lc[I] = Aarr[I] - L[I]
            for I in range(N):
                HK = self._get_HK(I, narr, Lc, R, corr)
                eigval, eigvec = np.linalg.eigh(HK)
                psiarr[I] = eigvec[:, 0]
                Ec[I] = eigval[0]
            if np.linalg.norm(np.concatenate(psiarr) - np.concatenate(prev_psiarr)) < 1e-10:
                break

        # ------------------------------------------

        # fock = mf.get_hcore() #+ mf.get_veff()
        # best_norm = np.inf
        # for j in range(50):
        #     print(f"Iteration {j}")
        #     R = self._compute_renormalizations(psiarr, narr)
        #     Aarr = self._get_A(narr, R, corr)

        #     Uarr = [None] * N
        #     for I in range(N):
        #         LM = self._get_block(fock, I, I)
        #         LC = Aarr[I] - LM
        #         LC = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        #         L[I] = LM
        #         Lc[I] = LC
        #         HK = self._get_HK(I, narr, Lc, R, corr)
        #         eigval, eigvec = np.linalg.eigh(HK)
        #         psiarr[I] = eigvec[:, 0]
        #         Ec[I] = eigval[0]
        #         Lc[I] = Aarr[I] - LM

        #         M = self.M[I]
        #         C = self._Cdict[M]
        #         Delta = np.zeros((M,M), dtype=np.complex128)
        #         psi = psiarr[I]
        #         for a in range(M):
        #             for b in range(M):
        #                 Delta[a, b] = psi.conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psi
        #         nI, U = np.linalg.eigh(Delta)

        #         L[I] = U.conj().T @ L[I] @ U
        #         Lc[I] = U.conj().T @ Lc[I] @ U
        #         narr[I] = nI
        #         Uarr[I] = U
        #     U = scipy.linalg.block_diag(*Uarr)
        #     corr = U.conj().T @ corr @ U

        #     x = self._pack_vector(psiarr, L, Lc, narr, Ec)
        #     norm = np.linalg.norm(self._compute_gradient(x))

        #     if (norm > best_norm):
        #         return prev_x
        #     else:
        #         best_norm = norm
        #         prev_x = x

        # -----------------------------------------

        # R = self._compute_renormalizations(psiarr, narr)
        # Aarr = self._get_A(narr, R, corr)
        # fock = mf.get_hcore() + mf.get_veff()

        # for K in range(N):
        #     LM = self._get_block(fock, K, K)
        #     LC = Aarr[K] - LM
        #     L[K] = LM
        #     Lc[K] = LC
        #     HK = self._get_HK(K, narr, Lc, R, corr)
        #     psi = psiarr[K]
        #     Ec[K] = (psi.conj().T @ HK @ psi).real / (np.linalg.norm(psi)**2)

        psiarr = [get_psi_matrix(psi) for psi in psiarr]
        x = self._pack_vector(psiarr, L, Lc, narr, Ec)
        if verbose:
            print(f"EGA={self._compute_lagrangian(x)}, EGHF={mf.e_tot}, (EGA - EGHF)/EGHF = {(self._compute_lagrangian(x) - mf.e_tot)/mf.e_tot}")
            Hqp = self._compute_Hqp(L, R) 
            qp_energy, qp_coeff = np.linalg.eigh(Hqp)
            print(f"|corrGA - corrGHF|/|corrGHF| = {np.linalg.norm(pcorr - self._compute_1body_correlations(qp_coeff, R, psiarr, qp_energy)) / np.linalg.norm(pcorr)}") 
            print(f"|guessGHF - static| = {np.linalg.norm(x - self._get_static_initial_guess())}")

        if hf:
            return x, {"converged":mf.converged, "e_tot":mf.e_tot}
        self._ren_time = []
        return x, None

    def _compute_1body_correlations(self, qp_coeff, R, psiarr, qp_energy):
        r"""
        Compute physical 1-body correlations under :math:`\ket{\Psi_G}` using:

        :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Jb} \ket{\Psi_G} = \sum_{cd} \left[ \mathcal{R}^I_{ac} \Delta^{IJ}_{cd} \mathcal{R}^{J*}_{bd} \right]`
        for :math:`I \neq J`.

        :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Ib} \ket{\Psi_G} = \text{Tr} \left[ \phi_I^\dagger c^\dagger_a c_b \phi_I  \right]` for each :math:`I`.
        """
        corr = self._compute_qp_density(qp_coeff, qp_energy)
        Rblock = scipy.linalg.block_diag(*R)
        expcorr = Rblock @ corr @ Rblock.T.conj()
        for I in range(self.N):
            psi = psiarr[I]
            M = self.M[I]
            C = self._Cdict[M]
            for a in range(M):
                for b in range(M):
                    expcorr[self._Moff[I]+a, self._Moff[I]+b] = _get_matrix_ip(psi, C[a].T @ C[b] @ psi)

        return expcorr

    def kernel(
        self,
        method="krylov",
        maxiter=None,
        x0=None,
        tolerance=1e-4,
        hf=True,
        x=False,
        verbose=True,
        n_bounds=(1e-8, 1 - 1e-8),
    ):
        r"""
        Use root finding (via scipy.optimize.root) on the gradient to calculate the ground state via Newton's method

        :param method: type of root finding method to use
        :param maxiter: maximum number of iterations (by default the solver iterates until convergence)
        :param x0: optional initial guess for Newton's method.  If not provided it will be computed by the kernel.
        :param tolerance: maximum acceptable gradient norm.  Default value is 1e-4.
        :param x: boolean specifying whether to included the packed vector in the result.  Default value is False.
        :param verbose: show verbose output regarding the Newton solver.  Default value is True.
        :param n_bounds: lower and upper bounds for occupation variables when using ``method="least_squares"``.

        The kernel returns a ``FermionGASCFResult`` object that can be queried for success status and values of Lagrange multipliers, Gutzwiller parameters and projectors, and correlation functions.
        """
        # create initial guess and solve
        self._put_harr()
        self._put_tblock()
        self._put_Uarr
        self._get_harr()
        self._get_tblock()
        self._get_Uarr()
        if (x0 is None):
            x0, hfres = self._get_initial_guess(verbose=verbose, hf=hf)
        elif hf:
            mf = self._GHF(self, verbose=verbose)
            mf.kernel()
            hfres = {"converged":mf.converged, "e_tot":mf.e_tot}
        else:
            hfres = None
        options = {}
        if maxiter:
            options['maxiter'] = maxiter
        if verbose:
            options['disp'] = True
        if method == "least_squares":
            nlo, nhi = n_bounds
            bounds_lo = np.full_like(x0, -np.inf, dtype=float)
            bounds_hi = np.full_like(x0, np.inf, dtype=float)
            n_slice = self._get_n_slice()
            bounds_lo[n_slice] = nlo
            bounds_hi[n_slice] = nhi
            x0 = np.array(x0, copy=True)
            x0[n_slice] = np.clip(x0[n_slice], nlo, nhi)

            def residual(z):
                try:
                    f = self._compute_gradient(z)
                except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                    return np.full_like(x0, 1e12, dtype=float)
                if not np.all(np.isfinite(f)):
                    return np.full_like(x0, 1e12, dtype=float)
                return f

            result = scipy.optimize.least_squares(
                residual,
                x0,
                bounds=(bounds_lo, bounds_hi),
                xtol=tolerance,
                ftol=tolerance,
                gtol=tolerance,
                max_nfev=maxiter,
                verbose=2 if verbose else 0,
            )
            result.fun_norm = np.linalg.norm(result.fun)
            if result.fun_norm <= tolerance:
                result.success = True
                result.message = f"{result.message} Residual norm {result.fun_norm:.6e} <= tolerance."
            else:
                result.success = False
                result.message = f"{result.message} Residual norm {result.fun_norm:.6e} > tolerance."
        else:
            options['fatol'] = tolerance
            result = scipy.optimize.root(self._compute_gradient, x0, method=method, options=options)

        # get runtime data
        print(f"Gradient total time: {sum(self._grad_time)}")
        print(f"Renormalization time: {sum(self._ren_time)}")
        print(f"psi gradient time: {sum(self._psi_time)}")
        print(f"QP correlation time: {sum(self._qp_time)}")
        print(f"L gradient time: {sum(self._L_time)}")
        print(f"Lc gradient time: {sum(self._Lc_time)}")
        print(f"n gradient time: {sum(self._n_time)}")
        print("-------------------------")
        print(f"Renormalization time: {100*sum(self._ren_time)/sum(self._grad_time)}%")
        print(f"psi gradient time: {100*sum(self._psi_time)/sum(self._grad_time)}%")
        print(f"QP correlation time: {100*sum(self._qp_time)/sum(self._grad_time)}%")
        print(f"L gradient time: {100*sum(self._L_time)/sum(self._grad_time)}%")
        print(f"Lc gradient time: {100*sum(self._Lc_time)/sum(self._grad_time)}%")
        print(f"n gradient time: {100*sum(self._n_time)/sum(self._grad_time)}%")

        # parse result
        psiarr, L, Lc, n, Ec = self._unpack_vector(result.x)
        R = self._compute_renormalizations(psiarr, n)
        Hqp = self._compute_Hqp(L, R)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        E = self._compute_lagrangian(result.x)
        if (not x):
            result.pop("x")
        expcorr = self._compute_1body_correlations(qp_coeff, R, psiarr, qp_energy)
        T, Tsrc = self._compute_density_renormalizations(psiarr, n)

        self._put_harr()
        self._put_tblock()
        self._put_Uarr()

        return FermionGASCFResult(self._Moff, self.Ne, E, psiarr, L, Lc, n, Ec, T=T, Tsrc=Tsrc, result=result, corr=expcorr, hfres=hfres)

class FermionGASCFResult:
    def __init__(self, Moff, Ne, E, psiarr, L, Lc, n, Ec, T=None, Tsrc=None, result=None, corr=None, hfres=None):
        N = len(Moff) - 1
        self.N = N
        self.Ne = Ne
        self._Moff = Moff
        self.E = E
        self.psiarr = psiarr
        self.L = L
        self.Lc = Lc
        self.n = n
        self.Ec = Ec
        self.result = result
        self.corr = corr
        self._has_T = (T is not None) and (Tsrc is not None)
        if self._has_T:
            self._T = T
            self._Tsrc = Tsrc
            TD = []
            for I in range(N):
                TD.append(np.einsum('abcc,c->ab', T[I], n[I]))
            self._TD = TD

        if (hfres is not None):
            self.hf = hfres

        projectors = []
        for I in range(N):
            projectors.append(self.psiarr[I] @ scipy.linalg.sqrtm(_compute_rdm(self.Delta(I))))
        self.projectors = projectors

    def Delta(self, I):
        r"""
        Return the matrix :math:`M_{ab} = \Delta^I_{ab} = \text{diag}(n^I)`.
        """
        return np.diag(self.n[I])

    def get_1body_corr(self, I, J):
        r"""
        Return the matrix :math:`M_{ab} = \bra{\Psi_G} c^\dagger_{Ia} c_{Jb} \ket{\Psi_G}`.
        """
        if (self.corr is None):
            return None
        return _get_block(self.corr, self._Moff, I, J)

    def get_density_corr(self, I, J):
        r"""
        Return the tensor :math:`M_{abcd} = \bra{\Psi_G} c^\dagger_{Ia} c_{Ib} c^\dagger_{Jc} c_{Jd} \ket{\Psi_G}`.
        """
        if not self._has_T:
            return None
        MI = self._Moff[I+1]-self._Moff[I]
        MJ = self._Moff[J+1]-self._Moff[J]
        corr2 = np.zeros((MI, MI, MJ, MJ), dtype=np.complex128)
        if (I == J):
            r"""
            :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Ib} c^\dagger_{Ic} c_{Id} \ket{\Psi_G} = \text{Tr} \left[ \phi_I^\dagger c^\dagger_a c_b c^\dagger_c c_d \phi_I  \right]`
            """
            M = MI
            psi = self.psiarr[I]
            C = _get_annahilation_operators(M)
            for a in range(M):
                for b in range(M):
                    for c in range(M):
                        for d in range(M):
                            corr2[a,b,c,d] = _get_matrix_ip(psi, C[a].T @ C[b] @ C[c].T @ C[d] @ psi)

        else:
            r"""
            :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Ib} c^\dagger_{Jc} c_{Jd} \ket{\Psi_G} = \bra{\Psi_0^e} \left(\displaystyle\sum_{ef} \left[ \mathcal{T}^I_{ab,ef} f^\dagger_{Ie} f_{If} \right] + \mathcal{T}^I_{ab} \right) \left(\displaystyle\sum_{gh} \left[ \mathcal{T}^J_{cd,gh} f^\dagger_{Jg} f_{Jh} \right] + \mathcal{T}^J_{cd} \right) \ket{\Psi_0^e}`
            Wick's theorem is OK here since the quaspiarticle states are a single Slater determinant.
            """
            corr2 = np.einsum('ab,cd->abcd', self._TD[I], self._TD[J])
            corr2 += -np.einsum('abef,cdgh,gf,eh->abcd', self._T[I], self._T[J], self.get_1body_corr(J, I), self.get_1body_corr(I, J))
            corr2 += np.einsum('ab,cd->abcd', self._TD[I], self._Tsrc[J])
            corr2 += np.einsum('ab,cd->abcd', self._Tsrc[I], self._TD[J])

        return corr2

    def get_number_corr(self, I, J):
        r"""
        Return the matrix :math:`M_{ab} = \bra{\Psi_G} n_{Ia} n_{Jb} \ket{\Psi_G}` where :math:`n_{Ia} \equiv c^\dagger_{Ia} c_{Ia}`.
        """
        return np.einsum('aabb->ab', self.get_density_corr(I, J)).real
