import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

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
        return res
    
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
    
    @abstractmethod
    def get_tt(self, I, J):
        r"""
        Return the matrix :math:`\tilde{t}^{IJ}_{ab}` between sites :math:`I \neq J`.
        Must be implemented by the  concrete class.
        """
        pass

    def _get_tt(self, I, J):
        res = self.get_tt(I, J)
        if (res.shape != (self.M[I], self.M[J])):
            raise ValueError(f"tt[{I}, {J}] has incorrect shape")
        return res

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

    def _compute_renormalizations(self, psiarr, n):
        r"""
        Compute all 1-body renormalization factors, given by

        .. math::

             \mathcal{R}^I_{\alpha a} = \dfrac{\bra{\Psi_I} c_{\alpha}^\dagger f_{b}^\dagger \ket{\Psi_I}}{\sqrt{n_a^I(1-n_a^I)}} 

        """
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
                for b in range(M):
                    opmat[alpha, b] = psi.conj().T @ (np.kron(C[alpha].T, C[b].T)) @ psi
            R.append(opmat * (1/np.sqrt(nI*(1-nI))))

        return R
    
    def _compute_density_renormalizations(self, psiarr, n):
        r"""
        Compute all 2-body (number-preserving) renormalization factors, given by

        .. math::

            \mathcal{T}^I_{\alpha\beta,ab} &= \dfrac{\bra{\Psi_I} c_{\alpha}^\dagger c_\beta (f_a^\dagger f_b - \delta_{ab}n^I_a \mathbb{I}) \ket{\Psi_I}}{\sqrt{n^I_an^I_b(1-n^I_a)(1-n^I_b)}} \\
            \mathcal{T}^I_{\alpha\beta} &= \bra{\Psi_I} c_\alpha^\dagger c_\beta \ket{\Psi_I} - \sum_{a} \mathcal{T}^I_{\alpha\beta, aa}n^I_{a}

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
                    P[alpha, beta] = psi.conj() @ np.kron(M1, np.eye(2**M)) @ psi
                    for a in range(M):
                        for b in range(M):
                            G[alpha, beta, a, b] = psi.conj().T @ np.kron(M1, X[a, b]) @ psi
                            
            nI = n[I]
            b = 1/np.sqrt(nI*(1-nI))
            bb = np.outer(b, b)
            T4 = (G - np.einsum('ab,ij->abij', P, np.diag(nI))) * bb[None, None, :, :]
            T.append(T4)
            Tsrc.append(P - np.einsum('abii,i->ab', T4, nI))

        return T, Tsrc

    
    def _compute_Hqp(self, L, R, tarr=None):
        r"""
        Compute

        .. math::

            \hat{H}_{qp} = & \sum_{IJ ab} \left[ \sum_{\alpha\beta}\tilde{t}^{IJ}_{\alpha\beta} \mathcal{R}^{I}_{\alpha a} \mathcal{R}^{J*}_{\beta b} \right] f^\dagger_{Ia}f_{Jb}  \\
            & + \sum_{I ab} \left[ \lambda^I_{ab}f_{Ia}^\dagger f_{Ib} + h.c.  \right]

        in the single-particle basis.
        """
        N = self.N
        if (tarr is None):
            tarr = self._get_tarr()
        t = np.block(tarr.tolist())
        Rblock = scipy.linalg.block_diag(*R)
        
        Hqp = Rblock.T @ t @ Rblock.conj()

        LH = []
        for I in range(N):
            LH.append(L[I] + L[I].conj().T)
        Hqp += scipy.linalg.block_diag(*LH)

        return Hqp
    
    def _compute_Hemb(self, I, Lc, C=None):
        r"""
        Compute

        .. math::

            \hat{H}_{emb}^I &= \displaystyle\sum_{\alpha\beta} \tilde{h}^{I}_{\alpha\beta}c^\dagger_{\alpha}c_{\beta} + \sum_{\alpha\beta\gamma\delta}U^{I}_{\alpha\beta\gamma\delta}c^\dagger_\alpha c_\beta^\dagger c_\gamma c_\delta \\
            &+ \displaystyle\sum_{ab} \left[ (\lambda_c^I)_{ab}f_b^\dagger f_a + h.c. \right]

        in the mixed tensor-product basis.
        """
        M = self.M[I]
        # get coefficients for embedding Hamiltonian
        h = self._get_ht(I)
        U = self._get_U(I)
        if (C is None):
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
        Lag += sum(qp_energy[i] for i in range(self.Ne))

        # get embedding Hamiltonian expectations and normalization terms
        for I in range(N):
            Hemb = self._compute_Hemb(I, Lc)
            psi = psiarr[I]
            Lag += psi.T.conj() @ Hemb @ psi
            # calculate Ec contribution
            Lag += Ec[I] * (1 - (psi.T.conj() @ psi))

        #  calculate Lmix
        Lmix = -sum(np.trace((L[I] + Lc[I]) @ np.diag(n[I])) for I in range(N))
        Lag += Lmix + Lmix.conj()
        
        return Lag.real
    
    def _compute_num_gradient(self, x):
        N = x.shape[0]
        base = self._compute_lagrangian(x)
        inc = 0.001*np.linalg.norm(x)
        grad = np.zeros(N)
        for I in range(N):
            y = x.copy()
            y[I] += inc
            new = self._compute_lagrangian(y)
            grad[I] = (new - base) / inc

        comp = self._compute_gradient(x)
        print(f"compute_num_gradient invoked, num = {np.linalg.norm(grad)}, comp = {np.linalg.norm(comp)}, percent = {100 * np.linalg.norm(grad-comp)/np.linalg.norm(comp)}")

        return grad

    def _compute_gradient(self, x):
        r"""
        Compute the first derivatives

        .. math::

            \dfrac{\partial \mathcal{L}_e}{\partial E_c^K} &= 1 - \braket{\Psi_K} \\
            \dfrac{\partial \mathcal{L}_e}{\partial \lambda^K_{ab}} &= \bra{\Psi_0^e} c_{Ka}^\dagger c_{Kb} \ket{\Psi_0^e} - \Delta^K_{ab} \\
            \dfrac{\partial \mathcal{L}_e}{\partial (\lambda^K_c)_{ab}} &= \bra{\Psi_K} f_b^\dagger f_a \ket{\Psi_K} - \Delta^K_{ab} \\
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
            &+ \sum_{\alpha\gamma} \left[ \left[\sum_I \tilde{t}^{{IK}^T} \mathcal{R}^I \Delta^{IK}  {B^K} \right]_{\alpha\gamma} c_\alpha f_\gamma\right] + h.c. \\
            
       Here, :math:`B^K` is a diagonal matrix with 

        .. math::

            (B^K)_{ab} = \dfrac{\delta_{ab}}{\sqrt{n^K_a(1-n^K_a)}}

        """
        N = self.N
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)
        tarr = self._get_tarr()
        Cdict = {}
        for M in set(self.M):
            Cdict[M] = _get_annahilation_operators(M)

        # get energy gradients
        grad_Ec = [1 - (psiarr[I].T.conj() @ psiarr[I]) for I in range(N)]

        # get quasiparticle and embedding Hamiltonians
        R = self._compute_renormalizations(psiarr, n)
        Hqp = self._compute_Hqp(L, R, tarr=tarr)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)

        # compute single particle correlations
        occ = [(i < self.Ne) for i in range(self._Moff[-1])]
        corr = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T

        # get <psi_K| gradients, each of size 4**M
        grad_psiarr = []
        for K in range(N):
            M = self.M[K]
            C = Cdict[M]
            nK = n[K]
            B = np.diag(1/np.sqrt(nK*(1- nK)))
            
            # construct derivative Hamiltonian
            HD = self._compute_Hemb(K, Lc, C=C) - Ec[K]*np.eye(4**M)
            HDqp = np.zeros((4**M, 4**M), dtype=np.complex128)
            for I in range(N):
                M1 = tarr[I, K].T @ R[I] @ self._get_block(corr, I, K) @ B
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
            idx = np.arange(M)
            Lm[idx, idx] -= n[I]
            grad_L.append(Lm)

        grad_Lc = []
        for I in range(N):
            M = self.M[I]
            Lm = np.zeros((M,M), dtype=np.complex128)
            C = Cdict[M]
            psivec = psiarr[I]
            for a in range(M):
                for b in range(M):
                    Lm[a,b] = psivec.conj().T @ np.kron(np.eye(2**M), C[b].T @ C[a]) @ psivec
            idx = np.arange(M)
            Lm[idx, idx] -= n[I]
            grad_Lc.append(Lm)

        # get n gradients
        grad_n = []
        for K in range(N):
            nK = n[K]
            r = 0.5 * R[K] * (1/(1-nK) - 1/nK)
            M1 = sum((r.conj().T @ tarr[I, K].T @ R[I] @ self._get_block(corr, I, K)) for I in range(N))
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

    def kernel(self, method="krylov", maxiter=None, x0=None, tolerance=1e-4, verbose=True):
        r"""
        Use root finding (via scipy.optimize.root) on the gradient to calculate the ground state via Newton's method

        :param method: type of root finding method to use
        :param maxiter: maximum number of iterations (by default the solver iterates until convergence)
        :param x0: optional initial guess for Newton's method.  If not provided it will be computed by the kernel.
        :param tolerance: maximum acceptable gradient norm.  Default value is 1e-4.
        :param verbose: show verbose output regarding the Newton solver.  Default value is True.

        The kernel returns a ``FermionGASCFResult`` object that can be queried for success status and values of Lagrange multipliers, Gutzwiller parameters and projectors, and correlation functions.
        """
        # create initial guess and solve
        N = self.N
        if (x0 == None):
            x0 = self._get_initial_guess()
        options = {}
        if maxiter:
            options['maxiter'] = maxiter
        if verbose:
            options['disp'] = True
        options['fatol'] = tolerance
        result = scipy.optimize.root(self._compute_num_gradient, x0, method=method, options=options)

        # parse result
        psiarr, L, Lc, n, Ec = self._unpack_vector(result.x)
        R = self._compute_renormalizations(psiarr, n)
        Hqp = self._compute_Hqp(L, R)
        qp_energy, qp_coeff = np.linalg.eigh(Hqp)
        occ = [(i < self.Ne) for i in range(self._Moff[-1])]
        corr = qp_coeff[:, occ].conj() @ qp_coeff[:, occ].T
        E = self._compute_lagrangian(result.x)
        result.pop("x")

        # compute 1-body correlations, then fix diagonal blocks
        r"""
        :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Jb} \ket{\Psi_G} = \sum_{cd} \left[ \mathcal{R}^I_{ac} \Delta^{IJ}_{cd} \mathcal{R}^{J*}_{bd} \right]`
        for :math:`I \neq J`
        """
        Rblock = scipy.linalg.block_diag(*R)
        expcorr = Rblock @ corr @ Rblock.T.conj()
        Cdict = {}
        for M in set(self.M):
            Cdict[M] = _get_annahilation_operators(M)
        for I in range(N):
            r"""
            :math:`\bra{\Psi_G} c^\dagger_{Ia} c_{Ib} \ket{\Psi_G} = \text{Tr} \left[ \phi_I^\dagger c^\dagger_a c_b \phi_I  \right]`
            """
            psi = get_psi_matrix(psiarr[I])
            M = self.M[I]
            C = Cdict[M]
            for a in range(M):
                for b in range(M):
                    expcorr[self._Moff[I]+a, self._Moff[I]+b] = np.trace(psi.conj().T @ C[a].T @ C[b] @ psi)

        T, Tsrc = self._compute_density_renormalizations(psiarr, n)

        return FermionGASCFResult(self._Moff, self.Ne, E, psiarr, L, Lc, n, Ec, T=T, Tsrc=Tsrc, result=result, corr=expcorr)
    
class FermionGASCFResult:
    def __init__(self, Moff, Ne, E, psiarr, L, Lc, n, Ec, T=None, Tsrc=None, result=None, corr=None):
        N = len(Moff) - 1
        self.N = N
        self.Ne = Ne
        self._Moff = Moff
        self.E = E
        self.psiarr = [get_psi_matrix(psiarr[I]) for I in range(N)]
        self.L = L
        self.Lc = Lc
        self.n = n
        self.Ec = Ec
        self.result = result
        self.corr = corr
        self._number = (T is not None) and (Tsrc is not None)
        if self._number:
            self._T = T
            self._Tsrc = Tsrc
            TD = []
            for I in range(N):
                TD.append(np.einsum('abcc,c->ab', T[I], n[I]))
            self._TD = TD

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
        if not self._number:
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
                            corr2[a,b,c,d] = np.trace(psi.conj().T @ C[a].T @ C[b] @ C[c].T @ C[d] @ psi)

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