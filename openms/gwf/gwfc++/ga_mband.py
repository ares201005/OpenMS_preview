import numpy as np
from abc import ABC, abstractmethod
from pyscf import scf, gto
from gafermion import FermionGACPP

import scipy.linalg

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
        self._gacpp = FermionGACPP(M, Ne)
        self.M = np.asarray(self._gacpp.M, dtype=np.int32)
        self.M.setflags(write=False)
        self.N = self._gacpp.N

        self.filling = Ne/self._Moff[-1]

        self._harr = None
        self._tarr = None
        self._Uarr = None

    def __getattr__(self, name):
        return getattr(self._gacpp, name)
    
    @abstractmethod
    def get_ht(self, I):
        r"""
        Return the matrix :math:`\tilde{h}^I_{ab}` for site :math:`I`.
        Must be implemented by the  concrete class.
        """
        pass

    def _get_ht(self, I):
        if (self._harr[I] is not None):
            return self._harr[I]
        M = self.M[I]
        res = self.get_ht(I)
        if (res.shape != (M, M)):
            raise ValueError(f"ht[{I}] has incorrect shape")
        self._harr[I] = res
        return res
    
    def _get_harr(self):
        if (self._harr is None):
            N = self.N
            self._harr = [None]*N
            for I in range(N):
                self._harr[I] = self._get_ht(I)
    
    def _put_harr(self):
        self._harr = None
    
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
        if (self._Uarr is None):
            N = self.N
            self._Uarr = [None]*N
            for I in range(N):
                self._Uarr[I] = self._get_U(I)

    def _put_Uarr(self):
        self._Uarr = None

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
    
    def _get_block(self, mat, I, J):
        return _get_block(mat, self._Moff, I, J)
    
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
            psivec = get_psi_vector(np.eye(2**M))
            psivec = psivec / np.linalg.norm(psivec)
            psiarr.append(psivec)
        
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
            self.max_cycle = 1000
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
        # get GHF 1-body correlations in the natural basis
        mf = self._GHF(self, verbose=verbose)
        mf.kernel()
        pcorr = mf.make_rdm1().conj().T
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

        fock = mf.get_hcore() + mf.get_veff()
        fock = U.conj().T @ fock @ U
        L = [0.5*self._get_block(fock, I, I) for I in range(N)]

        Lc = [None] * N
        Ec = np.zeros(N)
        # R = self._compute_renormalizations(psiarr, narr)
        # Aarr = self._get_A(narr, R, self._tblock, corr)
        # for I in range(N):
        #     Lc[I] = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        #     HK = self._get_HK(I, narr, Lc, R, self._tblock, corr)
        #     eigval, eigvec = np.linalg.eigh(HK)
        #     psiarr[I] = eigvec[:, 0]
        #     Ec[I] = eigval[0]
        #     Lc[I] = Aarr[I] - L[I]
        
        x = self._get_static_initial_guess()
        if hf:
            return x, {"converged":mf.converged, "e_tot":mf.e_tot}
        return x, None
    
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
        psiarr, L, Lc, n, Ec = self._unpack_vector(x)
        grad_psiarr, grad_L, grad_Lc, grad_n, grad_Ec = self._gacpp._compute_gradient(psiarr, L, Lc, n, Ec, self._harr, self._tblock, self._Uarr)
        f = lambda grad: [2*G.conj() for G in grad]
        g = lambda grad: [2*G for G in grad]
        return self._pack_vector(g(grad_psiarr), f(grad_L), f(grad_Lc), grad_n, grad_Ec)
    
    def kernel(self, method="krylov", maxiter=None, x0=None, tolerance=1e-4, hf=True, x=False, verbose=True):
        r"""
        Use root finding (via scipy.optimize.root) on the gradient to calculate the ground state via Newton's method

        :param method: type of root finding method to use
        :param maxiter: maximum number of iterations (by default the solver iterates until convergence)
        :param x0: optional initial guess for Newton's method.  If not provided it will be computed by the kernel.
        :param tolerance: maximum acceptable gradient norm.  Default value is 1e-4.
        :param x: boolean specifying whether to included the packed vector in the result.  Default value is False.
        :param verbose: show verbose output regarding the Newton solver.  Default value is True.

        The kernel returns a ``FermionGASCFResult`` object that can be queried for success status and values of Lagrange multipliers, Gutzwiller parameters and projectors, and correlation functions.
        """
        # create initial guess
        self._put_harr()
        self._put_tblock()
        self._put_Uarr()
        self._get_harr()
        self._get_tblock()
        self._get_Uarr()
        if (x0 is None):
            x0, hfres = self._get_initial_guess(verbose=verbose, hf=hf)
        # elif hf:
        #     mf = self._GHF(self, verbose=verbose)
        #     mf.kernel()
        #     hfres = {"converged":mf.converged, "e_tot":mf.e_tot}
        # else:
        #     hfres = None

        # solve
        options = {}
        if maxiter:
            options['maxiter'] = maxiter
        if verbose:
            options['disp'] = True
        options['fatol'] = tolerance
        result = scipy.optimize.root(self._compute_gradient, x0, method=method, options=options)
        breakpoint()

        self._put_harr()
        self._put_tblock()