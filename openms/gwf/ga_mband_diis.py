import numpy as np
import scipy.linalg
from pyscf import lib
from sys import argv



class GenericDIIS(lib.diis.DIIS):
    def __init__(self, space=8, rollback=0, filename=None):
        super().__init__(filename=filename)
        self.space = space
        self.rollback = rollback

    def update(self, var, var_grad):
        errvec = var_grad
        lib.logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(errvec))
        params = var
        xnew = lib.diis.DIIS.update(self, params, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

class FermionGASCF(lib.StreamObject):
    def __init__(self, h1e, U, N, Ne):
        # sanity check inputs
        if (Ne > h1e.shape[0]):
            raise ValueError
        self.N = N
        self.Ne = Ne

        h1e_shape = h1e.shape
        if (len(h1e_shape) != 2):
            raise ValueError
        if (h1e_shape[0] != h1e_shape[1]):
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
        for I in range(N):
            U_shape = U[I].shape
            if (len(U_shape) != 4):
                raise ValueError
            if (not(U_shape[0] == U_shape[1] == U_shape[2] == U_shape[3])): # add check for equaling site dependent M
                raise ValueError
        self.U = U


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

        # assert all(psiarr[i].size == 4**M for i in range(N))
        # assert all(L[i].size == M*M for i in range(N))
        # assert all(Lc[i].size == M*M for i in range(N))
        # assert all(Delta[i].size == M*M for i in range(N))
        # assert np.asarray(Ec).ravel().size == N

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
        psi_size = int(np.sqrt(psi.shape)[0])
        matrix = np.zeros((psi_size, psi_size), dtype=psi.dtype)
        for Gamma in range(psi_size):
            for n in range(psi_size):
                matrix[Gamma, n] = psi[Gamma*(psi_size) + n]
        return matrix

    def _compute_renormalizations(self, psiarr, Delta):
        # make R as a vector
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

    def _compute_Hqp(self, psiarr,
                    Delta, # local correlaiton on each site
                    L, # lambda variable
        ):
        r"""Compute the quasiparticle Hamiltonain

        .. math::
            \hat{H}_{qp} = & \sum_{IJ ab} \left[ \sum_{\alpha\beta}\tilde{t}^{IJ}_{\alpha\beta} \mathcal{R}^{I}_{\alpha a} \mathcal{R}^{J*}_{\beta b} \right] c^\dag_{Ia}c_{Jb} \\
                           &+\sum_{I ab} \left[ \lambda^I_{ab}c_{Ia}^\dag c_{Ib} + h.c.  \right]

        """
        #YZ: change this to vectorized code
        #    M should be a list of length N
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
        # YZ: looks problematic (bath part)
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

        # YZ: looks problematic, need to double check it!!!
        # lambda should be determined by the GWF constraints
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
    
    def _get_initial_guess(self):
        # initialize all psi to uniformly id on each block (unentangled)
        # L and Lc to 0 and Delta uniform identity on all sites
        # Ec to 0
        N = self.N
        psiarr = []
        L = []
        Lc = []
        Delta = []
        for I in range(N):
            M = self.M
            psi = np.zeros(4**M, dtype=np.complex128)
            for Gamma in range(2**M):
                psi[Gamma*(2**M) + Gamma] = 1
            psiarr.append(psi)

            ZM = np.eye(M, dtype=np.complex128)
            L.append(ZM.copy())
            Lc.append(ZM.copy())
            Delta.append((self.Ne/(N*M))*np.eye(M))
        Ec = np.zeros(N)
        return self._pack_vector(psiarr, L, Lc, Delta, Ec)

    # def project(self, x):
    #     psiarr, L, Lc, Delta, Ec = self._unpack_vector(x)
    #     for I in range(self.N):
    #         Delta[I] = 0.5*(Delta[I] + Delta[I].T.conj())
    #         L[I] = 0.5*(L[I] + L[I].T.conj())
    #         Lc[I] = 0.5*(Lc[I] + Lc[I].T.conj())
    #         psiarr[I] = psiarr[I] / np.linalg.norm(psiarr[I])
    #     return self._pack_vector(psiarr, L, Lc, Delta, Ec)

    def _project(self, x):
        return x

    def _do_simple_diis(self, x0, tolerance, verbose, diis_space=8, maxiter=200):
        diis = GenericDIIS(space=diis_space, rollback=0)
        x = x0
        max_iter=200
        for it in range(1, max_iter+1):
            g = self._compute_gradient(x)
            gnorm = np.linalg.norm(g)
            if verbose:
                print(f"iter {it}, norm = {gnorm}")

            # break if converged
            if (gnorm < tolerance):
                return x, {"converged":True, "niter":it, "gnorm":gnorm}
            # else perform update and try again
            x = self._project(diis.update(x, g))
            
        return x, {"converged": False, "niter": max_iter, "gnorm": float(np.linalg.norm(self._compute_gradient(x)))}

    def _do_fallback_diis(self, x0, tolerance, verbose, diis_space=8, diis_start=0, maxiter=200, alpha=0.05, beta=0.2):
        diis = GenericDIIS(space=diis_space, rollback=0)
        
        x = self._project(x0)

        for it in range(1, maxiter+1):
            g = self._compute_gradient(x)
            gnorm = np.linalg.norm(g)
            if verbose:
                print(f"iter {it}, ||g|| = {gnorm:.6e}")
            if  (gnorm < tolerance):
                return x, {"converged":True, "niter":it, "gnorm":gnorm}
            
            x_out = self._project(x - alpha*g)
            err = x_out - x

            if (it >= diis_start):
                x_diis = diis.update(x_out, err)
                x_try = x_diis
                # x_try = self._project((1-beta)*x_out +  beta*x_diis)
            else:
                x_try = x_out
            
            # if g got worse, forget about it
            gnorm_try = np.linalg.norm(self._compute_gradient((x_try)))
            if (np.isfinite(gnorm_try) and (gnorm_try <= gnorm)):
                x = x_try
            else:
                if verbose:
                    print("reject step, fallback and reset")
                x = x_out
                beta = 0.5*beta
                alpha = 0.5*alpha
                diis = GenericDIIS(space=diis_space, rollback=0)

        return x, {"converged":False, "niter":maxiter, "gnorm":gnorm}
    
    def _do_diis(self, x0, tolerance, verbose, diis_space=8, maxiter=200):
        return self._do_simple_diis(x0, tolerance, verbose, diis_space=diis_space, maxiter=maxiter)
        # return self._do_fallback_diis(x0, tolerance, verbose, diis_space=diis_space, maxiter=maxiter)

    def kernel(self, method="krylov", maxiter=None, tolerance=1e-6, verbose=True):
        print(f"kernel invoked: N={self.N}, M={self.M}, Ne={self.Ne}")
        N = self.N

        x0 = self._get_initial_guess()

        options = {}
        options['maxiter'] = 20
        if verbose:
            options['disp'] = True

        options['fatol'] = tolerance
        result = scipy.optimize.root(self._compute_gradient, x0, method=method, options=options)
        
        if (maxiter == None):
            maxiter = 200
        x, msg = self._do_diis(result.x, tolerance, verbose, diis_space=8, maxiter=maxiter)

        psiarr, L, Lc, Delta, Ec = self._unpack_vector(x)
        projectors = []
        for I in range(N):
            M = self.M
            K = scipy.linalg.logm((np.eye(M) - Delta[I]) @ np.linalg.inv(Delta[I]))
            C = self._get_annahilation_operators(M)
            rho = scipy.linalg.expm(sum(-K[a, b] * C[a].T @ C[b] for a in range(M) for b in range(M)))
            rho = (1/np.trace(rho)) * rho
            projectors.append(self.get_psi_matrix(psiarr[I]) @ scipy.linalg.sqrtm(rho))

        psidelta = []
        for I in range(N):
            M = self.M
            mat = np.zeros((M, M), dtype=np.complex128)
            for a in range(M):
                for b in range(M):
                    mat[a, b] = psiarr[I].conj().T @ np.kron(np.eye(2**M), C[b].T) @  np.kron(np.eye(2**M), C[a]) @ psiarr[I]
            psidelta.append(mat)

        print(f"Computed Ne = {sum(np.trace(Delta[I]) for I in range(N))}, self.Ne = {self.Ne}")
        breakpoint()

        print(msg)
        print(x)

        self._post_kernel()

    def _post_kernel(self):
        pass


def get_ga_model(N=12, filling=0.5, U=1.0, t=-1.0, PBC=True):
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
    eri = np.zeros((N, 2, 2, 2, 2))
    for I in range(N):
        eri[I, 0, 1, 0, 1] = -U

    return FermionGASCF(h1e, eri, N, Ne)

if __name__ == '__main__':
    gamf = get_ga_model(PBC=False)
    if (len(argv) == 1):
        gamf.kernel()
    elif (len(argv) == 2):
        gamf.kernel(method=argv[1])
    elif (len(argv) == 3):
        gamf.kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) == 4):
        gamf = get_ga_model(N=int(argv[3]), PBC=False)
        gamf.kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) == 5):
        gamf = get_ga_model(N=int(argv[3]), PBC=False)
        gamf.kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")
