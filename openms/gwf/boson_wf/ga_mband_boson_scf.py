import numpy as np
from abc import abstractmethod
from ga_mband import FermionGASCF

# return boson annihilation operator with M excitations
def _get_annihilation_operator(M):
    B = np.zeros((M, M))
    for n in range(1, M):
        B[n-1, n] = np.sqrt(n)
    return B

def _get_fock_annihilation_operator(nu, Bo, BM):
        B = np.eye(int(np.prod(BM[:nu])))
        B = np.kron(Bo, B)
        B = np.kron(np.eye(int(np.prod(BM[nu+1:]))), B)
        return B

def _boson_array_to_idx(tup, BM):
    BN = len(BM)
    if (len(tup) != BN):
        raise ValueError("Tuple does not correspond to a valid Fock state")
    idx = tup[-1]
    if not (0 <= idx < BM[-1]):
        raise ValueError(f"Invalid occupation number {idx} for mode {BN-1}")
    for i in range(BN - 1):
        j = BN - 2 - i
        n = tup[j]
        b = BM[j]
        if not (0 <= n < b):
            raise ValueError(f"Invalid occupation number {n} for mode {j}")
        idx = n + (idx * b)
    return idx

def _boson_idx_to_array(idx, BM):
    BN = len(BM)
    tup = np.zeros(BN, dtype=np.int32)
    for nu in range(BN):
        idx, tup[nu] = divmod(idx, BM[nu])
    return tup

class FermiBoseGASCF(FermionGASCF):
    def __init__(self, BM, omega, *args):
        self.BN = len(BM)
        if (len(omega) != self.BN):
            raise ValueError("Number of modes is not consistent")
        self.BM = BM
        self.omega = omega
        self._nfock = np.prod(BM)
        self._Bdict = {}
        for n in set(BM):
            self._Bdict[n] = _get_annihilation_operator(n)
        self._c = np.zeros(self._nfock, dtype=np.complex128)
        super().__init__(*args)

    def boson_array_to_idx(self, tup):
        return _boson_array_to_idx(tup, self.BM)

    def boson_idx_to_array(self, idx):
        if not (0 <= idx < self._nfock):
            raise ValueError("Index does not correspond to a valid Fock state")
        return _boson_idx_to_array(idx, self.BM)

    @abstractmethod  
    def get_g(self, nu, I, J):
        pass

    def _get_g(self, nu, I, J):
        res = self.get_g(nu, I, J)
        if (res.shape != (self.M[I], self.M[J])):
            raise ValueError(f"g{nu}[{I}, {J}] has incorrect shape")
        return res
    
    def _get_annihilation_operator(self, nu):
        return _get_fock_annihilation_operator(nu, self._Bdict[self.BM[nu]], self.BM)

    @abstractmethod
    def get_h(self, I):
        pass
    
    def get_ht(self, I):
        h = self.get_h(I)
        g = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        c = self._c
        for nu in range(self.BN):
            beta = c.conj().T @ self._get_annihilation_operator(nu) @ c
            g += beta.conj()*self.get_g(nu, I, I)
        return h + g + g.conj().T
    
    @abstractmethod
    def get_t(self, I, J):
        pass
    
    def get_tt(self, I, J):
        t = self.get_t(I, J)
        if (I == J):
            return t
        g = np.zeros((self.M[I], self.M[J]), dtype=np.complex128)
        c = self._c
        for nu in range(self.BN):
            beta = c.conj().T @ self._get_annihilation_operator(nu) @ c
            g += beta.conj()*self.get_g(nu, I, J) + beta*self.get_g(nu, J, I).conj().T
        return t + g
    
    def _compute_lagrangian(self, x):
        Lag = super()._compute_lagrangian(x)
        c = self._c
        for m in range(self._nfock):
            Lag += np.dot(self.boson_idx_to_array(m) + 0.5, self.omega) * (np.linalg.norm(c[m])**2)
        return Lag
    
    def kernel(self, etol=1e-6, x0=None, x=False, c0=None, verbose=True, **kwargs):
        N = self.N
        if (c0 is None):
            self._c = np.zeros(self._nfock, dtype=np.complex128)
            self._c[0] = 1
        else:
            self._c = c0
        prev_E = 0
        fermion_res = None
        initialized = False
        while True:
            # solve fermion problem and check for convergence
            if verbose:
                print(f"c = {self._c}")
            fermion_res = super().kernel(x0=x0, x=True, verbose=verbose, **kwargs)
            E = fermion_res.E

            if initialized:
                dE = E - prev_E
                if verbose:
                    print(f"E = {E}, dE = {dE}")
                if (abs(dE) <= etol):
                    break
            else:
                initialized = True
                if verbose:
                    print(f"E = {E}")
            prev_E = E
            x0 = fermion_res.result.x

            # construct Hb
            expcorr = fermion_res.corr
            vbare = np.zeros(self._nfock, dtype=np.complex128)
            for m in range(self._nfock):
                vbare[m] = np.dot(self.boson_idx_to_array(m) + 0.5, self.omega)
            Hb = np.diag(vbare)
            Hlin = np.zeros((self._nfock, self._nfock), dtype=np.complex128)
            for nu in range(self.BN):
                B = self._get_annihilation_operator(nu)
                Garr = np.zeros((N, N), dtype=object)
                for I in range(N):
                    for J in range(N):
                        Garr[I, J] = self._get_g(nu, I, J)
                Gnu = np.block(Garr.tolist())
                a = sum(np.dot(Gnu[i, :], expcorr[i, :]) for i in range(self._Moff[-1]))
                Hlin += a*B.T
            Hb += Hlin + Hlin.conj().T

            # get ground state of Hb
            eigval, eigvec = np.linalg.eigh(Hb)
            self._c = eigvec[:, 0]

        if (not x):
            fermion_res.result.pop("x")
            
        return FermiBoseGASCFResult(fermion_res, self.BM, self._c)

class FermiBoseGASCFResult:
    def __init__(self, fermion_result, BM, c):
        self._fermion_result = fermion_result
        self.BM = BM
        self.BN = len(BM)
        self.c = c

        # get annahilator expectations
        self._beta = np.zeros(self.BN, dtype=np.complex128)
        for nu in range(self.BN):
            B = _get_annihilation_operator(self.BM[nu])
            B = _get_fock_annihilation_operator(nu, B, self.BM)
            self._beta[nu] = c.conj().T @ B @ c

        # get boson occupancy numbers
        self._boson_number = np.zeros(self.BN)
        for m in range(len(c)):
            self._boson_number += (np.linalg.norm(c[m])**2) * _boson_idx_to_array(m, BM)

    def get_boson_ann_exp(self, nu):
        return self._beta[nu]
    
    def get_boson_number(self, nu):
        return self._boson_number[nu]
    
    def __getattr__(self, name):
        return getattr(self._fermion_result, name)
