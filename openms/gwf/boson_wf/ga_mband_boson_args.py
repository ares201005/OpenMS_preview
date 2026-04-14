import numpy as np
from abc import abstractmethod
from ga_mband_args import FermionGASCF, FermionGASCFResult

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
    
    def get_ht(self, I, c):
        h = self.get_h(I)
        g = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        for nu in range(self.BN):
            beta = c.conj().T @ self._get_annihilation_operator(nu) @ c
            g += beta.conj()*self.get_g(nu, I, I)
        return h + g + g.conj().T
    
    @abstractmethod
    def get_t(self, I, J):
        pass
    
    def get_tt(self, I, J, c):
        t = self.get_t(I, J)
        if (I == J):
            return t
        g = np.zeros((self.M[I], self.M[J]), dtype=np.complex128)
        for nu in range(self.BN):
            beta = c.conj().T @ self._get_annihilation_operator(nu) @ c
            g += beta.conj()*self.get_g(nu, I, J) + beta*self.get_g(nu, J, I).conj().T
        return t + g
    
    def _pack_boson_vector(self, x, c, Eb):
        assert (len(c) == self._nfock)
        return np.concatenate([x, c.real, c.imag, [Eb]])
    
    def _unpack_boson_vector(self, y):
        nfock = self._nfock
        assert (len(y) > 2*nfock + 1)
        Eb = y[-1]
        y = y[:-1]
        x = y[:-2*nfock]
        re = y[-2*nfock:-nfock]
        im = y[-nfock:]
        c = re + 1j*im
        return x, c, Eb
    
    def _compute_lagrangian(self, y):
        x, c, Eb = self._unpack_boson_vector(y)
        Lag = super()._compute_lagrangian(x, c)

        for m in range(self._nfock):
            Lag += (np.dot(self.boson_idx_to_array(m) + 0.5, self.omega) - Eb) * (np.linalg.norm(c[m])**2)
        Lag += Eb 

        return Lag
    
    def _compute_gradient(self, y):
        N = self.N
        x, c, Eb = self._unpack_boson_vector(y)
        fermion_grad, expcorr = super()._compute_gradient(x, c, expcorr=True)

        # compute gradient wrt cm*
        grad_c = -Eb * c
        for m in range(self._nfock):
            grad_c[m] += np.dot(self.boson_idx_to_array(m) + 0.5, self.omega) * c[m]
        for nu in range(self.BN):
            B = self._get_annihilation_operator(nu)
            Garr = np.zeros((N, N), dtype=object)
            for I in range(N):
                for J in range(N):
                    Garr[I, J] = self._get_g(nu, I, J)
            Gnu = np.block(Garr.tolist())
            a = sum(np.dot(Gnu[i, :], expcorr[i, :]) for i in range(self._Moff[-1]))
            grad_c += (a*B.T + a.conj()*B) @ c

        grad_Eb = 1 - (np.linalg.norm(c)**2)

        g = lambda grad: np.array([2*G for G in grad])
        return self._pack_boson_vector(fermion_grad, g(grad_c), grad_Eb)
    
    def _get_initial_guess(self, c0):
        fermion_guess = super()._get_initial_guess()
        c = c0
        if (c is None):
            c = np.zeros(self._nfock, dtype=np.complex128)
            c[0] = 1
        Eb = 0.5*sum(self.omega)
        return self._pack_boson_vector(fermion_guess, c, Eb)
    
    def kernel(self, method="krylov", maxiter=None, x0=None, c0=None, tolerance=1e-4, keep=False, verbose=True):
        result = self._get_result(method, maxiter, x0, tolerance, verbose, c0)
        x_fermion, c, Eb = self._unpack_boson_vector(result.x)
        rargs, rkwargs = self._parse_result(result, keep, c, x=x_fermion)
        return FermiBoseGASCFResult(self.BM, c, Eb, *rargs, **rkwargs)
    
class FermiBoseGASCFResult(FermionGASCFResult):
    def __init__(self, BM, c, Eb, *args, **kwargs):
        self.BM = BM
        self.BN = len(BM)
        self.c = c
        self.Eb = Eb

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

        super().__init__(*args, **kwargs)

    def get_boson_ann_exp(self, nu):
        return self._beta[nu]
    
    def get_boson_number(self, nu):
        return self._boson_number[nu]
