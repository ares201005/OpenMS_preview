import numpy as np
from abc import abstractmethod
from ga_mband_args import FermionGASCF, FermionGASCFResult

class FermiBoseGASCF(FermionGASCF):
    def __init__(self, omega, *args):
        self.BN = len(omega)
        self.omega = omega
        super().__init__(*args)

    @abstractmethod  
    def get_g(self, nu, I, J):
        pass

    def _get_g(self, nu, I, J):
        res = self.get_g(nu, I, J)
        if (res.shape != (self.M[I], self.M[J])):
            raise ValueError(f"g{nu}[{I}, {J}] has incorrect shape")
        return res

    @abstractmethod
    def get_h(self, I):
        pass
    
    def get_ht(self, I, beta):
        h = self.get_h(I)
        g = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        for nu in range(self.BN):
            g += beta[nu].conj()*self.get_g(nu, I, I)
        return h + g + g.conj().T
    
    @abstractmethod
    def get_t(self, I, J):
        pass
    
    def get_tt(self, I, J, beta):
        t = self.get_t(I, J)
        if (I == J):
            return t
        g = np.zeros((self.M[I], self.M[J]), dtype=np.complex128)
        for nu in range(self.BN):
            b = beta[nu]
            g += b.conj()*self.get_g(nu, I, J) + b*self.get_g(nu, J, I).conj().T
        return t + g
    
    def _pack_boson_vector(self, x, beta):
        assert (len(beta) == self.BN)
        return np.concatenate([x, beta.real, beta.imag])
    
    def _unpack_boson_vector(self, y):
        BN = self.BN
        assert (len(y) > 2*self.BN)
        x = y[:-2*BN]
        re = y[-2*BN:-BN]
        im = y[-BN:]
        c = re + 1j*im
        return x, c
    
    def _compute_lagrangian(self, y):
        x, beta = self._unpack_boson_vector(y)
        Lag = super()._compute_lagrangian(x, beta)
        Lag += np.dot(np.abs(beta)**2 + 0.5, self.omega)
        return Lag
    
    def _compute_gradient(self, y):
        N = self.N
        x, beta = self._unpack_boson_vector(y)
        fermion_grad, expcorr = super()._compute_gradient(x, beta)

        # compute gradient wrt beta_nu*
        grad_beta = np.zeros(self.BN, dtype=np.complex128)
        for nu in range(self.BN):
            Garr = np.zeros((N, N), dtype=object)
            for I in range(N):
                for J in range(N):
                    Garr[I, J] = self._get_g(nu, I, J)
            Gnu = np.block(Garr.tolist())
            a = sum(np.dot(Gnu[i, :], expcorr[i, :]) for i in range(self._Moff[-1]))
            grad_beta[nu] = a + beta[nu]*self.omega[nu]

        g = lambda grad: np.array([2*G for G in grad])
        return self._pack_boson_vector(fermion_grad, g(grad_beta))
    
    def _get_initial_guess(self, beta0):
        fermion_guess = super()._get_initial_guess()
        beta = beta0
        if (beta is None):
            beta = np.zeros(self.BN, dtype=np.complex128)
        return self._pack_boson_vector(fermion_guess, beta)
    
    def kernel(self, method="krylov", maxiter=None, x0=None, beta0=None, tolerance=1e-4, keep=False, verbose=True):
        result = self._get_result(method, maxiter, x0, tolerance, verbose, beta0)
        x_fermion, beta = self._unpack_boson_vector(result.x)
        rargs, rkwargs = self._parse_result(result, keep, beta, x=x_fermion)
        return FermiBoseGASCFResult(beta, *rargs, **rkwargs)
    
class FermiBoseGASCFResult(FermionGASCFResult):
    def __init__(self, beta, *args, **kwargs):
        self.beta = beta
        self.BN = len(beta)
        super().__init__(*args, **kwargs)

    def get_boson_ann_exp(self, nu):
        return self.beta[nu]
    
    def get_boson_number(self, nu):
        return np.abs(self.beta[nu])**2
