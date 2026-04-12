import numpy as np
from abc import abstractmethod
from ga_mband import FermionGASCF

class FermiBoseGASCF(FermionGASCF):
    def __init__(self, omega, *args):
        self.BN = len(omega)
        self.omega = omega
        self._beta = np.zeros(self.BN, dtype=np.complex128)
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
    
    def get_ht(self, I):
        h = self.get_h(I)
        g = np.zeros((self.M[I], self.M[I]), dtype=np.complex128)
        for nu in range(self.BN):
            g += self._beta[nu].conj()*self.get_g(nu, I, I)
        return h + g + g.conj().T
    
    @abstractmethod
    def get_t(self, I, J):
        pass
    
    def get_tt(self, I, J):
        t = self.get_t(I, J)
        if (I == J):
            return t
        g = np.zeros((self.M[I], self.M[J]), dtype=np.complex128)
        for nu in range(self.BN):
            beta = self._beta[nu]
            g += beta.conj()*self.get_g(nu, I, J) + beta*self.get_g(nu, J, I).conj().T
        return t + g
    
    def _compute_lagrangian(self, x):
        Lag = super()._compute_lagrangian(x)
        Lag += np.dot(np.abs(self._beta)**2 + 0.5, self.omega)
        return Lag
    
    def kernel(self, etol=1e-6, beta0=None, verbose=True, **kwargs):
        N = self.N
        if (beta0 is None):
            self._beta = np.zeros(self.BN, dtype=np.complex128)
        else:
            self._beta = beta0
        prev_E = 0
        fermion_res = None
        initialized = False
        while True:
            # solve fermion problem and check for convergence
            if verbose:
                print(f"beta = {self._beta}")
            fermion_res = super().kernel(verbose=verbose, **kwargs)
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

            # compute annihilator expectations
            expcorr = fermion_res.corr
            for nu in range(self.BN):
                Garr = np.zeros((N, N), dtype=object)
                for I in range(N):
                    for J in range(N):
                        Garr[I, J] = self._get_g(nu, I, J)
                Gnu = np.block(Garr.tolist())
                self._beta[nu] = -sum(np.dot(Gnu[i, :], expcorr[i, :]) for i in range(self._Moff[-1])) / self.omega[nu]

        return FermiBoseGASCFResult(fermion_res, self._beta)

class FermiBoseGASCFResult:
    def __init__(self, fermion_result, beta):
        self._fermion_result = fermion_result
        self.beta = beta
        self.BN = len(beta)

    def get_boson_ann_exp(self, nu):
        return self.beta[nu]
    
    def get_boson_number(self, nu):
        return np.abs(self.beta[nu])**2
    
    def __getattr__(self, name):
        return getattr(self._fermion_result, name)