import numpy
from pyscf import lib, gto

from pdb import set_trace as st

# matrix indices are given by (I, alpha) where I denotes the site, alpha the local state
# The ordering of matrix indices is given by (0,1), ... , (0, M), (1,0), ..., (1,M), ..., (N,M)

class GASCF(lib.StreamObject):
    def __init__(self, h1e, eri, N, Ne):
        # sanity check inputs
        if (Ne >= N):
            raise ValueError
        self.N = N
        self.Ne = Ne

        h1e_shape = h1e.shape
        if (len(h1e_shape) != 2):
            raise ValueError
        if (h1e_shape[0] != h1e_shape[1]):
            raise ValueError
        
        eri_shape = eri.shape
        if (len(eri_shape) != 4):
            raise ValueError
        if (not (eri_shape[0] == eri_shape[1] == eri_shape[2] == eri_shape[3] == h1e_shape[0])):
            raise ValueError
        
        if ((h1e_shape[0] % N) != 0):
            raise ValueError
        
        # extract 1-electron coefficients
        M = int(h1e_shape[0] / N)
        self.M = M
        h = numpy.zeros((N, M, M))
        h_int = numpy.zeros((N*M, N*M))
        for I in range(N):
            for a in range(M):
                for b in range(M):
                    h[I, a, b] = h1e[I*M + a, I*M + b]
                    h_int[I*M + a, I*M + b] = h1e[I*M + a, I*M + b]
        self.h = h
        self.t = h1e - h_int

        # extract 2-electron coefficients
        U = numpy.zeros((N, M, M, M, M))
        for I in range(N):
            for a in range(M):
                for b in range(M):
                    for c in range(M):
                        for d in range(M):
                            U[I, a, b, c, d] = eri[I*M + a, I*M + b, I*M + c, I*M + d]
        self.U = U

    def get_ht(self, I):
        return self.h[I]
    
    def get_U(self, I):
        return self.U[I]
    
    def get_tt(self):
        return self.t


def get_ga_model(N=12, filling=0.5, U=2.0, t=-1.0, PBC=True):
    dim = N * 2
    Ne = int(N * filling)

    # 1-electron interactions
    h1e = numpy.zeros((dim, dim))
    for I in range(N-1):
        for a in range(2):
            h1e[I*2 + a, (I+1)*2 + a] = t
    if PBC:
        for a in range(2):
            h1e[(N-1)*2 + a, a] = t

    # 2-electron interactions
    eri = numpy.zeros((dim, dim, dim, dim))
    for I in range(N):
        eri[I*2, I*2, I*2+1, I*2+1] = U

    return GASCF(h1e, eri, N, Ne)

if __name__ == '__main__':
    gamf = get_ga_model(PBC=False)