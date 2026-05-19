import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from ga_mband import FermionGASCF

class FermiHubbard(FermionGASCF):
    def __init__(self, n=4, filling=0.5, U=2.0, t=-1.0, J=0.0, PBC=True):
        if (n <= 0):
            raise ValueError("lattice size must be positive")
        if ((filling <= 0) or (filling >= 1)):
            raise ValueError("filling must be in between 0 and 1 exclusive")
        self.n = n
        N = n**2
        Ne = int(filling * N * 2)
        self.U = np.zeros((2,2,2,2))
        self.U[0,1,0,1] = -U
        self.t = (t*np.eye(2)) + (J*np.array([[-1, 1], [1, -1]]))
        self.PBC = PBC
        self.msg = f"n={n}, N={N}, t={t}, U={U}, J={J}, PBC={PBC}, filling={Ne/(N*2)}"
        super().__init__(N*[2], Ne)

    def get_ht(self, I):
        return np.zeros((2,2))
    
    def get_U(self, I):
        return self.U
    
    def tuple_to_idx(self, i, j):
        return i*self.n + j
    
    def idx_to_tuple(self, idx):
        return divmod(idx, self.n)
    
    def is_nn(self, x1, x2):
        i1, j1 = x1
        i2, j2 = x2
        di = abs(i1 - i2)
        dj = abs(j1 - j2)
        if self.PBC:
            di = min(di, self.n - di)
            dj = min(dj, self.n - dj)
        return (di + dj == 1)
    
    def get_tt(self, I, J):
        if self.is_nn(self.idx_to_tuple(I), self.idx_to_tuple(J)):
            return self.t
        return np.zeros((2,2))
    
    def _compute_structure_factor(self, A):
        n = self.n
        N = self.N
        x = np.arange(n)
        y = np.arange(n)
        qxs = None
        qys = None
        data = None
        if self.PBC:
            qxs = 2*np.pi*np.arange(n)/n
            qys = 2*np.pi*np.arange(n)/n
            data = np.zeros((n,n))
            for ix, qx in enumerate(qxs):
                vx = np.exp(1j*qx*x)
                for iy, qy in enumerate(qys):
                    vy = np.exp(1j*qy*y)
                    v = np.outer(vx, vy).reshape(N)
                    data[ix, iy] = (1/n)**2 * (v.conj().T @ A @ v).real
        else:
            qxs = np.pi*np.arange(1, n+1)/(n+1)
            qys = np.pi*np.arange(1, n+1)/(n+1)
            data = np.zeros((n,n))
            x = np.arange(1, n+1)
            y = np.arange(1, n+1)
            for ix, qx in enumerate(qxs):
                vx = np.sin(qx*x)
                for iy, qy in enumerate(qys):
                    vy = np.sin(qy*y)
                    v = np.outer(vx, vy).reshape(N)
                    data[ix, iy] = (2/(n+1))**2 * (v.T @ A @ v).real

        return qxs, qys, data

    def _charge_structure_factor(self, res):
        N = self.N
        A = np.zeros((N, N))
        for I in range(N):
            for J in range(N):
                M = res.get_number_corr(I, J) - np.outer(np.diag(res.get_1body_corr(I, I)), np.diag(res.get_1body_corr(J, J)))
                A[I, J] = M.real.sum()
                
        print(f"number (CDW) A sum = {A.sum()}")
        return self._compute_structure_factor(A)

    def _spin_structure_factor(self, res):
        N = self.N
        A = np.zeros((N, N))
        for I in range(N):
            for J in range(N):
                nIJ = res.get_number_corr(I, J)
                A[I, J] = 0.25 * (nIJ[0,0] + nIJ[1,1] - nIJ[0,1] - nIJ[1,0])
      
        return self._compute_structure_factor(A)
    
    def _3d_plot_structure_factors(self, qxS, qyS, S, qxN, qyN, N):
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(self.msg, fontsize=16)

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        surf1 = ax1.plot_surface(*np.meshgrid(qxS/np.pi, qyS/np.pi, indexing="ij"), S)
        surf2 = ax2.plot_surface(*np.meshgrid(qxN/np.pi, qyN/np.pi, indexing="ij"), N)

        ax1.set_title("Spin structure factor $S(\\mathbf{q})$", fontsize=16)
        ax1.set_xlabel(r"$q_x/\pi$")
        ax1.set_ylabel(r"$q_y/\pi$")
        ax1.set_zlabel("Structure Factor")

        ax2.set_title("Charge structure factor $N(\\mathbf{q})$", fontsize=16)
        ax2.set_xlabel(r"$q_x/\pi$")
        ax2.set_ylabel(r"$q_y/\pi$")
        ax2.set_zlabel("Structure Factor")

        plt.show()
    
    def _heat_plot_structure_factors(self, qxS, qyS, S, qxN, qyN, N):
        fig, axs = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle(self.msg, fontsize=16)

        # Spin
        im0 = axs[0].imshow(
            S.T,                    # transpose so qy runs vertically
            origin="lower",
            aspect="equal",
            extent=[qxS[0]/np.pi, qxS[-1]/np.pi, qyS[0]/np.pi, qyS[-1]/np.pi],
        )
        axs[0].set_title("Spin structure factor $S(\\mathbf{q})$", fontsize=16)
        axs[0].set_xlabel(r"$q_x/\pi$")
        axs[0].set_ylabel(r"$q_y/\pi$")
        fig.colorbar(im0, ax=axs[0])

        # Charge
        im1 = axs[1].imshow(
            N.T,
            origin="lower",
            aspect="equal",
            extent=[qxN[0]/np.pi, qxN[-1]/np.pi, qyN[0]/np.pi, qyN[-1]/np.pi],
        )
        axs[1].set_title("Charge structure factor $N(\\mathbf{q})$", fontsize=16)
        axs[1].set_xlabel(r"$q_x/\pi$")
        axs[1].set_ylabel(r"$q_y/\pi$")
        fig.colorbar(im1, ax=axs[1])

        plt.show()
    
    def kernel(self, verbose=True, **kwargs):
        if verbose:
            print("kernel invoked: " + self.msg)
        res = super().kernel(verbose=verbose, **kwargs)
        if verbose:
            print(f"self.Ne = {self.Ne}")
            print(f"Delta computed Ne = {sum(np.trace(res.Delta(I)) for I in range(self.N))}")
            print(f"correlation computed Ne = {sum(np.trace(res.get_1body_corr(I, I)) for I in range(self.N))}")
            print(f"E = {res.E}")
        return res
        
    def do_kernel(self, verbose=True, **kwargs):
        res = self.kernel(verbose=verbose, **kwargs)    
        
        qxS, qyS, S = self._spin_structure_factor(res)
        qxN, qyN, N = self._charge_structure_factor(res)

        if verbose:
            print(f"<Sz> = {0.5 * sum(np.trace(res.get_1body_corr(I, I) @ np.diag([1,-1])) for I in range(self.N))}")

            Min = self.idx_to_tuple(np.argmin(S))
            Min = np.array([qxS[Min[0]], qyS[Min[1]]])/np.pi
            Max = self.idx_to_tuple(np.argmax(S))
            Max = np.array([qxS[Max[0]], qyS[Max[1]]])/np.pi
            print(f"S(q) maxq/pi={Max}, minq/pi={Min}")

            Min = self.idx_to_tuple(np.argmin(N))
            Min = np.array([qxN[Min[0]], qyN[Min[1]]])/np.pi
            Max = self.idx_to_tuple(np.argmax(N))
            Max = np.array([qxN[Max[0]], qyN[Max[1]]])/np.pi
            print(f"N(q) maxq/pi={Max}, minq/pi={Min}")
            
        self._3d_plot_structure_factors(qxS, qyS, S, qxN, qyN, N)
        self._heat_plot_structure_factors(qxS, qyS, S, qxN, qyN, N)

if __name__ == '__main__':
    if (len(argv) <= 3):
        gamf = FermiHubbard()
        if (len(argv) == 1):
            gamf.do_kernel()
        elif (len(argv) == 2):
            gamf.do_kernel(method=argv[1])
        elif (len(argv) == 3):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = FermiHubbard(n=int(argv[3]))
        if (len(argv) == 4):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            gamf.do_kernel(method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")

