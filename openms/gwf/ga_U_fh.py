import numpy as np
import re
from sys import argv
import matplotlib.pyplot as plt
from ga_2d_fermi_hubbard import FermiHubbard

msg = ""

class FermiHubbardScanner(FermiHubbard): 
    def kernel(self, verbose=True, **kwargs):
        global msg
        if verbose:
            dim = 2 if ('2d' in FermiHubbard.__module__) else 1
            print(f"{dim}D ", end="")
        msg = re.sub(r',?\s*U=[^,]+', '', self.msg)
        return super().kernel(verbose=verbose, **kwargs)

def get_res(U, x0):
    hf = (x0 is None)
    if (len(argv) <= 3):
        gamf = FermiHubbardScanner(U=U)
        if (len(argv) == 1):
            return gamf.kernel(hf=hf, x=True, x0=x0)
        elif (len(argv) == 2):
            return gamf.kernel(hf=hf, x=True, x0=x0, method=argv[1])
        elif (len(argv) == 3):
            return gamf.kernel(hf=hf, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = None
        if ('2d' in FermiHubbard.__module__):
            gamf = FermiHubbardScanner(U=U, n=int(argv[3]))
        else:
            gamf = FermiHubbardScanner(U=U, N=int(argv[3]))
        if (len(argv) == 4):
            return gamf.kernel(hf=hf, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            return gamf.kernel(hf=hf, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")

def main():
    init_U = 2.0
    min_U = 6.0
    max_U = 9.0
    step = 0.125
    Uarr = []
    Earr = []
    x0 = None
    for U in np.arange(init_U, max_U, step):
        res = get_res(U, x0)
        E = res.E
        Uarr.append(U)
        Earr.append(E)
        x0 = res.result.x
        print(f"U={U}: E={E}")

    idx = Uarr.index(min_U)
    trunc_Uarr = Uarr[idx:-1]
    trunc_Earr = Earr[idx:-1]

    invU = 1/np.array(trunc_Uarr)
    slope, intercept = np.polyfit(invU, trunc_Earr, 1)
    Earr_fit = slope*invU + intercept
    ss_res = np.sum((trunc_Earr - Earr_fit)**2)
    ss_tot = np.sum((trunc_Earr - np.mean(trunc_Earr))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Fitted m={slope}, b={intercept}, R²={r_squared}")

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: E vs 1/U ----
    axes[0].scatter(invU, trunc_Earr, label='Data')
    axes[0].plot(invU, Earr_fit, '-', 
                label=f'Fit: slope={slope:.3f}, R²={r_squared:.3f}')
    axes[0].set_xlabel(r'$1/U$')
    axes[0].set_ylabel(r'$E$')
    axes[0].set_title(r'$E$ vs $1/U$')
    axes[0].legend()

    # ---- Right: E vs U ----
    axes[1].scatter(Uarr, Earr, label='Data')
    axes[1].set_xlabel(r'$U$')
    axes[1].set_ylabel(r'$E$')
    axes[1].set_title(r'$E$ vs $U$')
    axes[1].legend()

    plt.suptitle(msg)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()