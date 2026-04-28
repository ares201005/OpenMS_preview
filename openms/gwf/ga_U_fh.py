import numpy as np
import re
from sys import argv
import matplotlib.pyplot as plt
from ga_fermi_hubbard import FermiHubbard

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
    if (len(argv) <= 3):
        gamf = FermiHubbardScanner(U=U)
        if (len(argv) == 1):
            return gamf.kernel(hf=True, x=True, x0=x0)
        elif (len(argv) == 2):
            return gamf.kernel(hf=True, x=True, x0=x0, method=argv[1])
        elif (len(argv) == 3):
            return gamf.kernel(hf=True, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]))
    elif (len(argv) <= 5):
        gamf = None
        if ('2d' in FermiHubbard.__module__):
            gamf = FermiHubbardScanner(U=U, n=int(argv[3]))
        else:
            gamf = FermiHubbardScanner(U=U, N=int(argv[3]))
        if (len(argv) == 4):
            return gamf.kernel(hf=True, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]))
        elif (len(argv) == 5):
            return gamf.kernel(hf=True, x=True, x0=x0, method=argv[1], tolerance=float(argv[2]), maxiter=int(argv[4]))
    else:
        print(f"Usage: {argv[0]} [method] [tolerance] [nsites] [maxiter]")

def main():
    init_U = 2.0
    fin_U = 8.0
    step = 0.25
    Uarr = []
    corr_E = []
    x0 = None
    for U in np.arange(init_U, fin_U, step):
        res = get_res(U, x0)
        if res.hf["converged"]:
            diff = res.hf["e_tot"] - res.E
            Uarr.append(U)
            corr_E.append(diff)
            x0 = res.result.x
            print(f"U={U}: corrE={diff}")
        else:
            print(f"U={U}: GHF not converged, skipping")
    
    logU = np.log(Uarr)
    logE = np.log(corr_E)
    slope, intercept = np.polyfit(logU, logE, 1)
    logE_fit = slope*logU + intercept
    ss_res = np.sum((logE - logE_fit)**2)
    ss_tot = np.sum((logE - np.mean(logE))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Fitted m={slope}, b={intercept}, R²={r_squared}")

    plt.figure()
    plt.scatter(logU, logE, label='Data')
    plt.plot(logU, logE_fit, '-', label=f'Fit: slope={slope:.3f}, R²={r_squared:.3f}')
    plt.xlabel(r'$\text{log}(U)$')
    plt.ylabel(r'$\text{log}(\Delta E)$')
    plt.legend()
    plt.title(msg)
    plt.show()

if __name__ == '__main__':
    main()