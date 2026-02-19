import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

def load_qed_data(filename, nfock1=False):
    """Load QED ground state energies from HDF5 file."""

    qed_data = {'QEDFCI': {}, 'QEDHF': {}, 'SCQEDHF': {}, 'VTQEDHF': {}}

    with h5py.File(filename, 'r') as h5file:

        # Iterate over lambda groups
        for lambda_key in h5file.keys():
            if not lambda_key.startswith('lambda_'):
                continue

            # Extract lambda value from key
            lambda_str = lambda_key.replace('lambda_', '')
            lambda_val = float(lambda_str)

            lambda_group = h5file[lambda_key]

            # Extract converged energies for each method
            for method in qed_data.keys():

                if method == "QEDFCI":
                    fci_energy = lambda_group[method]["fci_energy"][:]
                    fci_elec = lambda_group[method]["fci_elec"][:]
                    fci_boson_energy = lambda_group[method]["fci_boson_energy"][:]

                    # Verify arrays have same length
                    if len(fci_energy) != len(fci_boson_energy) != len(fci_elec):
                        raise ValueError(f"Mismatched array lengths for lambda={lambda_val}")

                    total_energy = fci_elec + fci_boson_energy
                    qed_data[method][lambda_val] = total_energy

                else:
                    total_energy = lambda_group[method]["energy"][:]
                    qed_data[method][lambda_val] = total_energy

    return qed_data


def plot_nfock_data(nfock_data, fig_path, converged=True):

    # Enabling LaTeX text rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 8),
        sharex=True, constrained_layout=True,
        gridspec_kw={'hspace': 0.01}
    )

    # Style parameters
    labels = [r"\textbf{QED-FCI}", r"\textbf{QED-HF}",
        r"\textbf{SC-QED-HF}", r"\textbf{VT-QED-HF}"]
    colors = ["black", "#4477AA", "#228833", "#AA3377"]
    lstyles = ["-", "--", "-.", ":"]
    markers = ["", "o", "s", "^"]
    lwidths = [2.0, 1.75, 1.75, 1.75]

    # y-axis limits
    y1min = y1max = y2max = None


    # Create plots
    # ------------------------------
    for ind, method in enumerate(nfock_data.keys()):

        # Lambda values
        x = sorted(nfock_data[method].keys())

        # Converged or nfock=1 energies
        energies = [nfock_data[method][k] for k in x]
        if converged:
            y1 = [e_list[-1] for e_list in energies]
        else:
            y1 = [e_list[0] for e_list in energies]

        # Energy differences
        y2 = y1 - y1[0]

        # Determine y-axis limits
        if y1min is None:
            y1min = np.min(y1)
        else:
            if np.min(y1) < y1min:
                y1min = np.min(y1)

        if y1max is None:
            y1max = np.max(y1)
        else:
            if np.max(y1) > y1max:
                y1max = np.max(y1)

        if y2max is None:
            y2max = np.max(y2)
        else:
            if np.max(y2) > y2max:
                y2max = np.max(y2)


        # First plot, energy vs lambda
        # -----------------------------
        ax1.plot(x, y1,
            color = colors[ind],
            linestyle = lstyles[ind],
            marker = markers[ind],
            markevery = 5,
            markersize = 7,
            linewidth = lwidths[ind],
            label = labels[ind])

        # Second plot, e_diff vs lambda
        # ------------------------------
        ax2.plot(x, y2,
            color = colors[ind],
            linestyle = lstyles[ind],
            marker = markers[ind],
            markevery = 5,
            markersize = 7,
            linewidth = lwidths[ind],
            label = labels[ind])


    # Set x/y limits
    # ------------------------------
    padding = 0.05 * np.abs((y1max - y1min))

    ax1.set_xlim(-0.005, 0.305)
    ax1.set_ylim(y1min - padding, y1max + padding)

    ax2.set_xlim(-0.005, 0.305)
    ax2.set_ylim(-0.005, (y2max + 0.02 * y2max))


    # Set tickmark parameters
    # ------------------------------
    ax1.tick_params(bottom=True, labelbottom=True, labelsize=13)

    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_locator(
        MaxNLocator(nbins=9, prune=None))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax2.tick_params(axis="both", which="major", labelsize=13)

    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_locator(
        MaxNLocator(nbins=9, prune=None))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # Set x/y labels
    # ------------------------------
    ax1.set_ylabel(r"\textbf{total energy (a.u.)}", fontsize=14)

    ax2.set_xlabel(r"\textbf{coupling strength,} \boldmath${\lambda}_{\alpha}$", fontsize=14)
    ax2.set_ylabel(r"\textbf{QED energy (a.u.)}", fontsize=14)


    # Legend
    # ------------------------------
    ax1.legend(fontsize=12, fancybox=False, framealpha=0.6, edgecolor="black")

    ax2.legend(fontsize=12, fancybox=False, framealpha=0.6, edgecolor="black")


    # Align y-axis labels
    # ------------------------------
    fig.align_ylabels([ax1, ax2])


    # Subplot label
    # ------------------------------
    subplot_label = r"\textbf{a)}" if not converged else r"\textbf{b)}"
    ax1.text(-0.155, 1.05, subplot_label,
        transform=ax1.transAxes, fontsize=16, va="top", ha="left")


    # Save/show
    # ------------------------------
    plt.savefig(fig_path)
    #plt.show()


if __name__ == "__main__":

    # Molecule
    # ---------------------
    #mol = "LiF"
    mol = "LiH"

    # Basis set
    # ---------------------
    basis = "cc-pvdz"

    # ---------------------
    # Read data
    # ---------------------
    nfock_h5 = f"{basis}_{mol}/qed_e_vs_fci_lambda.h5"
    nfock_data = load_qed_data(nfock_h5)

    # ---------------------
    # nfock=1 figure
    # ---------------------
    fig3a_path = f"{basis}_{mol}/plots/fig3a_en_vs_la_nf1.pdf"
    plot_nfock_data(nfock_data, fig3a_path, converged=False)

    # ---------------------
    # Converged nfock figure
    # ---------------------
    fig3b_path = f"{basis}_{mol}/plots/fig3b_en_vs_la_nfconv.pdf"
    plot_nfock_data(nfock_data, fig3b_path, converged=True)

    exit()
