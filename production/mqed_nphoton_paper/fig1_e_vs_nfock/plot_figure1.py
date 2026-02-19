import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


def load_qed_data(filename, method):
    """Load QED ground state energies from HDF5 file."""

    qed_data = {}
    with h5py.File(filename, 'r') as h5file:

        # Iterate over lambda groups
        for lambda_key in h5file.keys():
            if not lambda_key.startswith('lambda_'):
                continue

            # Extract lambda value from key
            lambda_str = lambda_key.replace('lambda_', '')
            lambda_val = float(lambda_str)

            lambda_group = h5file[lambda_key]

            # Extract converged energies for specified method
            if method in lambda_group:
                energy = lambda_group[method]['energy'][:]
                nfock = lambda_group[method]["nfock"][()] \
                    if 'nfock' in lambda_group[method] else 4

                cs_energy = None
                if method == "QEDHF":
                    cs_energy = lambda_group[method]['cs_energy'][()]
                if method in ("SCQEDHF", "VTQEDHF"):
                    nfock = nfock - 1

                qed_data[lambda_val] = {
                    'energy': energy,
                    'cs_energy': cs_energy,
                    'nfock': nfock
                }

    return qed_data


def plot_nfock_data(nfock_dicts, fig_path):

    # Enabling LaTeX text rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Create figure and axes
    fig = plt.figure(figsize=(6, 9),  constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[1.5, 1.5, 1], hspace=0.01)

    # Initialize subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    # Style parameters
    labels = [r"\textbf{QED-HF}", r"\textbf{SC-QED-HF}", r"\textbf{VT-QED-HF}"]
    colors = ["#4477AA", "#228833", "#AA3377"]
    lstyles = ["--", "-.", ":"]
    markers = ["o", "s", "^"]
    lwidths = [1.75, 1.75, 1.75]

    # y-axis limits
    y1min = y1max = y2max = y3max = None


    # Create plots
    # ------------------------------
    for ind, method in enumerate(nfock_dicts):

        # Lambda values
        x1 = sorted(method.keys())
        x2 = x1.copy()

        # Create list for energies and nfock
        y1 = []
        y2 = []

        # Append energies and nfock values
        for k in x1:
            max_nf = method[k]["nfock"]

            # Adjust nfock for VTQEDHF method (over-convergence)
            if ind == 2 and max_nf == 4:
                y2.append(3)
                y1.append(method[k]["energy"][:][-2])
            else:
                y2.append(max_nf)
                y1.append(method[k]["energy"][:][-1])

        # # Total energies
        # y1 = [method[k]["energy"][:][-1] for k in x1]

        # # Number of Fock states
        # y2 = [method[k]["nfock"] for k in x2]

        # Offset points overlapping points for visual clarity
        offset = 0.15
        if ind == 0:
            y2 = [y2[k] + offset if y2[k] <= 3 else y2[k] for k in range(len(y2))]
        elif ind == 2:
            y2 = [y2[k] - offset for k in range(len(y2))]

        # Energy difference convergence
        energy_list = method[0.3]["energy"]
        if ind == 1 or ind == 2:
            energy_list = energy_list[:-1]
        y3 = np.abs(energy_list - energy_list[-1])
        y3[-1] = 1e-6 # Set last intensity such that it appears on log scale

        # x-axis datapoints from length of y datapoints
        x3 = list(range(1, len(y3) + 1))

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

        if y3max is None:
            y3max = np.max(y3)
        else:
            if np.max(y3) > y3max:
                y3max = np.max(y3)

        # First plot, total energy vs lambda
        # -----------------------------
        ax1.plot(x1, y1,
            color = colors[ind],
            linestyle = lstyles[ind],
            marker = markers[ind],
            markevery = 4,
            markersize = 7,
            linewidth = lwidths[ind],
            label = labels[ind])

        # Second plot, nfock vs lambda
        # -----------------------------
        ax2.scatter(x2, y2, s = 20,
            color = colors[ind],
            marker = markers[ind],
            label = labels[ind])

        # Third plot, ediff vs nfock
        # -----------------------------
        ax3.plot(x3, y3,
            color = colors[ind],
            linestyle = lstyles[ind],
            marker = markers[ind],
            markersize = 7,
            linewidth = lwidths[ind],
            label = labels[ind])


    # Set x/y limits
    # ------------------------------
    ax1.set_xlim(-0.005, 0.305)

    pad1 = 0.05 * np.abs((y1max - y1min))
    ax1.set_ylim(y1min - pad1, y1max + pad1)

    ax2.set_xlim(-0.005, 0.305)
    ax2.set_ylim(0.6, 7.4)

    ax3.set_xlim(0.8, 7.2)
    ax3.set_ylim(0.000001, y3max + 0.05 * y3max)
    ax3.set_yscale("log")


    # Set tickmark parameters
    # ------------------------------
    ax1.tick_params(bottom=True, labelbottom=True, labelsize=13)

    ax1.set_xticklabels([])
    ax1.yaxis.set_major_locator(
        MaxNLocator(nbins=9, prune=None))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax2.tick_params(axis="both", which="major", labelsize=13)

    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax2.yaxis.set_major_locator(plt.MultipleLocator(1))

    ax3.tick_params(axis='both', which='major', labelsize=13)

    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # Set x/y labels
    # ------------------------------
    ax1.set_ylabel(r"\textbf{ground state energy (a.u.)}", fontsize=14)

    ax2.set_xlabel(r"\textbf{coupling strength,} \boldmath${\lambda}_{\alpha}$", fontsize=14)
    ax2.set_ylabel(r"\textbf{number of photon states}", fontsize=14)

    ax3.set_xlabel(r"\textbf{number of photon states}", fontsize=14)
    ax3.set_ylabel(r"\textbf{energy error (a.u.)}", fontsize=14)


    # Legend
    # ------------------------------

    custom_handles = [
        Line2D([0], [0],
               color=colors[0],
               linestyle=lstyles[0],
               marker=markers[0],
               linewidth=lwidths[0],
               label=labels[0]),
        Line2D([0], [0],
               color=colors[1],
               linestyle=lstyles[1],
               marker=markers[1],
               linewidth=lwidths[1],
               label=labels[1]),
        Line2D([0], [0],
               color=colors[2],
               linestyle=lstyles[2],
               marker=markers[2],
               linewidth=lwidths[2],
               label=labels[2]),
    ]

    ax1.legend(handles=custom_handles, fontsize=12,
               fancybox=False, framealpha=0.6, edgecolor="black")
    ax2.legend(handles=custom_handles, fontsize=12,
               fancybox=False, framealpha=0.6, edgecolor="black")
    ax3.legend(handles=custom_handles, fontsize=12,
               fancybox=False, framealpha=0.6, edgecolor="black")


    # Align y-axis labels
    # ------------------------------
    fig.align_ylabels([ax2, ax3])


    # Subplot label
    # ------------------------------
    subplot_a = r"\textbf{a)}"
    ax1.text(-0.225, 1.05, subplot_a,
        transform=ax1.transAxes, fontsize=16, va="top", ha="left")

    subplot_b = r"\textbf{b)}"
    ax2.text(-0.225, 1.05, subplot_b,
        transform=ax2.transAxes, fontsize=16, va="top", ha="left")

    subplot_c = r"\textbf{c)}"
    ax3.text(-0.225, 1.05, subplot_c,
        transform=ax3.transAxes, fontsize=16, va="top", ha="left")

    ax3.text(0.44, 0.74, r"\boldmath${\lambda}_{\alpha}=0.30$",
             transform=ax3.transAxes, fontsize=12)
    ax3.text(0.31, 0.86, r"\textbf{coupling strength}",
             transform=ax3.transAxes, fontsize=12)


    # Save/show
    # ------------------------------
    #plt.show()
    plt.savefig(fig_path)


if __name__ == "__main__":

    # Basis set
    # ---------------------
    basis = "cc-pvdz"

    # QED-HF data
    # ---------------------
    qedhf_data = {}
    qedhf_h5 = f"{basis}/qedhf_e_vs_nfock.h5"
    qedhf_data = load_qed_data(qedhf_h5, method="QEDHF")

    # SC-QED-HF data
    # ---------------------
    scqedhf_data = {}
    scqedhf_h5 = f"{basis}/scqed_e_vs_nfock.h5"
    scqedhf_data = load_qed_data(scqedhf_h5, method="SCQEDHF")

    # VT-QED-HF data
    # ---------------------
    vtqedhf_data = {}
    vtqedhf_h5 = f"{basis}/vtqed_e_vs_nfock.h5"
    vtqedhf_data = load_qed_data(vtqedhf_h5, method="VTQEDHF")


    # ---------------------
    # All figures
    # ---------------------
    figure_path = f"{basis}/fig1_en_vs_la.pdf"
    plot_nfock_data(
        nfock_dicts=[qedhf_data, scqedhf_data, vtqedhf_data],
        fig_path = figure_path
    )
