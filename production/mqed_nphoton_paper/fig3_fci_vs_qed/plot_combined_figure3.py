import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

def load_qed_data(filename):
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

def plot_combined_figure(nfock_data, fig_path):
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
    methods = ['QEDFCI', 'QEDHF', 'SCQEDHF', 'VTQEDHF']
    labels = [r"\textbf{QED-FCI}", r"\textbf{QED-HF}",
        r"\textbf{SC-QED-HF}", r"\textbf{VT-QED-HF}"]
    colors = ["black", "#4477AA", "#228833", "#AA3377"]
    markers = ["", "o", "s", "^"]
    lwidths = [2.0, 1.75, 1.75, 1.75]

    # y-axis limits
    y1min = y1max = y2max = None

    # Create plots for both converged and nfock=1
    # ------------------------------
    for ind, method in enumerate(methods):
        # Lambda values
        x = sorted(nfock_data[method].keys())

        # Energies for both converged and nfock=1
        energies = [nfock_data[method][k] for k in x]
        y1_conv = [e_list[-1] for e_list in energies]  # Converged
        y1_nf1 = [e_list[0] for e_list in energies]   # nfock=1

        # Energy differences
        y2_conv = np.array(y1_conv) - y1_conv[0]
        y2_nf1 = np.array(y1_nf1) - y1_nf1[0]

        # Update y-axis limits based on all data
        all_y1 = np.concatenate([y1_conv, y1_nf1])
        all_y2 = np.concatenate([y2_conv, y2_nf1])

        if y1min is None:
            y1min = np.min(all_y1)
        else:
            y1min = min(y1min, np.min(all_y1))

        if y1max is None:
            y1max = np.max(all_y1)
        else:
            y1max = max(y1max, np.max(all_y1))

        if y2max is None:
            y2max = np.max(all_y2)
        else:
            y2max = max(y2max, np.max(all_y2))

        # Plot converged data (solid lines)
        ax1.plot(x, y1_conv,
            color=colors[ind],
            linestyle='-',
            marker=markers[ind],
            markevery=5,
            markersize=7,
            linewidth=lwidths[ind],
            label=f"{labels[ind]}")

        ax2.plot(x, y2_conv,
            color=colors[ind],
            linestyle='-',
            marker=markers[ind],
            markevery=5,
            markersize=7,
            linewidth=lwidths[ind],
            label=f"{labels[ind]}")

        # Plot nfock=1 data (dotted lines)
        ax1.plot(x, y1_nf1,
            color=colors[ind],
            linestyle=':',
            marker=markers[ind],
            markevery=5,
            markersize=7,
            linewidth=lwidths[ind])

        ax2.plot(x, y2_nf1,
            color=colors[ind],
            linestyle=':',
            marker=markers[ind],
            markevery=5,
            markersize=7,
            linewidth=lwidths[ind])

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

    # Legend with two columns - one for Nfock=1 and one for converged
    # ------------------------------
    # First clear any existing legends
    ax1.get_legend().remove() if ax1.get_legend() else None
    ax2.get_legend().remove() if ax2.get_legend() else None

    # Create custom handles and labels for two-column legend
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    # First column: Nfock=1 (dotted lines)
    for ind, method in enumerate(methods):
        legend_handles.append(Line2D([0], [0], color=colors[ind], linestyle=':',
                            marker=markers[ind], markersize=7, linewidth=lwidths[ind]))
        legend_labels.append(labels[ind])

    # Second column: Converged (solid lines)
    for ind, method in enumerate(methods):
        legend_handles.append(Line2D([0], [0], color=colors[ind], linestyle='-',
                            marker=markers[ind], markersize=7, linewidth=lwidths[ind]))
        legend_labels.append(labels[ind])

    # Add column headers
    header1 = Line2D([], [], color='none')
    header2 = Line2D([], [], color='none')
    legend_handles = [header1] + legend_handles[:4] + [header2] + legend_handles[4:]
    #legend_labels = [r'\textbf{\underline{$\mathbf{N}_{\textrm{Fock }}\mathbf{=1}$}}'] + legend_labels[:4] + [r'\textbf{\underline{Converged}}'] + legend_labels[4:]
    legend_labels = [r'\textbf{\underline{Vacuum}}'] + legend_labels[:4] + [r'\textbf{\underline{Converged}}'] + legend_labels[4:]

    # Create the legend with 2 columns
    ax1.legend(legend_handles, legend_labels, fontsize=12,
               fancybox=False, framealpha=0.6, edgecolor="black",
               loc='upper left', ncol=2, columnspacing=1.4,
               handletextpad=0.6, borderpad=0.4)
    # Align y-axis labels
    # ------------------------------
    fig.align_ylabels([ax1, ax2])

    # Subplot label
    # ------------------------------
    ax1.text(-0.155, 1.05, r"\textbf{a)}",
        transform=ax1.transAxes, fontsize=16, va="top", ha="left")
    ax2.text(-0.155, 1.05, r"\textbf{b)}",
        transform=ax2.transAxes, fontsize=16, va="top", ha="left")

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
    # Combined figure
    # ---------------------
    fig_path = f"{basis}_{mol}/plots/fig3_en_vs_la_combined.pdf"
    plot_combined_figure(nfock_data, fig_path)


