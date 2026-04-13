#!/usr/bin/env python3
"""Plot local vs MPI B-flux conservation diagnostic for a given refinement level."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_bflux_dat(fname):
    """Load a Bflux .dat file, return (time, data_dict).

    data_dict maps (level, dir, kind) -> array, where kind is 'loc' or 'mpi'.
    """
    raw = np.loadtxt(fname, comments="#")
    iteration = raw[:, 0]
    time = raw[:, 1]
    cols = raw[:, 2:]
    # columns are ordered: L0_x_loc L0_x_mpi L0_y_loc L0_y_mpi L0_z_loc L0_z_mpi L1_x_loc ...
    # so for level l, dir d: col index = (l * 3 + d) * 2 for loc, +1 for mpi
    data = {}
    dirs = ["x", "y", "z"]
    for l in range(16):
        for di, d in enumerate(dirs):
            base = (l * 3 + di) * 2
            if base + 1 < cols.shape[1]:
                data[(l, d, "loc")] = cols[:, base]
                data[(l, d, "mpi")] = cols[:, base + 1]
    return time, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to Bflux_hanging_max.dat or similar")
    parser.add_argument("-l", "--level", type=int, required=True, help="Refinement level to plot")
    parser.add_argument("--integral", action="store_true",
                        help="If set, also look for the _integral file and plot it")
    parser.add_argument("-o", "--output", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    time, data = load_bflux_dat(args.file)
    L = args.level
    dirs = ["x", "y", "z"]
    colors = {"x": "C0", "y": "C1", "z": "C2"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for d in dirs:
        loc = data.get((L, d, "loc"))
        mpi = data.get((L, d, "mpi"))
        if loc is None:
            continue
        has_signal = np.any(loc > 0) or np.any(mpi > 0)
        if not has_signal:
            continue
        ax.semilogy(time, np.maximum(loc, 1e-300), color=colors[d], linestyle="-",
                    label=f"dir={d} local", alpha=0.8)
        ax.semilogy(time, np.maximum(mpi, 1e-300), color=colors[d], linestyle="--",
                    label=f"dir={d} MPI", alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("max |dB|")
    ax.set_title(f"B-flux mismatch at L{L}/L{L+1} boundary — solid=local, dashed=MPI")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
