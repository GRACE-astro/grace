#!/usr/bin/env python3
"""Plot the per-cell symmetry-violation map for a GRACE plane output.

For each cell in a 2D slice, computes the symmetry residual against its
parity partner under a chosen discrete transform and renders it as a
scatter heatmap.  Useful for localising WHERE in the domain the asymmetry
lives — interior vs surface, scattered vs clustered at AMR boundaries.

Usage
-----
    source ~/.virtualenvs/numrel/bin/activate
    python examples/symmetry_audit/tools/plot_asymmetry_map.py path/to/desc_xy.xmf \\
        --transform pirot_z --var zvec --component x \\
        --time last --log --contour rho \\
        --out asym_map.png

Exit status: 0 on success.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree

from grace_tools.vtk_reader_utils import grace_xmf_reader


TRANSFORM_FLIPS = {
    "xmirror": (True,  False, False),
    "ymirror": (False, True,  False),
    "zmirror": (False, False, True ),
    "pirot_x": (False, True,  True ),
    "pirot_y": (True,  False, True ),
    "pirot_z": (True,  True,  False),
    "octant":  (True,  True,  True ),
}

DEFAULT_KIND = {
    "rho": "scalar", "press": "scalar", "eps": "scalar", "alp": "scalar",
    "temperature": "scalar", "ye": "scalar", "entropy": "scalar",
    "c2p_err": "scalar", "c2p_dens_corr": "scalar", "Bdiv": "scalar",
    "conf_fact": "scalar", "z4c_theta": "scalar", "z4c_Khat": "scalar",
    "zvec": "polar", "beta": "polar", "z4c_Gamma": "polar",
    "z4c_Bdriver": "polar",
    "Bvec": "axial",
}


def apply_transform(coords: np.ndarray, transform: str) -> np.ndarray:
    fx, fy, fz = TRANSFORM_FLIPS[transform]
    out = coords.copy()
    if fx: out[:, 0] = -out[:, 0]
    if fy: out[:, 1] = -out[:, 1]
    if fz: out[:, 2] = -out[:, 2]
    return out


def parity_factor(transform: str, kind: str):
    fx, fy, fz = TRANSFORM_FLIPS[transform]
    if kind == "scalar":
        return 1.0
    polar = np.array([-1.0 if fx else 1.0,
                      -1.0 if fy else 1.0,
                      -1.0 if fz else 1.0])
    if kind == "polar":
        return polar
    if kind == "axial":
        det = ((-1.0 if fx else 1.0)
               * (-1.0 if fy else 1.0)
               * (-1.0 if fz else 1.0))
        return det * polar
    raise ValueError(f"unknown kind {kind!r}")


def resolve_time(reader, selector: str) -> float:
    times = list(reader.available_times())
    if selector == "last":
        return times[-1]
    if selector == "first":
        return times[0]
    wanted = float(selector)
    idx = int(np.argmin(np.abs(np.array(times) - wanted)))
    return times[idx]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("descriptor", help="path to the XMF descriptor")
    p.add_argument("--transform", required=True,
                   choices=sorted(TRANSFORM_FLIPS.keys()))
    p.add_argument("--var", required=True, help="variable to map")
    p.add_argument("--kind", default=None,
                   help="override variable kind (scalar|polar|axial)")
    p.add_argument("--component", default="mag",
                   choices=["x", "y", "z", "mag"],
                   help="for vector variables: which component to plot "
                        "(default: vector magnitude)")
    p.add_argument("--time", default="last",
                   help="'last', 'first', or a numeric time value (nearest match)")
    p.add_argument("--log", action="store_true",
                   help="log-scale colour bar (zeros plotted as transparent)")
    p.add_argument("--contour", default=None,
                   help="overlay tricontour lines of this scalar variable "
                        "(e.g. rho) to show the stellar surface")
    p.add_argument("--contour-levels", type=int, default=8,
                   help="number of contour levels (default 8)")
    p.add_argument("--marker-size", type=float, default=1.5,
                   help="scatter-marker point size")
    p.add_argument("--out", default=None,
                   help="output PNG path; if omitted, shows interactively")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   help="restrict plot to this in-plane x-range")
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   help="restrict plot to this in-plane y-range")
    p.add_argument("--no-mark-max", action="store_true",
                   help="do NOT highlight the max-|Δ| cell on the plot")
    p.add_argument("--top-k", type=int, default=1,
                   help="highlight the top-K worst cells (default 1)")
    args = p.parse_args(argv)

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.tri as mtri

    reader = grace_xmf_reader(args.descriptor)
    t      = resolve_time(reader, args.time)
    reader.set_time(t)

    coords, values = reader.get_var(args.var, time=t, convert_to_numpy=True)
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    kind   = args.kind or DEFAULT_KIND.get(args.var, "scalar")
    parity = parity_factor(args.transform, kind)

    # Partner lookup (auto match-tol = 1% of typical NN distance).
    tree = cKDTree(coords)
    dx_nn, _ = tree.query(coords, k=2)
    match_tol = 0.01 * float(np.median(dx_nn[:, 1]))
    distances, partner_idx = tree.query(apply_transform(coords, args.transform), k=1)
    valid = distances < match_tol
    if not valid.all():
        print(f"WARNING: {int(np.sum(~valid))}/{len(valid)} cells unpaired "
              f"(match_tol={match_tol:.3g}); dropping from plot.",
              file=sys.stderr)

    if values.ndim == 1:
        delta = values - parity * values[partner_idx]
        plot_data = np.abs(delta)
        scale = float(np.abs(values).max())
        comp_label = ""
    else:
        delta = values - parity[np.newaxis, :] * values[partner_idx, :]
        scale = float(np.linalg.norm(values, axis=1).max())
        if args.component == "mag":
            plot_data = np.linalg.norm(delta, axis=1)
            comp_label = " (|Δ|)"
        else:
            ci = {"x": 0, "y": 1, "z": 2}[args.component]
            plot_data = np.abs(delta[:, ci])
            comp_label = f" (|Δ_{args.component}|)"

    # Detect which axis is the slice-constant one.
    extents     = coords.max(axis=0) - coords.min(axis=0)
    const_axis  = int(np.argmin(extents))
    plane_axes  = [a for a in range(3) if a != const_axis]
    axis_names  = ("x", "y", "z")

    xp = coords[valid, plane_axes[0]]
    yp = coords[valid, plane_axes[1]]
    cp = plot_data[valid]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    if args.log:
        cp_for_color = np.where(cp > 0, cp, np.nan)
        finite = cp_for_color[np.isfinite(cp_for_color)]
        if finite.size:
            vmin = max(finite.min(), 1e-300)
            vmax = finite.max()
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None
        cp_for_color = cp
    print(f"Max {np.nanmax(plot_data[valid])} Min {np.nanmin(plot_data[valid])}")
    sc = ax.scatter(xp, yp, c=cp_for_color, s=args.marker_size,
                    cmap="viridis", norm=norm, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(f"|Δ|{comp_label}  ({args.var})")

    # Optional underlay contours of e.g. rho to show stellar surface.
    if args.contour:
        _, cvals = reader.get_var(args.contour, time=t, convert_to_numpy=True)
        cvals = np.asarray(cvals, dtype=np.float64)
        if cvals.ndim > 1:
            cvals = np.linalg.norm(cvals, axis=1)
        try:
            triang = mtri.Triangulation(coords[:, plane_axes[0]],
                                        coords[:, plane_axes[1]])
            ax.tricontour(triang, cvals, levels=args.contour_levels,
                          colors="white", alpha=0.4, linewidths=0.5)
        except Exception as exc:
            print(f"WARNING: could not draw contours of {args.contour}: {exc}",
                  file=sys.stderr)

    # Symmetry-plane reference lines.
    fx, fy, fz = TRANSFORM_FLIPS[args.transform]
    axis_flips = (fx, fy, fz)
    for ax_idx in plane_axes:
        if axis_flips[ax_idx]:
            if ax_idx == plane_axes[0]:
                ax.axvline(0.0, color="red", lw=0.5, alpha=0.5, ls="--")
            else:
                ax.axhline(0.0, color="red", lw=0.5, alpha=0.5, ls="--")

    # Highlight the top-K cells carrying the worst symmetry residual.
    if not args.no_mark_max and cp.size:
        k = max(1, int(args.top_k))
        order = np.argsort(cp)[::-1][:k]   # indices into valid-filtered cp / xp / yp
        ax.scatter(xp[order], yp[order],
                   s=120, facecolors="none", edgecolors="red",
                   linewidths=1.6, zorder=10)
        # Label only the first (true max), with a callout showing value + position.
        i0 = order[0]
        u, v = xp[i0], yp[i0]
        ax.annotate(
            f"max |Δ| = {cp[i0]:.3e}\n"
            f"({axis_names[plane_axes[0]]}, {axis_names[plane_axes[1]]}) = "
            f"({u:.3f}, {v:.3f})",
            xy=(u, v),
            xytext=(12, 12), textcoords="offset points",
            fontsize=8, color="red",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", ec="red", lw=0.8, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            zorder=11,
        )

    if args.xlim is not None: ax.set_xlim(*args.xlim)
    if args.ylim is not None: ax.set_ylim(*args.ylim)
    ax.set_xlabel(axis_names[plane_axes[0]])
    ax.set_ylabel(axis_names[plane_axes[1]])
    ax.set_aspect("equal")
    ax.set_title(
        f"Symmetry residual: {args.var}{comp_label} under {args.transform}"
        f" at t = {t:.4g}\n"
        f"max|Δ| = {plot_data.max():.3e}    scale = {scale:.3e}    "
        f"rel = {plot_data.max() / (scale if scale > 0 else 1.0):.3e}"
    )

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
