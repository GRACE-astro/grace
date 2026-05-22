#!/usr/bin/env python3
"""Partner-equivariance check for a symmetric 2-tensor field whose six
independent components are exported as separate scalar variables (e.g.
A_tilde[0,0], A_tilde[0,1], ..., A_tilde[2,2]).

For a symmetric 2-tensor T_{ij}, each component carries its own parity
under a discrete-symmetry transform:
    parity(T_{ij}) = parity(x_i) × parity(x_j),
with per-axis parity +1 if the transform preserves that axis and -1 if it
flips it.  This script reads each component as a scalar field, applies
its parity factor, runs the partner-pair check, and reports the worst
component per time slice (mirroring check_symmetry.py's output format).

Usage
-----
    source ~/.virtualenvs/numrel/bin/activate
    python examples/symmetry_audit/tools/check_tensor_symmetry.py \\
        --base 'A_tilde' --transform pirot_z --time all \\
        ./plots/descriptors/xy_plane_descriptor.xmf

The variable names are constructed as f"{base}[{i},{j}]" for i ≤ j in
{0,1,2}.  Override the format with --name-pattern '{base}_{i}{j}' if your
output uses a different naming convention.

Exit status: 0 if every time slice passes tol, 1 otherwise.

Author: carlo.musolino@aei.mpg.de
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree

from grace_tools.vtk_reader_utils import grace_xmf_reader


# Upper-triangle index pairs (six independent components of a symmetric
# 3x3 tensor) and their plain-text labels.
VOIGT_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
VOIGT_LABEL = ["xx", "xy", "xz", "yy", "yz", "zz"]

# (fx, fy, fz) per transform: True = that axis flips under the transform.
TRANSFORM_FLIPS = {
    "xmirror": (True,  False, False),
    "ymirror": (False, True,  False),
    "zmirror": (False, False, True ),
    "pirot_x": (False, True,  True ),
    "pirot_y": (True,  False, True ),
    "pirot_z": (True,  True,  False),
    "octant":  (True,  True,  True ),
}


def apply_transform(coords: np.ndarray, transform: str) -> np.ndarray:
    fx, fy, fz = TRANSFORM_FLIPS[transform]
    out = coords.copy()
    if fx: out[:, 0] = -out[:, 0]
    if fy: out[:, 1] = -out[:, 1]
    if fz: out[:, 2] = -out[:, 2]
    return out


def component_parity(transform: str, i: int, j: int) -> float:
    flips = TRANSFORM_FLIPS[transform]
    par_i = -1.0 if flips[i] else 1.0
    par_j = -1.0 if flips[j] else 1.0
    return par_i * par_j


def resolve_times(reader, selector: str) -> list:
    times = list(reader.available_times())
    if selector == "all":
        return times
    if selector == "last":
        return [times[-1]]
    if selector == "first":
        return [times[0]]
    wanted = float(selector)
    idx = int(np.argmin(np.abs(np.array(times) - wanted)))
    return [times[idx]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("descriptor")
    p.add_argument("--base", required=True,
                   help="component-name prefix, e.g. 'A_tilde'")
    p.add_argument("--name-pattern", default="{base}[{i},{j}]",
                   help="Python format string with {base},{i},{j} (default: "
                        "'{base}[{i},{j}]')")
    p.add_argument("--transform", required=True,
                   choices=sorted(TRANSFORM_FLIPS.keys()))
    p.add_argument("--time", default="all",
                   help="'all', 'first', 'last', or a numeric time")
    p.add_argument("--tol", type=float, default=1e-13)
    p.add_argument("--quiet", "-q", action="store_true",
                   help="only print the final summary")
    p.add_argument("--per-component", action="store_true",
                   help="print one line per component (default: worst only)")
    args = p.parse_args(argv)

    reader = grace_xmf_reader(args.descriptor)
    times = resolve_times(reader, args.time)

    # Pre-resolve the six variable names.
    comp_names = [
        args.name_pattern.format(base=args.base, i=i, j=j)
        for (i, j) in VOIGT_PAIRS
    ]

    all_pass = True
    n_pass = 0
    n_total = 0
    first_failures = []

    for t in times:
        reader.set_time(t)

        # Read all six components; expect each to come back as a scalar
        # field over the same coordinate set.
        coords_ref = None
        comp_values = []
        for name in comp_names:
            try:
                coords, vals = reader.get_var(name, time=t,
                                              convert_to_numpy=True)
            except Exception as exc:
                print(f"ERROR: could not read component '{name}' at t={t}: "
                      f"{exc}", file=sys.stderr)
                return 2
            coords = np.asarray(coords, dtype=np.float64)
            vals   = np.asarray(vals,   dtype=np.float64).reshape(-1)
            if coords_ref is None:
                coords_ref = coords
            comp_values.append(vals)

        # Partner lookup, done once per time.
        tree = cKDTree(coords_ref)
        dx_nn, _ = tree.query(coords_ref, k=2)
        match_tol = 0.01 * float(np.median(dx_nn[:, 1]))
        distances, partner_idx = tree.query(
            apply_transform(coords_ref, args.transform), k=1)
        valid = distances < match_tol

        # Per-component partner-equivariance check.
        worst = {"k": -1, "abs": 0.0, "scale": 0.0, "rel": 0.0}
        per_comp = []
        for k, (i, j) in enumerate(VOIGT_PAIRS):
            vals = comp_values[k]
            parity = component_parity(args.transform, i, j)
            delta = vals - parity * vals[partner_idx]
            abs_delta = np.abs(delta)[valid]
            cmax = float(abs_delta.max()) if abs_delta.size else 0.0
            cscale = float(np.abs(vals).max())
            crel = cmax / cscale if cscale > 0 else 0.0
            per_comp.append((k, cmax, cscale, crel, parity))
            if crel > worst["rel"]:
                worst = {"k": k, "abs": cmax, "scale": cscale, "rel": crel}

        status = "PASS" if worst["rel"] < args.tol else "FAIL"
        n_total += 1
        if status == "PASS":
            n_pass += 1
        else:
            all_pass = False
            if len(first_failures) < 4:
                first_failures.append(
                    (t, worst["rel"], VOIGT_LABEL[worst["k"]]))

        if not args.quiet:
            wc = VOIGT_LABEL[worst["k"]] if worst["k"] >= 0 else "—"
            print(f"  [t={t}] {status} {args.base:14s} (sym2,worst={wc})  "
                  f"max|Δ|={worst['abs']:.3e}  scale={worst['scale']:.3e}  "
                  f"rel={worst['rel']:.3e}  N={int(valid.sum())}")
            if args.per_component:
                for k, cmax, cscale, crel, par in per_comp:
                    print(f"      {VOIGT_LABEL[k]:>2s} (par={int(par):+d})  "
                          f"|Δ|={cmax:.3e}  scale={cscale:.3e}  rel={crel:.3e}")

    print(f"\nSummary: {n_pass}/{n_total} times within tol={args.tol}")
    if first_failures:
        print("First failures:")
        for t, rel, comp in first_failures:
            print(f"  t={t}  {args.base}[{comp}]  rel={rel:.3e}  "
                  f"(tol={args.tol})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
