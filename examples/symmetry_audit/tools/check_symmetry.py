#!/usr/bin/env python3
"""Discrete-symmetry checker for GRACE plane-surface output.

Given an XMF descriptor produced by GRACE's plane HDF5 output and a discrete
spatial symmetry, verify that paired cells carry parity-consistent values
to within a user-set tolerance.

Supported transforms (T : (x,y,z) -> ...):
  xmirror   : (-x, y, z)
  ymirror   : (x, -y, z)
  zmirror   : (x, y, -z)
  pirot_x   : (x, -y, -z)
  pirot_y   : (-x, y, -z)
  pirot_z   : (-x, -y, z)
  octant    : (-x, -y, -z)

For each cell at position r the script looks up the cell at T(r) and
compares variable values with the parity that variable inherits from the
transform.  Three variable kinds are recognised:
  - scalar (rho, press, alp, ...)        -> sign = +1 always
  - polar  3-vector (zvec, beta, ...)    -> components along flipped axes flip
  - axial  3-vector (Bvec)               -> opposite parity to polar
                                            (extra det(T) factor)
Unknown variables default to scalar with a warning.

Tensors are not currently handled component-by-component; treat them by
exporting each component as a scalar in the parfile and listing them
explicitly.

Usage
-----
    # Inside the numrel venv (so GRACEpy / vtk / scipy are on PYTHONPATH).
    source ~/.virtualenvs/numrel/bin/activate
    python examples/symmetry_audit/tools/check_symmetry.py path/to/desc_xy.xmf \\
        --transform xmirror \\
        --var rho --var press --var alp --var zvec \\
        --time all --tol 1e-13

Exit status: 0 if every (time, variable) is within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from grace_tools.vtk_reader_utils import grace_xmf_reader


# ---------------------------------------------------------------------------
# Parity bookkeeping
# ---------------------------------------------------------------------------

KIND_SCALAR = "scalar"
KIND_POLAR  = "polar"
KIND_AXIAL  = "axial"

DEFAULT_KIND = {
    # Hydro primitives
    "rho":             KIND_SCALAR,
    "press":           KIND_SCALAR,
    "eps":             KIND_SCALAR,
    "temperature":     KIND_SCALAR,
    "ye":              KIND_SCALAR,
    "entropy":         KIND_SCALAR,
    "c2p_err":         KIND_SCALAR,
    "c2p_dens_corr":   KIND_SCALAR,
    "Bdiv":            KIND_SCALAR,
    # ADM / Z4c scalars
    "alp":             KIND_SCALAR,
    "conf_fact":       KIND_SCALAR,
    "z4c_theta":       KIND_SCALAR,
    "z4c_Khat":        KIND_SCALAR,
    # Polar vectors (transform as r does)
    "zvec":            KIND_POLAR,
    "beta":            KIND_POLAR,
    "z4c_Gamma":       KIND_POLAR,
    "z4c_Bdriver":     KIND_POLAR,
    # Axial / pseudo-vectors (extra det(T) factor)
    "Bvec":            KIND_AXIAL,
}

# axis-flip mask per named transform: True means that axis is reversed.
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
    """Return T(coords) where coords is (N, 3)."""
    fx, fy, fz = TRANSFORM_FLIPS[transform]
    out = coords.copy()
    if fx: out[:, 0] = -out[:, 0]
    if fy: out[:, 1] = -out[:, 1]
    if fz: out[:, 2] = -out[:, 2]
    return out


def parity_factor(transform: str, kind: str) -> np.ndarray | float:
    """Per-component parity sign carried by a variable under `transform`.

    Returns a scalar +1.0 for KIND_SCALAR or a length-3 array for vectors.
    """
    fx, fy, fz = TRANSFORM_FLIPS[transform]
    if kind == KIND_SCALAR:
        return 1.0
    polar = np.array([
        -1.0 if fx else 1.0,
        -1.0 if fy else 1.0,
        -1.0 if fz else 1.0,
    ])
    if kind == KIND_POLAR:
        return polar
    if kind == KIND_AXIAL:
        # axial vectors flip with det(T) relative to polar
        det = (-1.0 if fx else 1.0) * (-1.0 if fy else 1.0) * (-1.0 if fz else 1.0)
        return det * polar
    raise ValueError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Partner-cell lookup
# ---------------------------------------------------------------------------

def build_partner_indices(coords: np.ndarray, transform: str,
                          match_tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    """For every cell, return the index of the cell whose centre equals T(r).

    Returns (partner_idx, valid_mask, worst_distance):
      partner_idx[i] is the index j such that coords[j] ≈ T(coords[i]).
      valid_mask[i]  is True if the lookup succeeded within match_tol.
    """
    tree = cKDTree(coords)
    T_coords = apply_transform(coords, transform)
    distances, partner_idx = tree.query(T_coords, k=1)
    valid = distances < match_tol
    return partner_idx, valid, float(distances.max())


# ---------------------------------------------------------------------------
# Per-variable check
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    time:      float
    var:       str
    kind:      str
    scale:     float
    max_abs:   float
    rel:       float
    n_cells:   int
    passed:    bool


def check_variable(values: np.ndarray, partner_idx: np.ndarray,
                   valid: np.ndarray, parity, tol: float,
                   time: float, var: str, kind: str) -> CheckResult:
    """Compute max-|values - parity * values[partner]| / scale on valid cells."""
    if values.ndim == 1:                   # scalar field
        delta = values - parity * values[partner_idx]
    else:                                  # (N, 3) vector field
        delta = values - parity[np.newaxis, :] * values[partner_idx, :]
    delta = delta[valid]
    raw   = values[valid] if values.ndim == 1 else values[valid, :]
    scale = float(np.max(np.abs(raw))) if raw.size else 0.0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    denom = scale if scale > 0.0 else 1.0   # values that are uniformly zero
    rel   = max_abs / denom
    return CheckResult(time=time, var=var, kind=kind,
                       scale=scale, max_abs=max_abs, rel=rel,
                       n_cells=int(np.sum(valid)),
                       passed=(rel <= tol))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_times(reader: grace_xmf_reader, selector: str) -> list[float]:
    times = list(reader.available_times())
    if selector == "all":
        return times
    if selector == "first":
        return times[:1]
    if selector == "last":
        return times[-1:]
    try:
        wanted = float(selector)
    except ValueError as exc:
        raise SystemExit(f"--time must be 'all', 'first', 'last', or a number; got {selector!r}") from exc
    # nearest match
    idx = int(np.argmin(np.abs(np.array(times) - wanted)))
    return [times[idx]]


def warn_if_transform_flips_slice_axis(coords: np.ndarray, transform: str,
                                       has_vector_kind: bool) -> None:
    """Emit a warning if `transform` flips an axis along which the dataset is
    constant (i.e., a 2D slice).  Scalar tests still work correctly in that
    case, but vector tests get the wrong parity for the flipped component.
    """
    extents = coords.max(axis=0) - coords.min(axis=0)
    typical = float(np.max(extents))
    if typical <= 0.0:
        return
    flips = TRANSFORM_FLIPS[transform]
    compat_for = {
        0: "ymirror, zmirror, pirot_x",
        1: "xmirror, zmirror, pirot_y",
        2: "xmirror, ymirror, pirot_z",
    }
    axis_name = ("x", "y", "z")
    for axis_idx in range(3):
        if extents[axis_idx] >= 1e-6 * typical:
            continue  # axis varies — full 3D for this axis
        if not flips[axis_idx]:
            continue  # transform doesn't touch this constant axis — fine
        const_val = float(coords[:, axis_idx][0])
        msg = (f"WARNING: dataset is a {axis_name[axis_idx]}={const_val:.3g} slice, "
               f"but --transform {transform!r} flips the {axis_name[axis_idx]}-axis. "
               f"Scalar tests are still correct; vector tests will give WRONG parity "
               f"for the {axis_name[axis_idx]}-component. "
               f"For unambiguous vector tests on this slice, use one of: "
               f"{compat_for[axis_idx]}.")
        if has_vector_kind:
            print(msg, file=sys.stderr)
        # If no vector vars were requested, silence — the test is unambiguously
        # well-defined for scalars under any transform.


def resolve_kind(var: str, overrides: dict[str, str]) -> str:
    if var in overrides:
        return overrides[var]
    if var in DEFAULT_KIND:
        return DEFAULT_KIND[var]
    print(f"  [warn] {var}: kind unknown — treating as scalar. "
          f"Pass --kind {var}=polar|axial to override.", file=sys.stderr)
    return KIND_SCALAR


def parse_kind_overrides(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--kind expects var=kind, got {item!r}")
        k, v = item.split("=", 1)
        if v not in (KIND_SCALAR, KIND_POLAR, KIND_AXIAL):
            raise SystemExit(f"--kind: unknown kind {v!r}; expected scalar|polar|axial")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("descriptor", help="path to the XMF descriptor file")
    p.add_argument("--transform", required=True,
                   choices=sorted(TRANSFORM_FLIPS.keys()),
                   help="discrete symmetry to test")
    p.add_argument("--var", action="append", required=True,
                   help="variable to check (may be repeated)")
    p.add_argument("--kind", action="append", default=[],
                   help="override variable kind: 'name=polar' / 'name=axial' / 'name=scalar'")
    p.add_argument("--time", default="all",
                   help="'all', 'first', 'last', or a time value (nearest match)")
    p.add_argument("--tol", type=float, default=1e-13,
                   help="relative tolerance threshold for pass/fail (default 1e-13)")
    p.add_argument("--match-tol", type=float, default=None,
                   help="absolute physical-distance tolerance for partner-cell lookup "
                        "(default: auto = 1%% of dx_min)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="suppress per-(time, variable) lines; print only the summary "
                        "and any failures")
    args = p.parse_args(argv)

    overrides = parse_kind_overrides(args.kind)
    reader    = grace_xmf_reader(args.descriptor)
    times     = resolve_times(reader, args.time)

    failures: list[CheckResult] = []
    all_results: list[CheckResult] = []
    var_kinds = {v: resolve_kind(v, overrides) for v in args.var}
    has_vector = any(k in (KIND_POLAR, KIND_AXIAL) for k in var_kinds.values())

    geometry_warned = False
    for ti, t in enumerate(times):
        reader.set_time(t)
        # coords are the same for every variable at this time slice; only need to
        # build the KDTree + partner map once.
        coords, _ = reader.get_var(args.var[0], time=t, convert_to_numpy=True)
        coords = np.asarray(coords, dtype=np.float64)

        if not geometry_warned:
            warn_if_transform_flips_slice_axis(coords, args.transform, has_vector)
            geometry_warned = True

        # auto match-tol: take 1% of the typical NN distance.
        if args.match_tol is None:
            tree = cKDTree(coords)
            dx_nn, _ = tree.query(coords, k=2)
            dx_typ = float(np.median(dx_nn[:, 1]))
            match_tol = 0.01 * dx_typ
        else:
            match_tol = args.match_tol

        partner_idx, valid, worst = build_partner_indices(coords, args.transform, match_tol)
        n_unpaired = int(np.sum(~valid))
        if n_unpaired:
            print(f"[t={t:.6g}] WARNING: {n_unpaired}/{len(valid)} cells have no "
                  f"symmetry partner within match_tol={match_tol:.3g} "
                  f"(worst dist {worst:.3g}). These are skipped.",
                  file=sys.stderr)

        for var in args.var:
            kind = var_kinds[var]
            _, values = reader.get_var(var, time=t, convert_to_numpy=True)
            values = np.asarray(values, dtype=np.float64)
            parity = parity_factor(args.transform, kind)
            res = check_variable(values, partner_idx, valid, parity, args.tol,
                                 time=t, var=var, kind=kind)
            all_results.append(res)
            if not res.passed:
                failures.append(res)
            if (not args.quiet) or (not res.passed):
                tag = "PASS" if res.passed else "FAIL"
                print(f"  [t={res.time:.6g}] {tag} {res.var:<14s} ({res.kind}) "
                      f"max|Δ|={res.max_abs:.3e}  scale={res.scale:.3e}  "
                      f"rel={res.rel:.3e}  N={res.n_cells}")

    n_total = len(all_results)
    n_pass  = n_total - len(failures)
    print(f"\nSummary: {n_pass}/{n_total} (time, variable) pairs within tol={args.tol:g}")
    if failures:
        print("First failures:")
        for r in failures[:5]:
            print(f"  t={r.time:.6g}  {r.var}  rel={r.rel:.3e}  (tol={args.tol:g})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
