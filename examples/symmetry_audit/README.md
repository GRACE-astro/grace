# GRACE symmetry-preservation audit

A small suite of test problems used to verify that GRACE preserves
discrete spatial symmetries to bit-level (machine epsilon) over many
timesteps on a uniform Cartesian grid.

The long-term goal is SACRA/Kiuchi-class bit-exact symmetry preservation
through full BNS merger evolutions; this directory holds the Tier-1
unigrid harness that's the prerequisite for everything that follows.

## What's here

| File                         | What it tests              | Symmetry      |
|------------------------------|-----------------------------|----------------|
| `khi_hydro_pirot.yaml`       | Hydro pipeline              | π-rotation about origin |
| `blast_xmirror.yaml`         | MHD pipeline                | x-mirror across y-axis  |

The two ICs together cover:
- recon, hydro Riemann, C2P-hydro, RK, periodic BCs (KHI, σ_pol = 0)
- CT/EMF, magnetic Riemann branches, C2P-MHD (cylindrical blast)

π-rotation is *proper* (polar and axial vectors transform identically);
x-mirror is *improper* (polar and axial split). Forcing the checker to
handle both up front means the polar/axial bookkeeping is correct
for the BNS orbital-plane mirror later.

## Running the audit

1. Build GRACE with `-ffp-contract=off -fno-fast-math -fno-associative-math`
   (or equivalent for your compiler).  This kills the biggest hidden source
   of FP-asymmetry (FMA contraction).
2. Run one of the YAMLs:
   ```
   ./grace examples/symmetry_audit/khi_hydro_pirot.yaml
   ./grace examples/symmetry_audit/blast_xmirror.yaml
   ```
3. Verify with `examples/symmetry_audit/tools/check_symmetry.py`:
   ```
   # Hydro test (KHI σ=0, pi-rot, polar velocity field):
   python examples/symmetry_audit/tools/check_symmetry.py <run-dir>/desc_xy.xmf \
       --transform pirot --var zvec --time all --tol 1e-13

   # MHD test (blast, x-mirror, both polar v and axial B):
   python examples/symmetry_audit/tools/check_symmetry.py <run-dir>/desc_xy.xmf \
       --transform xmirror --var zvec --var Bvec --time all --tol 1e-13
   ```
   Exit code 0 = pass, 1 = fail.

## What "pass" means

`max(|Δ| / scale) ≤ tol` per (variable, time slice), where Δ is the
difference between cell `i` and the parity-flipped value at its symmetry
partner cell `j`. With `tol = 1e-13`, "pass" is roughly machine-ε on
the maximum-norm value of the variable.

## What to do when a test fails

1. Identify the *first* time slice where the symmetry breaks. The
   earlier the break, the closer to the IC and the easier to diagnose.
2. Re-run with `output_extra_quantities` and dump per-timestep at high
   cadence near the break.
3. Bisect kernel-by-kernel: dump state before/after each major call
   (recon, Riemann, EMF, cell update, C2P), assert symmetry on each
   pickle, find the first kernel that introduces the violation.
4. Rewrite the offending expressions to be parity-equivariant under
   the relevant symmetry.  Common culprits: cancellation-sensitive sums
   that erase IEEE +0/-0 (use a quadratic-form rewrite), and left-to-
   right reductions whose summation order is not partner-symmetric
   (pair-symmetric contractors fix these).  See e.g. the wavespeed
   discussion in `grmhd_helpers.hh` and the pair-symmetric
   prolong/restrict in `pr_helpers.hh`.

## Adding new audit tests

Tier-2 candidates (enable once Tier-1 is solid):
- Orszag-Tang vortex (multiple symmetries simultaneously: π-rot + diagonal mirrors)
- Doubled Brio-Wu shock tube (mirror across contact)
- Magnetized rotor π-rot variant
- Loop advection with a stream-function vortex flow (π-rot symmetric)

Each new IC should:
- Live as `include/grace/physics/id/<name>.hh`
- Be dispatched from `src/physics/grmhd.cpp`
- Have an example YAML here, with a clear comment specifying the
  expected symmetry and the `check_symmetry.py` invocation that
  verifies it.
