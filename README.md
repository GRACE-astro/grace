# GRACE

[![CI](https://github.com/GRACE-astro/grace/actions/workflows/ci.yml/badge.svg)](https://github.com/GRACE-astro/grace/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL_3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://grace-astro.github.io/grace/)

**GRACE** (General Relativistic Astrophysics Code for Exascale) is a
GPU-portable general-relativistic magnetohydrodynamics (GRMHD) evolution
framework, built on [Kokkos](https://github.com/kokkos/kokkos) for
performance portability and [p4est](https://www.p4est.org/) for block-based
adaptive mesh refinement. It targets BNS / BBH / accretion problems on
modern HPC architectures, from laptop OpenMP through CUDA / HIP exascale
systems.

## Features

- **GRMHD evolution** with HLLE / LLF Riemann solvers, WENO-Z and
  slope-limited reconstruction, Gardiner-Stone or UCT constrained
  transport for the magnetic field, first-order flux correction (FOFC),
  and Kastaun-based primitive recovery.
- **Z4c spacetime evolution** with codegen-emitted finite-difference
  operators at orders 2 / 4 / 6, Sommerfeld outer boundary, and
  Psi4 wave-extraction diagnostics.
- **Block-structured AMR** via p4est, with custom ghost-zone exchange,
  divergence-free prolongation, conservative restriction, and face-flux /
  edge-EMF refluxing.
- **Deterministic MPI reductions** for bit-exact reproducibility across
  rank counts.
- **Hybrid + tabulated EOS** with a cold-EOS table layer.

## Quickstart

```bash
git clone --recurse-submodules --shallow-submodules \
          https://github.com/GRACE-astro/grace.git
cd grace
cmake -B build -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGRACE_USE_BUNDLED_DEPS=ON \
      -DGRACE_ENABLE_OMP=ON \
      -DGRACE_METRIC_EVOL=COWLING
cmake --build build -j
./build/grace --grace-parfile examples/cowling_grmhd/shocktubes/balsara1.yaml
```

For GPU builds, see the
[building guide](https://grace-astro.github.io/grace/code_building_guide/).

## Documentation

Full documentation is hosted at
[grace-astro.github.io/grace](https://grace-astro.github.io/grace/) and
covers installation, configuration, examples, the Python analysis
interface, and the contribution workflow.

## Contributing

GRACE accepts contributions via pull requests from forks. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`doc/contributing/`](doc/contributing/index.rst) for the workflow,
coding standards, and review policy.

## Citing GRACE

If you use GRACE in academic work, please cite the code paper (citation
details will be added once the paper appears on arXiv). For now, please
reference the repository URL.

## License

GRACE is distributed under the GNU General Public License v3.0
(see [`LICENSE.md`](LICENSE.md)). Authors and contributors are listed
in [`AUTHORS`](AUTHORS).
