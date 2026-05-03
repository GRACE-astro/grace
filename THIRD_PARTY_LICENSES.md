# Third-Party Licences

GRACE is licensed under **GPL-3.0-or-later**. This file catalogues the
third-party software that GRACE links against, embeds, or bundles for its
documentation, together with the licence each component is distributed under.
Every entry below has been checked for compatibility with GPL-3; none of
these components require GRACE to be distributed under any other licence.

Two conventions throughout:

- **"Linked"** means GRACE calls into the library at build/runtime, but
  the library source is *not* vendored into this repository. Users install
  it separately (via their package manager, spack, CMake `find_package`, etc.).
- **"Embedded"** means source code from the upstream project lives
  inside this repository, either verbatim or with local modifications.

---

## Linked build/runtime libraries

| Component | Licence | Scope | Upstream |
|---|---|---|---|
| **Kokkos** | Apache-2.0 with LLVM Exception | Performance-portable parallel execution and memory hierarchies | https://github.com/kokkos/kokkos |
| **HDF5** | HDF Group BSD-style licence | Checkpoint I/O and EOS-table ingest | https://github.com/HDFGroup/hdf5 |
| **MPI** | Implementation-dependent (OpenMPI: BSD-3, MPICH: MPICH licence) | Distributed-memory parallelism (interface only; the implementation is supplied by the user's MPI stack) | https://www.mpi-forum.org |
| **p4est** | GPL-2-or-later | Forest-of-octrees AMR backbone | https://github.com/cburstedde/p4est |
| **yaml-cpp** | MIT | YAML configuration parsing | https://github.com/jbeder/yaml-cpp |
| **spdlog** | MIT | Logging (bundles `fmt`, also MIT) | https://github.com/gabime/spdlog |
| **Catch2** | Boost Software Licence 1.0 | Unit testing (test targets only; not in shipped binary) | https://github.com/catchorg/Catch2 |
| **LORENE** | GPL-2-or-later | Initial-data import (spectral GR solutions) | https://lorene.obspm.fr |
| **KADATH / FUKA** | GPL (confirm version with upstream) | Initial-data import (FUKA compact-object configurations) | https://kadath.obspm.fr (KADATH); FUKA public fork |

**Compatibility note.** GPL-3 can incorporate GPL-2-or-later code (the
`+` on those licences permits upgrade to GPL-3), MIT, BSD-style (HDF5,
MPI implementations), Apache-2.0 (including the Kokkos LLVM-exception
variant), and Boost Software Licence 1.0 without issue. The Kokkos
Apache-2 → GPL-3 compatibility in particular relies on a deliberate
compatibility provision added in GPL-3 (GPL-2 did not have it).

---

## Embedded code

| Component | Licence | Scope | Location in this repo | Upstream |
|---|---|---|---|---|
| **Brent rootfinder** (C++ port by John Burkardt of Richard Brent's FORTRAN77 algorithm) | GNU LGPL | `brent()` function in the root-finding utility header | `include/grace/utils/rootfinding.hh` (attribution block inline at the function) | https://people.sc.fsu.edu/~jburkardt/cpp_src/brent/brent.html |

---

## Documentation theme assets

GRACE's documentation is built with Doxygen and Sphinx and re-uses
theme assets from several upstream projects. Each file carries its own
SPDX header; the summary below is for convenience.

| Component | Licence | Location | Upstream |
|---|---|---|---|
| **doxygen-awesome-css** by *jothepro* | MIT | `doc/_doxygen/doxygen-awesome*.{css,js}` | https://github.com/jothepro/doxygen-awesome-css |
| **Godot community docs theme** (Juan Linietsky, Ariel Manzur et al.), with later polish by **Teslabs Engineering S.L.** and further tweaks inherited from the **Zephyr** project | CC-BY-3.0 | `doc/_static/css/light.css`, `doc/_static/css/dark.css`, `doc/_static/css/custom.css`, `doc/_static/js/custom.js` | https://github.com/zephyrproject-rtos/zephyr (docs subtree); originally from the Godot Engine documentation project |
| **Zephyr Google Programmable Search styles** by *Benjamin Cabé* | Apache-2.0 | `doc/_static/css/gcs.css` | https://github.com/zephyrproject-rtos/zephyr |
| **Sphinx search scoring** (Sphinx team + Intel) | Apache-2.0 | `doc/_static/js/scorer.js` | https://www.sphinx-doc.org |
| **GRACE Doxygen-Awesome branding** (Carlo Musolino) | CC-BY-3.0 | `doc/_doxygen/doxygen-awesome-grace.css` | This repository |

Minor colour/visual-identity modifications to the Godot-lineage CSS files
have been made for GRACE; those modifications are noted in the individual
file headers and are themselves distributed under CC-BY-3.0 per the
upstream licence.

---

## Notes for redistributors

- This file exists as a convenience index. The authoritative licence
  information for any given file is the SPDX header in the file itself,
  or the upstream project's own `LICENSE` / `COPYING` file.
- If you package GRACE into a binary distribution (container image,
  release tarball, etc.) you must include the notices required by each
  of the above licences. MIT and BSD-style licences require preserving
  their copyright notice; Apache-2.0 requires distributing a copy of
  the licence and any `NOTICE` file; GPL-2+ / GPL-3 / LGPL require
  providing (or offering) corresponding source; CC-BY-3.0 requires
  attribution.
- If you add a new linked or embedded dependency, please add it here
  and confirm its licence is GPL-3 compatible before merging.
