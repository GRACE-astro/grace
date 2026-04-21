.. _gracepy-userguide:

GRACEpy Library
================

The `GRACEpy <https://github.com/GRACE-astro/GRACEpy>`_ library is a set of companion Python tools for the GRACE code.

It includes:

- **Data analysis**: readers for volume, plane, scalar and gravitational wave output, unified detector views, and a high-level simulation interface
- **Physics utilities**: physical constants, unit systems, gravitational wave analysis, Kerr-Schild coordinate transformations, and equation of state table handling
- **Code generation**: SymPy-based utilities to generate optimized C99 code for GRACE kernels
- **Simulation management**: ``simpilot``, a tool to create, submit and monitor GRACE simulations on HPC clusters
- **Command-line tools**: descriptor creation, scalar export, grid inspection, source archival

Installation
**************

Download the code from the `repository <https://github.com/GRACE-astro/GRACEpy>`_ and, preferably within a virtual environment, run

.. code-block:: bash

    pip install -e ./

This installs all Python dependencies and registers the command-line entry points.

.. toctree::
   :maxdepth: 2
   :caption: Sections

   data_analysis.rst
   physics_utils.rst
   codegen.rst
   simpilot.rst
   cli_tools.rst
