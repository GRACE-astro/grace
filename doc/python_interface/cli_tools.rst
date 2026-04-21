.. _gracepy-cli-tools:

Command-Line Tools
====================

GRACEpy installs several command-line utilities via ``pip``. All are available after running ``pip install -e ./`` in the GRACEpy repository.


create_descriptor
*******************

Generate an XMF descriptor file from GRACE HDF5 output. Descriptors are required for loading data in both ParaView and the Python VTK reader.

.. code-block:: bash

    create_descriptor <input_dir> <output_file> [--mode {auto,temporal,spatial,spherical}]
                                                [--verbose] [--filter <pattern>]

Arguments:

- ``input_dir`` — directory (or directories) containing HDF5 files
- ``output_file`` — path for the generated XMF descriptor
- ``--mode`` — collection type (default ``auto``):

  - ``auto`` — detect from file names
  - ``temporal`` — timeseries of grids (most common for Python analysis)
  - ``spatial`` — spatial collection (e.g. multiple surfaces at one time)
  - ``spherical`` — spherical surface data

- ``--filter`` — regular expression to select a subset of files (e.g. ``"*xy*"`` for xy-plane data)
- ``--verbose`` — print progress information


export_scalars
****************

Export all scalar and gravitational wave data from a simulation to a single HDF5 file. Handles restart merging automatically.

.. code-block:: bash

    export_scalars <simdir> [-o <outfile>] [--parfile <parfile>]

Arguments:

- ``simdir`` — simulation directory (simpilot or flat layout)
- ``-o``, ``--output`` — output HDF5 file path (default: ``<simdir>/plots/<simname>_scalars.h5``)
- ``--parfile`` — explicit parameter file path (auto-detected if omitted)


grace_info
************

Inspect a GRACE parameter file and display grid information: domain size, block structure, refinement levels, and cell spacing.

.. code-block:: bash

    grace_info <parfile>

Output includes:

- Grid dimensions and total cell count
- Block size per direction (with validation: should be a power of 2 in the range 16--64)
- Number of refinement levels and the resulting finest cell spacing :math:`\Delta x`


archive_source
****************

Seal a source tree into an HDF5 archive for reproducibility. The archive stores every file as a binary dataset and records the current git commit hash and any unstaged changes.

.. code-block:: bash

    archive_source <source_dir> <output_file> [--exclude <patterns>...] [--include-git]

Arguments:

- ``source_dir`` — root of the source tree to archive
- ``output_file`` — output HDF5 file
- ``--exclude`` — one or more glob patterns for files to skip (e.g. ``"*.o" "build/*"``)
- ``--include-git`` — include the ``.git/`` directory (excluded by default)


unpack_archive
****************

Restore a source tree from an HDF5 archive created by ``archive_source``.

.. code-block:: bash

    unpack_archive <source_file> <output_dir> [--force]

Arguments:

- ``source_file`` — HDF5 archive file
- ``output_dir`` — target directory (must be empty unless ``--force`` is given)
- ``--force`` — allow unpacking into a non-empty directory


convert_timers_to_ascii
*************************

Convert binary Kokkos kernel timing output (GPU profiling data) to human-readable ASCII format.

.. code-block:: bash

    convert_timers_to_ascii <input_dir>

This scans for files matching ``gpu[0-9]*-[0-9]*.dat`` in the input directory and writes converted ASCII output.
