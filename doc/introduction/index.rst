.. _grace-intro:

Introduction
============

GRACE is a code framework utilising Finite Volume methods to solve hyperbolic partial differential equations.
It was mainly developed with the aim of solving the equations of General Relativistic Magnetohydrodynamics (GRMHD)
in the context of relativistic astrophysical systems, from which it takes its name. 
GRACE leverages the low level `p4est`_ library to handle its grid. Since p4est is based on a forest-of-oct-trees 
gridding algorithm, GRACE can support a variety of grid topologies and coordinate systems. Moreover, GRACE supports 
adaptive mesh refinement through user-supplied refinement criteria. 
GRACE is built and optimized to perform simulations on GPU backends. Its use of `Kokkos`_ for handling data allocation/deallocation
and kernel offloading ensures that the code remains readable while also being able to perform well on a variety of different 
platforms. 

Obtaining the code 
******************

The entire GRACE source code as well as the source for this documentation is available on `GitHub <https://github.com/carlomusolino/grace>`_. Please see the 
`Building The Code <_grace-building>`_ section of this documentation for a detailed guide on how to compile GRACE on your system.

Code dependencies
*****************

GRACE is built to have the minimum set of dependencies required for its performance and flexibility. Here is a list
of all external dependencies that need to be installed in order to compile GRACE:

p4est
++++++++ 

The p4est library is a fundamental component of GRACE since it is used to set up and 
modify the grid on which simulations are performed. The source code is available on 
the project's `GitHub <https://github.com/cburstedde/p4est>`__ along with an installation 
guide.

.. note::
    There are no particular requirements on build configuration options 
    for p4est to be compatible with GRACE. The code should be built in 
    Release mode unless you plan to do some heavy development with the 
    installation. All logs coming from p4est are redirected to GRACE's 
    internal logging system.


Kokkos 
+++++++++

Kokkos provides GRACE's memory allocation and kernel offloading policies. It is therefore 
a necessary requirement for the code's core functionality. The library can be obtained from its 
`GitHub <https://github.com/kokkos/kokkos>`__ repository. 

.. note::
    GRACE's built-in profilers rely on `Kokkos-Tools <https://github.com/kokkos/kokkos-tools>`_ to 
    measure performance of GPU kernels. Please download and 
    install the tools on your system if you plan on profiling 
    the code.

Building Kokkos 
~~~~~~~~~~~~~~~~

Since GRACE depends on CMake for its build system anyway, the recommended way of building Kokkos 
is to use CMake. Follow the build instructions in `Kokkos's Documentation <https://kokkos.org/kokkos-core-wiki/keywords.html>`_ for a detailed description 
of how to install the library. The important steps that need to be taken to ensure proper interoperability 
of Kokkos and GRACE are:
1. Enable the correct backend when building Kokoks by using the ``-DKokkos_ENABLE_{BACKEND}=ON`` configure option.
2. Ensure that you have selected the correct device microarchitecture to gain access to the more advanced optimizations available in Kokkos.

.. warning::
    GRACE requires device relocatable code to be compiled,
    for this reason, whatever backend you build Kokkos for, 
    you will need to specify the option 
    ``-DKokkos_ENABLE_{BACKEND}_RELOCATABLE_DEVICE_CODE=ON``
    in your CMake configuration.

MPI 
++++++

GRACE requires `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ for its parallel functionality. 
Since the programming model in GRACE has the data residing on device at all times, the MPI implementation 
that is linked against GRACE **must** support GPU-to-GPU data transfers. Most modern HPC facilities should already 
have such a library at your disposal. 

HDF5 
+++++

Parallel output of volume and surface data in GRACE utilises the `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ format by default.
This ensures efficient I/O as well as self-descriptive datasets that can be easily post-processed 
and analyzed. Note that all output produced with GRACE comes with `xdmf <https://www.xdmf.org/index.php/XDMF_Model_and_Format>`_ descriptors which make it 
natively readable by most VTK-compatible visualization software such as `Paraview <https://www.paraview.org/>`_ or `VisIt <https://visit-dav.github.io/visit-website/index.html>`_.

.. warning::
    GRACE uses MPI-parallel HDF5 input/output. Ensure that the version 
    of HDF5 being linked to your GRACE executable was built with MPI support 
    and using the same MPI as GRACE is using.

VTK
+++++

`VTK <https://vtk.org/>`_ can optionally be used to produce ``.vtu`` output in GRACE. The library is also 
used to slice the grid when producing codimension 1 output. The minimal set of components required for use of 
VTK in GRACE is: ``CommonCore``, ``CommonDataModel``, ``FiltersCore``, ``IOXML``, ``IOParallelXML``, ``ParallelMPI``.

spdlog
+++++++

`spdlog <https://github.com/gabime/spdlog>`_ is used for console and file logging in GRACE. It is a light-weight C++ utility that supports 
different levels of log priority as well as thread-safe operation. 

yaml-cpp
++++++++

`yaml-cpp <https://github.com/jbeder/yaml-cpp>`_ is a C++ `YAML <https://en.wikipedia.org/wiki/YAML>`_ parser and emitter 
that is used within GRACE to parse parameter files. 

CMake
++++++

`CMake <https://cmake.org/>`_ is the configure and build tool used by GRACE, it should be natively available 
on most platforms.

Catch2
+++++++

`Catch2 <https://github.com/catchorg/Catch2>`_ is GRACE's unit-test framework. It is not strictly required (but strongly recommended!) to install Catch2 in order 
to compile GRACE, since tests can be disabled.

The last two dependencies are only required for building this documentation. 

Doxygen
+++++++

`Doxygen <>` is necessary to build the detailed API reference guide. All functions/classes declarations in GRACE are 
documented using Doxygen-style comments that can be built into html or pdf documentation.

Sphinx
+++++++

The ReadTheDocs webpage hosting the documentation of GRACE utilizes the `Sphinx <>` Python utility. 
All the required python modules can be installed by running ``pip install -r ${GRACE_HOME}/doc/requirements.txt``.

Licensing 
*********

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Purus sit amet volutpat consequat mauris nunc congue nisi vitae. Urna nec tincidunt praesent semper. Vulputate dignissim suspendisse 
in est ante in nibh mauris cursus. Dui sapien eget mi proin. 

.. _Kokkos: https://github.com/kokkos/kokkos
.. _p4est: https://www.p4est.org/