.. _grace-building:

Building The Code
=====================

GRACE's build system is based on `CMake <https://cmake.org/>`__. This guide will provide a detailed explanation of all the configure flags that 
can be passed to the build system to customize the code's behaviour, and assumes that you already installed all the code's dependencies.
If you haven't done that yet, please consult the :doc:`Introduction <../introduction/index>`.
If you're unfamiliar with CMake, please consult their extensive documentation to learn more about this tool. For our purposes, 
CMake is a build system that we use to generate Makefiles that can then be used to generate the code. GRACE's build process 
is effectively divided into two parts: the configure step and the build step. The configure step is where all the build-time flags 
controlling the compilers to be used, the flags to be passed to the compiler, as well as all the GRACE-specific flags that set the 
compile-time environment are passed. The build step is simply a way to automatically call make on all the generated Makefiles.
We will now describe all the options that grace accepts as inputs during its configure step and how these influence the resulting 
executable, as well as some relevant CMake specific flags that are especially relevant to GRACE. Unless explicitly specified, all 
flags are options, meaning boolean type. The way to specify a flag to CMake is as follows: 

.. code-block:: bash 
    
    $ cmake -B build -S ./ -D<FLAG_NAME>=<FLAG_VALUE>

For a boolean flag (option), allowed values are ``ON`` or ``OFF``.

Environment File 
************************************

The recommended way of building GRACE is to first create an environment file. Some examples can be found in the GRACEpy repository as well as the main GRACE repository under `env`. 

In short, these files should load necessary modules and define environment variables pointing to the installation path of GRACE dependencies. 

The CMake build system looks for dependencies in the location pointed to by the environment variable `<DEP>_ROOT`, so for instance your environment file should contain `KOKKOS_ROOT=/my/kokkos/installation`

After creating and sourcing your environment file you're ready for the configure step.

Configure Options
************************************

What follows is a description of all the configure time flags that can be passed to GRACE's CMake build system, divided into categories.


Backend selection flags 
***********************************

GRACE needs to be made aware of the backend which it is being used on. Note that the Kokkos installation that is linked at compile time must also have this backend enabled. 

Note that these options are mutually exclusive, and that the `OpenMP` option enables it as the main backend. OpenMP is always used for host-only operations. 

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type 
     - Description
   * - GRACE_ENABLE_HIP
     - Boolean 
     - Enable HIP backend (AMD GPUs and APUs)
   * - GRACE_ENABLE_CUDA 
     - Boolean 
     - Enable CUDA backend (Nvidia GPUs)
   * - GRACE_ENABLE_OMP
     - Boolean
     - Enables Host-only OpenMP parallelism 

Code Modules Option
***********************************

GRACE supports various physical systems, and the following configure flags decide which ones are active 

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type 
     - Description
   * - GRACE_ENABLE_GRMHD
     - Boolean 
     - Enable GRMHD module (should be always ON)
   * - GRACE_ENABLE_COWLING_METRIC 
     - Boolean 
     - Enable Cowling (frozen) metric, mutually exclusive with Z4C. 
   * - GRACE_ENABLE_Z4C_METRIC
     - Boolean
     - Enable Z4c evolution of metric 
   * - GRACE_ENABLE_M1
     - Boolean
     - Enable M1 radiative transport (work in progress, developer only)
    
External Libraries Options 
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type 
     - Description
   * - GRACE_ENABLE_LORENE
     - Boolean 
     - Enable LORENE support 
   * - GRACE_ENABLE_TWOPUNCTURES
     - Boolean 
     - Enable TwoPuncture support 

Miscellaneous Options 
***********************************

These are mostly legacy options or options related to (potential) future developments, ignore them.

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type 
     - Description
   * - GRACE_ENABLE_TESTS
     - Boolean
     - Build unit tests 
   * - GRACE_BUILD_DOCS 
     - Boolean
     - Build this documentation 
   * - GRACE_BUILD_DOCS_ONLY
     - Boolean
     - Only build this documentation 
   * - GRACE_NSPACEDIM
     - Integer
     - Number of spatial dimensions (only 3 supported right now)
   * - GRACE_ENABLE_VTK
     - Boolean 
     - Enable VTK output (deprecated)
   * - GRACE_ENABLE_PROFILING 
     - Boolean 
     - Enable legacy HIP profiling hooks (deprecated)
   * - GRACE_ENABLE_CARTESIAN_COORDINATES
     - Boolean
     - Enable Cartesian coordinates (only option right now)


