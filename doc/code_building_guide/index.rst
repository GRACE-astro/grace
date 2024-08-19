.. _grace-building:

Building The Code
=====================

GRACE's build system is based on `CMake <>`__. This guide will provide a detailed explanation of all the configure flags that 
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

Configure Options
************************************

What follows is a description of all the configure time flags that can be passed to GRACE's CMake build system, divided into categories.

.. toctree::
    :maxdepth: 1
    :caption: Code Modules 

    config-params-cmake.rst
    config-params-backend.rst
    config-params-grace.rst

Backend selection flags 
=========================





An example build 
***********************************



Building The Tests
***********************************

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Purus sit amet volutpat consequat mauris nunc congue nisi vitae. Urna nec tincidunt praesent semper. Vulputate dignissim suspendisse 
in est ante in nibh mauris cursus. Dui sapien eget mi proin.

Building the docs 
***********************************



