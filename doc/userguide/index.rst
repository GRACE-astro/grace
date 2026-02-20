.. _grace-userguide:

User Guide
============

Welcome to the GRACE user guide! Below is a description of how to run GRACE for a variety of configurations, as well 
as of how to analyze the output it produces. 
If you're looking for a guide on how to build the code, please see the :doc:`related page <../code_building_guide/index>`.
If you're wondering what grace even is, please refer to the :doc:`Introduction <../introduction/index>`.


Input parameters
******************

Input parameters control the runtime behaviour of GRACE. They can be provided through a file, aptly called parameter file, which has 
to be written in `YAML <https://en.wikipedia.org/wiki/YAML>`__ format. The parameter file can be passed to the GRACE executable with 
the following invocation:

.. code-block:: bash

    $ mpirun -n <num_procs> ./grace --grace-parfile ./<parfile_name>.yaml

GRACE supports many different parameters that allow for very detailed customization of the runtime behaviour
of the code. We will now give a list of all possible parameters, their type, their default value and the range 
of values they support. Parameters in the ``yaml`` config file are split into sections, which roughly correspond 
with GRACE's modules, so we'll provide description of the parameters together grouped by the module they belong to.

.. toctree::
   :maxdepth: 1
   :caption: Code Modules

   params-m1
   params-gw_integrals
   params-bh_diagnostics
   params-coordinate_system
   params-checkpoints
   params-IO
   params-spherical_surfaces
   params-puncture_tracker
   params-grmhd
   params-evolution
   params-system
   params-z4c
   params-amr


Output Data Formats 
*********************

There are a few different kinds of output in GRACE. 

First, there is scalar output. This encompasses reductions of any registered variable (auxiliary or evolved) in GRACE.
Possible reductions are: coordinate volume integral, L2 norm, max and min. The output frequency of all scalar quantities is controlled by a single parameter, 
and all output can be found in the same directory. Output files contain three columns: iteration, simulation time, and the value. 

The second kind of output is also in the form of a timeseries and comprises all spherical surface integrals. These are implemented 
as custom modules in GRACE which have their own parameters. One example is the ``gw_integrals`` module, which simply takes the names 
of registered surfaces where integrals should be performed, and outputs the spherical harmonic decomposition of the Penrose-Newman scalar 
onto these surfaces. The output frequency here is controlled by the diagnostic output frequency, and the corresponding files are placed in 
the same directory as scalars. 

Other outputs are performed in HDF5 by default, with the legacy (deprecated) option of native VTK. GRACE supports volume and plane surface output (in xy, xz, and yz planes), as 
well as output of point data on spherical surfaces. All this data can be easily visualized in `ParaView <https://www.paraview.org/>`_ through the use of XDMF descriptors (see the :doc:`related page <../python_interface/index>` on how to generate them), 
as well as in python through the vtk Python interface. 