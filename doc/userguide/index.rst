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

    params-system.rst
    params-evolution.rst
    params-amr.rst
    params-IO.rst
    params-eos.rst
    params-scalar-advection.rst
    params-burgers.rst
    params-profiling.rst

Output Data Formats 
*********************