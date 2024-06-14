.. GRACE documentation Profiling Parameters file

.. _profiling-parameters:

Profiling Parameters
********************

This page contains a full reference of all parameters available for the profiling module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``output_base_directory``
   * - **Type:**
     - :data-type:`string`
   * - **Default Value:**
     - :default-value:`./data_profiling`
   * - **Allowed Range:**
     - :allowed-range:`Any valid relative or absolute path which is user writeable.`
   * - **Description:**
     - :param-description:`The profiling output files will be placed in this directory.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``enabled_hardware_counters``
   * - **Type:**
     - :data-type:`list of strings`
   * - **Default Value:**
     - :default-value:`["GPUBusy","MemUnitBusy","WaveFronts","VALUInsts","SALUInsts","VALUUtilization","VALUBusy","SALUBusy","FetchSize","WriteSize","L2CacheHit"]`
   * - **Allowed Range:**
     - :allowed-range:`Any valid hardware counter for the platform's microarchitecture.`
   * - **Description:**
     - :param-description:`The default counters are valid for AMD GPUs starting at microarchitecture gfx906 (codename Vega20).`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the profiling module to their default value.

.. code-block:: yaml

   profiling: {
     output_base_directory: "data_profiling",
     # The following hardware counters are hardware specific
     # This means that the entries will change based on the 
     # GPU vendor and the microarchitecture. The provided defaults
     # are good performance indicators that should be sufficient 
     # for a pretty detailed profiling of Kernel performance on 
     # AMD gfx906 chips (codename Vega20, cards Mi50, Mi60). Newer
     # cards should support the same options and might have additional
     # useful metrics. For older cards and / or Nvidia GPUs, please 
     # refer to GRACE documentation and the GPU technical specifications to find 
     # suitable options.
     enabled_hardware_counters: [
     # 1) Coarse metrics regarding time spent
     "GPUBusy", # Percentage of time the GPU was busy
     "MemUnitBusy", # Percentage of time the GPU memory unit was busy
     # 2) ALU instruction metrics 
     "WaveFronts", # Total number of wavefronts executed
     "VALUInsts",  # Avg number of Vector ALU instructions per wavefront
     "SALUInsts",  # Avg number of Scalar ALU instructions per wavefront
     "VALUUtilization", # % of active VALU threads in a wave
     "VALUBusy", # % of GPU time spent in VALU instructions
     "SALUBusy", # % of GPU time spent in SALU instructions
     # 3) Memory statistics
     "FetchSize", # total # of Kilobytes fetched from video memory 
     "WriteSize", # total # of Kilobytes written to video memory
     "L2CacheHit"  # % of write read atomic and other data insts that hit L2 cache data.
     ] #warning: only change these if you REALLY know what you're doing
   }
   


