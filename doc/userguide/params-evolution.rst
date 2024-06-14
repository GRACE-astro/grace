.. GRACE documentation Evolution Parameters file

.. _evolution-parameters:

Evolution Parameters
********************

This page contains a full reference of all parameters available for the evolution module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``time_stepper``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`euler`
   * - **Allowed Range:**
     - :allowed-range:`euler, rk2, rk3`
   * - **Description:**
     - :param-description:`Time-stepping algorithm used in GRACE.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``cfl_factor``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`0+:1.`
   * - **Description:**
     - :param-description:`CFL factor used to determine timestep.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``final_time``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`2.`
   * - **Allowed Range:**
     - :allowed-range:`0+:*`
   * - **Description:**
     - :param-description:`Time at which the simulation ends.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``reset_id_after_regrid``
   * - **Type:**
     - :data-type:`bool`
   * - **Default Value:**
     - :default-value:`false`
   * - **Allowed Range:**
     - :allowed-range:`true, false`
   * - **Description:**
     - :param-description:`Set initial data again after postinitial regrid.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``timestep_selection_mode``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`automatic`
   * - **Allowed Range:**
     - :allowed-range:`automatic, manual`
   * - **Description:**
     - :param-description:`Method to determine timestep, automatic chooses the maximum stable timestep based on characteristic speeds and CFL factor, manual sets it to a constant value.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``timestep``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.1`
   * - **Allowed Range:**
     - :allowed-range:`0+:*`
   * - **Description:**
     - :param-description:`If timestep_selection_mode is manual, this is the constant timestep. Typically used for testing. Beware that if this is set it's the user's responsibility to ensure stability.`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the evolution module to their default value.

.. code-block:: yaml

   evolution: {
     time_stepper: "euler",
     cfl_factor: 1.,
     final_time: 2.,
     reset_id_after_regrid: false,
     timestep_selection_mode: "automatic",
     timestep: 0.1
   }


