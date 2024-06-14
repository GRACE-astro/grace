.. GRACE documentation Burgers Equation Parameters file

.. _burgers-parameters:

Burgers Equation Parameters
***************************

This page contains a full reference of all parameters available for the burgers module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``which_id``
   * - **Type:**
     - :data-type:`keyword`
   * - **Default Value:**
     - :default-value:`gaussian`
   * - **Allowed Range:**
     - :allowed-range:`gaussian, shocktube, three_states_shocktube, oned_N_wave`
   * - **Description:**
     - :param-description:`Initial data type.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``shocktube_left_state``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`4.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For 2 or 3 state shocktube: the left state value.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``shocktube_central_state``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`2.0`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For 3 state shocktube: the center state value.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``shocktube_right_state``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For 2 or 3 state shocktube: the right state value.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``shocktube_x_location``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For 2 or 3 state shocktube: the interface location along x.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``shocktube_x_location_2``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`-1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`For 3 state shocktube: the second interface location along x.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``gaussian_sigma``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.1`
   * - **Allowed Range:**
     - :allowed-range:`0+:*`
   * - **Description:**
     - :param-description:`Initial gaussian pulse width.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``gaussian_x_c``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Initial gaussian pulse x center.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``gaussian_y_c``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Initial gaussian pulse y center.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``gaussian_z_c``
   * - **Type:**
     - :data-type:`floating point`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Initial gaussian pulse z center.`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the burgers module to their default value.

.. code-block:: yaml

   burgers_equation: {
     which_id: "gaussian",
     gaussian_sigma: 0.1,
     gaussian_x_c: 0.,
     gaussian_y_c: 0.,
     gaussian_z_c: 0.,
     shocktube_x_location: -1.,
     shocktube_x_location_2: 1.,
     shocktube_left_state: 4.,
     shocktube_central_state: 2.,
     shocktube_right_state: 0.
   }


