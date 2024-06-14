.. GRACE documentation Scalar Advection Equation Parameters file

.. _scalar-advection-parameters:

Scalar Advection Equation Parameters
************************************

This page contains a full reference of all parameters available for the scalar-advection module.
At the bottom of the page you can find an example parameter block that can be used in your parameter file.

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``ax``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Advection speed in x direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``ay``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Advection speed in y direction.`

.. raw:: html

  </div>

----

.. raw:: html

  <div class="custom-list-table-container">

.. list-table:: 
   :header-rows: 0
   :class: custom-list-table

   * - **Name:**
     - ``az``
   * - **Type:**
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`1.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Advection speed in z direction.`

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
     - :data-type:`double`
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
     - :data-type:`double`
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
     - :data-type:`double`
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
     - :data-type:`double`
   * - **Default Value:**
     - :default-value:`0.`
   * - **Allowed Range:**
     - :allowed-range:`*`
   * - **Description:**
     - :param-description:`Initial gaussian pulse z center.`

.. raw:: html

  </div>

----

The following snippet is an example of how to set all the parameters for the scalar-advection module to their default value.

.. code-block:: yaml

   scalar_advection: {
     ax: 1.,
     ay: 1.,
     az: 1.,
     gaussian_sigma: 0.1,
     gaussian_x_c: 0.,
     gaussian_y_c: 0.,
     gaussian_z_c: 0.
   }


