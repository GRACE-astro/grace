.. _grace-gallery:

Gallery
=======

A selection of images produced from GRACE simulations.  Each entry
links to the configuration and analysis tooling used so the figure can
be reproduced.


Post-merger of an SFHo binary neutron star
------------------------------------------

.. figure:: /_static/images/sfho_bns_merger.png
   :alt: 3D rendering of the merger of a BNS described by the SFHo EOS.
   :align: center
   :width: 90%

   Rendering of a magnetized BNS system around the time of merger. 

Rendered with `PyVista`_ from a native GRACE HDF5 volume dump
(``volume_out_*.h5``).  Source data is an asymmetric binary
(mass ratio :math:`q = 0.8`) at the GW170817 chirp mass
(:math:`\mathcal{M}_\mathrm{c} \approx 1.188\;M_\odot`,
i.e. component masses :math:`\sim 1.53\;M_\odot + 1.22\;M_\odot`)
on the SFHo tabulated equation of state, seeded with an initial
poloidal magnetic field of
:math:`B_\mathrm{max} = 10^{15}\;\mathrm{G}`.

.. _PyVista: https://pyvista.org/
