.. _grace-gallery:

Gallery
=======

A selection of images produced from GRACE simulations.  Each entry
links to the configuration and analysis tooling used so the figure can
be reproduced.


Post-merger of an SFHo binary neutron star
------------------------------------------

.. figure:: /_static/images/sfho_bns_postmerger.png
   :alt: 3D rendering of the SFHo BNS post-merger remnant, disk, and
         polar funnel around the central black hole.
   :align: center
   :width: 90%

   Post-merger state of an equal-mass SFHo binary neutron star
   simulation, :math:`t \approx 25\;\mathrm{ms}` after merger.
   Nested log-density isosurfaces trace the disk and the diffuse
   envelope; the bipolar mint-coloured cone is a high-entropy
   (:math:`s/k_\mathrm{B} = 25`) iso outlining the hot, low-density
   polar funnel.  The opaque black sphere at the centre is a lapse
   isosurface (:math:`\alpha = 0.35`), used as a visual proxy for the
   apparent horizon.  Cyan tubes are field lines of the magnetic field
   threading the remnant.

Rendered with `PyVista`_ from a native GRACE HDF5 volume dump
(``volume_out_*.h5``).  Source data is an asymmetric binary
(mass ratio :math:`q = 0.8`) at the GW170817 chirp mass
(:math:`\mathcal{M}_\mathrm{c} \approx 1.188\;M_\odot`,
i.e. component masses :math:`\sim 1.53\;M_\odot + 1.22\;M_\odot`)
on the SFHo tabulated equation of state, seeded with an initial
poloidal magnetic field of
:math:`B_\mathrm{max} = 10^{15}\;\mathrm{G}`.

.. _PyVista: https://pyvista.org/
