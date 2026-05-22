/**
 * @file cylindrical_blast.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Komissarov-1999-style cylindrical magnetic blast wave for the
 *        symmetry-preservation audit suite (Tier-1 MHD test).
 * @date 2026-05-08
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#ifndef GRACE_PHYSICS_ID_CYLINDRICAL_BLAST_HH
#define GRACE_PHYSICS_ID_CYLINDRICAL_BLAST_HH

#include <grace_config.h>

#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {

/**
 * @brief Cylindrical magnetic blast wave (Komissarov 1999).
 * \ingroup initial_data
 *
 * A high-pressure cylindrical pulse at the origin in a uniform poloidal
 * background field Bx = B0; v = 0 initially.  Used by the symmetry-audit
 * harness to test the MHD pipeline (CT/EMF, magnetic Riemann branches,
 * C2P with B) under x-mirror reflection symmetry across the y-axis.
 *
 * Symmetry under x-mirror (x -> -x, y unchanged):
 *   - rho(r), p(r): scalars depending only on r = sqrt(x^2 + y^2)  -> invariant
 *   - v = 0                                                         -> trivial
 *   - Bx = B0 (axial, x-component invariant under x-mirror)         -> invariant
 *   - By = Bz = 0                                                   -> trivial
 * So the IC is bit-exactly x-mirror symmetric.  (Note that the same IC is
 * NOT pi-rotation symmetric: uniform Bx is even under x->-x but axial
 * vectors require an x-flip under pi-rot.  Use sigma_pol=0 KHI for the
 * pi-rot audit instead.)
 *
 * Profile: smooth tanh transition centred at r_c = 0.5*(r_in + r_out)
 * with width delta = 0.5*(r_out - r_in).  Bumps from ~1 inside r_in to
 * ~0 outside r_out.
 *
 * @tparam eos_t equation-of-state type
 */
template < typename eos_t >
struct cylindrical_blast_id_t {
    using state_t = grace::var_array_t;

    cylindrical_blast_id_t(
          eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords
        , double const rho_in
        , double const rho_out
        , double const press_in
        , double const press_out
        , double const r_in
        , double const r_out
        , double const B0 )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho_in(rho_in)
        , _rho_out(rho_out)
        , _press_in(press_in)
        , _press_out(press_out)
        , _r_in(r_in)
        , _r_out(r_out)
        , _B0(B0)
    {}

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const
    {
        grmhd_id_t id;

        double const x = _pcoords(VEC(i,j,k), 0, q);
        double const y = _pcoords(VEC(i,j,k), 1, q);
        double const z = _pcoords(VEC(i,j,k), 2, q);

        // Flat (Minkowski) metric.
        id.alp   = 1;
        id.betax = 0; id.betay = 0; id.betaz = 0;
        id.gxx   = 1; id.gyy   = 1; id.gzz   = 1;
        id.gxy   = 0; id.gxz   = 0; id.gyz   = 0;
        id.kxx   = 0; id.kyy   = 0; id.kzz   = 0;
        id.kxy   = 0; id.kxz   = 0; id.kyz   = 0;

        // Smooth radial bump:  ~1 for r << r_in, ~0 for r >> r_out.
        double const r     = Kokkos::sqrt(x*x + y*y);
        double const r_c   = 0.5 * (_r_in + _r_out);
        double const delta = 0.5 * (_r_out - _r_in);
        double const bump  = 0.5 * (1.0 - Kokkos::tanh((r - r_c) / delta));

        id.rho   = _rho_out  + (_rho_in   - _rho_out)  * bump;
        id.press = _press_out + (_press_in - _press_out) * bump;

        id.vx = 0.0;
        id.vy = 0.0;
        id.vz = 0.0;

        // Uniform poloidal guiding field (Komissarov 1999).
        id.bx = _B0;
        id.by = 0.0;
        id.bz = 0.0;

        eos_err_t err;
        id.ye = _eos.ye_cold__press(id.press, err);
        double h, csnd2;
        id.eps = _eos.eps_h_csnd2_temp_entropy__press_rho_ye(
            h, csnd2, id.temp, id.entropy, id.press, id.rho, id.ye, err
        );

        return std::move(id);
    }

    eos_t   _eos       ;                            //!< Equation of state object
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords; //!< Physical coordinates of cell centers
    double  _rho_in    ;                            //!< Density inside the pulse
    double  _rho_out   ;                            //!< Density of the background medium
    double  _press_in  ;                            //!< Pressure inside the pulse
    double  _press_out ;                            //!< Pressure of the background medium
    double  _r_in      ;                            //!< Inner radius of the smooth transition shell
    double  _r_out     ;                            //!< Outer radius of the shell
    double  _B0        ;                            //!< Uniform poloidal Bx
};

}
#endif /* GRACE_PHYSICS_ID_CYLINDRICAL_BLAST_HH */
