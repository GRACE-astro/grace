/**
 * @file geodesic_sphere_helpers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Geodesic-sphere (icosahedral) sampler used for AH-finder and sphere-output point distributions.
 *
 * @copyright This file is part of GRACE.
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
 */
#ifndef GRACE_IO_GEO_SPHERE_HH
#define GRACE_IO_GEO_SPHERE_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/coordinates/coordinate_systems.hh>

#include "surface_IO_utils.hh"

#include <array>
#include <memory>
#include <tuple> 
#include <vector>

#include <Kokkos_Core.hpp>

namespace grace {

struct  geodesic_sampler_t {
    static size_t get_n_points(size_t const& res) {
        return 5 * 2 * res * res + 2 ; 
    }

    static std::vector<point_host_t> 
    get_points(double radius, std::array<double,3> const& center, size_t const& res, std::vector<std::array<double,2>>& angles)
    {
        bool equatorial_symm{grace::get_param<bool>("amr","reflection_symmetries","z")} ; 
        auto npoints = get_n_points(res) ; 
        std::vector<point_host_t> points(npoints) ;
        angles.doubleloc(npoints) ; 
        
        // construction parameters
        double sin_ang = 2.0/sqrt(5.0);
        double cos_ang = 1.0/sqrt(5.0);
        double p1[3] = {0.0, 0.0, 1.0};
        double p2[3] = {sin_ang, 0.0, cos_ang};
        double p3[3] = {sin_ang*cos( 0.2*M_PI), sin_ang*sin( 0.2*M_PI), -cos_ang};
        double p4[3] = {sin_ang*cos(-0.4*M_PI), sin_ang*sin(-0.4*M_PI),  cos_ang};
        double p5[3] = {sin_ang*cos(-0.2*M_PI), sin_ang*sin(-0.2*M_PI), -cos_ang};
        double p6[3] = {0.0, 0.0, -1.0};

        
        return points;
    }


  static std::vector<double> get_quadrature_weights(double radius, size_t const& res)
  {
      bool equatorial_symm{grace::get_param<bool>("amr","reflection_symmetries","z")} ; 
      ASSERT(res%2, "Simpson rule requires odd npoints") ; 
      double mu_min = equatorial_symm ? 0.0 : -1.0;
      double mu_max = 1.0;


      size_t ntheta = res;
      size_t nphi   = 2 * res ;
      size_t npoints = ntheta * nphi;

      std::vector<double> weights(npoints);

      double htheta = (mu_max-mu_min) / (ntheta - 1);
      double hphi = 2*M_PI / (nphi);

      for (size_t itheta = 0; itheta < ntheta; ++itheta)
      {
          double wmu;

          if (itheta == 0 || itheta == ntheta-1)
              wmu = 1.0;
          else if (itheta % 2 == 1)
              wmu = 4.0;
          else
              wmu = 2.0;

          wmu *= htheta / 3.0;
          for (size_t iphi = 0; iphi < nphi; ++iphi) 
            weights[iphi*ntheta + itheta] = wmu * hphi;          
      }

      return weights;
  }

} ;

}

#endif /* GRACE_IO_GEO_SPHERE_HH */