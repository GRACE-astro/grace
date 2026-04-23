/**
 * @file spherical_surface_utils.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-03
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
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

#ifndef GRACE_IO_SPHERICAL_SURFACE_HELPERS_HH
#define GRACE_IO_SPHERICAL_SURFACE_HELPERS_HH
#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/coordinates/coordinate_systems.hh>

#include "surface_IO_utils.hh"
#include "octree_search_class.hh"

#include <array>
#include <memory>
#include <tuple> 

#include <Kokkos_Core.hpp>


namespace grace {


struct  uniform_sampler_t {
    static size_t get_n_points(size_t const& res) {
        return (2 * res) * (res) ; 
    }

    static std::vector<point_host_t> 
    get_points(double radius, std::array<double,3> const& center, size_t const& res, std::vector<std::array<double,2>>& angles)
    {
      bool equatorial_symm{grace::get_param<bool>("amr","reflection_symmetries","z")} ; 
      ASSERT(res%2, "Simpson rule requires odd npoints") ; 
      double mu_min = equatorial_symm ? 0.0 : -1.0;
      double mu_max = 1.0;

      size_t ntheta = res ; 
      size_t nphi = 2 * res;
      size_t npoints = ntheta*nphi ; 
      auto& coord_system = grace::coordinate_system::get() ; 
      angles.clear() ; 
      angles.reserve(npoints); 
      
      for( int iphi=0; iphi<nphi; ++iphi) {
          double phi = 2 * M_PI / (nphi) * iphi ; // this excludes the endpoint 2pi == 0 
          for( int itheta=0; itheta<ntheta; ++itheta) {
              //double mu = mu_min + (mu_max-mu_min)/(ntheta-1) * itheta ; 
              double mu = mu_max - (mu_max - mu_min)/(ntheta-1) * itheta ; 
              double theta = acos(mu) ; 
              angles.push_back({theta,phi}) ; 
          }
      }

      std::vector<point_host_t> points;
      points.reserve(ntheta*nphi);

      for( size_t i=0; i<ntheta*nphi; i+=1UL) {
          double theta = angles[i][0] ; 
          double phi = angles[i][1] ; 
          std::array<double,3> p{radius,theta,phi} ; 
          // convert to cartesian, in CKS this 
          // is not the standard formula! 
          p = coord_system.sph_to_cart(p) ; 
          p[0] += center[0] ; 
          p[1] += center[1] ; 
          p[2] += center[2] ; 
          if (equatorial_symm) p[2] = fmax(p[2],1e-15) ; // ensure not == 0 otherwise might fall outside grid
          points.push_back(std::make_pair(i,p)) ; 
      }

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

struct no_tracking_policy_t {
    bool track(
        double& radius,
        std::array<double,3>& center
    ) {
        return false;
    }
} ; 


}

#endif /* GRACE_IO_SPHERICAL_SURFACE_HELPERS_HH */