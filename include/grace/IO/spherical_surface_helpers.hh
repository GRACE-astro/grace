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




namespace chealpix {

static int isqrt(int v)
{ return (int)(sqrt(v+0.5)); }

static void pix2ang_ring_z_phi (int nside_, int pix, double *z, double *phi)
{
  double const halfpi = M_PI * 0.5 ; 
  double const pi = M_PI ; 
  long ncap_=nside_*(nside_-1)*2;
  long npix_=12*nside_*nside_;
  double fact2_ = 4./npix_;
  if (pix<ncap_) /* North Polar cap */
    {
    int iring = (1+isqrt(1+2*pix))>>1; /* counted from North pole */
    int iphi  = (pix+1) - 2*iring*(iring-1);

    *z = 1.0 - (iring*iring)*fact2_;
    *phi = (iphi-0.5) * halfpi/iring;
    }
  else if (pix<(npix_-ncap_)) /* Equatorial region */
    {
    double fact1_  = (nside_<<1)*fact2_;
    int ip  = pix - ncap_;
    int iring = ip/(4*nside_) + nside_; /* counted from North pole */
    int iphi  = ip%(4*nside_) + 1;
    /* 1 if iring+nside is odd, 1/2 otherwise */
    double fodd = ((iring+nside_)&1) ? 1 : 0.5;

    int nl2 = 2*nside_;
    *z = (nl2-iring)*fact1_;
    *phi = (iphi-fodd) * pi/nl2;
    }
  else /* South Polar cap */
    {
    int ip = npix_ - pix;
    int iring = (1+isqrt(2*ip-1))>>1; /* counted from South pole */
    int iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));

    *z = -1.0 + (iring*iring)*fact2_;
    *phi = (iphi-0.5) * halfpi/iring;
    }
}

static inline void pix2vec_ring(long nside, long ipix, std::array<double,3>& vec)
{
  double z, phi;
  pix2ang_ring_z_phi (nside,ipix,&z,&phi);
  double stheta=sqrt((1.-z)*(1.+z));
  vec[0]=stheta*cos(phi);
  vec[1]=stheta*sin(phi);
  vec[2]=z;
}

static inline void pix2ang_ring(long nside, long ipix, std::array<double,3>& vec)
{
  double z, phi;
  pix2ang_ring_z_phi (nside,ipix,&z,&phi);
  double stheta=sqrt((1.-z)*(1.+z));
  vec[0]=1;
  vec[1]=asin(stheta);
  vec[2]=phi;
}

static void ang2vec(double theta, double phi, std::array<double,3>& vec)
{
  double sz = sin(theta);
  vec[0] = sz * cos(phi);
  vec[1] = sz * sin(phi);
  vec[2] = cos(theta);
}

static void vec2ang(const double *vec, double *theta, double *phi)
{
  double const twopi = 2 * M_PI ; 
  *theta = atan2(sqrt(vec[0]*vec[0]+vec[1]*vec[1]),vec[2]);
  *phi = atan2 (vec[1],vec[0]);
  if (*phi<0.) *phi += twopi;
}

static long npix2nside(long npix)
{
  long res = isqrt(npix/12);
  return (res*res*12==npix) ? res : -1;
}

static long nside2npix(const long nside)
{ return 12*nside*nside; }

}

struct healpix_sampler_t {

    static size_t get_n_points(size_t const& res) {
      return chealpix::nside2npix(res);
    }

    static std::vector<point_host_t>
    get_points(double radius, std::array<double,3> const& center, size_t const& res, std::vector<std::array<double,2>>& angles)
    {
      using namespace chealpix ;
      auto& coord_system = grace::coordinate_system::get() ; 
      size_t nside = res ; 
      size_t npix = nside2npix(nside);
      std::vector<point_host_t> points;
      points.reserve(nside2npix(res));
      angles.clear() ; angles.reserve(nside2npix(res));

      for( size_t ipix=0; ipix<npix; ipix+=1UL) {
          std::array<double,3> p; 
          pix2ang_ring(nside,ipix,p) ; 
          angles.push_back({p[1],p[2]}) ; 
          p[0] = radius ; 
          p = coord_system.sph_to_cart(p) ; 
          for( int i=0; i<3; ++i) p[i] += center[i] ; 
          points.push_back(std::make_pair(ipix,p)) ;  
      }

      return points;
    }
    //! TODO (?) this is simply dA for all 
    // points
    static std::vector<double> get_quadrature_weights(double radius,size_t const& res) {
      size_t npix = chealpix::nside2npix(res); 
      double A = 4 * M_PI / npix * radius * radius ; 
      return std::vector<double>(npix, A) ; 
    }
};

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