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

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>



#include "surface_IO_utils.hh"

#include <array>
#include <memory>
#include <tuple> 

#include <Kokkos_Core.hpp>


namespace grace {

using point_host_t = std::pair<size_t,std::array<double,3>> ; 



namespace chealpix {

static int isqrt(int v)
{ return (int)(sqrt(v+0.5)); }

static void pix2ang_ring_z_phi (int nside_, int pix, double *z, double *phi)
  {
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

inline void pix2vec_ring(long nside, long ipix, std::array<double,3>& vec)
  {
  double z, phi;
  pix2ang_ring_z_phi (nside,ipix,&z,&phi);
  double stheta=sqrt((1.-z)*(1.+z));
  vec[0]=stheta*cos(phi);
  vec[1]=stheta*sin(phi);
  vec[2]=z;
  }

void ang2vec(double theta, double phi, std::array<double,3>& vec)
  {
  double sz = sin(theta);
  vec[0] = sz * cos(phi);
  vec[1] = sz * sin(phi);
  vec[2] = cos(theta);
  }

void vec2ang(const double *vec, double *theta, double *phi)
  {
  *theta = atan2(sqrt(vec[0]*vec[0]+vec[1]*vec[1]),vec[2]);
  *phi = atan2 (vec[1],vec[0]);
  if (*phi<0.) *phi += twopi;
  }

long npix2nside(long npix)
  {
  long res = isqrt(npix/12);
  return (res*res*12==npix) ? res : -1;
  }

long nside2npix(const long nside)
  { return 12*nside*nside; }

}

struct healpix_sampler_t {

    static size_t get_n_points(size_t const& res) {
        return nside2npix(res);
    }

    static std::vector<point_host_t>
    get_points(double radius, std::array<double,3> const& center, size_t const& res)
    {
        size_t nside = res ; 
        size_t npix = nside2npix(nside);
        std::vector<point_host_t> points;
        points.reserve(nside2npix(res));

        for( size_t ipix=0; ipix<npix; ipix+=1UL) {
            std::array<double,3> p; 
            pix2vec_ring(nside,ipix,p) ; 
            for( int i=0; i<3; ++i) p[i] += center[i] ; 
            points.push_back(std::make_pair(ipix,p)) ;  
        }

        return points;
    }
    //! TODO (?) this is simply dA for all 
    // points
    static std::vector<double> get_quadrature_weights(double radius,size_t const& res) {
        size_t npix = nside2npix(res); 
        double A = 4 * M_PI / npix * radius * radius ; 
        return std::vector<double>(npix, A) ; 
    }
};

struct  uniform_sampler_t {
    static size_t get_n_points(size_t const& res) {
        return 2 * res * res ; 
    }

    static std::vector<point_host_t> 
    get_points(double radius, std::array<double,3> const& center, size_t const& res)
    {
        size_t ntheta = res ; 
        size_t nphi = 2*res ; 
        size_t npoints = ntheta*nphi ; 

        std::vector<std::array<double,2>> angles ; 
        angles.reserve(npoints); 

        for( int iphi=0; iphi<nphi; ++iphi) {
            double phi = M_PI / ntheta * iphi ; 
            for( int itheta=0; itheta<ntheta; ++itheta) {
                double mu = -1.0 + 2.0/(ntheta-1) * itheta ; 
                double theta = acos(mu) ; 
                angles.emplace_back({{theta,phi}})
            }
        }

        std::vector<point_host_t> points;
        points.reserve(ntheta*nphi);

        for( size_t i=0; i<ntheta*nphi; i+=1UL) {
            double theta = angles[i][0] ; 
            double phi = angles[i][1] ; 
            std::array<double,3> p ; 
            p[0] = center[0] + radius * cos(phi) * sin(theta) ; 
            p[1] = center[1] + radius * sin(phi) * sin(theta) ; 
            p[2] = center[2] + radius * cos(theta) ; 
            points.push_back(std::make_pair(i,p)) ; 
        }

        return points;
    }


    static std::vector<double> get_quadrature_weights(double radius,size_t const& res) {
        size_t n_points = get_n_points(res) ; 
        double A = M_PI / (res) * (2.0)/res ; 
        return std::vector<double>(n_points, A) ; 
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