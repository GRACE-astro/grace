/**
 * @file spherical_surfaces.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
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

#ifndef GRACE_SPHERICAL_SURFACES_HH
#define GRACE_SPHERICAL_SURFACES_HH 

#include <grace_config.h>

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>
#include <grace/utils/device_vector.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

namespace grace {

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

    static size_t get_n_points(std::pair<size_t, size_t> const& res) {
        return nside2npix(res.fist);
    }

    static std::vector<std::array<double,3>>
    get_points(double radius, std::array<double,3> const& center, std::pair<size_t,size_t> const& res)
    {
        size_t nside = res.first ; 
        size_t npix = nside2npix(nside);
        std::vector<std::array<double,3>> points;
        points.reserve(nside2npix(res.fist));

        for( size_t ipix=0; ipix<npix; ipix+=1UL) {
            std::array<double,3> p; 
            pix2vec_ring(nside,ipix,p) ; 
            for( int i=0; i<3; ++i) p[i] += center[i] ; 
            points.push_back(p) ;  
        }

        return points;
    }
    //! TODO (?) this is simply dA for all 
    // points
    static std::vector<double> get_quadrature_weights(double radius, std::pair<size_t,size_t> const& res) {
        size_t npix = nside2npix(res.fist); 
        double A = 4 * M_PI / npix * radius * radius ; 
        return std::vector<double>(npix, A) ; 
    }
};

struct no_tracking_policy_t {
    void track(
        double& radius,
        std::array<double,3>& center
    ) {}
} ; 

struct spherical_surface_iface {
    virtual ~spherical_surface_iface() = default;

    virtual void update_if_needed() = 0;

    std::string name ; 
    double radius ; 
    std::array<double,3> center ;
    size_t npoints   ; 
    size_t res ; 
    std::vector<std::array<double,3>> points_h ; 
    std::vector<double> weights_h ; 
    readonly_twod_view_t<double,3> points ; // maybe don't store here in case some points are not intersected.
    readonly_view_t<double> weights ;

};


template< typename SamplingPolicy 
        , typename TrackingPolicy > 
struct spherical_surface_t: public spherical_surface_iface {

    spherical_surface_t(
        std::string const& _name,
        double _r,
        std::array<double,3> const& c,
        size_t const& _res
    ) : name(_name), radius(_r), center(_c), res(_res)
    {
        tracker = TrackingPolicy() ; 
        npoints = SamplingPolicy::get_n_points(res) ;
        point_h =  SamplingPolicy::get_points(radius, center, res) ; 
        weights_h = SamplingPolicy::get_quadrature_weights(radius,res) ; 
        grace::deep_copy_vec_to_const_2D_view(
            points, points_h
        ) ; 
        grace::deep_copy_vec_to_const_view(weights,weights_h) ;
    }

    /**
     * @brief Update sphere, if tracking is active, this will 
     *        update center and radius and recompute points and 
     *        quadrature weigths
     * 
     */
    void update_if_needed() override {
        // this function is responsible for checking if update is needed
        tracker.track(radius, center) ; 
        point_h =  SamplingPolicy::get_points(radius, center, res) ; 
        weights_h = SamplingPolicy::get_quadrature_weights(radius,res) ; 
        grace::deep_copy_vec_to_const_2D_view(
            points, points_h
        ) ; 
        grace::deep_copy_vec_to_const_view(weights,weights_h) ;
    }

    TrackingPolicy tracker ; 

    #if 0
    void _append_var(std::string const& vname, std::vector<size_t>& vidx, std::vector<size_t>& aidx)  {
        auto& vnames = grace::variables::detail::_varnames ; 
        auto& auxnames = grace::variables::detail::_auxnames ;
        if(std::find(vnames.begin(), vnames.end(), vname) != vnames.end()) {
            vidx.push_back(
                grace::get_variable_index(vname,false) ;
            ) ; 
            return ; 
        }
        if(std::find(auxnames.begin(), auxnames.end(), vname) != auxnames.end()) {
            aidx.push_back(
                grace::get_variable_index(vname,true) ;
            ) ; 
            return ; 
        }

        // handle "special" vars here
        if ( vname == "mass_fluxes" ) {
            // hydro vars needed
            aidx.push_back(RHO) ; aidx.push_back(VELX) ; aidx.push_back(VELY) ; aidx.push_back(VELZ) ;
            aidx.push_back(EPS) ; aidx.push_back(PRESS) ; 

            vidx.push_back(GXX) ; vidx.push_back(GXY) ; vidx.push_back(GXZ) ;
            vidx.push_back(GYY) ; vidx.push_back(GYZ) ; vidx.push_back(GZZ) ; 
            vidx.push_back(BETAX) ; vidx.push_back(BETAY) ; vidx.push_back(BETAZ) ;
            vidx.push_back(ALP) ; 

        } 

        ERROR("Variable " << vname << " not found requested for interpolation") ; 
    }
    #endif 
} ; 
//**************************************************************************************************
//**************************************************************************************************
/**
 * @brief Container for active spherical surfaces 
 * \cond grace_detail
 */
struct spherical_surface_manager_impl_t {
    //**************************************************************************************************
    using ptr_t = std::unique_ptr<spherical_surface_iface> ;
    using ref_t = spherical_surface_iface& ;
    using cref_t = const spherical_surface_iface&;
    //**************************************************************************************************
 public:
    //**************************************************************************************************
    void update() {
        for( auto& d: detectors ) {
            d->update_if_needed() ; 
        }
    }
    //**************************************************************************************************
    ref_t get(size_t i) {
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
    cref_t get(size_t i)  const {
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
    ref_t get(std::string const& n) {
        size_t const i = name_map[n] ; 
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
    cref_t get(std::string const& n)  const {
        size_t const i = name_map[n] ; 
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
 protected:
    //**************************************************************************************************
    spherical_surface_manager_impl_t() ; // here we need to set up from parfiles etc 
    //**************************************************************************************************
    ~spherical_surface_manager_impl_t() = default ; // Right? std unique_ptr cleans up 
    //**************************************************************************************************
    std::vector<ptr_t> detectors ; 
    std::unordered_map<std::string, size_t> name_map ; 
    //**************************************************************************************************
    static constexpr unsigned long longevity = unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES ; 
    //**************************************************************************************************
    //**************************************************************************************************
    friend class utils::singleton_holder<spherical_surface_manager_impl_t> ;
    friend class memory::new_delete_creator<spherical_surface_manager_impl_t, memory::new_delete_allocator> ; 
    //**************************************************************************************************

} ; 
//**************************************************************************************************
using spherical_surface_manager = utils::singleton_holder<spherical_surface_manager_impl_t> ; 
//**************************************************************************************************

}


#endif /* GRACE_SPHERICAL_SURFACES_HH */