/**
 * @file spherical_surfaces.hh
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

#ifndef GRACE_SPHERICAL_SURFACES_HH
#define GRACE_SPHERICAL_SURFACES_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/device_vector.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>
#include <grace/utils/lagrange_interpolation.hh>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

#include "surface_IO_utils.hh"
#include "spherical_surface_helpers.hh"
#include "octree_search_class.hh"

#include <array>
#include <memory>

namespace grace {

struct spherical_surface_iface {

    spherical_surface_iface(
        std::string const& _name,
        double _r,
        std::array<double,3> const& _c,
        size_t _res
    ): name(_name), radius(_r), center(_c), res(_res), interpolator(1 /*NB for psi4 we need this!*/)
    {}

    virtual ~spherical_surface_iface() = default;

    virtual void update_if_needed(bool mesh_updated) = 0;


    std::string name                   ; //!< Name of this surface 
    double radius                      ; //!< Radius 
    std::array<double,3> center        ; //!< Cartesian coordinates of the center 
    size_t npoints_glob, npoints_loc   ; //!< Number of points on the surface
    size_t res                         ; //!< "Resolution"
    // host storage 
    std::vector<point_host_t> points_h                              ; //!< Points host array -> std::pair<index, {x,y,z}>
    std::vector<std::array<double,2>> angles_h                      ; //!< Angles
    std::vector<double> weights_h                                   ; //!< Quadrature weights
    std::vector<intersected_cell_descriptor_t> intersected_cells_h  ; //!< i,j,k, q of intersected cells
    std::vector<size_t> intersecting_points_h                       ; //!< Indices of points contained in local grid
    // interpolator 
    lagrange_interpolator_t<LAGRANGE_INTERP_ORDER> interpolator ;  
};

template< typename SamplingPolicy 
        , typename TrackingPolicy > 
struct spherical_surface_t: public spherical_surface_iface {
    using base_t = spherical_surface_iface ;
    

    spherical_surface_t(
        std::string const& _name,
        double _r,
        std::array<double,3> const& c,
        size_t const& _res
    ) : base_t(_name,_r,c,_res)
    {
        tracker = TrackingPolicy() ; 
        this->update() ; 
    }

    /**
     * @brief Update sphere, if tracking is active, this will 
     *        update center and radius and recompute points and 
     *        quadrature weigths
     * 
     */
    void update_if_needed(bool mesh_changed) override {
        // this function is responsible for checking if update is needed
        auto updated = tracker.track(this->radius, this->center) ; 
        if (!updated and !mesh_changed) return ; 
        this->update() ; 
    }

    TrackingPolicy tracker ; 

    private:

    void update() {
        // construct the surface 
        this->npoints_glob = SamplingPolicy::get_n_points(this->res) ;
        this->points_h     =  SamplingPolicy::get_points(this->radius, this->center, this->res, this->angles_h) ; 
        this->weights_h    = SamplingPolicy::get_quadrature_weights(this->radius,this->res) ; 
        // find intersection with local forest 
        slice_oct_tree() ; 
        // pre-compute interpolation weights 
        interpolator.compute_weights(
            this->points_h,
            this->intersecting_points_h,
            this->intersected_cells_h
        ) ; 
    }

    void slice_oct_tree() {
        GRACE_VERBOSE("Slicing oct-tree, total n points: {}", this->points_h.size() );
        auto points_array = sc_array_new_data(
            this->points_h.data(), sizeof(point_host_t), this->points_h.size() 
        ) ; 
        // search 
        auto p4est = grace::amr::forest::get().get() ; 
        this->intersected_cells_h.clear() ; 
        this->intersecting_points_h.clear() ; 
        intersected_cell_set_t set{
            &this->intersected_cells_h,
            &this->intersecting_points_h
        }; 
        p4est->user_pointer = static_cast<void*>(&set) ; 
        p4est_search_local(
            p4est, 
            false, 
            nullptr, 
            &grace_search_points,
            points_array
        ) ; 
        this->npoints_loc = this->intersecting_points_h.size() ; 
        GRACE_VERBOSE("Spherical surface {}, number of local points {}", this->name, this->intersecting_points_h.size()) ; 
    }

} ; 
//**************************************************************************************************
void interpolate_on_sphere( spherical_surface_iface const& surf
                       , std::vector<int> const& var_idx_h 
                       , std::vector<int> const& aux_idx_h 
                       , Kokkos::View<double**,grace::default_space>& out
                       , Kokkos::View<double**,grace::default_space>& out_aux ); 


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
    void update(bool mesh_changed) {
        for( auto& d: detectors ) {
            d->update_if_needed(mesh_changed) ; 
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
        size_t i;
        auto it = name_map.find(n);
        if (it != name_map.end()) {
            i = it->second;
        } else {
            ERROR("Invalid surface requested " << n) ; 
        }
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
    cref_t get(std::string const& n)  const {
        size_t i;
        auto it = name_map.find(n);
        if (it != name_map.end()) {
            i = it->second;
        } else {
            ERROR("Invalid surface requested " << n) ; 
        }
        ASSERT(i < detectors.size(), 
        "Requested detector " << i << " exceeds maximum " << detectors.size() ) ; 
        return *detectors[i] ; // note this is a reference! 
    }
    //**************************************************************************************************
    int get_index(std::string const& name) const {
        auto it = name_map.find(name);
        if (it != name_map.end()) {
            return it->second;
        } else {
            return -1 ;
        }
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