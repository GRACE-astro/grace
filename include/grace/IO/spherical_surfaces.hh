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

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>
#include <grace/utils/device_vector.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

#include "surface_IO_utils.hh"
#include "spherical_surface_helpers.hh"

#include <array>
#include <memory>

namespace grace {



struct spherical_surface_iface {
    virtual ~spherical_surface_iface() = default;

    virtual void update_if_needed() = 0;

    virtual void slice_oct_tree() = 0 ;

    virtual void compute_interpolation_weights() = 0 ; 

    std::string name ; 
    double radius ; 
    std::array<double,3> center ;
    size_t npoints   ; 
    size_t res ; 
    std::vector<point_host_t> points_h ; 
    std::vector<double> weights_h ; 
    std::vector<intersected_cell_descriptor_t> intersected_cells ; //!< i,j,k, q of intersected cells
    std::vector<size_t> intersecting_points; 
    std::vector<std::array<std::array<double,4>,3>> interp_weights ; //!< Interpolation weights 
    readonly_twod_view_t<double,3> points ; // maybe don't store here in case some points are not intersected.
    readonly_view_t<double> weights ;
};

int grace_search_points(
    p4est_t* forest,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant, 
    p4est_locidx_t local_num,
    void* point
)
{
    DECLARE_GRID_EXTENTS ; 
    auto point_desc = static_cast<point_host_t*>(point) ; 
    auto p_idx = point_desc->first ;
    auto pcoords = point_desc->second; 
    // now construct a cube from the quadrant 
    auto cube  = detail::make_cube(quadrant_t{quadrant}, which_tree) ; 
    bool contained = (
        pcoords[0] <= cube.v[1][0] and pcoords[0] >= cube.v[0][0] and 
        pcoords[1] <= cube.v[2][1] and pcoords[1] >= cube.v[0][1] and 
        pcoords[2] <= cube.v[4][2] and pcoords[2] >= cube.v[0][2] and 
    ) ; 

    if ( local_num >=0 and contained ) {
        auto quadid = local_num ; 
        // find the indices of the cell within the quad that contains the point ; 
        double xoff = pcoords[0] - cube.v[0][0] ; 
        double yoff = pcoords[1] - cube.v[0][1] ; 
        double zoff = pcoords[2] - cube.v[0][2] ; 
        double idx = static_cast<double>(nx)/(cube.v[1][0] - cube.v[0][0]);
        int i = static_cast<int>(xoff * idx) ; 
        int j = static_cast<int>(yoff * idx) ; 
        int k = static_cast<int>(zoff * idx) ; 
        auto intersected_cells = static_cast<std::vector<intersected_cell_set_t>*>(p4est->user_pointer) ; 

        intersected_cells->cells->push_back(
            std::make_tuple(i,j,k,quadid)
        ) ; 
        intersected_cells->point_idx->push_back(p_idx) ;
    }

    return static_cast<int>(contained) ;
}

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
        slice_oct_tree() ; 
        compute_interpolation_weights() ; 
    }

    /**
     * @brief Update sphere, if tracking is active, this will 
     *        update center and radius and recompute points and 
     *        quadrature weigths
     * 
     */
    void update_if_needed() override {
        // this function is responsible for checking if update is needed
        auto updated = tracker.track(radius, center) ; 
        if (!updated) return ; 
        point_h =  SamplingPolicy::get_points(radius, center, res) ; 
        weights_h = SamplingPolicy::get_quadrature_weights(radius,res) ; 
        grace::deep_copy_vec_to_const_2D_view(
            points, points_h
        ) ; 
        grace::deep_copy_vec_to_const_view(weights,weights_h) ;
    }

    void slice_oct_tree() override {
        auto points_array = sc_array_new_data(
            points_h.data(), sizeof(std::array<double,3>), points_h.size() 
        ) ; 
        // search 
        auto p4est = grace::amr::forest::get().get() ; 
        intersected_cells.clear() ; 
        intersecting_points.clear() ; 
        intersected_cell_set_t set{
            &intersected_cells,
            &intersecting_points
        }; 
        p4est->user_pointer = static_cast<void*>(&set) ; 
        p4est_search_local(
            p4est, 
            false, 
            nullptr, 
            &grace_search_points,
            points_array
        ) ; 
        GRACE_TRACE("Spherical surface {}, number of local points {}", name, intersecting_points.size()) ; 
    }

    void compute_interpolation_weights() override {
        
    };

    TrackingPolicy tracker ; 

    private:

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