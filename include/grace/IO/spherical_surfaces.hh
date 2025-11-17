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

template< size_t order >
struct interp_weights_t {
    double w[order+1][3] ;
} ; 

template< size_t interp_order >
struct spherical_surface_iface {
    virtual ~spherical_surface_iface() = default;

    virtual void update_if_needed() = 0;


    std::string name ; //!< Name of this surface 
    double radius ; //!< Radius 
    std::array<double,3> center ; //!< Cartesian coordinates of the center 
    size_t npoints   ; //!< Number of points on the surface
    size_t res ; //!< "Resolution"
    // host storage 
    std::vector<point_host_t> points_h ; //!< Points host array -> std::pair<index, {x,y,z}>
    std::vector<double> weights_h ; //!< Quadrature weights
    std::vector<intersected_cell_descriptor_t> intersected_cells_h ; //!< i,j,k, q of intersected cells
    std::vector<size_t> intersecting_points_h; //!< Indices of points contained in local grid
    std::vector<interp_weights_t> interp_weights_h ; //!< Interpolation weights 
    // device storage 
    readonly_twod_view_t<double,3> points ; //!< Device storage of points coordinates
    readonly_view_t<double> weights ; //!< Device storage of quadrature weights 
    redonly_view_t<intersected_cell_descriptor_t> intersected_cells; //!< Device storage of local cells 
    readonly_view_t<size_t> intersecting_points ; 
    readonly_view_t<interp_weights_t<interp_order>> interp_weights ; //!< Device storage of interpolation weights 
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
        pcoords[0] < cube.v[1][0] and pcoords[0] >= cube.v[0][0] and 
        pcoords[1] < cube.v[2][1] and pcoords[1] >= cube.v[0][1] and 
        pcoords[2] < cube.v[4][2] and pcoords[2] >= cube.v[0][2] and 
    ) ; 
    if (! contained ) return 0 ; 

    if ( local_num >=0 ) {
        auto quadid = local_num ; 
        // find the indices of the cell within the quad that contains the point ; 
        
        double xoff = pcoords[0] - cube.v[0][0] ; 
        double yoff = pcoords[1] - cube.v[0][1] ; 
        double zoff = pcoords[2] - cube.v[0][2] ; 
        double idx = static_cast<double>(nx)/(cube.v[1][0] - cube.v[0][0]);
        // clamp you never know 
        // note 0 here means ngz in the quad 
        int i = std::min(nx-1, std::max(0, int(xoff * idx)));
        int j = std::min(ny-1, std::max(0, int(yoff * idx)));
        int k = std::min(nz-1, std::max(0, int(zoff * idx)));

        intersected_cell_descriptor_t desc{i,j,k,quadid} ; 
        auto intersected_cells = static_cast<intersected_cell_set_t*>(p4est->user_pointer) ; 
        intersected_cells->cells->push_back(
            desc
        ) ; 
        intersected_cells->point_idx->push_back(p_idx) ;
    }

    return 1 ;
}

template< typename SamplingPolicy 
        , typename TrackingPolicy 
        , size_t interp_order > 
struct spherical_surface_t: public spherical_surface_iface<interp_order> {

    spherical_surface_t(
        std::string const& _name,
        double _r,
        std::array<double,3> const& c,
        size_t const& _res
    ) : name(_name), radius(_r), center(_c), res(_res)
    {
        tracker = TrackingPolicy() ; 
        update() ; 
    }

    /**
     * @brief Update sphere, if tracking is active, this will 
     *        update center and radius and recompute points and 
     *        quadrature weigths
     * 
     */
    void update_if_needed(bool mesh_changed) override {
        // this function is responsible for checking if update is needed
        auto updated = tracker.track(radius, center) ; 
        if (!updated and !mesh_changed) return ; 
        update() ; 
    }

    TrackingPolicy tracker ; 

    private:

    void update() {
        npoints = SamplingPolicy::get_n_points(res) ;
        points_h =  SamplingPolicy::get_points(radius, center, res) ; 
        weights_h = SamplingPolicy::get_quadrature_weights(radius,res) ; 

        slice_oct_tree() ; 
        compute_interpolation_weights(); 

        grace::deep_copy_vec_to_const_view(interp_weights,interp_weights_h );
        grace::deep_copy_vec_to_const_view(intersected_cells, intersected_cells_h) ; 
        grace::deep_copy_vec_to_const_view(intersecting_points, intersecting_points_h) ; 
        grace::deep_copy_vec_to_const_2D_view(points, points_h) ; 
        grace::deep_copy_vec_to_const_view(weights,weights_h) ;
    }

    void slice_oct_tree() {
        auto points_array = sc_array_new_data(
            points_h.data(), sizeof(std::array<double,3>), points_h.size() 
        ) ; 
        // search 
        auto p4est = grace::amr::forest::get().get() ; 
        intersected_cells_h.clear() ; 
        intersecting_points_h.clear() ; 
        intersected_cell_set_t set{
            &intersected_cells_h,
            &intersecting_points_h
        }; 
        p4est->user_pointer = static_cast<void*>(&set) ; 
        p4est_search_local(
            p4est, 
            false, 
            nullptr, 
            &grace_search_points,
            points_array
        ) ; 
        GRACE_TRACE("Spherical surface {}, number of local points {}", name, intersecting_points_h.size()) ; 
    }

    void compute_interpolation_weights() {
        DECLARE_GRID_EXTENTS ; 
        auto& coord_system = grace::coordinate_system::get() ; 
        
        auto n_points = intersecting_points.size() ;
        interp_weights.reserve(n_points) ; 

        std::array<double,4> wj = {-1./6.,1./2.,-1./2.,1./6.} ;  
        for( int i=0; i<n_points; ++i) {
            auto point_idx = intersecting_points[i] ; 
            auto const& point_coords = points_h[point_idx] ;
            auto const ijkq = intersected_cells[i] ; 
            double const dx = coord_system::get_spacing(ijkq.q) ; 
            std::array<int,3> bias{{0,0,0}} ; 
            // decide if we bias the stencil down (0) or up (1)
            {
                std::array<size_t,3> ijk {{
                        std::get<0>(ijkq),
                        std::get<1>(ijkq),
                        std::get<2>(ijkq)
                    }} ; 
                auto pcoords = coord_system.get_physical_coordinates(
                        ijk, std::get<3>(ijkq), {0.5,0.5,0.5}, false
                    ) ; 
                for( int idir=0; idir<3; ++idir ) bias[idir] = point_coords[idir] > pcoords[idir] ; 
            }
            interp_weights_t<interp_order> iweights ; 
            for( int idir=0; idir<3; ++idir) {
                double norm {0} ; 
                for( int ic=0; ic<4; ++ic) {
                    int off = ic - 2 + bias[idir] ;
                    std::array<size_t,3> ijk {{
                        ijkq.i + off * (idir==0),
                        ijkq.j + off * (idir==1),
                        ijkq.k + off * (idir==2)
                    }} ; 
                    auto pcoords = coord_system.get_physical_coordinates(
                        ijk, ijkq.q, {0.5,0.5,0.5}, false
                    ) ; 
                    double wL = dx * wj[ic]/(point_coords[idir]-pcoords[idir]) ;
                    norm += wL ; 
                    iweights.w[ic][idir]= wL ;     
                }
                for( int ic=0; ic<4; ++ic) {
                    iweights.w[ic][idir]/=norm ; 
                }
            }
            interp_weights.push_back(iweights) ; 
        }
    };

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
    int get_index(std::string const& name) const {
        auto it = name_map.find(name);
        if (it != name_map.end()) {
            // ✔ key exists
            return it->second;
            // do something with val
        } else {
            // ✘ key not found
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