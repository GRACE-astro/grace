/**
 * @file lagrange_interpolation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Slope limiters for use in reconstruction and/or prolongation.
 * @version 0.1
 * @date 2024-04-09
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_UTILS_LAGRANGE_INTERP_HH
#define GRACE_UTILS_LAGRANGE_INTERP_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/math.hh>
#include <grace/IO/surface_IO_utils.hh> 
#include <grace/coordinates/coordinate_systems.hh>

#include <Kokkos_Core.hpp> 


#include <vector>
#include <array> 

namespace grace {

/**
 * @brief Point descriptor, to be used for searching the oct-tree
 */
using point_host_t = std::pair<size_t,std::array<double,3>> ; 
/**
 * @brief Cell descriptor
 */
struct intersected_cell_descriptor_t {
    size_t i,j,k;
    size_t q ; 
} ; 
/**
 * @brief Set of cells plus identifieds tying them to points they contain
 */
struct intersected_cell_set_t {
    std::vector<intersected_cell_descriptor_t>* cells ;
    std::vector<size_t>* point_idx ;
} ; 
/**
 * @brief Callback for searching the local forest for a set of points
 * NB: The user pointer in p4est must be set to an intersected_cells_set_t 
 * **before** this is called.
 */
static int 
grace_search_points(
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
    auto cube  = amr::detail::make_cube(amr::quadrant_t{quadrant}, which_tree) ; 
    bool contained = (
        pcoords[0] < cube.v[1][0] and pcoords[0] >= cube.v[0][0] and 
        pcoords[1] < cube.v[2][1] and pcoords[1] >= cube.v[0][1] and 
        pcoords[2] < cube.v[4][2] and pcoords[2] >= cube.v[0][2]  
    ) ; 
    #if 0 
    GRACE_VERBOSE("Debug info: ip {} x y z {} {} {}\n"
            "                                  qid {} treeid {} x {} {} y {} {} z {} {} contained {}",
            p_idx, pcoords[0],pcoords[1],pcoords[2], local_num, which_tree, cube.v[0][0], cube.v[1][0],
            cube.v[0][1], cube.v[2][1], cube.v[0][2], cube.v[4][2], contained);
    #endif 
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
        size_t i = std::min(nx-1, std::max(0UL, size_t(xoff * idx)));
        size_t j = std::min(ny-1, std::max(0UL, size_t(yoff * idx)));
        size_t k = std::min(nz-1, std::max(0UL, size_t(zoff * idx)));
        intersected_cell_descriptor_t desc;
        desc.i = i ; desc.j = j; desc.k = k; desc.q = quadid ; 
        auto intersected_cells = static_cast<intersected_cell_set_t*>(forest->user_pointer) ; 
        intersected_cells->cells->push_back(
            desc
        ) ; 
        intersected_cells->point_idx->push_back(p_idx) ;
    }

    return 1 ;
}

/**
 * @brief Weights for Lagrange interpolation of given order
 */
template< size_t order >
struct interp_weights_t {
    double w[order+1][3] ;
} ; 

template< size_t order >
struct lagrange_interpolator_t {

    lagrange_interpolator_t(
        int valid_gz
    ) : _valid_gz(valid_gz)
    { }


    void interpolate(
        grace::var_array_t data,
        std::vector<int> const& var_idx_h,
        Kokkos::View<double**,grace::default_space>& out 
    ) const 
    {
        DECLARE_GRID_EXTENTS ; 
        using namespace grace ; 
        using namespace Kokkos ; 

        auto nvars = var_idx_h.size() ; 

        readonly_view_t<int> var_idx;
        deep_copy_vec_to_const_view(var_idx,var_idx_h) ; 

        Kokkos::realloc(out, npoints, nvars) ;

        auto icells = intersected_cells ; 
        auto iweights = interp_weights;
        auto istencils = interp_stencils;

        MDRangePolicy<Rank<2>> policy({0,0},{npoints, static_cast<long>(nvars)}) ;
        parallel_for(
            GRACE_EXECUTION_TAG("IO", "interp_to_sphere"),
            policy,
            KOKKOS_LAMBDA (int const& ip, int const& iv) {
                auto const& cell = icells(ip);
                auto q = cell.q ; 
                auto w = iweights(ip) ; 
                auto u = subview(data, ALL(), ALL(), ALL(), var_idx(iv), q) ; 
                int bx = istencils(ip,0); 
                int by = istencils(ip,1);
                int bz = istencils(ip,2);

                double val{0};
                for( int i=0; i<order+1; ++i) {
                    for( int j=0; j<order+1; ++j) {
                        for( int k=0; k<order+1; ++k) {
                            // FIXME this should be - (order+1)/2 but hardcoded 4 so fine.. 
                            int io = i - 2 + bx ; 
                            int jo = j - 2 + by ; 
                            int ko = k - 2 + bz ; 
                            int ic = static_cast<int>(ngz+cell.i) + io ; 
                            int jc = static_cast<int>(ngz+cell.j) + jo ; 
                            int kc = static_cast<int>(ngz+cell.k) + ko ; 
                            val += w.w[i][0] * w.w[j][1] * w.w[k][2] * u(ic,jc,kc) ;
                        }
                    }
                }
                out(ip,iv) = val ;
            }   
        ) ;
    }

    void compute_weights(
        std::vector<point_host_t> const & points ,
        std::vector<size_t> const & ipoints ,
        std::vector<intersected_cell_descriptor_t> const & icells
    ) {
        DECLARE_GRID_EXTENTS ; 
        auto& coord_system = grace::coordinate_system::get() ; 

        npoints = ipoints.size() ; 
        weights.resize(npoints) ; 
        stencils.resize(npoints) ; 
        
        auto wj = get_barycentric_weights() ; 
        for( int i=0; i<npoints; ++i) {
            auto pidx = ipoints[i] ; 
            auto const& point_coords = points[pidx].second ;
            auto const ijkq = icells[i] ; 
            double const dx = coord_system.get_spacing(ijkq.q) ; 

            // figure out if we need to bias the stencil
            std::array<int,3> bias{{0,0,0}} ; 
            {
                std::array<size_t,3> ijk {{
                        ijkq.i,
                        ijkq.j,
                        ijkq.k
                    }} ; 

                for( int idir=0; idir<3; ++idir ) { 
                     bias[idir] = 0 ;
                    if ( ijk[idir] - (order+1)/2 < ngz - _valid_gz ) {
                        bias[idir] = +1 ;
                    } 
                    if ( ijk[idir] - (order+1)/2 + order >= nx + ngz + _valid_gz ) {
                        bias[idir] = -1 ;
                    }
                }
            }
            stencils[i] = bias ; 

            interp_weights_t<order> iweights ; 
            for( int idir=0; idir<3; ++idir) {
                double norm {0} ; 
                for( int ic=0; ic<order+1; ++ic) {
                    int off = ic - (order+1)/2 + bias[idir] ;
                    std::array<size_t,3> ijk {{
                        ijkq.i + off * (idir==0),
                        ijkq.j + off * (idir==1),
                        ijkq.k + off * (idir==2)
                    }} ; 
                    auto pcoords = coord_system.get_physical_coordinates(
                        ijk, ijkq.q, {0.5,0.5,0.5}, false
                    ) ; 
                    double wL = dx * wj[ic]/(point_coords[idir]-pcoords[idir]+1e-15*std::copysign(1.0,point_coords[idir]-pcoords[idir])) ;
                    norm += wL ; 
                    iweights.w[ic][idir]= wL ;     
                }
                for( int ic=0; ic<order+1; ++ic) {
                    iweights.w[ic][idir]/=norm ; 
                }
            }
            weights[i] = iweights ; 
        }

        deep_copy_vec_to_const_view(intersected_cells,icells)    ; 
        deep_copy_vec_to_const_2D_view(interp_stencils,stencils) ; 
        deep_copy_vec_to_const_view(interp_weights,weights)      ; 
    }

    std::array<double,order+1> get_barycentric_weights() const
    {
        std::vector<double> nodes(order+1);
        for ( int inode=0; inode<order+1; ++inode) {
            nodes[inode] = inode ; 
        }
        std::array<double,order+1> w; 
        for( int i=0; i<order+1; ++i) {
            double ww=1;
            for( int j=0; j<order+1; ++j) {
                if ( i== j ) continue ; 
                ww *= nodes[i] - nodes[j] ; 
            }
            w[i] = 1/ww ; 
        }
        return w ; 
    }

    int _valid_gz ; //!< How many ghostzones are valid in the target var? 

    // host storage 
    std::vector<std::array<int,3>> stencils         ; 
    std::vector<interp_weights_t<order>>    weights ; 

    // device storage 
    int npoints ; 
    readonly_view_t<intersected_cell_descriptor_t> intersected_cells; //!< Device storage of local cells 
    readonly_view_t<interp_weights_t<order>> interp_weights  ; //!< Device storage of interpolation weights 
    readonly_twod_view_t<int,3> interp_stencils                     ; //!< Bias of the stencils in each direction 

} ; 
}

#endif /* GRACE_UTILS_LAGRANGE_INTERP_HH */