/**
 * @file amr_functions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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

#include <grace/amr/amr_functions.hh>

#include <grace/amr/tree.hh>
#include <grace/amr/connectivity.hh>
#include <grace/amr/forest.hh> 

#include <grace/config/config_parser.hh>

namespace grace { namespace amr {
namespace detail {
int64_t _nx; 
int64_t _ny;
int64_t _nz;
int _ngz ;
}

std::tuple<size_t,size_t,size_t>
get_quadrant_extents()
{
    return std::make_tuple(detail::_nx,detail::_ny,detail::_nz) ;  
}

int 
get_n_ghosts()
{
    return detail::_ngz ; 
}

size_t 
get_local_num_quadrants()
{
    return grace::amr::forest::get().local_num_quadrants() ; 
}

int 
get_quadrant_owner(size_t iquad)
{
    auto& forest = grace::amr::forest::get() ;
    for(size_t itree=forest.first_local_tree();
        itree <= forest.last_local_tree(); 
        itree+=1UL)
    {
        auto tree = forest.tree(itree) ; 
        int iquad_loc = iquad - tree.quadrants_offset() ; 
        if(     (iquad_loc >= 0)
            and (iquad_loc < tree.num_quadrants() ) ){
            return itree ; 
        }
    }
    ASSERT_DBG(0, 
    "In get_quadrant_owner: " << iquad << " is not owned by any local tree.") ;
    return -1 ; 
}

size_t get_local_quadrants_offset(size_t itree)
{
    tree_t tree = grace::amr::forest::get().tree(itree);
    return tree.quadrants_offset() ; 
}

quadrant_t  
get_quadrant(size_t which_tree, size_t iquad)
{
    tree_t tree = grace::amr::forest::get().tree(which_tree) ;
    return tree.quadrant(iquad-tree.quadrants_offset()) ; 
}

quadrant_t  
get_quadrant(size_t iquad)
{
    tree_t tree = 
        grace::amr::forest::get().tree(get_quadrant_owner(iquad)) ;
    return tree.quadrant(iquad-tree.quadrants_offset()) ; 
}

int64_t 
get_quadrant_locidx(quadrant_t quad)
{
    auto tree = forest::get().tree(quad.tree_index()); 
    int64_t locidx{0L} ; 
    for( int64_t locidx=0; locidx<tree.num_quadrants(); ++locidx ){
        auto q = tree.quadrant(locidx) ; 
        if ( p4est_quadrant_is_equal(q.get(), quad.get()) ){
            break ; 
        } ;
    }
    return locidx + tree.quadrants_offset() ;
}

int64_t 
get_quadrant_locidx(p4est_quadrant_t* quad)
{
    return get_quadrant_locidx(quadrant_t(quad)) ; 
}

int 
get_halo_quad_owner(quadrant_t& quad)
{
    return quad.halo_owner_rank() ; 
};

int 
get_halo_quad_owner(p4est_quadrant_t* quad)
{
    return quadrant_t(quad).halo_owner_rank() ;
}

int 
trees_have_opposite_polarity( int64_t treeid, int face )
{
    return connectivity::get().tree_to_tree_polarity(treeid, face) ; 
}

std::array<double,GRACE_NSPACEDIM> 
get_tree_vertex(size_t which_tree, size_t which_vertex)
{
    return grace::amr::connectivity::get().vertex_coordinates(which_tree,which_vertex);
}

std::array<double,GRACE_NSPACEDIM> 
get_tree_spacing(size_t which_tree)
{
    return grace::amr::connectivity::get().tree_coordinate_extents(which_tree);
}

std::vector<int64_t>
get_global_quadrant_offsets()
{
    std::vector<int64_t> offsets(parallel::mpi_comm_size()+1) ; 
    for(int i=0; i<=parallel::mpi_comm_size(); ++i)
        offsets[i] = forest::get().global_quadrant_offset(i) ; 
    return std::move(offsets) ; 
}

}} /* namespace grace::amr */ 