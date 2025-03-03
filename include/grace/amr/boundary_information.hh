/**
 * @file boundary_information.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-03
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_AMR_BOUNDARY_INFO_HH
#define GRACE_AMR_BOUNDARY_INFO_HH


#include <grace_config.h> 

#include <grace/utils/inline.h>
#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/sc_wrappers.hh>

#include <grace/amr/p4est_headers.hh>

#include <grace/amr/tree.hh>
#include <grace/amr/quadrant.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/bc_iterate.hh>

#include <grace/parallel/mpi_wrappers.hh>

namespace grace { namespace amr {
//*****************************************************************************************************
class boundary_info_impl_t 
{
    //*****************************************************************************************************
 private:
    //*****************************************************************************************************
    grace_neighbor_info_t info ; 
    //*****************************************************************************************************
    p4est_ghost_t *halos       ;
    //*****************************************************************************************************
 public:
    //*****************************************************************************************************
    void update() {
        // If the halo is not empty destroy it first 
        if ( halos ) {
            p4est_ghost_destroy(halos) ; 
        }
        // Fill the halo info from the forest 
        halos = p4est_ghost_new( 
              forest::get().get() 
            , P4EST_CONNECT_FULL 
        ) ; 
        // Iterate and store boundary information
        p4est_iterate(
            forest::get().get()
          , halos 
          , reinterpret_cast<void*>( &info ) 
          , nullptr
          , grace_iterate_faces 
          #ifdef GRACE_3D 
          , grace_iterate_edges 
          #endif 
          , grace_iterate_corners ) ;
    }; 
    p4est_ghost_t * get_halos() const { return halos ; }
    grace_neighbor_info_t& get_boundary_info(){ return info ;  } ; 
    //*****************************************************************************************************
 private:
    //*****************************************************************************************************
    boundary_info_impl_t() 
        : halos{nullptr}, info{grace_neighbor_info_t()}
    { 
        update() ; 
    } ; 
    //*****************************************************************************************************
    ~boundary_info_impl_t() {
        if(halos) p4est_ghost_destroy(halos) ;
    } ; 
    //*****************************************************************************************************
    friend class utils::singleton_holder<boundary_info_impl_t, memory::default_create> ;          //!< Give access 
    friend class memory::new_delete_creator<boundary_info_impl_t, memory::new_delete_allocator> ; //!< Give access
    static constexpr unsigned int longevity = AMR_BC_INFO ; //!< Longevity of p4est object. 
   //*****************************************************************************************************
} ; 
//*****************************************************************************************************

using boundary_info = utils::singleton_holder<boundary_info_impl_t> ; 

}} /* namespace grace::amr */
//*****************************************************************************************************
#endif /* GRACE_AMR_BOUNDARY_INFO_HH */