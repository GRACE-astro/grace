/**
 * @file admbase.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-28
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

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/errors/error.hh>

#include <Kokkos_Core.hpp>

#include <string>

namespace grace {


static void set_admbase_minkowski_id() {
    using namespace grace ;
    using namespace Kokkos ;

    auto& state   = variable_list::get().getstate() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    parallel_for( GRACE_EXECUTION_TAG("ID","admbase_Minkowski_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    state(VEC(i,j,k),GXX_,q)   = 1. ; state(VEC(i,j,k),GXY_,q)   = 0. ; state(VEC(i,j,k),GXZ_,q)   = 0. ;
                    state(VEC(i,j,k),GYY_,q)   = 1. ; state(VEC(i,j,k),GYZ_,q)   = 0. ; state(VEC(i,j,k),GZZ_,q)   = 1. ;
                    state(VEC(i,j,k),BETAX_,q) = 0. ; state(VEC(i,j,k),BETAY_,q) = 0. ; state(VEC(i,j,k),BETAZ_,q) = 0. ;
                    state(VEC(i,j,k),ALP_,q) = 1. ;
                    state(VEC(i,j,k),KXX_,q) = 0. ; state(VEC(i,j,k),KXY_,q) = 0. ; state(VEC(i,j,k),KXZ_,q) = 0. ;
                    state(VEC(i,j,k),KYY_,q) = 0. ; state(VEC(i,j,k),KYZ_,q) = 0. ; state(VEC(i,j,k),KZZ_,q) = 0. ;
                }
    ) ;
}

void set_admbase_id() {
     

    auto const metric_type = 
        grace::get_param<std::string>("admbase","metric_kind") ;

    GRACE_VERBOSE("Setting {} metric initial data.", metric_type) ;

    if ( metric_type == "Minkowski" ) {
        set_admbase_minkowski_id() ; 
    } else {
        ERROR("Metric type " << metric_type << " not supported.") ;
    }
}

}