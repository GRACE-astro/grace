/**
 * @file regrid_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-20
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

#ifndef GRACE_AMR_REGRID_HELPERS_TPP 
#define GRACE_AMR_REGRID_HELPERS_TPP

#include <grace_config.h>

#include <grace/amr/amr_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/regrid/regridding_policy_kernels.tpp>

#include <grace/utils/interpolators.hh>
#include <grace/utils/device_vector.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

/**
 * @brief Decide whether a quadrant needs to be refined/coarsened
 *        based on custom criterion.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 * @tparam KerT  Type of the kernel.
 * @tparam KerArgT Type of extra arguments to the kernel.
 * @param flag_view View containing regrid flags. 
 * @param kernel    Cell-wise kernel to decide whether to regrid.
 * @param kernel_args Extra arguments to kernel.
 */
template< typename ViewT 
    , typename KerT 
    , typename ... KerArgT> 
void evaluate_regrid_criterion( ViewT flag_view
                              , KerT kernel  
                              , KerArgT&& ... kernel_args) 
{
    using namespace grace  ;  
    auto& params = config_parser::get() ; 
    auto  state  = variable_list::get().getstate() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 
    size_t ngz = amr::get_n_ghosts() ; 

    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 

    double CTORE = params["amr"]["refinement_criterion_CTORE"].as<double>() ; 
    double CTODE = params["amr"]["refinement_criterion_CTODE"].as<double>() ;
    
    /* Each thread league deals with a single quadrant */ 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ; 
    Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                        , policy 
                        , KOKKOS_LAMBDA (member_type team_member)
    {
        double eps ; 
        /* 
        * parallel reduction of regridding criterion 
        * over quadrant cells 
        */ 
        auto reduce_range = 
            Kokkos::TeamThreadRange( 
                    team_member 
                , EXPR(nx,*ny,*nz) ) ; 
        int const q = team_member.league_rank() ; 
        Kokkos::parallel_reduce(  
                reduce_range 
            , KOKKOS_LAMBDA (int64_t& icell, double& leps )
            {
                int const i = icell%nx ;
                int const j = icell/nx%ny; 
                #ifdef GRACE_3D 
                int const k = icell/nx/ny ; 
                #endif  
                auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q, kernel_args...) ; 
                if( eps_new > leps ) {
                    leps = eps_new ;
                }
            } 
            , Kokkos::Max<double>(eps)  
        ) ; 
        team_member.team_barrier() ; 
        if( team_member.team_rank() == 0 ) 
        {
            flag_view(q) = REFINE_FLAG  * ( eps > CTORE )
                         + COARSEN_FLAG * ( eps < CTODE ) ; 
        } 
    }) ;
}


}} /* namespace grace::amr */ 

#endif 