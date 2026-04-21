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

#include <grace/utils/device_vector.hh>
#include <grace/utils/reductions.hh>

#include <grace/coordinates/coordinate_systems.hh>
#include <grace/IO/diagnostics/co_tracker.hh>


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
 */
template< typename ViewT 
    , typename KerT 
    , typename ... KerArgT> 
void evaluate_regrid_criterion( ViewT flag_view
                              , KerT kernel) 
{
    using namespace grace  ;  
    // Get arrays 
    auto  state  = variable_list::get().getstate() ; 
    // Get array sizes 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 
    size_t ngz = amr::get_n_ghosts() ; 
    // Store flags 
    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 
    // Store parameters 
    double CTORE = get_param<double>("amr","refinement_criterion_CTORE");  
    double CTODE = get_param<double>("amr","refinement_criterion_CTODE");
    auto reduction = get_param<std::string>("amr", "refinement_criterion_reduction") ; 
    // Policy 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ;

    if ( reduction == "max" ) {
        /* Each thread league deals with a  single quadrant */ 
     
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
                , [=] (int64_t const icell, double& leps )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q) ; 
                    if( eps_new > leps ) {
                        leps = eps_new ;
                    }
                } 
                , Kokkos::Max<double>(eps)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = REFINE_FLAG  * ( eps >= CTORE )  
                                 + COARSEN_FLAG * ( eps <= CTODE ) ; 
                }
            ) ; 
        }) ;
    } else if ( reduction == "min" ) {
        /* Each thread league deals with a  single quadrant */ 
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
                , [=] (int64_t const icell, double& leps )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    auto eps_new = kernel(VEC(i+ngz,j+ngz,k+ngz), q) ; 
                    if( eps_new < leps ) {
                        leps = eps_new ;
                    }
                } 
                , Kokkos::Min<double>(eps)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = REFINE_FLAG  * ( eps <= CTORE )  
                                 + COARSEN_FLAG * ( eps >= CTODE ) ; 
                }
            ) ; 
        }) ;
    } else {
        ERROR("Unrecognized reduction for refinement.") ; 
    }
    
}


template< typename view_t >
void evaluate_binary_tracker_criterion(view_t & flag_view )
{
    DECLARE_GRID_EXTENTS; 

    using namespace Kokkos ; 
    using namespace grace  ; 
    auto& co_tracker = grace::co_tracker::get() ; 


    ASSERT(co_tracker.is_active(), "CO tracker specified as AMR criterion is inactive.") ; 
    ASSERT(co_tracker.get_n_cos()==2, "CO tracker only tracking one object" ) ; 

    double co1_re_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_1_refine_radius_factor") ; 
    double co1_de_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_1_coarsen_radius_factor") ; 

    double co2_re_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_2_refine_radius_factor") ; 
    double co2_de_radius = get_param<double>("amr","binary_tracker_amr_criterion","compact_object_2_coarsen_radius_factor") ; 

    double post_merger_re_radius = get_param<double>("amr", "binary_tracker_amr_criterion", "post_merger_refine_radius") ; 
    double post_merger_de_radius = get_param<double>("amr", "binary_tracker_amr_criterion", "post_merger_coarsen_radius") ; 

    Kokkos::View<double[3], grace::default_space> co1_loc("co1_location"), co2_loc("co2_location") ; 
    auto co1_h = create_mirror_view(co1_loc) ; 
    auto co2_h = create_mirror_view(co2_loc) ; 

    auto l1 = co_tracker.get(0)->get_loc() ; 
    auto l2 = co_tracker.get(1)->get_loc() ; 
    for(int ii=0; ii<3; ++ii) {
        co1_h(ii) = l1[ii] ; 
        co2_h(ii) = l2[ii] ; 
    } 

    Kokkos::deep_copy(co1_loc,co1_h) ; 
    Kokkos::deep_copy(co2_loc,co2_h) ; 

    double const r1 = co_tracker.get(0)->get_radius() ; 
    double const r2 = co_tracker.get(1)->get_radius() ; 

    bool has_merged = co_tracker.get_merged() ; 

    // Policy 
    Kokkos::TeamPolicy<default_execution_space> policy(nq, Kokkos::AUTO() ) ; 
    using member_type = Kokkos::TeamPolicy<default_execution_space>::member_type ;

    // coordinates 
    auto dc = grace::coordinate_system::get().get_device_coord_system() ;
    
    // Store flags 
    size_t REFINE_FLAG  = amr::quadrant_flags_t::REFINE  ;  
    size_t COARSEN_FLAG = amr::quadrant_flags_t::COARSEN ; 

    if (has_merged) {

        Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                            , policy 
                            , KOKKOS_LAMBDA (member_type team_member)
        {
            double rmin ; 
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
                , [=] (int64_t const icell, double& lrmin )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #else 
                    int const k = 0 
                    #endif   
                    double xyz[3];
                    dc.get_physical_coordinates(i+ngz,j+ngz,k+ngz,q,xyz) ; 
                    double r = Kokkos::sqrt(
                        SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
                    ) ; 
                    if( r < lrmin ) {
                        lrmin = r ;
                    }
                } 
                , Kokkos::Min<double>(rmin)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    flag_view(q) = ( rmin <= post_merger_re_radius ) ? REFINE_FLAG 
                                        : (( rmin >= post_merger_de_radius ) ? COARSEN_FLAG : 0) ;
                }
            ) ; 
        }) ;

    } else {
        Kokkos::parallel_for( GRACE_EXECUTION_TAG("AMR","eval_refine_coarsen_criterion")
                            , policy 
                            , KOKKOS_LAMBDA (member_type team_member)
        {
            block_dist_t dist ; 
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
                , [=] (int64_t const icell, block_dist_t& ldist )
                {
                    int const i = icell%nx ;
                    int const j = icell/nx%ny; 
                    #ifdef GRACE_3D 
                    int const k = icell/nx/ny ; 
                    #endif  
                    double xyz[3];
                    dc.get_physical_coordinates(i+ngz,j+ngz,k+ngz,q,xyz) ; 
                    double d1 = Kokkos::sqrt(
                        SQR(xyz[0]-co1_loc(0)) + SQR(xyz[1]-co1_loc(1)) + SQR(xyz[2]-co1_loc(2))
                    ) ;
                    double d2 = Kokkos::sqrt(
                        SQR(xyz[0]-co2_loc(0)) + SQR(xyz[1]-co2_loc(1)) + SQR(xyz[2]-co2_loc(2))
                    ) ; 
                    
                    if ( d1 < ldist.min_d0 ) {
                        ldist.min_d0 = d1 ; 
                    }

                    if ( d2 < ldist.min_d1 ) {
                        ldist.min_d1 = d2 ; 
                    }
                } 
                , Kokkos::Sum<block_dist_t>(dist)  
            ) ; 
            team_member.team_barrier() ; 
            Kokkos::single( 
                Kokkos::PerTeam(team_member),
                [&] () {
                    bool refine   = (dist.min_d0 < co1_re_radius * r1) || (dist.min_d1 < co2_re_radius * r2) ;
                    bool derefine = (dist.min_d0 > co1_de_radius * r1) && (dist.min_d1 > co2_de_radius * r2) ;
                    flag_view(q) = refine ? REFINE_FLAG : (derefine ? COARSEN_FLAG : 0) ;
                }
            ) ; 
        }) ;
    }
}

}} /* namespace grace::amr */ 

#endif 
